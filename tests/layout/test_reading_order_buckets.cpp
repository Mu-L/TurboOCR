// Unit tests for the recursive XY-cut reading-order algorithm.
//
// Exercises projection_by_bboxes, split_projection_profile, and
// assign_reading_order on synthetic layouts: single column, two
// columns, header + two columns, single box, and empty input.

#include <catch_amalgamated.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/layout/blocks/child_blocks.h"
#include "turbo_ocr/layout/blocks/match_unsorted.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/layout/blocks/text_line_cluster.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::layout::assign_reading_order;
using turbo_ocr::layout::assign_reading_order_for_results;
using turbo_ocr::layout::LayoutBox;
using turbo_ocr::layout::projection_by_bboxes;
using turbo_ocr::layout::recursive_xy_cut;
using turbo_ocr::layout::split_projection_profile;

namespace {

// Build a 4-corner Box from (x0, y0, x1, y1).
Box make_box(int x0, int y0, int x1, int y1) {
  return Box{{{{{x0, y0}}, {{x1, y0}}, {{x1, y1}}, {{x0, y1}}}}};
}

LayoutBox make_layout(int x0, int y0, int x1, int y1, int class_id = 22) {
  LayoutBox lb;
  lb.class_id = class_id;
  lb.score = 0.99f;
  lb.box = make_box(x0, y0, x1, y1);
  return lb;
}

} // namespace

namespace {
// Make an OCRResultItem with center inside the given AABB and a given
// layout_id. Used to test results-level reading-order grouping.
OCRResultItem make_result(int x0, int y0, int x1, int y1, int layout_id) {
  OCRResultItem r;
  r.text = "x";
  r.confidence = 0.9f;
  r.box = make_box(x0, y0, x1, y1);
  r.layout_id = layout_id;
  return r;
}
} // namespace



// ---- Class-aware bucketing (header → body → footer/reference) -----------

TEST_CASE("assign_reading_order hoists header above body and sinks footer",
          "[xy_cut][bucket]") {
  // Geometric position alone would order: header, body1, body2, footer
  // (top-to-bottom). We also add a malformed layout where the footer is
  // accidentally placed mid-page — the class-aware bucket sort must
  // still push it to the end.
  std::vector<LayoutBox> layout = {
      make_layout(50, 200, 450, 280, /*class_id=*/22),  // body text 1
      make_layout(50, 50,  450, 100, /*class_id=*/12),  // header
      make_layout(50, 320, 450, 380, /*class_id=*/8),   // footer (mis-placed mid-body)
      make_layout(50, 290, 450, 310, /*class_id=*/22),  // body text 2 (after footer y)
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 4);
  CHECK(order[0] == 1);  // header
  // body bucket: text 1 above text 2
  CHECK(order[1] == 0);
  CHECK(order[2] == 3);
  CHECK(order[3] == 2);  // footer last regardless of geometric position
}

TEST_CASE("assign_reading_order sinks reference/footnote/vision_footnote to bottom",
          "[xy_cut][bucket]") {
  // Reference should land after body. Multi-line footnote keeps internal
  // top-to-bottom order within the bottom bucket.
  std::vector<LayoutBox> layout = {
      make_layout(50, 200, 450, 240, /*class_id=*/18),  // reference (early)
      make_layout(50,  60, 450, 100, /*class_id=*/22),  // body 1
      make_layout(50, 110, 450, 150, /*class_id=*/22),  // body 2
      make_layout(50, 260, 450, 280, /*class_id=*/10),  // footnote (lower)
      make_layout(50, 290, 450, 310, /*class_id=*/24),  // vision_footnote
  };
  auto order = assign_reading_order(layout);
  REQUIRE(order.size() == 5);
  CHECK(order[0] == 1);  // body 1
  CHECK(order[1] == 2);  // body 2
  // bottom bucket order is XY-cut by y_min: reference (y=200), footnote (260), vision_footnote (290)
  CHECK(order[2] == 0);  // reference
  CHECK(order[3] == 3);  // footnote
  CHECK(order[4] == 4);  // vision_footnote
}

TEST_CASE("assign_reading_order_for_results: header text reads first across buckets",
          "[xy_cut][bucket]") {
  // Real-world shape: header line + two body paragraphs + footnote.
  // Each layout region holds one OCR result.
  std::vector<LayoutBox> layout = {
      make_layout(50, 200, 450, 240, /*class_id=*/22),  // body 1
      make_layout(50,  60, 450, 100, /*class_id=*/12),  // header
      make_layout(50, 250, 450, 290, /*class_id=*/22),  // body 2
      make_layout(50, 320, 450, 360, /*class_id=*/10),  // footnote
  };
  std::vector<OCRResultItem> results = {
      make_result(50, 250, 450, 290, /*layout_id=*/2),  // body 2 line
      make_result(50,  60, 450, 100, /*layout_id=*/1),  // header line
      make_result(50, 320, 450, 360, /*layout_id=*/3),  // footnote line
      make_result(50, 200, 450, 240, /*layout_id=*/0),  // body 1 line
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 4);
  CHECK(order[0] == 1);  // header
  CHECK(order[1] == 3);  // body 1
  CHECK(order[2] == 0);  // body 2
  CHECK(order[3] == 2);  // footnote (bottom bucket)
}

TEST_CASE("assign_reading_order_for_results: row tolerance handles table cell jitter",
          "[xy_cut][table]") {
  // A 3-column × 2-row table inside one layout box. OCR detection
  // produces a few pixels of y-jitter per cell — strict (y, x) sort
  // would interleave columns. The within-block sort must bucket by row
  // first, then sort x within each row.
  std::vector<LayoutBox> layout = {make_layout(50, 100, 950, 250, /*class_id=*/21)};  // table
  std::vector<OCRResultItem> results = {
      // Row 1, with 1-3 px y-jitter per cell.
      make_result( 60, 110, 200, 140, /*layout_id=*/0),  // R1-LEFT  cy ≈ 125
      make_result(360, 112, 500, 142, /*layout_id=*/0),  // R1-MID   cy ≈ 127
      make_result(660, 113, 800, 143, /*layout_id=*/0),  // R1-RIGHT cy ≈ 128
      // Row 2.
      make_result( 60, 200, 200, 230, /*layout_id=*/0),  // R2-LEFT  cy ≈ 215
      make_result(360, 201, 500, 231, /*layout_id=*/0),  // R2-MID   cy ≈ 216
      make_result(660, 202, 800, 232, /*layout_id=*/0),  // R2-RIGHT cy ≈ 217
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 6);
  // Expected row-major: R1-LEFT, R1-MID, R1-RIGHT, R2-LEFT, R2-MID, R2-RIGHT
  CHECK(order[0] == 0);
  CHECK(order[1] == 1);
  CHECK(order[2] == 2);
  CHECK(order[3] == 3);
  CHECK(order[4] == 4);
  CHECK(order[5] == 5);
}

TEST_CASE("assign_reading_order_for_results: orphan stays in body even near header band",
          "[xy_cut][bucket]") {
  // An orphan with no layout match goes into the body bucket. If it
  // happens to sit at the very top of the page (y=10) the body XY-cut
  // places it above the body region, but it still reads AFTER any
  // explicit header.
  std::vector<LayoutBox> layout = {
      make_layout(50,  60, 450, 100, /*class_id=*/12),   // header
      make_layout(50, 200, 450, 280, /*class_id=*/22),   // body
  };
  std::vector<OCRResultItem> results = {
      make_result(50, 220, 200, 240, /*layout_id=*/1),    // body line
      make_result(60,  10, 200,  30, /*layout_id=*/-1),   // orphan near top
      make_result(50,  60, 450, 100, /*layout_id=*/0),    // header line
  };
  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 3);
  CHECK(order[0] == 2);  // header line first (top bucket)
  // Body bucket: orphan (y=10) above body line (y=220).
  CHECK(order[1] == 1);  // orphan
  CHECK(order[2] == 0);  // body line
}

TEST_CASE("assign_layout_ids synthesises SupplementaryRegion for orphans",
          "[layout_ids][supplementary]") {
  // Two real layout boxes; result #1 falls inside layout[0], result #2
  // falls inside layout[1], result #0 has its centroid OUTSIDE both —
  // that's the orphan case. After assign_layout_ids:
  //   - layout vector grows by one entry (index 2) tagged
  //     class_id == kSupplementaryRegionClassId
  //   - the synthetic block's bbox encloses the orphan's bbox
  //   - the orphan's layout_id points at the synthetic block
  //   - the matched results keep their original layout_id
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 200, 200),   // idx 0
      make_layout(300, 300, 400, 400),   // idx 1
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(500, 500, 540, 520, /*layout_id=*/-1), // orphan
      make_result(110, 110, 190, 190, /*layout_id=*/-1), // → layout[0]
      make_result(310, 310, 390, 390, /*layout_id=*/-1), // → layout[1]
  };

  turbo_ocr::assign_layout_ids(results, layout);

  REQUIRE(layout.size() == 3);
  CHECK(layout[2].class_id ==
        turbo_ocr::layout::kSupplementaryRegionClassId);
  CHECK(layout[2].id == 2);
  CHECK(turbo_ocr::layout::label_name(layout[2].class_id) ==
        "SupplementaryRegion");

  // Synthetic bbox covers the orphan's AABB exactly (single orphan).
  auto [sx0, sy0, sx1, sy1] = turbo_ocr::aabb(layout[2].box);
  CHECK(sx0 == 500);
  CHECK(sy0 == 500);
  CHECK(sx1 == 540);
  CHECK(sy1 == 520);

  CHECK(results[0].layout_id == 2);  // orphan → SupplementaryRegion
  CHECK(results[1].layout_id == 0);  // matched
  CHECK(results[2].layout_id == 1);  // matched
}

TEST_CASE("assign_layout_ids: SupplementaryRegion encloses ALL orphans",
          "[layout_ids][supplementary]") {
  // Multiple scattered orphans → one SupplementaryRegion whose bbox
  // is the minimum-enclosing rectangle of all of them.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 200, 200),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result( 50,  60,  80,  80, /*layout_id=*/-1),  // top-left orphan
      make_result(110, 110, 190, 190, /*layout_id=*/-1),  // matched
      make_result(500, 500, 540, 540, /*layout_id=*/-1),  // bottom-right orphan
      make_result(300,  20, 320,  40, /*layout_id=*/-1),  // top-right orphan
  };

  turbo_ocr::assign_layout_ids(results, layout);

  REQUIRE(layout.size() == 2);
  auto [sx0, sy0, sx1, sy1] = turbo_ocr::aabb(layout[1].box);
  // Min-enclosing of {50,60,80,80}, {500,500,540,540}, {300,20,320,40}
  CHECK(sx0 == 50);
  CHECK(sy0 == 20);
  CHECK(sx1 == 540);
  CHECK(sy1 == 540);

  CHECK(results[0].layout_id == 1);  // orphan → SupplementaryRegion
  CHECK(results[1].layout_id == 0);  // matched
  CHECK(results[2].layout_id == 1);
  CHECK(results[3].layout_id == 1);
}

TEST_CASE("assign_layout_ids: no orphans → no SupplementaryRegion appended",
          "[layout_ids][supplementary]") {
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(100, 100, 200, 200),
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(110, 110, 190, 190, /*layout_id=*/-1),
  };
  turbo_ocr::assign_layout_ids(results, layout);
  REQUIRE(layout.size() == 1);   // unchanged
  CHECK(results[0].layout_id == 0);
}

TEST_CASE("assign_layout_ids: empty layout stays empty (backward-compat)",
          "[layout_ids][supplementary]") {
  // When the caller did not request layout (empty input) we DO NOT
  // synthesise a SupplementaryRegion. The serializer then omits the
  // layout key + per-result layout_id keys entirely, keeping responses
  // byte-identical to pre-layout clients.
  std::vector<turbo_ocr::layout::LayoutBox> layout;
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(10, 20, 30, 40, /*layout_id=*/-1),
      make_result(50, 60, 70, 80, /*layout_id=*/-1),
  };
  turbo_ocr::assign_layout_ids(results, layout);
  REQUIRE(layout.empty());
  CHECK(results[0].layout_id == -1);
  CHECK(results[1].layout_id == -1);
}

TEST_CASE("assign_reading_order_for_results: orphans inside SupplementaryRegion "
          "still placed individually by XY-cut",
          "[layout_ids][supplementary][xy_cut]") {
  // Real layout that misses both result boxes — both become orphans,
  // get assigned to a synthesised SupplementaryRegion, and the
  // reading-order code must still emit them in geometric order rather
  // than treating the synthetic region as one indivisible block.
  std::vector<turbo_ocr::layout::LayoutBox> layout = {
      make_layout(800, 800, 900, 900),  // far-away real layout, contains nothing
  };
  std::vector<turbo_ocr::OCRResultItem> results = {
      make_result(50, 200, 100, 220, /*layout_id=*/-1),  // bottom
      make_result(50,  20, 100,  40, /*layout_id=*/-1),  // top
  };
  turbo_ocr::assign_layout_ids(results, layout);
  REQUIRE(layout.size() == 2);
  CHECK(results[0].layout_id == 1);
  CHECK(results[1].layout_id == 1);

  auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order.size() == 2);
  // Top result should come first geometrically.
  CHECK(order[0] == 1);
  CHECK(order[1] == 0);
}

// ---- Columns the plain cut cannot separate ---------------------------------

namespace {

// Text columns of stacked paragraphs whose lines tilt by `slope` (dy/dx);
// each column's x drifts by -slope * y like a skewed scan. Paragraphs are
// staggered per column so no Y gap crosses the whole page.
struct ColumnPage {
  std::vector<LayoutBox> layout;
  std::vector<OCRResultItem> results;
  std::vector<int> column;  // per result
};

ColumnPage make_column_page(const std::vector<int> &col_x0, int col_w,
                            double slope, int para_gap = 24) {
  constexpr int kParas = 4, kLines = 4, kLineH = 30, kLineStep = 44;
  const int para_h = kLines * kLineStep;
  ColumnPage p;
  for (size_t c = 0; c < col_x0.size(); ++c) {
    int y = 100 + static_cast<int>(c) * 60;
    for (int k = 0; k < kParas; ++k) {
      const int x0 = col_x0[c] + static_cast<int>(std::lround(-slope * (y + para_h / 2)));
      const int x1 = x0 + col_w;
      const int lid = static_cast<int>(p.layout.size());
      p.layout.push_back(make_layout(x0, y, x1, y + para_h));
      const int dy = static_cast<int>(std::lround(slope * (col_w - 10)));
      for (int l = 0; l < kLines; ++l) {
        const int ly = y + l * kLineStep + 6;
        OCRResultItem r;
        r.text = "c" + std::to_string(c) + "p" + std::to_string(k) + "l" + std::to_string(l);
        r.confidence = 0.9f;
        r.box = Box{{{{{x0 + 5, ly}}, {{x1 - 5, ly + dy}},
                      {{x1 - 5, ly + dy + kLineH}}, {{x0 + 5, ly + kLineH}}}}};
        r.layout_id = lid;
        p.results.push_back(r);
        p.column.push_back(static_cast<int>(c));
      }
      y += para_h + para_gap;
    }
  }
  return p;
}

bool reads_column_by_column(const std::vector<int> &order, const ColumnPage &p) {
  for (size_t i = 1; i < order.size(); ++i) {
    const auto a = static_cast<size_t>(order[i - 1]), b = static_cast<size_t>(order[i]);
    if (a >= p.column.size() || b >= p.column.size()) continue;
    if (p.column[b] < p.column[a]) return false;
    if (p.column[b] == p.column[a] && p.results[b].box[0][1] < p.results[a].box[0][1])
      return false;
  }
  return true;
}

// Whether the plain cut alone keeps the paragraph boxes column by column.
bool plain_cut_separates_columns(const ColumnPage &p) {
  std::vector<std::array<int, 4>> rects;
  for (const auto &lb : p.layout) {
    const auto [x0, y0, x1, y1] = turbo_ocr::aabb(lb.box);
    rects.push_back({x0, y0, x1, y1});
  }
  std::vector<int> idx(rects.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<int> out;
  recursive_xy_cut(rects, idx, out);
  for (size_t i = 1; i < out.size(); ++i)
    if (out[i] / 4 < out[i - 1] / 4) return false;  // 4 paragraphs per column
  return true;
}

} // namespace

TEST_CASE("assign_reading_order_for_results: skewed columns read column by column",
          "[xy_cut][columns]") {
  // ~1 degree of skew drifts each column edge ~15 px over the page, past
  // the 10 px gutters, so no vertical line clears every column.
  auto p = make_column_page({50, 510, 970}, 450, std::tan(1.0 * 3.14159265358979 / 180.0));
  REQUIRE_FALSE(plain_cut_separates_columns(p));
  const auto order = assign_reading_order_for_results(p.results, p.layout);
  REQUIRE(order.size() == p.results.size());
  CHECK(reads_column_by_column(order, p));
}

TEST_CASE("assign_reading_order_for_results: touching columns read column by column",
          "[xy_cut][columns]") {
  // Straight page, adjacent column boxes overlapping by 4 px.
  auto p = make_column_page({50, 496, 942}, 450, 0.0);
  REQUIRE_FALSE(plain_cut_separates_columns(p));
  const auto order = assign_reading_order_for_results(p.results, p.layout);
  REQUIRE(order.size() == p.results.size());
  CHECK(reads_column_by_column(order, p));
}

TEST_CASE("assign_reading_order_for_results: a box spanning two columns does not merge them",
          "[xy_cut][columns]") {
  // A stamp over the top of both columns bridges the gutter.
  auto p = make_column_page({50, 540}, 450, 0.0);
  const int stamp = static_cast<int>(p.layout.size());
  p.layout.push_back(make_layout(300, 110, 790, 250));
  p.results.push_back(make_result(320, 160, 770, 190, stamp));
  const auto order = assign_reading_order_for_results(p.results, p.layout);
  REQUIRE(order.size() == p.results.size());
  CHECK(std::find(order.begin(), order.end(), static_cast<int>(p.column.size())) != order.end());
  CHECK(reads_column_by_column(order, p));  // the stamp's own line is skipped
}

TEST_CASE("assign_reading_order_for_results: one column of touching paragraphs keeps y order",
          "[xy_cut][columns]") {
  auto p = make_column_page({50}, 450, 0.0, /*para_gap=*/-10);
  const auto order = assign_reading_order_for_results(p.results, p.layout);
  REQUIRE(order.size() == p.results.size());
  CHECK(reads_column_by_column(order, p));
}

// ---- Order predicted by the layout model -----------------------------------

TEST_CASE("assign_reading_order_for_results: the layout model's read_order wins over geometry",
          "[reading_order][model]") {
  // Geometry alone reads the left box first; the model says right first.
  std::vector<LayoutBox> layout = {make_layout(50, 100, 450, 300),
                                   make_layout(500, 100, 900, 300)};
  layout[0].read_order = 1;
  layout[1].read_order = 0;
  std::vector<OCRResultItem> results = {
      make_result(60, 110, 440, 140, 0), make_result(60, 150, 440, 180, 0),
      make_result(510, 110, 890, 140, 1), make_result(510, 150, 890, 180, 1)};
  const auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order == std::vector<int>{2, 3, 0, 1});
}

TEST_CASE("assign_reading_order_for_results: a line outside every box follows the box above it",
          "[reading_order][model]") {
  // Two columns read left then right; the orphan sits under the left box.
  std::vector<LayoutBox> layout = {make_layout(50, 100, 450, 300),
                                   make_layout(500, 100, 900, 600)};
  layout[0].read_order = 0;
  layout[1].read_order = 1;
  std::vector<OCRResultItem> results = {
      make_result(60, 110, 440, 140, 0),
      make_result(510, 110, 890, 140, 1),
      make_result(60, 400, 440, 430, -1),  // orphan, left column, below box 0
      make_result(510, 300, 890, 330, 1)};
  const auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order == std::vector<int>{0, 2, 1, 3});
}

TEST_CASE("assign_reading_order_for_results: a box without read_order falls back to geometry",
          "[reading_order][model]") {
  std::vector<LayoutBox> layout = {make_layout(50, 100, 450, 300),
                                   make_layout(500, 100, 900, 300)};
  layout[0].read_order = 1;  // layout[1] has none
  std::vector<OCRResultItem> results = {make_result(60, 110, 440, 140, 0),
                                        make_result(510, 110, 890, 140, 1)};
  const auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order == std::vector<int>{0, 1});
}

TEST_CASE("assign_reading_order_for_results: tied read_order values fall back to geometry",
          "[reading_order][model]") {
  // Every box ranked the same carries no order. The right box is listed
  // first; geometry reads the left one first.
  std::vector<LayoutBox> layout = {make_layout(500, 100, 900, 300),
                                   make_layout(50, 100, 450, 300)};
  layout[0].read_order = 0;
  layout[1].read_order = 0;
  std::vector<OCRResultItem> results = {make_result(510, 110, 890, 140, 0),
                                        make_result(60, 110, 440, 140, 1)};
  const auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order == std::vector<int>{1, 0});
}

TEST_CASE("assign_reading_order_for_results: paragraphs the model swaps within one column stay top to bottom",
          "[reading_order][model]") {
  // Three stacked paragraphs of one column; the model swaps the lower two.
  std::vector<LayoutBox> layout = {make_layout(50, 100, 450, 200),
                                   make_layout(50, 210, 450, 310),
                                   make_layout(50, 320, 450, 420)};
  layout[0].read_order = 0;
  layout[1].read_order = 2;
  layout[2].read_order = 1;
  std::vector<OCRResultItem> results = {make_result(60, 110, 440, 140, 0),
                                        make_result(60, 220, 440, 250, 1),
                                        make_result(60, 330, 440, 360, 2)};
  const auto order = assign_reading_order_for_results(results, layout);
  REQUIRE(order == std::vector<int>{0, 1, 2});
}
