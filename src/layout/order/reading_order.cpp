#include "turbo_ocr/layout/order/reading_order.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/layout/blocks/child_blocks.h"
#include "turbo_ocr/layout/blocks/match_unsorted.h"
#include "turbo_ocr/layout/blocks/text_line_cluster.h"

namespace turbo_ocr::layout {

namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
// Below this a gutter stays straight enough for the plain cut; above the max
// it is rotated content, not scanner skew.
constexpr double kMinSkewRad = 0.1 * kDegToRad;
constexpr double kMaxSkewRad = 5.0 * kDegToRad;
constexpr size_t kMinSkewLines = 8;
// mark_cross_layout is cubic in the worst case.
constexpr size_t kMaxCrossLayoutBlocks = 256;

// Median top-edge angle of lines long enough to carry one; 0 if unmeasurable.
double estimate_page_skew(const std::vector<OCRResultItem> &results) {
  std::vector<double> angles;
  angles.reserve(results.size());
  for (const auto &r : results) {
    const double dx = r.box[1][0] - r.box[0][0];
    const double dy = r.box[1][1] - r.box[0][1];
    const double h = std::hypot(double(r.box[3][0] - r.box[0][0]),
                                double(r.box[3][1] - r.box[0][1]));
    if (dx <= 0.0 || dx < 3.0 * h) continue;
    angles.push_back(std::atan2(dy, dx));
  }
  if (angles.size() < kMinSkewLines) return 0.0;
  auto mid = angles.begin() + angles.size() / 2;
  std::nth_element(angles.begin(), mid, angles.end());
  const double mag = std::abs(*mid);
  return (mag < kMinSkewRad || mag > kMaxSkewRad) ? 0.0 : *mid;
}

// A box is the axis-aligned hull of content tilted by theta: recover the
// content's own (w, h) from the hull, rotate its centre by -theta, and pull
// each x-edge in by edge_tol so glyph bleed doesn't bridge a narrow gutter.
void deskew_cut_rects(std::vector<std::array<int, 4>> &rects, double theta,
                      int edge_tol) {
  const double c = std::cos(theta), s = std::sin(theta), as = std::abs(s);
  const double det = c * c - as * as;
  std::vector<std::array<double, 4>> out(rects.size());
  double min_x = std::numeric_limits<double>::infinity(), min_y = min_x;
  for (size_t i = 0; i < rects.size(); ++i) {
    const auto &r = rects[i];
    const double W = r[2] - r[0], H = r[3] - r[1];
    const double w = std::clamp((W * c - H * as) / det, 1.0, std::max(1.0, W));
    const double h = std::clamp((H * c - W * as) / det, 1.0, std::max(1.0, H));
    const double cx = 0.5 * (r[0] + r[2]), cy = 0.5 * (r[1] + r[3]);
    const double rx = cx * c + cy * s, ry = -cx * s + cy * c;
    double x0 = rx - 0.5 * w + edge_tol, x1 = rx + 0.5 * w - edge_tol;
    if (x1 - x0 < 1.0) { x0 = rx - 0.5; x1 = rx + 0.5; }
    out[i] = {x0, ry - 0.5 * h, x1, ry + 0.5 * h};
    min_x = std::min(min_x, x0);
    min_y = std::min(min_y, out[i][1]);
  }
  // recursive_xy_cut treats a negative x_min as a right-to-left page.
  const double sx = std::max(0.0, -min_x), sy = std::max(0.0, -min_y);
  for (size_t i = 0; i < rects.size(); ++i) {
    rects[i] = {static_cast<int>(std::lround(out[i][0] + sx)),
                static_cast<int>(std::lround(out[i][1] + sy)),
                static_cast<int>(std::lround(out[i][2] + sx)),
                static_cast<int>(std::lround(out[i][3] + sy))};
  }
}

// PaddleX calculate_projection_overlap_ratio (mode "union"); axis 0 = x.
double projection_overlap(const std::array<int, 4> &a,
                          const std::array<int, 4> &b, int axis) {
  const int s = axis, e = axis + 2;
  const int ov = std::min(a[e], b[e]) - std::max(a[s], b[s]);
  if (ov <= 0) return 0.0;
  const int uni = std::max(a[e], b[e]) - std::min(a[s], b[s]);
  return uni > 0 ? static_cast<double>(ov) / uni : 0.0;
}

long long rect_area(const std::array<int, 4> &r) {
  return static_cast<long long>(std::max(0, r[2] - r[0])) *
         std::max(0, r[3] - r[1]);
}

double area_iou(const std::array<int, 4> &a, const std::array<int, 4> &b) {
  const long long iw = std::max(0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  const long long ih = std::max(0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const long long inter = iw * ih;
  const long long uni = rect_area(a) + rect_area(b) - inter;
  return uni > 0 ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
}

// PaddleX get_layout_structure for text blocks: a block that overlaps a larger
// one, or X-overlaps two side-by-side paragraphs, spans columns and would keep
// the X projection from ever reaching zero.
std::vector<char> mark_cross_layout(const std::vector<std::array<int, 4>> &rects,
                                    const std::vector<int> &line_height) {
  const size_t n = rects.size();
  std::vector<char> cross(n, 0);
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (rects[a][0] != rects[b][0]) return rects[a][0] < rects[b][0];
    return rects[a][2] - rects[a][0] < rects[b][2] - rects[b][0];
  });
  auto is_paragraph = [&](size_t k) {
    const auto &r = rects[k];
    return std::max(r[2] - r[0], r[3] - r[1]) > 8 * line_height[k];
  };
  auto side_by_side = [&](size_t a, size_t b) {
    return projection_overlap(rects[a], rects[b], 0) == 0.0 &&
           projection_overlap(rects[a], rects[b], 1) > 0.0;
  };
  std::vector<char> has_side(n, 0);
  for (size_t a = 0; a < n; ++a)
    for (size_t b = a + 1; b < n; ++b)
      if (side_by_side(a, b)) has_side[a] = has_side[b] = 1;
  for (size_t bi : order) {
    if (cross[bi]) continue;
    const auto &b = rects[bi];
    for (size_t ri : order) {
      if (ri == bi || cross[ri]) continue;
      const auto &r = rects[ri];
      if (area_iou(b, r) > 0.1 && rect_area(b) < rect_area(r)) {
        cross[bi] = 1;
        break;
      }
      if (!has_side[ri] || !is_paragraph(ri) ||
          projection_overlap(b, r, 0) <= 0.0)
        continue;
      for (size_t si : order) {
        if (si == bi || si == ri || cross[si]) continue;
        if (projection_overlap(b, rects[si], 0) > 0.0 && side_by_side(ri, si) &&
            is_paragraph(si)) {
          cross[bi] = 1;
          break;
        }
      }
      if (cross[bi]) break;
    }
  }
  return cross;
}

// Columns of `node` on the robust geometry, left to right, once its
// cross-layout blocks are set aside. Fewer than two means no split and
// flags nothing.
std::vector<std::vector<int>>
split_columns(const std::vector<int> &node,
              const std::vector<std::array<int, 4>> &rects,
              const std::vector<int> &line_height, std::vector<char> &cross) {
  std::vector<std::array<int, 4>> sub;
  std::vector<int> sub_lh;
  sub.reserve(node.size());
  sub_lh.reserve(node.size());
  for (int i : node) {
    sub.push_back(rects[static_cast<size_t>(i)]);
    sub_lh.push_back(line_height[static_cast<size_t>(i)]);
  }
  const std::vector<char> sub_cross = node.size() <= kMaxCrossLayoutBlocks
                                          ? mark_cross_layout(sub, sub_lh)
                                          : std::vector<char>(node.size(), 0);
  std::vector<size_t> kept;
  for (size_t k = 0; k < node.size(); ++k)
    if (!sub_cross[k]) kept.push_back(k);
  std::stable_sort(kept.begin(), kept.end(),
                   [&](size_t a, size_t b) { return sub[a][0] < sub[b][0]; });
  std::vector<std::vector<int>> groups;
  int cur_end = 0;
  for (size_t k : kept) {
    if (groups.empty() || sub[k][0] > cur_end) groups.emplace_back();
    groups.back().push_back(node[k]);
    cur_end = groups.back().size() == 1 ? sub[k][2] : std::max(cur_end, sub[k][2]);
  }
  if (groups.size() < 2) return {};
  for (size_t k = 0; k < node.size(); ++k)
    if (sub_cross[k]) cross[static_cast<size_t>(node[k])] = 1;
  return groups;
}

// Reading order as predicted by the layout model itself (PP-DocLayoutV3's
// read_order column) -- the ordering PaddleOCR-VL uses. Applies only when
// every real box carries one; lines outside every box go right after the box
// directly above them in the same column, else right before the one directly
// below, else after the nearest. `by_layout` holds each box's lines in row
// order. Returns false to fall back to the geometric cut.
bool emit_model_order(const std::vector<OCRResultItem> &results,
                      const std::vector<LayoutBox> &layout,
                      const std::vector<std::vector<int>> &by_layout,
                      std::vector<int> &out) {
  std::vector<int> boxes;
  for (size_t li = 0; li < layout.size(); ++li) {
    if (layout[li].class_id == kSupplementaryRegionClassId) continue;
    if (layout[li].read_order < 0) return false;
    boxes.push_back(static_cast<int>(li));
  }
  if (boxes.empty()) return false;
  std::stable_sort(boxes.begin(), boxes.end(), [&](int a, int b) {
    return layout[static_cast<size_t>(a)].read_order <
           layout[static_cast<size_t>(b)].read_order;
  });
  // A real ranking gives every box its own slot; ties (an export whose order
  // head collapsed to a constant) carry no order at all.
  for (size_t k = 1; k < boxes.size(); ++k)
    if (layout[static_cast<size_t>(boxes[k - 1])].read_order ==
        layout[static_cast<size_t>(boxes[k])].read_order)
      return false;
  std::vector<std::array<int, 4>> box_aabb(layout.size());
  for (int li : boxes) {
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[static_cast<size_t>(li)].box);
    box_aabb[static_cast<size_t>(li)] = {x0, y0, x1, y1};
  }
  // The model now and then swaps neighbouring paragraphs of one column, which
  // geometry never gets wrong: consecutive boxes that share a column (x-extents
  // overlapping >= 80% of their union) stay top to bottom.
  auto same_column = [&](int a, int b) {
    const auto &A = box_aabb[static_cast<size_t>(a)];
    const auto &B = box_aabb[static_cast<size_t>(b)];
    const int ov = std::min(A[2], B[2]) - std::max(A[0], B[0]);
    const int uni = std::max(A[2], B[2]) - std::min(A[0], B[0]);
    return uni > 0 && ov * 10 >= uni * 8;
  };
  for (size_t i = 0; i < boxes.size();) {
    size_t j = i + 1;
    while (j < boxes.size() && same_column(boxes[j - 1], boxes[j])) ++j;
    std::stable_sort(boxes.begin() + static_cast<std::ptrdiff_t>(i),
                     boxes.begin() + static_cast<std::ptrdiff_t>(j),
                     [&](int a, int b) {
                       return box_aabb[static_cast<size_t>(a)][1] <
                              box_aabb[static_cast<size_t>(b)][1];
                     });
    i = j;
  }
  std::vector<int> rank(layout.size(), -1);
  for (size_t k = 0; k < boxes.size(); ++k)
    rank[static_cast<size_t>(boxes[k])] = static_cast<int>(k);

  // Orphan slot: (box rank, -1 before / +1 after, y0, x0, result index).
  struct Slot { int rank, side, y0, x0, ri; };
  std::vector<Slot> slots;
  for (size_t ri = 0; ri < results.size(); ++ri) {
    const int lid = results[ri].layout_id;
    if (lid >= 0 && static_cast<size_t>(lid) < layout.size() && rank[static_cast<size_t>(lid)] >= 0)
      continue;
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(results[ri].box);
    const int cy2 = y0 + y1;  // doubled centre, keeps the compare integral
    int above = -1, below = -1, nearest = -1;
    long long nearest_d = std::numeric_limits<long long>::max();
    int above_ov = 0, below_ov = 0;
    for (int li : boxes) {
      const auto &a = box_aabb[static_cast<size_t>(li)];
      const int ov = std::min(x1, a[2]) - std::max(x0, a[0]);
      if (ov > 0 && 2 * a[3] <= cy2 &&
          (above < 0 || a[3] > box_aabb[static_cast<size_t>(above)][3] ||
           (a[3] == box_aabb[static_cast<size_t>(above)][3] && ov > above_ov))) {
        above = li;
        above_ov = ov;
      }
      if (ov > 0 && 2 * a[1] >= cy2 &&
          (below < 0 || a[1] < box_aabb[static_cast<size_t>(below)][1] ||
           (a[1] == box_aabb[static_cast<size_t>(below)][1] && ov > below_ov))) {
        below = li;
        below_ov = ov;
      }
      const long long d = std::max({0, a[0] - x1, x0 - a[2]}) +
                          std::max({0, a[1] - y1, y0 - a[3]});
      if (d < nearest_d) { nearest_d = d; nearest = li; }
    }
    if (above >= 0) slots.push_back({rank[static_cast<size_t>(above)], 1, y0, x0, static_cast<int>(ri)});
    else if (below >= 0) slots.push_back({rank[static_cast<size_t>(below)], -1, y0, x0, static_cast<int>(ri)});
    else slots.push_back({rank[static_cast<size_t>(nearest)], 1, y0, x0, static_cast<int>(ri)});
  }
  std::stable_sort(slots.begin(), slots.end(), [](const Slot &a, const Slot &b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    if (a.side != b.side) return a.side < b.side;
    if (a.y0 != b.y0) return a.y0 < b.y0;
    return a.x0 < b.x0;
  });

  size_t s = 0;
  for (size_t k = 0; k < boxes.size(); ++k) {
    const int r = static_cast<int>(k);
    for (; s < slots.size() && slots[s].rank == r && slots[s].side < 0; ++s)
      out.push_back(slots[s].ri);
    for (int ri : by_layout[static_cast<size_t>(boxes[k])]) out.push_back(ri);
    for (; s < slots.size() && slots[s].rank == r; ++s) out.push_back(slots[s].ri);
  }
  return true;
}

} // namespace

// XY-cut over a subset of layout indices. Helper extracted so callers can
// reuse it on each priority bucket without duplicating the AABB build.
static void
xy_cut_subset(const std::vector<LayoutBox> &layout,
              const std::vector<int> &subset,
              std::vector<int> &out, int min_gap) {
  if (subset.empty()) return;
  std::vector<std::array<int, 4>> rects;
  rects.reserve(subset.size());
  for (int idx : subset) {
    auto [x0, y0, x1, y1] = aabb(layout[static_cast<size_t>(idx)].box);
    rects.push_back({x0, y0, x1, y1});
  }
  std::vector<int> local(subset.size());
  std::iota(local.begin(), local.end(), 0);
  std::vector<int> local_order;
  local_order.reserve(subset.size());
  recursive_xy_cut(rects, local, local_order, min_gap);

  // Defense in depth: degenerate inputs (overlapping AABBs, zero areas)
  // can drop indices from the recursion. Append any missed in input
  // order so callers always see a complete permutation of the subset.
  std::vector<char> seen(subset.size(), 0);
  for (int li : local_order) {
    if (li >= 0 && static_cast<size_t>(li) < seen.size()) seen[li] = 1;
  }
  for (size_t k = 0; k < local.size(); ++k) {
    if (!seen[k]) local_order.push_back(static_cast<int>(k));
  }

  for (int li : local_order) out.push_back(subset[static_cast<size_t>(li)]);
}

std::vector<int>
assign_reading_order(const std::vector<LayoutBox> &layout, int min_gap) {
  std::vector<int> result;
  if (layout.empty()) return result;

  // Class-aware bucketing: header → body → footer/reference. Each bucket
  // gets its own XY-cut so multi-line headers and reference lists keep
  // their internal order. PaddleX's xycut_enhanced does the same — page
  // furniture should not interleave with the body.
  std::array<std::vector<int>, 3> buckets;
  for (size_t i = 0; i < layout.size(); ++i) {
    int b = reading_priority_bucket(layout[i].class_id);
    buckets[static_cast<size_t>(b)].push_back(static_cast<int>(i));
  }

  result.reserve(layout.size());
  for (auto &subset : buckets) xy_cut_subset(layout, subset, result, min_gap);
  return result;
}

std::vector<int>
assign_reading_order_for_results(const std::vector<OCRResultItem> &results,
                                 std::vector<LayoutBox> &layout,
                                 int min_gap) {
  std::vector<int> out;
  out.reserve(results.size());
  if (results.empty()) return out;

  // Shared "no usable layout signal" exit: sort every result by
  // (y_center, x_center) and emit. Decorate-sort-undecorate keeps the key
  // a pure per-result computation instead of recomputing it inside the
  // comparator. Used by both fall-back branches below.
  const auto emit_yx_fallback = [&results, &out]() {
    struct K { int y4, x4, idx; };
    std::vector<K> keys;
    keys.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      int sx = 0, sy = 0;
      for (int k = 0; k < 4; ++k) {
        sx += results[i].box[k][0];
        sy += results[i].box[k][1];
      }
      keys.push_back({sy, sx, static_cast<int>(i)});
    }
    std::stable_sort(keys.begin(), keys.end(), [](const K &a, const K &b) {
      if (a.y4 != b.y4) return a.y4 < b.y4;
      return a.x4 < b.x4;
    });
    for (const auto &k : keys) out.push_back(k.idx);
  };

  // Layout-empty fast path: text-line detection boxes are line-level,
  // not paragraph-level. Running XY-cut on raw detection boxes can
  // over-split a one-column document into spurious "columns" the moment
  // there's a horizontal gap between two short lines. Without any layout
  // signal there's nothing better than y-then-x.
  if (layout.empty()) {
    emit_yx_fallback();
    return out;
  }

  // ----- Class-aware bucketing + augmented XY-cut over (layout ∪ orphans) ----
  //
  // The PP-DocLayoutV3 layout model emits 25 classes. PaddleX's
  // xycut_enhanced pipeline doesn't run all of them through one flat
  // XY-cut: page furniture (header / footer / footnote / reference /
  // vision_footnote) is hoisted out into top/bottom strata so it doesn't
  // interleave with the body, and the body proper runs through XY-cut.
  // A 'reference' block geometrically placed mid-page in a malformed
  // document still belongs at the end of the reading order.
  //
  // Inside each bucket we still need the orphan handling: results whose
  // centroid falls outside every layout region (page numbers the layout
  // model missed, OCR detections in the gutter, etc.) get a synthetic
  // XY-cut entry from their detection AABB so they land in their natural
  // geometric position rather than trailing the bucket.
  //
  // Orphans always go into the BODY bucket. They could in theory fall
  // inside the header/footer band of the page, but with no class signal
  // we bias toward the safer placement (let XY-cut decide their y/x slot
  // within the body). Headers and footers themselves are explicit layout
  // regions, not orphans.
  //
  // Tagged-rect kinds:
  //   0 = real layout region; payload = layout index
  //   1 = orphan result;       payload = result index
  struct AugRect {
    std::array<int, 4> aabb;
    int kind;
    int payload;
  };

  // Pre-compute layout AABBs (used by both the bucket sort and the XY-cut).
  std::vector<std::array<int, 4>> layout_aabb(layout.size());
  for (size_t li = 0; li < layout.size(); ++li) {
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[li].box);
    layout_aabb[li] = {x0, y0, x1, y1};
  }

  // Group results by their layout_id and pre-sort each group into row
  // order. The naive (y_center, x_center) sort fails on tables: cells in
  // the same row routinely have 1-3 pixels of y-jitter from text-line
  // detection, so a strict y-tiebreak interleaves columns. We bucket
  // y-centroids by a row tolerance derived from the median text-line
  // height of the group: tol = max(4, median_height/3). This is scale
  // invariant — works at 100 dpi and at 600 dpi alike. `floor(cy / tol)`
  // is a function of one input, giving the strict weak ordering
  // std::stable_sort requires.
  std::vector<std::vector<int>> by_layout(layout.size());
  for (size_t ri = 0; ri < results.size(); ++ri) {
    int lid = results[ri].layout_id;
    if (lid >= 0 && static_cast<size_t>(lid) < by_layout.size())
      by_layout[lid].push_back(static_cast<int>(ri));
  }
  // Per-group row tolerance from median bbox height.
  std::vector<int> group_row_tol(layout.size(), 8);
  for (size_t li = 0; li < by_layout.size(); ++li) {
    const auto &v = by_layout[li];
    if (v.size() < 2) continue;
    std::vector<int> heights;
    heights.reserve(v.size());
    for (int ri : v) {
      int y_min = INT_MAX, y_max = INT_MIN;
      for (int k = 0; k < 4; ++k) {
        y_min = std::min(y_min, results[ri].box[k][1]);
        y_max = std::max(y_max, results[ri].box[k][1]);
      }
      heights.push_back(std::max(1, y_max - y_min));
    }
    auto mid = heights.begin() + heights.size() / 2;
    std::nth_element(heights.begin(), mid, heights.end());
    int median_h = *mid;
    group_row_tol[li] = std::max(4, median_h / 3);
  }
  // Decorate-sort-undecorate: the (row, x) key is a pure function of each
  // result, so materialise it once per element rather than recomputing 8
  // corner sums per comparison. Keys are built in ascending result-index
  // order and stable_sort preserves that on ties, so the emitted
  // permutation is byte-identical to the in-comparator version. Groups of
  // size < 2 need no sort. The key buffer is reused across groups.
  struct RowKey { int row, xsum, ri; };
  std::vector<RowKey> row_keys;
  for (size_t li = 0; li < by_layout.size(); ++li) {
    auto &v = by_layout[li];
    if (v.size() < 2) continue;
    const int tol = group_row_tol[li];
    row_keys.clear();
    row_keys.reserve(v.size());
    for (int ri : v) {
      int sy = 0, sx = 0;
      for (int k = 0; k < 4; ++k) {
        sy += results[ri].box[k][1];
        sx += results[ri].box[k][0];
      }
      row_keys.push_back({(sy / 4) / tol, sx, ri});
    }
    std::stable_sort(row_keys.begin(), row_keys.end(),
                     [](const RowKey &a, const RowKey &b) {
                       if (a.row != b.row) return a.row < b.row;
                       return a.xsum < b.xsum;
                     });
    for (size_t i = 0; i < v.size(); ++i) v[i] = row_keys[i].ri;
  }

  // Mostly-orphans fast path: if very few results matched a layout box
  // (e.g. layout model nearly missed the page) the body bucket would
  // run XY-cut over LINE-level detection boxes, which can spuriously
  // split a single column into "columns" the moment two short lines
  // have a horizontal gap. Layout AABBs typically dwarf line AABBs,
  // so a single matched layout containing 99 orphan rects also hits
  // the recursion's "no progress" bail-out at recursive_xy_cut(). Fall
  // back to the y-then-x sort whenever the matched fraction is below
  // 5% — that's well into "layout missed it" territory.
  size_t matched_count = 0;
  for (size_t ri = 0; ri < results.size(); ++ri) {
    int lid = results[ri].layout_id;
    if (lid < 0 || static_cast<size_t>(lid) >= layout.size()) continue;
    if (layout[static_cast<size_t>(lid)].class_id ==
        kSupplementaryRegionClassId) continue;
    ++matched_count;
  }
  // 5% threshold: fewer than ceil(0.05 * N) matches means "almost no
  // signal from layout" — better to fall back than risk the regression.
  size_t min_matches = std::max<size_t>(1, (results.size() * 5 + 99) / 100);
  if (matched_count < min_matches) {
    emit_yx_fallback();
    return out;
  }

  if (emit_model_order(results, layout, by_layout, out)) return out;

  // Cluster the OCR detection boxes into per-cell TextLines. This
  // populates each LayoutBox with direction, num_of_lines,
  // text_line_height, text_line_width, and seg_*_coordinate — which
  // feed the label-aware insertion (weighted_distance_insert
  // disperse term + get_seg_flag look-ahead) and the child-block
  // detection (real proximity threshold per block instead of the
  // height-over-text_line_height approximation).
  cluster_text_lines(results, layout);

  // Page-level direction (majority vote across text-class cells).
  // Drives axis selection in weighted_distance_insert and the
  // bucket-level sort key.
  const Direction page_direction = infer_page_direction(layout);

  // Page-level text_line_width / text_line_height as means across
  // text-class cells — the disperse-term scale for doc_title in
  // weighted_distance_insert, and the cross-cell proximity threshold
  // for child-block detection. Falls back to 0 when no text cell got
  // any clustered lines.
  int text_line_width = 0;
  int text_line_height = 0;
  {
    long long sum_w = 0, sum_h = 0;
    int n = 0;
    for (const auto &lb : layout) {
      if (lb.class_id != 22 /*text*/) continue;
      if (lb.text_line_height <= 0) continue;
      sum_w += lb.text_line_width;
      sum_h += lb.text_line_height;
      ++n;
    }
    if (n > 0) {
      text_line_width = static_cast<int>(sum_w / n);
      text_line_height = static_cast<int>(sum_h / n);
    }
  }

  // A scan skewed ~1 degree drifts a column edge further than a narrow
  // gutter is wide; the column retry in run_bucket cuts in de-tilted
  // coordinates when the page measures skewed.
  const double skew_rad = page_direction == Direction::kHorizontal
                              ? estimate_page_skew(results)
                              : 0.0;
  const int edge_tol =
      std::max(2, static_cast<int>(std::lround(0.2 * text_line_height)));

  // Detect parent → children relationships once for the page; the
  // sidecar links survive across buckets so vision_footnote can stay
  // glued to its (body-bucket) parent vision block even if the bucket
  // sweep would otherwise emit them in different strata.
  const auto child_links = detect_child_blocks(layout, text_line_height);

  // Set of layout indices that ARE children of some parent. These
  // never participate in bucket collection or XY-cut directly — they
  // emit under their parent's slot via the emit loop's child splice.
  std::vector<char> is_child(layout.size(), 0);
  for (const auto &cl : child_links) {
    for (int ci : cl.child_indices) {
      if (ci >= 0 && static_cast<size_t>(ci) < is_child.size())
        is_child[static_cast<size_t>(ci)] = 1;
    }
  }

  // Process each bucket in TOP→BODY→BOTTOM order. Each bucket splits
  // layout boxes into:
  //   - regulars (kBody) → run through XY-cut
  //   - unsorted (titles, captions, vision, cross-refs, unordered) →
  //     inserted via match_unsorted_blocks AFTER the XY-cut so each
  //     uses the strategy keyed by its order label (weighted distance
  //     for titles/vision, manhattan for unordered, reference for
  //     cross-refs).
  // Orphan results (synthetic SupplementaryRegion members) are inlined
  // into the body bucket as XY-cut entries so they keep their
  // geometric placement.
  std::vector<char> emitted(results.size(), 0);
  auto run_bucket = [&](int bucket) {
    std::vector<AugRect> aug;
    std::vector<UnsortedBlock> unsorted;
    aug.reserve(layout.size());
    for (size_t li = 0; li < layout.size(); ++li) {
      if (layout[li].class_id == kSupplementaryRegionClassId) continue;
      if (is_child[li]) continue;  // emitted under its parent
      if (reading_priority_bucket(layout[li].class_id) != bucket) continue;
      const OrderLabel ol = order_label_for(layout[li].class_id);
      if (ol == OrderLabel::kBody) {
        aug.push_back({layout_aabb[li], 0, static_cast<int>(li)});
      } else {
        unsorted.push_back({static_cast<int>(li), layout_aabb[li], ol,
                            layout[li].class_id});
      }
    }
    if (bucket == 1) {
      for (size_t ri = 0; ri < results.size(); ++ri) {
        int lid = results[ri].layout_id;
        const bool is_orphan =
            (lid < 0) ||
            (static_cast<size_t>(lid) >= layout.size()) ||
            (layout[static_cast<size_t>(lid)].class_id ==
             kSupplementaryRegionClassId);
        if (is_orphan) {
          auto [x0, y0, x1, y1] = turbo_ocr::aabb(results[ri].box);
          aug.push_back({{x0, y0, x1, y1}, 1, static_cast<int>(ri)});
        }
      }
    }
    if (aug.empty() && unsorted.empty()) return;

    // 1. XY-cut over the regulars. For vertical-direction pages we
    //    mirror x coordinates around max_x BEFORE the cut so the
    //    algorithm's left-to-right behaviour produces a right-to-left
    //    column order (CJK tategaki). Coordinates are restored only
    //    in the final layout-idx mapping so callers see the original
    //    bbox.
    std::vector<std::array<int, 4>> rects;
    rects.reserve(aug.size());
    int mirror_x = 0;
    if (page_direction == Direction::kVertical) {
      for (const auto &a : aug) {
        mirror_x = std::max(mirror_x, a.aabb[2]);
      }
    }
    for (const auto &a : aug) {
      if (page_direction == Direction::kVertical) {
        // Reflect: new_x0 = mirror_x - old_x1, new_x1 = mirror_x - old_x0.
        // Keeps width identical; flips order so the rightmost column
        // becomes the leftmost in the cut input.
        rects.push_back({mirror_x - a.aabb[2], a.aabb[1],
                         mirror_x - a.aabb[0], a.aabb[3]});
      } else {
        rects.push_back(a.aabb);
      }
    }
    // Where the plain cut splits on neither axis it would read straight
    // across the columns; retry there on skew-corrected, edge-tolerant
    // geometry. Blocks that retry sets aside as cross-layout are inserted by
    // distance below.
    std::vector<char> cross(aug.size(), 0);
    std::vector<std::array<int, 4>> split_rects;
    std::vector<int> line_h;
    ColumnSplitFn split;
    if (bucket == 1 && page_direction == Direction::kHorizontal) {
      split_rects = rects;
      deskew_cut_rects(split_rects, skew_rad, edge_tol);
      line_h.resize(aug.size());
      for (size_t k = 0; k < aug.size(); ++k) {
        const auto &a = aug[k];
        const int lh =
            a.kind == 0 ? layout[static_cast<size_t>(a.payload)].text_line_height
                        : 0;
        line_h[k] = lh > 0 ? lh : std::max(1, a.aabb[3] - a.aabb[1]);
      }
      split = [&](const std::vector<int> &node) {
        return split_columns(node, split_rects, line_h, cross);
      };
    }
    std::vector<int> aug_indices(aug.size());
    std::iota(aug_indices.begin(), aug_indices.end(), 0);
    std::vector<int> aug_order;
    aug_order.reserve(aug.size());
    recursive_xy_cut(rects, aug_indices, aug_order, min_gap, split);
    std::vector<char> seen(aug.size(), 0);
    for (int ai : aug_order) {
      if (ai >= 0 && static_cast<size_t>(ai) < seen.size()) seen[ai] = 1;
    }
    for (size_t k = 0; k < aug.size(); ++k) {
      if (!seen[k] && !cross[k]) aug_order.push_back(static_cast<int>(k));
    }
    for (size_t k = 0; k < aug.size(); ++k) {
      if (!cross[k]) continue;
      const auto &a = aug[k];
      if (a.kind == 0) {
        unsorted.push_back({a.payload, a.aabb, OrderLabel::kCrossLayout,
                            layout[static_cast<size_t>(a.payload)].class_id});
      } else {
        unsorted.push_back({-2 - a.payload, a.aabb, OrderLabel::kCrossLayout, -1});
      }
    }

    // 2. Convert XY-cut output to a list of (UnsortedBlock + emit
    //    payload) so match_unsorted_blocks can mutate ordering. Layout
    //    AABB carries the original aug payload (kind, idx) via the
    //    `class_id` field's sign trick: positive class_id = real
    //    layout idx into outer layout vector; -1 sentinel is reserved
    //    for orphan rects (kind == 1) which we encode via order_label.
    std::vector<UnsortedBlock> sorted_blocks;
    sorted_blocks.reserve(aug.size());
    for (int ai : aug_order) {
      const auto &a = aug[static_cast<size_t>(ai)];
      if (a.kind == 0) {
        sorted_blocks.push_back({a.payload, a.aabb, OrderLabel::kBody,
                                  layout[static_cast<size_t>(a.payload)].class_id});
      } else {
        // Orphan rect: payload is a result index. Encode via negative
        // layout_idx so the emitter can recover it (-2 - ri).
        sorted_blocks.push_back({-2 - a.payload, a.aabb,
                                  OrderLabel::kBody, -1});
      }
    }

    // 3. Insert label-aware unsorted blocks.
    if (!unsorted.empty()) {
      match_unsorted_blocks(sorted_blocks, unsorted, text_line_width,
                              page_direction, layout);
    }

    // 4. Emit results. For each sorted layout entry, also splice in
    //    the layout's descendants (children, grandchildren, …) in
    //    top-down geometric order — mirrors PaddleX's
    //    insert_child_blocks. Descendants come from
    //    flatten_descendants which handles arbitrary tree depth and
    //    is cycle-safe.
    auto emit_layout = [&](int layout_idx) {
      for (int ri : by_layout[static_cast<size_t>(layout_idx)]) {
        if (!emitted[static_cast<size_t>(ri)]) {
          out.push_back(ri);
          emitted[static_cast<size_t>(ri)] = 1;
        }
      }
    };
    for (const auto &sb : sorted_blocks) {
      if (sb.layout_idx >= 0) {
        emit_layout(sb.layout_idx);
        const auto descendants =
            flatten_descendants(sb.layout_idx, child_links, layout);
        for (int ci : descendants) emit_layout(ci);
      } else {
        const int ri = -2 - sb.layout_idx;
        if (ri >= 0 && static_cast<size_t>(ri) < results.size() &&
            !emitted[static_cast<size_t>(ri)]) {
          out.push_back(ri);
          emitted[static_cast<size_t>(ri)] = 1;
        }
      }
    }
  };
  run_bucket(0);  // headers
  run_bucket(1);  // body (with orphans)
  run_bucket(2);  // footers / footnotes / references

  // Final defense in depth: a result whose layout_id pointed at a region
  // that somehow yielded no XY-cut entry across all three buckets.
  for (size_t ri = 0; ri < results.size(); ++ri) {
    if (!emitted[ri]) out.push_back(static_cast<int>(ri));
  }
  return out;
}

} // namespace turbo_ocr::layout
