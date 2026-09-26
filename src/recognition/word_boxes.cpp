#include "turbo_ocr/recognition/word_boxes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/geometry/perspective.h"

namespace turbo_ocr::recognition {

namespace {

// The line is re-sampled at its own height in pixels, kept within these
// bounds: tiny text is upsampled so a gap still spans a few columns, huge
// text is capped to bound the work.
constexpr int kMinLineH = 16;
constexpr int kMaxLineH = 160;
constexpr int kMaxLineW = 16384;

float edge_len(const Box &b, int i, int j) {
  const float dx = static_cast<float>(b[i][0] - b[j][0]);
  const float dy = static_cast<float>(b[i][1] - b[j][1]);
  return std::sqrt(dx * dx + dy * dy);
}

// Widest run of columns in [lo, hi) whose ink is at most `floor_ink`, as
// [begin, end). Empty when no column qualifies.
std::pair<int, int> widest_run(const std::vector<int> &col, int lo, int hi,
                               int floor_ink) {
  std::pair<int, int> best{lo, lo};
  int begin = -1;
  for (int x = lo; x <= hi; ++x) {
    const bool clear = x < hi && col[static_cast<size_t>(x)] <= floor_ink;
    if (clear && begin < 0) begin = x;
    if (!clear && begin >= 0) {
      if (x - begin > best.second - best.first) best = {begin, x};
      begin = -1;
    }
  }
  return best;
}

// From an anchor column outwards (dir -1 = left, +1 = right) past the word's
// ink to the first clear run of `gap` columns, or to the crop border; returns
// that side's boundary. Ink beyond such a gap belongs to a neighbour the
// line box grazed, not to the line. Clear columns before any ink is met do
// not count: the anchor itself may sit just outside its glyph.
int outer_edge(const std::vector<int> &col, int anchor, int dir, int gap,
               int noise) {
  const int w = static_cast<int>(col.size());
  bool seen_ink = false;
  int run = 0;
  for (int x = anchor; x >= 0 && x < w; x += dir) {
    if (col[static_cast<size_t>(x)] > noise) {
      seen_ink = true;
      run = 0;
    } else if (seen_ink && ++run >= gap) {
      return dir < 0 ? x + run : x - run + 1;
    }
  }
  return dir < 0 ? 0 : w;
}

// Whether the light pixels are the ground: most of the light inside `core`
// reaches the crop's border through light, the way paper runs between and
// around heavy letters. A light glyph on a dark ground is enclosed by it.
bool light_is_ground(const cv::Mat &ink, const cv::Rect &core) {
  cv::Mat lab;
  const cv::Mat light = 1 - ink;
  const int n = cv::connectedComponents(light, lab, 4, CV_32S);
  std::vector<unsigned char> border(static_cast<size_t>(n), 0);
  for (int x = 0; x < lab.cols; ++x) {
    border[static_cast<size_t>(lab.at<int>(0, x))] = 1;
    border[static_cast<size_t>(lab.at<int>(lab.rows - 1, x))] = 1;
  }
  for (int y = 0; y < lab.rows; ++y) {
    border[static_cast<size_t>(lab.at<int>(y, 0))] = 1;
    border[static_cast<size_t>(lab.at<int>(y, lab.cols - 1))] = 1;
  }
  int light_px = 0, reaching = 0;
  for (int y = core.y; y < core.y + core.height; ++y) {
    const int *row = lab.ptr<int>(y);
    for (int x = core.x; x < core.x + core.width; ++x) {
      if (row[x] == 0) continue;  // label 0 is the ink
      ++light_px;
      reaching += border[static_cast<size_t>(row[x])];
    }
  }
  return 2 * reaching >= light_px;
}

}  // namespace

std::vector<OCRWord> locate_words(const cv::Mat &img, const Box &line,
                                  const std::vector<CtcWord> &words,
                                  const std::vector<Box> &page_lines) {
  std::vector<OCRWord> out;
  if (words.empty() || img.empty()) return out;

  // compute_crop_transform's orientation rule: after its vertical-text swap
  // the text runs along the crop's x axis, the other edge is its height.
  const float e01 = edge_len(line, 0, 1);
  const float e03 = edge_len(line, 0, 3);
  const bool vertical = e03 >= e01 * kVerticalAspectRatio;
  const int h = std::clamp(static_cast<int>(std::lround(vertical ? e01 : e03)),
                           kMinLineH, kMaxLineH);
  const CropTransform ct = compute_crop_transform(line, h, kMaxLineW);
  const int w = ct.crop_width;

  // The crop reaches `m` pixels past the line box on every side, so glyphs
  // the detector's box clips (a descender, a comma's tail, an accent, a tall
  // bracket) are still whole. Extended crop pixel (x, y) is line-crop pixel
  // (x - m, y - m); the line box itself is the `core` rectangle.
  const int m = std::max(2, h / 2);
  const cv::Rect core(m, m, w, h);
  const cv::Matx33f m_inv(ct.M_inv[0], ct.M_inv[1], ct.M_inv[2],
                          ct.M_inv[3], ct.M_inv[4], ct.M_inv[5],
                          ct.M_inv[6], ct.M_inv[7], ct.M_inv[8]);
  const cv::Matx33f shift(1, 0, static_cast<float>(-m),
                          0, 1, static_cast<float>(-m), 0, 0, 1);
  cv::Mat crop;
  cv::warpPerspective(img, crop, m_inv * shift, cv::Size(w + 2 * m, h + 2 * m),
                      cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                      cv::BORDER_REPLICATE);
  cv::Mat gray;
  if (crop.channels() == 3)
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
  else if (crop.channels() == 4)
    cv::cvtColor(crop, gray, cv::COLOR_BGRA2GRAY);
  else
    gray = crop;
  if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);

  // Ink: Otsu's threshold of the line itself, applied to the margins too.
  cv::Mat ink, scratch;
  const double thr = cv::threshold(gray(core), scratch, 0, 1,
                                   cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::threshold(gray, ink, thr, 1, cv::THRESH_BINARY_INV);
  // Dark text on light paper is the norm. A box that is mostly dark holds
  // either light text on a dark ground or heavy type the box fits tightly,
  // and the ground is the side that surrounds the other.
  if (2 * cv::countNonZero(ink(core)) > w * h && !light_is_ground(ink, core))
    ink = 1 - ink;
  const auto W = w + 2 * m, H = h + 2 * m;

  // Other detected lines reaching into the crop claim the ink inside their own
  // box that lies nearer their middle than this line's -- a headline's letters
  // stay above the small line's box, so it can never take them. Ink claimed
  // beyond the other line's middle ("deep") means a blob runs right through
  // that line: glyphs of two lines touching, not one glyph.
  cv::Mat claimed = cv::Mat::zeros(ink.size(), CV_8U);  // 1 + claiming line
  cv::Mat deep = cv::Mat::zeros(ink.size(), CV_8U);
  cv::Mat boxed = cv::Mat::zeros(ink.size(), CV_8U);  // inside any other box
  bool any_claim = false;
  struct Midline { cv::Point2f left; float slope = 0.0f; };
  std::vector<Midline> others;
  const float own_mid = static_cast<float>(m) + 0.5f * static_cast<float>(h);
  {
    const cv::Matx33f to_crop = (m_inv * shift).inv();
    for (const Box &o : page_lines) {
      if (others.size() >= 254) break;
      if (o == line) continue;
      std::array<cv::Point2f, 4> q;
      for (int k = 0; k < 4; ++k) {
        const cv::Vec3f p = to_crop * cv::Vec3f(static_cast<float>(o[k][0]),
                                                static_cast<float>(o[k][1]), 1.0f);
        q[static_cast<size_t>(k)] = {p[0] / p[2], p[1] / p[2]};
      }
      float qx0 = q[0].x, qx1 = q[0].x, qy0 = q[0].y, qy1 = q[0].y;
      for (const auto &p : q) {
        qx0 = std::min(qx0, p.x); qx1 = std::max(qx1, p.x);
        qy0 = std::min(qy0, p.y); qy1 = std::max(qy1, p.y);
      }
      if (qx1 < 0 || qy1 < 0 || qx0 >= static_cast<float>(W) || qy0 >= static_cast<float>(H))
        continue;
      std::array<cv::Point, 4> poly;
      for (int k = 0; k < 4; ++k)
        poly[static_cast<size_t>(k)] = {static_cast<int>(std::lround(q[static_cast<size_t>(k)].x)),
                                        static_cast<int>(std::lround(q[static_cast<size_t>(k)].y))};
      cv::fillConvexPoly(boxed, poly.data(), 4, 1);
      const cv::Point2f L = 0.5f * (q[0] + q[3]), R = 0.5f * (q[1] + q[2]);
      if (std::abs(R.x - L.x) < 1.0f) continue;
      const float slope = (R.y - L.y) / (R.x - L.x);
      if (std::abs(slope) > 0.6f) continue;  // not a line running this way
      const auto mid_at = [&](float x) { return L.y + slope * (x - L.x); };
      // The same line cut into two boxes shares this middle: nothing to claim.
      if (std::abs(mid_at(0.5f * static_cast<float>(W)) - own_mid) < 0.25f * static_cast<float>(h))
        continue;
      others.push_back({L, slope});
      const auto tag = static_cast<uchar>(others.size());
      cv::Mat inside = cv::Mat::zeros(ink.size(), CV_8U);
      cv::fillConvexPoly(inside, poly.data(), 4, 1);
      const int y0 = std::max(0, static_cast<int>(qy0));
      const int y1 = std::min(H, static_cast<int>(qy1) + 1);
      const int x0 = std::max(0, static_cast<int>(qx0));
      const int x1 = std::min(W, static_cast<int>(qx1) + 1);
      for (int y = y0; y < y1; ++y) {
        const uchar *ir = ink.ptr<uchar>(y), *mr = inside.ptr<uchar>(y);
        uchar *cr = claimed.ptr<uchar>(y), *dr = deep.ptr<uchar>(y);
        for (int x = x0; x < x1; ++x) {
          if (!ir[x] || !mr[x]) continue;
          const float fy = static_cast<float>(y), oy = mid_at(static_cast<float>(x));
          if (std::abs(fy - oy) >= std::abs(fy - own_mid)) continue;
          if (cr[x] && std::abs(fy - others[cr[x] - 1u].left.y -
                                others[cr[x] - 1u].slope *
                                    (static_cast<float>(x) - others[cr[x] - 1u].left.x)) <=
                           std::abs(fy - oy))
            continue;  // a nearer line already claims it
          cr[x] = tag;
          any_claim = true;
          if ((oy - own_mid) * (fy - oy) > 0) dr[x] = 1;
        }
      }
    }
  }

  // Glyphs = connected ink components. One is the line's own when it stays
  // within the margin and most of it lies inside the line box -- a glyph the
  // box clips stays, a letter of the line above reaching down with its
  // descender does not -- or, for a short stroke, when its middle is on the
  // box's edge (the bar under a "⊆").
  // Specks are no glyph at all.
  cv::Mat labels, stats, centroids;
  int ncomp = cv::connectedComponentsWithStats(ink, labels, stats, centroids, 8,
                                               CV_32S);
  // A blob reaching deep into another line may hold glyphs of both, joined
  // by bleed. Where its row profile shows a clear valley between the two
  // middles -- a hairline of bleed -- it is cut there, and the part past the
  // cut leaves this line's ink.
  std::vector<int> claimed_px(static_cast<size_t>(ncomp), 0);
  if (any_claim) {
    std::vector<int> deep_px(static_cast<size_t>(ncomp), 0);
    std::vector<int> taken(static_cast<size_t>(ncomp), 0);
    std::vector<std::vector<int>> by_line(static_cast<size_t>(ncomp));
    for (int y = 0; y < H; ++y) {
      const int *lr = labels.ptr<int>(y);
      const uchar *dr = deep.ptr<uchar>(y), *cr = claimed.ptr<uchar>(y);
      for (int x = 0; x < W; ++x) {
        if (cr[x]) ++taken[static_cast<size_t>(lr[x])];
        if (dr[x]) {
          const auto c = static_cast<size_t>(lr[x]);
          ++deep_px[c];
          if (by_line[c].empty()) by_line[c].assign(others.size() + 1, 0);
          ++by_line[c][cr[x]];
        }
      }
    }
    bool split = false;
    for (int c = 1; c < ncomp; ++c) {
      const auto cu = static_cast<size_t>(c);
      const int area = stats.at<int>(c, cv::CC_STAT_AREA);
      // Both lines' ink: some reaching past the other line's middle, and a
      // real share left on this line's side (a glyph of the other line alone
      // is all its own and is simply the other line's).
      if (deep_px[cu] < std::max(3, area / 50) || 4 * (area - taken[cu]) < area)
        continue;
      const auto j = static_cast<size_t>(
          std::max_element(by_line[cu].begin() + 1, by_line[cu].end()) - by_line[cu].begin() - 1);
      const Midline &o = others[j];
      // Position of a pixel between this line's middle (0) and the other's (1).
      const auto frac = [&](int x, int y) {
        const float oy = o.left.y + o.slope * (static_cast<float>(x) - o.left.x);
        return (static_cast<float>(y) - own_mid) / (oy - own_mid);
      };
      const int cx0 = stats.at<int>(c, cv::CC_STAT_LEFT), cy0 = stats.at<int>(c, cv::CC_STAT_TOP);
      const int cx1 = cx0 + stats.at<int>(c, cv::CC_STAT_WIDTH);
      const int cy1 = cy0 + stats.at<int>(c, cv::CC_STAT_HEIGHT);
      const float cx = 0.5f * static_cast<float>(cx0 + cx1);
      const float span = std::abs(o.left.y + o.slope * (cx - o.left.x) - own_mid);
      // Bins at least two rows tall: finer ones leave empty bins between rows
      // of one continuous stroke, which would pass for a gap. The profile runs
      // on past the other line's middle, so its glyph counts whole; the cut
      // itself lies between the two middles.
      const int nb = std::max(4, static_cast<int>(span / 2.0f));
      const int nb_all = nb + nb / 2;
      std::vector<int> prof(static_cast<size_t>(nb_all), 0);
      for (int y = cy0; y < cy1; ++y) {
        const int *lr = labels.ptr<int>(y);
        for (int x = cx0; x < cx1; ++x) {
          if (lr[x] != c) continue;
          const float f = frac(x, y);
          if (f >= 0.0f && f < 1.5f)
            ++prof[std::min(static_cast<size_t>(f * static_cast<float>(nb)), prof.size() - 1)];
        }
      }
      // The valley lies inside the blob, with its ink on both sides.
      size_t lo = 0, hi = prof.size();
      while (lo < hi && prof[lo] == 0) ++lo;
      while (hi > lo && prof[hi - 1] == 0) --hi;
      if (hi - lo < 3) continue;
      size_t best = lo + 1;
      for (size_t k = lo + 1; k + 1 < hi && k < static_cast<size_t>(nb); ++k)
        if (prof[k] < prof[best] ||
            (prof[k] == prof[best] &&
             std::abs(static_cast<int>(k) - nb / 2) < std::abs(static_cast<int>(best) - nb / 2)))
          best = k;
      const int before = *std::max_element(prof.begin() + static_cast<std::ptrdiff_t>(lo),
                                           prof.begin() + static_cast<std::ptrdiff_t>(best));
      const int after = *std::max_element(prof.begin() + static_cast<std::ptrdiff_t>(best) + 1,
                                          prof.begin() + static_cast<std::ptrdiff_t>(hi));
      // No clear valley -- a bridge as thick as a stroke, or this line's own
      // ascender reaching into an overlapping box -- keeps the blob whole.
      if (3 * prof[best] > std::min(before, after)) continue;
      const float cut = (static_cast<float>(best) + 0.5f) / static_cast<float>(nb);
      for (int y = cy0; y < cy1; ++y) {
        const int *lr = labels.ptr<int>(y);
        uchar *ir = ink.ptr<uchar>(y);
        for (int x = cx0; x < cx1; ++x)
          if (lr[x] == c && frac(x, y) > cut) ir[x] = 0;
      }
      split = true;
    }
    if (split)
      ncomp = cv::connectedComponentsWithStats(ink, labels, stats, centroids, 8,
                                               CV_32S);
    claimed_px.assign(static_cast<size_t>(ncomp), 0);
    for (int y = 0; y < H; ++y) {
      const int *lr = labels.ptr<int>(y);
      const uchar *cr = claimed.ptr<uchar>(y), *ir = ink.ptr<uchar>(y);
      for (int x = 0; x < W; ++x)
        if (cr[x] && ir[x]) ++claimed_px[static_cast<size_t>(lr[x])];
    }
  }
  // Extended coordinates; `label` is the component.
  struct Glyph { int x0, y0, x1, y1, core_x0, core_x1, cx, label; };
  std::vector<int> in_core(static_cast<size_t>(ncomp), 0);
  std::vector<int> core_x0(static_cast<size_t>(ncomp), w + 2 * m);
  std::vector<int> core_x1(static_cast<size_t>(ncomp), -1);
  for (int y = m; y < m + h; ++y) {
    const int *row = labels.ptr<int>(y);
    for (int x = m; x < m + w; ++x) {
      const auto c = static_cast<size_t>(row[x]);
      ++in_core[c];
      core_x0[c] = std::min(core_x0[c], x);
      core_x1[c] = std::max(core_x1[c], x + 1);
    }
  }
  const int min_area = std::max(3, h * h / 400);
  std::vector<char> own(static_cast<size_t>(ncomp), 0);
  std::vector<Glyph> glyphs;
  std::vector<Glyph> theirs;  // glyphs another line claimed
  for (int c = 1; c < ncomp; ++c) {
    const int area = stats.at<int>(c, cv::CC_STAT_AREA);
    const int y0 = stats.at<int>(c, cv::CC_STAT_TOP);
    const int gh = stats.at<int>(c, cv::CC_STAT_HEIGHT);
    const int core_px = in_core[static_cast<size_t>(c)];
    if (area < min_area || core_px == 0) continue;
    // Running through the whole margin to the crop's edge -- half a line
    // height past the box -- is no glyph of this line: a frame, a column
    // rule, a border.
    if (y0 == 0 || y0 + gh >= h + 2 * m) continue;
    const int mid2 = 2 * y0 + gh;  // doubled vertical middle
    const int tol2 = 2 * std::max(1, h / 8);
    const bool edge_stroke = 4 * gh <= h && mid2 >= 2 * m - tol2 &&
                             mid2 < 2 * (m + h) + tol2;
    const int x0 = stats.at<int>(c, cv::CC_STAT_LEFT);
    if (2 * claimed_px[static_cast<size_t>(c)] >= area) {  // another line's
      theirs.push_back({x0, y0, x0 + stats.at<int>(c, cv::CC_STAT_WIDTH), y0 + gh, 0, 0, 0, c});
      continue;
    }
    if (2 * core_px < area && !edge_stroke) continue;
    own[static_cast<size_t>(c)] = 1;
    glyphs.push_back({x0, y0, x0 + stats.at<int>(c, cv::CC_STAT_WIDTH), y0 + gh,
                      core_x0[static_cast<size_t>(c)], core_x1[static_cast<size_t>(c)],
                      static_cast<int>(centroids.at<double>(c, 0)), c});
  }
  // A box that stops at the capitals' height leaves their accents -- and the
  // top strokes of CJK characters -- above it: a small, whole mark mostly
  // above the box and right over one of the line's own glyphs is that
  // glyph's. (The line above reaching down is a whole letter, too tall to
  // pass; and only above: the next line's accents and i-dots sit below.)
  const size_t n_own = glyphs.size();
  const int touch = std::max(2, h / 6);
  for (int c = 1; c < ncomp; ++c) {
    if (own[static_cast<size_t>(c)]) continue;
    const int area = stats.at<int>(c, cv::CC_STAT_AREA);
    const int x0 = stats.at<int>(c, cv::CC_STAT_LEFT);
    const int x1 = x0 + stats.at<int>(c, cv::CC_STAT_WIDTH);
    const int y0 = stats.at<int>(c, cv::CC_STAT_TOP);
    const int y1 = y0 + stats.at<int>(c, cv::CC_STAT_HEIGHT);
    if (area < min_area || y0 == 0 || 3 * (y1 - y0) > h || y0 + y1 >= 2 * m ||
        2 * claimed_px[static_cast<size_t>(c)] >= area)
      continue;
    for (size_t g = 0; g < n_own; ++g) {
      const auto &o = glyphs[g];
      if (std::min(x1, o.x1) > std::max(x0, o.x0) && o.y0 >= y1 - 1 &&
          o.y0 - y1 <= touch) {
        glyphs.push_back({x0, y0, x1, y1, std::max(x0, m), std::min(x1, m + w),
                          static_cast<int>(centroids.at<double>(c, 0)), c});
        break;
      }
    }
  }

  // A small mark -- an i-dot, an accent, a period -- belongs with the nearest
  // full-size glyph: the next line's i-dots, reaching into this box where the
  // lines sit close, go with that line's letters. Small = under a third of the
  // median glyph's area.
  if (!theirs.empty()) {
    const auto gap = [](const Glyph &a, const Glyph &b) {
      const int dx = std::max({0, a.x0 - b.x1, b.x0 - a.x1});
      const int dy = std::max({0, a.y0 - b.y1, b.y0 - a.y1});
      return dx * dx + dy * dy;
    };
    const auto area_of = [&](const Glyph &g) {
      return stats.at<int>(g.label, cv::CC_STAT_AREA);
    };
    std::vector<int> areas;
    for (const auto &g : glyphs) areas.push_back(area_of(g));
    for (const auto &g : theirs) areas.push_back(area_of(g));
    std::nth_element(areas.begin(), areas.begin() + areas.size() / 2, areas.end());
    const int median_area = areas[areas.size() / 2];
    const auto small = [&](const Glyph &g) { return 3 * area_of(g) < median_area; };
    std::vector<Glyph> kept;
    for (const auto &g : glyphs) {
      if (small(g)) {
        int mine = std::numeric_limits<int>::max(), other = mine;
        const Glyph *mine_g = nullptr, *other_g = nullptr;
        for (const auto &o : glyphs)
          if (&o != &g && !small(o) && gap(g, o) < mine) { mine = gap(g, o); mine_g = &o; }
        for (const auto &o : theirs)
          if (!small(o) && gap(g, o) < other) { other = gap(g, o); other_g = &o; }
        // Near-equal: a dot or accent sits on top of the glyph it belongs to.
        const auto under = [&](const Glyph *o) { return o && o->y0 >= g.y1 - 1; };
        if (other < mine || (other_g && 4 * other <= 9 * mine && under(other_g) &&
                             !under(mine_g))) {
          own[static_cast<size_t>(g.label)] = 0;
          continue;
        }
      }
      kept.push_back(g);
    }
    glyphs.swap(kept);
  }

  // Column profile of the line's own glyphs inside the line box.
  std::vector<int> col(static_cast<size_t>(w), 0);
  for (int y = m; y < m + h; ++y) {
    const int *row = labels.ptr<int>(y);
    for (int x = m; x < m + w; ++x)
      if (own[static_cast<size_t>(row[x])]) ++col[static_cast<size_t>(x - m)];
  }
  // A column with a pixel or two of glyph (an anti-aliased serif) is still a
  // gap between glyphs.
  const int noise = h / 24;

  const int n = static_cast<int>(words.size());
  auto at = [&](float frac) {
    return std::clamp(static_cast<int>(std::lround(frac * static_cast<float>(w))),
                      0, w - 1);
  };
  // Each word's column span [left, right) in line-crop coordinates.
  std::vector<int> left(static_cast<size_t>(n)), right(static_cast<size_t>(n));
  std::vector<int> gaps;
  for (int k = 1; k < n; ++k) {
    const int lo = at(words[static_cast<size_t>(k - 1)].last);
    const int hi = std::max(at(words[static_cast<size_t>(k)].first) + 1, lo + 1);
    // Across a clean gap the window's floor is 0; an underline running
    // through it raises the floor to the line's own thickness.
    const int floor_ink = std::max(
        noise, *std::min_element(col.begin() + lo, col.begin() + hi));
    const auto [gb, ge] = widest_run(col, lo, hi, floor_ink);
    right[static_cast<size_t>(k - 1)] = gb;
    left[static_cast<size_t>(k)] = ge;
    if (ge > gb) gaps.push_back(ge - gb);
  }
  // Ink past a clear run as wide as this line's own word spaces (at least a
  // third of the line height) is a neighbour's; anything closer -- a period
  // set apart in old justified print -- still belongs to the line. Where the
  // line's ink runs to the box's edge, the span runs on into the margin.
  int margin_gap = std::max(2, h / 3);
  int word_space = margin_gap;
  if (!gaps.empty()) {
    std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
    word_space = std::max(2, gaps[gaps.size() / 2]);
    margin_gap = std::max(margin_gap, word_space);
  }
  left[0] = outer_edge(col, at(words[0].first), -1, margin_gap, noise);
  if (left[0] == 0) left[0] = -m;
  auto &last_right = right[static_cast<size_t>(n - 1)];
  last_right = outer_edge(col, at(words[static_cast<size_t>(n - 1)].last), +1,
                          margin_gap, noise);
  if (last_right == w) last_right = w + m;

  // A box that stops short of its line clips the first or last letter, or
  // leaves it just outside. Where the line's ink runs to the box's end, the
  // glyphs continuing it across the margin are the line's too: in its band,
  // each closer to the one before than the line's own word space, and in no
  // other line's box. A stroke running past both the box's top and bottom is
  // a rule, and one the crop's side cuts off is too far out to tell.
  if (!glyphs.empty() && (left[0] == -m || last_right == w + m)) {
    std::vector<int> band_px(static_cast<size_t>(ncomp), 0);
    std::vector<int> boxed_px(static_cast<size_t>(ncomp), 0);
    for (int y = 0; y < H; ++y) {
      const int *row = labels.ptr<int>(y);
      const uchar *br = boxed.ptr<uchar>(y);
      const int in_band = y >= m && y < m + h ? 1 : 0;
      for (int x = 0; x < W; ++x) {
        const auto c = static_cast<size_t>(row[x]);
        band_px[c] += in_band;
        boxed_px[c] += br[x];
      }
    }
    std::vector<Glyph> cand;
    for (int c = 1; c < ncomp; ++c) {
      const auto cu = static_cast<size_t>(c);
      const int area = stats.at<int>(c, cv::CC_STAT_AREA);
      const int x0 = stats.at<int>(c, cv::CC_STAT_LEFT);
      const int x1 = x0 + stats.at<int>(c, cv::CC_STAT_WIDTH);
      const int y0 = stats.at<int>(c, cv::CC_STAT_TOP);
      const int y1 = y0 + stats.at<int>(c, cv::CC_STAT_HEIGHT);
      if (own[cu] || area < min_area || y0 == 0 || y1 >= H || (x0 >= m && x1 <= m + w) ||
          (y0 < m && y1 > m + h) || x0 == 0 || x1 >= W ||
          2 * claimed_px[cu] >= area || 2 * band_px[cu] < area || 2 * boxed_px[cu] >= area)
        continue;
      cand.push_back({x0, y0, x1, y1, x0, x1, static_cast<int>(centroids.at<double>(c, 0)), c});
    }
    int lo = W, hi = 0;
    for (const auto &g : glyphs) { lo = std::min(lo, g.x0); hi = std::max(hi, g.x1); }
    if (last_right == w + m) {
      std::sort(cand.begin(), cand.end(),
                [](const Glyph &a, const Glyph &b) { return a.x0 < b.x0; });
      for (const auto &g : cand) {
        if (g.x1 <= m + w || own[static_cast<size_t>(g.label)]) continue;
        if (g.x0 - hi >= word_space) break;
        glyphs.push_back(g);
        own[static_cast<size_t>(g.label)] = 1;
        hi = std::max(hi, g.x1);
      }
    }
    if (left[0] == -m) {
      std::sort(cand.begin(), cand.end(),
                [](const Glyph &a, const Glyph &b) { return a.x1 > b.x1; });
      for (const auto &g : cand) {
        if (g.x0 >= m || own[static_cast<size_t>(g.label)]) continue;
        if (lo - g.x1 >= word_space) break;
        glyphs.push_back(g);
        own[static_cast<size_t>(g.label)] = 1;
        lo = std::min(lo, g.x0);
      }
    }
  }

  const auto pt = [&](float u, float v) {
    const auto p = apply_crop_transform_inv(ct, u, v);
    return std::array<int, 2>{static_cast<int>(std::lround(p[0])),
                              static_cast<int>(std::lround(p[1]))};
  };
  // Word spans in extended coordinates.
  std::vector<int> sl(static_cast<size_t>(n)), sr(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    sl[static_cast<size_t>(k)] = left[static_cast<size_t>(k)] + m;
    sr[static_cast<size_t>(k)] =
        std::max(right[static_cast<size_t>(k)] + m, sl[static_cast<size_t>(k)] + 1);
  }
  // Each glyph belongs whole to the word nearest its ink's centre -- a
  // hairline tail or hook reaching back across the gap (a "j", a CJK "了")
  // included. Only a stroke running on into the next word by more than half
  // a line height (an underline, a dash across words) is shared, cut at each
  // span's sides. Glyphs centred past the outer spans are a neighbour's.
  struct Part { int word, x0, x1, y0, y1; };
  std::vector<Part> parts;
  const int overhang = std::max(2, h / 2);
  for (const auto &g : glyphs) {
    if (g.cx < sl.front() || g.cx >= sr.back()) continue;
    int owner = 0, best = std::numeric_limits<int>::max();
    for (int k = 0; k < n; ++k) {
      const auto ku = static_cast<size_t>(k);
      const int dist = std::max({0, sl[ku] - g.cx, g.cx - (sr[ku] - 1)});
      if (dist < best) { best = dist; owner = k; }
    }
    const auto ou = static_cast<size_t>(owner);
    if (g.core_x0 >= sl[ou] - overhang && g.core_x1 <= sr[ou] + overhang) {
      parts.push_back({owner, g.x0, g.x1, g.y0, g.y1});
      continue;
    }
    for (int k = 0; k < n; ++k) {
      const auto ku = static_cast<size_t>(k);
      if (g.core_x1 <= sl[ku] || g.core_x0 >= sr[ku]) continue;
      parts.push_back({k, g.core_x0 < sl[ku] ? sl[ku] : g.x0,
                       g.core_x1 > sr[ku] ? sr[ku] : g.x1, g.y0, g.y1});
    }
  }

  out.reserve(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    const int wl = sl[static_cast<size_t>(k)], wr = sr[static_cast<size_t>(k)];
    int x0 = wr, x1 = wl, y0 = h + 2 * m, y1 = 0;
    for (const auto &p : parts) {
      if (p.word != k) continue;
      x0 = std::min(x0, p.x0);
      x1 = std::max(x1, p.x1);
      y0 = std::min(y0, p.y0);
      y1 = std::max(y1, p.y1);
    }
    if (x0 >= x1) {  // no ink of its own: the span across the line box
      x0 = std::max(wl, m);
      x1 = std::min(wr, m + w);
      y0 = m;
      y1 = m + h;
    }
    // One pixel of margin for the anti-aliased rim the threshold drops; back
    // to line-crop coordinates (the right/bottom edges are exclusive).
    const float u0 = static_cast<float>(x0 - 1 - m);
    const float u1 = static_cast<float>(x1 - m);
    const float v0 = static_cast<float>(y0 - 1 - m);
    const float v1 = static_cast<float>(y1 - m);
    OCRWord word;
    word.text = words[static_cast<size_t>(k)].text;
    word.confidence = words[static_cast<size_t>(k)].score;
    // Same point order as the line box: compute_crop_transform maps the
    // box's point 0 to the crop's top-left, or, for vertical text, to its
    // top-right.
    if (ct.vertical) {
      word.box[0] = pt(u1, v0);
      word.box[1] = pt(u1, v1);
      word.box[2] = pt(u0, v1);
      word.box[3] = pt(u0, v0);
    } else {
      word.box[0] = pt(u0, v0);
      word.box[1] = pt(u1, v0);
      word.box[2] = pt(u1, v1);
      word.box[3] = pt(u0, v1);
    }
    out.push_back(std::move(word));
  }
  return out;
}

}  // namespace turbo_ocr::recognition
