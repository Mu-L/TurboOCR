#include <catch_amalgamated.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/recognition/ctc_decode.h"
#include "turbo_ocr/recognition/word_boxes.h"

using turbo_ocr::Box;
using turbo_ocr::OCRWord;
using turbo_ocr::recognition::CtcWord;
using turbo_ocr::recognition::ctc_argmax;
using turbo_ocr::recognition::ctc_greedy_decode;
using turbo_ocr::recognition::ctc_greedy_decode_raw;
using turbo_ocr::recognition::ctc_greedy_decode_words;
using turbo_ocr::recognition::locate_words;

// ---- ctc_greedy_decode_words ------------------------------------------------

namespace {
// index 0 = blank, 1 = "a", 2 = "b", 3 = "c", 4 = ".", 5 = "中", 6 = "文", 7 = " "
std::vector<std::string> word_labels() {
  return {"blank", "a", "b", "c", ".", "中", "文", " "};
}
}  // namespace

TEST_CASE("ctc_greedy_decode_words splits at spaces and keeps positions", "[ctc][words]") {
  const auto labels = word_labels();
  // a a _ b ␠ ␠ c . _ _  -> "ab" (t 0..3), "c." (t 6..7)
  const int indices[] = {1, 1, 0, 2, 7, 7, 3, 4, 0, 0};
  const float scores[] = {0.9f, 0.5f, 0.9f, 0.7f, 0.9f, 0.9f, 0.6f, 0.8f, 0.9f, 0.9f};
  // Ran at 80 px wide, text in the first 60: a timestep is 8 px = 8/60.
  const auto words = ctc_greedy_decode_words(indices, scores, 10, 80, 60, labels);
  REQUIRE(words.size() == 2);
  CHECK(words[0].text == "ab");
  CHECK(words[0].score == Catch::Approx((0.9f + 0.7f) / 2));
  CHECK(words[0].first == Catch::Approx(0.5f * 8 / 60));
  CHECK(words[0].last == Catch::Approx(3.5f * 8 / 60));
  CHECK(words[1].text == "c.");
  CHECK(words[1].score == Catch::Approx((0.6f + 0.8f) / 2));
  CHECK(words[1].first == Catch::Approx(6.5f * 8 / 60));
  CHECK(words[1].last == Catch::Approx(7.5f * 8 / 60));
  // The words are the line text split at its spaces.
  CHECK(ctc_greedy_decode(indices, scores, 10, labels).first == "ab c.");
}

TEST_CASE("ctc_greedy_decode_words makes every CJK character its own word", "[ctc][words]") {
  const auto labels = word_labels();
  const int indices[] = {5, 0, 6, 1, 2, 0, 5};  // 中 文 ab 中
  const float scores[] = {0.9f, 0.9f, 0.8f, 0.7f, 0.6f, 0.9f, 0.5f};
  const auto words = ctc_greedy_decode_words(indices, scores, 7, 56, 56, labels);
  REQUIRE(words.size() == 4);
  CHECK(words[0].text == "中");
  CHECK(words[1].text == "文");
  CHECK(words[2].text == "ab");
  CHECK(words[3].text == "中");
  CHECK(words[3].score == Catch::Approx(0.5f));
}

TEST_CASE("ctc_greedy_decode_words: blank or all-space lines give no words", "[ctc][words]") {
  const auto labels = word_labels();
  const int blanks[] = {0, 0, 0};
  const int spaces[] = {7, 0, 7};
  const float scores[] = {0.9f, 0.9f, 0.9f};
  CHECK(ctc_greedy_decode_words(blanks, scores, 3, 24, 24, labels).empty());
  CHECK(ctc_greedy_decode_words(spaces, scores, 3, 24, 24, labels).empty());
  CHECK(ctc_greedy_decode_words(nullptr, nullptr, 0, 24, 24, labels).empty());
}

TEST_CASE("ctc_argmax + ctc_greedy_decode equals ctc_greedy_decode_raw", "[ctc][words]") {
  const auto labels = word_labels();
  const int seq_len = 40, num_classes = static_cast<int>(labels.size());
  std::mt19937 rng(35);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> logits(static_cast<size_t>(seq_len * num_classes));
  for (auto &v : logits) v = u(rng);
  std::vector<int> idx(seq_len);
  std::vector<float> sc(seq_len);
  ctc_argmax(logits.data(), seq_len, num_classes, idx.data(), sc.data());
  const auto a = ctc_greedy_decode(idx.data(), sc.data(), seq_len, labels);
  const auto b = ctc_greedy_decode_raw(logits.data(), seq_len, num_classes, labels);
  CHECK(a.first == b.first);
  CHECK(a.second == Catch::Approx(b.second));
}

// ---- locate_words -------------------------------------------------------------

namespace {

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kScale = 1.0;
constexpr int kThick = 2;

struct RenderedLine {
  cv::Mat img;                 // BGR page
  std::vector<cv::Mat> masks;  // per-word ink, page-sized
  Box line{};                  // the detector's (padded) line box
  std::vector<CtcWord> ctc;    // anchors at the first/last characters
  std::vector<std::string> words;
};

// Where a recognizer's CTC peak sits inside a glyph: models differ, and a
// peak can land anywhere from the glyph's left edge to its right edge.
enum class Peak { kCentre, kLeftEdge, kRightEdge };

// Renders `words` on one baseline with `gap` px between them and builds CTC
// anchors at the first/last characters, the way a recognizer reports them.
// The line box pads the ink like DB's unclip does.
RenderedLine render_line(const std::vector<std::string> &words, int gap = 14,
                         int left_pad = 12, int right_pad = 12,
                         Peak peak = Peak::kCentre, int thick = kThick) {
  RenderedLine r;
  r.words = words;
  const int x_start = 60, baseline = 80;
  r.img = cv::Mat(160, 900, CV_8UC3, cv::Scalar(255, 255, 255));
  int x = x_start;
  std::vector<std::pair<int, int>> char_centres;  // per word: first, last
  for (const auto &w : words) {
    cv::Mat mask = cv::Mat::zeros(r.img.size(), CV_8U);
    cv::putText(mask, w, {x, baseline}, kFont, kScale, 255, thick, cv::LINE_AA);
    cv::putText(r.img, w, {x, baseline}, kFont, kScale, {0, 0, 0}, thick, cv::LINE_AA);
    r.masks.push_back(mask);
    int base = 0;
    const int wwidth = cv::getTextSize(w, kFont, kScale, thick, &base).width;
    const int first_w = cv::getTextSize(w.substr(0, 1), kFont, kScale, thick, &base).width;
    const int last_w = cv::getTextSize(w.substr(w.size() - 1), kFont, kScale, thick, &base).width;
    const int first_x = peak == Peak::kLeftEdge    ? x
                        : peak == Peak::kRightEdge ? x + first_w
                                                   : x + first_w / 2;
    const int last_x = peak == Peak::kLeftEdge    ? x + wwidth - last_w
                       : peak == Peak::kRightEdge ? x + wwidth
                                                  : x + wwidth - last_w / 2;
    char_centres.emplace_back(first_x, last_x);
    x += wwidth + gap;
  }
  cv::Mat all = cv::Mat::zeros(r.img.size(), CV_8U);
  for (const auto &m : r.masks) all |= m;
  const cv::Rect ink = cv::boundingRect(all);
  const int x0 = ink.x - left_pad, x1 = ink.x + ink.width + right_pad;
  const int y0 = ink.y - 10, y1 = ink.y + ink.height + 10;
  r.line[0] = {x0, y0};
  r.line[1] = {x1, y0};
  r.line[2] = {x1, y1};
  r.line[3] = {x0, y1};
  const float len = static_cast<float>(x1 - x0);
  for (size_t k = 0; k < words.size(); ++k)
    r.ctc.push_back({words[k], 0.9f,
                     static_cast<float>(char_centres[k].first - x0) / len,
                     static_cast<float>(char_centres[k].second - x0) / len});
  return r;
}

cv::Rect ink_rect(const cv::Mat &mask) {
  cv::Mat bin;
  cv::threshold(mask, bin, 96, 255, cv::THRESH_BINARY);  // the visible glyph
  return cv::boundingRect(bin);
}

// Moves the line box to `[x0, y0, x1, y1]`, keeping the anchors on the same
// page columns.
void refit_line(RenderedLine &r, int x0, int y0, int x1, int y1) {
  const float old_x0 = static_cast<float>(r.line[0][0]);
  const float old_len = static_cast<float>(r.line[1][0] - r.line[0][0]);
  const float len = static_cast<float>(x1 - x0);
  for (auto &c : r.ctc) {
    c.first = (old_x0 + c.first * old_len - static_cast<float>(x0)) / len;
    c.last = (old_x0 + c.last * old_len - static_cast<float>(x0)) / len;
  }
  r.line[0] = {x0, y0};
  r.line[1] = {x1, y0};
  r.line[2] = {x1, y1};
  r.line[3] = {x0, y1};
}

// The word box never cuts its word's ink and hugs it within `slack` px.
void check_tight(const OCRWord &w, const cv::Mat &mask, int slack = 2) {
  const auto [x0, y0, x1, y1] = turbo_ocr::aabb(w.box);
  const cv::Rect g = ink_rect(mask);
  INFO("word '" << w.text << "' box [" << x0 << "," << y0 << "," << x1 << ","
                << y1 << "] ink [" << g.x << "," << g.y << ","
                << g.x + g.width - 1 << "," << g.y + g.height - 1 << "]");
  CHECK(x0 <= g.x);
  CHECK(y0 <= g.y);
  CHECK(x1 >= g.x + g.width - 1);
  CHECK(y1 >= g.y + g.height - 1);
  CHECK(g.x - x0 <= slack);
  CHECK(g.y - y0 <= slack);
  CHECK(x1 - (g.x + g.width - 1) <= slack);
  CHECK(y1 - (g.y + g.height - 1) <= slack);
}

}  // namespace

TEST_CASE("locate_words: each word gets a tight box, punctuation stays attached",
          "[words]") {
  const auto r = render_line({"Hello,", "world.", "Foo", "bar"});
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 4);
  for (size_t k = 0; k < words.size(); ++k) {
    CHECK(words[k].text == r.words[k]);
    CHECK(words[k].confidence == Catch::Approx(0.9f));
    check_tight(words[k], r.masks[k]);
  }
}

TEST_CASE("locate_words: anchors a timestep off still find the true gaps", "[words]") {
  auto r = render_line({"state", "of", "the", "art"});
  // A timestep is 8 px of the 48 px recognizer crop.
  const float line_h = static_cast<float>(r.line[3][1] - r.line[0][1]);
  const float line_w = static_cast<float>(r.line[1][0] - r.line[0][0]);
  const float ts = 8.0f / 48.0f * line_h / line_w;
  for (const float shift : {-ts, ts}) {
    auto ctc = r.ctc;
    for (auto &c : ctc) {  // outward, then inward
      c.first -= shift;
      c.last += shift;
    }
    const auto words = locate_words(r.img, r.line, ctc);
    REQUIRE(words.size() == 4);
    for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
  }
}

TEST_CASE("locate_words: CTC peaks at glyph edges, wide letters at the boundaries",
          "[words]") {
  // Early peaks put the window between two words over the whole last glyph
  // (a wide "m") plus the gap, late peaks over the gap plus the next word's
  // first glyph (a wide "W"): halfway between the anchors falls inside a
  // letter, the gap does not.
  for (const Peak peak : {Peak::kLeftEdge, Peak::kRightEdge}) {
    const auto r = render_line({"swim", "Wow", "mom", "Wim"}, 12, 12, 12, peak);
    const auto words = locate_words(r.img, r.line, r.ctc);
    REQUIRE(words.size() == 4);
    for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
  }
}

TEST_CASE("locate_words: a period set apart from its word stays in the box", "[words]") {
  // Old justified print sets ")." with a clear gap before the period, about a
  // quarter of the line box's height -- narrower than a word space, so it is
  // still part of the line, and the CTC puts the period's peak early, back on
  // the parenthesis.
  auto r = render_line({"see", "(paid)"});
  const cv::Rect paren = ink_rect(r.masks[1]);
  const int line_h = r.line[3][1] - r.line[0][1];
  const int dot_x = paren.x + paren.width + line_h / 4 + 1;
  const int dot_y = paren.y + paren.height - 8;
  cv::rectangle(r.img, {dot_x, dot_y}, {dot_x + 4, dot_y + 4}, {0, 0, 0}, cv::FILLED);
  cv::rectangle(r.masks[1], {dot_x, dot_y}, {dot_x + 4, dot_y + 4}, 255, cv::FILLED);
  r.words[1] = "(paid).";
  r.ctc[1].text = "(paid).";
  // The line box still ends a margin past the period; the anchors keep their
  // pixel positions in the wider box.
  const float old_len = static_cast<float>(r.line[1][0] - r.line[0][0]);
  const int x1 = dot_x + 5 + 12;
  r.line[1][0] = r.line[2][0] = x1;
  const float scale = old_len / static_cast<float>(x1 - r.line[0][0]);
  for (auto &c : r.ctc) {
    c.first *= scale;
    c.last *= scale;
  }
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  check_tight(words[0], r.masks[0]);
  check_tight(words[1], r.masks[1]);
}

TEST_CASE("locate_words: glyphs the line box clips are covered whole", "[words]") {
  // The detector's box stops above the descenders of "gypsy" and "jump".
  auto r = render_line({"gypsy", "jump", "on"});
  const int clip = ink_rect(r.masks[0]).y + ink_rect(r.masks[0]).height - 6;
  r.line[2][1] = r.line[3][1] = clip;
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 3);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
  CHECK(turbo_ocr::aabb(words[0].box)[3] > clip);  // past the line box
}

TEST_CASE("locate_words: the line above reaching into the box is not a word's ink",
          "[words]") {
  auto r = render_line({"plain", "words"});
  // Descenders of the previous line dip 5 px into this line's box.
  const int top = r.line[0][1];
  cv::putText(r.img, "gjpqy gjpqy gjpqy", {r.line[0][0], top - 2}, kFont, kScale,
              {0, 0, 0}, kThick, cv::LINE_AA);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a stroke on the line box's edge belongs to its word", "[words]") {
  // The bar under a "⊆" is a separate stroke that sits on the box's bottom
  // edge, its middle a pixel past it -- still the symbol's.
  auto r = render_line({"subset", "of"});
  const cv::Rect w0 = ink_rect(r.masks[0]);
  const int bottom = r.line[3][1];
  cv::line(r.img, {w0.x + 4, bottom}, {w0.x + 30, bottom}, {0, 0, 0}, 3);
  cv::line(r.masks[0], {w0.x + 4, bottom}, {w0.x + 30, bottom}, 255, 3);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  check_tight(words[0], r.masks[0]);
  check_tight(words[1], r.masks[1]);
}

TEST_CASE("locate_words: an accent above a tight line box stays with its letter",
          "[words]") {
  // The detector's box stops at the capitals' height; the accent over the
  // "E" -- a separate stroke -- lies wholly above it.
  auto r = render_line({"ECOLE", "NORMALE"});
  const cv::Rect w0 = ink_rect(r.masks[0]);
  r.line[0][1] = r.line[1][1] = w0.y + 1;
  const cv::Point a0(w0.x + 6, w0.y - 4), a1(w0.x + 13, w0.y - 9);
  cv::line(r.img, a0, a1, {0, 0, 0}, 3);
  cv::line(r.masks[0], a0, a1, 255, 3);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  check_tight(words[0], r.masks[0]);
  check_tight(words[1], r.masks[1]);
}

TEST_CASE("locate_words: a glyph overhanging the gap stays whole", "[words]") {
  // A "j" whose hairline tail reaches back across the gap (the hook of a
  // CJK "了" does the same): the tail's columns hold too little ink to be
  // anything but gap, yet the glyph is this word's, whole.
  auto r = render_line({"will", "just"}, 20);
  const cv::Rect j = ink_rect(r.masks[1]);
  const int base = j.y + j.height;
  const cv::Point h0(j.x + 2, base - 1), h1(j.x - 14, base - 1);
  cv::line(r.img, h0, h1, {0, 0, 0}, 1);
  cv::line(r.masks[1], h0, h1, 255, 1);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  check_tight(words[0], r.masks[0]);
  check_tight(words[1], r.masks[1]);
}

TEST_CASE("locate_words: in widely spaced justified text a far period still belongs",
          "[words]") {
  // Justification stretched the spaces to 34 px; the final period sits 20 px
  // out -- beyond a third of the line height, but closer than this line's own
  // word spaces, so it is the line's.
  auto r = render_line({"wide", "spaced", "line"}, 34);
  const cv::Rect last = ink_rect(r.masks[2]);
  const int dot_x = last.x + last.width + 20;
  const int dot_y = last.y + last.height - 5;
  cv::rectangle(r.img, {dot_x, dot_y}, {dot_x + 4, dot_y + 4}, {0, 0, 0}, cv::FILLED);
  cv::rectangle(r.masks[2], {dot_x, dot_y}, {dot_x + 4, dot_y + 4}, 255, cv::FILLED);
  r.words[2] = r.ctc[2].text = "line.";
  const float old_len = static_cast<float>(r.line[1][0] - r.line[0][0]);
  const int x1 = dot_x + 5 + 12;
  r.line[1][0] = r.line[2][0] = x1;
  const float scale = old_len / static_cast<float>(x1 - r.line[0][0]);
  for (auto &c : r.ctc) {
    c.first *= scale;
    c.last *= scale;
  }
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 3);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

namespace {
// A second line drawn under `r`, `gap` px below its ink, with its own box.
Box add_line_below(RenderedLine &r, const std::string &text, int gap, cv::Mat &mask) {
  const cv::Rect a = ink_rect(r.masks.front()) | ink_rect(r.masks.back());
  int base = 0;
  const cv::Size sz = cv::getTextSize(text, kFont, kScale, kThick, &base);
  const int baseline = a.y + a.height + gap + sz.height;
  mask = cv::Mat::zeros(r.img.size(), CV_8U);
  cv::putText(r.img, text, {a.x, baseline}, kFont, kScale, {0, 0, 0}, kThick, cv::LINE_AA);
  cv::putText(mask, text, {a.x, baseline}, kFont, kScale, 255, kThick, cv::LINE_AA);
  const cv::Rect b = ink_rect(mask);
  Box box{};
  box[0] = {b.x - 10, b.y - 8};
  box[1] = {b.x + b.width + 10, b.y - 8};
  box[2] = {b.x + b.width + 10, b.y + b.height + 8};
  box[3] = {b.x - 10, b.y + b.height + 8};
  return box;
}
}  // namespace

TEST_CASE("locate_words: a line box reaching over the next line leaves it alone",
          "[words]") {
  // The detector's box for this line runs down over most of the line below
  // (big type, tight leading); the page's other line owns those glyphs.
  auto r = render_line({"head", "line"});
  cv::Mat below;
  const Box next = add_line_below(r, "small print under it", 4, below);
  const cv::Rect b = ink_rect(below);
  r.line[2][1] = r.line[3][1] = b.y + b.height * 3 / 4;
  const auto words = locate_words(r.img, r.line, r.ctc, {r.line, next});
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a hairline of bleed between lines is cut where the lines meet",
          "[words]") {
  auto r = render_line({"bled", "ink"});
  cv::Mat below;
  const Box next = add_line_below(r, "touching line", 6, below);
  const cv::Rect a = ink_rect(r.masks[0]), b = ink_rect(below);
  const int x = a.x + 3;
  cv::line(r.img, {x, a.y + a.height - 2}, {x, b.y + 4}, {0, 0, 0}, 1);
  const auto words = locate_words(r.img, r.line, r.ctc, {r.line, next});
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) {
    const auto [x0, y0, x1, y1] = turbo_ocr::aabb(words[k].box);
    INFO("word " << words[k].text << " bottom " << y1 << " next line top " << b.y);
    CHECK(y1 < b.y + 2);  // at most the bridge's own end
  }
}

TEST_CASE("locate_words: a frame rule running through the line is no glyph", "[words]") {
  // Old newspapers box their mastheads: a vertical rule starts inside the
  // line box, beside the first word, and runs on far below it.
  auto r = render_line({"framed", "title"});
  const cv::Rect first = ink_rect(r.masks[0]);
  const int x = first.x - 6, top = r.line[0][1] + 4;
  cv::line(r.img, {x, top}, {x, r.img.rows - 1}, {0, 0, 0}, 3);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: light text on a dark background", "[words]") {
  auto r = render_line({"white", "on", "black"});
  cv::bitwise_not(r.img, r.img);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 3);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: light text on a dark band that ends just past its box",
          "[words]") {
  // A header cell: white type on a fill that stops 4 px outside the
  // detector's box, paper beyond -- most of the margin is light, yet the
  // light letters inside the fill are the text.
  auto r = render_line({"white", "band"});
  const cv::Rect band(r.line[0][0] - 4, r.line[0][1] - 4,
                      r.line[1][0] - r.line[0][0] + 8, r.line[3][1] - r.line[0][1] + 8);
  cv::Mat fill = r.img(band);
  cv::bitwise_not(fill, fill);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: heavy type filling a tight box is still dark ink", "[words]") {
  // Display type so heavy, in a detector box so tight (it clips the ink by
  // 3 px top and bottom), that ink covers most of the box. The paper around
  // the letters is still the ground.
  auto r = render_line({"summer", "news"}, 18, 12, 12, Peak::kCentre, /*thick=*/8);
  cv::Mat all = cv::Mat::zeros(r.img.size(), CV_8U);
  for (const auto &m : r.masks) all |= m;
  const cv::Rect g = ink_rect(all);
  refit_line(r, g.x - 4, g.y + 3, g.x + g.width + 4, g.y + g.height - 3);
  cv::Mat gray, dark;
  cv::cvtColor(r.img, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, dark, 128, 1, cv::THRESH_BINARY_INV);
  const cv::Rect core(r.line[0][0], r.line[0][1], r.line[1][0] - r.line[0][0],
                      r.line[3][1] - r.line[0][1]);
  REQUIRE(2 * cv::countNonZero(dark(core)) > core.area());  // mostly ink
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a box that stops short of the line keeps its end letters",
          "[words]") {
  // The detector's box ends inside the last letter and starts inside the
  // first: both letters lie mostly outside it, yet they are the line's.
  auto r = render_line({"short", "boxes"});
  const cv::Rect a = ink_rect(r.masks.front()), b = ink_rect(r.masks.back());
  refit_line(r, a.x + 8, r.line[0][1], b.x + b.width - 9, r.line[3][1]);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a column rule beside a box that stops short stays out",
          "[words]") {
  // The box starts inside the first letter, and a rule runs down the column
  // edge a few pixels before it -- past the box's top and bottom, short of
  // the crop's.
  auto r = render_line({"ruled", "column"});
  const cv::Rect a = ink_rect(r.masks.front());
  refit_line(r, a.x + 8, r.line[0][1], r.line[1][0], r.line[3][1]);
  const int x = a.x - 6;
  cv::line(r.img, {x, r.line[0][1] - 5}, {x, r.line[3][1] + 5}, {0, 0, 0}, 2);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: ink the crop's side cuts off does not continue the line",
          "[words]") {
  // A sliver of a rule's edge right where the crop ends, in the line's band:
  // too far out to tell what it is, so the first word stops at its letter.
  auto r = render_line({"sliver", "edge"});
  const cv::Rect a = ink_rect(r.masks.front());
  refit_line(r, a.x + 8, r.line[0][1], r.line[1][0], r.line[3][1]);
  const int m = (r.line[3][1] - r.line[0][1]) / 2;
  const int x = r.line[0][0] - m;
  cv::rectangle(r.img, {x - 1, a.y + 2}, {x + 2, a.y + a.height - 3}, {0, 0, 0}, cv::FILLED);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a box that stops short gains letters, not the next word",
          "[words]") {
  // The box ends inside "two"'s last letter, and the detector found no box
  // at all for the word after it: that word, a word space on (narrower than
  // a third of the line height), is not "two"'s.
  auto r = render_line({"one", "two", "lit"}, 8);
  const cv::Rect b = ink_rect(r.masks[1]);
  refit_line(r, r.line[0][0], r.line[0][1], b.x + b.width - 6, r.line[3][1]);
  r.ctc.pop_back();
  const auto words = locate_words(r.img, r.line, r.ctc, {r.line});
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a line cut into two boxes keeps each box's words apart",
          "[words]") {
  // One printed line the detector split in two, the cut in a narrow word
  // space: the second box's word lies in the first box's band, a few pixels
  // past its end, and stays with its own box.
  auto r = render_line({"split", "line", "here"}, 9);
  const cv::Rect second = ink_rect(r.masks[1]), third = ink_rect(r.masks[2]);
  const int y0 = r.line[0][1], y1 = r.line[3][1];
  Box other{};
  other[0] = {third.x - 2, y0};
  other[1] = {r.line[1][0], y0};
  other[2] = {r.line[1][0], y1};
  other[3] = {third.x - 2, y1};
  refit_line(r, r.line[0][0], y0, second.x + second.width + 2, y1);
  r.ctc.pop_back();
  const auto words = locate_words(r.img, r.line, r.ctc, {r.line, other});
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) check_tight(words[k], r.masks[k]);
}

TEST_CASE("locate_words: a neighbour's glyph inside the line box is left out", "[words]") {
  // The box reaches 40 px left of the text; a stray bar from the next column
  // sits in that margin, well clear of the first word.
  auto r = render_line({"column", "text"}, 14, /*left_pad=*/40);
  const int bar_x = r.line[0][0] + 4;
  cv::line(r.img, {bar_x, r.line[0][1] + 6}, {bar_x, r.line[3][1] - 6}, {0, 0, 0}, 3);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 2);
  check_tight(words[0], r.masks[0]);
  check_tight(words[1], r.masks[1]);
}

TEST_CASE("locate_words: an underline through the gaps does not merge words", "[words]") {
  auto r = render_line({"under", "lined", "words"});
  const cv::Rect first = ink_rect(r.masks.front()), last = ink_rect(r.masks.back());
  const int y = first.y + first.height + 3;
  cv::line(r.img, {first.x, y}, {last.x + last.width - 1, y}, {0, 0, 0}, 3);
  const auto words = locate_words(r.img, r.line, r.ctc);
  REQUIRE(words.size() == 3);
  for (size_t k = 0; k + 1 < words.size(); ++k) {
    // Each word ends before the next one's ink starts.
    CHECK(turbo_ocr::aabb(words[k].box)[2] < ink_rect(r.masks[k + 1]).x);
    CHECK(turbo_ocr::aabb(words[k].box)[0] <= ink_rect(r.masks[k]).x);
  }
}

TEST_CASE("locate_words: a rotated line gives rotated word boxes around each word",
          "[words]") {
  const auto r = render_line({"tilted", "text", "line"});
  const cv::Point2f centre(450, 80);
  const cv::Mat rot = cv::getRotationMatrix2D(centre, 6.0, 1.0);  // 6 deg CCW
  cv::Mat img;
  cv::warpAffine(r.img, img, rot, r.img.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  Box line{};
  for (int k = 0; k < 4; ++k) {
    const double x = r.line[k][0], y = r.line[k][1];
    const double u = rot.at<double>(0, 0) * x + rot.at<double>(0, 1) * y + rot.at<double>(0, 2);
    const double v = rot.at<double>(1, 0) * x + rot.at<double>(1, 1) * y + rot.at<double>(1, 2);
    line[k] = {static_cast<int>(std::lround(u)), static_cast<int>(std::lround(v))};
  }
  const auto words = locate_words(img, line, r.ctc);
  REQUIRE(words.size() == 3);
  for (size_t k = 0; k < words.size(); ++k) {
    cv::Mat mask;
    cv::warpAffine(r.masks[k], mask, rot, r.img.size());
    std::vector<cv::Point> quad;
    for (const auto &p : words[k].box.pts) quad.emplace_back(p[0], p[1]);
    std::vector<cv::Point> ink;
    cv::findNonZero(mask > 128, ink);
    REQUIRE_FALSE(ink.empty());
    int outside = 0;
    for (const auto &p : ink)
      if (cv::pointPolygonTest(quad, cv::Point2f(p), true) < -1.5) ++outside;
    INFO("word " << words[k].text);
    CHECK(outside == 0);
    // Tight along the text: the quad's long side is at most the ink's own
    // extent (its min-area rect) plus a few pixels.
    const auto rect = cv::minAreaRect(ink);
    const float ink_len = std::max(rect.size.width, rect.size.height);
    const float top_len = static_cast<float>(cv::norm(quad[1] - quad[0]));
    CHECK(top_len <= ink_len + 6.0f);
  }
}

TEST_CASE("locate_words: vertical text keeps the line box's point order", "[words]") {
  auto r = render_line({"vertical", "words"});
  // Text reading top to bottom, the orientation compute_crop_transform reads
  // tall boxes in.
  cv::Mat img;
  cv::rotate(r.img, img, cv::ROTATE_90_CLOCKWISE);
  const int H = r.img.rows;
  auto to_rot = [H](int x, int y) { return std::array<int, 2>{H - 1 - y, x}; };
  const auto [lx0, ly0, lx1, ly1] = turbo_ocr::aabb(r.line);
  const auto a = to_rot(lx0, ly0), b = to_rot(lx1, ly1);
  Box line{};
  line[0] = {std::min(a[0], b[0]), std::min(a[1], b[1])};
  line[1] = {std::max(a[0], b[0]), std::min(a[1], b[1])};
  line[2] = {std::max(a[0], b[0]), std::max(a[1], b[1])};
  line[3] = {std::min(a[0], b[0]), std::max(a[1], b[1])};
  const auto words = locate_words(img, line, r.ctc);
  REQUIRE(words.size() == 2);
  for (size_t k = 0; k < words.size(); ++k) {
    cv::Mat mask;
    cv::rotate(r.masks[k], mask, cv::ROTATE_90_CLOCKWISE);
    check_tight(words[k], mask);
    // [tl, tr, br, bl] like the line box.
    const auto &q = words[k].box;
    CHECK(q[0][0] < q[1][0]);
    CHECK(q[0][1] < q[3][1]);
    CHECK(q[2][0] == q[1][0]);
    CHECK(q[2][1] == q[3][1]);
  }
  // Reading order top to bottom: the first word sits higher on the page.
  CHECK(words[0].box[3][1] < words[1].box[3][1]);
}
