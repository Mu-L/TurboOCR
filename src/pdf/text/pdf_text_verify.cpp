// Text-layer trust checks: the sanity heuristics for a native-text box and
// the auto_verified per-item replacement pass. Pure consumers of the public
// PdfDocument API — no direct PDFium calls, so no pdfium_lock here.

#include "turbo_ocr/pdf/pdf_text_layer.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/word_split.h"

namespace turbo_ocr::pdf {

SanityVerdict passes_sanity_check(const std::string &text,
                                  float box_width_pt,
                                  float box_height_pt) {
  if (text.empty())
    return {false, "no native text in box"};

  int n = 0, fffd = 0, nonprint = 0;
  for (size_t i = 0; i < text.size(); ) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    uint32_t cp = 0;
    int step = 1;
    if (c < 0x80) { cp = c; step = 1; }
    else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp = (c & 0x1F) << 6 | (static_cast<unsigned char>(text[i+1]) & 0x3F);
      step = 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = (c & 0x0F) << 12
         | (static_cast<unsigned char>(text[i+1]) & 0x3F) << 6
         | (static_cast<unsigned char>(text[i+2]) & 0x3F);
      step = 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
      cp = (c & 0x07) << 18
         | (static_cast<unsigned char>(text[i+1]) & 0x3F) << 12
         | (static_cast<unsigned char>(text[i+2]) & 0x3F) << 6
         | (static_cast<unsigned char>(text[i+3]) & 0x3F);
      step = 4;
    }
    if (cp == 0xFFFD) ++fffd;
    else if (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r') ++nonprint;
    ++n;
    i += step;
  }
  if (n == 0) return {false, "empty after decode"};
  if (fffd * 20 > n) return {false, "too many U+FFFD replacement chars"};
  if (nonprint * 10 > n) return {false, "too many non-printable chars"};

  if (box_width_pt > 0 && box_height_pt > 0) {
    float min_expected = box_width_pt / 30.0f;
    float max_expected = box_width_pt / 2.0f;
    if (static_cast<float>(n) < min_expected * 0.5f ||
        static_cast<float>(n) > max_expected * 2.0f)
      return {false, "char count implausible for box width"};
  }

  return {true, "trusted"};
}

namespace {

// Next code point of UTF-8 text at byte `i`, advancing `i` past it.
char32_t next_code_point(const std::string &s, size_t &i) {
  const auto c = static_cast<unsigned char>(s[i++]);
  if (c < 0x80) return c;
  const int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
  char32_t cp = c & (0x3Fu >> extra);
  for (int k = 0; k < extra && i < s.size(); ++k)
    cp = (cp << 6) | (static_cast<unsigned char>(s[i++]) & 0x3Fu);
  return cp;
}

// Page characters inside a box (text order): centred in it, or -- for a
// second try -- merely overlapping it, as FPDFText_GetBoundedText counts them.
std::vector<const PdfTextChar *> chars_in(const std::vector<PdfTextChar> &chars,
                                          float x0, float y0, float x1,
                                          float y1, bool centred) {
  std::vector<const PdfTextChar *> out;
  for (const auto &c : chars) {
    const bool in =
        centred ? (0.5f * (c.x0_pt + c.x1_pt) >= x0 && 0.5f * (c.x0_pt + c.x1_pt) <= x1 &&
                   0.5f * (c.y0_pt + c.y1_pt) >= y0 && 0.5f * (c.y0_pt + c.y1_pt) <= y1)
                : (c.x1_pt > x0 && c.x0_pt < x1 && c.y1_pt > y0 && c.y0_pt < y1);
    if (in) out.push_back(&c);
  }
  return out;
}

// The words of `text` (common/word_split.h rules), each boxed from the page
// characters it was read from: `chars` are matched in turn against the
// words' characters, skipping any the text leaves out (inserted spaces, wrap
// marks). Empty when a character finds no match.
std::vector<OCRWord> box_words(const std::string &text,
                               const std::vector<const PdfTextChar *> &chars,
                               float px_to_pt) {
  constexpr size_t kLookahead = 16;
  const auto px = [px_to_pt](float v) {
    return static_cast<int>(std::lround(v / px_to_pt));
  };
  std::vector<OCRWord> out;
  OCRWord cur;
  float bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
  bool open = false;
  const auto flush = [&] {
    if (open) {
      cur.confidence = 1.0f;
      cur.box[0] = {px(bx0), px(by0)};
      cur.box[1] = {px(bx1), px(by0)};
      cur.box[2] = {px(bx1), px(by1)};
      cur.box[3] = {px(bx0), px(by1)};
      out.push_back(std::move(cur));
    }
    cur = OCRWord{};
    open = false;
  };
  size_t next = 0;
  for (size_t i = 0; i < text.size();) {
    const size_t start = i;
    const char32_t cp = next_code_point(text, i);
    if (cp < 0x20 || is_word_space(cp)) {
      flush();
      continue;
    }
    const bool standalone = is_standalone_word_char(cp);
    if (standalone) flush();
    const size_t stop = std::min(chars.size(), next + kLookahead);
    size_t j = next;
    while (j < stop && chars[j]->cp != cp) ++j;
    if (j == stop) return {};
    const PdfTextChar &c = *chars[j];
    next = j + 1;
    if (!open) {
      bx0 = c.x0_pt; by0 = c.y0_pt; bx1 = c.x1_pt; by1 = c.y1_pt;
      open = true;
    } else {
      bx0 = std::min(bx0, c.x0_pt); by0 = std::min(by0, c.y0_pt);
      bx1 = std::max(bx1, c.x1_pt); by1 = std::max(by1, c.y1_pt);
    }
    cur.text.append(text, start, i - start);
    if (standalone) flush();
  }
  flush();
  return out;
}

}  // namespace

void verify_results_with_text_layer(std::vector<OCRResultItem> &results,
                                    const PdfDocument &doc, int page_index,
                                    int dpi, const PdfPageText *page_text) {
  // dpi is caller-supplied; a zero/negative value would make px_to_pt inf/nan
  // and silently blank the text-layer coords for every item (mis-verifying the
  // whole page). Callers validate DPI, but this guard keeps the invariant local.
  if (dpi <= 0) return;
  const float px_to_pt = 72.0f / static_cast<float>(dpi);
  for (auto &item : results) {
    auto [ix0, iy0, ix1, iy1] = turbo_ocr::aabb(item.box);
    float x0 = ix0 * px_to_pt, y0 = iy0 * px_to_pt;
    float x1 = ix1 * px_to_pt, y1 = iy1 * px_to_pt;
    std::string native = doc.text_in_rect_pt(page_index, x0, y0, x1, y1);
    // Control marks PDFium reports for wrap hyphens and unmapped glyphs are
    // no text (the text-layer lines drop them the same way).
    std::erase_if(native, [](char ch) {
      const auto c = static_cast<unsigned char>(ch);
      return c < 0x20 && c != '\t' && c != '\n' && c != '\r';
    });
    auto verdict = passes_sanity_check(native, x1 - x0, y1 - y0);
    if (verdict.accept) {
      item.text = std::move(native);
      item.source = "pdf";
      item.confidence = 1.0f;
      // The OCR words spelled the OCR text: box the new text's own words from
      // the characters it came from.
      if (!item.words.empty() && page_text) {
        auto words = box_words(item.text,
                               chars_in(page_text->chars, x0, y0, x1, y1, true),
                               px_to_pt);
        if (words.empty())
          words = box_words(item.text,
                            chars_in(page_text->chars, x0, y0, x1, y1, false),
                            px_to_pt);
        // No match at all: the OCR words stay -- they still mark the text.
        if (!words.empty()) item.words = std::move(words);
      }
    }
  }
}

} // namespace turbo_ocr::pdf
