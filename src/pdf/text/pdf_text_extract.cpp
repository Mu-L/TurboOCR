#include "pdf_text_internal.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/word_split.h"

namespace turbo_ocr::pdf {

using detail::pdfium_lock;
using detail::utf16le_to_utf8;

namespace {

// A character's box in PDFium page space (bottom-left origin). The tight glyph
// box first. The invisible text layer OCR tools write uses a glyphless font
// whose empty glyphs give a flat tight box -- then the loose box from the
// font's metrics, and for a font without metrics (FontBBox [0 0 0 0]) a box
// built from the baseline origin and the font size. False for characters
// PDFium generated without any geometry (inserted spaces).
bool char_box(FPDF_TEXTPAGE tp, int idx, double &l, double &r, double &b,
              double &t) {
  const bool tight = FPDFText_GetCharBox(tp, idx, &l, &r, &b, &t) != 0;
  const double size = FPDFText_GetFontSize(tp, idx);
  const double min_h = size > 0 ? 0.25 * size : 0.0;
  if (tight && r > l && t - b > min_h && t > b) return true;
  FS_RECTF loose{};
  if (FPDFText_GetLooseCharBox(tp, idx, &loose) &&
      loose.right > loose.left && loose.top - loose.bottom > min_h &&
      loose.top > loose.bottom) {
    if (!tight || r <= l) { l = loose.left; r = loose.right; }
    b = loose.bottom;
    t = loose.top;
    return true;
  }
  double ox = 0, oy = 0;
  if (tight && r > l && size > 0 && FPDFText_GetCharOrigin(tp, idx, &ox, &oy)) {
    b = oy - 0.2 * size;
    t = oy + 0.8 * size;
    return true;
  }
  return tight && r > l && t > b;
}

}  // namespace

PdfPageText PdfDocument::extract_page(int page_index) const {
  PdfPageText out;
  if (!doc_ || !impl_) return out;

  // Per-document cache mutex first (cheap, protects the unordered_map),
  // then the global PDFium lock for the actual API calls.
  std::lock_guard<std::mutex> lock(impl_->mtx);
  std::lock_guard<std::mutex> gl(pdfium_lock());
  auto *ph = impl_->get_locked(static_cast<FPDF_DOCUMENT>(doc_), page_index);
  if (!ph) return out;

  // Report visual dimensions so downstream code (PdfRenderer, layout,
  // client coord conversions) sees a single post-rotation coordinate
  // system regardless of /Rotate.
  out.page_width_pt  = ph->visual_w_pt;
  out.page_height_pt = ph->visual_h_pt;
  out.rotation_deg   = ph->rotation_deg;

  const int n_chars = FPDFText_CountChars(ph->textpage);
  out.char_count = std::max(0, n_chars);
  if (n_chars <= 0) return out;

  // Pull the full page text once and scan it for U+FFFD / non-printable
  // counts. This is cheap — PDFium returns UTF-16 that we transcode below.
  {
    std::vector<unsigned short> buf(static_cast<size_t>(n_chars) + 1);
    int written = FPDFText_GetText(ph->textpage, 0, n_chars, buf.data());
    for (int i = 0; i < written; ++i) {
      uint32_t cp = buf[static_cast<size_t>(i)];
      if (cp == 0) continue;
      if (cp == 0xFFFD) ++out.fffd_count;
      else if (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r')
        ++out.nonprint_count;
    }
  }

  // Line grouping: walk chars and split on PDFium's OWN layout line breaks
  // (the generated \r\n in the char stream). NOT FPDFText_CountRects/GetRect:
  // those rects are per same-font/style RUN, so a mid-line font-size change
  // (small-caps headings: "M"+"AMBA") fragments one visual line into several
  // out-of-order boxes. PDFium's char-flow segmentation handles those
  // correctly; the per-char box union gives the true line bbox.
  const int n_units = FPDFText_CountChars(ph->textpage);
  std::vector<unsigned short> line_units;
  line_units.reserve(128);
  double lleft = 0, ltop = 0, lright = 0, lbottom = 0;
  bool have_box = false;
  // The word being built, and the finished words of the current line.
  std::vector<unsigned short> word_units;
  double wleft = 0, wtop = 0, wright = 0, wbottom = 0;
  bool word_box = false;
  std::vector<PdfTextWord> line_words;
  // The box of the word's last character, to see a hyphenated word jump to
  // the next line.
  double pleft = 0, pbottom = 0;

  // PDFium page-space box (bottom-left origin) -> visual top-left-origin AABB.
  auto to_visual = [&](double left, double top, double right, double bottom,
                       float &x0, float &y0, float &x1, float &y1) {
    // Common path: no rotation, no cropbox offset. Single subtract + flip,
    // no 4-corner transform. This is the shape of ~99% of real PDFs.
    if (ph->rotation_deg == 0 && ph->origin_x_pt == 0.0f &&
        ph->origin_y_pt == 0.0f) [[likely]] {
      const float page_h = ph->visual_h_pt;
      x0 = static_cast<float>(left);
      x1 = static_cast<float>(right);
      y0 = page_h - static_cast<float>(top);
      y1 = page_h - static_cast<float>(bottom);
      return;
    }
    // Rotated / trimmed page: transform all 4 corners through
    // pre_to_visual and take the AABB.
    const float pre_x[4] = {
        static_cast<float>(left), static_cast<float>(right),
        static_cast<float>(right), static_cast<float>(left)};
    const float pre_y[4] = {
        static_cast<float>(top), static_cast<float>(top),
        static_cast<float>(bottom), static_cast<float>(bottom)};
    float vx, vy;
    pre_to_visual(*ph, pre_x[0], pre_y[0], vx, vy);
    float vx0 = vx, vx1 = vx, vy0 = vy, vy1 = vy;
    for (int k = 1; k < 4; ++k) {
      pre_to_visual(*ph, pre_x[k], pre_y[k], vx, vy);
      if (vx < vx0) vx0 = vx; else if (vx > vx1) vx1 = vx;
      if (vy < vy0) vy0 = vy; else if (vy > vy1) vy1 = vy;
    }
    x0 = vx0;
    y0 = vy0;
    x1 = vx1;
    y1 = vy1;
  };

  auto flush_word = [&]() {
    if (!word_units.empty() && word_box) {
      PdfTextWord w;
      w.text = utf16le_to_utf8(word_units.data(),
                               static_cast<int>(word_units.size()));
      if (!w.text.empty()) {
        to_visual(wleft, wtop, wright, wbottom, w.x0_pt, w.y0_pt, w.x1_pt,
                  w.y1_pt);
        line_words.push_back(std::move(w));
      }
    }
    word_units.clear();
    word_box = false;
  };

  auto flush_line = [&]() {
    flush_word();
    if (line_units.empty() || !have_box) {
      line_units.clear();
      line_words.clear();
      have_box = false;
      return;
    }
    std::string utf8 =
        utf16le_to_utf8(line_units.data(), static_cast<int>(line_units.size()));
    detail::strip_trailing_ws(utf8);
    line_units.clear();
    have_box = false;
    if (utf8.empty()) {
      line_words.clear();
      return;
    }

    PdfTextLine line;
    line.text = std::move(utf8);
    to_visual(lleft, ltop, lright, lbottom, line.x0_pt, line.y0_pt, line.x1_pt,
              line.y1_pt);
    line.words = std::move(line_words);
    line_words.clear();
    out.lines.push_back(std::move(line));
  };

  for (int idx = 0; idx < n_units; ++idx) {
    const unsigned int u = FPDFText_GetUnicode(ph->textpage, idx);
    if (u == '\r' || u == '\n') {
      flush_line();
      continue;
    }
    // Skip control characters (U+0000–U+001F except the \r/\n handled above and
    // \t). Glyphs with no proper ToUnicode mapping — soft/discretionary hyphens
    // at line-wrap points, ligature components — report a low control code
    // (U+0000, or e.g. U+0002 for a wrap hyphen: "de\x02ployment"). Dropping
    // them rejoins the word ("deployment") and avoids embedding a NUL that would
    // truncate the line, or a control glyph that renders as tofu (□).
    if (u < 0x20 && u != '\t') continue;
    // FPDFText_GetUnicode returns a full code point (uint32). Re-encode astral
    // (>U+FFFF) code points as a UTF-16 surrogate pair so utf16le_to_utf8
    // reconstructs them; a bare cast to unsigned short would truncate to 16
    // bits and corrupt emoji / SMP CJK / math-alphanumeric glyphs.
    const size_t first_unit = line_units.size();
    if (u > 0xFFFF) {
      const unsigned int v = u - 0x10000u;
      line_units.push_back(static_cast<unsigned short>(0xD800u + (v >> 10)));
      line_units.push_back(static_cast<unsigned short>(0xDC00u + (v & 0x3FFu)));
    } else {
      line_units.push_back(static_cast<unsigned short>(u));
    }
    double cl = 0, cr = 0, cb = 0, ct = 0;
    // Generated chars (inserted spaces) have no box — keep their text, skip
    // them in the bbox union.
    const bool boxed = char_box(ph->textpage, idx, cl, cr, cb, ct);
    // Words: a space ends one; a CJK character is one on its own; so does a
    // jump down and back to the next visual line -- a hyphenated word PDFium
    // joins across a line break ("founda-" + "tion") -- so each part keeps a
    // true box. (A superscript sits higher and runs on: no break.)
    const bool space = is_word_space(u);
    const bool standalone = is_standalone_word_char(u);
    const bool new_row = boxed && word_box && ct <= pbottom && cl < pleft;
    if (space || standalone || new_row) flush_word();
    if (!space)
      word_units.insert(word_units.end(),
                        line_units.begin() +
                            static_cast<std::ptrdiff_t>(first_unit),
                        line_units.end());
    if (boxed) {
      PdfTextChar ch;
      ch.cp = static_cast<char32_t>(u);
      to_visual(cl, ct, cr, cb, ch.x0_pt, ch.y0_pt, ch.x1_pt, ch.y1_pt);
      out.chars.push_back(ch);
      if (!have_box) {
        lleft = cl; lright = cr; lbottom = cb; ltop = ct;
        have_box = true;
      } else {
        lleft = std::min(lleft, cl);
        lright = std::max(lright, cr);
        lbottom = std::min(lbottom, cb);
        ltop = std::max(ltop, ct);
      }
      if (!space) {
        if (!word_box) {
          wleft = cl; wright = cr; wbottom = cb; wtop = ct;
          word_box = true;
        } else {
          wleft = std::min(wleft, cl);
          wright = std::max(wright, cr);
          wbottom = std::min(wbottom, cb);
          wtop = std::max(wtop, ct);
        }
        pleft = cl;
        pbottom = cb;
      }
    }
    if (standalone) flush_word();
  }
  flush_line();

  return out;
}

} // namespace turbo_ocr::pdf
