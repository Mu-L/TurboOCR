#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/recognition/ctc_decode.h"

namespace turbo_ocr::recognition {

// Places the recognized words of one text line in the image.
//
// The recognizer only says roughly where a word is -- the CTC timestep a
// character is emitted at lies somewhere inside its glyph -- so the line's own
// ink decides the rest: the line is re-sampled at its native resolution, each
// boundary between two words goes to the widest ink-free gap between the last
// character of one and the first character of the next, and every box is the
// bounding box of its word's glyphs, whole even where the line box clips them
// (a descender, a comma's tail). Punctuation touching a word has no gap to cut
// at and stays with it. Backend-independent: the recognizers contribute only the
// CTC positions, the pixels come from the host image (BGR, BGRA or gray).
// `line` must be the box the recognizer read (after any angle flip).
// `page_lines` are the page's other detected lines (the list may include
// `line` itself): ink one of them owns never joins this line's words, even
// where glyphs of two lines touch.
[[nodiscard]] std::vector<OCRWord>
locate_words(const cv::Mat &img, const Box &line,
             const std::vector<CtcWord> &words,
             const std::vector<Box> &page_lines = {});

} // namespace turbo_ocr::recognition
