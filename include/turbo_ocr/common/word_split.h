#pragma once

// How text is cut into words wherever words are produced (OCR word boxes,
// PDF text-layer words), so both sources agree on what a word is.
namespace turbo_ocr {

// Word separators: ASCII space and tab, no-break space, ideographic space.
[[nodiscard]] constexpr bool is_word_space(char32_t cp) noexcept {
  return cp == U' ' || cp == U'\t' || cp == 0xA0 || cp == 0x3000;
}

// Characters of scripts written without spaces between words -- CJK
// ideographs, kana, CJK punctuation, full/halfwidth forms. Each one is a word
// of its own.
[[nodiscard]] constexpr bool is_standalone_word_char(char32_t cp) noexcept {
  return (cp >= 0x3001 && cp <= 0x30FF) ||   // CJK punctuation, kana
         (cp >= 0x31F0 && cp <= 0x31FF) ||   // katakana phonetic extensions
         (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK extension A
         (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK unified ideographs
         (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK compatibility ideographs
         (cp >= 0xFF00 && cp <= 0xFFEF) ||   // full/halfwidth forms
         (cp >= 0x20000 && cp <= 0x2FA1F);   // CJK extensions B.. + supplement
}

} // namespace turbo_ocr
