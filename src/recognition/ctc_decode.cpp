#include "turbo_ocr/recognition/ctc_decode.h"

#include "turbo_ocr/common/word_split.h"

#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define TURBO_CTC_HAVE_AVX2 1
#else
#define TURBO_CTC_HAVE_AVX2 0
#endif

namespace turbo_ocr::recognition {

namespace {
#if TURBO_CTC_HAVE_AVX2
// AVX2 argmax over a contiguous row of floats. Matches the scalar reference
// exactly, INCLUDING tie-breaking (lowest index wins on equal values): the
// per-lane compare uses strict >, so a lane keeps its lowest index on ties,
// and the 8-lane horizontal reduce also breaks ties toward the lower index.
inline std::pair<float, int> argmax_avx2(const float *row, int n) {
  int j = 0;
  __m256 vmax = _mm256_set1_ps(-3.402823e38f);
  __m256i vidx = _mm256_setzero_si256();
  __m256i vj = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i v8 = _mm256_set1_epi32(8);
  for (; j + 8 <= n; j += 8) {
    __m256 v = _mm256_loadu_ps(row + j);
    __m256 gt = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
    vmax = _mm256_max_ps(vmax, v);
    vidx = _mm256_blendv_epi8(vidx, vj, _mm256_castps_si256(gt));
    vj = _mm256_add_epi32(vj, v8);
  }
  alignas(32) float vals[8];
  alignas(32) int idxs[8];
  _mm256_store_ps(vals, vmax);
  _mm256_store_si256(reinterpret_cast<__m256i *>(idxs), vidx);
  float best = vals[0];
  int bi = idxs[0];
  for (int k = 1; k < 8; k++) {
    if (vals[k] > best || (vals[k] == best && idxs[k] < bi)) {
      best = vals[k];
      bi = idxs[k];
    }
  }
  for (; j < n; j++) {
    if (row[j] > best) {
      best = row[j];
      bi = j;
    }
  }
  return {best, bi};
}

inline bool simd_ctc_enabled() {
  static const bool e = [] {
    const char *v = std::getenv("SIMD_CTC");
    return v && v[0] == '1';
  }();
  return e;
}
#endif // TURBO_CTC_HAVE_AVX2

bool use_simd_ctc() {
#if TURBO_CTC_HAVE_AVX2
  return simd_ctc_enabled();
#else
  return false;
#endif
}

// (max, argmax) of one logit row, lowest index on ties -- the one argmax both
// ctc_greedy_decode_raw and ctc_argmax use, so they can never disagree.
inline std::pair<float, int> row_argmax(const float *row, int n, bool use_simd) {
#if TURBO_CTC_HAVE_AVX2
  if (use_simd) return argmax_avx2(row, n);
#else
  (void)use_simd;
#endif
  int index = 0;
  float max_val = row[0];
  for (int j = 1; j < n; j++) {
    if (row[j] > max_val) {
      max_val = row[j];
      index = j;
    }
  }
  return {max_val, index};
}

// First code point of a UTF-8 label (every dictionary label is one character).
char32_t first_codepoint(const std::string &s) {
  if (s.empty()) return 0;
  const auto c = static_cast<unsigned char>(s[0]);
  if (c < 0x80) return c;
  const int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
  char32_t cp = c & (0x3Fu >> extra);
  for (int k = 1; k <= extra && k < static_cast<int>(s.size()); ++k)
    cp = (cp << 6) | (static_cast<unsigned char>(s[static_cast<size_t>(k)]) & 0x3Fu);
  return cp;
}
} // namespace

std::pair<std::string, float>
ctc_greedy_decode(const int *indices, const float *scores, int seq_len,
                  const std::vector<std::string> &label_list) {
  std::string text;
  text.reserve(seq_len);
  float score = 0.0f;
  int count = 0;
  int last_index = -1;

  for (int i = 0; i < seq_len; i++) {
    int index = indices[i];
    if (index != last_index) {
      if (index != 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
        score += scores[i];
        count++;
      }
    }
    last_index = index;
  }
  if (count > 0)
    score /= count;
  return {text, score};
}

std::pair<std::string, float>
ctc_greedy_decode_raw(const float *logits, int seq_len, int num_classes,
                      const std::vector<std::string> &label_list) {
  std::string text;
  text.reserve(seq_len);
  float score = 0.0f;
  int count = 0;
  int last_index = -1;

  const bool use_simd = use_simd_ctc();
  for (int i = 0; i < seq_len; i++) {
    const auto [max_val, index] =
        row_argmax(logits + i * num_classes, num_classes, use_simd);

    if (index != last_index) {
      if (index != 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
        score += max_val;
        count++;
      }
    }
    last_index = index;
  }
  if (count > 0)
    score /= count;
  return {text, score};
}

void ctc_argmax(const float *logits, int seq_len, int num_classes,
                int *indices, float *scores) {
  const bool use_simd = use_simd_ctc();
  for (int i = 0; i < seq_len; i++) {
    const auto [max_val, index] =
        row_argmax(logits + i * num_classes, num_classes, use_simd);
    indices[i] = index;
    scores[i] = max_val;
  }
}

std::vector<CtcWord>
ctc_greedy_decode_words(const int *indices, const float *scores, int seq_len,
                        int input_w, int content_w,
                        const std::vector<std::string> &label_list) {
  std::vector<CtcWord> words;
  if (seq_len <= 0 || content_w <= 0) return words;
  const int n_labels = static_cast<int>(label_list.size());
  // Timestep centre -> fraction of the content width.
  const float step = static_cast<float>(input_w) /
                     (static_cast<float>(seq_len) * static_cast<float>(content_w));

  CtcWord cur;
  int count = 0;
  auto flush = [&] {
    if (count > 0) {
      cur.score /= static_cast<float>(count);
      words.push_back(std::move(cur));
    }
    cur = CtcWord{};
    count = 0;
  };

  int last_index = -1;
  for (int t = 0; t < seq_len; t++) {
    const int index = indices[t];
    if (index != last_index && index != 0 && index < n_labels) {
      const std::string &ch = label_list[static_cast<size_t>(index)];
      const char32_t cp = first_codepoint(ch);
      if (turbo_ocr::is_word_space(cp)) {
        flush();
      } else {
        const bool own_word = turbo_ocr::is_standalone_word_char(cp);
        if (own_word) flush();
        const float x = (static_cast<float>(t) + 0.5f) * step;
        if (count == 0) cur.first = x;
        cur.last = x;
        cur.text += ch;
        cur.score += scores[t];
        ++count;
        if (own_word) flush();
      }
    }
    last_index = index;
  }
  flush();
  return words;
}

bool load_label_dict(const std::string &dict_path,
                     std::vector<std::string> &label_list) {
  std::ifstream in(dict_path);
  if (!in) [[unlikely]] {
    std::cerr << std::format("[Rec] Failed to open dictionary: {}",
                             dict_path)
              << '\n';
    return false;
  }
  std::string line;
  while (getline(in, line)) {
    if (!line.empty() && line.back() == '\n')
      line.pop_back();
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    label_list.push_back(std::move(line));
  }
  label_list.push_back(" ");
  return true;
}

} // namespace turbo_ocr::recognition
