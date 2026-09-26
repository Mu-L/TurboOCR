#pragma once

#include <string>
#include <utility>
#include <vector>

namespace turbo_ocr::recognition {

// GPU path: indices and scores already computed by GPU argmax kernel.
[[nodiscard]] std::pair<std::string, float>
ctc_greedy_decode(const int *indices, const float *scores, int seq_len,
                  const std::vector<std::string> &label_list);

// CPU path: raw logits, needs inline argmax.
[[nodiscard]] std::pair<std::string, float>
ctc_greedy_decode_raw(const float *logits, int seq_len, int num_classes,
                      const std::vector<std::string> &label_list);

// Per-timestep argmax of a raw [seq_len, num_classes] logit block (the same
// argmax ctc_greedy_decode_raw runs inline), for the CPU path when it needs
// the indices themselves -- ctc_greedy_decode on them then gives exactly what
// ctc_greedy_decode_raw returns.
void ctc_argmax(const float *logits, int seq_len, int num_classes,
                int *indices, float *scores);

// One word of a decoded line.
struct CtcWord {
  std::string text;
  float score = 0.0f;  // mean probability of the word's characters
  // Centres of the timesteps the word's first and last characters were
  // emitted at, as fractions of the line crop's content width (0 = where the
  // text line starts, 1 = where it ends).
  float first = 0.0f;
  float last = 0.0f;
};

// The greedy CTC collapse of ctc_greedy_decode, split into words the way
// common/word_split.h defines them: at spaces, and around every character of
// a script written without spaces (CJK), which is a word of its own.
// `input_w` is the width the recognizer ran at, `content_w` the part of it the
// text occupies (the rest is right-padding); timestep t sits at
// (t + 0.5) * input_w / seq_len input pixels. For space-separated scripts,
// joining the words with single spaces gives the line text with runs of
// spaces collapsed.
[[nodiscard]] std::vector<CtcWord>
ctc_greedy_decode_words(const int *indices, const float *scores, int seq_len,
                        int input_w, int content_w,
                        const std::vector<std::string> &label_list);

// Shared dictionary loader.
// Prepends "blank" and appends " " around the file contents.
// label_list should be empty on entry (or pre-populated with "blank").
[[nodiscard]] bool load_label_dict(const std::string &dict_path,
                                    std::vector<std::string> &label_list);

} // namespace turbo_ocr::recognition
