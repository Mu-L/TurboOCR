#pragma once

// Device side of the GPU word placer (gpu_word_placer.cpp): the pixel stages
// of recognition::locate_words for every line of a page at once. Each stage
// reproduces the CPU one to the bit -- OpenCV 4's warpPerspective (INTER_LINEAR,
// BORDER_REPLICATE) and BGR2GRAY, Otsu's threshold, the neighbour claims -- and
// the connected components come out numbered by their first pixel in raster
// order, the canonical order word_boxes_internal.h defines.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace turbo_ocr::recognition::gpu {

// One line's crop, as recognition::detail::plan_line lays it out.
struct LineDesc {
  double M[9];        // crop pixel -> page pixel (the float matrix, widened)
  int W, H, w, h, m;  // crop W x H, line box w x h at (m, m)
  int off;            // first pixel of the crop in the chunk
  int tile_first;     // its first tile
  int sub_x, sub_y;   // origin when the crop is an exact sub-image, else sub_x < 0
  int bw0;            // column block of OpenCV's warp for this crop
  float own_mid;
  int nb_first, nb_count;  // neighbours [nb_first, nb_first + nb_count)
};

// Another line's box reaching into the crop (recognition::detail::Neighbour).
struct NbDesc {
  int claims;
  int wx0, wy0, wx1, wy1;  // claim window
  float lx, ly, slope;     // its midline
  int run_first, run_y0, run_rows;  // its rows [run_y0, run_y0 + run_rows) in the runs table
};

// A run of up to 1024 pixels of one line: [start, start + count).
struct Tile {
  int line, start, count;
};
inline constexpr int kTilePixels = 1024;
inline constexpr int kTileWords = kTilePixels / 32;  // a bit a pixel

// Per component, accumulated with atomics (index = component number in the
// chunk, lines' components consecutive in canonical order).
struct CompStats {
  int area, x0, y0, x1, y1;  // x1/y1 inclusive
  int in_core, core_x0, core_x1;
  int claimed, deep, band, boxed;
  int line;
  int unused;  // no padding bytes: the host copies whole records
  unsigned long long sumx;
};

// A glyph the bleed split examines: the claiming line it reaches into, its
// row-profile bins, and where (if anywhere) it is cut.
struct CandPar {
  int comp, line, nb, nb_all, do_cut;
  float lx, ly, slope, cut;
};

// Per line, after the pass.
struct LineOut {
  int comp_first;  // its first component
  int core_ink;    // ink pixels inside the line box (polarity check)
  int any_claim;   // some pixel claimed by another line
  int thr;         // Otsu threshold
  int light_px, reaching;  // polarity: light core pixels, and those reaching the border
  int cpu;         // hand the line to the CPU code (a profile too fine to bin here)
};

// Totals the host reads before copying the results back (zeroed per chunk).
struct Totals {
  int comps;          // components in the chunk
  int hist;           // column-histogram cells used
  int comps_overflow; // more components than the buffers hold
  int hist_overflow;  // more histogram cells than the buffer holds
  int cands;          // glyphs the bleed split examines
  int any_split;      // some glyph was cut
  int dark_tiles;     // tiles of dark lines, for the polarity pass
};

struct PageView {
  const std::uint8_t *data;  // BGR, pitched
  std::size_t step;
  int rows, cols;
};

// Device buffers of one chunk; sizes are capacities.
struct ChunkBuffers {
  const LineDesc *lines;
  const NbDesc *nbs;
  const std::int16_t *runs;  // 4 per row
  const Tile *tiles;
  int n_lines, n_tiles, n_pixels;
  std::uint8_t *ink;       // gray, then ink (n_pixels)
  std::uint16_t *claims;   // tag | deep << 8 | boxed << 9 (n_pixels)
  int *labels;             // union-find (n_pixels)
  int *cid;                // component number at roots (n_pixels)
  int *tile_roots, *tile_base;  // components rooted in each tile, and before it (n_tiles)
  int *dark_tiles;         // the dark lines' tiles (n_tiles)
  unsigned *root_mask;     // each tile's roots (n_tiles * kTileWords)
  int *hist256;            // per-line core histogram (n_lines * 256)
  LineOut *line_out;       // n_lines
  CompStats *comps;        // comp_capacity
  int *colw, *hoff;        // comp_capacity each
  int *colhist;            // hist_capacity
  Totals *totals;
  void *scan_tmp;          // CUB scratch
  std::size_t scan_tmp_bytes;
  int comp_capacity, hist_capacity;
  CandPar *cands;          // cand_capacity
  int *bytag, *prof;       // 256 per candidate each
  int cand_capacity;
};
inline constexpr int kProfileBins = 256;

// CUB scratch needed to scan `n` ints.
std::size_t scan_scratch_bytes(int n);

// The stages of one chunk, enqueued on `stream`; the host sequences them,
// reading `totals` in between (none reads anything back itself):
//  pixels  crop, threshold, polarity, claims, components (labels, numbering)
//  stats   per-component statistics, and the bleed split's candidates
//  split   the bleed split: profile, valley, cut (when candidates exist)
//  relabel components again after a cut (then stats again)
//  hist    per-component column histograms
void stage_pixels(const PageView &page, const ChunkBuffers &b, cudaStream_t stream);
void stage_stats(const ChunkBuffers &b, cudaStream_t stream);
void stage_split(const ChunkBuffers &b, cudaStream_t stream);
void stage_relabel(const ChunkBuffers &b, cudaStream_t stream);
void stage_hist(const ChunkBuffers &b, cudaStream_t stream);

// Only the crop and gray conversion, into b.ink (for the unit tests).
void crop_gray(const PageView &page, const ChunkBuffers &b, cudaStream_t stream);

}  // namespace turbo_ocr::recognition::gpu
