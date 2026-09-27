// Pixel stages of recognition::locate_words on the GPU, for a chunk of lines
// at once. Every stage is the CPU one to the bit; the arithmetic below is
// spelled with round-to-nearest intrinsics (no fused multiply-add) in the
// order OpenCV 4.6 evaluates it (imgwarp.cpp, color_rgb.simd.hpp, thresh.cpp).

#include "word_boxes_gpu.h"

#include <cfloat>
#include <climits>

#include <cub/device/device_scan.cuh>

#include "turbo_ocr/common/cuda/cuda_check.h"

namespace turbo_ocr::recognition::gpu {

namespace {

constexpr int kThreads = 256;

// --- crop ------------------------------------------------------------------

__device__ __forceinline__ int clip_to(int x, int a, int b) {
  return x >= a ? (x < b ? x : b - 1) : a;
}
__device__ __forceinline__ int sat_short(int v) { return max(-32768, min(32767, v)); }
__device__ __forceinline__ int sat_u8(int v) { return max(0, min(255, v)); }

__device__ __forceinline__ std::uint8_t bgr_to_gray(int b, int g, int r) {
  // cvtColor(COLOR_BGR2GRAY), 8 bits: 15-bit fixed point, rounded.
  return static_cast<std::uint8_t>((b * 3735 + g * 19235 + r * 9798 + (1 << 14)) >> 15);
}

__device__ __forceinline__ bool in_core(const LineDesc &L, int x, int y) {
  return x >= L.m && x < L.m + L.w && y >= L.m && y < L.m + L.h;
}

// cv::warpPerspective(src, crop, M, size, INTER_LINEAR | WARP_INVERSE_MAP,
// BORDER_REPLICATE) at crop pixel (x, y), then BGR2GRAY. The source
// coordinate is computed per column block exactly as the warp's line routine
// does (X0, Y0, W0 for the block start xb, then + M * x1 per pixel), rounded
// to 1/32 px; the tap weights are initInterTab2D's fixed-point bilinear table,
// whose zero-offset entry is (32767, 0, 0, 1).
__device__ __forceinline__ void block_base(const LineDesc &L, int xb, int y, double &X0,
                                           double &Y0, double &W0) {
  const double *M = L.M;
  const double dxb = static_cast<double>(xb), dy = static_cast<double>(y);
  X0 = __dadd_rn(__dadd_rn(__dmul_rn(M[0], dxb), __dmul_rn(M[1], dy)), M[2]);
  Y0 = __dadd_rn(__dadd_rn(__dmul_rn(M[3], dxb), __dmul_rn(M[4], dy)), M[5]);
  W0 = __dadd_rn(__dadd_rn(__dmul_rn(M[6], dxb), __dmul_rn(M[7], dy)), M[8]);
}

// The warp's 32 / W, 0 where W is.
__device__ __forceinline__ double tab_scale(double W) {
  return W != 0.0 ? __ddiv_rn(32.0, W) : 0.0;
}

// Pixel x1 of the block with bases X0, Y0 and scale Wd = tab_scale(W0 + M6 * x1).
__device__ std::uint8_t warp_tap(const PageView &pg, const LineDesc &L, double X0, double Y0,
                                 double Wd, int x1) {
  const double dx1 = static_cast<double>(x1);
  const double fX = fmax(static_cast<double>(INT_MIN),
                         fmin(static_cast<double>(INT_MAX),
                              __dmul_rn(__dadd_rn(X0, __dmul_rn(L.M[0], dx1)), Wd)));
  const double fY = fmax(static_cast<double>(INT_MIN),
                         fmin(static_cast<double>(INT_MAX),
                              __dmul_rn(__dadd_rn(Y0, __dmul_rn(L.M[3], dx1)), Wd)));
  const int X = __double2int_rn(fX), Y = __double2int_rn(fY);
  const int sx = sat_short(X >> 5), sy = sat_short(Y >> 5);
  const int fx = X & 31, fy = Y & 31;
  int w0, w1, w2, w3;
  if (fx == 0 && fy == 0) {
    w0 = 32767; w1 = 0; w2 = 0; w3 = 1;
  } else {
    w0 = (32 - fy) * (32 - fx) * 32;
    w1 = (32 - fy) * fx * 32;
    w2 = fy * (32 - fx) * 32;
    w3 = fy * fx * 32;
  }
  const int sx0 = clip_to(sx, 0, pg.cols), sx1 = clip_to(sx + 1, 0, pg.cols);
  const int sy0 = clip_to(sy, 0, pg.rows), sy1 = clip_to(sy + 1, 0, pg.rows);
  const std::uint8_t *r0 = pg.data + static_cast<std::size_t>(sy0) * pg.step;
  const std::uint8_t *r1 = pg.data + static_cast<std::size_t>(sy1) * pg.step;
  int c[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int t = r0[sx0 * 3 + k] * w0 + r0[sx1 * 3 + k] * w1 +
                  r1[sx0 * 3 + k] * w2 + r1[sx1 * 3 + k] * w3;
    c[k] = sat_u8((t + (1 << 14)) >> 15);
  }
  return bgr_to_gray(c[0], c[1], c[2]);
}

__device__ std::uint8_t warp_gray(const PageView &pg, const LineDesc &L, int x, int y) {
  const int xb = (x / L.bw0) * L.bw0, x1 = x - xb;
  double X0, Y0, W0;
  block_base(L, xb, y, X0, Y0, W0);
  const double Wd = tab_scale(__dadd_rn(W0, __dmul_rn(L.M[6], static_cast<double>(x1))));
  return warp_tap(pg, L, X0, Y0, Wd, x1);
}

// Block bases a tile can keep in shared memory (its rows times the blocks of
// a row); a tile of a wider crop computes them per pixel.
constexpr int kBaseCache = 128;

// The crop in gray, and with kHist each line's core histogram (the input of
// Otsu's threshold).
template <bool kHist>
__global__ void k_crop_gray(PageView pg, ChunkBuffers b) {
  __shared__ double base[3][kBaseCache];
  __shared__ int hist[256];
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  const int ty0 = t.start / L.W;
  const int nbx = (L.W + L.bw0 - 1) / L.bw0;
  const int nbase = ((t.start + t.count - 1) / L.W - ty0 + 1) * nbx;
  const bool cached = L.sub_x < 0 && nbase <= kBaseCache;
  // No perspective term along x: every pixel of a block divides by its W0
  // (W0 + M6 * x1 is W0 to the bit), so the scale is the block's.
  const bool affine = L.M[6] == 0.0;
  if (kHist)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  if (cached)
    for (int e = threadIdx.x; e < nbase; e += blockDim.x) {
      double X0, Y0, W0;
      block_base(L, (e % nbx) * L.bw0, ty0 + e / nbx, X0, Y0, W0);
      base[0][e] = X0;
      base[1][e] = Y0;
      base[2][e] = affine ? tab_scale(W0) : W0;
    }
  __syncthreads();
  const int rounds = (t.count + kThreads - 1) / kThreads;
  for (int r = 0; r < rounds; ++r) {
    const int i = r * kThreads + threadIdx.x;
    const int p = t.start + i, x = p % L.W, y = p / L.W;
    int key = -1;  // the pixel's gray level when it counts in the histogram
    if (i < t.count) {
      std::uint8_t g;
      if (L.sub_x >= 0) {
        const std::uint8_t *s = pg.data + static_cast<std::size_t>(L.sub_y + y) * pg.step +
                                static_cast<std::size_t>(L.sub_x + x) * 3;
        g = bgr_to_gray(s[0], s[1], s[2]);
      } else if (cached) {
        const int bx = x / L.bw0, x1 = x - bx * L.bw0;
        const int e = (y - ty0) * nbx + bx;
        const double Wd =
            affine ? base[2][e]
                   : tab_scale(__dadd_rn(base[2][e], __dmul_rn(L.M[6], static_cast<double>(x1))));
        g = warp_tap(pg, L, base[0][e], base[1][e], Wd, x1);
      } else {
        g = warp_gray(pg, L, x, y);
      }
      b.ink[L.off + p] = g;
      if (in_core(L, x, y)) key = g;
    }
    if (kHist && key >= 0) atomicAdd(&hist[key], 1);
  }
  if (kHist) {
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      if (hist[i]) atomicAdd(&b.hist256[t.line * 256 + i], hist[i]);
  }
}

// --- threshold ---------------------------------------------------------------

// cv::threshold(core, _, 0, 1, THRESH_BINARY_INV | THRESH_OTSU):
// getThreshVal_Otsu's double arithmetic, operation for operation, one warp a
// line. Only the running class-1 mean is a chain (each step divides by the
// running weight); lane 0 runs it, and the lanes then score the 256 levels
// and keep the first of the best, as the sequential loop does.
__global__ void k_otsu(ChunkBuffers b) {
  __shared__ double pp[256], ip[256], q1s[256], mu1s[256];
  __shared__ bool ok[256];
  const int l = blockIdx.x, lane = threadIdx.x;
  const LineDesc &L = b.lines[l];
  const int *h = b.hist256 + l * 256;
  const double scale = 1.0 / static_cast<double>(L.w * L.h);
  long long sum = 0;
  for (int i = lane; i < 256; i += 32) {
    const int hi = h[i];
    sum += static_cast<long long>(i) * hi;
    const double p_i = __dmul_rn(static_cast<double>(hi), scale);
    pp[i] = p_i;
    ip[i] = __dmul_rn(static_cast<double>(i), p_i);
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  // OpenCV's running double sum of i * h[i]: integers below 2^53 all the way,
  // so exactly this integer.
  const double mu = __dmul_rn(static_cast<double>(sum), scale);
  __syncwarp();
  if (lane == 0) {
    double mu1 = 0.0, q1 = 0.0;
    for (int i = 0; i < 256; ++i) {
      const double t = __dmul_rn(mu1, q1);
      q1 = __dadd_rn(q1, pp[i]);
      const double q2 = __dsub_rn(1.0, q1);
      const bool skip = fmin(q1, q2) < static_cast<double>(FLT_EPSILON) ||
                        fmax(q1, q2) > __dsub_rn(1.0, static_cast<double>(FLT_EPSILON));
      mu1 = skip ? t : __ddiv_rn(__dadd_rn(t, ip[i]), q1);
      q1s[i] = q1;
      mu1s[i] = mu1;
      ok[i] = !skip;
    }
  }
  __syncwarp();
  double best = 0.0;
  int best_i = 256;  // none yet: the threshold stays 0
  for (int i = lane; i < 256; i += 32) {
    if (!ok[i]) continue;
    const double q1 = q1s[i], mu1 = mu1s[i];
    const double q2 = __dsub_rn(1.0, q1);
    const double mu2 = __ddiv_rn(__dsub_rn(mu, __dmul_rn(q1, mu1)), q2);
    const double d = __dsub_rn(mu1, mu2);
    const double sigma = __dmul_rn(__dmul_rn(__dmul_rn(q1, q2), d), d);
    if (sigma > best) {
      best = sigma;
      best_i = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const double ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
    if (ob > best || (ob == best && oi < best_i)) {
      best = ob;
      best_i = oi;
    }
  }
  const int thr = best_i == 256 ? 0 : best_i;
  // Ink is gray <= thr: the core's ink pixels are the histogram up to it.
  int core_ink = 0;
  for (int i = lane; i <= thr; i += 32) core_ink += h[i];
  for (int o = 16; o > 0; o >>= 1) core_ink += __shfl_xor_sync(0xffffffffu, core_ink, o);
  if (lane == 0) {
    b.line_out[l].thr = thr;
    b.line_out[l].core_ink = core_ink;
  }
  // A line box mostly dark holds light type on a dark ground or heavy type the
  // box fits tightly; the ground is the side that surrounds the other: the
  // light is ground when most of the core's light pixels reach the crop's
  // border through light (4-connected), else the ink is inverted. Such a
  // line's tiles go to the polarity pass.
  if (2 * core_ink > L.w * L.h) {
    const int n = (L.W * L.H + kTilePixels - 1) / kTilePixels;
    int first = 0;
    if (lane == 0) first = atomicAdd(&b.totals->dark_tiles, n);
    first = __shfl_sync(0xffffffffu, first, 0);
    for (int k = lane; k < n; k += 32) b.dark_tiles[first + k] = L.tile_first + k;
  }
}

__global__ void k_ink(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const int thr = b.line_out[t.line].thr;
  const int base = b.lines[t.line].off + t.start;
  for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
    const int g = base + i;
    b.ink[g] = b.ink[g] > thr ? 0 : 1;
  }
}

// --- union-find ------------------------------------------------------------------
// Links only ever go to smaller indices (the first ones k_uf_init makes and
// every union), so each component's root is its first pixel in raster order:
// numbering roots in index order is the canonical order.

__device__ __forceinline__ int find_root(const int *lab, int n) {
  while (lab[n] != n) n = lab[n];
  return n;
}

__device__ void unite(int *lab, int a, int b) {
  bool done;
  do {
    a = find_root(lab, a);
    b = find_root(lab, b);
    if (a < b) {
      const int old = atomicMin(&lab[b], a);
      done = old == b;
      b = old;
    } else if (b < a) {
      const int old = atomicMin(&lab[a], b);
      done = old == a;
      a = old;
    } else {
      done = true;
    }
  } while (!done);
}

// --- union-find start --------------------------------------------------------------
// Each pixel of a side (ink 1, light 0) starts linked to the start of its
// horizontal run within its warp's 32 pixels, or, when the run comes in from
// the warp before, to the pixel left of the warp's first: the left unions,
// made without atomics. Every lane of the warp calls it, lane k on pixel g of
// column x (`in`: on that side), the warp's pixels consecutive.
__device__ __forceinline__ void link_run(const ChunkBuffers &b, int g, int x, bool in,
                                         std::uint8_t side) {
  const int lane = threadIdx.x & 31;
  const unsigned ins = __ballot_sync(0xffffffffu, in);
  const bool left_in = lane > 0 ? ((ins >> (lane - 1)) & 1u) != 0
                                : in && x > 0 && (b.ink[g - 1] != 0) == (side != 0);
  const bool joined = in && x > 0 && left_in;
  const unsigned starts = __ballot_sync(0xffffffffu, !joined);
  if (!in) return;
  const unsigned upto = starts & (0xffffffffu >> (31 - lane));  // lanes <= this one
  b.labels[g] = upto ? g - lane + (31 - __clz(upto)) : g - lane - 1;
}

// --- polarity (dark lines) ------------------------------------------------------
// Only a dark line's tiles (k_otsu lists them), each block striding over them.

constexpr int kPolarityBlocks = 256;

// The light pixels' sets (4-connected) start as their runs; none reaches the
// border yet.
__global__ void k_light_init(ChunkBuffers b) {
  for (int k = blockIdx.x; k < b.totals->dark_tiles; k += gridDim.x) {
    const Tile t = b.tiles[b.dark_tiles[k]];
    const LineDesc &L = b.lines[t.line];
    const int rounds = (t.count + kThreads - 1) / kThreads;
    for (int r = 0; r < rounds; ++r) {
      const int i = r * kThreads + threadIdx.x;
      const int p = t.start + i;
      const int g = L.off + p;
      const bool light = i < t.count && b.ink[g] == 0;
      link_run(b, g, p % L.W, light, 0);
      if (i < t.count) b.cid[g] = 0;
    }
  }
}

// The unions with the pixel above that no run makes: where the light above
// and to the left joins both already, the left neighbour's run has it.
__global__ void k_light_merge(ChunkBuffers b) {
  for (int k = blockIdx.x; k < b.totals->dark_tiles; k += gridDim.x) {
    const Tile t = b.tiles[b.dark_tiles[k]];
    const LineDesc &L = b.lines[t.line];
    for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
      const int p = t.start + i, x = p % L.W, y = p / L.W;
      const int g = L.off + p;
      if (b.ink[g] || y == 0 || b.ink[g - L.W]) continue;
      if (x > 0 && !b.ink[g - 1] && !b.ink[g - L.W - 1]) continue;
      unite(b.labels, g, g - L.W);
    }
  }
}

__global__ void k_light_roots(ChunkBuffers b) {
  for (int k = blockIdx.x; k < b.totals->dark_tiles; k += gridDim.x) {
    const Tile t = b.tiles[b.dark_tiles[k]];
    const LineDesc &L = b.lines[t.line];
    for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
      const int p = t.start + i, x = p % L.W, y = p / L.W;
      const int g = L.off + p;
      if (b.ink[g]) continue;
      const int root = find_root(b.labels, g);
      b.labels[g] = root;
      if (x == 0 || y == 0 || x == L.W - 1 || y == L.H - 1) b.cid[root] = 1;
    }
  }
}

__global__ void k_light_count(ChunkBuffers b) {
  for (int k = blockIdx.x; k < b.totals->dark_tiles; k += gridDim.x) {
    const Tile t = b.tiles[b.dark_tiles[k]];
    const LineDesc &L = b.lines[t.line];
    int light = 0, reach = 0;
    for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
      const int p = t.start + i, x = p % L.W, y = p / L.W;
      const int g = L.off + p;
      if (b.ink[g] || !in_core(L, x, y)) continue;
      ++light;
      reach += b.cid[b.labels[g]];
    }
    for (int o = 16; o > 0; o >>= 1) {
      light += __shfl_down_sync(0xffffffffu, light, o);
      reach += __shfl_down_sync(0xffffffffu, reach, o);
    }
    if ((threadIdx.x & 31) == 0) {
      if (light) atomicAdd(&b.line_out[t.line].light_px, light);
      if (reach) atomicAdd(&b.line_out[t.line].reaching, reach);
    }
  }
}

__global__ void k_invert(ChunkBuffers b) {
  for (int k = blockIdx.x; k < b.totals->dark_tiles; k += gridDim.x) {
    const Tile t = b.tiles[b.dark_tiles[k]];
    const LineOut &o = b.line_out[t.line];
    if (2 * o.reaching >= o.light_px) continue;  // the light is the ground
    const int base = b.lines[t.line].off + t.start;
    for (int i = threadIdx.x; i < t.count; i += blockDim.x)
      b.ink[base + i] = static_cast<std::uint8_t>(1 - b.ink[base + i]);
  }
}

// --- other lines' boxes --------------------------------------------------------

__device__ __forceinline__ bool nb_contains(const ChunkBuffers &b, const NbDesc &n, int x, int y) {
  const int i = y - n.run_y0;
  if (i < 0 || i >= n.run_rows) return false;
  const std::int16_t *r = b.runs + 4 * (n.run_first + i);
  return (x >= r[0] && x <= r[1]) || (x >= r[2] && x <= r[3]);
}

// The claims pass of locate_words, per ink pixel: `boxed` inside any other box;
// the claiming line (1-based among the claimers, in page order) nearest the
// pixel's row of those whose box holds it; `deep` once a claim passes that
// line's middle. The union-find starts in the same pass.
__global__ void k_claims(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  bool any = false;
  const int rounds = (t.count + kThreads - 1) / kThreads;
  for (int r = 0; r < rounds; ++r) {
    const int i = r * kThreads + threadIdx.x;
    const int p = t.start + i, x = p % L.W, y = p / L.W;
    const bool ink = i < t.count && b.ink[L.off + p] != 0;
    link_run(b, L.off + p, x, ink, 1);
    if (!ink) continue;  // only a component's pixels are read back
    int boxed = 0, tag = 0, deep = 0, claimer = 0;
    float tlx = 0.0f, tly = 0.0f, tslope = 0.0f;
    const float fx = static_cast<float>(x), fy = static_cast<float>(y);
    for (int j = 0; j < L.nb_count; ++j) {
      const NbDesc &n = b.nbs[L.nb_first + j];
      const bool inside = nb_contains(b, n, x, y);
      boxed |= inside ? 1 : 0;
      if (!n.claims) continue;
      ++claimer;
      if (!inside || x < n.wx0 || x >= n.wx1 || y < n.wy0 || y >= n.wy1) continue;
      const float oy = __fadd_rn(n.ly, __fmul_rn(n.slope, __fsub_rn(fx, n.lx)));
      if (fabsf(__fsub_rn(fy, oy)) >= fabsf(__fsub_rn(fy, L.own_mid))) continue;
      if (tag) {
        const float d = __fsub_rn(__fsub_rn(fy, tly), __fmul_rn(tslope, __fsub_rn(fx, tlx)));
        if (fabsf(d) <= fabsf(__fsub_rn(fy, oy))) continue;  // a nearer line claims it
      }
      tag = claimer;
      tlx = n.lx;
      tly = n.ly;
      tslope = n.slope;
      if (__fmul_rn(__fsub_rn(oy, L.own_mid), __fsub_rn(fy, oy)) > 0.0f) deep = 1;
    }
    b.claims[L.off + p] = static_cast<std::uint16_t>(tag | (deep << 8) | (boxed << 9));
    any |= tag != 0;
  }
  if (__syncthreads_or(any) && threadIdx.x == 0) b.line_out[t.line].any_claim = 1;
}

// --- connected components (8-connectivity) ---------------------------------------

__global__ void k_uf_init(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  const int rounds = (t.count + kThreads - 1) / kThreads;
  for (int r = 0; r < rounds; ++r) {
    const int i = r * kThreads + threadIdx.x;
    const int p = t.start + i;
    const int g = L.off + p;
    link_run(b, g, p % L.W, i < t.count && b.ink[g] != 0, 1);
  }
}

__global__ void k_uf_merge(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
    const int p = t.start + i, g = L.off + p;
    if (!b.ink[g] || p < L.W) continue;  // row 0: only the left union
    const int x = p % L.W;
    // Only the unions no neighbour makes (the left one k_uf_init made): a
    // left neighbour in ink has joined the pixels above it (this one's
    // up-left and up); an up neighbour in ink has joined its left and right
    // (this one's up-left and up-right).
    const bool left = x > 0 && b.ink[g - 1];
    const int up = g - L.W;
    if (b.ink[up]) {
      if (!left) unite(b.labels, g, up);
    } else {
      if (!left && x > 0 && b.ink[up - 1]) unite(b.labels, g, up - 1);
      if (x + 1 < L.W && b.ink[up + 1]) unite(b.labels, g, up + 1);
    }
  }
}

// A thread's pixels of a tile: k * kThreads + threadIdx.x for k < kPerThread.
constexpr int kPerThread = kTilePixels / kThreads;
static_assert(kTilePixels % kThreads == 0);

// Every ink pixel to its root; each tile's roots are counted, and marked in
// its mask (bit i % 32 of word i / 32 for the tile's pixel i).
__global__ void k_uf_roots(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const int base = b.lines[t.line].off + t.start;
  // The thread's pixels climb to their roots together, a step each per turn.
  int at[kPerThread];
  unsigned ink = 0;
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    const int i = k * kThreads + threadIdx.x;
    at[k] = base + i;
    if (i < t.count && b.ink[base + i]) ink |= 1u << k;
  }
  for (unsigned climbing = ink; climbing;) {
#pragma unroll
    for (int k = 0; k < kPerThread; ++k) {
      if (!((climbing >> k) & 1u)) continue;
      const int up = b.labels[at[k]];
      if (up == at[k]) climbing &= ~(1u << k);
      else at[k] = up;
    }
  }
  int roots = 0;
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    const int i = k * kThreads + threadIdx.x;
    const bool in = (ink >> k) & 1u;
    if (in) b.labels[base + i] = at[k];
    const bool root = in && at[k] == base + i;
    const unsigned mask = __ballot_sync(0xffffffffu, root);
    if ((threadIdx.x & 31) == 0) b.root_mask[blockIdx.x * kTileWords + i / 32] = mask;
    roots += __syncthreads_count(root);
  }
  if (threadIdx.x == 0) b.tile_roots[blockIdx.x] = roots;
}

// Roots numbered in pixel order, a warp a tile: the tile's base (the scan of
// the counts), then their rank in its mask.
__global__ void k_number(ChunkBuffers b) {
  const int tile = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
  if (tile >= b.n_tiles) return;
  const int lane = threadIdx.x & 31;
  const Tile t = b.tiles[tile];
  unsigned mask = lane * 32 < t.count ? b.root_mask[tile * kTileWords + lane] : 0u;
  const int n = __popc(mask);
  int upto = n;
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, upto, o);
    if (lane >= o) upto += v;
  }
  int next = b.tile_base[tile] + upto - n;
  const int base = b.lines[t.line].off + t.start + lane * 32;
  for (; mask; mask &= mask - 1) b.cid[base + __ffs(mask) - 1] = next++;
}

__global__ void k_comp_totals(ChunkBuffers b) {
  // The chunk's count and each line's first component.
  const int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l == 0) {
    const int last = b.n_tiles - 1;
    const int n = b.tile_base[last] + b.tile_roots[last];
    b.totals->comps = n;
    b.totals->comps_overflow = n > b.comp_capacity ? 1 : 0;
  }
  if (l < b.n_lines) b.line_out[l].comp_first = b.tile_base[b.lines[l].tile_first];
}

__global__ void k_comp_init(ChunkBuffers b) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= b.comp_capacity || c >= b.totals->comps) return;
  CompStats s;
  s.line = -1;
  s.area = 0;
  s.x0 = INT_MAX; s.y0 = INT_MAX; s.x1 = -1; s.y1 = -1;
  s.in_core = 0; s.core_x0 = INT_MAX; s.core_x1 = -1;
  s.claimed = 0; s.deep = 0; s.band = 0; s.boxed = 0;
  s.unused = 0;
  s.sumx = 0;
  b.comps[c] = s;
}

// Per-component statistics. Lanes of a warp on the same component combine
// their pixels first (the group's first lane reads the others' from shared
// memory), so each field takes one atomic per group, not one per pixel.
__global__ void k_stats(ChunkBuffers b) {
  constexpr int kWarps = kThreads / 32;
  __shared__ int px[kWarps][32], py[kWarps][32], pf[kWarps][32];
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int rounds = (t.count + kThreads - 1) / kThreads;
  for (int r = 0; r < rounds; ++r) {
    const int i = r * kThreads + threadIdx.x;
    int key = -1, x = 0, y = 0, flags = 0;
    if (i < t.count) {
      const int p = t.start + i, g = L.off + p;
      if (b.ink[g]) {
        x = p % L.W;
        y = p / L.W;
        const int c = b.cid[b.labels[g]];
        if (c < b.comp_capacity) {
          key = c;
          const std::uint16_t cl = b.claims[g];
          flags = (in_core(L, x, y) ? 1 : 0) | ((cl & 0xff) ? 2 : 0) | ((cl & 0x100) ? 4 : 0) |
                  ((cl & 0x200) ? 8 : 0) | ((y >= L.m && y < L.m + L.h) ? 16 : 0);
        }
      }
    }
    if (__ballot_sync(0xffffffffu, key >= 0) == 0) continue;
    px[warp][lane] = x;
    py[warp][lane] = y;
    pf[warp][lane] = flags;
    const unsigned peers = __match_any_sync(0xffffffffu, key);
    __syncwarp();
    if (key >= 0 && lane == __ffs(peers) - 1) {
      int x0 = INT_MAX, x1 = -1, y0 = INT_MAX, y1 = -1, core = 0, cx0 = INT_MAX, cx1 = -1;
      int claimed = 0, deep = 0, boxed = 0, band = 0;
      unsigned sumx = 0;
      for (unsigned m = peers; m; m &= m - 1) {
        const int src = __ffs(m) - 1;
        const int sx = px[warp][src], sy = py[warp][src], sf = pf[warp][src];
        x0 = min(x0, sx); x1 = max(x1, sx);
        y0 = min(y0, sy); y1 = max(y1, sy);
        sumx += static_cast<unsigned>(sx);
        if (sf & 1) { ++core; cx0 = min(cx0, sx); cx1 = max(cx1, sx + 1); }
        claimed += (sf >> 1) & 1;
        deep += (sf >> 2) & 1;
        boxed += (sf >> 3) & 1;
        band += (sf >> 4) & 1;
      }
      CompStats &s = b.comps[key];
      if (atomicAdd(&s.area, __popc(peers)) == 0) s.line = t.line;
      atomicMin(&s.x0, x0);
      atomicMax(&s.x1, x1);
      atomicMin(&s.y0, y0);
      atomicMax(&s.y1, y1);
      atomicAdd(&s.sumx, static_cast<unsigned long long>(sumx));
      if (core) {
        atomicAdd(&s.in_core, core);
        atomicMin(&s.core_x0, cx0);
        atomicMax(&s.core_x1, cx1);
      }
      if (claimed) atomicAdd(&s.claimed, claimed);
      if (deep) atomicAdd(&s.deep, deep);
      if (boxed) atomicAdd(&s.boxed, boxed);
      if (band) atomicAdd(&s.band, band);
    }
    __syncwarp();
  }
}

// --- bleed split -------------------------------------------------------------------
// A glyph reaching deep into another line, with a real share left on this
// line's side, may hold glyphs of both joined by bleed: where its row profile
// between the two middles shows a clear valley, the part past it leaves the
// ink. locate_words' CPU loop, per candidate.

__global__ void k_cand_flags(ChunkBuffers b) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= b.comp_capacity) return;
  int flag = 0;
  if (c < b.totals->comps) {
    const CompStats &s = b.comps[c];
    flag = s.line >= 0 && b.line_out[s.line].any_claim &&
           s.deep >= max(3, s.area / 50) && 4 * (s.area - s.claimed) >= s.area;
  }
  b.colw[c] = flag;
}

__global__ void k_cand_totals(ChunkBuffers b) {
  const int last = b.comp_capacity - 1;
  b.totals->cands = b.hoff[last] + b.colw[last];
  b.totals->any_split = 0;
}

__global__ void k_cand_list(ChunkBuffers b) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= b.comp_capacity || c >= b.totals->comps || !b.colw[c]) return;
  CandPar &p = b.cands[b.hoff[c]];
  p.comp = c;
  p.line = b.comps[c].line;
  p.nb = 0;
  p.do_cut = 0;
}

__global__ void k_bytag(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
    const int g = L.off + t.start + i;
    if (!b.ink[g] || !(b.claims[g] & 0x100)) continue;
    const int c = b.cid[b.labels[g]];
    if (c >= b.comp_capacity || !b.colw[c]) continue;
    atomicAdd(&b.bytag[b.hoff[c] * kProfileBins + (b.claims[g] & 0xff)], 1);
  }
}

// The claiming line most of its deep pixels belong to (the first of equals),
// and the profile's bins.
__global__ void k_split_plan(ChunkBuffers b) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= b.totals->cands) return;
  CandPar &p = b.cands[k];
  const LineDesc &L = b.lines[p.line];
  const int *tag = b.bytag + k * kProfileBins;
  int claimers = 0, best = 0, best_n = -1;
  float lx = 0.0f, ly = 0.0f, slope = 0.0f;
  for (int j = 0; j < L.nb_count; ++j) {
    const NbDesc &n = b.nbs[L.nb_first + j];
    if (!n.claims) continue;
    ++claimers;
    if (tag[claimers] > best_n) {
      best_n = tag[claimers];
      best = claimers;
      lx = n.lx;
      ly = n.ly;
      slope = n.slope;
    }
  }
  if (best == 0) return;
  const CompStats &s = b.comps[p.comp];
  const float cx = __fmul_rn(0.5f, static_cast<float>(s.x0 + s.x1 + 1));
  const float span =
      fabsf(__fsub_rn(__fadd_rn(ly, __fmul_rn(slope, __fsub_rn(cx, lx))), L.own_mid));
  const int nb = max(4, static_cast<int>(__fdiv_rn(span, 2.0f)));
  const int nb_all = nb + nb / 2;
  if (nb_all > kProfileBins) {  // finer than the bins here: the CPU code places the line
    b.line_out[p.line].cpu = 1;
    return;
  }
  p.lx = lx;
  p.ly = ly;
  p.slope = slope;
  p.nb = nb;
  p.nb_all = nb_all;
}

// Position of a pixel between the line's middle (0) and the other's (1).
__device__ __forceinline__ float frac_of(const CandPar &p, float own_mid, int x, int y) {
  const float oy = __fadd_rn(p.ly, __fmul_rn(p.slope, __fsub_rn(static_cast<float>(x), p.lx)));
  return __fdiv_rn(__fsub_rn(static_cast<float>(y), own_mid), __fsub_rn(oy, own_mid));
}

__global__ void k_prof(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
    const int p = t.start + i, x = p % L.W, y = p / L.W;
    const int g = L.off + p;
    if (!b.ink[g]) continue;
    const int c = b.cid[b.labels[g]];
    if (c >= b.comp_capacity || !b.colw[c]) continue;
    const int k = b.hoff[c];
    const CandPar &cp = b.cands[k];
    if (cp.nb == 0) continue;
    const float f = frac_of(cp, L.own_mid, x, y);
    if (f >= 0.0f && f < 1.5f) {
      const int bin = min(static_cast<int>(__float2uint_rz(__fmul_rn(f, static_cast<float>(cp.nb)))),
                          cp.nb_all - 1);
      atomicAdd(&b.prof[k * kProfileBins + bin], 1);
    }
  }
}

__global__ void k_split_decide(ChunkBuffers b) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= b.totals->cands) return;
  CandPar &p = b.cands[k];
  if (p.nb == 0) return;
  const int *pr = b.prof + k * kProfileBins;
  const int nb = p.nb;
  int lo = 0, hi = p.nb_all;
  while (lo < hi && pr[lo] == 0) ++lo;
  while (hi > lo && pr[hi - 1] == 0) --hi;
  if (hi - lo < 3) return;
  int best = lo + 1;
  for (int j = lo + 1; j + 1 < hi && j < nb; ++j)
    if (pr[j] < pr[best] || (pr[j] == pr[best] && abs(j - nb / 2) < abs(best - nb / 2)))
      best = j;
  int before = 0, after = 0;
  for (int j = lo; j < best; ++j) before = max(before, pr[j]);
  for (int j = best + 1; j < hi; ++j) after = max(after, pr[j]);
  if (3 * pr[best] > min(before, after)) return;
  p.cut = __fdiv_rn(__fadd_rn(static_cast<float>(best), 0.5f), static_cast<float>(nb));
  p.do_cut = 1;
  b.totals->any_split = 1;
}

__global__ void k_cut(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
    const int p = t.start + i, x = p % L.W, y = p / L.W;
    const int g = L.off + p;
    if (!b.ink[g]) continue;
    const int c = b.cid[b.labels[g]];
    if (c >= b.comp_capacity || !b.colw[c]) continue;
    const CandPar &cp = b.cands[b.hoff[c]];
    if (cp.do_cut && frac_of(cp, L.own_mid, x, y) > cp.cut) b.ink[g] = 0;
  }
}

__global__ void k_colw(ChunkBuffers b) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= b.comp_capacity) return;
  const CompStats &s = b.comps[c];
  b.colw[c] = c < b.totals->comps && s.in_core > 0 ? s.core_x1 - s.core_x0 : 0;
}

__global__ void k_hist_totals(ChunkBuffers b) {
  const int last = b.comp_capacity - 1;
  const int n = b.hoff[last] + b.colw[last];
  b.totals->hist = n;
  b.totals->hist_overflow = n > b.hist_capacity ? 1 : 0;
}

// Per component, its core pixels per column: the line box's column profile of
// any set of components is the sum of theirs.
__global__ void k_colhist(ChunkBuffers b) {
  const Tile t = b.tiles[blockIdx.x];
  const LineDesc &L = b.lines[t.line];
  for (int i = threadIdx.x; i < t.count; i += blockDim.x) {
    const int p = t.start + i, g = L.off + p;
    if (!b.ink[g]) continue;
    const int x = p % L.W, y = p / L.W;
    if (!in_core(L, x, y)) continue;
    const int c = b.cid[b.labels[g]];
    if (c >= b.comp_capacity) continue;
    const int idx = b.hoff[c] + x - b.comps[c].core_x0;
    if (idx < b.hist_capacity) atomicAdd(&b.colhist[idx], 1);
  }
}

}  // namespace

std::size_t scan_scratch_bytes(int n) {
  std::size_t bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const int *>(nullptr),
                                           static_cast<int *>(nullptr), n));
  return bytes;
}

namespace {

void scan(const ChunkBuffers &b, const int *in, int *out, int n, cudaStream_t stream) {
  std::size_t tmp = b.scan_tmp_bytes;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(b.scan_tmp, tmp, in, out, n, stream));
}

// Components of the ink, the union-find started (k_claims or k_uf_init).
void label(const ChunkBuffers &b, cudaStream_t stream) {
  k_uf_merge<<<b.n_tiles, kThreads, 0, stream>>>(b);
  k_uf_roots<<<b.n_tiles, kThreads, 0, stream>>>(b);
  scan(b, b.tile_roots, b.tile_base, b.n_tiles, stream);
  k_number<<<(b.n_tiles + kThreads / 32 - 1) / (kThreads / 32), kThreads, 0, stream>>>(b);
  k_comp_totals<<<(b.n_lines + kThreads - 1) / kThreads, kThreads, 0, stream>>>(b);
}

int blocks(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void stage_pixels(const PageView &page, const ChunkBuffers &b, cudaStream_t stream) {
  k_crop_gray<true><<<b.n_tiles, kThreads, 0, stream>>>(page, b);
  k_otsu<<<b.n_lines, 32, 0, stream>>>(b);
  k_ink<<<b.n_tiles, kThreads, 0, stream>>>(b);
  k_light_init<<<kPolarityBlocks, kThreads, 0, stream>>>(b);
  k_light_merge<<<kPolarityBlocks, kThreads, 0, stream>>>(b);
  k_light_roots<<<kPolarityBlocks, kThreads, 0, stream>>>(b);
  k_light_count<<<kPolarityBlocks, kThreads, 0, stream>>>(b);
  k_invert<<<kPolarityBlocks, kThreads, 0, stream>>>(b);
  k_claims<<<b.n_tiles, kThreads, 0, stream>>>(b);
  label(b, stream);
  CUDA_CHECK(cudaGetLastError());
}

void stage_stats(const ChunkBuffers &b, cudaStream_t stream) {
  k_comp_init<<<blocks(b.comp_capacity), kThreads, 0, stream>>>(b);
  k_stats<<<b.n_tiles, kThreads, 0, stream>>>(b);
  k_cand_flags<<<blocks(b.comp_capacity), kThreads, 0, stream>>>(b);
  scan(b, b.colw, b.hoff, b.comp_capacity, stream);
  k_cand_totals<<<1, 1, 0, stream>>>(b);
  CUDA_CHECK(cudaGetLastError());
}

void stage_split(const ChunkBuffers &b, cudaStream_t stream) {
  const std::size_t bins = sizeof(int) * kProfileBins * static_cast<std::size_t>(b.cand_capacity);
  CUDA_CHECK(cudaMemsetAsync(b.bytag, 0, bins, stream));
  CUDA_CHECK(cudaMemsetAsync(b.prof, 0, bins, stream));
  k_cand_list<<<blocks(b.comp_capacity), kThreads, 0, stream>>>(b);
  k_bytag<<<b.n_tiles, kThreads, 0, stream>>>(b);
  k_split_plan<<<blocks(b.cand_capacity), kThreads, 0, stream>>>(b);
  k_prof<<<b.n_tiles, kThreads, 0, stream>>>(b);
  k_split_decide<<<blocks(b.cand_capacity), kThreads, 0, stream>>>(b);
  k_cut<<<b.n_tiles, kThreads, 0, stream>>>(b);
  CUDA_CHECK(cudaGetLastError());
}

void stage_relabel(const ChunkBuffers &b, cudaStream_t stream) {
  k_uf_init<<<b.n_tiles, kThreads, 0, stream>>>(b);
  label(b, stream);
  CUDA_CHECK(cudaGetLastError());
}

void stage_hist(const ChunkBuffers &b, cudaStream_t stream) {
  k_colw<<<blocks(b.comp_capacity), kThreads, 0, stream>>>(b);
  scan(b, b.colw, b.hoff, b.comp_capacity, stream);
  k_hist_totals<<<1, 1, 0, stream>>>(b);
  CUDA_CHECK(cudaMemsetAsync(b.colhist, 0,
                             sizeof(int) * static_cast<std::size_t>(b.hist_capacity), stream));
  k_colhist<<<b.n_tiles, kThreads, 0, stream>>>(b);
  CUDA_CHECK(cudaGetLastError());
}

void crop_gray(const PageView &page, const ChunkBuffers &b, cudaStream_t stream) {
  k_crop_gray<false><<<b.n_tiles, kThreads, 0, stream>>>(page, b);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbo_ocr::recognition::gpu
