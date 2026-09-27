#pragma once

#include <algorithm>
#include <cstddef>

#include "turbo_ocr/common/log/logger.h"

namespace turbo_ocr::server {

// VRAM left free when sizing the pool, so the renderer daemons / CUDA context
// / OS don't get squeezed to the byte.
inline constexpr size_t kVramHeadroomBytes = size_t{1} << 30;

// Pipeline pool auto-sizing policy. Throughput is GPU-compute-bound on this
// stack: a clean pool sweep (RTX 5090, FUNSD) plateaus by ~5 pipelines
// (5→278, 8→263, 10→258 img/s) because the det/rec kernels already saturate
// the GPU and extra pipelines only add scheduling/cache pressure. So the
// ladder caps at 5 deliberately — raising it costs VRAM for negative
// throughput. An explicit --pool-size / PIPELINE_POOL_SIZE override bypasses
// this function entirely.
//
// Pure policy over the cudaMemGetInfo numbers (no CUDA dependency) so it is
// unit-testable on the CPU build.
[[nodiscard]] inline int compute_pipeline_pool_size(size_t free_mem,
                                                    size_t total_mem) {
  int pool_size;
  int vram_gb = static_cast<int>(total_mem >> 30);
  if (vram_gb >= 14) pool_size = 5;
  else if (vram_gb >= 12) pool_size = 3;
  else if (vram_gb >= 8)  pool_size = 2;
  else                     pool_size = 1;

  // Footprint-based safety floor: the tier above keys off TOTAL VRAM, but a
  // card that *reports* 16 GB while another process already holds most of
  // it would OOM during warmup. Estimate each pipeline's resident footprint
  // (engines + activation/workspace buffers, generous to stay conservative)
  // and reduce the tier so it fits in the FREE VRAM measured right now.
  // This only ever LOWERS the count — never raises it above the tier cap —
  // and never below 1, so a healthy 32 GB card with the tiny model still
  // picks 5 (a few hundred MB per pipeline against ~31 GB free).
  constexpr size_t kPerPipelineFootprintBytes = size_t{2} << 30;  // 2 GiB
  size_t budget = free_mem > kVramHeadroomBytes ? free_mem - kVramHeadroomBytes : 0;
  int fits = static_cast<int>(budget / kPerPipelineFootprintBytes);
  if (fits < 1) fits = 1;
  if (fits < pool_size) {
    TOCR_LOG_WARN("Reducing pipeline pool to fit available VRAM",
                  "tier_pool_size", pool_size, "footprint_capped", fits,
                  "free_mem_gb", static_cast<int>(free_mem >> 30));
    pool_size = fits;
  }
  TOCR_LOG_INFO("Auto-detected pipeline pool size", "pool_size", pool_size, "vram_gb", vram_gb);
  return pool_size;
}

// What a pipeline grows by after warmup (per-replica decoders, scratch sized
// by the first large requests): measured ~0.36 GiB per replica on the tiny
// models over a mixed-traffic soak.
inline constexpr size_t kPipelineRuntimeReserveBytes = size_t{512} << 20;

// Auto mode, once the first pipeline is built and warmed up: how many the pool
// gets. The fixed 2 GiB guess above is right for the tiny models but the
// medium ones (det + rec scratch, 18,710 recognizer classes) take ~5 GiB, so a
// card with room for three pipelines ran out of memory building the fourth and
// the server never started. Each further pipeline costs what the first one
// measurably took (`first_bytes`) plus the runtime reserve; the headroom stays
// free. Only lowers `cap`, never below 1; no measurement keeps `cap`.
[[nodiscard]] inline int fit_pipeline_count(int cap, size_t first_bytes,
                                            size_t free_after) {
  if (cap <= 1 || first_bytes == 0) return std::max(cap, 1);
  const size_t keep = kVramHeadroomBytes + kPipelineRuntimeReserveBytes;
  const size_t budget = free_after > keep ? free_after - keep : 0;
  const size_t more = budget / (first_bytes + kPipelineRuntimeReserveBytes);
  return static_cast<int>(std::min<size_t>(1 + more, static_cast<size_t>(cap)));
}

// Work-pool threads in front of the replica pool: they decode, parse and
// serialise while the replicas infer, so a few per replica keep the GPU fed
// and more only add idle threads. Measured flat from 20 to 48 threads on an
// RTX 5090 (pool 5); the previous max(pool*32, 128) put 128-160 threads on a
// 20-core box. Every extra thread is one more per-thread scratch set and one
// more allocator arena holding its own high-water mark of freed request
// buffers, which is host RSS that never comes back. HTTP_THREADS overrides.
[[nodiscard]] constexpr int compute_work_threads(int pool_size) noexcept {
  return std::clamp(pool_size * 4, 16, 64);
}

} // namespace turbo_ocr::server
