// GPU_CCL=2: every accepted component's oriented rectangle, fully on the GPU.
//
// The detector's text regions come out of CCL (ccl_kernels.cu) as compact ids
// with a score sum and a pixel count. The components that pass the size and
// score filters are labelled, and one PCA reduction per component gives its
// rotated bounding rectangle -- the GPU's stand-in for cv::minAreaRect on the
// component's contour. The host then unclips that rectangle exactly as the
// other detector paths do (detection::region_to_box), so every path shares one
// geometry: the rectangle grown by area * ratio / perimeter on every side.

#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

namespace turbo_ocr::kernels {

// One thread per pixel: a foreground pixel takes its component's label
// (compact id + 1) when the component passes the filters every detector path
// applies before the unclip -- at least 3 pixels, a bounding box at least 3 px
// a side, a mean score of at least box_thresh; everything else is 0.
__global__ void label_accepted_kernel(uint32_t *labels, const uint8_t *bitmap,
                                      const int32_t *compact_ids,
                                      const GpuDetBox *bboxes, int num_slots,
                                      float box_thresh, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;
    uint32_t out = 0;
    int cid = bitmap[idx] ? compact_ids[idx] : -1;
    if (cid >= 0 && cid < num_slots) {
        const GpuDetBox &b = bboxes[cid];
        const int pc = b.pixel_count;
        if (pc >= 3 && b.xmax - b.xmin + 1 >= 3 && b.ymax - b.ymin + 1 >= 3 &&
            b.score / (float)pc >= box_thresh)
            out = (uint32_t)(cid + 1);
    }
    labels[idx] = out;
}

// Init bbox slots with sentinels so the atomic min/max scatter works.
__global__ void init_bboxes_kernel(GpuDetBox *bboxes, int num_slots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    GpuDetBox &b = bboxes[i];
    b.xmin = INT_MAX; b.ymin = INT_MAX;
    b.xmax = INT_MIN; b.ymax = INT_MIN;
    b.pixel_count = 0;
    b.score = 0.0f;
}

// ---------------------------------------------------------------------------
// Oriented (rotated) min-area-rect via PCA over the component's pixels.
//
// cv2.minAreaRect is rotating-calipers on the convex hull (exact min area).
// PCA minimizes the SECOND MOMENT, not the bounding area, so the two agree only
// for a uniform-density rectangle; on real text blobs (ascenders/descenders,
// ragged ends, uneven stroke density) the PCA angle differs from minAreaRect's
// by typically a degree or two — negligible for crop-and-recognize, and measured
// at parity with the CPU reference on skewed text. One known residual:
// near-isotropic blobs, where the PCA angle is ill-conditioned -- guarded below
// by an anisotropy fallback to axis-aligned. PCA is chosen because it is a
// two-pass atomic reduction (GPU-friendly, no per-component hull construction),
// which is what the sibling detector used to reach official parity.
//
// Layout: mom[cid*6 + {0..5}] = {n, sx, sy, sxx, syy, sxy} (uint64).
//         orient[cid*6 + {0..5}] = {cos, sin, umin, umax, vmin, vmax} (float).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float atomic_min_f(float *addr, float val) {
    int *ai = (int *)addr;
    int old = *ai, assumed;
    do {
        assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomic_max_f(float *addr, float val) {
    int *ai = (int *)addr;
    int old = *ai, assumed;
    do {
        assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Pass A: one thread per labelled pixel — axis-aligned bbox plus the integer
// second-moment accumulation for PCA.
__global__ void oriented_extract_kernel(
    const uint32_t *labels, int w, int h,
    GpuDetBox *bboxes, unsigned long long *mom, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    uint32_t label = labels[idx];
    if (label == 0) return;
    int cid = (int)label - 1;
    if (cid >= num_slots) return;
    int x = idx % w, y = idx / w;
    GpuDetBox &b = bboxes[cid];
    atomicMin(&b.xmin, x); atomicMax(&b.xmax, x);
    atomicMin(&b.ymin, y); atomicMax(&b.ymax, y);
    atomicAdd(&b.pixel_count, 1);
    unsigned long long *m = mom + (size_t)cid * 6;
    // Widen BEFORE multiplying: int x*x overflows at coords >= 46341. Today's
    // det resize caps coords at 4096, but the cliff would be silent.
    const unsigned long long ux = (unsigned long long)x, uy = (unsigned long long)y;
    atomicAdd(&m[0], 1ULL);
    atomicAdd(&m[1], ux);
    atomicAdd(&m[2], uy);
    atomicAdd(&m[3], ux * ux);
    atomicAdd(&m[4], uy * uy);
    atomicAdd(&m[5], ux * uy);
}

// Pass B (per component): principal-axis angle from the covariance, then seed
// the projection extents. Double math here is safe — one thread per component,
// not a hot per-pixel loop.
__global__ void oriented_axis_kernel(const unsigned long long *mom,
                                     int num_slots, float *orient) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    const unsigned long long *m = mom + (size_t)i * 6;
    float *o = orient + (size_t)i * 6;
    unsigned long long n = m[0];
    o[2] = INFINITY; o[3] = -INFINITY; o[4] = INFINITY; o[5] = -INFINITY;
    if (n < 3) { o[0] = 1.0f; o[1] = 0.0f; return; } // degenerate → axis-aligned
    double dn = (double)n;
    double mx = (double)m[1] / dn, my = (double)m[2] / dn;
    double cxx = (double)m[3] / dn - mx * mx;
    double cyy = (double)m[4] / dn - my * my;
    double cxy = (double)m[5] / dn - mx * my;
    // Near-isotropic guard: with no dominant axis (cxx≈cyy, cxy≈0) the principal
    // angle is ill-conditioned — atan2(2cxy, cxx-cyy) snaps ~45° on tiny
    // perturbations, so a near-square blob (single glyph, CJK char) would get a
    // spuriously rotated rect where minAreaRect stays axis-aligned. The anisotropy
    // (λmax-λmin)/(λmax+λmin) = sqrt((cxx-cyy)^2 + 4cxy^2)/(cxx+cyy); below a few
    // percent the axis is meaningless, so fall back to axis-aligned (angle 0).
    double trace = cxx + cyy;
    double aniso = sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy);
    if (trace <= 0.0 || aniso < 0.05 * trace) { o[0] = 1.0f; o[1] = 0.0f; return; }
    double theta = 0.5 * atan2(2.0 * cxy, cxx - cyy);
    o[0] = (float)cos(theta);
    o[1] = (float)sin(theta);
}

// Pass C: one thread per labelled pixel — project onto the component's
// oriented basis and track the min/max extent along each axis.
__global__ void oriented_project_kernel(
    const uint32_t *labels, int w, int h,
    const unsigned long long *mom, float *orient, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    uint32_t label = labels[idx];
    if (label == 0) return;
    int cid = (int)label - 1;
    if (cid >= num_slots) return;
    if (mom[(size_t)cid * 6] < 3) return;
    int x = idx % w, y = idx / w;
    float *o = orient + (size_t)cid * 6;
    float c = o[0], s = o[1];
    float u = x * c + y * s;
    float v = -x * s + y * c;
    atomic_min_f(&o[2], u); atomic_max_f(&o[3], u);
    atomic_min_f(&o[4], v); atomic_max_f(&o[5], v);
}

// Pass D (per component): reconstruct the 4 oriented-rect corners from the axis
// and the per-axis extents. Degenerate components fall back to the axis-aligned
// bbox (a rotated rect with angle 0).
__global__ void oriented_emit_kernel(const unsigned long long *mom,
                                     const float *orient, int num_slots,
                                     GpuDetBox *bboxes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    GpuDetBox &b = bboxes[i];
    const float *o = orient + (size_t)i * 6;
    if (mom[(size_t)i * 6] < 3 || o[2] > o[3]) {
        float x0 = (float)b.xmin, y0 = (float)b.ymin;
        float x1 = (float)b.xmax, y1 = (float)b.ymax;
        b.ox[0] = x0; b.oy[0] = y0;
        b.ox[1] = x1; b.oy[1] = y0;
        b.ox[2] = x1; b.oy[2] = y1;
        b.ox[3] = x0; b.oy[3] = y1;
        return;
    }
    float c = o[0], s = o[1];
    float umin = o[2], umax = o[3], vmin = o[4], vmax = o[5];
    // corner(u,v) = u*(c,s) + v*(-s,c)
    const float uu[4] = {umin, umax, umax, umin};
    const float vv[4] = {vmin, vmin, vmax, vmax};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        b.ox[k] = uu[k] * c - vv[k] * s;
        b.oy[k] = uu[k] * s + vv[k] * c;
    }
}

void cuda_extract_oriented_rects(const uint32_t *d_labels,
                                 int w, int h,
                                 GpuDetBox *d_bboxes, int num_slots,
                                 unsigned long long *d_moments, float *d_orient,
                                 cudaStream_t stream) {
    if (num_slots <= 0) return;
    int total = w * h;
    int block = 256;
    int comp_grid = (num_slots + block - 1) / block;
    int pix_grid = (total + block - 1) / block;

    init_bboxes_kernel<<<comp_grid, block, 0, stream>>>(d_bboxes, num_slots);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(d_moments, 0,
                               (size_t)num_slots * 6 * sizeof(unsigned long long), stream));

    oriented_extract_kernel<<<pix_grid, block, 0, stream>>>(
        d_labels, w, h, d_bboxes, d_moments, num_slots);
    CUDA_CHECK(cudaGetLastError());

    oriented_axis_kernel<<<comp_grid, block, 0, stream>>>(d_moments, num_slots, d_orient);
    CUDA_CHECK(cudaGetLastError());

    oriented_project_kernel<<<pix_grid, block, 0, stream>>>(
        d_labels, w, h, d_moments, d_orient, num_slots);
    CUDA_CHECK(cudaGetLastError());

    oriented_emit_kernel<<<comp_grid, block, 0, stream>>>(
        d_moments, d_orient, num_slots, d_bboxes);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_label_accepted_components(const uint8_t *d_bitmap,
                                    const int32_t *d_compact_ids,
                                    const GpuDetBox *d_bboxes, int num_slots,
                                    float box_thresh, uint32_t *d_labels,
                                    int w, int h, cudaStream_t stream) {
    int block = 256;
    int grid = (w * h + block - 1) / block;
    label_accepted_kernel<<<grid, block, 0, stream>>>(
        d_labels, d_bitmap, d_compact_ids, d_bboxes, num_slots, box_thresh, w, h);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::kernels
