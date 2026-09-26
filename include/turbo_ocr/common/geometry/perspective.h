#pragma once

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/geometry/perspective_math.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace turbo_ocr {

/// Result of computing a crop transform for a box.
struct CropTransform {
  float M_inv[9];  // 3x3 inverse perspective matrix
  int crop_width;  // width of the resized crop in pixels
  bool vertical;   // true if the box was rotated (vertical text)
};

/// Which corner of the crop -- 0 top-left, 1 top-right, 2 bottom-right,
/// 3 bottom-left -- the box's point k lands on. A tall (vertical-text) box is
/// turned a quarter anticlockwise, as PaddleOCR's np.rot90 turns it and so as
/// its recognition and orientation models were trained to read it: the top of
/// the column goes to the crop's left, its first character first.
[[nodiscard]] constexpr int crop_corner(bool vertical, int k) noexcept {
  return vertical ? (k + 3) % 4 : k;
}

/// Compute the inverse perspective transform that maps a destination rectangle
/// (0,0)-(resize_w, target_h) back to the quadrilateral defined by @p box in
/// the source image.  The destination width is clamped to [1, max_width].
///
/// The single source of truth for crop-transform logic: both recognizers
/// (TRT warp kernel, ORT cv::warpPerspective), both classifiers, and
/// rec_input_width all derive their geometry from this function.
inline CropTransform compute_crop_transform(const Box &box, int target_h,
                                            int max_width) {
  auto f = [](int v) { return static_cast<float>(v); };
  float bx0 = f(box[0][0]), by0 = f(box[0][1]);
  float bx1 = f(box[1][0]), by1 = f(box[1][1]);
  float bx3 = f(box[3][0]), by3 = f(box[3][1]);

  float crop_w =
      std::sqrt((bx0 - bx1) * (bx0 - bx1) + (by0 - by1) * (by0 - by1));
  float crop_h =
      std::sqrt((bx0 - bx3) * (bx0 - bx3) + (by0 - by3) * (by0 - by3));

  bool vertical = (crop_h >= crop_w * kVerticalAspectRatio);

  // src_f lists the box points in crop-corner order.
  float src_f[8];
  for (int k = 0; k < 4; ++k) {
    const int c = crop_corner(vertical, k);
    src_f[2 * c] = f(box[k][0]);
    src_f[2 * c + 1] = f(box[k][1]);
  }
  if (vertical) std::swap(crop_w, crop_h);

  float ar = (crop_h > 0) ? (crop_w / crop_h) : 0;
  int resize_w =
      std::min(static_cast<int>(std::ceil(target_h * ar)), max_width);
  resize_w = std::max(resize_w, 1);

  float rw = f(resize_w), rh = f(target_h);
  float dst_f[8] = {0, 0, rw, 0, rw, rh, 0, rh};

  CropTransform ct{};
  ct.crop_width = resize_w;
  ct.vertical = vertical;
  compute_perspective_inv(dst_f, src_f, ct.M_inv);
  return ct;
}

/// Map a point in crop (destination) space back to the source image through
/// a CropTransform's inverse perspective matrix.
[[nodiscard]] inline std::array<float, 2>
apply_crop_transform_inv(const CropTransform &ct, float x, float y) noexcept {
  const float *m = ct.M_inv;
  float w = m[6] * x + m[7] * y + m[8];
  if (std::fabs(w) < 1e-6f) w = (w < 0) ? -1e-6f : 1e-6f;
  return {(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w};
}

} // namespace turbo_ocr
