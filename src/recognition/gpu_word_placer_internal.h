#pragma once

// The GPU placer's line descriptor, shared with its unit tests.

#include "word_boxes_gpu.h"
#include "word_boxes_internal.h"

namespace turbo_ocr::recognition::gpu {

// A planned line as the kernels read it; its crop starts at pixel `off`.
LineDesc line_desc(const detail::LinePlan &p, int off);

}  // namespace turbo_ocr::recognition::gpu
