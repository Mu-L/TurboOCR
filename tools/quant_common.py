"""Shared preprocessing that mirrors the C++ CPU OCR pipeline exactly, so that
calibration / eval crops have the same activation statistics the deployed engine
sees. References:
  - det preprocess + DB postproc: src/detection/cpu_paddle_det.cpp,
    src/detection/det_postprocess.cpp, include/.../det_config.h
  - crop warp: include/.../perspective.h (compute_crop_transform),
    include/.../rec_geometry.h
  - rec preprocess: src/recognition/cpu_paddle_rec.cpp (preprocess_box)
"""
import glob
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNSD = os.path.join(os.path.dirname(REPO), "compare-ocrs", "funsd_cache")

# ---- det constants (cpu_paddle_det.h / det_config.h) ----
DET_LIMIT_SIDE_LEN = 64    # kDetResizeDefault: "min" policy, shorter side >= 64
DET_MAX_SIDE_LIMIT = 1280  # ... longer output side capped at 1280
DET_DB_THRESH = 0.2        # kDbDefaults (det.onnx; the tiny model uses box 0.40)
DET_DB_BOX_THRESH = 0.45
DET_DB_UNCLIP_RATIO = 1.4
DET_MAX_CANDIDATES = 3000  # det_postprocess.cpp kMaxCandidates
MIN_BOX_SIDE = 3.0
MIN_UNCLIPPED_SIDE = 5.0

# ---- rec constants (cpu_paddle_rec.h / rec_geometry.h) ----
REC_IMAGE_H = 48
REC_MIN_WIDTH = 32         # kMinRecWidth: narrower crops are right-padded
REC_MAX_WIDTH = 4000       # kMaxRecWidth
VERTICAL_AR = 1.5          # box.h kVerticalAspectRatio


def list_pages(n=None):
    imgs = sorted(glob.glob(f"{FUNSD}/*.png"))
    return imgs if n is None else imgs[:n]


def make_cpu_session(model_path, intra=0, inter=1, opt_all=True):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if opt_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


# ---------------- detection ----------------
def _round32(v):
    """det_config.h round32_floor (C++ std::round: halves away from zero)."""
    return max(int(math.floor(v / 32.0 + 0.5)) * 32, 32)


def det_resize(h, w):
    """Mirrors det_config.h compute_det_resize for kDetResizeDefault."""
    ratio = 1.0
    if min(h, w) < DET_LIMIT_SIDE_LEN:
        ratio = DET_LIMIT_SIDE_LEN / min(h, w)
    resize_h, resize_w = _round32(h * ratio), _round32(w * ratio)
    longest = max(resize_h, resize_w)
    if longest > DET_MAX_SIDE_LIMIT:
        rescale = DET_MAX_SIDE_LIMIT / longest
        resize_h, resize_w = _round32(resize_h * rescale), _round32(resize_w * rescale)
    return resize_h, resize_w


def det_preprocess(img_bgr):
    """Returns (nchw float32 input, resize_h, resize_w). Mirrors cpu_paddle_det.cpp."""
    h, w = img_bgr.shape[:2]
    resize_h, resize_w = det_resize(h, w)

    resized = cv2.resize(img_bgr, (resize_w, resize_h))
    f = resized.astype(np.float32) / 255.0
    # BGR plane order; mean/std apply positionally (plane 0 = B gets 0.485),
    # as the det models were exported with img_mode BGR.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    nchw = ((f - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(nchw), resize_h, resize_w


def _order_quad(pts):
    """Stable sort by x -> [tl,tr,br,bl]. Mirrors order_quad_tl_tr_br_bl."""
    p0, p1, p2, p3 = sorted(pts.tolist(), key=lambda p: p[0])
    tl, bl = (p0, p1) if p0[1] < p1[1] else (p1, p0)
    tr, br = (p2, p3) if p2[1] < p3[1] else (p3, p2)
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _box_score_fast(pred_map, contour):
    h, w = pred_map.shape
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    xmin = max(0, int(xs.min())); xmax = min(w - 1, int(xs.max()))
    ymin = max(0, int(ys.min())); ymax = min(h - 1, int(ys.max()))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= xmin
    shifted[:, 0, 1] -= ymin
    cv2.fillPoly(mask, [shifted], 1)
    roi = pred_map[ymin:ymax + 1, xmin:xmax + 1]
    return float(cv2.mean(roi, mask)[0])


def det_postprocess(pred_map, orig_h, orig_w, resize_h, resize_w):
    """Returns list of boxes (4x2 int, [tl,tr,br,bl]) in original image coords."""
    bitmap = (pred_map > DET_DB_THRESH).astype(np.uint8) * 255
    ratio_h = resize_h / orig_h
    ratio_w = resize_w / orig_w
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours[:DET_MAX_CANDIDATES]:
        if len(c) <= 2:
            continue
        br = cv2.boundingRect(c)
        if br[2] < 3 or br[3] < 3:
            continue
        center, (rw, rh), angle = cv2.minAreaRect(c)
        if min(rw, rh) < MIN_BOX_SIDE:
            continue
        if _box_score_fast(pred_map, c) < DET_DB_BOX_THRESH:
            continue
        # region_to_box: grow the min-area rect by d = r*w*h / (2(w+h)) per side.
        d = DET_DB_UNCLIP_RATIO * rw * rh / (2.0 * (rw + rh)) if rw + rh > 0 else 0.0
        grown = (center, (rw + 2 * d, rh + 2 * d), angle)
        if min(grown[1]) < MIN_UNCLIPPED_SIDE:
            continue
        box = _order_quad(cv2.boxPoints(grown))
        # floor(v + 0.5) == std::round once clamped at 0 (np.round is half-even).
        box[:, 0] = np.clip(np.floor(box[:, 0] / ratio_w + 0.5), 0, orig_w - 1)
        box[:, 1] = np.clip(np.floor(box[:, 1] / ratio_h + 0.5), 0, orig_h - 1)
        b = box.astype(np.int32)
        d01 = (b[0] - b[1]).astype(np.int64)
        d03 = (b[0] - b[3]).astype(np.int64)
        if d01 @ d01 < 16 or d03 @ d03 < 16:
            continue
        boxes.append(b)
    return boxes


# ---------------- rotate crop + rec preprocess ----------------
def get_rotate_crop_image(img_bgr, box):
    """One warp from the page straight to the recognizer height, (content_w x 48).
    Mirrors compute_crop_transform + the warp in preprocess_box."""
    b = box.astype(np.float32)
    crop_w = math.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1])
    crop_h = math.hypot(b[0][0] - b[3][0], b[0][1] - b[3][1])
    vertical = crop_h >= crop_w * VERTICAL_AR
    if vertical:
        # crop_corner(): box corner k lands on crop corner (k+3)%4, i.e. the
        # column is turned anticlockwise (PaddleOCR np.rot90).
        src = np.array([b[1], b[2], b[3], b[0]], dtype=np.float32)
        crop_w, crop_h = crop_h, crop_w
    else:
        src = np.array([b[0], b[1], b[2], b[3]], dtype=np.float32)
    ar = crop_w / crop_h if crop_h > 0 else 0.0
    content_w = max(min(int(math.ceil(REC_IMAGE_H * ar)), REC_MAX_WIDTH), 1)
    dst = np.array([[0, 0], [content_w, 0], [content_w, REC_IMAGE_H], [0, REC_IMAGE_H]],
                   dtype=np.float32)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img_bgr, m_inv, (content_w, REC_IMAGE_H),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def _to_rec_height(crop):
    """Crops from get_rotate_crop_image are already 48 px high; scale others."""
    h, w = crop.shape[:2]
    if h == REC_IMAGE_H:
        return crop
    content_w = max(min(int(math.ceil(REC_IMAGE_H * w / h)), REC_MAX_WIDTH), 1)
    return cv2.resize(crop, (content_w, REC_IMAGE_H))


def rec_target_width(crop):
    """rec_input_width: the natural content width, floored at kMinRecWidth."""
    return max(_to_rec_height(crop).shape[1], REC_MIN_WIDTH)


def rec_preprocess(crop, target_w=None):
    """/127.5-1, BGR->RGB, NCHW, right-padded with 0 (mid-gray) to target_w.
    Mirrors preprocess_box. A target_w narrower than the content (fixed-width
    calibration) squeezes the crop to fit; the server never does that."""
    crop = _to_rec_height(crop)
    if target_w is None:
        target_w = rec_target_width(crop)
    if crop.shape[1] > target_w:
        crop = cv2.resize(crop, (target_w, REC_IMAGE_H))
    f = crop.astype(np.float32) / 127.5 - 1.0
    nchw = np.zeros((1, 3, REC_IMAGE_H, target_w), dtype=np.float32)
    nchw[0, :, :, :f.shape[1]] = f[:, :, ::-1].transpose(2, 0, 1)
    return nchw, target_w


# ---------------- crop extraction driver ----------------
def extract_line_crops(det_sess, pages, max_crops=None, verbose=False):
    """Run det fp32 on pages, return list of BGR line crops (real text lines)."""
    in_name = det_sess.get_inputs()[0].name
    crops = []
    for p in pages:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        inp, rh, rw = det_preprocess(img)
        out = det_sess.run(None, {in_name: inp})[0]
        pred = out[0, 0].astype(np.float32)
        boxes = det_postprocess(pred, h, w, rh, rw)
        crops.extend(get_rotate_crop_image(img, box) for box in boxes)
        if verbose:
            print(f"  {os.path.basename(p)}: {len(boxes)} boxes (total crops {len(crops)})")
        if max_crops and len(crops) >= max_crops:
            break
    return crops[:max_crops] if max_crops else crops


def load_keys(path=f"{REPO}/models/keys.txt"):
    """Returns label_list matching C++ load_label_dict: ['blank', <keys...>, ' ']."""
    labels = ["blank"]
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            labels.append(line)
    labels.append(" ")
    return labels


def ctc_greedy_decode(logits, label_list):
    """Greedy CTC decode. logits: (seq, num_classes). Mirrors ctc_greedy_decode_raw."""
    idx = logits.argmax(axis=1)
    text = []
    last = -1
    for i in idx:
        i = int(i)
        if i != last:
            if i != 0 and i < len(label_list):
                text.append(label_list[i])
        last = i
    return "".join(text)
