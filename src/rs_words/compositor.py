from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from rs_words.data_engine.patch_bank import Patch
from rs_words.glyph import Stroke


def _resize_patch(patch_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Resize a patch image to the width/height defined by the stroke bbox (ymin, xmin, ymax, xmax)."""
    ymin, xmin, ymax, xmax = bbox
    target_w = xmax - xmin
    target_h = ymax - ymin
    if target_w <= 0 or target_h <= 0:
        return np.empty((0, 0, patch_image.shape[2]), dtype=patch_image.dtype)
    return cv2.resize(patch_image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _feather_mask(mask: np.ndarray) -> np.ndarray:
    """Compute a normalized distance-transform feather from a binary mask, shape (H,W)."""
    binary = (mask > 0).astype(np.uint8) * 255
    if binary.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    max_dt = dt.max()
    if max_dt <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    return dt / max_dt


def _build_lut(template_channel: np.ndarray, source_channel: np.ndarray) -> np.ndarray:
    """Build a 256-entry LUT for histogram matching from source to template."""
    template_flat = template_channel.ravel()
    source_flat = source_channel.ravel()

    template_hist, _ = np.histogram(template_flat, bins=256, range=(0, 256))
    source_hist, _ = np.histogram(source_flat, bins=256, range=(0, 256))

    template_cdf = template_hist.cumsum()
    source_cdf = source_hist.cumsum()

    t_total = template_cdf[-1]
    s_total = source_cdf[-1]
    if t_total == 0 or s_total == 0:
        return np.arange(256, dtype=np.uint8)

    template_cdf_norm = template_cdf / t_total
    source_cdf_norm = source_cdf / s_total

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        idx = np.searchsorted(template_cdf_norm, source_cdf_norm[i], side="left")
        if idx >= 256:
            idx = 255
        lut[i] = idx
    return lut


def match_histograms(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Match source RGB histograms to template RGB histograms channel-wise."""
    if source.shape != template.shape:
        raise ValueError("source and template must have the same shape")
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source and template must be RGB images with shape (H, W, 3)")

    matched = np.zeros_like(source)
    for c in range(3):
        lut = _build_lut(template[:, :, c], source[:, :, c])
        matched[:, :, c] = cv2.LUT(source[:, :, c], lut)
    return matched


def compose_text(
    text_mask: np.ndarray,
    stroke_matches: List[Tuple[Stroke, Patch]],
    tone_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose the final mosaic.

    - `text_mask` is the full binary text mask, shape (H, W).
    - For each (stroke, patch) pair, place the resized patch into the stroke bbox.
    - Use the stroke mask (resized to bbox size) to generate a feather alpha and
      alpha-blend the patch over a black canvas.
    - Clip the canvas to [0, 255] and convert to uint8.
    - If `tone_reference` is provided, resize it to (W, H) and apply
      `match_histograms(canvas, tone_reference)` so the output has a consistent tone.
    """
    if text_mask.ndim != 2:
        raise ValueError("text_mask must be a 2D array")
    h, w = text_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    for stroke, patch in stroke_matches:
        ymin, xmin, ymax, xmax = stroke.bbox
        if ymin >= ymax or xmin >= xmax:
            continue
        target_w = xmax - xmin
        target_h = ymax - ymin

        resized_patch = _resize_patch(patch.image, stroke.bbox)
        if resized_patch.size == 0:
            continue
        if resized_patch.ndim == 2:
            resized_patch = np.stack([resized_patch] * 3, axis=-1)

        resized_mask = cv2.resize(
            (stroke.mask > 0).astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = _feather_mask(resized_mask)
        alpha = alpha[:, :, np.newaxis]

        roi = canvas[ymin:ymax, xmin:xmax]
        if roi.shape[:2] != alpha.shape[:2] or roi.shape[:2] != resized_patch.shape[:2]:
            continue
        canvas[ymin:ymax, xmin:xmax] = alpha * resized_patch.astype(np.float32) + (1.0 - alpha) * roi

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    if tone_reference is not None:
        tone_resized = cv2.resize(tone_reference, (w, h), interpolation=cv2.INTER_AREA)
        if tone_resized.ndim == 2:
            tone_resized = np.stack([tone_resized] * 3, axis=-1)
        if tone_resized.shape == canvas.shape:
            canvas = match_histograms(canvas, tone_resized)

    return canvas
