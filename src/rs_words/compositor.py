from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from rs_words.data_engine.patch_bank import Patch
from rs_words.glyph import Stroke


def _resize_patch(patch_image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a patch image to the target width and height.

    Supports both 2-D (H, W) and 3-D (H, W, C) inputs. The resized output is
    always returned as float32 so callers can blend it without repeated casts.
    """
    if target_w <= 0 or target_h <= 0:
        if patch_image.ndim == 2:
            return np.empty((0, 0), dtype=np.float32)
        return np.empty((0, 0, patch_image.shape[2]), dtype=np.float32)
    resized = cv2.resize(patch_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def _feather_mask(mask: np.ndarray) -> np.ndarray:
    """Compute a normalized distance-transform feather from a binary mask, shape (H,W)."""
    binary = (mask > 0).astype(np.uint8) * 255
    if binary.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_dt = dt.max()
    if max_dt <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    return dt / max_dt


def _stroke_angle(mask: np.ndarray) -> float:
    """Return the principal axis angle of a binary mask in degrees."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 0.0
    cov = np.cov(xs, ys)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]
    return np.degrees(np.arctan2(vec[1], vec[0]))


def _zoom_crop_patch(patch_image: np.ndarray, stroke_mask: np.ndarray, zoom: float = 3.0) -> np.ndarray:
    """Crop a center band around the river and upscale so the river fills more of the stroke.

    The stroke mask's principal angle tells us whether the stroke (and therefore the
    river in the patch) runs roughly horizontally or vertically. We then crop the
    patch perpendicular to that direction and resize back to the original patch size,
    producing a digital zoom that makes the river width occupy a larger share of the tile.
    """
    h, w = patch_image.shape[:2]
    angle = abs(_stroke_angle(stroke_mask))
    if angle > 90:
        angle = 180 - angle

    # Horizontal-ish stroke: the river runs left-right, so crop a vertical band.
    if angle <= 45:
        crop_h = max(int(h / zoom), 1)
        y0 = (h - crop_h) // 2
        cropped = patch_image[y0 : y0 + crop_h, :, ...]
    else:
        # Vertical-ish stroke: the river runs top-bottom, so crop a horizontal band.
        crop_w = max(int(w / zoom), 1)
        x0 = (w - crop_w) // 2
        cropped = patch_image[:, x0 : x0 + crop_w, ...]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


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

    if np.array_equal(source_cdf_norm, template_cdf_norm):
        return np.arange(256, dtype=np.uint8)

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
    """Compose a collage mosaic from stroke-to-patch matches.

    - `text_mask` is used only to determine the output canvas dimensions (H, W).
    - For each (stroke, patch) pair, the patch is first zoom-cropped around the
      river so the river fills more of the tile, then resized to the stroke bbox
      and pasted directly onto a white canvas. No stroke-shaped mask is applied,
      so the resulting mosaic is an arrangement of zoomed satellite image tiles.
    - Clip the canvas to [0, 255] and convert to uint8.
    - If `tone_reference` is provided, resize it to (W, H) and apply
      `match_histograms(canvas, tone_reference)` so the output has a consistent tone.
    """
    if text_mask.ndim != 2:
        raise ValueError("text_mask must be a 2D array")
    h, w = text_mask.shape
    canvas = np.full((h, w, 3), 255.0, dtype=np.float32)

    for stroke, patch in stroke_matches:
        ymin, xmin, ymax, xmax = stroke.bbox
        if ymin >= ymax or xmin >= xmax:
            continue
        target_w = xmax - xmin
        target_h = ymax - ymin

        zoomed_patch = _zoom_crop_patch(patch.image, stroke.mask, zoom=3.0)
        resized_patch = _resize_patch(zoomed_patch, target_w, target_h)
        if resized_patch.size == 0:
            continue
        if resized_patch.ndim == 2:
            resized_patch = np.stack([resized_patch] * 3, axis=-1)

        y0 = max(ymin, 0)
        y1 = min(ymax, h)
        x0 = max(xmin, 0)
        x1 = min(xmax, w)
        if y1 <= y0 or x1 <= x0:
            continue

        # Crop the resized patch to the actual in-bounds ROI when a bbox extends
        # past the canvas edges.
        crop_top = y0 - ymin
        crop_left = x0 - xmin
        crop_bottom = crop_top + (y1 - y0)
        crop_right = crop_left + (x1 - x0)
        resized_patch = resized_patch[crop_top:crop_bottom, crop_left:crop_right]

        canvas[y0:y1, x0:x1] = resized_patch

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    if tone_reference is not None:
        tone_resized = cv2.resize(tone_reference, (w, h), interpolation=cv2.INTER_AREA)
        if tone_resized.ndim == 2:
            tone_resized = np.stack([tone_resized] * 3, axis=-1)
        if tone_resized.shape == canvas.shape:
            canvas = match_histograms(canvas, tone_resized)

    return canvas
