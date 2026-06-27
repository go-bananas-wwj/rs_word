from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from rs_words.data_engine.patch_bank import Patch, PatchBank
from rs_words.glyph import Stroke
from rs_words.matcher import RiverMatcher


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


def _aspect_crop_patch(
    patch_image: np.ndarray, target_w: int, target_h: int, zoom: float = 2.0
) -> np.ndarray:
    """Crop a region from the patch that matches the stroke bbox aspect ratio.

    The patch is square (256x256), but strokes can be long horizontal or tall
    vertical. Instead of stretching the whole square patch to fit the stroke,
    we crop a centered band whose aspect ratio matches the stroke, optionally
    zooming in so the river fills more of the tile, and then resize it to the
    stroke bbox. This preserves the natural proportions of the river imagery
    while making the river clearly visible.
    """
    h, w = patch_image.shape[:2]
    if target_h <= 0 or target_w <= 0:
        return patch_image
    target_aspect = target_w / target_h
    patch_aspect = w / h

    if target_aspect >= patch_aspect:
        # Stroke is wider than the patch: crop a horizontal band (full width, less height).
        crop_h = max(int(w / target_aspect / zoom), 1)
        y0 = (h - crop_h) // 2
        cropped = patch_image[y0 : y0 + crop_h, :, ...]
    else:
        # Stroke is taller than the patch: crop a vertical band (less width, full height).
        crop_w = max(int(h * target_aspect / zoom), 1)
        x0 = (w - crop_w) // 2
        cropped = patch_image[:, x0 : x0 + crop_w, ...]

    return cropped


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
    """Compose a stroke-level collage mosaic from stroke-to-patch matches.

    - `text_mask` is used only to determine the output canvas dimensions (H, W).
    - For each (stroke, patch) pair, a centered region matching the stroke's
      aspect ratio is cropped from the square patch, resized to the stroke bbox
      without distortion, and pasted directly onto a white canvas. No stroke-shaped
      mask is applied, so the result is an arrangement of proportionally-cropped
      satellite image tiles.
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

        cropped_patch = _aspect_crop_patch(patch.image, target_w, target_h)
        resized_patch = _resize_patch(cropped_patch, target_w, target_h)
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


def compose_grid(
    text_mask: np.ndarray,
    bank: PatchBank,
    tile_size: int = 128,
    min_ink_ratio: float = 0.15,
    tone_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compose a grid mosaic where each tile is a satellite patch.

    The text mask is divided into a grid of `tile_size` cells. For each cell that
    contains enough text ink, a patch whose shape best matches the local ink is
    selected and pasted into the cell. This produces a denser, more recognizable
    mosaic than stroke-level collage, especially for characters whose strokes are
    merged by skeletonization.
    """
    if text_mask.ndim != 2:
        raise ValueError("text_mask must be a 2D array")
    h, w = text_mask.shape
    canvas = np.full((h, w, 3), 255.0, dtype=np.float32)
    matcher = RiverMatcher()

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            ymax = min(y + tile_size, h)
            xmax = min(x + tile_size, w)
            if ymax <= y or xmax <= x:
                continue
            cell_mask = text_mask[y:ymax, x:xmax]
            cell_area = (ymax - y) * (xmax - x)
            if cell_area == 0 or (cell_mask.sum() / cell_area) < min_ink_ratio:
                continue
            stroke = Stroke(char_index=0, bbox=(y, x, ymax, xmax), mask=cell_mask)
            top_k = matcher.match(stroke, bank, k=5)
            if not top_k:
                continue
            best_patch, _ = top_k[0]
            cropped = _aspect_crop_patch(best_patch.image, xmax - x, ymax - y)
            resized = _resize_patch(cropped, xmax - x, ymax - y)
            if resized.size == 0:
                continue
            if resized.ndim == 2:
                resized = np.stack([resized] * 3, axis=-1)
            canvas[y:ymax, x:xmax] = resized

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    if tone_reference is not None:
        tone_resized = cv2.resize(tone_reference, (w, h), interpolation=cv2.INTER_AREA)
        if tone_resized.ndim == 2:
            tone_resized = np.stack([tone_resized] * 3, axis=-1)
        if tone_resized.shape == canvas.shape:
            canvas = match_histograms(canvas, tone_resized)

    return canvas
