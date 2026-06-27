from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

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


def _resolve_patch_meta_path(patch: Patch, key: str) -> Path | None:
    value = patch.meta.get(key)
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    data_root = patch.meta.get("_data_root")
    if data_root:
        return Path(data_root) / path
    return path


def _load_patch_water_mask(patch: Patch) -> np.ndarray | None:
    mask_path = _resolve_patch_meta_path(patch, "water_mask_path")
    if not mask_path or not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.sum() == 0:
        return None
    return mask


def _crop_around_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_w: int,
    target_h: int,
    padding: float = 0.35,
) -> np.ndarray:
    h, w = image.shape[:2]
    if target_h <= 0 or target_w <= 0 or h <= 0 or w <= 0:
        return image
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return _aspect_crop_patch(image, target_w, target_h)

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    target_aspect = target_w / max(target_h, 1)

    crop_w = box_w * (1 + padding)
    crop_h = box_h * (1 + padding)
    if crop_w / crop_h < target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect

    crop_w = min(max(int(round(crop_w)), 1), w)
    crop_h = min(max(int(round(crop_h)), 1), h)
    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))
    left = min(max(left, 0), max(w - crop_w, 0))
    top = min(max(top, 0), max(h - crop_h, 0))
    return image[top : top + crop_h, left : left + crop_w, ...]


def _stroke_patch_crop(patch: Patch, target_w: int, target_h: int) -> np.ndarray:
    mask = _load_patch_water_mask(patch)
    if mask is not None:
        return _crop_around_mask(patch.image, mask, target_w, target_h)
    return _aspect_crop_patch(patch.image, target_w, target_h)


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

        cropped_patch = _stroke_patch_crop(patch, target_w, target_h)
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



def _smooth_text_mask(text_mask: np.ndarray, close_radius: int = 5) -> np.ndarray:
    if text_mask.ndim != 2:
        raise ValueError("text_mask must be a 2D array")
    binary = (text_mask > 0).astype(np.uint8) * 255
    if binary.size == 0 or binary.sum() == 0:
        return binary
    radius = max(int(close_radius), 0)
    if radius == 0:
        return binary
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def _best_texture_patch(bank: PatchBank) -> Patch:
    if not bank.patches:
        raise ValueError("patch bank is empty")

    def quality(patch: Patch) -> tuple[float, float]:
        metrics = patch.meta.get("river_metrics") or {}
        water_fraction = float(metrics.get("water_fraction", 0.0) or 0.0)
        skeleton_length = float(metrics.get("skeleton_length_px", 0.0) or 0.0)
        # Prefer visible water without choosing masks that are nearly all water.
        balance = 1.0 - min(abs(water_fraction - 0.18), 0.18) / 0.18 if water_fraction else 0.0
        return (balance, skeleton_length)

    return max(bank.patches, key=quality)


def _cover_resize(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if target_w <= 0 or target_h <= 0 or h <= 0 or w <= 0:
        return np.empty((max(target_h, 0), max(target_w, 0), 3), dtype=np.float32)
    scale = max(target_w / w, target_h / h)
    resized_w = max(int(round(w * scale)), target_w)
    resized_h = max(int(round(h * scale)), target_h)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    x0 = (resized_w - target_w) // 2
    y0 = (resized_h - target_h) // 2
    return resized[y0 : y0 + target_h, x0 : x0 + target_w].astype(np.float32)


def compose_connected_text(
    text_mask: np.ndarray,
    bank: PatchBank,
    close_radius: int = 5,
    background: tuple[int, int, int] = (246, 246, 240),
    outline_color: tuple[int, int, int] = (18, 72, 82),
    outline_alpha: float = 0.18,
) -> np.ndarray:
    """Render the whole glyph as one connected, texture-filled river-word shape.

    This mode intentionally avoids per-stroke tiles. It smooths the glyph mask,
    chooses one high-quality texture patch, covers the full canvas with that texture,
    and clips it through a feathered whole-character mask. The result is closer to
    river-logo typography: connected strokes, one visual tone, and no rectangular seams.
    """
    if text_mask.ndim != 2:
        raise ValueError("text_mask must be a 2D array")
    h, w = text_mask.shape
    canvas = np.full((h, w, 3), np.array(background, dtype=np.float32), dtype=np.float32)
    if h == 0 or w == 0:
        return canvas.astype(np.uint8)
    smooth_mask = _smooth_text_mask(text_mask, close_radius=close_radius)
    if smooth_mask.sum() == 0:
        return canvas.astype(np.uint8)

    patch = _best_texture_patch(bank)
    texture = _cover_resize(patch.image, w, h)
    if texture.ndim == 2:
        texture = np.stack([texture] * 3, axis=-1)

    # Slightly cool the texture so it reads more like water typography than raw tiles.
    tint = np.array([0.86, 1.02, 1.08], dtype=np.float32)
    texture = np.clip(texture * tint, 0, 255)

    alpha = _feather_mask(smooth_mask)
    alpha = np.clip(alpha * 1.25, 0, 1)[..., None]
    canvas = canvas * (1 - alpha) + texture * alpha

    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(outline, contours, -1, 255, thickness=max(close_radius // 2, 1), lineType=cv2.LINE_AA)
    outline_a = (outline.astype(np.float32) / 255.0 * outline_alpha)[..., None]
    outline_rgb = np.array(outline_color, dtype=np.float32)
    canvas = canvas * (1 - outline_a) + outline_rgb * outline_a

    return np.clip(canvas, 0, 255).astype(np.uint8)


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
