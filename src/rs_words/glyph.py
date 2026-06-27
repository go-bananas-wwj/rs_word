from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


@dataclass
class Stroke:
    char_index: int
    bbox: Tuple[int, int, int, int]  # ymin, xmin, ymax, xmax
    mask: np.ndarray  # uint8 二值掩码，形状与 bbox 一致


def render_text(
    text: str,
    font_path: Optional[Path] = None,
    font_size: int = 256,
    padding: int = 20,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    font_path = font_path or ""
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + padding * 2
    height = bbox[3] - bbox[1] + padding * 2

    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, fill=255, font=font)

    char_bboxes = []
    x_offset = padding - bbox[0]
    y_offset = padding - bbox[1]
    for ch in text:
        cb = draw.textbbox((0, 0), ch, font=font)
        w = cb[2] - cb[0]
        char_bboxes.append((y_offset + cb[1], x_offset + cb[0], y_offset + cb[3], x_offset + cb[2]))
        x_offset += w

    return np.array(img, dtype=np.uint8), char_bboxes


def decompose_mask(mask: np.ndarray, char_index: int = 0, min_area: int = 50) -> List[Stroke]:
    # Support both grayscale masks (0-255) and already-binary masks (0/1).
    if mask.max() <= 1:
        binary = mask.astype(np.uint8)
    else:
        binary = (mask > 127).astype(np.uint8)
    skel = skeletonize(binary).astype(np.uint8)
    labeled = label(skel, connectivity=2)
    strokes = []
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        ymin, xmin, ymax, xmax = region.bbox
        crop = (labeled[ymin:ymax, xmin:xmax] == region.label).astype(np.uint8)
        strokes.append(Stroke(char_index=char_index, bbox=(ymin, xmin, ymax, xmax), mask=crop))
    return strokes


def decompose_text(
    text: str,
    font_path: Optional[Path] = None,
    font_size: int = 256,
    min_area: int = 50,
) -> Tuple[np.ndarray, List[Stroke]]:
    gray, char_bboxes = render_text(text, font_path=font_path, font_size=font_size)
    full_mask = (gray > 127).astype(np.uint8)
    all_strokes = []
    for idx, (ymin, xmin, ymax, xmax) in enumerate(char_bboxes):
        char_mask = full_mask[ymin:ymax, xmin:xmax]
        strokes = decompose_mask(char_mask, char_index=idx, min_area=min_area)
        for s in strokes:
            all_strokes.append(
                Stroke(
                    char_index=idx,
                    bbox=(ymin + s.bbox[0], xmin + s.bbox[1], ymin + s.bbox[2], xmin + s.bbox[3]),
                    mask=s.mask,
                )
            )
    return full_mask, all_strokes
