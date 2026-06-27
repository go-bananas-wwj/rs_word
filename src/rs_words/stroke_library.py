from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from rs_words.data_engine.patch_bank import Patch, PatchBank
from rs_words.glyph import Stroke
from rs_words.matcher import RiverMatcher


CANVAS_SIZE = 256
STROKE_THICKNESS = 24


@dataclass
class StrokeSpec:
    name: str
    description: str
    draw: Callable[[np.ndarray], None]


def _heng(canvas: np.ndarray) -> None:
    y = CANVAS_SIZE // 2
    cv2.line(canvas, (40, y), (CANVAS_SIZE - 40, y), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _shu(canvas: np.ndarray) -> None:
    x = CANVAS_SIZE // 2
    cv2.line(canvas, (x, 40), (x, CANVAS_SIZE - 40), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _pie(canvas: np.ndarray) -> None:
    cv2.line(canvas, (CANVAS_SIZE - 60, 60), (60, CANVAS_SIZE - 60), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _na(canvas: np.ndarray) -> None:
    cv2.line(canvas, (60, 60), (CANVAS_SIZE - 60, CANVAS_SIZE - 60), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _dian(canvas: np.ndarray) -> None:
    # Small dot / short blotch near the center.
    center = (CANVAS_SIZE // 2, CANVAS_SIZE // 2 + 20)
    axes = (18, 12)
    cv2.ellipse(canvas, center, axes, 30, 0, 360, 255, -1, lineType=cv2.LINE_AA)


def _ti(canvas: np.ndarray) -> None:
    cv2.line(canvas, (50, CANVAS_SIZE - 70), (CANVAS_SIZE - 50, CANVAS_SIZE - 110), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _heng_zhe(canvas: np.ndarray) -> None:
    # Horizontal bar then turn downward (L shape).
    y = CANVAS_SIZE // 3
    cv2.line(canvas, (40, y), (CANVAS_SIZE - 80, y), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)
    cv2.line(canvas, (CANVAS_SIZE - 80, y), (CANVAS_SIZE - 80, CANVAS_SIZE - 40), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _shu_gou(canvas: np.ndarray) -> None:
    # Vertical bar with a small left hook at the bottom.
    x = CANVAS_SIZE // 2
    cv2.line(canvas, (x, 40), (x, CANVAS_SIZE - 70), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x, CANVAS_SIZE - 70), (x - 40, CANVAS_SIZE - 40), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _heng_pie(canvas: np.ndarray) -> None:
    # Short horizontal then left-falling diagonal.
    y = CANVAS_SIZE // 3
    cv2.line(canvas, (60, y), (CANVAS_SIZE // 2 + 20, y), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)
    cv2.line(canvas, (CANVAS_SIZE // 2 + 20, y), (60, CANVAS_SIZE - 40), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


def _shu_wan_gou(canvas: np.ndarray) -> None:
    # Vertical, curve right, hook up.
    x = CANVAS_SIZE // 2 - 30
    cv2.line(canvas, (x, 40), (x, CANVAS_SIZE // 2), 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)
    # Curved rightward segment approximated by an elliptical arc.
    center = (x + 40, CANVAS_SIZE // 2 + 40)
    axes = (40, 50)
    cv2.ellipse(canvas, center, axes, 0, 180, 270, 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)
    # Small upward hook at the end.
    hook_start = (x + 80, CANVAS_SIZE // 2 + 40)
    hook_end = (x + 100, CANVAS_SIZE // 2)
    cv2.line(canvas, hook_start, hook_end, 255, STROKE_THICKNESS, lineType=cv2.LINE_AA)


STROKE_SPECS: List[StrokeSpec] = [
    StrokeSpec("heng", "horizontal bar", _heng),
    StrokeSpec("shu", "vertical bar", _shu),
    StrokeSpec("pie", "left-falling diagonal", _pie),
    StrokeSpec("na", "right-falling diagonal", _na),
    StrokeSpec("dian", "small dot / short blotch", _dian),
    StrokeSpec("ti", "short rising stroke", _ti),
    StrokeSpec("heng-zhe", "horizontal then down (L shape)", _heng_zhe),
    StrokeSpec("shu-gou", "vertical with left hook", _shu_gou),
    StrokeSpec("heng-pie", "horizontal then left-falling", _heng_pie),
    StrokeSpec("shu-wan-gou", "vertical, curve right, hook up", _shu_wan_gou),
]


def _draw_reference_mask(draw_fn: Callable[[np.ndarray], None]) -> np.ndarray:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    draw_fn(canvas)
    return canvas


def _has_water_mask(patch: Patch) -> bool:
    return bool(patch.meta.get("water_mask_path"))


def _is_usable_water_patch(patch: Patch) -> bool:
    if not _has_water_mask(patch):
        return False
    metrics = patch.meta.get("river_metrics") or {}
    water_fraction = float(metrics.get("water_fraction", 0.0) or 0.0)
    largest_fraction = float(metrics.get("largest_component_fraction", 0.0) or 0.0)
    skeleton_length = int(metrics.get("skeleton_length_px", 0) or 0)
    if water_fraction < 0.002 or water_fraction > 0.75:
        return False
    if largest_fraction < 0.001:
        return False
    return skeleton_length >= 8


def _candidate_patches(bank: PatchBank) -> List[Patch]:
    masked = [patch for patch in bank.patches if _has_water_mask(patch)]
    if not masked:
        return bank.patches
    usable = [patch for patch in masked if _is_usable_water_patch(patch)]
    return usable or masked


def _source_image_path(general_bank_dir: Path, patch: Patch) -> Path:
    image_path = patch.meta.get("image_path")
    if image_path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate
        return general_bank_dir.parent / candidate
    return general_bank_dir / f"{patch.patch_id}.png"


def build_stroke_reference_masks(output_dir: Path) -> Dict[str, Path]:
    ref_dir = output_dir / "_references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for spec in STROKE_SPECS:
        mask = _draw_reference_mask(spec.draw)
        path = ref_dir / f"{spec.name}.png"
        Image.fromarray(mask).save(path)
        paths[spec.name] = path
    return paths


def build_stroke_patch_bank(
    general_bank_dir: Path,
    output_dir: Path = Path("/data/rs_word/stroke_patch_bank"),
    top_k: int = 20,
) -> Path:
    """Build a stroke-specific patch library from a general PatchBank."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save reference masks.
    reference_paths = build_stroke_reference_masks(output_dir)

    # Load general patch bank.
    metadata_path = general_bank_dir / "metadata.jsonl"
    bank = PatchBank.load(metadata_path)

    matcher = RiverMatcher()
    summary: Dict[str, Dict] = {}
    candidates = _candidate_patches(bank)

    for spec in STROKE_SPECS:
        stroke_mask = _draw_reference_mask(spec.draw)
        stroke = Stroke(char_index=0, bbox=(0, 0, CANVAS_SIZE, CANVAS_SIZE), mask=stroke_mask)

        scored: List[Tuple[Patch, float]] = []
        seen_ids: set[str] = set()
        for patch in candidates:
            if patch.patch_id in seen_ids:
                continue
            score = matcher.score(stroke, patch)
            scored.append((patch, score))
            seen_ids.add(patch.patch_id)

        scored.sort(key=lambda x: x[1])
        selected = scored[:top_k]

        stroke_out_dir = output_dir / spec.name
        stroke_out_dir.mkdir(parents=True, exist_ok=True)

        for patch, score in selected:
            src_img_path = _source_image_path(general_bank_dir, patch)
            if not src_img_path.exists():
                # Fallback: search within the bank directory.
                image_candidates = list(general_bank_dir.rglob(f"{patch.patch_id}.png"))
                if image_candidates:
                    src_img_path = image_candidates[0]
            dst_img_path = stroke_out_dir / f"{patch.patch_id}.png"
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)

            sidecar = dict(patch.meta)
            sidecar.pop("_data_root", None)
            sidecar["stroke_match_score"] = score
            sidecar["stroke_name"] = spec.name
            sidecar["match_shape_source"] = matcher.patch_shape_source(patch)
            sidecar_path = stroke_out_dir / f"{patch.patch_id}.json"
            sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")

        summary[spec.name] = {
            "description": spec.description,
            "reference_mask_path": str(reference_paths[spec.name]),
            "output_directory": str(stroke_out_dir),
            "saved_patches": len(selected),
            "candidate_patches": len(candidates),
            "masked_candidates": sum(1 for patch in candidates if _has_water_mask(patch)),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_dir
