from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rasterio
from PIL import Image
from skimage import measure, morphology

from rs_words.config import DATA_DIR, PATCH_BANK_DIR

MaskBackend = Literal["ndwi", "omniwatermask"]


def resolve_data_path(data_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path


def relative_to_data_root(path: Path, data_root: Path) -> str:
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def read_four_band_geotiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"Expected at least 4 bands in {path}, found {src.count}")
        return src.read([1, 2, 3, 4]).astype(np.float32)


def water_mask_ndwi(stack: np.ndarray, threshold: float = 0.0, min_size: int = 48) -> np.ndarray:
    if stack.shape[0] < 4:
        raise ValueError("NDWI backend expects B02/B03/B04/B08 stack")
    green = stack[1]
    nir = stack[3]
    ndwi = (green - nir) / (green + nir + 1e-6)
    mask = ndwi > threshold
    mask = morphology.remove_small_objects(mask, min_size)
    mask = morphology.remove_small_holes(mask, min_size)
    return (mask.astype(np.uint8) * 255)


def water_mask_omniwatermask(geotiff_path: Path, output_path: Path) -> np.ndarray:
    command = os.environ.get("OMNIWATERMASK_COMMAND")
    if command:
        subprocess.run(
            [command, str(geotiff_path), str(output_path)],
            check=True,
        )
        return np.array(Image.open(output_path).convert("L"), dtype=np.uint8)

    try:
        import omniwatermask  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OmniWaterMask is not installed in the rs_words Conda environment. "
            "Install it there or set OMNIWATERMASK_COMMAND to a CLI wrapper; "
            "use --backend ndwi to run the built-in fallback."
        ) from exc

    if hasattr(omniwatermask, "predict"):
        result = omniwatermask.predict(str(geotiff_path))
    elif hasattr(omniwatermask, "OmniWaterMask"):
        result = omniwatermask.OmniWaterMask().predict(str(geotiff_path))
    else:
        raise RuntimeError(
            "Installed omniwatermask package does not expose a recognized predict API. "
            "Set OMNIWATERMASK_COMMAND to a wrapper that accepts input GeoTIFF and output mask path."
        )

    mask = np.asarray(result)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return ((mask > 0).astype(np.uint8) * 255)


def save_mask(mask: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(output_path)
    return output_path


def _principal_angle(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return None
    cov = np.cov(xs, ys)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180
    return angle


def _skeleton_branch_points(skeleton: np.ndarray) -> int:
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) - skeleton.astype(np.uint8)
    return int(((skeleton > 0) & (neighbors > 2)).sum())


def river_metrics(mask: np.ndarray) -> dict:
    binary = mask > 0
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    if not props:
        return {
            "metrics_backend": "skimage",
            "water_fraction": 0.0,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "skeleton_length_px": 0,
            "branch_points": 0,
        }

    largest = max(props, key=lambda p: p.area)
    min_row, min_col, max_row, max_col = largest.bbox
    height = max(max_row - min_row, 1)
    width = max(max_col - min_col, 1)
    skeleton = morphology.skeletonize(binary).astype(np.uint8)
    skeleton_length = int(skeleton.sum())
    branch_points = _skeleton_branch_points(skeleton) if skeleton_length else 0

    return {
        "metrics_backend": "skimage",
        "water_fraction": float(binary.mean()),
        "component_count": int(len(props)),
        "largest_component_fraction": float(largest.area / binary.size),
        "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)],
        "bbox_aspect_ratio": float(width / height),
        "orientation_degrees": _principal_angle(labels == largest.label),
        "skeleton_length_px": skeleton_length,
        "branch_points": branch_points,
        "branch_density": float(branch_points / max(skeleton_length, 1)),
    }


def generate_water_mask(
    geotiff_path: Path,
    output_path: Path,
    backend: MaskBackend = "ndwi",
    ndwi_threshold: float = 0.0,
) -> tuple[Path, dict]:
    if backend == "ndwi":
        stack = read_four_band_geotiff(geotiff_path)
        mask = water_mask_ndwi(stack, threshold=ndwi_threshold)
    elif backend == "omniwatermask":
        mask = water_mask_omniwatermask(geotiff_path, output_path)
    else:
        raise ValueError(f"Unsupported water mask backend: {backend}")

    save_mask(mask, output_path)
    metrics = river_metrics(mask)
    metrics["mask_backend"] = backend
    return output_path, metrics


def build_water_masks(
    metadata_path: Path = PATCH_BANK_DIR / "metadata.jsonl",
    output_dir: Path = DATA_DIR / "water_masks",
    backend: MaskBackend = "ndwi",
    ndwi_threshold: float = 0.0,
    overwrite: bool = False,
) -> int:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Patch bank metadata not found: {metadata_path}")

    data_root = metadata_path.parent.parent
    output_dir = Path(output_dir)
    updated = []
    generated = 0

    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        meta = json.loads(line)
        geotiff_path = resolve_data_path(data_root, meta.get("geotiff_path"))
        if geotiff_path is None or not geotiff_path.exists():
            updated.append(meta)
            continue

        basin = meta.get("basin", "unknown")
        patch_id = meta.get("patch_id") or geotiff_path.stem
        mask_path = output_dir / basin / f"{patch_id}.png"
        if overwrite or not mask_path.exists():
            _, metrics = generate_water_mask(
                geotiff_path,
                mask_path,
                backend=backend,
                ndwi_threshold=ndwi_threshold,
            )
            generated += 1
        else:
            metrics = river_metrics(np.array(Image.open(mask_path).convert("L"), dtype=np.uint8))
            metrics["mask_backend"] = backend

        meta["water_mask_path"] = relative_to_data_root(mask_path, data_root)
        meta["mask_backend"] = backend
        meta["river_metrics"] = metrics
        updated.append(meta)

    tmp_path = metadata_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text(
        "\n".join(json.dumps(meta, ensure_ascii=False) for meta in updated) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(metadata_path)
    return generated
