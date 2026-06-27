import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

from rs_words.water_masks import build_water_masks, generate_water_mask, river_metrics, water_mask_ndwi


def _write_four_band_tif(path: Path) -> None:
    stack = np.zeros((4, 16, 16), dtype=np.uint16)
    stack[1] = 40
    stack[3] = 220
    stack[1, 4:12, 3:13] = 220
    stack[3, 4:12, 3:13] = 40
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=4,
        height=16,
        width=16,
        dtype="uint16",
        transform=Affine.identity(),
    ) as dst:
        dst.write(stack)


def test_ndwi_backend_detects_high_green_low_nir_water():
    stack = np.zeros((4, 16, 16), dtype=np.float32)
    stack[1] = 40
    stack[3] = 220
    stack[1, 4:12, 3:13] = 220
    stack[3, 4:12, 3:13] = 40

    mask = water_mask_ndwi(stack, threshold=0.0)

    assert mask[6, 6] == 255
    assert mask[0, 0] == 0


def test_generate_water_mask_writes_mask_and_metrics(tmp_path: Path):
    tif_path = tmp_path / "chip.tif"
    mask_path = tmp_path / "mask.png"
    _write_four_band_tif(tif_path)

    out_path, metrics = generate_water_mask(tif_path, mask_path, backend="ndwi")

    assert out_path == mask_path
    assert mask_path.exists()
    assert metrics["mask_backend"] == "ndwi"
    assert metrics["water_fraction"] > 0
    assert metrics["skeleton_length_px"] > 0


def test_river_metrics_empty_mask():
    metrics = river_metrics(np.zeros((8, 8), dtype=np.uint8))

    assert metrics["water_fraction"] == 0.0
    assert metrics["component_count"] == 0


def test_build_water_masks_updates_patch_bank_metadata(tmp_path: Path):
    data_root = tmp_path
    patch_bank = data_root / "patch_bank"
    raw = data_root / "satellite_chips" / "raw" / "basin"
    patch_bank.mkdir()
    raw.mkdir(parents=True)
    tif_path = raw / "seg_1.tif"
    _write_four_band_tif(tif_path)
    Image.new("RGB", (16, 16)).save(patch_bank / "preview.png")
    metadata_path = patch_bank / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "patch_id": "seg_1",
                "basin": "basin",
                "image_path": "patch_bank/preview.png",
                "geotiff_path": "satellite_chips/raw/basin/seg_1.tif",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    generated = build_water_masks(
        metadata_path=metadata_path,
        output_dir=data_root / "water_masks",
        backend="ndwi",
    )

    assert generated == 1
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["water_mask_path"] == "water_masks/basin/seg_1.png"
    assert meta["mask_backend"] == "ndwi"
    assert meta["river_metrics"]["water_fraction"] > 0
