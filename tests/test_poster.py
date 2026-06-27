"""Tests for the info poster generator."""

import json
from pathlib import Path

import pytest
from PIL import Image

from rs_words.poster import create_info_poster, INFO_PANEL_WIDTH


@pytest.fixture
def dummy_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a dummy mosaic image and metadata file."""
    mosaic_path = tmp_path / "mosaic.png"
    meta_path = tmp_path / "meta.json"
    output_path = tmp_path / "poster.png"

    mosaic = Image.new("RGB", (200, 150), (128, 128, 128))
    mosaic.save(mosaic_path)

    meta = {
        "text": "测试",
        "strokes": [
            {
                "patch_id": "patch_001",
                "basin": "basin_A",
                "name": "river_one",
                "cloud_cover": 12.5,
            },
        ],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    return mosaic_path, meta_path, output_path


def test_poster(dummy_assets: tuple[Path, Path, Path]) -> None:
    """create_info_poster should produce a correctly sized poster."""
    mosaic_path, meta_path, output_path = dummy_assets

    result = create_info_poster(mosaic_path, meta_path, output_path)

    assert result == output_path
    assert output_path.exists()

    with Image.open(output_path) as poster:
        with Image.open(mosaic_path) as mosaic:
            num_lines = 2 + len(json.loads(meta_path.read_text(encoding="utf-8"))["strokes"])
            text_block_height = num_lines * 30 + 20 * 2
            assert poster.width == mosaic.width + INFO_PANEL_WIDTH
            assert poster.height == max(mosaic.height, text_block_height)
