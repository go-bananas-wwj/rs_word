import numpy as np

from rs_words.data_engine.patch_bank import Patch
from rs_words.glyph import Stroke
from rs_words.matcher import RiverMatcher


def test_matcher_prefers_horizontal():
    h_patch = Patch("h", "yangtze", np.full((64, 64, 3), 200, dtype=np.uint8), {})
    h_patch.image[28:36, 8:56] = [0, 0, 0]
    v_patch = Patch("v", "yangtze", np.full((64, 64, 3), 200, dtype=np.uint8), {})
    v_patch.image[8:56, 28:36] = [0, 0, 0]
    bank = type("B", (), {"patches": [h_patch, v_patch]})()

    h_mask = np.zeros((32, 64), dtype=np.uint8)
    h_mask[14:18, 8:56] = 1
    stroke = Stroke(0, (0, 0, 32, 64), h_mask)
    matcher = RiverMatcher()
    best, _ = matcher.match(stroke, bank, k=1)[0]
    assert best.patch_id == "h"


def test_matcher_prefers_water_mask_over_rgb_edges(tmp_path):
    from PIL import Image

    misleading_rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
    misleading_rgb[8:56, 28:36] = [0, 0, 0]

    h_mask = np.zeros((64, 64), dtype=np.uint8)
    h_mask[28:36, 8:56] = 255
    h_mask_path = tmp_path / "h_mask.png"
    Image.fromarray(h_mask).save(h_mask_path)

    v_mask = np.zeros((64, 64), dtype=np.uint8)
    v_mask[8:56, 28:36] = 255
    v_mask_path = tmp_path / "v_mask.png"
    Image.fromarray(v_mask).save(v_mask_path)

    h_patch = Patch("h", "yangtze", misleading_rgb.copy(), {"water_mask_path": str(h_mask_path)})
    v_patch = Patch("v", "yangtze", misleading_rgb.copy(), {"water_mask_path": str(v_mask_path)})
    bank = type("B", (), {"patches": [v_patch, h_patch]})()

    stroke_mask = np.zeros((32, 64), dtype=np.uint8)
    stroke_mask[14:18, 8:56] = 1
    stroke = Stroke(0, (0, 0, 32, 64), stroke_mask)
    matcher = RiverMatcher()

    best, _ = matcher.match(stroke, bank, k=1)[0]

    assert best.patch_id == "h"
    assert matcher.patch_shape_source(best) == "water_mask"
