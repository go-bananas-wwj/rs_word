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
