import numpy as np

from rs_words.data_engine.patch_bank import Patch, PatchBank
from rs_words.stroke_library import _candidate_patches


def test_candidate_patches_filter_invalid_water_masks():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    good = Patch(
        "good",
        "basin",
        image,
        {
            "water_mask_path": "/tmp/good.png",
            "river_metrics": {
                "water_fraction": 0.1,
                "largest_component_fraction": 0.08,
                "skeleton_length_px": 12,
            },
        },
    )
    bad = Patch(
        "bad",
        "basin",
        image,
        {
            "water_mask_path": "/tmp/bad.png",
            "river_metrics": {
                "water_fraction": 0.0,
                "largest_component_fraction": 0.0,
                "skeleton_length_px": 0,
            },
        },
    )

    candidates = _candidate_patches(PatchBank([bad, good]))

    assert candidates == [good]


def test_candidate_patches_keeps_legacy_bank_when_no_masks():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    patch = Patch("legacy", "basin", image, {})

    assert _candidate_patches(PatchBank([patch])) == [patch]
