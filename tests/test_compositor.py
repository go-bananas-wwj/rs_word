import numpy as np

from rs_words.compositor import (
    _build_lut,
    _feather_mask,
    _resize_patch,
    compose_text,
    match_histograms,
)
from rs_words.data_engine.patch_bank import Patch
from rs_words.glyph import Stroke


def test_resize_patch_basic():
    patch = np.full((64, 64, 3), 255, dtype=np.uint8)
    resized = _resize_patch(patch, (10, 20, 30, 70))
    assert resized.shape == (20, 50, 3)


def test_resize_patch_degenerate():
    patch = np.full((64, 64, 3), 255, dtype=np.uint8)
    resized = _resize_patch(patch, (10, 20, 10, 70))
    assert resized.size == 0


def test_feather_mask_empty():
    mask = np.zeros((32, 32), dtype=np.uint8)
    feather = _feather_mask(mask)
    assert feather.shape == (32, 32)
    assert np.allclose(feather, 0)


def test_feather_mask_inside_higher_than_edge():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    feather = _feather_mask(mask)
    assert feather.shape == (32, 32)
    center = feather[16, 16]
    edge = feather[8, 8]
    assert center > edge


def test_build_lut_identity():
    src = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    lut = _build_lut(src, src)
    expected = np.arange(256, dtype=np.uint8)
    assert np.array_equal(lut, expected)


def test_match_histograms_shape():
    source = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    template = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    matched = match_histograms(source, template)
    assert matched.shape == source.shape
    assert matched.dtype == np.uint8


def test_compose_basic():
    text_mask = np.zeros((100, 200), dtype=np.uint8)
    text_mask[30:70, 50:150] = 255

    patch_image = np.zeros((64, 64, 3), dtype=np.uint8)
    patch_image[:, :] = [200, 50, 50]
    patch = Patch("red", "basin", patch_image, {})

    stroke_mask = np.ones((40, 100), dtype=np.uint8)
    stroke = Stroke(0, (30, 50, 70, 150), stroke_mask)

    output = compose_text(text_mask, [(stroke, patch)])

    assert output.shape == (100, 200, 3)
    assert output.dtype == np.uint8

    inside = output[35:65, 55:145]
    assert np.all(inside[:, :, 0] > 100)
    assert np.all(inside[:, :, 1] < 100)
    assert np.all(inside[:, :, 2] < 100)

    outside = output[0:20, 0:20]
    assert np.all(outside == 0)


def test_compose_with_tone_reference():
    text_mask = np.zeros((100, 200), dtype=np.uint8)
    text_mask[30:70, 50:150] = 255

    patch_image = np.zeros((64, 64, 3), dtype=np.uint8)
    patch_image[:, :] = [200, 50, 50]
    patch = Patch("red", "basin", patch_image, {})

    stroke_mask = np.ones((40, 100), dtype=np.uint8)
    stroke = Stroke(0, (30, 50, 70, 150), stroke_mask)

    tone_reference = np.zeros((50, 50, 3), dtype=np.uint8)
    tone_reference[:, :] = [50, 150, 50]

    output_without = compose_text(text_mask, [(stroke, patch)])
    output_with = compose_text(text_mask, [(stroke, patch)], tone_reference=tone_reference)

    assert output_with.shape == output_without.shape
    assert not np.array_equal(output_with, output_without)
    assert np.mean(output_with[:, :, 1]) > np.mean(output_without[:, :, 1])


def test_compose_degenerate_bbox_skipped():
    text_mask = np.zeros((64, 64), dtype=np.uint8)
    patch = Patch("p", "basin", np.full((32, 32, 3), 128, dtype=np.uint8), {})
    stroke = Stroke(0, (10, 10, 10, 40), np.ones((0, 30), dtype=np.uint8))
    output = compose_text(text_mask, [(stroke, patch)])
    assert output.shape == (64, 64, 3)
    assert np.all(output == 0)
