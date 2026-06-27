from rs_words.glyph import decompose_text, render_text


def test_render_ascii():
    gray, bboxes = render_text("AB", font_size=64)
    assert gray.max() > 0
    assert len(bboxes) == 2


def test_decompose_text():
    mask, strokes = decompose_text("AB", font_size=64, min_area=5)
    assert len(strokes) >= 2
