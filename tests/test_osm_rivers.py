from shapely.geometry import LineString

from rs_words.data_engine.osm_rivers import segment_line


def test_segment_line():
    # 创建一个跨 1 度的水平线（约 111 km）
    line = LineString([(0, 0), (1, 0)])
    segs = segment_line(line, 30000)
    assert len(segs) >= 3
    for s in segs:
        assert isinstance(s, LineString)
