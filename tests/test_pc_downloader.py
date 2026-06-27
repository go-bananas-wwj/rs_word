from pathlib import Path

from shapely.geometry import LineString

from rs_words.data_engine.pc_downloader import _buffer_in_degrees, _segment_center


def test_segment_center():
    seg = LineString([(0, 0), (2, 2)])
    lon, lat = _segment_center(seg)
    assert lon == 1.0
    assert lat == 1.0


def test_buffer_in_degrees():
    dlon, dlat = _buffer_in_degrees(0, 0, 1280)
    assert dlon > 0 and dlat > 0
