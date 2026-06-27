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


def test_download_chip_geotiff4_writes_tif_and_preview(tmp_path, monkeypatch):
    import json

    import numpy as np
    from affine import Affine
    from shapely.geometry import LineString

    from rs_words.data_engine import pc_downloader

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        id = "sentinel-item"
        collection_id = "sentinel-2-l2a"
        datetime = "2024-01-01T00:00:00Z"
        properties = {"eo:cloud_cover": 1}
        assets = {key: Asset(f"/fake/{key}.tif") for key in ["B02", "B03", "B04", "B08"]}

    stack = np.stack(
        [
            np.full((4, 4), 100, dtype=np.uint16),
            np.full((4, 4), 200, dtype=np.uint16),
            np.full((4, 4), 300, dtype=np.uint16),
            np.full((4, 4), 400, dtype=np.uint16),
        ]
    )
    profile = {
        "driver": "GTiff",
        "count": 4,
        "height": 4,
        "width": 4,
        "dtype": "uint16",
        "transform": Affine.identity(),
    }

    monkeypatch.setattr(pc_downloader, "search_sentinel_items", lambda *args, **kwargs: [Item()])
    monkeypatch.setattr(pc_downloader.planetary_computer, "sign", lambda item: item)
    monkeypatch.setattr(pc_downloader, "_read_four_band_stack", lambda assets, bbox: (stack, profile))

    result = pc_downloader.download_chip(
        catalog=object(),
        segment=LineString([(0, 0), (1, 1)]),
        segment_id="seg_1",
        basin="basin",
        output_dir=tmp_path,
        output_format="geotiff4",
    )

    assert result == tmp_path / "basin" / "seg_1.tif"
    assert result.exists()
    assert (tmp_path / "basin" / "seg_1.png").exists()
    meta = json.loads((tmp_path / "basin" / "seg_1.json").read_text())
    assert meta["bands"] == ["B02", "B03", "B04", "B08"]
    assert meta["geotiff_path"].endswith("seg_1.tif")


def test_download_chip_geotiff4_requires_all_sentinel_assets(tmp_path, monkeypatch):
    from shapely.geometry import LineString

    from rs_words.data_engine import pc_downloader

    class Item:
        id = "sentinel-item"
        collection_id = "sentinel-2-l2a"
        datetime = "2024-01-01T00:00:00Z"
        properties = {}
        assets = {}

    monkeypatch.setattr(pc_downloader, "search_sentinel_items", lambda *args, **kwargs: [Item()])
    monkeypatch.setattr(pc_downloader.planetary_computer, "sign", lambda item: item)

    result = pc_downloader.download_chip(
        catalog=object(),
        segment=LineString([(0, 0), (1, 1)]),
        segment_id="seg_missing",
        basin="basin",
        output_dir=tmp_path,
        output_format="geotiff4",
    )

    assert result is None
