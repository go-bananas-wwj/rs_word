from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import LineString, mapping

from rs_words.config import (
    CHIP_SIZE,
    CHIP_SIZE_METERS,
    PC_COLLECTION_LANDSAT,
    PC_COLLECTION_SENTINEL,
    PC_STAC_URL,
    SATELLITE_DIR,
)


def _segment_center(seg: LineString) -> tuple:
    return seg.centroid.x, seg.centroid.y


def _buffer_in_degrees(lon: float, lat: float, meters: float) -> float:
    """基于纬度的粗略米转度（用于 Sentinel-2 10m 近似搜索）。"""
    lat_rad = np.radians(lat)
    delta_lon = meters / (111320 * np.cos(lat_rad))
    delta_lat = meters / 110540
    return delta_lon, delta_lat


def search_sentinel_items(
    catalog: pystac_client.Client,
    bbox: tuple,
    datetime_range: str = "2022-01-01/2024-12-31",
    cloud_cover_lt: int = 20,
    max_items: int = 10,
):
    search = catalog.search(
        collections=[PC_COLLECTION_SENTINEL],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
        max_items=max_items,
    )
    return search.item_collection()


def download_chip(
    catalog: pystac_client.Client,
    segment: LineString,
    segment_id: str,
    basin: str,
    output_dir: Path = SATELLITE_DIR,
    datetime_range: str = "2022-01-01/2024-12-31",
    chip_size_meters: float = CHIP_SIZE_METERS,
    fallback_collection: Optional[str] = PC_COLLECTION_LANDSAT,
) -> Optional[Path]:
    """为一条河流段下载以其中点为中心的 RGB 卫星切片。"""
    output_dir = output_dir / basin
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / f"{segment_id}.png"
    meta_path = output_dir / f"{segment_id}.json"

    if img_path.exists() and meta_path.exists():
        return img_path

    lon, lat = _segment_center(segment)
    dlon, dlat = _buffer_in_degrees(lon, lat, chip_size_meters / 2)
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    items = search_sentinel_items(catalog, bbox, datetime_range=datetime_range)
    if not items and fallback_collection:
        search = catalog.search(
            collections=[fallback_collection],
            bbox=bbox,
            datetime=datetime_range,
            max_items=5,
        )
        items = search.item_collection()
    if not items:
        return None

    item = items[0]
    signed = planetary_computer.sign(item)
    asset = signed.assets.get("visual") or signed.assets.get("rendered_preview")
    if asset is None:
        return None

    with rasterio.open(asset.href) as src:
        win = from_bounds(*bbox, src.transform)
        data = src.read(window=win)
        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0))
        else:
            rgb = np.squeeze(data)
        rgb = np.clip(rgb, 0, None)
        if rgb.max() > 0:
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    from PIL import Image

    Image.fromarray(rgb).save(img_path)

    meta = {
        "segment_id": segment_id,
        "basin": basin,
        "item_id": item.id,
        "collection": item.collection_id,
        "datetime": str(item.datetime),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "bbox": bbox,
        "asset_href": asset.href,
        "segment_geometry": mapping(segment),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return img_path
