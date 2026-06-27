from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from PIL import Image
from pystac import ItemCollection
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import LineString, mapping

from rs_words.config import (
    CHIP_SIZE_METERS,
    PC_ASSET_KEYS,
    PC_CLOUD_COVER_LT,
    PC_COLLECTION_LANDSAT,
    PC_COLLECTION_SENTINEL,
    PC_DATETIME_RANGE,
    PC_FALLBACK_MAX_ITEMS,
    PC_SENTINEL_4BAND_ASSET_KEYS,
    SATELLITE_DIR,
)

logger = logging.getLogger(__name__)

# Approximate meters per degree at the equator
METERS_PER_DEGREE_LON_AT_EQUATOR = 111320.0
METERS_PER_DEGREE_LAT = 110540.0
DownloadFormat = Literal["rgb", "geotiff4"]


def _segment_center(seg: LineString) -> tuple[float, float]:
    return seg.centroid.x, seg.centroid.y


def _buffer_in_degrees(lon: float, lat: float, meters: float) -> tuple[float, float]:
    """基于纬度的粗略米转度（用于 Sentinel-2 10m 近似搜索）。"""
    lat_rad = np.radians(lat)
    delta_lon = meters / (METERS_PER_DEGREE_LON_AT_EQUATOR * np.cos(lat_rad))
    delta_lat = meters / METERS_PER_DEGREE_LAT
    return delta_lon, delta_lat


def _asset_by_key(assets: dict, key: str):
    return assets.get(key) or assets.get(key.lower())


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    rgb = np.clip(rgb, 0, None).astype(np.float32)
    if rgb.size == 0:
        return rgb.astype(np.uint8)
    hi = float(np.nanpercentile(rgb, 98))
    if hi <= 0:
        hi = float(np.nanmax(rgb))
    if hi > 0:
        rgb = np.clip(rgb / hi * 255, 0, 255)
    return rgb.astype(np.uint8)


def _relative_to_data_root(path: Path, output_dir: Path) -> str:
    if output_dir.parent.name == "raw" and output_dir.parent.parent.name == "satellite_chips":
        root = output_dir.parent.parent.parent
    else:
        root = output_dir.parent
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def search_sentinel_items(
    catalog: pystac_client.Client,
    bbox: tuple[float, float, float, float],
    datetime_range: str = PC_DATETIME_RANGE,
    cloud_cover_lt: int = PC_CLOUD_COVER_LT,
    max_items: int = 10,
) -> ItemCollection:
    search = catalog.search(
        collections=[PC_COLLECTION_SENTINEL],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
        max_items=max_items,
    )
    return search.item_collection()


def _read_visual_rgb(asset, bbox: tuple[float, float, float, float]) -> np.ndarray:
    with rasterio.open(asset.href) as src:
        src_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*src_bbox, src.transform)
        data = src.read(window=win)

    if data.size == 0:
        return data
    if data.shape[0] >= 3:
        rgb = np.transpose(data[:3], (1, 2, 0))
    else:
        rgb = np.squeeze(data)
    return _to_uint8_rgb(rgb)


def _read_four_band_stack(
    assets: dict,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict] | None:
    bands = []
    profile: dict | None = None
    transform = None

    for key in PC_SENTINEL_4BAND_ASSET_KEYS:
        asset = _asset_by_key(assets, key)
        if asset is None:
            logger.warning("Sentinel asset %s is missing; cannot build four-band chip", key)
            return None
        with rasterio.open(asset.href) as src:
            src_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*src_bbox, src.transform)
            band = src.read(1, window=win)
            if band.size == 0:
                return None
            bands.append(band)
            if profile is None:
                transform = src.window_transform(win)
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=len(PC_SENTINEL_4BAND_ASSET_KEYS),
                    height=band.shape[0],
                    width=band.shape[1],
                    transform=transform,
                    dtype=band.dtype,
                )

    shapes = {band.shape for band in bands}
    if len(shapes) != 1:
        logger.warning("Four-band Sentinel assets have mismatched shapes: %s", shapes)
        return None
    return np.stack(bands, axis=0), profile or {}


def _save_rgb_preview_from_four_band(stack: np.ndarray, img_path: Path) -> None:
    # Stack order is B02, B03, B04, B08; RGB preview is B04/B03/B02.
    rgb = np.transpose(stack[[2, 1, 0]], (1, 2, 0))
    Image.fromarray(_to_uint8_rgb(rgb)).save(img_path)


def download_chip(
    catalog: pystac_client.Client,
    segment: LineString,
    segment_id: str,
    basin: str,
    output_dir: Path = SATELLITE_DIR,
    datetime_range: str = PC_DATETIME_RANGE,
    chip_size_meters: float = CHIP_SIZE_METERS,
    fallback_collection: Optional[str] = PC_COLLECTION_LANDSAT,
    output_format: DownloadFormat = "rgb",
    save_rgb_preview: bool = True,
) -> Optional[Path]:
    """为一条河流段下载卫星切片。

    ``output_format="rgb"`` keeps the historical PNG behavior.
    ``output_format="geotiff4"`` stores Sentinel-2 B02/B03/B04/B08 as a
    four-band GeoTIFF and writes an RGB PNG preview when requested.
    """
    if output_format not in {"rgb", "geotiff4"}:
        raise ValueError(f"Unsupported output_format: {output_format}")

    output_dir = output_dir / basin
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / f"{segment_id}.png"
    tif_path = output_dir / f"{segment_id}.tif"
    meta_path = output_dir / f"{segment_id}.json"
    primary_path = tif_path if output_format == "geotiff4" else img_path

    preview_ready = not save_rgb_preview or img_path.exists()
    if primary_path.exists() and meta_path.exists() and preview_ready:
        return primary_path

    lon, lat = _segment_center(segment)
    dlon, dlat = _buffer_in_degrees(lon, lat, chip_size_meters / 2)
    bbox: tuple[float, float, float, float] = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    try:
        items = search_sentinel_items(catalog, bbox, datetime_range=datetime_range)
    except Exception as exc:
        logger.warning("STAC search failed for %s: %s", segment_id, exc)
        return None

    if not items and fallback_collection and output_format == "rgb":
        try:
            search = catalog.search(
                collections=[fallback_collection],
                bbox=bbox,
                datetime=datetime_range,
                max_items=PC_FALLBACK_MAX_ITEMS,
            )
            items = search.item_collection()
        except Exception as exc:
            logger.warning("Fallback STAC search failed for %s: %s", segment_id, exc)
            return None

    if not items:
        return None

    item = items[0]

    try:
        signed = planetary_computer.sign(item)
    except Exception as exc:
        logger.warning("Failed to sign item for %s: %s", segment_id, exc)
        return None

    asset = None
    for key in PC_ASSET_KEYS:
        asset = _asset_by_key(signed.assets, key)
        if asset is not None:
            break

    meta = {
        "segment_id": segment_id,
        "basin": basin,
        "item_id": item.id,
        "collection": item.collection_id,
        "datetime": str(item.datetime),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "bbox": bbox,
        "segment_geometry": mapping(segment),
    }

    if output_format == "geotiff4":
        if item.collection_id != PC_COLLECTION_SENTINEL:
            logger.warning("Four-band chips are only supported for Sentinel-2 items: %s", item.id)
            return None
        stack_profile = _read_four_band_stack(signed.assets, bbox)
        if stack_profile is None:
            return None
        stack, profile = stack_profile
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(stack)
        if save_rgb_preview:
            _save_rgb_preview_from_four_band(stack, img_path)
        meta.update(
            {
                "asset_hrefs": {
                    key: _asset_by_key(signed.assets, key).href
                    for key in PC_SENTINEL_4BAND_ASSET_KEYS
                },
                "geotiff_path": _relative_to_data_root(tif_path, output_dir),
                "image_path": _relative_to_data_root(img_path, output_dir) if img_path.exists() else None,
                "bands": PC_SENTINEL_4BAND_ASSET_KEYS,
            }
        )
    else:
        if asset is None:
            return None
        try:
            rgb = _read_visual_rgb(asset, bbox)
        except Exception as exc:
            logger.warning("Failed to read raster for %s: %s", segment_id, exc)
            return None

        if rgb.size == 0:
            logger.warning("Empty raster read for %s", segment_id)
            return None

        Image.fromarray(rgb).save(img_path)
        meta.update(
            {
                "asset_href": asset.href,
                "image_path": _relative_to_data_root(img_path, output_dir),
            }
        )

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return primary_path
