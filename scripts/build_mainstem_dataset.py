"""Download 2025 summer/autumn Sentinel-2 chips along Yangtze and Yellow mainstems."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from PIL import Image
from pyproj import Geod
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rs_words.config import PC_COLLECTION_SENTINEL, PC_SENTINEL_4BAND_ASSET_KEYS, PC_STAC_URL
from rs_words.data_engine.pc_downloader import _buffer_in_degrees, _to_uint8_rgb

logger = logging.getLogger(__name__)
GEOD = Geod(ellps="WGS84")

# Approximate mainstem control points, downstream order. These are used to make
# reproducible along-river sampling independent of fragile full-basin Overpass calls.
RIVER_WAYPOINTS = {
    "yangtze": [
        (91.15, 33.10), (95.60, 32.95), (97.20, 32.65), (99.60, 31.80),
        (102.25, 29.95), (104.65, 28.75), (106.55, 29.55), (108.40, 30.80),
        (111.30, 30.70), (114.30, 30.55), (116.35, 29.85), (118.75, 31.95),
        (120.20, 32.05), (121.50, 31.35),
    ],
    "yellow": [
        (95.90, 35.05), (98.20, 35.90), (101.75, 36.55), (103.85, 36.05),
        (106.35, 37.45), (108.95, 39.20), (110.45, 37.65), (111.15, 35.70),
        (112.70, 34.85), (114.35, 34.90), (116.10, 35.95), (118.65, 37.70),
        (119.15, 37.75),
    ],
}

SEASON_WINDOWS = {
    "summer": "2025-06-01/2025-08-31",
    "autumn": "2025-09-01/2025-11-30",
}


@dataclass
class SamplePoint:
    river: str
    index: int
    lon: float
    lat: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/rs_word_mainstem"))
    parser.add_argument("--rivers", default="yangtze,yellow")
    parser.add_argument("--seasons", default="summer,autumn")
    parser.add_argument("--spacing-km", type=float, default=5.0)
    parser.add_argument("--chip-size-meters", type=float, default=2560.0)
    parser.add_argument("--max-per-river-season", type=int, default=None)
    parser.add_argument("--cloud-cover", type=float, default=10.0)
    parser.add_argument("--fallback-cloud-cover", type=float, default=20.0)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_list(value: str, allowed: Iterable[str], label: str) -> list[str]:
    requested = [v.strip() for v in value.split(",") if v.strip()]
    allowed_set = set(allowed)
    unknown = sorted(set(requested) - allowed_set)
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return requested


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    _, _, dist = GEOD.inv(a[0], a[1], b[0], b[1])
    return float(dist)


def sample_mainstem(river: str, spacing_km: float) -> list[SamplePoint]:
    points = RIVER_WAYPOINTS[river]
    spacing_m = spacing_km * 1000.0
    samples: list[SamplePoint] = []
    carry = 0.0
    index = 0
    for start, end in zip(points[:-1], points[1:]):
        seg_len = _distance_m(start, end)
        if seg_len <= 0:
            continue
        distance = 0.0 if not samples else spacing_m - carry
        while distance <= seg_len:
            frac = distance / seg_len
            lon = start[0] + (end[0] - start[0]) * frac
            lat = start[1] + (end[1] - start[1]) * frac
            samples.append(SamplePoint(river=river, index=index, lon=lon, lat=lat))
            index += 1
            distance += spacing_m
        carry = seg_len - (distance - spacing_m)
        if math.isclose(carry, spacing_m):
            carry = 0.0
    return samples


def bbox_for_point(lon: float, lat: float, chip_size_meters: float) -> tuple[float, float, float, float]:
    dlon, dlat = _buffer_in_degrees(lon, lat, chip_size_meters / 2)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _asset_by_key(assets: dict, key: str):
    return assets.get(key) or assets.get(key.lower())


def search_items(catalog, bbox, season: str, cloud_cover: float, max_items: int):
    search = catalog.search(
        collections=[PC_COLLECTION_SENTINEL],
        bbox=bbox,
        datetime=SEASON_WINDOWS[season],
        query={"eo:cloud_cover": {"lt": cloud_cover}},
        max_items=max_items,
    )
    items = list(search.items())
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover") or 999)
    return items


def read_four_band(item, bbox):
    signed = planetary_computer.sign(item)
    bands = []
    profile = None
    for key in PC_SENTINEL_4BAND_ASSET_KEYS:
        asset = _asset_by_key(signed.assets, key)
        if asset is None:
            return None
        with rasterio.open(asset.href) as src:
            src_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*src_bbox, src.transform)
            band = src.read(1, window=win)
            if band.size == 0:
                return None
            bands.append(band)
            if profile is None:
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=4,
                    height=band.shape[0],
                    width=band.shape[1],
                    transform=src.window_transform(win),
                    dtype=band.dtype,
                )
    if len({b.shape for b in bands}) != 1:
        return None
    return np.stack(bands), profile, signed


def save_preview(stack: np.ndarray, path: Path) -> None:
    rgb = np.transpose(stack[[2, 1, 0]], (1, 2, 0))
    Image.fromarray(_to_uint8_rgb(rgb)).save(path)


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_one(catalog, point: SamplePoint, season: str, args, manifest_path: Path, failed_path: Path) -> bool:
    segment_id = f"{point.river}_{point.index:05d}_{season}_2025"
    raw_dir = args.output_root / "raw" / point.river / season
    preview_dir = args.output_root / "preview" / point.river / season
    raw_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    tif_path = raw_dir / f"{segment_id}.tif"
    png_path = preview_dir / f"{segment_id}.png"
    json_path = raw_dir / f"{segment_id}.json"
    bbox = bbox_for_point(point.lon, point.lat, args.chip_size_meters)

    if tif_path.exists() and png_path.exists() and json_path.exists():
        return True

    items = search_items(catalog, bbox, season, args.cloud_cover, args.max_items)
    if not items and args.fallback_cloud_cover > args.cloud_cover:
        items = search_items(catalog, bbox, season, args.fallback_cloud_cover, args.max_items)

    for item in items:
        try:
            result = read_four_band(item, bbox)
            if result is None:
                continue
            stack, profile, signed = result
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stack)
            save_preview(stack, png_path)
            meta = {
                "river": point.river,
                "season": season,
                "year": 2025,
                "segment_id": segment_id,
                "sample_index": point.index,
                "lon": point.lon,
                "lat": point.lat,
                "bbox": bbox,
                "datetime": str(item.datetime),
                "month": str(item.datetime)[:7],
                "collection": item.collection_id,
                "item_id": item.id,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "bands": PC_SENTINEL_4BAND_ASSET_KEYS,
                "geotiff_path": str(tif_path),
                "preview_path": str(png_path),
                "asset_hrefs": {key: _asset_by_key(signed.assets, key).href for key in PC_SENTINEL_4BAND_ASSET_KEYS},
                "source": "Microsoft Planetary Computer",
            }
            json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            write_jsonl(manifest_path, meta)
            return True
        except Exception as exc:
            logger.warning("Failed %s %s %s on item %s: %s", point.river, point.index, season, item.id, exc)

    write_jsonl(failed_path, {"river": point.river, "season": season, "sample_index": point.index, "bbox": bbox})
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    rivers = _parse_list(args.rivers, RIVER_WAYPOINTS, "river")
    seasons = _parse_list(args.seasons, SEASON_WINDOWS, "season")
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_points = {river: sample_mainstem(river, args.spacing_km) for river in rivers}
    for river, points in all_points.items():
        logger.info("%s sample points: %d", river, len(points))

    if args.dry_run:
        planned = sum(
            min(len(points), args.max_per_river_season or len(points)) * len(seasons)
            for points in all_points.values()
        )
        logger.info("Dry run planned downloads: %d", planned)
        return

    manifest_path = metadata_dir / "download_manifest.jsonl"
    failed_path = metadata_dir / "failed_segments.jsonl"
    catalog = pystac_client.Client.open(PC_STAC_URL)
    downloaded = 0
    for river, points in all_points.items():
        selected = points[: args.max_per_river_season] if args.max_per_river_season else points
        for season in seasons:
            for point in selected:
                ok = download_one(catalog, point, season, args, manifest_path, failed_path)
                if ok:
                    downloaded += 1
                    logger.info("Saved %s %05d %s (%d)", river, point.index, season, downloaded)
    logger.info("Done. Saved or reused %d chips", downloaded)


if __name__ == "__main__":
    main()
