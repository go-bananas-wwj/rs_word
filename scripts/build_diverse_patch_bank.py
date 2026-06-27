"""Build a diverse patch bank by downloading river segments from multiple small areas.

This avoids the huge OSM download of an entire basin by sampling several smaller,
geographically varied bboxes across the Yangtze, Yellow, and Pearl river systems.
"""
from __future__ import annotations

import argparse
import logging
import random
from typing import Iterable, List, Tuple

import geopandas as gpd
import pystac_client
from shapely.geometry import LineString, MultiLineString

from rs_words.config import (
    OSM_DIR,
    PATCH_BANK_DIR,
    PC_STAC_URL,
    SATELLITE_DIR,
    SEGMENT_LENGTH_METERS,
)
from rs_words.data_engine.osm_rivers import download_rivers_for_basin, segment_line
from rs_words.data_engine.pc_downloader import download_chip
from rs_words.data_engine.patch_bank import PatchBank

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Small, diverse sampling areas across China's three major basins.
AREAS: List[Tuple[str, Tuple[float, float, float, float]]] = [
    ("yz_yichang", (111.0, 30.5, 112.0, 31.5)),      # Yangtze: gorges / winding
    ("yz_wuhan", (113.5, 29.5, 115.0, 31.0)),        # Yangtze: broad meanders
    ("hr_zhengzhou", (112.5, 34.5, 114.5, 35.5)),    # Yellow River: braided
    ("hr_bend", (108.0, 37.0, 110.0, 39.0)),          # Yellow River: bends
    ("pr_delta", (112.5, 22.5, 114.0, 23.5)),         # Pearl River: delta channels
]

MAX_CHIPS_PER_AREA = 30
MAX_TOTAL_CHIPS = 150
MAX_CANDIDATE_SEGMENTS_PER_AREA = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=["rgb", "geotiff4"],
        default="rgb",
        help="Satellite chip format to download.",
    )
    parser.add_argument(
        "--rgb-preview",
        action="store_true",
        help="Write RGB PNG previews when downloading four-band GeoTIFF chips.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=MAX_TOTAL_CHIPS,
        help="Maximum chips to download across all areas.",
    )
    parser.add_argument(
        "--max-per-area",
        type=int,
        default=MAX_CHIPS_PER_AREA,
        help="Maximum chips to download from each sampling area.",
    )
    parser.add_argument(
        "--max-candidates-per-area",
        type=int,
        default=MAX_CANDIDATE_SEGMENTS_PER_AREA,
        help="Maximum candidate river segments retained per area before STAC searches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible segment sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save the segment catalog without opening STAC or downloading chips.",
    )
    parser.add_argument(
        "--areas",
        default=None,
        help="Comma-separated area names to include, e.g. yz_yichang,yz_wuhan.",
    )
    return parser.parse_args()


def _positive_limit(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _selected_areas(
    areas: Iterable[Tuple[str, Tuple[float, float, float, float]]],
    include: str | None = None,
) -> list[Tuple[str, Tuple[float, float, float, float]]]:
    all_areas = list(areas)
    if include is None:
        return all_areas
    requested = {name.strip() for name in include.split(",") if name.strip()}
    known = {name for name, _ in all_areas}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown area(s): {', '.join(sorted(unknown))}")
    return [(name, bbox) for name, bbox in all_areas if name in requested]


def _collect_segments_for_area(name: str, bbox: tuple, max_candidates: int, rng: random.Random) -> List[dict]:
    """Download OSM rivers for a small area and segment them with prefixed IDs."""
    river_path = download_rivers_for_basin(name, bbox, output_dir=OSM_DIR)
    gdf = gpd.read_file(river_path)

    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            lines = [geom]
        for line in lines:
            if not isinstance(line, LineString):
                continue
            for seg in segment_line(line, SEGMENT_LENGTH_METERS):
                rows.append(
                    {
                        "basin": name,
                        "name": str(row.get("name", "unknown")),
                        "osm_id": str(row.get("osmid", row.get("id", "unknown"))),
                        "geometry": seg,
                    }
                )

    rng.shuffle(rows)
    if len(rows) > max_candidates:
        rows = rows[:max_candidates]
    return rows


def main() -> None:
    args = parse_args()
    max_total = _positive_limit(args.max_total, "--max-total")
    max_per_area = _positive_limit(args.max_per_area, "--max-per-area")
    max_candidates = _positive_limit(args.max_candidates_per_area, "--max-candidates-per-area")
    rng = random.Random(args.seed)

    OSM_DIR.mkdir(parents=True, exist_ok=True)
    SATELLITE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    areas = _selected_areas(AREAS, include=args.areas)
    for name, bbox in areas:
        logger.info("Collecting segments for %s ...", name)
        try:
            rows = _collect_segments_for_area(name, bbox, max_candidates=max_candidates, rng=rng)
        except Exception as exc:
            logger.warning("Failed to collect segments for %s: %s", name, exc)
            continue
        logger.info("Area %s produced %d segments", name, len(rows))
        all_rows.extend(rows)

    # Assign globally unique segment IDs with area prefix.
    for i, row in enumerate(all_rows):
        row["segment_id"] = f"seg_{i:08d}"

    segments_gdf = gpd.GeoDataFrame(all_rows, crs="EPSG:4326")
    segments_path = OSM_DIR / "diverse_segments.geojson"
    segments_gdf.to_file(segments_path, driver="GeoJSON")
    logger.info("Saved combined segment catalog with %d segments to %s", len(segments_gdf), segments_path)
    logger.info(
        "Run limits: max_total=%d, max_per_area=%d, max_candidates_per_area=%d, seed=%d",
        max_total,
        max_per_area,
        max_candidates,
        args.seed,
    )
    if args.dry_run:
        logger.info("Dry run requested; skipping STAC search, downloads, and patch bank rebuild.")
        return

    logger.info("Opening Planetary Computer catalog ...")
    catalog = pystac_client.Client.open(PC_STAC_URL)

    downloaded = 0
    per_area_counts = {name: 0 for name, _ in areas}
    for _, row in segments_gdf.iterrows():
        if downloaded >= max_total:
            break
        area = row["basin"]
        if per_area_counts[area] >= max_per_area:
            continue
        try:
            result = download_chip(
                catalog=catalog,
                segment=row.geometry,
                segment_id=row["segment_id"],
                basin="diverse",
                output_format=args.format,
                save_rgb_preview=args.rgb_preview or args.format == "rgb",
            )
            if result:
                downloaded += 1
                per_area_counts[area] += 1
                logger.info("Downloaded %s (%d/%d)", result, downloaded, max_total)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", row["segment_id"], exc)

    logger.info("Downloaded %d chips", downloaded)
    if downloaded == 0:
        logger.error("No chips downloaded; aborting patch bank build.")
        return

    logger.info("Building patch bank ...")
    bank = PatchBank.build_from_raw_chips(SATELLITE_DIR / "diverse", PATCH_BANK_DIR)
    logger.info("Patch bank size: %d", len(bank))


if __name__ == "__main__":
    main()
