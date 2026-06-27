"""一键构建河流影像块库：OSM 提取 → PC 下载 → 块库规范化。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pystac_client

# Ensure project source is importable when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rs_words.config import (
    BASINS,
    OSM_DIR,
    PATCH_BANK_DIR,
    SATELLITE_DIR,
    PC_STAC_URL,
)
from rs_words.data_engine.osm_rivers import (
    build_segment_catalog,
    download_rivers_for_basin,
)
from rs_words.data_engine.pc_downloader import download_chip
from rs_words.data_engine.patch_bank import PatchBank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    print("=" * 60)
    print("Step 1/5: Ensure OSM output directory exists")
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  OSM_DIR: {OSM_DIR}")

    print("=" * 60)
    print("Step 2/5: Download river networks for each basin from OSM")
    basin_files: Dict[str, Path] = {}
    for basin_name, bbox in BASINS.items():
        print(f"  Downloading rivers for basin: {basin_name} (bbox={bbox})")
        path = download_rivers_for_basin(basin_name, bbox)
        basin_files[basin_name] = path
        print(f"    Saved to: {path}")
    print(f"  Collected {len(basin_files)} basin files.")

    print("=" * 60)
    print("Step 3/5: Build river segment catalog")
    segment_catalog = build_segment_catalog(basin_files)
    print(f"  Total segments: {len(segment_catalog)}")

    print("=" * 60)
    print("Step 4/5: Download satellite chips from Planetary Computer")
    catalog = pystac_client.Client.open(PC_STAC_URL)
    print(f"  Opened STAC catalog: {PC_STAC_URL}")

    downloaded: List[Path] = []
    skipped: List[str] = []
    for idx, row in segment_catalog.iterrows():
        segment_id = row.segment_id
        basin = row.basin
        print(f"  [{idx + 1}/{len(segment_catalog)}] Downloading {segment_id} ({basin})...")
        chip_path = download_chip(
            catalog=catalog,
            segment=row.geometry,
            segment_id=segment_id,
            basin=basin,
        )
        if chip_path is not None:
            downloaded.append(chip_path)
            print(f"    -> saved: {chip_path}")
        else:
            skipped.append(segment_id)
            print("    -> skipped")
    print(f"  Downloaded {len(downloaded)} chips, skipped {len(skipped)} segments.")

    print("=" * 60)
    print("Step 5/5: Normalize raw chips into PatchBank")
    patch_bank = PatchBank.build_from_raw_chips(SATELLITE_DIR, PATCH_BANK_DIR)
    print(f"  PatchBank created with {len(patch_bank)} patches at: {PATCH_BANK_DIR}")

    print("=" * 60)
    print("Build complete!")


if __name__ == "__main__":
    main()
