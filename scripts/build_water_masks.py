"""Build water masks and river geometry metrics for a patch bank."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rs_words.config import DATA_DIR, PATCH_BANK_DIR
from rs_words.water_masks import build_water_masks

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-bank",
        type=Path,
        default=PATCH_BANK_DIR,
        help="Patch bank directory containing metadata.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "water_masks",
        help="Directory where water mask PNGs will be written.",
    )
    parser.add_argument(
        "--backend",
        choices=["omniwatermask", "ndwi"],
        default="omniwatermask",
        help="Water segmentation backend.",
    )
    parser.add_argument(
        "--ndwi-threshold",
        type=float,
        default=0.0,
        help="NDWI threshold used by the fallback backend.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks even when a mask file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    metadata_path = args.patch_bank / "metadata.jsonl"
    logger.info("Building water masks with backend=%s", args.backend)
    generated = build_water_masks(
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        backend=args.backend,
        ndwi_threshold=args.ndwi_threshold,
        overwrite=args.overwrite,
    )
    logger.info("Generated or refreshed %d water masks", generated)


if __name__ == "__main__":
    main()
