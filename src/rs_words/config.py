from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# All project data (OSM, satellite chips, patch bank, outputs, fonts) lives outside
# the repository so large files are never accidentally committed.
DATA_DIR = Path("/data/rs_word")

OSM_DIR = DATA_DIR / "osm"
SATELLITE_DIR = DATA_DIR / "satellite_chips" / "raw"
PATCH_BANK_DIR = DATA_DIR / "patch_bank"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "outputs"
FONT_DIR = DATA_DIR / "fonts"
WEB_DIR = ROOT / "web"

CHIP_SIZE = 256
CHIP_SIZE_METERS = 2560  # 256 px @ 10 m/px
SEGMENT_LENGTH_METERS = 1500

DEFAULT_FONT_SIZE = 256
DEFAULT_K = 5

BASINS = {
    "yangtze": (98.0, 24.0, 122.0, 35.0),
    "yellow": (96.0, 32.0, 113.0, 42.0),
    "pearl": (105.0, 21.0, 116.0, 27.0),
}

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_COLLECTION_SENTINEL = "sentinel-2-l2a"
PC_COLLECTION_LANDSAT = "landsat-c2-l2"

PC_DATETIME_RANGE = "2022-01-01/2024-12-31"
PC_CLOUD_COVER_LT = 20
PC_FALLBACK_MAX_ITEMS = 5
PC_ASSET_KEYS = ["visual", "rendered_preview"]
