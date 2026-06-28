from pathlib import Path
import importlib.util
import sys

from shapely.geometry import LineString


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mine_scaled_water_strokes.py"
SPEC = importlib.util.spec_from_file_location("mine_scaled_water_strokes", SCRIPT_PATH)
mine_scaled_water_strokes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mine_scaled_water_strokes
SPEC.loader.exec_module(mine_scaled_water_strokes)


def test_merge_line_geometries_accepts_single_line():
    line = LineString([(0, 0), (1, 0)])

    merged = mine_scaled_water_strokes.merge_line_geometries([line])

    assert merged.equals(line)


def test_source_name_ignores_nan_lake_name():
    name = mine_scaled_water_strokes.source_name({"Lake_name": float("nan")})

    assert name == "unknown"
