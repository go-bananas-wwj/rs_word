from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from shapely.geometry import LineString

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_vector_strokes.py"
spec = importlib.util.spec_from_file_location("mine_vector_strokes", SCRIPT)
mine_vector_strokes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mine_vector_strokes
spec.loader.exec_module(mine_vector_strokes)


def _types_for(
    line: LineString,
    min_straight_km: float = 8.0,
    min_diagonal_km: float = 6.0,
    min_turn_km: float = 12.0,
) -> set[str]:
    candidates = mine_vector_strokes.score_segment(
        line_m=line,
        line_lonlat=LineString([(x / 100000, y / 100000) for x, y in line.coords]),
        source_index=1,
        target_river="test",
        width_mean_m=300,
        width_max_m=500,
        api_buffer_km=2,
        min_straight_km=min_straight_km,
        min_diagonal_km=min_diagonal_km,
        min_turn_km=min_turn_km,
    )
    return {candidate.stroke_type for candidate in candidates}


def test_split_line_window_ranges_includes_tail() -> None:
    line = LineString([(0, 0), (100, 0)])

    ranges = mine_vector_strokes.split_line_window_ranges(line, window_m=35, step_m=30)

    assert ranges == [(0.0, 35.0), (30.0, 65.0), (60.0, 95.0), (65.0, 100.0)]


def test_score_segment_classifies_horizontal_and_vertical() -> None:
    assert "heng" in _types_for(LineString([(0, 0), (50000, 0)]))
    assert "shu" in _types_for(LineString([(0, 0), (0, 50000)]))


def test_score_segment_classifies_diagonals() -> None:
    assert "na" in _types_for(LineString([(0, 0), (40000, 40000)]))
    assert "pie" in _types_for(LineString([(0, 40000), (40000, 0)]))


def test_score_segment_classifies_bend() -> None:
    types = _types_for(LineString([(0, 0), (25000, 0), (25000, 25000)]))

    assert "heng-zhe" in types or "shu-wan-gou" in types


def test_normalize_stroke_types_supports_dictionary_and_aliases() -> None:
    expected = {
        "heng",
        "shu",
        "pie",
        "na",
        "dian",
        "ti",
        "heng-zhe",
        "shu-gou",
        "heng-pie",
        "shu-wan-gou",
    }

    assert mine_vector_strokes.normalize_stroke_types("all") == expected
    assert mine_vector_strokes.normalize_stroke_types("hengzhe,shu_gou,hengpie,wan_gou") == {
        "heng-zhe",
        "shu-gou",
        "heng-pie",
        "shu-wan-gou",
    }


def test_score_segment_classifies_all_dictionary_strokes() -> None:
    samples = {
        "dian": LineString([(0, 0), (700, 600), (1400, 0)]),
        "ti": LineString([(0, 0), (3500, 1200)]),
        "heng-zhe": LineString([(0, 0), (12000, 0), (12000, 9000)]),
        "shu-gou": LineString([(0, 0), (0, 10000), (-3500, 12500)]),
        "heng-pie": LineString([(0, 0), (9000, 0), (15000, -6000)]),
        "shu-wan-gou": LineString([(0, 0), (0, 7000), (4500, 11000), (8000, 10500)]),
    }

    for stroke_type, line in samples.items():
        assert stroke_type in _types_for(line, min_straight_km=2.0, min_diagonal_km=1.2, min_turn_km=2.0)
