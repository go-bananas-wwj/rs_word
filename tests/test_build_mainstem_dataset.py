import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_mainstem_dataset.py"
    spec = importlib.util.spec_from_file_location("build_mainstem_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_list_rejects_unknown_values():
    module = _load_script_module()

    assert module._parse_list("yangtze,yellow", module.RIVER_WAYPOINTS, "river") == ["yangtze", "yellow"]
    with pytest.raises(ValueError, match="Unknown river"):
        module._parse_list("pearl", module.RIVER_WAYPOINTS, "river")


def test_sample_mainstem_produces_points():
    module = _load_script_module()

    points = module.sample_mainstem("yangtze", spacing_km=1000)

    assert points
    assert points[0].river == "yangtze"
    assert points[0].index == 0


def test_bbox_for_point_has_area():
    module = _load_script_module()

    bbox = module.bbox_for_point(114.0, 30.0, 2560)

    assert bbox[0] < 114.0 < bbox[2]
    assert bbox[1] < 30.0 < bbox[3]
