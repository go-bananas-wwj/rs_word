from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_vector_stroke_imagery.py"
    spec = importlib.util.spec_from_file_location("download_vector_stroke_imagery", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_strokes_accepts_all_and_rejects_unknown() -> None:
    module = _load_module()

    assert module.parse_strokes("all") == list(module.STROKE_TYPES)
    assert module.parse_strokes("heng,shu") == ["heng", "shu"]
    with pytest.raises(ValueError, match="Unknown strokes"):
        module.parse_strokes("fake")


def test_iter_candidates_limits_per_stroke(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "candidates"
    stroke_dir = root / "heng"
    stroke_dir.mkdir(parents=True)
    rows = [
        {"stroke_type": "heng", "source_index": 1, "target_river": "yangtze", "api_bbox": [0, 0, 1, 1]},
        {"stroke_type": "heng", "source_index": 2, "target_river": "yellow", "api_bbox": [1, 1, 2, 2]},
    ]
    (stroke_dir / "heng_top.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = module.iter_candidates([root], ["heng"], per_stroke=1)

    assert len(selected) == 1
    assert selected[0]["target_river"] == "yangtze"


def test_scale_bbox_expands_around_center() -> None:
    module = _load_module()

    scaled = module.scale_bbox((0.0, 10.0, 10.0, 20.0), 1.2)

    assert scaled == pytest.approx((-1.0, 9.0, 11.0, 21.0))
