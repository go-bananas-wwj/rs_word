import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_diverse_patch_bank.py"
    spec = importlib.util.spec_from_file_location("build_diverse_patch_bank", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_exposes_bounded_run_controls(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_diverse_patch_bank.py",
            "--format",
            "geotiff4",
            "--rgb-preview",
            "--max-total",
            "12",
            "--max-per-area",
            "3",
            "--max-candidates-per-area",
            "9",
            "--seed",
            "7",
            "--dry-run",
        ],
    )

    args = module.parse_args()

    assert args.format == "geotiff4"
    assert args.rgb_preview is True
    assert args.max_total == 12
    assert args.max_per_area == 3
    assert args.max_candidates_per_area == 9
    assert args.seed == 7
    assert args.dry_run is True


def test_positive_limit_rejects_non_positive_values():
    module = _load_script_module()

    assert module._positive_limit(1, "--max-total") == 1
    with pytest.raises(ValueError, match="--max-total must be >= 1"):
        module._positive_limit(0, "--max-total")
