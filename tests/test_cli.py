from pathlib import Path

import typer
from typer.testing import CliRunner

from rs_words.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "中文" in result.output or "河流" in result.output


def test_cli_mock(tmp_path: Path):
    patch_bank = tmp_path / "empty_bank"
    patch_bank.mkdir()
    # metadata.jsonl is intentionally missing
    result = runner.invoke(app, ["河", "--patch-bank", str(patch_bank)])
    assert result.exit_code != 0
