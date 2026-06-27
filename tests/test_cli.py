from pathlib import Path

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
    assert "patch bank not found" in result.output


def test_cli_connected_mode(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from PIL import Image

    from rs_words.cli import app

    bank_dir = tmp_path / "bank"
    patch_dir = bank_dir / "basin"
    patch_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (40, 120, 180)).save(patch_dir / "p.png")
    (bank_dir / "metadata.jsonl").write_text(
        '{"patch_id":"p","basin":"basin","image_path":"patch_bank/basin/p.png","river_metrics":{"water_fraction":0.2,"skeleton_length_px":20}}\n',
        encoding="utf-8",
    )

    # PatchBank paths are resolved relative to metadata_path.parent.parent.
    data_root = tmp_path
    (data_root / "patch_bank").symlink_to(bank_dir, target_is_directory=True)
    output = tmp_path / "out.png"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "河",
            "--mode",
            "connected",
            "--patch-bank",
            str(data_root / "patch_bank"),
            "--output",
            str(output),
            "--font-size",
            "128",
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
