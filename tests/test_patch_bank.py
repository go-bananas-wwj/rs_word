from pathlib import Path

from PIL import Image

from rs_words.data_engine.patch_bank import PatchBank


def test_build_and_load(tmp_path: Path):
    import json
    raw = tmp_path / "raw" / "yangtze"
    raw.mkdir(parents=True)
    img = Image.new("RGB", (128, 128), (0, 100, 200))
    img.save(raw / "seg_00000001.png")
    meta = {"item_id": "test", "collection": "sentinel-2-l2a"}
    (raw / "seg_00000001.json").write_text(json.dumps(meta))

    bank = PatchBank.build_from_raw_chips(raw, output_dir=tmp_path / "bank")
    assert len(bank) == 1
    loaded = PatchBank.load(tmp_path / "bank" / "metadata.jsonl")
    assert len(loaded) == 1


def test_build_from_jpg_chip(tmp_path: Path):
    import json
    raw = tmp_path / "raw" / "amazon"
    raw.mkdir(parents=True)
    img = Image.new("RGB", (128, 128), (200, 50, 0))
    img.save(raw / "seg_00000002.jpg", quality=90)
    meta = {"item_id": "test-jpg", "collection": "sentinel-2-l2a"}
    (raw / "seg_00000002.json").write_text(json.dumps(meta))

    bank = PatchBank.build_from_raw_chips(raw, output_dir=tmp_path / "bank")
    assert len(bank) == 1
    assert bank.get("seg_00000002") is not None
    loaded = PatchBank.load(tmp_path / "bank" / "metadata.jsonl")
    assert len(loaded) == 1
