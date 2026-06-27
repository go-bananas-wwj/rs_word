from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from rs_words.config import PATCH_BANK_DIR


@dataclass
class Patch:
    patch_id: str
    basin: str
    image: np.ndarray  # RGB uint8, shape (H, W, 3)
    meta: Dict


class PatchBank:
    def __init__(self, patches: List[Patch]):
        self.patches = patches
        self._by_id = {p.patch_id: p for p in patches}

    @classmethod
    def build_from_raw_chips(
        cls,
        raw_dir: Path,
        output_dir: Path = PATCH_BANK_DIR,
        target_size: int = 256,
    ) -> "PatchBank":
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.jsonl"
        patches = []

        # Support raw_dir being either a parent of basin directories or a single basin.
        img_exts = {".png", ".jpg", ".jpeg"}
        if any(
            p.is_file() and p.suffix.lower() in img_exts
            for p in raw_dir.iterdir()
        ):
            basin_dirs = [raw_dir]
        else:
            basin_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]

        with metadata_path.open("w", encoding="utf-8") as f:
            for basin_dir in basin_dirs:
                out_basin_dir = output_dir / basin_dir.name
                out_basin_dir.mkdir(exist_ok=True)
                img_paths = sorted(
                    p for p in basin_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in img_exts
                )
                for img_path in img_paths:
                    meta_path = basin_dir / f"{img_path.stem}.json"
                    if not meta_path.exists():
                        continue
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    arr = np.array(img, dtype=np.uint8)
                    out_path = out_basin_dir / img_path.name
                    img.save(out_path)
                    meta["patch_id"] = img_path.stem
                    meta["basin"] = basin_dir.name
                    meta["image_path"] = str(out_path.relative_to(output_dir.parent))
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    patches.append(Patch(patch_id=img_path.stem, basin=basin_dir.name, image=arr, meta=meta))
        return cls(patches)

    @classmethod
    def load(cls, metadata_path: Path = PATCH_BANK_DIR / "metadata.jsonl") -> "PatchBank":
        if not metadata_path.exists():
            raise FileNotFoundError(f"Patch bank metadata not found: {metadata_path}")
        patches = []
        root = metadata_path.parent.parent  # data/
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            meta = json.loads(line)
            img_path = root / meta["image_path"]
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            patches.append(Patch(patch_id=meta["patch_id"], basin=meta["basin"], image=arr, meta=meta))
        return cls(patches)

    def __len__(self) -> int:
        return len(self.patches)

    def get(self, patch_id: str) -> Optional[Patch]:
        return self._by_id.get(patch_id)
