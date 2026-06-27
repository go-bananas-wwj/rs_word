"""Info poster generator for rs-words mosaics."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INFO_PANEL_WIDTH = 500
LINE_HEIGHT = 30
MARGIN = 20


def create_info_poster(
    mosaic_path: Path,
    meta_path: Path,
    output_path: Path,
    font_path: Path | None = None,
) -> Path:
    """Create an information poster for a mosaic.

    Args:
        mosaic_path: Path to the mosaic PNG image.
        meta_path: Path to the metadata JSON file.
        output_path: Path where the poster will be saved.
        font_path: Optional path to a TrueType/OpenType font.

    Returns:
        The path to the saved poster image.
    """
    mosaic = Image.open(mosaic_path).convert("RGB")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    strokes = meta.get("strokes", [])
    text = meta.get("text", "")

    if font_path is not None:
        font = ImageFont.truetype(str(font_path), size=20)
    else:
        font = ImageFont.load_default()

    lines: list[str] = [
        f"文本: {text}",
        f"笔画数: {len(strokes)}",
    ]
    for s in strokes:
        patch_id = s.get("patch_id", "")
        basin = s.get("basin", "")
        name = s.get("name", "")
        cloud_cover = s.get("cloud_cover", "?")
        lines.append(f"{patch_id} | {basin} | {name} | 云量{cloud_cover}")

    text_block_height = len(lines) * LINE_HEIGHT + MARGIN * 2
    canvas_height = max(mosaic.height, text_block_height)
    canvas_width = mosaic.width + INFO_PANEL_WIDTH

    poster = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    poster.paste(mosaic, (0, 0))

    draw = ImageDraw.Draw(poster)
    x = mosaic.width + MARGIN
    y = MARGIN
    for line in lines:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += LINE_HEIGHT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    poster.save(output_path)
    return output_path
