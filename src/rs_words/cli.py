from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from rs_words.compositor import compose_text
from rs_words.config import DEFAULT_FONT_SIZE, DEFAULT_K, OUTPUT_DIR, PATCH_BANK_DIR
from rs_words.data_engine.patch_bank import PatchBank
from rs_words.glyph import decompose_text
from rs_words.matcher import RiverMatcher

app = typer.Typer(help="用真实河流卫星影像拼出汉字")


@app.command()
def create(
    text: str = typer.Argument(..., help="要渲染的中文文本"),
    output: Path = typer.Option(OUTPUT_DIR / "out.png", "--output", "-o"),
    font_path: Optional[Path] = typer.Option(None, "--font", help="CJK 字体路径"),
    patch_bank_dir: Path = typer.Option(PATCH_BANK_DIR, "--patch-bank"),
    font_size: int = typer.Option(DEFAULT_FONT_SIZE, "--font-size"),
    k: int = typer.Option(DEFAULT_K, "--k"),
    meta_output: Optional[Path] = typer.Option(None, "--meta"),
) -> None:
    """Render Chinese text as a river satellite-image mosaic."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Rendering and decomposing text: {text!r}")
    mask, strokes = decompose_text(text, font_path=font_path, font_size=font_size)
    typer.echo(f"Found {len(strokes)} strokes")

    metadata_path = patch_bank_dir / "metadata.jsonl"
    typer.echo(f"Loading patch bank from {metadata_path}")
    try:
        bank = PatchBank.load(metadata_path)
    except FileNotFoundError:
        typer.echo(f"Error: patch bank not found at {metadata_path}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Error loading patch bank: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Loaded {len(bank)} patches")

    matcher = RiverMatcher()
    matches = []
    stroke_entries = []
    for i, stroke in enumerate(strokes):
        top_k = matcher.match(stroke, bank, k=k)
        if not top_k:
            typer.echo(f"No patch match for stroke {i}; skipping")
            continue
        best_patch, best_score = top_k[0]
        matches.append((stroke, best_patch))
        entry = {
            "char_index": stroke.char_index,
            "bbox": stroke.bbox,
            "patch_id": best_patch.patch_id,
            "basin": best_patch.basin,
            "score": best_score,
            **best_patch.meta,
        }
        stroke_entries.append(entry)
        typer.echo(f"Stroke {i}: matched {best_patch.patch_id} (score {best_score:.4f})")

    typer.echo("Composing final mosaic")
    composed = compose_text(mask, matches)
    Image.fromarray(composed).save(output)
    typer.echo(f"Saved mosaic to {output}")

    if meta_output is not None:
        meta_output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"text": text, "strokes": stroke_entries}
        meta_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Saved metadata to {meta_output}")
