"""Download Sentinel-2 chips for vector stroke candidates."""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rs_words.config import PC_COLLECTION_SENTINEL, PC_SENTINEL_4BAND_ASSET_KEYS, PC_STAC_URL
from rs_words.data_engine.pc_downloader import _asset_by_key, _to_uint8_rgb

logger = logging.getLogger(__name__)
STROKE_TYPES = (
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
)
DEFAULT_CANDIDATE_ROOTS = [
    Path("/data2/rs_word_vectors/stroke_candidates/sentinel10_all_grwl_w5_v1"),
    Path("/data2/rs_word_vectors/stroke_candidates/sentinel10_all_grwl_w2p5_v1"),
    Path("/data2/rs_word_vectors/stroke_candidates/sentinel10_all_hydrorivers_w5_v1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-roots", type=Path, nargs="*", default=DEFAULT_CANDIDATE_ROOTS)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/rs_word_vectors/sentinel10_stroke_imagery_v1"))
    parser.add_argument("--strokes", default="all")
    parser.add_argument("--per-stroke", type=int, default=3)
    parser.add_argument("--datetime", default="2025-06-01/2025-11-30")
    parser.add_argument("--cloud-cover", type=float, default=15.0)
    parser.add_argument("--fallback-cloud-cover", type=float, default=30.0)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_strokes(value: str) -> list[str]:
    if value == "all":
        return list(STROKE_TYPES)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(STROKE_TYPES))
    if unknown:
        raise ValueError(f"Unknown strokes: {', '.join(unknown)}")
    return requested


def iter_candidates(candidate_roots: Iterable[Path], strokes: list[str], per_stroke: int) -> list[dict]:
    selected: dict[str, list[dict]] = {stroke: [] for stroke in strokes}
    seen: set[tuple[str, int, str]] = set()
    for root in candidate_roots:
        for stroke in strokes:
            path = root / stroke / f"{stroke}_top.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if len(selected[stroke]) >= per_stroke:
                    break
                row = json.loads(line)
                key = (stroke, int(row.get("source_index", -1)), str(row.get("target_river", "")))
                if key in seen:
                    continue
                row["candidate_root"] = str(root)
                row["stroke_type"] = stroke
                selected[stroke].append(row)
                seen.add(key)
    rows = []
    for stroke in strokes:
        rows.extend(selected[stroke])
    return rows


def search_items(catalog, bbox: tuple[float, float, float, float], datetime: str, cloud_cover: float, max_items: int):
    search = catalog.search(
        collections=[PC_COLLECTION_SENTINEL],
        bbox=bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": cloud_cover}},
        max_items=max_items,
    )
    items = list(search.items())
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover") or 999)
    return items


def read_four_band(item, bbox: tuple[float, float, float, float]):
    signed = planetary_computer.sign(item)
    bands = []
    profile = None
    for key in PC_SENTINEL_4BAND_ASSET_KEYS:
        asset = _asset_by_key(signed.assets, key)
        if asset is None:
            return None
        with rasterio.open(asset.href) as src:
            src_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*src_bbox, src.transform)
            band = src.read(1, window=win)
            if band.size == 0:
                return None
            bands.append(band)
            if profile is None:
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=4,
                    height=band.shape[0],
                    width=band.shape[1],
                    transform=src.window_transform(win),
                    dtype=band.dtype,
                )
    if len({band.shape for band in bands}) != 1:
        return None
    return np.stack(bands), profile, signed


def save_preview(stack: np.ndarray, path: Path) -> None:
    rgb = np.transpose(stack[[2, 1, 0]], (1, 2, 0))
    Image.fromarray(_to_uint8_rgb(rgb)).save(path)


def draw_overlay(preview_path: Path, row: dict, output_path: Path) -> None:
    image = Image.open(preview_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = f"{row['stroke_type']} {row.get('target_river')} score={row.get('score', 0):.2f}"
    draw.rectangle((0, 0, min(image.width, 360), 26), fill=(255, 255, 255))
    draw.text((6, 6), text, fill=(0, 0, 0))
    image.save(output_path)


def safe_id(row: dict, rank: int) -> str:
    stroke = row["stroke_type"]
    river = row.get("target_river", "river")
    lon = float(row.get("center_lon", 0.0))
    lat = float(row.get("center_lat", 0.0))
    return f"{stroke}_{rank:03d}_{river}_lon{lon:.3f}_lat{lat:.3f}".replace("-", "_")


def download_one(catalog, row: dict, rank: int, args: argparse.Namespace) -> dict | None:
    stroke = row["stroke_type"]
    bbox = tuple(float(v) for v in row["api_bbox"])
    out_dir = args.output_root / stroke
    out_dir.mkdir(parents=True, exist_ok=True)
    chip_id = safe_id(row, rank)
    tif_path = out_dir / f"{chip_id}.tif"
    png_path = out_dir / f"{chip_id}.png"
    overlay_path = out_dir / f"{chip_id}_label.png"
    json_path = out_dir / f"{chip_id}.json"
    if tif_path.exists() and png_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["preview_path"] = str(png_path)
        meta["overlay_path"] = str(overlay_path if overlay_path.exists() else png_path)
        return meta

    items = search_items(catalog, bbox, args.datetime, args.cloud_cover, args.max_items)
    if not items and args.fallback_cloud_cover > args.cloud_cover:
        items = search_items(catalog, bbox, args.datetime, args.fallback_cloud_cover, args.max_items)
    for item in items:
        try:
            result = read_four_band(item, bbox)
            if result is None:
                continue
            stack, profile, signed = result
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(stack)
            save_preview(stack, png_path)
            meta = {
                "chip_id": chip_id,
                "stroke_type": stroke,
                "rank": rank,
                "target_river": row.get("target_river"),
                "candidate": row,
                "bbox": bbox,
                "geotiff_path": str(tif_path),
                "preview_path": str(png_path),
                "overlay_path": str(overlay_path),
                "item_id": item.id,
                "collection": item.collection_id,
                "datetime": str(item.datetime),
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "bands": PC_SENTINEL_4BAND_ASSET_KEYS,
                "asset_hrefs": {key: _asset_by_key(signed.assets, key).href for key in PC_SENTINEL_4BAND_ASSET_KEYS},
                "source": "Microsoft Planetary Computer Sentinel-2 L2A",
            }
            json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            draw_overlay(png_path, row, overlay_path)
            return meta
        except Exception as exc:
            logger.warning("Failed %s item %s: %s", chip_id, item.id, exc)
    return None


def make_gallery(rows: list[dict], output_root: Path) -> None:
    css = """body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;background:#f7f7f4;color:#151515}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.card{background:white;border:1px solid #ddd;border-radius:6px;padding:10px}.card img{width:100%;display:block;border:1px solid #eee}.meta{font-size:12px;line-height:1.45}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;word-break:break-all}h2{margin-top:30px}"""
    parts = ["<!doctype html><meta charset='utf-8'><title>Sentinel stroke imagery</title>", f"<style>{css}</style>"]
    parts.append("<h1>Sentinel-2 10m 笔画候选遥感图像</h1>")
    current = None
    for row in sorted(rows, key=lambda r: (r["stroke_type"], r["rank"])):
        stroke = row["stroke_type"]
        if stroke != current:
            if current is not None:
                parts.append("</div>")
            parts.append(f"<h2>{html.escape(stroke)}</h2><div class='grid'>")
            current = stroke
        img_rel = Path(row["overlay_path"]).relative_to(output_root)
        parts.append("<div class='card'>")
        parts.append(f"<img src='{html.escape(str(img_rel))}' loading='lazy'>")
        parts.append("<div class='meta'>")
        parts.append(f"<b>{html.escape(row['chip_id'])}</b><br>")
        parts.append(f"date={html.escape(str(row['datetime'])[:10])} cloud={row.get('cloud_cover')}<br>")
        parts.append(f"river={html.escape(str(row.get('target_river')))} item={html.escape(str(row.get('item_id')))}")
        parts.append(f"<div class='mono'>{html.escape(str(row['bbox']))}</div>")
        parts.append("</div></div>")
    if current is not None:
        parts.append("</div>")
    (output_root / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


def make_sheet(rows: list[dict], output_root: Path, thumb_w: int = 240) -> None:
    if not rows:
        return
    thumbs = []
    for row in rows:
        im = Image.open(row["overlay_path"]).convert("RGB")
        scale = thumb_w / im.width
        im = im.resize((thumb_w, max(80, int(im.height * scale))), Image.Resampling.LANCZOS)
        thumbs.append((row, im))
    cols = 5
    label_h = 28
    cell_h = max(im.height for _, im in thumbs) + label_h
    sheet = Image.new("RGB", (cols * thumb_w, ((len(thumbs) + cols - 1) // cols) * cell_h + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "Sentinel-2 10m stroke imagery", fill=(0, 0, 0))
    for i, (row, im) in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = 40 + (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + im.height + 4), f"{row['stroke_type']} #{row['rank']} {row.get('target_river')}", fill=(0, 0, 0))
    sheet.save(output_root / "contact_sheet.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    strokes = parse_strokes(args.strokes)
    candidates = iter_candidates(args.candidate_roots, strokes, args.per_stroke)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    if args.dry_run:
        print(json.dumps(candidates[:5], ensure_ascii=False, indent=2))
        print(f"planned={len(candidates)}")
        return
    catalog = pystac_client.Client.open(PC_STAC_URL)
    rows = []
    for index, row in enumerate(candidates, 1):
        rank = sum(1 for existing in rows if existing["stroke_type"] == row["stroke_type"]) + 1
        meta = download_one(catalog, row, rank, args)
        if meta is None:
            logger.warning("No Sentinel chip saved for %s candidate %s", row["stroke_type"], index)
            continue
        rows.append(meta)
        logger.info("Saved %s (%d/%d)", meta["chip_id"], len(rows), len(candidates))
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    make_gallery(rows, args.output_root)
    make_sheet(rows, args.output_root)
    print(f"saved={len(rows)}")
    print(f"output={args.output_root}")


if __name__ == "__main__":
    main()
