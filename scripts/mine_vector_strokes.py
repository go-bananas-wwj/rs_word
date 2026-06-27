"""Mine stroke-like river segments from vector river datasets."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
from PIL import Image, ImageDraw
from pyproj import Geod
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GEOD = Geod(ellps="WGS84")
DEFAULT_INPUTS = [
    Path("/data2/rs_word_vectors/clipped/yangtze_yellow_grwl_width_candidates.gpkg"),
    Path("/data2/rs_word_vectors/clipped/yangtze_yellow_hydrorivers_candidates.gpkg"),
]


@dataclass
class StrokeCandidate:
    stroke_type: str
    score: float
    source_index: int
    target_river: str
    width_mean_m: float | None
    width_max_m: float | None
    length_km: float
    straightness: float
    sinuosity: float
    angle_deg: float
    start_angle_deg: float
    end_angle_deg: float
    corner_angle_deg: float
    turn_angle_deg: float
    deviation_ratio: float
    turn_sign_consistency: float
    turn_reversals: int
    aspect_ratio: float
    bbox: tuple[float, float, float, float]
    center_lon: float
    center_lat: float
    api_bbox: tuple[float, float, float, float]
    geometry_wkt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-vector", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/rs_word_vectors/stroke_candidates"))
    parser.add_argument("--window-km", type=float, default=35.0)
    parser.add_argument("--step-km", type=float, default=10.0)
    parser.add_argument("--api-buffer-km", type=float, default=3.0)
    parser.add_argument("--min-width-mean", type=float, default=80.0)
    parser.add_argument("--max-per-type", type=int, default=80)
    parser.add_argument("--diversity-radius-km", type=float, default=18.0)
    parser.add_argument("--per-source-limit", type=int, default=2)
    parser.add_argument("--min-straight-km", type=float, default=8.0)
    parser.add_argument("--min-diagonal-km", type=float, default=6.0)
    parser.add_argument("--min-turn-km", type=float, default=12.0)
    parser.add_argument("--stroke-types", default="heng,shu,pie,na,bend,hengzhe")
    return parser.parse_args()


def iter_lines(geom) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms


def split_line_window_ranges(line: LineString, window_m: float, step_m: float) -> list[tuple[float, float]]:
    if line.length <= 0:
        return []
    if line.length <= window_m:
        return [(0.0, line.length)]
    ranges = []
    start = 0.0
    while start + window_m <= line.length:
        ranges.append((start, start + window_m))
        start += step_m
    tail = (max(0.0, line.length - window_m), line.length)
    if not ranges or tail != ranges[-1]:
        ranges.append(tail)
    return ranges


def split_line_windows(line: LineString, window_m: float, step_m: float) -> list[LineString]:
    windows = []
    for start, end in split_line_window_ranges(line, window_m, step_m):
        seg = substring(line, start, end)
        if isinstance(seg, LineString) and len(seg.coords) >= 2:
            windows.append(seg)
    return windows


def _angle_deg(dx: float, dy: float) -> float:
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 180:
        angle -= 360
    if angle <= -180:
        angle += 360
    return angle


def _axis_distance(angle: float, axis: float) -> float:
    diff = abs((angle - axis + 180) % 360 - 180)
    return min(diff, 180 - diff)


def _line_angle(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    sx, sy = coords[0]
    ex, ey = coords[-1]
    return _angle_deg(ex - sx, ey - sy)


def _leg_angles(line: LineString) -> tuple[float, float, float]:
    if line.length <= 0:
        return (0.0, 0.0, 0.0)
    first = substring(line, 0.0, min(line.length * 0.35, line.length))
    last = substring(line, max(0.0, line.length * 0.65), line.length)
    if not isinstance(first, LineString) or not isinstance(last, LineString):
        return (0.0, 0.0, 0.0)
    start_angle = _line_angle(first)
    end_angle = _line_angle(last)
    corner = abs((end_angle - start_angle + 180) % 360 - 180)
    return (start_angle, end_angle, min(corner, 180 - corner))


def _max_deviation_ratio(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0
    ax, ay = coords[0]
    bx, by = coords[-1]
    chord = math.hypot(bx - ax, by - ay)
    if chord <= 0:
        return 1.0
    max_dev = 0.0
    for px, py in coords[1:-1]:
        dev = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / chord
        max_dev = max(max_dev, dev)
    return max_dev / chord


def _turn_sign_stats(line: LineString) -> tuple[float, int]:
    coords = list(line.coords)
    signs = []
    for a, b, c in zip(coords[:-2], coords[1:-1], coords[2:]):
        abx, aby = b[0] - a[0], b[1] - a[1]
        bcx, bcy = c[0] - b[0], c[1] - b[1]
        cross = abx * bcy - aby * bcx
        if abs(cross) > 1e-6:
            signs.append(1 if cross > 0 else -1)
    if not signs:
        return (1.0, 0)
    dominant = max(signs.count(1), signs.count(-1)) / len(signs)
    reversals = sum(1 for prev, cur in zip(signs[:-1], signs[1:]) if prev != cur)
    return (dominant, reversals)


def _width_quality(width_mean_m: float | None) -> float:
    if width_mean_m is None:
        return 1.0
    if 120 <= width_mean_m <= 1800:
        return 1.08
    if width_mean_m > 3000:
        return 0.84
    if width_mean_m < 60:
        return 0.9
    return 1.0


def _band_score(value: float, low: float, high: float, feather: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / feather)
    return max(0.0, 1.0 - (value - high) / feather)


def _principal_turn(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0
    angles = []
    for a, b in zip(coords[:-1], coords[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if math.hypot(dx, dy) > 1e-6:
            angles.append(_angle_deg(dx, dy))
    if len(angles) < 2:
        return 0.0
    total = 0.0
    for prev, cur in zip(angles[:-1], angles[1:]):
        total += abs((cur - prev + 180) % 360 - 180)
    return min(total, 180.0)


def _api_bbox(line_lonlat: LineString, buffer_km: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = line_lonlat.bounds
    lon = (minx + maxx) / 2
    lat = (miny + maxy) / 2
    west, _, _ = GEOD.fwd(lon, lat, 270, buffer_km * 1000)
    east, _, _ = GEOD.fwd(lon, lat, 90, buffer_km * 1000)
    _, south, _ = GEOD.fwd(lon, lat, 180, buffer_km * 1000)
    _, north, _ = GEOD.fwd(lon, lat, 0, buffer_km * 1000)
    return (min(minx, west), min(miny, south), max(maxx, east), max(maxy, north))


def score_segment(
    line_m: LineString,
    line_lonlat: LineString,
    source_index: int,
    target_river: str,
    width_mean_m: float | None,
    width_max_m: float | None,
    api_buffer_km: float,
    min_straight_km: float = 8.0,
    min_diagonal_km: float = 6.0,
    min_turn_km: float = 12.0,
) -> list[StrokeCandidate]:
    coords = list(line_m.coords)
    if len(coords) < 2 or line_m.length <= 0:
        return []
    sx, sy = coords[0]
    ex, ey = coords[-1]
    dx = ex - sx
    dy = ey - sy
    chord = math.hypot(dx, dy)
    if chord <= 0:
        return []
    straightness = min(chord / line_m.length, 1.0)
    sinuosity = line_m.length / chord
    angle = _angle_deg(dx, dy)
    minx, miny, maxx, maxy = line_m.bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    aspect = width / height
    simplified = line_m.simplify(max(line_m.length / 18, 1.0))
    turn = _principal_turn(simplified)
    start_angle, end_angle, corner_angle = _leg_angles(line_m)
    deviation_ratio = _max_deviation_ratio(line_m)
    turn_sign_consistency, turn_reversals = _turn_sign_stats(simplified)
    length_km = line_m.length / 1000
    center = line_lonlat.interpolate(0.5, normalized=True)
    bbox = tuple(float(v) for v in line_lonlat.bounds)
    base = {
        "source_index": source_index,
        "target_river": target_river,
        "width_mean_m": width_mean_m,
        "width_max_m": width_max_m,
        "length_km": length_km,
        "straightness": straightness,
        "sinuosity": sinuosity,
        "angle_deg": angle,
        "start_angle_deg": start_angle,
        "end_angle_deg": end_angle,
        "corner_angle_deg": corner_angle,
        "turn_angle_deg": turn,
        "deviation_ratio": deviation_ratio,
        "turn_sign_consistency": turn_sign_consistency,
        "turn_reversals": turn_reversals,
        "aspect_ratio": aspect,
        "bbox": bbox,
        "center_lon": float(center.x),
        "center_lat": float(center.y),
        "api_bbox": _api_bbox(line_lonlat, api_buffer_km),
        "geometry_wkt": line_lonlat.wkt,
    }
    width_quality = _width_quality(width_mean_m)
    candidates: list[StrokeCandidate] = []

    horizontal = max(0.0, 1.0 - _axis_distance(angle, 0) / 25.0)
    vertical = max(0.0, 1.0 - _axis_distance(angle, 90) / 25.0)
    diag_pos = max(0.0, 1.0 - _axis_distance(angle, 45) / 24.0)
    diag_neg = max(0.0, 1.0 - _axis_distance(angle, -45) / 24.0)
    straight_score = max(0.0, (straightness - 0.86) / 0.14)
    clean_axis = max(0.0, 1.0 - deviation_ratio / 0.13)
    low_turn_penalty = _band_score(turn, 0.0, 16.0, 14.0)
    bend_turn_score = _band_score(turn, 55.0, 125.0, 35.0)
    bend_sinuosity_score = _band_score(sinuosity, 1.05, 1.6, 0.35)
    bend_clean_curve = turn_sign_consistency * max(0.0, 1.0 - turn_reversals / 2.0)
    hengzhe_start = max(0.0, 1.0 - _axis_distance(start_angle, 0) / 24.0)
    hengzhe_end = max(0.0, 1.0 - _axis_distance(end_angle, 90) / 28.0)
    hengzhe_corner = _band_score(corner_angle, 70.0, 105.0, 24.0)
    long_straight = _band_score(length_km, min_straight_km, 80.0, max(min_straight_km * 0.45, 0.5))
    diagonal_length = _band_score(length_km, min_diagonal_km, 70.0, max(min_diagonal_km * 0.45, 0.5))
    turn_length = _band_score(length_km, min_turn_km, 90.0, max(min_turn_km * 0.45, 0.5))

    stroke_scores = {
        "heng": horizontal * straight_score * clean_axis * low_turn_penalty * long_straight * min(aspect / 4.0, 1.15),
        "shu": vertical * straight_score * clean_axis * low_turn_penalty * long_straight * min((1.0 / max(aspect, 0.05)) / 4.0, 1.15),
        "pie": diag_neg * straight_score * clean_axis * low_turn_penalty * diagonal_length,
        "na": diag_pos * straight_score * clean_axis * low_turn_penalty * diagonal_length,
        "bend": bend_turn_score * bend_sinuosity_score * bend_clean_curve * turn_length * max(0.0, min((1.0 - straightness) / 0.35, 1.15)),
        "hengzhe": hengzhe_start * hengzhe_end * hengzhe_corner * turn_length * max(0.0, min(aspect / 1.4, 1.15)),
    }
    for stroke_type, raw_score in stroke_scores.items():
        score = raw_score * width_quality
        if score >= 0.20:
            candidates.append(StrokeCandidate(stroke_type=stroke_type, score=score, **base))
    return candidates


def _row_width(row) -> tuple[float | None, float | None]:
    mean = row.get("width_mean")
    max_width = row.get("width_max_")
    if mean is None and row.get("DIS_AV_CMS") is not None:
        # HydroRIVERS has no width; use flow only for ordering, not as a fake width.
        mean = None
    return (float(mean) if mean is not None and not math.isnan(float(mean)) else None,
            float(max_width) if max_width is not None and not math.isnan(float(max_width)) else None)


def mine_candidates(
    input_vector: Path,
    window_km: float,
    step_km: float,
    api_buffer_km: float,
    min_width_mean: float,
    stroke_types: set[str],
    min_straight_km: float = 8.0,
    min_diagonal_km: float = 6.0,
    min_turn_km: float = 12.0,
) -> list[StrokeCandidate]:
    gdf = gpd.read_file(input_vector)
    if gdf.empty:
        return []
    original = gdf.to_crs("EPSG:4326")
    projected = original.to_crs("EPSG:3857")
    candidates: list[StrokeCandidate] = []
    for idx, (row_ll, row_m) in enumerate(zip(original.to_dict("records"), projected.to_dict("records"))):
        width_mean, width_max = _row_width(row_ll)
        if width_mean is not None and width_mean < min_width_mean and (width_max or 0) < min_width_mean * 1.8:
            continue
        river = str(row_ll.get("target_river") or row_ll.get("target_riv") or "unknown")
        for line_m, line_ll in zip(iter_lines(row_m["geometry"]), iter_lines(row_ll["geometry"])):
            ranges_m = split_line_window_ranges(line_m, window_km * 1000, step_km * 1000)
            for start_m, end_m in ranges_m:
                seg_m = substring(line_m, start_m, end_m)
                start_norm = start_m / line_m.length if line_m.length else 0.0
                end_norm = end_m / line_m.length if line_m.length else 1.0
                seg_ll = substring(line_ll, start_norm, end_norm, normalized=True)
                if not isinstance(seg_m, LineString) or not isinstance(seg_ll, LineString):
                    continue
                for cand in score_segment(
                    seg_m,
                    seg_ll,
                    idx,
                    river,
                    width_mean,
                    width_max,
                    api_buffer_km,
                    min_straight_km=min_straight_km,
                    min_diagonal_km=min_diagonal_km,
                    min_turn_km=min_turn_km,
                ):
                    if cand.stroke_type in stroke_types:
                        candidates.append(cand)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def render_candidate(candidate: StrokeCandidate, path: Path, size: tuple[int, int] = (320, 220)) -> None:
    line = wkt.loads(candidate.geometry_wkt)
    minx, miny, maxx, maxy = candidate.api_bbox
    pad_x = max((maxx - minx) * 0.08, 1e-6)
    pad_y = max((maxy - miny) * 0.08, 1e-6)
    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y

    def xy(lon: float, lat: float) -> tuple[int, int]:
        x = int((lon - minx) / max(maxx - minx, 1e-9) * (size[0] - 28) + 14)
        y = int((maxy - lat) / max(maxy - miny, 1e-9) * (size[1] - 44) + 14)
        return x, y

    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, size[0] - 8, size[1] - 32), outline=(210, 210, 210))
    pts = [xy(x, y) for x, y in line.coords]
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 95, 180), width=8, joint="curve")
        draw.line(pts, fill=(18, 140, 220), width=4, joint="curve")
    label = f"{candidate.stroke_type} {candidate.target_river} score={candidate.score:.2f} w={candidate.width_mean_m or 0:.0f}m"
    draw.text((10, size[1] - 24), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_sheet(images: list[Path], output: Path, title: str, cols: int = 5) -> None:
    if not images:
        return
    thumbs = [Image.open(p).convert("RGB") for p in images]
    cell_w = max(im.width for im in thumbs)
    cell_h = max(im.height for im in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 38), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(0, 0, 0))
    for i, im in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = 38 + (i // cols) * cell_h
        sheet.paste(im, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _distance_km(a: StrokeCandidate, b: StrokeCandidate) -> float:
    _, _, meters = GEOD.inv(a.center_lon, a.center_lat, b.center_lon, b.center_lat)
    return meters / 1000.0


def diverse_top(candidates: list[StrokeCandidate], max_items: int, radius_km: float, per_source_limit: int) -> list[StrokeCandidate]:
    picked: list[StrokeCandidate] = []
    source_counts: dict[tuple[str, int], int] = {}
    for cand in candidates:
        source_key = (cand.target_river, cand.source_index)
        if source_counts.get(source_key, 0) >= per_source_limit:
            continue
        if any(_distance_km(cand, existing) < radius_km for existing in picked):
            continue
        picked.append(cand)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        if len(picked) >= max_items:
            break
    if len(picked) < max_items:
        for cand in candidates:
            if cand in picked:
                continue
            picked.append(cand)
            if len(picked) >= max_items:
                break
    return picked


def make_gallery(candidates_by_type: dict[str, list[StrokeCandidate]], output_root: Path, max_per_type: int) -> None:
    cards = []
    for stroke_type, rows in sorted(candidates_by_type.items()):
        for rank, cand in enumerate(rows[:max_per_type], 1):
            img_path = Path(stroke_type) / f"{stroke_type}_{rank:03d}.png"
            cards.append((cand.target_river, stroke_type, rank, img_path, cand))
    css = """
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f7f4;color:#161616}.top{position:sticky;top:0;background:white;border-bottom:1px solid #ddd;padding:14px 18px;z-index:2}.group{padding:16px 18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}.card{background:white;border:1px solid #ddd;border-radius:6px;padding:8px}.card img{width:100%;display:block}.meta{font-size:12px;line-height:1.45;color:#333}.id{font-weight:700;font-size:13px;color:#111}.bbox{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;word-break:break-all;color:#555}.badge{display:inline-block;margin-right:6px;margin-top:5px;padding:2px 8px;border:1px solid #ddd;border-radius:999px;background:#eee;font-size:12px}
"""
    parts = ["<!doctype html><meta charset='utf-8'><title>Vector Stroke Candidates</title>", f"<style>{css}</style>"]
    parts.append("<div class='top'><h2 style='margin:0'>长江 / 黄河矢量笔画候选</h2>")
    for river in sorted({c[0] for c in cards}):
        for stroke in sorted({c[1] for c in cards if c[0] == river}):
            count = sum(1 for c in cards if c[0] == river and c[1] == stroke)
            parts.append(f"<span class='badge'>{html.escape(river)} / {html.escape(stroke)}: {count}</span>")
    parts.append("</div>")
    current = None
    for river, stroke_type, rank, img_path, cand in sorted(cards, key=lambda x: (x[0], x[1], x[2])):
        group = (river, stroke_type)
        if group != current:
            if current is not None:
                parts.append("</div></section>")
            parts.append(f"<section class='group'><h3>{html.escape(river)} / {html.escape(stroke_type)}</h3><div class='grid'>")
            current = group
        bbox = ", ".join(f"{v:.6f}" for v in cand.api_bbox)
        ident = f"{river}_{stroke_type}_{rank:03d}_src{cand.source_index}_lon{cand.center_lon:.3f}_lat{cand.center_lat:.3f}"
        parts.append("<div class='card'>")
        parts.append(f"<img src='{html.escape(str(img_path))}' loading='lazy'>")
        parts.append("<div class='meta'>")
        parts.append(f"<div class='id'>{html.escape(ident)}</div>")
        parts.append(f"score={cand.score:.3f} width={cand.width_mean_m or 0:.0f}m len={cand.length_km:.1f}km angle={cand.angle_deg:.1f} turn={cand.turn_angle_deg:.1f}<br>")
        parts.append(f"center=({cand.center_lon:.5f}, {cand.center_lat:.5f})<br>")
        parts.append(f"<div class='bbox'>api_bbox={html.escape(bbox)}</div>")
        parts.append("</div></div>")
    if current is not None:
        parts.append("</div></section>")
    (output_root / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


def write_outputs(
    candidates: list[StrokeCandidate],
    output_root: Path,
    max_per_type: int,
    diversity_radius_km: float,
    per_source_limit: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_jsonl = output_root / "stroke_candidates.jsonl"
    with all_jsonl.open("w", encoding="utf-8") as f:
        for cand in candidates:
            f.write(json.dumps(asdict(cand), ensure_ascii=False) + "\n")

    by_type: dict[str, list[StrokeCandidate]] = {}
    for cand in candidates:
        by_type.setdefault(cand.stroke_type, []).append(cand)

    selected_by_type: dict[str, list[StrokeCandidate]] = {}
    for stroke_type, rows in sorted(by_type.items()):
        top = diverse_top(rows, max_per_type, diversity_radius_km, per_source_limit)
        selected_by_type[stroke_type] = top
        type_dir = output_root / stroke_type
        type_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        with (type_dir / f"{stroke_type}_top.jsonl").open("w", encoding="utf-8") as f:
            for rank, cand in enumerate(top, 1):
                row = asdict(cand) | {"rank": rank}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                img_path = type_dir / f"{stroke_type}_{rank:03d}.png"
                render_candidate(cand, img_path)
                image_paths.append(img_path)
        make_sheet(image_paths, output_root / f"{stroke_type}_top_sheet.png", f"{stroke_type} vector river candidates")
    make_gallery(selected_by_type, output_root, max_per_type)


def pick_default_input() -> Path:
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    raise FileNotFoundError("No default vector input found. Pass --input-vector.")


def main() -> None:
    args = parse_args()
    input_vector = args.input_vector or pick_default_input()
    stroke_types = {s.strip() for s in args.stroke_types.split(",") if s.strip()}
    candidates = mine_candidates(
        input_vector=input_vector,
        window_km=args.window_km,
        step_km=args.step_km,
        api_buffer_km=args.api_buffer_km,
        min_width_mean=args.min_width_mean,
        stroke_types=stroke_types,
        min_straight_km=args.min_straight_km,
        min_diagonal_km=args.min_diagonal_km,
        min_turn_km=args.min_turn_km,
    )
    write_outputs(candidates, args.output_root, args.max_per_type, args.diversity_radius_km, args.per_source_limit)
    print(f"input={input_vector}")
    print(f"candidates={len(candidates)}")
    for stroke_type in sorted(stroke_types):
        count = sum(c.stroke_type == stroke_type for c in candidates)
        print(f"{stroke_type}={count}")
    print(f"output={args.output_root}")


if __name__ == "__main__":
    main()
