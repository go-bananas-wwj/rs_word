"""Mine stroke-like river segments from vector river datasets."""

from __future__ import annotations

import argparse
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
    turn_angle_deg: float
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
    turn = _principal_turn(line_m.simplify(max(line_m.length / 18, 1.0)))
    center = line_lonlat.interpolate(0.5, normalized=True)
    bbox = tuple(float(v) for v in line_lonlat.bounds)
    base = {
        "source_index": source_index,
        "target_river": target_river,
        "width_mean_m": width_mean_m,
        "width_max_m": width_max_m,
        "length_km": line_m.length / 1000,
        "straightness": straightness,
        "sinuosity": sinuosity,
        "angle_deg": angle,
        "turn_angle_deg": turn,
        "aspect_ratio": aspect,
        "bbox": bbox,
        "center_lon": float(center.x),
        "center_lat": float(center.y),
        "api_bbox": _api_bbox(line_lonlat, api_buffer_km),
        "geometry_wkt": line_lonlat.wkt,
    }
    width_bonus = min((width_mean_m or 80.0) / 800.0, 1.5)
    candidates: list[StrokeCandidate] = []

    horizontal = max(0.0, 1.0 - _axis_distance(angle, 0) / 35.0)
    vertical = max(0.0, 1.0 - _axis_distance(angle, 90) / 35.0)
    diag_pos = max(0.0, 1.0 - _axis_distance(angle, 45) / 30.0)
    diag_neg = max(0.0, 1.0 - _axis_distance(angle, -45) / 30.0)
    straight_score = max(0.0, (straightness - 0.62) / 0.38)

    stroke_scores = {
        "heng": horizontal * straight_score * min(aspect / 3.0, 1.2),
        "shu": vertical * straight_score * min((1.0 / max(aspect, 0.05)) / 3.0, 1.2),
        "pie": diag_neg * straight_score,
        "na": diag_pos * straight_score,
        "bend": max(0.0, min(turn / 80.0, 1.4)) * max(0.0, min((sinuosity - 1.02) / 0.45, 1.2)),
        "hengzhe": horizontal * max(0.0, min((turn - 35.0) / 70.0, 1.2)) * max(0.0, min(aspect / 2.2, 1.2)),
    }
    for stroke_type, raw_score in stroke_scores.items():
        score = raw_score * (1.0 + 0.25 * width_bonus)
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
                for cand in score_segment(seg_m, seg_ll, idx, river, width_mean, width_max, api_buffer_km):
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


def write_outputs(candidates: list[StrokeCandidate], output_root: Path, max_per_type: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_jsonl = output_root / "stroke_candidates.jsonl"
    with all_jsonl.open("w", encoding="utf-8") as f:
        for cand in candidates:
            f.write(json.dumps(asdict(cand), ensure_ascii=False) + "\n")

    by_type: dict[str, list[StrokeCandidate]] = {}
    for cand in candidates:
        by_type.setdefault(cand.stroke_type, []).append(cand)

    for stroke_type, rows in sorted(by_type.items()):
        top = rows[:max_per_type]
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
    )
    write_outputs(candidates, args.output_root, args.max_per_type)
    print(f"input={input_vector}")
    print(f"candidates={len(candidates)}")
    for stroke_type in sorted(stroke_types):
        count = sum(c.stroke_type == stroke_type for c in candidates)
        print(f"{stroke_type}={count}")
    print(f"output={args.output_root}")


if __name__ == "__main__":
    main()
