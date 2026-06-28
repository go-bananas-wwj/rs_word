"""Mine China-wide water stroke candidates with stroke-specific scale rules."""

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
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, substring, unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GEOD = Geod(ellps="WGS84")
CHINA_BBOX = (73.0, 18.0, 135.5, 54.5)
DEFAULT_RIVER_VECTOR = Path(
    "/data2/rs_word_vectors/hydrorivers_asia/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp"
)
DEFAULT_LAKE_VECTOR = Path(
    "/data2/rs_word_vectors/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"
)


@dataclass(frozen=True)
class StrokeScale:
    window_km: float
    step_km: float
    min_length_km: float
    max_length_km: float
    target_aspect: float
    axis: float | None
    angle_tol: float
    min_straightness: float
    max_turn: float
    crop_scale: float
    render_size: tuple[int, int]
    source: str = "river"


STROKE_SCALES = {
    "heng": StrokeScale(95, 16, 38, 160, 5.8, 0, 28, 0.72, 55, 1.18, (520, 170)),
    "shu": StrokeScale(95, 16, 38, 160, 1 / 5.8, 90, 28, 0.72, 55, 1.18, (180, 520)),
    "pie": StrokeScale(76, 12, 26, 125, 1.35, -45, 32, 0.7, 60, 1.25, (360, 360)),
    "na": StrokeScale(76, 12, 26, 125, 1.35, 45, 32, 0.7, 60, 1.25, (360, 360)),
    "ti": StrokeScale(42, 7, 10, 60, 2.8, 18, 30, 0.72, 50, 1.25, (380, 220)),
    "heng-zhe": StrokeScale(72, 12, 28, 110, 1.25, None, 30, 0.55, 145, 1.35, (360, 360)),
    "heng-pie": StrokeScale(72, 12, 28, 110, 1.25, None, 30, 0.55, 145, 1.35, (360, 360)),
    "shu-gou": StrokeScale(62, 10, 24, 95, 0.62, None, 34, 0.55, 135, 1.35, (270, 430)),
    "shu-wan-gou": StrokeScale(80, 12, 30, 125, 0.85, None, 38, 0.5, 165, 1.45, (340, 430)),
    "dian": StrokeScale(12, 4, 0.5, 18, 1.0, None, 180, 0.0, 180, 1.4, (260, 260), source="lake"),
}


@dataclass
class ScaledStrokeCandidate:
    chip_id: str
    stroke_type: str
    score: float
    source_index: int
    target_river: str
    water_source: str
    length_km: float
    area_km2: float | None
    straightness: float
    angle_deg: float
    start_angle_deg: float
    end_angle_deg: float
    corner_angle_deg: float
    turn_angle_deg: float
    aspect_ratio: float
    scale_profile: dict
    bbox: tuple[float, float, float, float]
    center_lon: float
    center_lat: float
    api_bbox: tuple[float, float, float, float]
    geometry_wkt: str
    preview_path: str
    overlay_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--river-vector", type=Path, default=DEFAULT_RIVER_VECTOR)
    parser.add_argument("--lake-vector", type=Path, default=DEFAULT_LAKE_VECTOR)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/rs_word_vectors/scaled_water_strokes_v1"))
    parser.add_argument("--bbox", type=float, nargs=4, default=CHINA_BBOX)
    parser.add_argument("--max-per-type", type=int, default=180)
    parser.add_argument("--candidate-limit", type=int, default=4_000)
    parser.add_argument("--min-discharge-cms", type=float, default=80.0)
    parser.add_argument("--max-river-features", type=int, default=30_000)
    parser.add_argument("--no-merge-main-rivers", action="store_true")
    parser.add_argument("--diversity-radius-km", type=float, default=22.0)
    parser.add_argument("--strokes", default="all")
    return parser.parse_args()


def parse_strokes(value: str) -> list[str]:
    if value == "all":
        return list(STROKE_SCALES)
    strokes = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(strokes) - set(STROKE_SCALES))
    if unknown:
        raise ValueError(f"Unknown strokes: {', '.join(unknown)}")
    return strokes


def iter_lines(geom) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms


def iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms


def angle_deg(dx: float, dy: float) -> float:
    angle = math.degrees(math.atan2(dy, dx))
    if angle <= -180:
        angle += 360
    if angle > 180:
        angle -= 360
    return angle


def axis_distance(angle: float, axis: float) -> float:
    diff = abs((angle - axis + 180) % 360 - 180)
    return min(diff, 180 - diff)


def band_score(value: float, low: float, high: float, feather: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / feather)
    return max(0.0, 1.0 - (value - high) / feather)


def line_angle(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    sx, sy = coords[0]
    ex, ey = coords[-1]
    return angle_deg(ex - sx, ey - sy)


def leg_angles(line: LineString) -> tuple[float, float, float]:
    if line.length <= 0:
        return (0.0, 0.0, 0.0)
    first = substring(line, 0.0, min(line.length * 0.38, line.length))
    last = substring(line, max(0.0, line.length * 0.62), line.length)
    if not isinstance(first, LineString) or not isinstance(last, LineString):
        return (0.0, 0.0, 0.0)
    start = line_angle(first)
    end = line_angle(last)
    corner = abs((end - start + 180) % 360 - 180)
    return (start, end, min(corner, 180 - corner))


def principal_turn(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0
    angles = []
    for a, b in zip(coords[:-1], coords[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) > 1e-6:
            angles.append(angle_deg(dx, dy))
    total = 0.0
    for prev, cur in zip(angles[:-1], angles[1:]):
        total += abs((cur - prev + 180) % 360 - 180)
    return min(total, 180.0)


def crop_bbox_for_line(line_lonlat: LineString, aspect: float, scale: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = line_lonlat.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    _, _, diag_m = GEOD.inv(minx, miny, maxx, maxy)
    if diag_m <= 0:
        diag_m = 1000.0
    if aspect >= 1:
        width_m = max(diag_m * scale, 1000.0)
        height_m = width_m / aspect
    else:
        height_m = max(diag_m * scale, 1000.0)
        width_m = height_m * aspect
    west, _, _ = GEOD.fwd(cx, cy, 270, width_m / 2)
    east, _, _ = GEOD.fwd(cx, cy, 90, width_m / 2)
    _, south, _ = GEOD.fwd(cx, cy, 180, height_m / 2)
    _, north, _ = GEOD.fwd(cx, cy, 0, height_m / 2)
    return (west, south, east, north)


def crop_bbox_for_polygon(poly_lonlat: Polygon, scale: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = poly_lonlat.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    _, _, width_m = GEOD.inv(minx, cy, maxx, cy)
    _, _, height_m = GEOD.inv(cx, miny, cx, maxy)
    side_m = max(width_m, height_m, 600.0) * scale
    west, _, _ = GEOD.fwd(cx, cy, 270, side_m / 2)
    east, _, _ = GEOD.fwd(cx, cy, 90, side_m / 2)
    _, south, _ = GEOD.fwd(cx, cy, 180, side_m / 2)
    _, north, _ = GEOD.fwd(cx, cy, 0, side_m / 2)
    return (west, south, east, north)


def split_line_ranges(line: LineString, window_m: float, step_m: float) -> list[tuple[float, float]]:
    if line.length <= 0:
        return []
    if line.length <= window_m:
        return [(0.0, line.length)]
    ranges = []
    start = 0.0
    while start + window_m <= line.length:
        ranges.append((start, start + window_m))
        start += step_m
    tail = (line.length - window_m, line.length)
    if not ranges or tail != ranges[-1]:
        ranges.append(tail)
    return ranges


def score_axis_line(line_m: LineString, scale: StrokeScale) -> tuple[float, dict] | None:
    coords = list(line_m.coords)
    if len(coords) < 2 or line_m.length <= 0:
        return None
    sx, sy = coords[0]
    ex, ey = coords[-1]
    chord = math.hypot(ex - sx, ey - sy)
    if chord <= 0:
        return None
    length_km = line_m.length / 1000
    straightness = min(chord / line_m.length, 1.0)
    angle = angle_deg(ex - sx, ey - sy)
    start, end, corner = leg_angles(line_m)
    turn = principal_turn(line_m.simplify(max(line_m.length / 24, 1.0)))
    minx, miny, maxx, maxy = line_m.bounds
    aspect = max(maxx - minx, 1.0) / max(maxy - miny, 1.0)

    length_score = band_score(length_km, scale.min_length_km, scale.max_length_km, scale.min_length_km * 0.45)
    straight_score = max(0.0, (straightness - scale.min_straightness) / max(1.0 - scale.min_straightness, 1e-6))
    turn_score = band_score(turn, 0, scale.max_turn, max(scale.max_turn * 0.8, 8))
    if scale.axis is None:
        direction_score = 1.0
    else:
        direction_score = max(0.0, 1.0 - axis_distance(angle, scale.axis) / scale.angle_tol)
    if scale.target_aspect >= 1:
        aspect_score = min(aspect / scale.target_aspect, 1.15)
    else:
        aspect_score = min((1 / max(aspect, 0.01)) / (1 / scale.target_aspect), 1.15)
    score = length_score * straight_score * turn_score * direction_score * aspect_score
    return score, {
        "length_km": length_km,
        "straightness": straightness,
        "angle": angle,
        "start_angle": start,
        "end_angle": end,
        "corner_angle": corner,
        "turn": turn,
        "aspect": aspect,
    }


def score_compound_line(line_m: LineString, stroke_type: str, scale: StrokeScale) -> tuple[float, dict] | None:
    basic = score_axis_line(line_m, scale)
    if basic is None:
        return None
    _, metrics = basic
    length_score = band_score(metrics["length_km"], scale.min_length_km, scale.max_length_km, scale.min_length_km * 0.45)
    turn_score = band_score(metrics["corner_angle"], 35, scale.max_turn, 26)
    if stroke_type == "heng-zhe":
        shape = max(
            max(0.0, 1 - axis_distance(metrics["start_angle"], 0) / 25)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 90) / 34),
            max(0.0, 1 - axis_distance(metrics["start_angle"], 90) / 34)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 0) / 25),
        )
    elif stroke_type == "heng-pie":
        shape = max(
            max(0.0, 1 - axis_distance(metrics["start_angle"], 0) / 25)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], -45) / 38),
            max(0.0, 1 - axis_distance(metrics["start_angle"], -45) / 38)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 0) / 25),
        )
    elif stroke_type == "shu-gou":
        shape = max(
            max(0.0, 1 - axis_distance(metrics["start_angle"], 90) / 28)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 150) / 45),
            max(0.0, 1 - axis_distance(metrics["start_angle"], 150) / 45)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 90) / 28),
        )
    else:
        shape = max(
            max(0.0, 1 - axis_distance(metrics["start_angle"], 90) / 34)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 0) / 45),
            max(0.0, 1 - axis_distance(metrics["start_angle"], 0) / 45)
            * max(0.0, 1 - axis_distance(metrics["end_angle"], 90) / 34),
        )
    score = length_score * turn_score * shape * band_score(metrics["straightness"], 0.45, 0.92, 0.25)
    return score, metrics


def source_name(row: dict) -> str:
    main = row.get("MAIN_RIV")
    if main is not None and not (isinstance(main, float) and math.isnan(main)):
        return f"hydroriver_{main}"
    lake = row.get("Lake_name") or row.get("Lake_type") or row.get("Hylak_id")
    if lake is not None and not (isinstance(lake, float) and math.isnan(lake)):
        return f"lake_{lake}"
    return "unknown"


def merge_line_geometries(geometries: list) -> LineString | MultiLineString:
    merged = unary_union(geometries)
    if isinstance(merged, LineString):
        return merged
    return linemerge(merged)


def make_line_candidate(
    stroke_type: str,
    score: float,
    metrics: dict,
    row_ll: dict,
    line_ll: LineString,
    source_index: int,
    output_root: Path,
) -> ScaledStrokeCandidate:
    scale = STROKE_SCALES[stroke_type]
    center = line_ll.interpolate(0.5, normalized=True)
    api_bbox = crop_bbox_for_line(line_ll, scale.target_aspect, scale.crop_scale)
    chip_id = f"{stroke_type}_{source_index}_lon{center.x:.3f}_lat{center.y:.3f}".replace("-", "_")
    rel = Path(stroke_type) / f"{chip_id}.png"
    return ScaledStrokeCandidate(
        chip_id=chip_id,
        stroke_type=stroke_type,
        score=float(score),
        source_index=source_index,
        target_river=source_name(row_ll),
        water_source="HydroRIVERS",
        length_km=float(metrics["length_km"]),
        area_km2=None,
        straightness=float(metrics["straightness"]),
        angle_deg=float(metrics["angle"]),
        start_angle_deg=float(metrics["start_angle"]),
        end_angle_deg=float(metrics["end_angle"]),
        corner_angle_deg=float(metrics["corner_angle"]),
        turn_angle_deg=float(metrics["turn"]),
        aspect_ratio=float(metrics["aspect"]),
        scale_profile={
            "target_aspect": scale.target_aspect,
            "render_size": scale.render_size,
            "crop_scale": scale.crop_scale,
            "relative_role": "long" if stroke_type in {"heng", "shu", "pie", "na"} else "compound",
        },
        bbox=tuple(float(v) for v in line_ll.bounds),
        center_lon=float(center.x),
        center_lat=float(center.y),
        api_bbox=api_bbox,
        geometry_wkt=line_ll.wkt,
        preview_path=str(output_root / rel),
        overlay_path=str(output_root / rel),
    )


def mine_river_candidates(
    river_vector: Path,
    bbox: tuple[float, float, float, float],
    strokes: list[str],
    output_root: Path,
    candidate_limit: int,
    min_discharge_cms: float,
    max_river_features: int,
    merge_main_rivers: bool,
) -> list[ScaledStrokeCandidate]:
    line_strokes = [s for s in strokes if STROKE_SCALES[s].source == "river"]
    if not line_strokes:
        return []
    gdf = gpd.read_file(river_vector, bbox=bbox, engine="pyogrio")
    if gdf.empty:
        return []
    if "DIS_AV_CMS" in gdf.columns:
        gdf = gdf[gdf["DIS_AV_CMS"].fillna(0) >= min_discharge_cms]
        gdf = gdf.sort_values("DIS_AV_CMS", ascending=False).head(max_river_features)
    if merge_main_rivers and "MAIN_RIV" in gdf.columns:
        merged_rows = []
        for main_riv, group in gdf.groupby("MAIN_RIV", sort=False):
            geom = merge_line_geometries(group.geometry.to_list())
            row = group.iloc[0].copy()
            row["geometry"] = geom
            row["DIS_AV_CMS"] = group["DIS_AV_CMS"].max()
            row["MAIN_RIV"] = main_riv
            merged_rows.append(row)
        gdf = gpd.GeoDataFrame(merged_rows, crs=gdf.crs)
    original = gdf.to_crs("EPSG:4326")
    projected = original.to_crs("EPSG:3857")
    candidates: list[ScaledStrokeCandidate] = []
    counts = {stroke_type: 0 for stroke_type in line_strokes}
    for idx, (row_ll, row_m) in enumerate(zip(original.to_dict("records"), projected.to_dict("records"))):
        if all(counts[stroke_type] >= candidate_limit for stroke_type in line_strokes):
            return candidates
        flow = row_ll.get("DIS_AV_CMS")
        if flow is not None and not math.isnan(float(flow)) and float(flow) < min_discharge_cms:
            continue
        for line_m, line_ll in zip(iter_lines(row_m["geometry"]), iter_lines(row_ll["geometry"])):
            if line_m.length < 1000:
                continue
            for stroke_type in line_strokes:
                if counts[stroke_type] >= candidate_limit:
                    continue
                scale = STROKE_SCALES[stroke_type]
                for start_m, end_m in split_line_ranges(line_m, scale.window_km * 1000, scale.step_km * 1000):
                    if counts[stroke_type] >= candidate_limit:
                        break
                    seg_m = substring(line_m, start_m, end_m)
                    start_norm = start_m / line_m.length
                    end_norm = end_m / line_m.length
                    seg_ll = substring(line_ll, start_norm, end_norm, normalized=True)
                    if not isinstance(seg_m, LineString) or not isinstance(seg_ll, LineString):
                        continue
                    scored = (
                        score_axis_line(seg_m, scale)
                        if scale.axis is not None
                        else score_compound_line(seg_m, stroke_type, scale)
                    )
                    if scored is None:
                        continue
                    score, metrics = scored
                    if score >= 0.18:
                        candidates.append(make_line_candidate(stroke_type, score, metrics, row_ll, seg_ll, idx, output_root))
                        counts[stroke_type] += 1
    return candidates


def mine_lake_dots(
    lake_vector: Path,
    bbox: tuple[float, float, float, float],
    output_root: Path,
    candidate_limit: int,
) -> list[ScaledStrokeCandidate]:
    if not lake_vector.exists():
        return []
    gdf = gpd.read_file(lake_vector, bbox=bbox, engine="pyogrio")
    if gdf.empty:
        return []
    original = gdf.to_crs("EPSG:4326")
    projected = original.to_crs("EPSG:3857")
    candidates: list[ScaledStrokeCandidate] = []
    scale = STROKE_SCALES["dian"]
    for idx, (row_ll, row_m) in enumerate(zip(original.to_dict("records"), projected.to_dict("records"))):
        geom_m = row_m["geometry"]
        geom_ll = row_ll["geometry"]
        for poly_m, poly_ll in zip(iter_polygons(geom_m), iter_polygons(geom_ll)):
            area_km2 = poly_m.area / 1_000_000
            if not 0.2 <= area_km2 <= 120:
                continue
            minx, miny, maxx, maxy = poly_m.bounds
            aspect = max(maxx - minx, 1.0) / max(maxy - miny, 1.0)
            compact = min(aspect, 1 / max(aspect, 0.01))
            area_score = band_score(area_km2, 0.8, 45, 20)
            score = area_score * compact
            if score < 0.18:
                continue
            center = poly_ll.centroid
            api_bbox = crop_bbox_for_polygon(poly_ll, scale.crop_scale)
            chip_id = f"dian_{idx}_lon{center.x:.3f}_lat{center.y:.3f}"
            rel = Path("dian") / f"{chip_id}.png"
            candidates.append(
                ScaledStrokeCandidate(
                    chip_id=chip_id,
                    stroke_type="dian",
                    score=float(score),
                    source_index=idx,
                    target_river=source_name(row_ll),
                    water_source="HydroLAKES",
                    length_km=0.0,
                    area_km2=float(area_km2),
                    straightness=0.0,
                    angle_deg=0.0,
                    start_angle_deg=0.0,
                    end_angle_deg=0.0,
                    corner_angle_deg=0.0,
                    turn_angle_deg=0.0,
                    aspect_ratio=float(aspect),
                    scale_profile={
                        "target_aspect": scale.target_aspect,
                        "render_size": scale.render_size,
                        "crop_scale": scale.crop_scale,
                        "relative_role": "small-dot",
                    },
                    bbox=tuple(float(v) for v in poly_ll.bounds),
                    center_lon=float(center.x),
                    center_lat=float(center.y),
                    api_bbox=api_bbox,
                    geometry_wkt=poly_ll.wkt,
                    preview_path=str(output_root / rel),
                    overlay_path=str(output_root / rel),
                )
            )
            if len(candidates) >= candidate_limit:
                return candidates
    return candidates


def distance_km(a: ScaledStrokeCandidate, b: ScaledStrokeCandidate) -> float:
    _, _, meters = GEOD.inv(a.center_lon, a.center_lat, b.center_lon, b.center_lat)
    return meters / 1000


def diverse_top(rows: list[ScaledStrokeCandidate], max_items: int, radius_km: float) -> list[ScaledStrokeCandidate]:
    picked: list[ScaledStrokeCandidate] = []
    for row in sorted(rows, key=lambda item: item.score, reverse=True):
        if any(distance_km(row, existing) < radius_km for existing in picked):
            continue
        picked.append(row)
        if len(picked) >= max_items:
            return picked
    for row in sorted(rows, key=lambda item: item.score, reverse=True):
        if row not in picked:
            picked.append(row)
        if len(picked) >= max_items:
            break
    return picked


def draw_candidate(candidate: ScaledStrokeCandidate, path: Path) -> None:
    scale = STROKE_SCALES[candidate.stroke_type]
    geom = wkt.loads(candidate.geometry_wkt)
    minx, miny, maxx, maxy = candidate.api_bbox
    width, height = scale.render_size

    def xy(lon: float, lat: float) -> tuple[int, int]:
        x = int((lon - minx) / max(maxx - minx, 1e-9) * (width - 28) + 14)
        y = int((maxy - lat) / max(maxy - miny, 1e-9) * (height - 48) + 14)
        return x, y

    img = Image.new("RGB", scale.render_size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, width - 8, height - 34), outline=(205, 210, 216))
    if isinstance(geom, (Polygon, MultiPolygon)):
        for poly in iter_polygons(geom):
            pts = [xy(x, y) for x, y in poly.exterior.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=(46, 144, 205), outline=(9, 90, 150))
    else:
        for line in iter_lines(geom):
            pts = [xy(x, y) for x, y in line.coords]
            if len(pts) >= 2:
                line_width = max(4, min(width, height) // 28)
                draw.line(pts, fill=(0, 86, 150), width=line_width + 4, joint="curve")
                draw.line(pts, fill=(29, 153, 218), width=line_width, joint="curve")
    label = f"{candidate.stroke_type} {candidate.water_source} score={candidate.score:.2f}"
    draw.text((10, height - 25), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_gallery(selected: dict[str, list[ScaledStrokeCandidate]], output_root: Path) -> None:
    css = """body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f6f7;color:#161616}.top{position:sticky;top:0;z-index:2;background:white;border-bottom:1px solid #ddd;padding:14px 18px}.group{padding:16px 18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}.card{background:white;border:1px solid #ddd;border-radius:6px;padding:8px}.card img{width:100%;height:220px;object-fit:contain;background:#eef1f4;display:block}.meta{font-size:12px;line-height:1.45;color:#333}.id{font-weight:700;color:#111}.badge{display:inline-block;margin:6px 6px 0 0;padding:2px 8px;border:1px solid #ddd;border-radius:999px;background:#eee;font-size:12px}.bbox{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;word-break:break-all;color:#555}"""
    parts = ["<!doctype html><meta charset='utf-8'><title>Scaled Water Stroke Candidates</title>", f"<style>{css}</style>"]
    parts.append("<div class='top'><h2 style='margin:0'>中国水系尺度化笔画候选</h2>")
    for stroke, rows in sorted(selected.items()):
        parts.append(f"<span class='badge'>{html.escape(stroke)}: {len(rows)}</span>")
    parts.append("</div>")
    for stroke, rows in sorted(selected.items()):
        parts.append(f"<section class='group'><h3>{html.escape(stroke)}</h3><div class='grid'>")
        for rank, cand in enumerate(rows, 1):
            img_rel = Path(cand.preview_path).relative_to(output_root)
            bbox = ", ".join(f"{v:.5f}" for v in cand.api_bbox)
            parts.append("<div class='card'>")
            parts.append(f"<img src='{html.escape(str(img_rel))}' loading='lazy'>")
            parts.append("<div class='meta'>")
            parts.append(f"<div class='id'>{rank:03d} {html.escape(cand.chip_id)}</div>")
            parts.append(f"source={html.escape(cand.water_source)} score={cand.score:.3f} len={cand.length_km:.1f}km area={cand.area_km2 or 0:.1f}km2<br>")
            parts.append(f"angle={cand.angle_deg:.1f} turn={cand.turn_angle_deg:.1f} aspect={cand.aspect_ratio:.2f}<br>")
            parts.append(f"<div class='bbox'>api_bbox={html.escape(bbox)}</div>")
            parts.append("</div></div>")
        parts.append("</div></section>")
    (output_root / "gallery.html").write_text("\n".join(parts), encoding="utf-8")


def write_outputs(candidates: list[ScaledStrokeCandidate], output_root: Path, max_per_type: int, radius_km: float) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[ScaledStrokeCandidate]] = {}
    for cand in candidates:
        by_type.setdefault(cand.stroke_type, []).append(cand)
    selected = {stroke: diverse_top(rows, max_per_type, radius_km) for stroke, rows in by_type.items()}

    manifest_rows = []
    for stroke, rows in sorted(selected.items()):
        type_dir = output_root / stroke
        type_dir.mkdir(parents=True, exist_ok=True)
        with (type_dir / f"{stroke}_top.jsonl").open("w", encoding="utf-8") as f:
            for rank, cand in enumerate(rows, 1):
                cand.preview_path = str(type_dir / f"{stroke}_{rank:03d}.png")
                cand.overlay_path = cand.preview_path
                draw_candidate(cand, Path(cand.preview_path))
                row = asdict(cand) | {
                    "rank": rank,
                    "target_river": cand.target_river,
                    "candidate": asdict(cand),
                }
                manifest_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    make_gallery(selected, output_root)


def main() -> None:
    args = parse_args()
    strokes = parse_strokes(args.strokes)
    bbox = tuple(args.bbox)
    candidates = mine_river_candidates(
        args.river_vector,
        bbox,
        strokes,
        args.output_root,
        args.candidate_limit,
        args.min_discharge_cms,
        args.max_river_features,
        not args.no_merge_main_rivers,
    )
    if "dian" in strokes:
        candidates.extend(mine_lake_dots(args.lake_vector, bbox, args.output_root, args.candidate_limit))
    write_outputs(candidates, args.output_root, args.max_per_type, args.diversity_radius_km)
    print(f"river_vector={args.river_vector}")
    print(f"lake_vector={args.lake_vector} exists={args.lake_vector.exists()}")
    print(f"candidates={len(candidates)}")
    print(f"output={args.output_root}")
    for stroke in strokes:
        print(f"{stroke}={sum(c.stroke_type == stroke for c in candidates)}")


if __name__ == "__main__":
    main()
