from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd
import pyproj
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform

from rs_words.config import OSM_DIR, SEGMENT_LENGTH_METERS


def _to_utm(geom, lat: float, lon: float):
    """将 WGS84 几何投影到对应 UTM 带（米）。"""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    crs = f"EPSG:{epsg}"
    project = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    return transform(project, geom), crs


def _to_wgs84(geom, utm_crs: str):
    project = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform
    return transform(project, geom)


def download_rivers_for_basin(
    basin_name: str,
    bbox: Tuple[float, float, float, float],
    output_dir: Path = OSM_DIR,
) -> Path:
    """从 OSM 下载指定流域的河流线，保存为 GeoJSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{basin_name}_rivers.geojson"

    tags = {"waterway": ["river", "stream"]}
    gdf = ox.features.features_from_bbox(bbox=bbox, tags=tags)
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    gdf = gdf.to_crs("EPSG:4326")
    gdf["basin"] = basin_name
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def segment_line(line: LineString, segment_length_m: float) -> List[LineString]:
    """将一条 LineString 等距分段。"""
    if line.is_empty or line.length <= 1e-12:
        return []
    lat, lon = line.centroid.y, line.centroid.x
    line_utm, crs = _to_utm(line, lat, lon)
    length = line_utm.length
    if length <= segment_length_m:
        return [line]
    segments = []
    n = max(int(round(length / segment_length_m)), 1)
    step = length / n
    for i in range(n):
        start = i * step
        end = (i + 1) * step if i < n - 1 else length
        p1 = line_utm.interpolate(start)
        p2 = line_utm.interpolate(end)
        seg_utm = LineString([p1, p2])
        seg_wgs84 = _to_wgs84(seg_utm, crs)
        segments.append(seg_wgs84)
    return segments


def build_segment_catalog(
    basin_files: Dict[str, Path],
    output_path: Path = OSM_DIR / "river_segments.geojson",
    segment_length_m: float = SEGMENT_LENGTH_METERS,
) -> gpd.GeoDataFrame:
    """把多个流域的河流线合并并分段，输出 GeoDataFrame。"""
    rows = []
    for basin_name, path in basin_files.items():
        gdf = gpd.read_file(path)
        for _, row in gdf.iterrows():
            geom = row.geometry
            name = row.get("name", "unknown")
            osm_id = row.get("osmid", row.get("id", "unknown"))
            if geom is None:
                continue
            if isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                lines = [geom]
            for line in lines:
                if not isinstance(line, LineString):
                    continue
                for seg in segment_line(line, segment_length_m):
                    rows.append(
                        {
                            "basin": basin_name,
                            "name": str(name) if pd.notna(name) else "unknown",
                            "osm_id": str(osm_id),
                            "geometry": seg,
                        }
                    )
    gdf_out = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf_out["segment_id"] = [f"seg_{i:08d}" for i in range(len(gdf_out))]
    gdf_out.to_file(output_path, driver="GeoJSON")
    return gdf_out
