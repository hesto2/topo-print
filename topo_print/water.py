import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

from .geo import Square

NHD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/{layer}/query"
FLOWLINE_LAYER = 6
WATERBODY_LAYER = 12
LAKE_POND, RESERVOIR = 390, 436
PERENNIAL_STREAM = 46006
TILE_DEG = 0.05
PARALLEL_DOWNLOADS = 6


@dataclass
class Lake:
    rings: list[np.ndarray]


@dataclass
class Waterway:
    line: np.ndarray
    kind: str


def _tile_features(layer, tile, cache_dir: Path):
    tx, ty = tile
    path = cache_dir / "nhd-tiles" / str(layer) / f"{tx}_{ty}.json"
    if not path.exists():
        w, s = tx * TILE_DEG, ty * TILE_DEG
        params = {
            "where": "1=1",
            "geometry": f"{w},{s},{w + TILE_DEG},{s + TILE_DEG}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "f": "geojson",
        }
        features = _fetch_all_pages(NHD_URL.format(layer=layer), params)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(features))
        partial.replace(path)
    return json.loads(path.read_text())


def _tiles(bbox):
    s, w, n, e = bbox
    xs = range(math.floor(w / TILE_DEG), math.floor(e / TILE_DEG) + 1)
    ys = range(math.floor(s / TILE_DEG), math.floor(n / TILE_DEG) + 1)
    return [(x, y) for x in xs for y in ys]


def _query_nhd(layer, bbox, cache_dir: Path):
    """Features in bbox, downloaded and cached per fixed tile so nearby areas reuse downloads."""
    with ThreadPoolExecutor(PARALLEL_DOWNLOADS) as pool:
        pages = pool.map(lambda tile: _tile_features(layer, tile, cache_dir), _tiles(bbox))
        unique = {}
        for features in pages:
            for feature in features:
                unique[_properties(feature)["permanent_identifier"]] = feature
    return list(unique.values())


def _get_with_retries(url, params, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)


def _fetch_all_pages(url, params):
    features = []
    while True:
        page = _get_with_retries(url, {**params, "resultOffset": len(features)})
        features += page["features"]
        if not page.get("exceededTransferLimit") and not page.get("properties", {}).get("exceededTransferLimit"):
            return features


def _lines(geometry):
    coords = geometry["coordinates"]
    parts = [coords] if geometry["type"] == "LineString" else coords
    return [np.array(part)[:, :2] for part in parts]


def _polygons(geometry):
    coords = geometry["coordinates"]
    return [coords] if geometry["type"] == "Polygon" else coords


def _properties(feature):
    return {k.lower(): v for k, v in feature["properties"].items()}


def _waterway_kind(props):
    if props["ftype"] not in (460, 558):
        return None
    if props.get("gnis_name"):
        return "river"
    return "stream" if props["fcode"] == PERENNIAL_STREAM else None


def fetch_lakes(square: Square, cache_dir: Path):
    lakes = []
    for feature in _query_nhd(WATERBODY_LAYER, square.bbox(), cache_dir):
        if feature["geometry"] and _properties(feature)["ftype"] in (LAKE_POND, RESERVOIR):
            for polygon in _polygons(feature["geometry"]):
                lakes.append(Lake([np.array(ring)[:, :2] for ring in polygon]))
    return lakes


def fetch_waterways(square: Square, cache_dir: Path):
    waterways = []
    for feature in _query_nhd(FLOWLINE_LAYER, square.bbox(), cache_dir):
        kind = _waterway_kind(_properties(feature))
        if kind and feature["geometry"]:
            waterways += [Waterway(line, kind) for line in _lines(feature["geometry"])]
    return waterways


def _grid_points(lonlat, square, cells):
    col, row = square.to_grid(lonlat[:, 1], lonlat[:, 0], cells)
    return list(zip((col - 0.5).tolist(), (row - 0.5).tolist()))


def lake_mask(lake: Lake, square: Square, cells: int) -> np.ndarray:
    image = Image.new("1", (cells, cells), 0)
    draw = ImageDraw.Draw(image)
    outer, *holes = lake.rings
    draw.polygon(_grid_points(outer, square, cells), fill=1)
    for hole in holes:
        draw.polygon(_grid_points(hole, square, cells), fill=0)
    return np.asarray(image, dtype=bool)


def waterway_mask(waterways, widths_cells, square: Square, cells: int) -> np.ndarray:
    image = Image.new("1", (cells, cells), 0)
    draw = ImageDraw.Draw(image)
    for way in waterways:
        width = widths_cells.get(way.kind)
        if width and len(way.line) > 1:
            draw.line(_grid_points(way.line, square, cells), fill=1, width=width, joint="curve")
    return np.asarray(image, dtype=bool)
