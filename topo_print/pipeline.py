from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .elevation import elevation_grid
from .geo import Square
from .mesh import column_mesh
from .model import PrintSpec, TerrainModel, build_model, default_band_edges
from .preview import render_preview
from .threemf import write_colored_3mf
from .water import fetch_lakes, fetch_waterways, lake_mask, waterway_mask

DEFAULT_BAND_COLORS = ["#3f6b3a", "#8a6f4e", "#9a968e", "#f4f4f2"]
DEFAULT_WATER_COLOR = "#2f6fb5"
A1_PLATE_CENTER_MM = 128
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


@dataclass
class Params:
    lat: float = 42.7775
    lon: float = -109.2155
    size_km: float = 6
    print_mm: float = 150
    cell_mm: float = 0.4
    base_mm: float = 3
    exaggeration: float = 1.5
    bands: list[float] | None = None
    band_count: int = 3
    colors: list[str] | None = None
    river_width_mm: float = 1.2
    stream_width_mm: float = 0.8
    min_lake_mm2: float = 2
    recess_mm: float = 0.4
    water_depth_mm: float = 1.2
    include_waterways: bool = True
    contour_interval_m: float = 0
    min_layer_mm2: float = 3
    name: str = "terrain"


@dataclass
class Build:
    params: Params
    model: TerrainModel
    cell_mm: float
    elev_m: np.ndarray
    band_edges_m: list[float]
    band_colors: list[str]
    water_color: str
    lake_count: int


def _colors(params: Params, band_count):
    picks = np.linspace(0, len(DEFAULT_BAND_COLORS) - 1, band_count).round().astype(int)
    defaults = [DEFAULT_BAND_COLORS[k] for k in picks]
    if params.colors and len(params.colors) == band_count + 1:
        return params.colors[:-1], params.colors[-1]
    return defaults, DEFAULT_WATER_COLOR


def _water_masks(params: Params, square, cells, cell_mm):
    min_lake_cells = params.min_lake_mm2 / cell_mm**2
    lake_masks = [lake_mask(lake, square, cells) for lake in fetch_lakes(square, CACHE_DIR)]
    lake_masks = [m for m in lake_masks if m.sum() >= min_lake_cells]
    if not params.include_waterways:
        return lake_masks, np.zeros((cells, cells), dtype=bool)

    widths_mm = {"river": params.river_width_mm, "stream": params.stream_width_mm}
    widths_cells = {kind: max(1, round(mm / cell_mm)) for kind, mm in widths_mm.items() if mm > 0}
    waterways = fetch_waterways(square, CACHE_DIR)
    return lake_masks, waterway_mask(waterways, widths_cells, square, cells)


TERRACE_SMOOTHING_CELLS = 1.5


def _gaussian_blur(grid, sigma):
    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    rows, cols = grid.shape
    padded = np.pad(grid, radius, mode="edge")
    across = sum(w * padded[:, k : k + cols] for k, w in enumerate(kernel))
    return sum(w * across[k : k + rows, :] for k, w in enumerate(kernel))


def _merge_small_regions(levels, min_cells, direction, water):
    """Move every land region at or beyond a level that is smaller than min_cells one level back."""
    order = range(levels.max(), levels.min(), -1) if direction < 0 else range(levels.min(), levels.max())
    for level in order:
        region = (levels >= level if direction < 0 else levels <= level) & ~water
        labels, _ = ndimage.label(region)
        small = np.bincount(labels.ravel()) < min_cells
        small[0] = False
        levels[small[labels]] = level + direction
    return levels


def terrace(elev_m, interval_m, min_cells, water):
    if interval_m <= 0:
        return elev_m
    levels = np.floor(_gaussian_blur(elev_m, TERRACE_SMOOTHING_CELLS) / interval_m).astype(int)
    levels = _merge_small_regions(levels, min_cells, direction=-1, water=water)
    levels = _merge_small_regions(levels, min_cells, direction=+1, water=water)
    return levels * interval_m


def snap_to_layers(edges_m, interval_m):
    if interval_m <= 0:
        return edges_m
    return [float(round(edge / interval_m) * interval_m) for edge in edges_m]


def build(params: Params) -> Build:
    cells = round(params.print_mm / params.cell_mm)
    cell_mm = params.print_mm / cells
    size_m = params.size_km * 1000
    square = Square.around(params.lat, params.lon, size_m, min_px=cells * 1.5)

    elev_m = elevation_grid(square, cells, CACHE_DIR)
    lakes, flowing = _water_masks(params, square, cells, cell_mm)

    band_edges_m = sorted(params.bands) if params.bands else default_band_edges(elev_m, params.band_count)
    band_edges_m = snap_to_layers(band_edges_m, params.contour_interval_m)
    band_colors, water_color = _colors(params, len(band_edges_m) + 1)
    spec = PrintSpec(params.print_mm, params.base_mm, params.exaggeration, params.recess_mm, params.water_depth_mm)
    min_cells = params.min_layer_mm2 / cell_mm**2
    water = np.logical_or.reduce([flowing, *lakes])
    surface_m = terrace(elev_m, params.contour_interval_m, min_cells, water)
    model = build_model(surface_m, size_m, lakes, flowing, band_edges_m, band_colors, water_color, spec, min_cells)
    return Build(params, model, cell_mm, elev_m, band_edges_m, band_colors, water_color, len(lakes))


def export(result: Build, out_dir: Path) -> list[Path]:
    name = result.params.name
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, files = [], []
    for body in result.model.bodies:
        mesh = column_mesh(body.bot, body.top, result.cell_mm)
        stl = out_dir / f"{name}-{body.name}.stl"
        mesh.export(stl)
        parts.append((body.name, body.color, mesh))
        files.append(stl)

    project = out_dir / f"{name}.3mf"
    write_colored_3mf(project, name, parts, A1_PLATE_CENTER_MM, result.params.print_mm)
    preview = out_dir / f"{name}-preview.png"
    render_preview(result.model, result.band_colors, result.water_color, result.cell_mm, preview)
    return [project, preview, *files]
