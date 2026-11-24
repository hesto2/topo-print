from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .mesh import EMPTY


@dataclass(frozen=True)
class PrintSpec:
    size_mm: float
    base_mm: float
    exaggeration: float
    recess_mm: float
    water_depth_mm: float


@dataclass
class Body:
    name: str
    color: str
    bot: np.ndarray
    top: np.ndarray

    def is_empty(self):
        return not np.isfinite(self.top).any()


@dataclass
class TerrainModel:
    surface_mm: np.ndarray
    is_water: np.ndarray
    band_index: np.ndarray
    band_edges_mm: list[float]
    bodies: list[Body]


def _absorb_small_islands(water_top, min_cells):
    """Land specks smaller than min_cells take the level of the nearest water around them."""
    land = ~np.isfinite(water_top)
    labels, _ = ndimage.label(land)
    small = np.bincount(labels.ravel()) < min_cells
    small[0] = False
    specks = small[labels]
    if specks.any():
        _, nearest = ndimage.distance_transform_edt(land, return_indices=True)
        water_top[specks] = water_top[tuple(axis[specks] for axis in nearest)]
    return water_top


EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def _diagonal_saddles(grid):
    """2x2 blocks where one diagonal pair stands above both cells of the other pair.

    Columns meeting like that share a single vertical edge between four faces (non-manifold).
    Yields (high-diagonal min height, the two other cells) for every such block.
    """
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    rows, cols = np.indices(a.shape)
    for high_1, high_2, low_1, low_2, low_cells in (
        (a, d, b, c, ((0, 1), (1, 0))),
        (b, c, a, d, ((0, 0), (1, 1))),
    ):
        level = np.minimum(high_1, high_2)
        for i, j in zip(*np.nonzero(level > np.maximum(low_1, low_2))):
            yield level[i, j], [(rows[i, j] + di, cols[i, j] + dj) for di, dj in low_cells]


def _fill_saddles(grid, raisable):
    """Raise one in-between cell of every diagonal saddle until the columns form no pinch edges."""
    grid = grid.copy()
    while True:
        changed = False
        for level, cells in _diagonal_saddles(grid):
            candidates = sorted((c for c in cells if raisable[c]), key=lambda c: grid[c], reverse=True)
            if candidates and grid[candidates[0]] < level:
                grid[candidates[0]] = level
                changed = True
        if not changed:
            return grid


def _flat_bottoms(water_top, depth):
    """Every connected body of water gets one bottom, below its lowest surface."""
    labels, count = ndimage.label(np.isfinite(water_top), structure=EIGHT_CONNECTED)
    lowest = ndimage.minimum(water_top, labels, index=np.arange(1, count + 1))
    return np.where(labels > 0, np.concatenate([[np.nan], lowest])[labels] - depth, np.nan)


def build_model(elev_m, size_m, lakes, flowing, band_edges_m, band_colors, water_color, spec: PrintSpec, min_cells=0):
    mm_per_m = spec.size_mm / size_m * spec.exaggeration
    floor_m = elev_m.min()

    def to_mm(meters):
        return spec.base_mm + (np.asarray(meters, dtype=float) - floor_m) * mm_per_m

    surface = to_mm(elev_m)
    water_top = np.where(flowing, surface - spec.recess_mm, np.nan)
    for lake in lakes:
        if lake.any():
            water_top[lake] = np.percentile(surface[lake], 10) - spec.recess_mm

    if np.isfinite(water_top).any():
        water_top = _absorb_small_islands(water_top, min_cells)
    water_top = _fill_saddles(np.nan_to_num(water_top, nan=-np.inf), raisable=np.ones_like(surface, dtype=bool))
    water_top[np.isneginf(water_top)] = np.nan
    is_water = np.isfinite(water_top)
    water_bot = _flat_bottoms(water_top, spec.water_depth_mm)
    land_top = _fill_saddles(np.where(is_water, water_bot, surface), raisable=~is_water)

    band_edges_mm = to_mm(band_edges_m).tolist()
    edges = [0.0, *band_edges_mm, np.inf]
    bodies = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        top = np.minimum(land_top, hi)
        present = top > lo
        bodies.append(
            Body(f"band{k + 1}", band_colors[k], np.where(present, lo, EMPTY), np.where(present, top, EMPTY))
        )
    bodies.append(
        Body("water", water_color, np.where(is_water, water_bot, EMPTY), np.where(is_water, water_top, EMPTY))
    )

    return TerrainModel(
        surface_mm=np.where(is_water, water_top, surface),
        is_water=is_water,
        band_index=np.searchsorted(band_edges_mm, land_top, side="left"),
        band_edges_mm=band_edges_mm,
        bodies=[b for b in bodies if not b.is_empty()],
    )


def default_band_edges(elev_m, band_count):
    return np.quantile(elev_m, np.linspace(0, 1, band_count + 1)[1:-1]).round().tolist()
