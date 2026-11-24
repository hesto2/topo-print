from typing import NamedTuple

import numpy as np
import trimesh

EMPTY = np.inf
WALL_CHUNK = 100_000


class WallSide(NamedTuple):
    neighbor: tuple[int, int]
    start_corner: tuple[int, int]
    end_corner: tuple[int, int]


WALL_SIDES = [
    WallSide(neighbor=(0, 1), start_corner=(1, 1), end_corner=(0, 1)),
    WallSide(neighbor=(0, -1), start_corner=(0, 0), end_corner=(1, 0)),
    WallSide(neighbor=(-1, 0), start_corner=(0, 1), end_corner=(0, 0)),
    WallSide(neighbor=(1, 0), start_corner=(1, 0), end_corner=(1, 1)),
]


def column_mesh(bot: np.ndarray, top: np.ndarray, cell: float) -> trimesh.Trimesh:
    """Watertight solid made of one vertical column per grid cell spanning [bot, top].

    Cells with bot == top == EMPTY are skipped. Row 0 is north (+y).
    Walls are split at every height used around their corner lines so no T-junctions are left.
    """
    corner_z = _corner_breakpoints(bot, top)
    parts = [_horizontal_faces(top, cell, up=True), _horizontal_faces(bot, cell, up=False)]
    parts += [_walls(bot, top, corner_z, cell, side) for side in WALL_SIDES]
    triangles = np.concatenate(parts)
    return trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles), process=True)


def _corner_xy(ci, cj, rows, cell):
    return cj * cell, (rows - ci) * cell


def _pad(grid):
    return np.pad(grid, 1, constant_values=EMPTY)


def _corner_breakpoints(bot, top):
    heights = []
    for grid in (_pad(bot), _pad(top)):
        heights += [grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]]
    return np.sort(np.stack(heights, axis=-1), axis=-1)


def _quads_to_triangles(quads):
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def _horizontal_faces(z, cell, up):
    rows = z.shape[0]
    i, j = np.nonzero(np.isfinite(z))
    heights = z[i, j]
    corners = [(i + 1, j), (i + 1, j + 1), (i, j + 1), (i, j)]
    quad = np.stack(
        [np.stack([*_corner_xy(ci, cj, rows, cell), heights], axis=-1) for ci, cj in corners], axis=1
    )
    return _quads_to_triangles(quad if up else quad[:, ::-1])


def _exposed_spans(bot, top, neighbor_bot, neighbor_top):
    lower_hi = np.minimum(top, neighbor_bot)
    upper_lo = np.maximum(bot, neighbor_top)
    lower = np.nonzero(lower_hi > bot)
    upper = np.nonzero(top > upper_lo)
    i = np.concatenate([lower[0], upper[0]])
    j = np.concatenate([lower[1], upper[1]])
    zlo = np.concatenate([bot[lower], upper_lo[upper]])
    zhi = np.concatenate([lower_hi[lower], top[upper]])
    return i, j, zlo, zhi


def _walls(bot, top, corner_z, cell, side: WallSide):
    (di, dj), p_offset, q_offset = side
    rows, cols = bot.shape
    window = (slice(1 + di, 1 + di + rows), slice(1 + dj, 1 + dj + cols))
    i, j, zlo, zhi = _exposed_spans(bot, top, _pad(bot)[window], _pad(top)[window])
    chunks = [
        _wall_triangles(i[k], j[k], zlo[k], zhi[k], p_offset, q_offset, corner_z, rows, cell)
        for k in np.array_split(np.arange(len(i)), max(1, len(i) // WALL_CHUNK))
    ]
    return np.concatenate(chunks) if chunks else np.empty((0, 3, 3))


def _wall_triangles(i, j, zlo, zhi, p_offset, q_offset, corner_z, rows, cell):
    pi, pj = i + p_offset[0], j + p_offset[1]
    qi, qj = i + q_offset[0], j + q_offset[1]
    px, py = _corner_xy(pi, pj, rows, cell)
    qx, qy = _corner_xy(qi, qj, rows, cell)
    lo, hi = zlo[:, None], zhi[:, None]
    q_heights = np.clip(corner_z[qi, qj], lo, hi)
    p_heights = np.clip(corner_z[pi, pj], lo, hi)[:, ::-1]

    zs = np.hstack([lo, q_heights, hi, hi, p_heights, lo])
    half = zs.shape[1] // 2
    xs = np.hstack([np.repeat(qx[:, None], half, 1), np.repeat(px[:, None], half, 1)])
    ys = np.hstack([np.repeat(qy[:, None], half, 1), np.repeat(py[:, None], half, 1)])
    boundary = np.stack([xs, ys, zs], axis=-1)
    following = np.roll(boundary, -1, axis=1)
    center = np.stack([(px + qx) / 2, (py + qy) / 2, (zlo + zhi) / 2], axis=-1)

    keep = np.any(boundary != following, axis=-1)
    wall_index = np.nonzero(keep)[0]
    return np.stack([center[wall_index], boundary[keep], following[keep]], axis=1)
