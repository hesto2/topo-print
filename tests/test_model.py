import numpy as np
import pytest

from topo_print.mesh import column_mesh
from topo_print.model import PrintSpec, build_model
from topo_print.pipeline import terrace

CELLS = 60
SPEC = PrintSpec(size_mm=24, base_mm=3, exaggeration=2, recess_mm=0.4, water_depth_mm=1.2)


def random_terrain(seed):
    rng = np.random.default_rng(seed)
    rough = rng.normal(size=(CELLS, CELLS)).cumsum(axis=0).cumsum(axis=1)
    return 3000 + 40 * (rough - rough.min()) / rough.std()


def random_water(seed):
    rng = np.random.default_rng(seed + 1000)
    lake = np.zeros((CELLS, CELLS), dtype=bool)
    r, c = rng.integers(10, 50, size=2)
    lake[r - 6 : r + 6, c - 4 : c + 8] = True
    river = np.zeros((CELLS, CELLS), dtype=bool)
    path = np.clip(np.cumsum(rng.integers(-1, 2, size=CELLS)) + CELLS // 2, 0, CELLS - 2)
    river[np.arange(CELLS), path] = river[np.arange(CELLS), path + 1] = True
    return [lake], river


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("interval_m", [0, 25])
def test_every_part_is_a_closed_manifold_solid(seed, interval_m):
    elev = random_terrain(seed)
    lakes, river = random_water(seed)
    surface = terrace(elev, interval_m, min_cells=6, water=lakes[0] | river)
    edges = np.quantile(surface, [0.33, 0.66]).tolist()
    model = build_model(surface, 3000, lakes, river, edges, ["#111111"] * 3, "#0000ff", SPEC, min_cells=6)

    for body in model.bodies:
        mesh = column_mesh(body.bot, body.top, SPEC.size_mm / CELLS)
        _, edge_uses = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        assert (edge_uses == 2).all(), f"{body.name}: non-manifold edges"
        assert mesh.is_winding_consistent and mesh.volume > 0, body.name


def test_parts_never_overlap():
    elev = random_terrain(3)
    lakes, river = random_water(3)
    model = build_model(elev, 3000, lakes, river, [3030, 3060], ["#111111"] * 3, "#0000ff", SPEC, min_cells=6)
    spans = [(np.where(np.isfinite(b.top), b.bot, np.nan), b.top) for b in model.bodies]
    for k, (bot_a, top_a) in enumerate(spans):
        for bot_b, top_b in spans[k + 1 :]:
            overlap = np.minimum(top_a, top_b) - np.maximum(bot_a, bot_b)
            assert not (np.nan_to_num(overlap, nan=-1) > 1e-9).any()
