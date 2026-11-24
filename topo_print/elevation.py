import io
import math
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .geo import TILE_SIZE, Square

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _tile_png(z, x, y, cache_dir: Path) -> bytes:
    path = cache_dir / "terrarium" / str(z) / str(x) / f"{y}.png"
    if not path.exists():
        response = requests.get(TERRARIUM_URL.format(z=z, x=x, y=y), timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path.read_bytes()


def _decode_terrarium(png: bytes) -> np.ndarray:
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def elevation_grid(square: Square, cells: int, cache_dir: Path) -> np.ndarray:
    """Elevation in meters, resampled to a cells x cells grid (row 0 = north)."""
    tx0 = math.floor(square.left / TILE_SIZE)
    ty0 = math.floor(square.top / TILE_SIZE)
    tx1 = math.floor((square.left + square.size_px) / TILE_SIZE)
    ty1 = math.floor((square.top + square.size_px) / TILE_SIZE)

    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE_SIZE, (tx1 - tx0 + 1) * TILE_SIZE), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = _decode_terrarium(_tile_png(square.zoom, tx, ty, cache_dir))
            r, c = (ty - ty0) * TILE_SIZE, (tx - tx0) * TILE_SIZE
            mosaic[r : r + TILE_SIZE, c : c + TILE_SIZE] = tile

    left = square.left - tx0 * TILE_SIZE
    top = square.top - ty0 * TILE_SIZE
    box = (left, top, left + square.size_px, top + square.size_px)
    resampled = Image.fromarray(mosaic, mode="F").resize((cells, cells), Image.BILINEAR, box=box)
    return np.asarray(resampled, dtype=np.float64)
