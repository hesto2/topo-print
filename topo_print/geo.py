import math
from dataclasses import dataclass

import numpy as np

TILE_SIZE = 256
MAX_ZOOM = 15
EARTH_CIRCUMFERENCE_M = 40075016.686


def world_px(lat, lon, zoom):
    scale = TILE_SIZE * 2**zoom
    lat_rad = np.radians(lat)
    x = (np.asarray(lon) + 180) / 360 * scale
    y = (1 - np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / math.pi) / 2 * scale
    return x, y


def world_latlon(x, y, zoom):
    scale = TILE_SIZE * 2**zoom
    lon = x / scale * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / scale))))
    return lat, lon


def meters_per_px(lat, zoom):
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE_SIZE * 2**zoom)


@dataclass(frozen=True)
class Square:
    """A ground-square area expressed in Web Mercator world pixels at a zoom level."""

    left: float
    top: float
    size_px: float
    zoom: int

    @classmethod
    def around(cls, lat, lon, size_m, min_px):
        zoom = next(
            (z for z in range(1, MAX_ZOOM + 1) if size_m / meters_per_px(lat, z) >= min_px),
            MAX_ZOOM,
        )
        size_px = size_m / meters_per_px(lat, zoom)
        cx, cy = world_px(lat, lon, zoom)
        return cls(float(cx) - size_px / 2, float(cy) - size_px / 2, size_px, zoom)

    def bbox(self):
        north, west = world_latlon(self.left, self.top, self.zoom)
        south, east = world_latlon(self.left + self.size_px, self.top + self.size_px, self.zoom)
        return south, west, north, east

    def to_grid(self, lat, lon, cells):
        x, y = world_px(lat, lon, self.zoom)
        per_cell = self.size_px / cells
        return (x - self.left) / per_cell, (y - self.top) / per_cell
