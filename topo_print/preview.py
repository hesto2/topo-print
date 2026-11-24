import numpy as np
from PIL import Image, ImageColor

from .model import TerrainModel

LIGHT_FROM_NORTHWEST = np.array([-1.0, 1.0, 1.4]) / np.linalg.norm([-1.0, 1.0, 1.4])


def _hillshade(surface_mm, cell_mm):
    d_south, d_east = np.gradient(surface_mm, cell_mm)
    normals = np.stack([-d_east, d_south, np.ones_like(surface_mm)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.clip(normals @ LIGHT_FROM_NORTHWEST, 0, 1)


def render_preview(model: TerrainModel, band_colors, water_color, cell_mm, path, size_px=900):
    palette = np.array([ImageColor.getrgb(c) for c in [*band_colors, water_color]], dtype=float)
    color_index = np.where(model.is_water, len(band_colors), model.band_index)
    shade = 0.55 + 0.45 * _hillshade(model.surface_mm, cell_mm)
    rgb = palette[color_index] * shade[..., None]
    image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    image.resize((size_px, size_px), Image.NEAREST).save(path)
