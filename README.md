# topo-print

Turn any square of US terrain into a multi-color 3D print: elevation bands in different colors, optional
stacked contour layers, and lakes and rivers inset in blue. Built for Bambu Lab printers with an AMS, but
the output is a standard 3MF that any slicer can read.

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python and dependencies for you)
- Internet access for elevation and water data (cached locally after the first download)

## Quick start: web editor

```sh
git clone https://github.com/hesto2/topo-print.git
cd topo-print
uv run topo-print-web
```

This opens `http://127.0.0.1:8765`, where you can:

1. **Pick the area**: click the map, paste a Google Maps link, or type lat/lon, then set the area width.
2. **Tweak the print**: print width, vertical exaggeration, base thickness, resolution.
3. **Add contour layers** (optional): step the terrain every 40–400 ft, like a stacked topo map.
   Tiny specks smaller than "Smallest land piece" are merged away.
4. **Set colors**: 1–4 terrain colors plus water. Color boundaries snap to contour layers.
5. **Water**: include or skip rivers and streams, set their widths, and hide tiny lakes.
6. **Export Bambu 3MF**: files are written to `out/<name>/`.

Every setting is kept in the URL, so **Copy share link** reproduces the exact model.

The first time you view a new area it can take about a minute while USGS water data downloads.

## Command line

```sh
# From an editor share link
uv run topo-print --url 'http://127.0.0.1:8765/?lat=42.78&lon=-109.26&size_km=8&contour_interval_m=30.48' --name cirque

# Or with flags
uv run topo-print --lat 43.10 --lon -109.62 --size-km 8 --print-mm 150 --contour-interval-m 30.48 --name titcomb
```

Run `uv run topo-print --help` for every option.

## Output

`out/<name>/` contains:

- `<name>.3mf`: one colored object per filament (terrain bands + water)
- `<name>-<part>.stl`: the same parts as STLs
- `<name>-preview.png`: a top-down preview

## Printing in Bambu Studio

1. Start a new project (**File → New Project**) so the model's colors become filaments 1–4.
2. Open the `.3mf`.
3. Click **OK** on "The 3mf file has invalid config, load geometry data only" (expected for non-Bambu 3MFs).
4. Click **Yes** to "load as a single object with multiple parts".
5. Load the matching colors in your AMS slots (or map them when sending the print).

## Data sources

- **Elevation**: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) (USGS 3DEP, ~10 m in the US)
- **Lakes, rivers, streams**: [USGS National Hydrography Dataset](https://www.usgs.gov/national-hydrography)
  (US only)

## How it works

Each grid cell (about one nozzle width) becomes a vertical column split into stacked, non-overlapping color
parts. Terrain is cut into elevation bands; lakes are leveled flat and rivers recessed slightly, then filled
with a water part. Diagonal-only contacts are bridged so every part is a closed, manifold solid.

## Development

```sh
uv run pytest
```

Inspired by [TouchTerrain](https://github.com/ChHarding/TouchTerrain_for_CAGEO),
[dem2stl](https://github.com/ansonl/dem2stl), and multi-color topo maps on MakerWorld.
