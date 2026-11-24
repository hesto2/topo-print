import argparse
from dataclasses import fields
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .pipeline import Params, build, export


def _floats(text):
    return [float(v) for v in text.split(",")]


def _strings(text):
    return text.split(",")


LIST_PARAMS = {"bands": float, "colors": str}


def _parse_value(field, raw):
    if field.name in LIST_PARAMS:
        return [LIST_PARAMS[field.name](v) for v in raw.split(",")]
    if field.type is bool:
        return raw == "true"
    return field.type(raw)


def params_from_url(url, name):
    """Params from an editor share link; settings missing from the link keep the editor defaults."""
    query = parse_qs(urlsplit(url).query)
    values = {f.name: _parse_value(f, query[f.name][0]) for f in fields(Params) if f.name in query}
    return Params(**{**values, "name": name})


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a multi-color 3D-printable terrain square.")
    parser.add_argument("--url", help="editor share link; replaces all other model settings")
    parser.add_argument("--lat", type=float, default=42.7775, help="center latitude")
    parser.add_argument("--lon", type=float, default=-109.2155, help="center longitude")
    parser.add_argument("--size-km", type=float, default=6, help="ground width of the square")
    parser.add_argument("--print-mm", type=float, default=150, help="printed width of the square")
    parser.add_argument("--cell-mm", type=float, default=0.4, help="horizontal resolution (~nozzle width)")
    parser.add_argument("--base-mm", type=float, default=3, help="thickness below the lowest point")
    parser.add_argument("--exaggeration", type=float, default=1.5, help="vertical exaggeration")
    parser.add_argument("--bands", type=_floats, help="comma separated elevation boundaries in meters")
    parser.add_argument("--band-count", type=int, default=3, help="terrain colors when --bands is omitted")
    parser.add_argument("--colors", type=_strings, help="comma separated hex colors: bands..., water")
    parser.add_argument("--river-width-mm", type=float, default=1.2)
    parser.add_argument("--stream-width-mm", type=float, default=0.8, help="0 leaves out unnamed streams")
    parser.add_argument("--min-lake-mm2", type=float, default=2, help="skip lakes smaller than this on the print")
    parser.add_argument("--recess-mm", type=float, default=0.4, help="how far water sits below the shore")
    parser.add_argument("--water-depth-mm", type=float, default=1.2, help="thickness of the water body")
    parser.add_argument("--no-waterways", dest="include_waterways", action="store_false", help="leave out rivers and streams")
    parser.add_argument("--contour-interval-m", type=float, default=0, help="terrace height in meters (0 = smooth)")
    parser.add_argument("--min-layer-mm2", type=float, default=3, help="merge layer pieces smaller than this")
    parser.add_argument("--name", default="terrain")
    parser.add_argument("--out", type=Path, default=Path("out"))
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.url:
        params = params_from_url(args.url, args.name)
    else:
        params = Params(**{f.name: getattr(args, f.name) for f in fields(Params)})
    result = build(params)
    print(f"elevation {result.elev_m.min():.0f} m - {result.elev_m.max():.0f} m")
    print(f"water: {result.lake_count} lakes")

    files = export(result, args.out)
    for slot, body in enumerate(result.model.bodies, start=1):
        print(f"  filament {slot}: {body.name:6} {body.color}")
    print(f"max height {np.nanmax(result.model.surface_mm):.1f} mm")
    for edge_m, edge_mm in zip(result.band_edges_m, result.model.band_edges_mm):
        print(f"  color boundary {edge_m:.0f} m -> z = {edge_mm:.2f} mm")
    print(f"wrote {files[0].resolve()} (+ per-color STLs and preview png)")


if __name__ == "__main__":
    main()
