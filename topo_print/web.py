import argparse
import json
import re
import traceback
import webbrowser
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from .pipeline import Build, Params, build, export

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"
SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
CONTENT_TYPES = {".html": "text/html", ".png": "image/png", ".stl": "model/stl", ".3mf": "model/3mf"}


def _params(payload) -> Params:
    known = {f.name for f in fields(Params)}
    params = Params(**{k: v for k, v in payload.items() if k in known})
    params.name = SAFE_NAME.sub("-", params.name).strip("-") or "terrain"
    return params


def _preview_payload(result: Build):
    model = result.model
    color_index = np.where(model.is_water, len(result.band_colors), model.band_index)
    return {
        "params": asdict(result.params),
        "cells": int(model.surface_mm.shape[0]),
        "cellMm": result.cell_mm,
        "heights": np.round(model.surface_mm, 2).ravel().tolist(),
        "colorIndex": color_index.astype(int).ravel().tolist(),
        "palette": [*result.band_colors, result.water_color],
        "bandEdgesM": [float(e) for e in result.band_edges_m],
        "bandEdgesMm": model.band_edges_mm,
        "elevationM": [float(result.elev_m.min()), float(result.elev_m.max())],
        "maxHeightMm": float(np.nanmax(model.surface_mm)),
        "mmPerM": result.params.print_mm / (result.params.size_km * 1000) * result.params.exaggeration,
        "lakes": result.lake_count,
        "filaments": [{"name": b.name, "color": b.color} for b in model.bodies],
    }


def _export_payload(result: Build):
    out_dir = OUT_DIR / result.params.name
    files = export(result, out_dir)
    return {
        "dir": str(out_dir),
        "files": [{"name": f.name, "url": f"/files/{out_dir.name}/{f.name}", "bytes": f.stat().st_size} for f in files],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html")
        if path.startswith("/files/"):
            target = (OUT_DIR / unquote(path.removeprefix("/files/"))).resolve()
            if target.is_relative_to(OUT_DIR) and target.is_file():
                return self._send_file(target, download=target.suffix != ".png")
        self.send_error(404)

    def do_POST(self):
        routes = {"/api/preview": _preview_payload, "/api/export": _export_payload}
        if self.path not in routes:
            return self.send_error(404)
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = routes[self.path](build(_params(payload)))
            self._send_json(200, body)
        except Exception as error:
            traceback.print_exc()
            self._send_json(400, {"error": str(error)})

    def _send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, download=False):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description="Web editor for topo-print models.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"topo-print editor running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
