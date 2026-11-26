"""Serve the review UI and run one local detection job at a time."""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

logger = logging.getLogger(__name__)
MAX_CLIP_BYTES = 200 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/viewer.js": ("viewer.js", "text/javascript"),
}


def serve(
    viewer: Path, port: int, model: str, device: str | None, contact_model: Path | None = None,
) -> None:
    if not all((viewer / name).is_file() for name, _ in ASSETS.values()):
        raise ValueError(f"Viewer files not found in {viewer}. Run from the repository root.")
    job = Lock()
    classifier = None
    if contact_model is not None:
        from .contact_model import ContactModel

        classifier = ContactModel.model_validate_json(contact_model.read_bytes())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asset = ASSETS.get(self.path)
            if not asset:
                self.send_error(404)
                return
            name, media_type = asset
            content = (viewer / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{media_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path not in {"/detect", "/detect?profile=broad"}:
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if origin != expected or self.headers.get("Host") != expected.removeprefix("http://"):
                self.send_error(403, "Open the viewer using the address printed by CourtVision")
                return
