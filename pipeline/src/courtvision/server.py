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
