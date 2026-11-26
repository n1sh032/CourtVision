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

