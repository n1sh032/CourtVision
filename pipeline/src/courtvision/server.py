"""Serve the review UI and run one local detection job at a time."""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from .application import DetectionService

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
    service = DetectionService(model, device, classifier)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") not in {
                f"courtvision.localhost:{self.server.server_port}",
                f"127.0.0.1:{self.server.server_port}",
            }:
                self.send_error(403, "Use the address printed by CourtVision")
                return
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
            host = self.headers.get("Host")
            allowed = {
                f"courtvision.localhost:{self.server.server_port}",
                f"127.0.0.1:{self.server.server_port}",
            }
            if host not in allowed or origin != f"http://{host}":
                self.send_error(403, "Open the viewer using the address printed by CourtVision")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_CLIP_BYTES:
                self.send_error(413, "Choose a clip no larger than 200 MiB")
                return
            if not job.acquire(blocking=False):
                self.send_error(409, "Detection is already running")
                return
            try:
                self.run_detection(size)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                job.release()

        def send_event(self, event):
            self.wfile.write(json.dumps(event, separators=(",", ":")).encode() + b"\n")
            self.wfile.flush()

        def run_detection(self, size):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with TemporaryDirectory(prefix="courtvision-") as folder:
                    source = Path(folder) / "clip.mp4"
                    self.connection.settimeout(60)
                    with source.open("wb") as output:
                        remaining = size
                        while remaining:
                            chunk = self.rfile.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise ValueError("Clip upload ended early")
                            output.write(chunk)
                            remaining -= len(chunk)

                    result = service.run(
                        source, self.path.endswith("profile=broad"), self.send_event,
                    )
                    result.frames = frames
                    self.send_event({"status": "Finding contact candidates…"})
                    if self.path.endswith("profile=broad"):
                        result.contacts = detect(result)
                    elif classifier is None:
                        result.contacts = detect_for_review(result)
                    else:
                        from .contact_model import features

                        result.contacts = [
                            event for event in detect(result)
                            if classifier.accepts(features(result, event))
                        ]
                    self.wfile.write(
                        b'{"result":' + result.model_dump_json().encode() + b"}\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                self.send_event({"error": str(exc) or type(exc).__name__})
            except Exception:
                logger.exception("Unexpected detection failure")
                self.send_event({"error": "Unexpected error. See the CourtVision terminal."})

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{server.server_port} to run CourtVision.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
