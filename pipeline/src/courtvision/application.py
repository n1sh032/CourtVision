"""Detection workflow shared by the local HTTP adapter."""
from pathlib import Path


class DetectionService:
    def __init__(self, model: str, device: str | None, classifier=None):
        self.model = model
        self.device = device
        self.classifier = classifier

    def run(self, source: Path, broad: bool, progress):
        from . import track
        from .contact import detect, detect_for_review

        frames = []

        def on_frame(frame):
            frames.append(frame)
            if frame.index == 0 or frame.index % 15 == 0:
                progress({"status": f"Processed {frame.index + 1} frames…"})

        progress({"status": "Loading pose model…"})
        result = track.run(source, model_name=self.model, device=self.device, on_frame=on_frame)
        result.frames = frames
