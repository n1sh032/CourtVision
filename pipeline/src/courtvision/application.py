"""Detection workflow shared by the local HTTP adapter."""
from pathlib import Path


class DetectionService:
    def __init__(self, model: str, device: str | None, classifier=None):
        self.model = model
        self.device = device
        self.classifier = classifier

    def run(self, source: Path, broad: bool, progress):
        from . import track
