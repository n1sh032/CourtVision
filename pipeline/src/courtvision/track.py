"""Detection + tracking + pose via Ultralytics (needs the `ml` extra)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from .schema import ClipResult, Frame, PlayerFrame, VideoMeta


def run(
    video_path: str | Path,
    model_name: str = "yolo11n-pose.pt",
    conf: float = 0.25,
    tracker: str = "bytetrack.yaml",
    device: str | None = None,
) -> ClipResult:
    import cv2
    from ultralytics import YOLO

