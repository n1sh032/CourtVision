"""Draw boxes, track IDs and skeletons onto a clip (needs the `ml` extra)."""
from __future__ import annotations

from pathlib import Path

from .schema import ClipResult

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]  # fmt: skip


def render(video_path: str | Path, result: ClipResult, out_path: str | Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    v = result.video
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), v.fps, (v.width, v.height)
    )
    by_index = {f.index: f for f in result.frames}
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
