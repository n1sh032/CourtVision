"""Draw boxes, track IDs and skeletons onto a clip (needs the `ml` extra)."""
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from .schema import ClipResult, Frame, VideoMeta
from .source import fingerprint

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]  # fmt: skip


def render(video_path: str | Path, result: ClipResult, out_path: str | Path) -> None:
    render_frames(video_path, result.video, result.frames, out_path)


def render_frames(
    video_path: str | Path, v: VideoMeta, frames: Iterable[Frame], out_path: str | Path,
) -> None:
    import cv2

    video_path, out_path = Path(video_path), Path(out_path)
    if not video_path.is_file() or out_path.exists() or out_path.suffix.lower() != ".mp4":
        raise ValueError("Overlay needs an existing input file and a new .mp4 output")
    if v.source_sha256 and fingerprint(video_path) != v.source_sha256:
        raise ValueError("Input video does not match the pose JSON source")
    if not math.isfinite(v.fps) or v.fps <= 0 or min(v.width, v.height) <= 0:
        raise ValueError("Invalid video metadata in pose JSON")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), v.fps, (v.width, v.height)
    )
    try:
        if not writer.isOpened():
            raise ValueError(f"Cannot open MP4 writer: {out_path}")
        records = iter(frames)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if img.shape[:2] != (v.height, v.width):
                raise ValueError("Video dimensions differ from pose JSON")
            frame = next(records, None)
            if frame is None or frame.index != i:
                raise ValueError("Decoded frame count or order differs from pose JSON")
            if frame:
                for p in frame.players:
                    x1, y1, x2, y2 = map(int, p.bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(img, f"#{p.track_id}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                    for a, b in SKELETON:
                        (xa, ya, ca), (xb, yb, cb) = p.keypoints[a], p.keypoints[b]
                        if min(ca, cb) > 0.3:
                            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 0), 2)
            writer.write(img)
            i += 1
        if not i or i != len(result.frames):
            raise ValueError("Decoded frame count differs from pose JSON")
    finally:
        cap.release()
        writer.release()
