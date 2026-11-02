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

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(
        f"Video: {n_frames} frames, {width}x{height}, {fps:.2f} FPS "
        f"({n_frames / fps:.1f}s). Loading {model_name}...",
        file=sys.stderr,
        flush=True,
    )
    model = YOLO(model_name)
    frames: list[Frame] = []
    started = time.perf_counter()
    last_report = started
    print("Starting tracking; waiting for the first frame...", file=sys.stderr, flush=True)
    stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=tracker,
        conf=conf,
        classes=[0],  # person
        device=device,
        verbose=False,
    )
    for i, r in enumerate(stream):
        players: list[PlayerFrame] = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            boxes = r.boxes.xyxy.tolist()
            kps = r.keypoints.data.tolist()
            for tid, box, kp in zip(ids, boxes, kps):
                players.append(
                    PlayerFrame(
                        track_id=tid,
                        bbox=tuple(box),
                        keypoints=[tuple(p) for p in kp],
                    )
                )
        frames.append(Frame(index=i, t=i / fps, players=players))
        now = time.perf_counter()
        processed = i + 1
        if processed == 1 or now - last_report >= 5 or processed == n_frames:
            rate = processed / max(now - started, 0.001)
            if processed == 1:
                total = f"/{n_frames}" if n_frames > 0 else ""
                progress = f"1{total} frames; first frame ready, estimating speed..."
            elif n_frames > 0:
                remaining = max(n_frames - processed, 0) / rate
                progress = (
                    f"{processed}/{n_frames} frames ({processed / n_frames:.1%}), "
                    f"{rate:.1f} frames/s, ~{remaining / 60:.1f} min remaining"
                )
            else:
                progress = f"{processed} frames, {rate:.1f} frames/s"
            print(progress, file=sys.stderr, flush=True)
            last_report = now

    return ClipResult(
        video=VideoMeta(
            source=video_path.name, fps=fps, width=width, height=height, n_frames=n_frames
        ),
        frames=frames,
    )
