"""JSON contract between the ML pipeline and the web app.

Keypoints use the COCO-17 layout. Coordinates are source-video pixels.
Bump SCHEMA_VERSION on any breaking change.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"

Keypoint = tuple[float, float, float]  # x, y, confidence


class VideoMeta(BaseModel):
    source: str
    fps: float
    width: int
    height: int
    n_frames: int


class PlayerFrame(BaseModel):
    track_id: int
    bbox: tuple[float, float, float, float]
    keypoints: list[Keypoint] = Field(min_length=17, max_length=17)


class Frame(BaseModel):
    index: int
    t: float  # seconds
    players: list[PlayerFrame]


class ClipResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video: VideoMeta
    frames: list[Frame]
