"""JSON contract between the ML pipeline and the web app.

Keypoints use the COCO-17 layout. Coordinates are source-video pixels.
Bump SCHEMA_VERSION on any breaking change.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat, model_validator

SCHEMA_VERSION = "0.1.0"

Keypoint = tuple[FiniteFloat, FiniteFloat, FiniteFloat]  # x, y, confidence


class VideoMeta(BaseModel):
    source: str
    source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    fps: FiniteFloat = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    n_frames: int = Field(ge=0)


class PlayerFrame(BaseModel):
    track_id: int
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    keypoints: list[Keypoint] = Field(min_length=17, max_length=17)


class Frame(BaseModel):
    index: int = Field(ge=0)
    t: FiniteFloat = Field(ge=0)  # seconds
    players: list[PlayerFrame]


class ClipResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video: VideoMeta
    frames: list[Frame]
