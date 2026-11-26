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


class ContactEvent(BaseModel):
    track_ids: tuple[int, int]
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_t: FiniteFloat = Field(ge=0)
    end_t: FiniteFloat = Field(ge=0)
    hit_frames: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_interval(self) -> ContactEvent:
        if self.track_ids[0] >= self.track_ids[1]:
            raise ValueError("track_ids must be distinct and in ascending order")
        if self.end_frame < self.start_frame or self.end_t < self.start_t:
            raise ValueError("event end must not precede its start")
        if self.hit_frames > self.end_frame - self.start_frame + 1:
            raise ValueError("hit_frames cannot exceed the event's frame span")
        return self

class ClipResult(BaseModel):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    video: VideoMeta
    frames: list[Frame]

    @model_validator(mode="after")
    def validate_order(self):
        previous_time = 0.0
        for index, frame in enumerate(self.frames):
            if frame.index != index or frame.t < previous_time:
                raise ValueError("Frame indices must be consecutive and times nondecreasing")
            previous_time = frame.t
        return self
