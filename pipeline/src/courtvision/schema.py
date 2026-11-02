"""JSON contract between the ML pipeline and the web app.

Keypoints use the COCO-17 layout. Coordinates are source-video pixels.
Bump SCHEMA_VERSION on any breaking change.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

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


class ContactEvent(BaseModel):
    """A candidate for one unordered track pair; endpoints are inclusive hit frames."""

    track_ids: tuple[int, int]
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_t: float = Field(ge=0, allow_inf_nan=False)
    end_t: float = Field(ge=0, allow_inf_nan=False)
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
    schema_version: str = SCHEMA_VERSION
    video: VideoMeta
    frames: list[Frame]
    contacts: list[ContactEvent] = Field(default_factory=list)
