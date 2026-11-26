"""Extract event features and train a small classifier from reviewed clips."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat

from .contact import BODY_SEGMENTS, WRISTS, _usable, _valid_box, point_segment_distance
from .schema import ClipResult, ContactEvent


class Label(BaseModel):
    event_index: int = Field(ge=0)
    label: Literal["contact", "no_contact", "unclear"]


class Review(BaseModel):
    version: Literal[1]
    result: ClipResult
    labels: list[Label]


Vector = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


class ContactModel(BaseModel):
    version: Literal[1] = 1
    mean: Vector
    scale: Vector
    weight: Vector
    intercept: FiniteFloat

    def accepts(self, features: Vector) -> bool:
        return self.intercept + sum(
            w * (x - m) / (s or 1)
            for x, m, s, w in zip(features, self.mean, self.scale, self.weight)
        ) >= 0


def features(result: ClipResult, event: ContactEvent) -> Vector:
    distances = []
    height_ratios = []
    track_a, track_b = event.track_ids
    for frame in result.frames:
        if frame.index < event.start_frame:
            continue
        if frame.index > event.end_frame:
            break
        players = {p.track_id: p for p in frame.players if _valid_box(p)}
        if track_a not in players or track_b not in players:
            continue
