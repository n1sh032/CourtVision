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

