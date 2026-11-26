"""Synthetic geometry and event tests; no footage or ML dependencies required."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from courtvision.cli import main
from courtvision.contact import BODY_SEGMENTS, detect, point_segment_distance
from courtvision.schema import ClipResult, ContactEvent, Frame, PlayerFrame, VideoMeta
from pydantic import ValidationError


def pair(distance: float = 5) -> list[PlayerFrame]:
    wrist = [(0.0, 0.0, 0.0)] * 17
    body = wrist.copy()
    wrist[9] = (50 - distance, 50, 1)
    body[5], body[11] = (50, 20, 1), (50, 80, 1)
    return [
        PlayerFrame(track_id=1, bbox=(0, 0, 50, 100), keypoints=wrist),
        PlayerFrame(track_id=2, bbox=(50, 0, 100, 100), keypoints=body),
    ]


def clip(frames: list[Frame]) -> ClipResult:
    return ClipResult(
        video=VideoMeta(
            source="synthetic.mp4",
            fps=30,
            width=640,
            height=360,
            n_frames=max((frame.index for frame in frames), default=-1) + 1,
        ),
        frames=frames,
    )


def sequence(hits: set[int], indices: list[int] | None = None) -> ClipResult:
    if indices is None:
        indices = list(range(max(hits, default=-1) + 1))
    return clip(
        [Frame(index=i, t=i / 30, players=pair() if i in hits else []) for i in indices]
    )


def single(players: list[PlayerFrame], **kwargs: object) -> list[ContactEvent]:
    return detect(
        clip([Frame(index=0, t=0, players=players)]), min_hit_frames=1, **kwargs
    )


@pytest.mark.parametrize(
    ("point", "start", "end", "expected"),
    [
        ((3, 5), (0, 0), (0, 10), 3),  # perpendicular projection
        ((3, -4), (0, 0), (0, 10), 5),  # before the first endpoint
        ((3, 14), (0, 0), (0, 10), 5),  # beyond the last endpoint
        ((5, 5), (0, 0), (10, 10), 0),
        ((3, 4), (0, 0), (0, 0), 5),  # collapsed segment
        ((-3, -4), (-6, -8), (0, 0), 0),
    ],
)
def test_point_segment_distance(point, start, end, expected):
    assert point_segment_distance(point, start, end) == pytest.approx(expected)


@pytest.mark.parametrize(("distance", "expected"), [(0, 1), (8, 1), (8.01, 0)])
def test_inclusive_distance_threshold(distance, expected):
    assert len(single(pair(distance))) == expected


def test_zero_distance_threshold_is_supported():
    assert len(single(pair(0), distance_ratio=0)) == 1
    assert not single(pair(0.01), distance_ratio=0)


def test_distance_scales_with_target_height_not_wrist_owner():
    players = pair(10)
    players[0].bbox = (0, 0, 50, 1000)
    assert not single(players)  # target height 100: threshold 8, not 80
