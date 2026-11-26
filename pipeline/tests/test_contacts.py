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
    players[1].bbox = (50, 0, 100, 200)
    assert len(single(players)) == 1  # target height 200: threshold 16


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((50, 0, 100, 100), 1),  # shared edge
        ((45, 0, 100, 100), 1),  # overlap
        ((50.01, 0, 100, 100), 0),
        ((50, 100.01, 100, 200), 0),
        ((50, 0, 100, 0), 0),  # no height
        ((100, 0, 50, 100), 0),  # inverted width
        ((50, 0, math.inf, 100), 0),
        ((math.nan, 0, 100, 100), 0),
    ],
)
def test_boxes_must_be_valid_and_touch(bbox, expected):
    players = pair()
    players[1].bbox = bbox
    assert len(single(players)) == expected


def test_corner_touch_is_inclusive():
    players = pair(0)
    players[0].keypoints[9] = (50, 100, 1)
    players[1].bbox = (50, 100, 100, 200)
    players[1].keypoints[5] = (50, 100, 1)
    players[1].keypoints[11] = (50, 180, 1)
    assert len(single(players)) == 1


@pytest.mark.parametrize(("player_index", "joint"), [(0, 9), (1, 5), (1, 11)])
@pytest.mark.parametrize("confidence", [0, 0.3, math.nan, math.inf, 1.01])
def test_wrist_and_both_endpoints_need_confidence(player_index, joint, confidence):
    players = pair()
    x, y, _ = players[player_index].keypoints[joint]
    players[player_index].keypoints[joint] = (x, y, confidence)
    assert not single(players)


@pytest.mark.parametrize(("player_index", "joint"), [(0, 9), (1, 5), (1, 11)])
@pytest.mark.parametrize("coordinate", [math.nan, math.inf])
def test_nonfinite_keypoints_are_ignored(player_index, joint, coordinate):
    players = pair()
    _, y, confidence = players[player_index].keypoints[joint]
    players[player_index].keypoints[joint] = (coordinate, y, confidence)
    assert not single(players)


@pytest.mark.parametrize("edge", BODY_SEGMENTS)
def test_torso_arm_and_leg_segments_are_checked(edge):
    players = pair()
    players[1].keypoints = [(0, 0, 0)] * 17
    a, b = edge
    players[1].keypoints[a], players[1].keypoints[b] = (50, 20, 1), (50, 80, 1)
    assert len(single(players)) == 1


@pytest.mark.parametrize("wrist", [9, 10])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_both_wrists_and_both_player_directions(wrist, reverse_order):
    players = pair()
    players[0].keypoints[9] = (0, 0, 0)
    players[0].keypoints[wrist] = (45, 50, 1)
    if reverse_order:
        players.reverse()
    assert single(players)[0].track_ids == (1, 2)


def test_only_wrists_start_a_hit_and_face_edges_are_excluded():
    players = pair()
    players[0].keypoints[7], players[0].keypoints[9] = (45, 50, 1), (0, 0, 0)
    assert not single(players)
    players = pair()
    players[1].keypoints = [(0, 0, 0)] * 17
    players[1].keypoints[0], players[1].keypoints[1] = (50, 20, 1), (50, 80, 1)
    assert not single(players)


def test_multiple_witnesses_still_count_once_per_pair_per_frame():
    players = pair()
    players[0].keypoints[10] = (45, 51, 1)
    players[1].keypoints[6] = (50, 20, 1)
    assert single(players)[0].hit_frames == 1
    assert not detect(clip([Frame(index=0, t=0, players=players)]), min_hit_frames=2)


def test_same_track_id_never_contacts_itself():
    players = pair()
    players[1].track_id = players[0].track_id
    assert not single(players)

