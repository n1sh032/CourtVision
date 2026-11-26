"""Find 2D wrist-to-body proximity candidates in saved pose results.

This module requires only the base package: no video decoding or model inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from .schema import ClipResult, ContactEvent, Keypoint, PlayerFrame

Point = tuple[float, float]
TrackPair = tuple[int, int]

# COCO-17: torso, arms, and legs. Face keypoints are intentionally excluded.
BODY_SEGMENTS: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
WRISTS = (9, 10)
TORSO = (5, 6, 11, 12)
TORSO_SEGMENTS = ((5, 6), (5, 11), (6, 12), (11, 12))
WRIST_ELBOW = ((9, 7), (10, 8))


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Euclidean pixel distance to the finite segment, including its endpoints."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    fraction = max(0.0, min(1.0, projection))
    return math.hypot(
        point[0] - (start[0] + fraction * dx),
        point[1] - (start[1] + fraction * dy),
    )


def _valid_box(player: PlayerFrame) -> bool:
    x1, y1, x2, y2 = player.bbox
    return all(math.isfinite(v) for v in player.bbox) and x2 > x1 and y2 > y1


def _boxes_touch(a: PlayerFrame, b: PlayerFrame) -> bool:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def _usable(keypoint: Keypoint, min_conf: float) -> bool:
    return all(math.isfinite(v) for v in keypoint) and min_conf < keypoint[2] <= 1.0
