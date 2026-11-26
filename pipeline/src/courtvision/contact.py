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
