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
