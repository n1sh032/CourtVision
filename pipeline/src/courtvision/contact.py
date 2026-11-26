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
