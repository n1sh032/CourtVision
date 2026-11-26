"""Draw boxes, track IDs and skeletons onto a clip (needs the `ml` extra)."""
from __future__ import annotations

from pathlib import Path

from .schema import ClipResult

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]  # fmt: skip

