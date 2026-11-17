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


def _wrist_near_body(
    source: PlayerFrame, target: PlayerFrame, distance_ratio: float, min_conf: float
) -> bool:
    threshold = distance_ratio * (target.bbox[3] - target.bbox[1])
    for wrist_index in WRISTS:
        wrist = source.keypoints[wrist_index]
        if not _usable(wrist, min_conf):
            continue
        for start_index, end_index in BODY_SEGMENTS:
            start, end = target.keypoints[start_index], target.keypoints[end_index]
            if not (_usable(start, min_conf) and _usable(end, min_conf)):
                continue
            if point_segment_distance(wrist[:2], start[:2], end[:2]) <= threshold:
                return True
    return False


@dataclass
class _PendingEvent:
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    hit_frames: int = 1

    def finish(self, track_ids: TrackPair) -> ContactEvent:
        return ContactEvent(track_ids=track_ids, **vars(self))


def detect(
    result: ClipResult,
    distance_ratio: float = 0.08,
    min_conf: float = 0.3,
    min_hit_frames: int = 3,
    max_gap_frames: int = 2,
) -> list[ContactEvent]:
    """Return candidates without modifying the input poses or existing contacts.

    Distance is scaled by the target player's box height. Both wrist directions
    are tested, but a track pair contributes at most one hit per frame. Gaps are
    measured using frame indices, including indices absent from the input JSON.
    """
    if not math.isfinite(distance_ratio) or distance_ratio < 0:
        raise ValueError("distance_ratio must be finite and nonnegative")
    if not math.isfinite(min_conf) or not 0 <= min_conf <= 1:
        raise ValueError("min_conf must be between 0 and 1")
    if not isinstance(min_hit_frames, int) or min_hit_frames < 1:
        raise ValueError("min_hit_frames must be a positive integer")
    if not isinstance(max_gap_frames, int) or max_gap_frames < 0:
        raise ValueError("max_gap_frames must be a nonnegative integer")

    events: list[ContactEvent] = []
    active: dict[TrackPair, _PendingEvent] = {}
    previous_index = -1
    previous_t = -1.0

    def finish(pair: TrackPair) -> None:
        pending = active.pop(pair)
        if pending.hit_frames >= min_hit_frames:
            events.append(pending.finish(pair))

    for frame in result.frames:
        if frame.index <= previous_index:
            raise ValueError(
                "frame indices must be nonnegative and strictly increasing"
            )
        if not math.isfinite(frame.t) or frame.t < 0 or frame.t < previous_t:
            raise ValueError(
                "frame times must be finite, nonnegative, and nondecreasing"
            )
        previous_index, previous_t = frame.index, frame.t

        for pair in list(active):
            if frame.index - active[pair].end_frame > max_gap_frames + 1:
                finish(pair)

        hits: set[TrackPair] = set()
        players = [player for player in frame.players if _valid_box(player)]
        for a, b in combinations(players, 2):
            if a.track_id == b.track_id or not _boxes_touch(a, b):
                continue
            if _wrist_near_body(a, b, distance_ratio, min_conf) or _wrist_near_body(
                b, a, distance_ratio, min_conf
            ):
                hits.add((min(a.track_id, b.track_id), max(a.track_id, b.track_id)))

        for pair in hits:
            if pair in active:
                pending = active[pair]
                pending.end_frame, pending.end_t = frame.index, frame.t
                pending.hit_frames += 1
            else:
                active[pair] = _PendingEvent(
                    start_frame=frame.index,
                    end_frame=frame.index,
                    start_t=frame.t,
                    end_t=frame.t,
                )

    for pair in list(active):
        finish(pair)
    return sorted(events, key=lambda event: (event.start_frame, event.track_ids))
