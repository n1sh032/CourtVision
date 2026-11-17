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
        a, b = players[track_a], players[track_b]
        ha, hb = a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1]
        height_ratios.append(min(ha, hb) / max(ha, hb))
        nearest = 1.0
        for source, target, height in ((a, b, hb), (b, a, ha)):
            for i in WRISTS:
                wrist = source.keypoints[i]
                if not _usable(wrist, 0.3):
                    continue
                for j, k in BODY_SEGMENTS:
                    start, end = target.keypoints[j], target.keypoints[k]
                    if _usable(start, 0.3) and _usable(end, 0.3):
                        nearest = min(nearest, point_segment_distance(wrist[:2], start[:2], end[:2]) / height)
        distances.append(nearest)
    span = event.end_frame - event.start_frame + 1
    return (
        span / result.video.fps,
        event.hit_frames / span,
        min(distances, default=1.0),
        statistics.median(distances) if distances else 1.0,
        statistics.median(height_ratios) if height_ratios else 0.0,
    )


def train(paths: list[Path]) -> tuple[ContactModel, dict]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ValueError('Install training dependencies with pip install -e "./pipeline[train]"') from exc

    samples = {}
    for path in paths:
        review = Review.model_validate_json(path.read_bytes())
        result = review.result
        clip = result.video.source_sha256 or result.video.source
        for label in review.labels:
            if label.event_index >= len(result.contacts):
                raise ValueError(f"{path}: label points to an event that does not exist")
            if label.label == "unclear":
                continue
            event = result.contacts[label.event_index]
            key = (clip, event.track_ids, event.start_frame, event.end_frame)
            value = int(label.label == "contact")
            if key in samples and samples[key][1] != value:
                raise ValueError(f"{path}: conflicting labels for the same event")
            samples[key] = (features(result, event), value)

    rows = [(key[0], x, y) for key, (x, y) in samples.items()]
    clips = sorted({clip for clip, _, _ in rows})
    if len(clips) < 2 or {y for _, _, y in rows} != {0, 1}:
        raise ValueError("Label visible contact and no contact across at least two different clips")

    def fit(training):
        x = [row[1] for row in training]
        y = [row[2] for row in training]
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(class_weight="balanced", random_state=0).fit(scaler.transform(x), y)
        return ContactModel(
            mean=tuple(scaler.mean_), scale=tuple(scaler.scale_),
            weight=tuple(model.coef_[0]), intercept=model.intercept_[0],
        )

    report = {"clips": len(clips), "labels": len(rows), "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for holdout in clips:
        training = [row for row in rows if row[0] != holdout]
        if {y for _, _, y in training} != {0, 1}:
            raise ValueError("Label both classes across multiple clips so held-out clips can be evaluated")
        model = fit(training)
        for clip, x, y in rows:
            if clip == holdout:
                key = (("tn", "fp"), ("fn", "tp"))[y][model.accepts(x)]
                report[key] += 1
    return fit(rows), report
