import pytest
from courtvision.contact import detect, detect_for_review
from courtvision.contact_model import ContactModel, Label, Review, train
from courtvision.schema import ClipResult, ContactEvent, Frame, PlayerFrame, VideoMeta


def player(track, left, *, visible=True):
    points = [(0, 0, 0)] * 17
    points[9] = (45, 45, 1)
    if visible:
        points[5], points[6] = (left + 5, 25, 1), (left + 20, 25, 1)
        points[11], points[12] = (left + 5, 65, 1), (left + 20, 65, 1)
        points[7] = (left + 12, 38, 1)
    return PlayerFrame(track_id=track, bbox=(left, 0, left + 60, 100), keypoints=points)


def clip(name, annotated_frames, *, target_left=42, visible=True):
    frames = [
        Frame(index=i, t=i / 30, players=[player(1, 0), player(2, target_left, visible=visible)])
        for i in range(annotated_frames)
    ]
    return ClipResult(
        video=VideoMeta(source=name, fps=30, width=200, height=100, n_frames=len(frames)),
        frames=frames,
    )


def test_strict_review_uses_pose_evidence_and_duration():
    assert detect(clip("short.mp4", 3))
    assert not detect_for_review(clip("short.mp4", 3))
