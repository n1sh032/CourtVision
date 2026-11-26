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
    assert not detect_for_review(clip("occluded.mp4", 10, visible=False))
    (candidate,) = detect_for_review(clip("visible.mp4", 10))
    assert candidate.hit_frames == 10


def test_exported_reviews_train_with_clip_holdouts(tmp_path):
    pytest.importorskip("sklearn")
    paths = []
    for name in ("clip-a.mp4", "clip-b.mp4"):
        result = clip(name, 12)
        result.contacts = [
            ContactEvent(track_ids=(1, 2), start_frame=0, end_frame=2,
                         start_t=0, end_t=2 / 30, hit_frames=3),
            ContactEvent(track_ids=(1, 2), start_frame=3, end_frame=11,
                         start_t=3 / 30, end_t=11 / 30, hit_frames=9),
        ]
        review = Review(version=1, result=result, labels=[
            Label(event_index=0, label="no_contact"),
            Label(event_index=1, label="contact"),
        ])
        path = tmp_path / f"{name}.json"
        path.write_text(review.model_dump_json(), encoding="utf-8")
        paths.append(path)
    model, report = train(paths)
