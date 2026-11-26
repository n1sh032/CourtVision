from pathlib import Path

from courtvision import track
from courtvision.application import DetectionService
from courtvision.schema import ClipResult, Frame, VideoMeta


def test_service_preserves_streamed_frames_and_reports_progress(monkeypatch):
    def fake_run(path, *, model_name, device, on_frame):
        assert path == Path("clip.mp4")
        on_frame(Frame(index=0, t=0, players=[]))
        return ClipResult(
            video=VideoMeta(source="clip.mp4", fps=30, width=640, height=360, n_frames=1),
            frames=[],
        )
