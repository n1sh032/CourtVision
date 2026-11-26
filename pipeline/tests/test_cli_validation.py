import pytest

from courtvision.cli import main
from courtvision.schema import ClipResult, Frame, VideoMeta


def test_process_streams_readable_pose_json(tmp_path, monkeypatch):
    from courtvision import track

    def run(video, *, model_name, device, on_frame):
        for index in range(2):
            on_frame(Frame(index=index, t=index / 30, players=[]))
        return ClipResult(
            video=VideoMeta(source=video.name, fps=30, width=640, height=480, n_frames=2),
            frames=[],
        )

    monkeypatch.setattr(track, "run", run)


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "input.mp4"
    output.write_bytes(b"original")
    with pytest.raises(SystemExit) as error:
        main(["overlay", str(output), "pose.json", "-o", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"original"


def test_overlay_requires_mp4(tmp_path):
    output = tmp_path / "overlay.txt"
    with pytest.raises(SystemExit) as error:
        main(["overlay", "input.mp4", "pose.json", "-o", str(output)])
    assert error.value.code == 2
    assert not output.exists()
