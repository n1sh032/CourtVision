import pytest

from courtvision.cli import main
from courtvision.schema import ClipResult, Frame, VideoMeta


def test_overlay_reads_pose_records_in_order(tmp_path, monkeypatch):
    from courtvision import overlay

    frames = [Frame(index=index, t=index / 30, players=[]) for index in range(2)]
    result = ClipResult(
        video=VideoMeta(source="input.mp4", fps=30, width=640, height=480, n_frames=2),
        frames=frames,
    )
    pose = tmp_path / "pose.json"
    pose.write_text(result.model_dump_json(), encoding="utf-8")


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
    output = tmp_path / "pose.json"
    main(["process", "input.mp4", "-o", str(output)])
    result = ClipResult.model_validate_json(output.read_bytes())
    assert [frame.index for frame in result.frames] == [0, 1]
    assert result.video.n_frames == 2


def test_failed_process_leaves_no_output(tmp_path, monkeypatch):
    from courtvision import track

    def run(*args, on_frame, **kwargs):
        on_frame(Frame(index=0, t=0, players=[]))
        raise ValueError("failed to decode video")

    monkeypatch.setattr(track, "run", run)
    output = tmp_path / "pose.json"
    with pytest.raises(SystemExit):
        main(["process", "input.mp4", "-o", str(output)])
    assert not output.exists()


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
