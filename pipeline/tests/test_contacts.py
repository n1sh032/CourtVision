"""Synthetic geometry and event tests; no footage or ML dependencies required."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from courtvision.cli import main
from courtvision.contact import BODY_SEGMENTS, detect, point_segment_distance
from courtvision.schema import ClipResult, ContactEvent, Frame, PlayerFrame, VideoMeta
from pydantic import ValidationError


def pair(distance: float = 5) -> list[PlayerFrame]:
    wrist = [(0.0, 0.0, 0.0)] * 17
    body = wrist.copy()
    wrist[9] = (50 - distance, 50, 1)
    body[5], body[11] = (50, 20, 1), (50, 80, 1)
    return [
        PlayerFrame(track_id=1, bbox=(0, 0, 50, 100), keypoints=wrist),
        PlayerFrame(track_id=2, bbox=(50, 0, 100, 100), keypoints=body),
    ]


def clip(frames: list[Frame]) -> ClipResult:
    return ClipResult(
        video=VideoMeta(
            source="synthetic.mp4",
            fps=30,
            width=640,
            height=360,
            n_frames=max((frame.index for frame in frames), default=-1) + 1,
        ),
        frames=frames,
    )


def sequence(hits: set[int], indices: list[int] | None = None) -> ClipResult:
    if indices is None:
        indices = list(range(max(hits, default=-1) + 1))
    return clip(
        [Frame(index=i, t=i / 30, players=pair() if i in hits else []) for i in indices]
    )


def single(players: list[PlayerFrame], **kwargs: object) -> list[ContactEvent]:
    return detect(
        clip([Frame(index=0, t=0, players=players)]), min_hit_frames=1, **kwargs
    )


@pytest.mark.parametrize(
    ("point", "start", "end", "expected"),
    [
        ((3, 5), (0, 0), (0, 10), 3),  # perpendicular projection
        ((3, -4), (0, 0), (0, 10), 5),  # before the first endpoint
        ((3, 14), (0, 0), (0, 10), 5),  # beyond the last endpoint
        ((5, 5), (0, 0), (10, 10), 0),
        ((3, 4), (0, 0), (0, 0), 5),  # collapsed segment
        ((-3, -4), (-6, -8), (0, 0), 0),
    ],
)
def test_point_segment_distance(point, start, end, expected):
    assert point_segment_distance(point, start, end) == pytest.approx(expected)


@pytest.mark.parametrize(("distance", "expected"), [(0, 1), (8, 1), (8.01, 0)])
def test_inclusive_distance_threshold(distance, expected):
    assert len(single(pair(distance))) == expected


def test_zero_distance_threshold_is_supported():
    assert len(single(pair(0), distance_ratio=0)) == 1
    assert not single(pair(0.01), distance_ratio=0)


def test_distance_scales_with_target_height_not_wrist_owner():
    players = pair(10)
    players[0].bbox = (0, 0, 50, 1000)
    assert not single(players)  # target height 100: threshold 8, not 80
    players[1].bbox = (50, 0, 100, 200)
    assert len(single(players)) == 1  # target height 200: threshold 16


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((50, 0, 100, 100), 1),  # shared edge
        ((45, 0, 100, 100), 1),  # overlap
        ((50.01, 0, 100, 100), 0),
        ((50, 100.01, 100, 200), 0),
        ((50, 0, 100, 0), 0),  # no height
        ((100, 0, 50, 100), 0),  # inverted width
        ((50, 0, math.inf, 100), 0),
        ((math.nan, 0, 100, 100), 0),
    ],
)
def test_boxes_must_be_valid_and_touch(bbox, expected):
    players = pair()
    players[1].bbox = bbox
    assert len(single(players)) == expected


def test_corner_touch_is_inclusive():
    players = pair(0)
    players[0].keypoints[9] = (50, 100, 1)
    players[1].bbox = (50, 100, 100, 200)
    players[1].keypoints[5] = (50, 100, 1)
    players[1].keypoints[11] = (50, 180, 1)
    assert len(single(players)) == 1


@pytest.mark.parametrize(("player_index", "joint"), [(0, 9), (1, 5), (1, 11)])
@pytest.mark.parametrize("confidence", [0, 0.3, math.nan, math.inf, 1.01])
def test_wrist_and_both_endpoints_need_confidence(player_index, joint, confidence):
    players = pair()
    x, y, _ = players[player_index].keypoints[joint]
    players[player_index].keypoints[joint] = (x, y, confidence)
    assert not single(players)


@pytest.mark.parametrize(("player_index", "joint"), [(0, 9), (1, 5), (1, 11)])
@pytest.mark.parametrize("coordinate", [math.nan, math.inf])
def test_nonfinite_keypoints_are_ignored(player_index, joint, coordinate):
    players = pair()
    _, y, confidence = players[player_index].keypoints[joint]
    players[player_index].keypoints[joint] = (coordinate, y, confidence)
    assert not single(players)


@pytest.mark.parametrize("edge", BODY_SEGMENTS)
def test_torso_arm_and_leg_segments_are_checked(edge):
    players = pair()
    players[1].keypoints = [(0, 0, 0)] * 17
    a, b = edge
    players[1].keypoints[a], players[1].keypoints[b] = (50, 20, 1), (50, 80, 1)
    assert len(single(players)) == 1


@pytest.mark.parametrize("wrist", [9, 10])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_both_wrists_and_both_player_directions(wrist, reverse_order):
    players = pair()
    players[0].keypoints[9] = (0, 0, 0)
    players[0].keypoints[wrist] = (45, 50, 1)
    if reverse_order:
        players.reverse()
    assert single(players)[0].track_ids == (1, 2)


def test_only_wrists_start_a_hit_and_face_edges_are_excluded():
    players = pair()
    players[0].keypoints[7], players[0].keypoints[9] = (45, 50, 1), (0, 0, 0)
    assert not single(players)
    players = pair()
    players[1].keypoints = [(0, 0, 0)] * 17
    players[1].keypoints[0], players[1].keypoints[1] = (50, 20, 1), (50, 80, 1)
    assert not single(players)


def test_multiple_witnesses_still_count_once_per_pair_per_frame():
    players = pair()
    players[0].keypoints[10] = (45, 51, 1)
    players[1].keypoints[6] = (50, 20, 1)
    assert single(players)[0].hit_frames == 1
    assert not detect(clip([Frame(index=0, t=0, players=players)]), min_hit_frames=2)


def test_same_track_id_never_contacts_itself():
    players = pair()
    players[1].track_id = players[0].track_id
    assert not single(players)


def test_consecutive_hits_have_inclusive_frame_and_time_bounds():
    (event,) = detect(sequence({10, 11, 12}))
    assert (event.start_frame, event.end_frame, event.hit_frames) == (10, 12, 3)
    assert event.start_t == pytest.approx(10 / 30)
    assert event.end_t == pytest.approx(12 / 30)


def test_tolerated_gaps_do_not_count_as_hits_or_extend_the_end():
    (event,) = detect(sequence({0, 3, 4}, indices=list(range(8))))
    assert (event.start_frame, event.end_frame, event.hit_frames) == (0, 4, 3)


def test_gap_beyond_limit_splits_events():
    events = detect(sequence({0, 1, 2, 6, 7, 8}))
    assert [(e.start_frame, e.end_frame) for e in events] == [(0, 2), (6, 8)]


def test_gaps_use_frame_indices_even_when_frames_are_absent():
    assert detect(sequence({0, 3, 4}, [0, 3, 4]))[0].hit_frames == 3
    events = detect(sequence({0, 4, 5}, [0, 4, 5]), min_hit_frames=2)
    assert [(e.start_frame, e.end_frame) for e in events] == [(4, 5)]


def test_zero_gap_requires_consecutive_hits():
    events = detect(sequence({0, 1, 3, 4}), min_hit_frames=2, max_gap_frames=0)
    assert [(e.start_frame, e.end_frame) for e in events] == [(0, 1), (3, 4)]


def test_short_flickers_are_removed_by_actual_hit_count():
    assert not detect(sequence({0, 3}))  # span is four frames but only two hits
    assert not detect(sequence({0}))
    assert detect(sequence({0}), min_hit_frames=1)[0].hit_frames == 1


def test_empty_frames_and_single_players_have_no_events():
    assert not detect(sequence(set()))
    assert not detect(sequence(set(), [0, 1, 2]))
    assert not single(pair()[:1])


def test_track_id_changes_split_history():
    frames = []
    for index in range(4):
        players = pair()
        players[0].track_id = 1 if index < 2 else 9
        frames.append(Frame(index=index, t=index / 30, players=players))
    events = detect(clip(frames), min_hit_frames=2)
    assert [(e.track_ids, e.start_frame, e.end_frame) for e in events] == [
        ((1, 2), 0, 1),
        ((2, 9), 2, 3),
    ]


def test_pairs_merge_independently_and_output_is_sorted():
    frames = []
    for index in range(4):
        players = pair() if index < 3 else []
        if index > 0:
            for player in pair():
                player.track_id += 2
                x1, y1, x2, y2 = player.bbox
                player.bbox = (x1 + 200, y1, x2 + 200, y2)
                player.keypoints = [(x + 200, y, c) for x, y, c in player.keypoints]
                players.insert(0, player)
        frames.append(Frame(index=index, t=index / 30, players=players))
    events = detect(clip(frames))
    assert [(e.track_ids, e.start_frame, e.end_frame) for e in events] == [
        ((1, 2), 0, 2),
        ((3, 4), 1, 3),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance_ratio": -0.1},
        {"distance_ratio": math.nan},
        {"distance_ratio": math.inf},
        {"min_conf": -0.01},
        {"min_conf": 1.01},
        {"min_conf": math.nan},
        {"min_hit_frames": 0},
        {"min_hit_frames": 1.5},
        {"max_gap_frames": -1},
        {"max_gap_frames": 0.5},
    ],
)
def test_invalid_tuning_is_rejected(kwargs):
    with pytest.raises(ValueError):
        detect(sequence(set()), **kwargs)


@pytest.mark.parametrize("indices", [[-1], [1, 0], [0, 0]])
def test_invalid_frame_order_is_rejected(indices):
    result = clip([Frame(index=i, t=0, players=[]) for i in indices])
    with pytest.raises(ValueError, match="frame indices"):
        detect(result)


@pytest.mark.parametrize("times", [[-1], [math.nan], [math.inf], [1, 0]])
def test_invalid_frame_times_are_rejected(times):
    result = clip([Frame(index=i, t=t, players=[]) for i, t in enumerate(times)])
    with pytest.raises(ValueError, match="frame times"):
        detect(result)


def test_detect_preserves_input_and_existing_contacts():
    result = sequence({0, 1, 2})
    result.contacts = detect(result)
    before = result.model_dump_json()
    assert not detect(result, distance_ratio=0)
    assert result.model_dump_json() == before


def test_legacy_json_and_contact_json_round_trip():
    legacy = sequence({0, 1, 2}).model_dump()
    legacy.pop("contacts")
    result = ClipResult.model_validate_json(json.dumps(legacy))
    assert result.contacts == []
    result.contacts = detect(result)
    assert result.schema_version == "0.1.0"
    assert ClipResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "update",
    [
        {"track_ids": (2, 1)},
        {"track_ids": (1, 1)},
        {"start_frame": -1},
        {"end_frame": 0},
        {"start_t": -1},
        {"end_t": 0},
        {"end_t": math.inf},
        {"hit_frames": 0},
        {"hit_frames": 4},
    ],
)
def test_contact_event_contract_rejects_invalid_intervals(update):
    fields = {
        "track_ids": (1, 2),
        "start_frame": 1,
        "end_frame": 3,
        "start_t": 1 / 30,
        "end_t": 3 / 30,
        "hit_frames": 3,
    }
    with pytest.raises(ValidationError):
        ContactEvent(**(fields | update))


def test_contacts_cli_preserves_poses_and_replaces_existing_events(tmp_path, capsys):
    result = sequence({0, 3, 4})
    source, target = tmp_path / "poses.json", tmp_path / "nested" / "contacts.json"
    source.write_text(result.model_dump_json(), encoding="utf-8")
    main(["contacts", str(source), "-o", str(target)])
    saved = ClipResult.model_validate_json(target.read_text(encoding="utf-8"))
    assert saved.video == result.video and saved.frames == result.frames
    assert len(saved.contacts) == 1
    assert "1 candidate contact events" in capsys.readouterr().out
    main(["contacts", str(target), "-o", str(target)])
    assert len(ClipResult.model_validate_json(target.read_text()).contacts) == 1
    assert ClipResult.model_validate_json(source.read_text()).contacts == []


@pytest.mark.parametrize(
    "options",
    [
        ["--distance-ratio", "0.01"],
        ["--min-conf", "1"],
        ["--min-hit-frames", "4"],
        ["--max-gap-frames", "0"],
    ],
)
def test_each_cli_threshold_can_retune_saved_json(tmp_path, options):
    source, target = tmp_path / "poses.json", tmp_path / "contacts.json"
    source.write_text(sequence({0, 3, 4}).model_dump_json(), encoding="utf-8")
    main(["contacts", str(source), "-o", str(target), *options])
    assert ClipResult.model_validate_json(target.read_text()).contacts == []


def test_invalid_cli_tuning_reports_error_without_writing_output(tmp_path, capsys):
    source, target = tmp_path / "poses.json", tmp_path / "contacts.json"
    source.write_text(sequence({0, 1, 2}).model_dump_json(), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["contacts", str(source), "-o", str(target), "--distance-ratio", "-1"])
    assert exc.value.code == 2
    assert "distance_ratio" in capsys.readouterr().err
    assert not target.exists()


def test_contacts_command_does_not_import_ml_packages(tmp_path: Path):
    source, target = tmp_path / "poses.json", tmp_path / "contacts.json"
    legacy = sequence({0, 1, 2}).model_dump()
    legacy.pop("contacts")
    source.write_text(json.dumps(legacy), encoding="utf-8")
    code = (
        "import sys; from courtvision.cli import main; main(); "
        "assert not any(m in sys.modules for m in ('cv2', 'torch', 'ultralytics'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, "contacts", str(source), "-o", str(target)],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(ClipResult.model_validate_json(target.read_text()).contacts) == 1
