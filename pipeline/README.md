# CourtVision pipeline

The Python package processes clips, exports poses, finds contact candidates, renders overlays and serves the local viewer. The [project README](../README.md) covers installation and browser review.

## CLI

From the repository root, create a virtual environment and install the package. The base package can run contact analysis on existing pose JSON. Use the `ml` extra for video decoding, pose inference and overlays.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./pipeline[ml,train,dev]"
$cv = ".\.venv\Scripts\courtvision.exe"
& $cv --help
& $cv process clip.mp4 -o pose.json --device cpu
& $cv contacts pose.json -o contacts.json
& $cv overlay clip.mp4 contacts.json -o annotated.mp4
```

Use a new output filename on each run. For `process` and `serve`, `--model` accepts a path to existing YOLO weights, and `--device cpu` forces CPU inference. The `contacts` command reads saved poses without running inference again. Overlay draws track IDs, bounding boxes and skeletons onto a matching video. The browser viewer highlights the selected candidate's tracks.

## Candidate detection

The CLI uses the same rules as **Broad search**. A frame counts for a track pair if their boxes overlap or touch, and a wrist from either person is near a segment of the other person's body. The wrist and both segment endpoints must exceed the confidence threshold. The distance threshold scales with the height of the target person's bounding box.

| Option | Default | Meaning |
| --- | --- | --- |
| `--distance-ratio` | `0.08` | Maximum distance as a fraction of the target box height |
| `--min-conf` | `0.3` | Minimum confidence, exceeded by each keypoint |
| `--min-hit-frames` | `3` | Actual hits required to emit an event |
| `--max-gap-frames` | `2` | Missing or non-hit frames allowed between hits |

Hits at frames 10, 13 and 14 produce one event with three hits and inclusive endpoints 10–14. Frames 11 and 12 bridge the gap without counting as hits. Both directions are checked and a pair counts at most once per frame.

```powershell
& $cv contacts pose.json -o contacts-retuned.json --distance-ratio 0.05 --min-conf 0.5 --min-hit-frames 4 --max-gap-frames 1
```

Retuning replaces the candidate array in the output while preserving the poses. The browser's stricter profile also checks torso visibility, elbow confidence, and relative track size, so CLI thresholds alone do not reproduce that profile.

## Result format

- `video` contains source name, frame rate, dimensions and frame count. Fresh exports include a SHA-256 hash to identify the exact input.
- `frames` contains frame indices, timestamps in seconds, track IDs, boxes and 17 COCO keypoints in source-video pixels.
- `contacts` contains track pairs, frame and time intervals, and hit counts. Frame endpoints are inclusive.
- Review exports contain a `result` and labels keyed by zero-based candidate index. The classifier uses event duration, hit density, wrist-distance statistics and relative box height. It does not change YOLO's pose model.

## Working proof and checks

A previous run passed **111 pipeline tests** with `train` and `dev` dependencies available. The suite covers geometry, event grouping, CLI validation and synthetic training. The real CPU and browser smoke check is summarized in the [main README](../README.md). For your own video, inspect the printed frame and candidate counts and open the matching clip and `contacts.json` in the viewer. A count of zero is valid.

```powershell
.\.venv\Scripts\python.exe -m pytest -q pipeline/tests
.\.venv\Scripts\python.exe -m ruff check pipeline
node --check viewer/viewer.js
```

Node.js is only needed for the JavaScript syntax check. See the [main README](../README.md#limits) for detection and codec limitations.
