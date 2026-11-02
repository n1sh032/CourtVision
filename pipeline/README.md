# CourtVision pipeline

Run these PowerShell commands from the repository root. They use the existing
venv directly, so activation is optional.

```powershell
.\pipeline\.venv\Scripts\python.exe -m courtvision.cli contacts data\results\sample.json -o data\results\sample_contacts.json
```

Expected output: `N candidate contact events; wrote data\results\sample_contacts.json`.
The output contains the original video metadata and poses, plus a `contacts`
array. This command reads saved JSON and does not load ML packages, decode the
video, or rerun pose inference. It can also be used as `courtvision contacts`
when the venv is activated.

## Geometry and tuning

A frame contributes one hit for a pair of different track IDs when:

1. Their bounding boxes overlap or touch, including a shared edge or corner.
2. Either player's wrist is within `distance_ratio * target_box_height` pixels
   of a body segment of the other player.
3. The wrist and both segment endpoints have confidence strictly greater than
   `min_conf`. Nonfinite coordinates/confidences and invalid boxes are ignored.

Body segments use COCO-17 shoulders, elbows, wrists, hips, knees, and ankles:
the shoulder and hip crossbars, both sides of the torso, upper/lower arms, and
upper/lower legs. Distance is to the finite segment, including its endpoints.
Both wrists and both directions are checked, but each pair has only one hit per
frame. Distance uses the other player's box height, so the two directions may
have different pixel thresholds.

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--distance-ratio` | `0.08` | Maximum distance as a fraction of the target's box height |
| `--min-conf` | `0.3` | Confidence must be strictly above this value |
| `--min-hit-frames` | `3` | Minimum number of actual hit frames per event |
| `--max-gap-frames` | `2` | Maximum missing/non-hit frames between hits |

For example, hits at frames 10, 13, and 14 merge with the defaults: the two
missing frames between 10 and 13 are tolerated. The event has three hit frames
and inclusive endpoints 10 and 14. Trailing gaps do not extend its end. A single
hit is discarded even if surrounded by tolerated gaps. Missing frame indices
count toward gaps; gap settings use frames, not seconds. Frames must have
strictly increasing nonnegative indices and nondecreasing finite times.

Retune without running the model again:

```powershell
.\pipeline\.venv\Scripts\python.exe -m courtvision.cli contacts data\results\sample.json -o data\results\sample_contacts_strict.json --distance-ratio 0.05 --min-conf 0.5 --min-hit-frames 4 --max-gap-frames 1
```

## JSON contract

`ContactEvent` stores `track_ids` (two distinct IDs in ascending order),
`start_frame`, `end_frame`, `start_t`, `end_t`, and `hit_frames`. Frame/time
endpoints refer to the first and last actual hits; times come from saved frames.
Events are sorted by start frame, then track pair. Retuning replaces the
`contacts` array rather than appending duplicate results.

The optional `contacts` field defaults to `[]`, so existing Step 2 JSON can still
be read. This additive change retains schema version `0.1.0` and does not change
the existing pose fields.

## Limitations

These are 2D proximity candidates, not verified physical contacts or foul
decisions. Perspective can make separated players appear close. Pose jitter,
occlusion, and changing box height can cause false hits or missed contacts.
Confidence filtering and flicker removal help suppress noise but do not fix
pose estimates. ID changes split a player's history and can fragment events;
the detector does not reconnect IDs or detect camera cuts. It checks wrists
against body segments, so other kinds of contact can be missed. Defaults are
starting values; precision and recall have not been measured.

## Tests

```powershell
.\pipeline\.venv\Scripts\python.exe -m pytest pipeline\tests -q
.\pipeline\.venv\Scripts\python.exe -m ruff check pipeline
```

Tests use synthetic keypoints and saved JSON. They require the `dev` extra but
do not need video footage, model weights, or the `ml` extra.
