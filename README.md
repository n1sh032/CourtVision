# CourtVision

CourtVision helps find and review candidate player contact in basketball clips.
It tracks people, estimates poses, and flags frames where a wrist appears close
to another player's body. Candidates are not foul decisions.

## Run a clip

Use Python 3.10 or later. Run these commands from the repository root with a
clip you have permission to review:

```powershell
python -m pip install -e "./pipeline[ml]"
courtvision process clip.mp4 -o pose.json
courtvision contacts pose.json -o contacts.json
courtvision overlay clip.mp4 pose.json -o annotated.mp4
```

The overlay command is optional. To review candidates, open
[viewer/index.html](viewer/index.html) in a recent browser. Select the **original
clip** and `contacts.json`. Select a candidate to jump to its first frame and
highlight the two track IDs. Use Previous frame, Next frame, or the timeline to
inspect surrounding frames. The browser reads both files locally.

If your browser restricts local pages, run
`python -m http.server 8000 --bind 127.0.0.1 --directory viewer` and open
`http://127.0.0.1:8000`.

See the [pipeline guide](pipeline/README.md) for tuning and tests. Retuning
replaces the candidates in a new output file and preserves the original poses.

## Limitations

Tracking IDs can change when players overlap or leave the frame, and pose
estimates can jitter or miss joints. A 2D view can make separated players appear
close. The detector can miss contact that does not involve a wrist. Precision
and recall have not been measured against labeled footage.

The viewer maps playback time to pose frames using the reported frame rate.
Use constant frame rate clips for frame stepping. Browser seeking can vary by
codec. New exports include a source hash that the viewer checks in browsers with
Web Crypto. Older exports are checked by filename, dimensions and duration.

Processing and overlay stream pose records. The contact command and viewer load
the complete pose result. Short clips work best for review.

See the [development log](docs/devlog.md). CourtVision does not redistribute NBA
footage and is independent of the NBA and its teams.
