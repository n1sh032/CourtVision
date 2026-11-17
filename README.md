# CourtVision

CourtVision helps find and review candidate player contact in basketball clips.
It tracks people, estimates poses, and flags frames where a wrist appears close
to another player's body. Candidates are not foul decisions.

## Run a clip

Use Python 3.10 or later. For browser review, start with a constant frame rate
H.264 MP4. If the browser cannot open your source, convert a copy before
processing so the pose records and viewer refer to the same file. Run these
commands from the repository root with a clip you have permission to review:

```powershell
python -m pip install -e "./pipeline[ml]"
courtvision serve
```

Open the printed URL, normally `http://127.0.0.1:8000`, choose the clip, and
click **Run Detection**. The local service runs pose estimation and contact
detection, reports progress, and opens the results in the same viewer. The
server binds to your computer only and accepts clips up to 200 MiB. The pose
weights may be downloaded on the first run. Press Ctrl+C to stop the server.

### Reduce false positives and train a candidate filter

Browser detection defaults to **Stricter review**. It looks for a confident
wrist and elbow near the torso of a similarly sized, visible track for about
0.2 seconds. This can filter some walk-bys and occlusions, and can also drop
real brief contacts or contacts away from the torso. **Broad search** keeps
the original rules and is useful for gathering a wider range of examples.

Run Broad search on several different clips. Select each candidate and label
it **Visible contact**, **No contact**, or **Unclear or occluded**. Export
labeled examples for each clip. Keep exports local if they contain private
pose data. Use both labels across multiple clips. Unclear examples are
excluded from training.

```powershell
python -m pip install -e "./pipeline[ml,train]"
courtvision train-contacts clip-a-labels.json clip-b-labels.json -o contact-model.json
courtvision serve --contact-model contact-model.json
```

The trainer evaluates on held-out clips and writes a local classifier. When
the server is started with that model, **Stricter review** uses it to filter
Broad search candidates. Broad search itself remains available for labeling.
The report covers reviewed candidates only. It does not measure contacts the
initial candidate finder never proposed. Keep clips from the same play
together when collecting data so near-duplicates do not inflate results.

### Terminal commands and saved results

```powershell
courtvision process clip.mp4 -o pose.json
courtvision contacts pose.json -o contacts.json
courtvision overlay clip.mp4 pose.json -o annotated.mp4
```

The overlay command is optional. To review saved results, open
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
