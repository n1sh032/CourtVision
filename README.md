# CourtVision

CourtVision finds candidate player contact in basketball footage. Choose a clip in the browser, run detection, then jump between candidates with the two player tracks highlighted. Candidates are moments to review, not foul decisions.

## Requirements

- Python 3.10 or later and a recent browser. The full workflow was exercised on Windows with Chrome and CPU inference.
- The `ml` extra for Ultralytics YOLO pose and OpenCV. The first run may download `yolo11n-pose.pt`; use `--model` if you already have the weights.
- A constant-frame-rate H.264 MP4 you have permission to use. A short clip is easiest to process and review. The browser service accepts files up to 200 MiB.
- The optional `train` extra for fitting a contact filter, or `dev` for running checks.

## Start the viewer

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./pipeline[ml]"
.\.venv\Scripts\courtvision.exe serve --device cpu
```

1. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. Choose the original clip and click **Run Detection**. The page shows progress and opens the results automatically.
3. Select a candidate to jump to its first frame. Use the timeline or the frame buttons to inspect it.
4. Press Ctrl+C in the terminal to stop the service. If port 8000 is occupied, add `--port 8001` and use the address printed by CourtVision.

The service binds to loopback and processes the video locally. If a browser cannot play the source, convert a copy to H.264 MP4 before processing so the browser and pose records refer to the same file.

## How detection works

YOLO estimates body keypoints and ByteTrack links detections into tracks. The contact stage checks how close one player's wrist appears to another player's body in the image, then groups repeated hits into an interval.

| Mode | What it does |
| --- | --- |
| **Broad search** | Proposes candidates using wrist-to-body distance. Use it when collecting training examples. |
| **Stricter review** | Requires clearer upper-body pose evidence, similar track sizes and roughly 0.2 seconds of hits. This can suppress false positives and miss brief genuine contact. |
| **Stricter review with a loaded model** | Scores Broad search candidates with the supplied classifier. Broad search remains unchanged. |

The viewer can also open existing contact results. Choose the matching source clip and **Or open saved results**. See [pipeline commands and tuning](pipeline/README.md).

## Working proof

| What was exercised | Observed result |
| --- | --- |
| Clip to candidates in Chrome | A local 150-frame H.264 clip ran through real CPU inference. The browser loaded 15 Broad search candidates and verified the source clip. Candidate selection and a review-label export were exercised. |
| Annotated video | OpenCV rendered a 150-frame MP4 and reopened it successfully. |
| First review batch | A separate 1,112-frame clip produced 50 candidates. The reviewer marked 2 visible contacts, 14 no-contact examples and 34 unclear cases. |
| Automated checks | A previous run passed 111 pipeline tests, including synthetic classifier training. Ruff and JavaScript syntax checks passed. |

The [review export](experiments/contact-pilot/game-000-reviewed.json) and [pilot model](experiments/contact-pilot/contact-model-single-clip-pilot.json) are included. The export contains poses and labels, and the source footage stays outside the repository. The pilot model was fitted on one clip and checked for loadability. These results establish that the workflow runs. Accuracy on independent footage has not been measured.

## Train a filter

Run **Broad search** on clips from multiple independent games or recording sessions. Mark clear touches **Visible contact**, clear walk-bys or perspective overlaps **No contact**, and blocked views **Unclear or occluded**. Export each review as a separate JSON file. The trainer skips unclear labels.

```powershell
.\.venv\Scripts\python.exe -m pip install -e "./pipeline[ml,train]"
.\.venv\Scripts\courtvision.exe train-contacts clip-a-labels.json clip-b-labels.json clip-c-labels.json -o contact-model.json
.\.venv\Scripts\courtvision.exe serve --contact-model contact-model.json
```

Stop the earlier server before restarting with the model. Replace the example filenames with your actual exports. The trainer holds out one clip at a time, so use clips from separate sources for this check. It cannot infer that two excerpts came from the same game. The report measures classification on labeled candidates only.

## Limits

One camera can make players at different depths appear to touch. Occlusion, pose jitter and tracker ID changes can produce false hits or missed contacts. The viewer uses frame rate to map playback time to pose frames, so variable frame rate clips or some codecs can affect stepping. Contact analysis and browser review load the pose result into memory, which is another reason to use short clips.

See the [development log](docs/devlog.md). CourtVision is independent of the NBA and its teams.

## Application structure and local address

Run `courtvision serve` from the repository root and open
[CourtVision](http://courtvision.localhost:8000). The branded address resolves to
your own computer in Chrome and Edge. Detection now lives in an application
service, while the HTTP server handles request validation and upload limits.

See [architecture and development](docs/ARCHITECTURE.md).
