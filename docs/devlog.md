# Devlog

## Step 1: Project foundation
**Why this project:** I wanted a portfolio piece that combines web engineering with computer vision, built around a problem fans and analysts actually argue about.

**Scope decision:** I'm building a *contact moment finder*, not a foul classifier. Whether contact is a foul depends on rules and referee judgment that even officials disagree on, so claiming a verdict would overstate what the tool can do. Surfacing candidate moments with evidence is honest and still useful.

**Naming:** CourtVision.

## Step 2: Tracking and pose pipeline
**Built:** Ultralytics YOLO11 pose with ByteTrack, saving per-frame player boxes, track IDs and 17 keypoints to JSON, plus an overlay renderer to check the output visually.

**Decision:** The pipeline writes JSON to a defined schema (`schema.py`). The web app will only read that schema, so models can be swapped later without touching the frontend.

**Run feedback:** Added frame progress, processing speed, and a rough remaining-time estimate after a long CPU run appeared unresponsive. JSON is written after tracking finishes.

**Observations:** The short sample processed successfully. Visual review found decent tracking, with skeleton jitter during falls and occasional ID changes. Quality has not been quantitatively evaluated.
