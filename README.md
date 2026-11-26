# CourtVision

CourtVision tracks people and estimates their poses in basketball clips. The
default branch exports per-frame boxes, track IDs and keypoints as JSON and can
render an annotated MP4.

```powershell
python -m pip install -e "./pipeline[ml]"
courtvision process clip.mp4 -o pose.json
courtvision overlay clip.mp4 pose.json -o annotated.mp4
```

The frame review viewer is planned. Contact detection is under development on
the contact detection branch. Its output consists of candidate events, not foul
decisions.

The tracker can change IDs when players overlap or leave the frame, and pose
estimates can jitter or miss joints. Use short clips because processing and
overlay currently keep the full pose result in memory.

See the [development log](docs/devlog.md). CourtVision does not redistribute NBA
footage and is independent of the NBA and its teams.
