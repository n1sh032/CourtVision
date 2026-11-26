from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .schema import ClipResult, Frame


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="courtvision")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="run tracking + pose on a clip, save JSON")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--device", default=None)

    p = sub.add_parser("overlay", help="render an annotated video from saved JSON")
    p.add_argument("video", type=Path)
    p.add_argument("json", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)

    p = sub.add_parser("contacts", help="find contact candidates in saved pose JSON")
    p.add_argument("json", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--distance-ratio", type=float, default=0.08)
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--min-hit-frames", type=int, default=3)
    p.add_argument("--max-gap-frames", type=int, default=2)

    p = sub.add_parser("serve", help="run detection and review from a local browser")
    args = ap.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        ap.error(f"output already exists: {args.out}")

    if args.cmd == "process":
        from . import track

        detections = 0
        ids: set[int] = set()
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=args.out.parent) as folder:
                temp = Path(folder) / "pose.json"
                with temp.open("w", encoding="utf-8") as output:
                    output.write('{"frames":[')

                    def write_frame(frame: Frame) -> None:
                        nonlocal detections
                        if frame.index:
                            output.write(",")
                        output.write(frame.model_dump_json())
                        detections += len(frame.players)
                        ids.update(player.track_id for player in frame.players)

                    result = track.run(
                        args.video, model_name=args.model, device=args.device, on_frame=write_frame,
                    )
                    metadata = result.model_dump_json(exclude={"frames"})
                    output.write("]," + metadata[1:])
                temp.rename(args.out)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        print(f"{result.video.n_frames} frames, {detections} player detections, {len(ids)} unique track IDs")
    else:
        from . import overlay
        from .pose_reader import read_frames, read_metadata
        from ijson import JSONError

        if args.out.suffix.lower() != ".mp4":
            ap.error("overlay output must end in .mp4")
        try:
            metadata = read_metadata(args.json)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=args.out.parent) as folder:
                temp = Path(folder) / "overlay.mp4"
                frames = read_frames(args.json)
                try:
                    overlay.render_frames(args.video, metadata, frames, temp)
                finally:
                    frames.close()
                temp.rename(args.out)
        except (OSError, ValueError, JSONError) as exc:
            ap.error(str(exc))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
