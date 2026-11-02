from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .schema import ClipResult


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

    args = ap.parse_args(argv)

    if args.cmd == "process":
        from . import track

        result = track.run(args.video, model_name=args.model, device=args.device)
        print(f"Writing {args.out}...", file=sys.stderr, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.model_dump_json())
        n = sum(len(f.players) for f in result.frames)
        ids = {p.track_id for f in result.frames for p in f.players}
        print(f"{len(result.frames)} frames, {n} player detections, {len(ids)} unique track IDs")
    else:
        from . import overlay

        result = ClipResult.model_validate_json(args.json.read_text())
        overlay.render(args.video, result, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
