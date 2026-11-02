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

    p = sub.add_parser("contacts", help="find contact candidates in saved pose JSON")
    p.add_argument("json", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--distance-ratio", type=float, default=0.08)
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--min-hit-frames", type=int, default=3)
    p.add_argument("--max-gap-frames", type=int, default=2)

    args = ap.parse_args(argv)

    if args.cmd == "process":
        from . import track

        result = track.run(args.video, model_name=args.model, device=args.device)
        print(f"Writing {args.out}...", file=sys.stderr, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.model_dump_json())
        n = sum(len(f.players) for f in result.frames)
        ids = {p.track_id for f in result.frames for p in f.players}
        print(
            f"{len(result.frames)} frames, {n} player detections, {len(ids)} unique track IDs"
        )
    elif args.cmd == "overlay":
        from . import overlay

        result = ClipResult.model_validate_json(args.json.read_text())
        overlay.render(args.video, result, args.out)
        print(f"wrote {args.out}")
    else:
        from .contact import detect

        try:
            result = ClipResult.model_validate_json(
                args.json.read_text(encoding="utf-8")
            )
            events = detect(
                result,
                distance_ratio=args.distance_ratio,
                min_conf=args.min_conf,
                min_hit_frames=args.min_hit_frames,
                max_gap_frames=args.max_gap_frames,
            )
        except ValueError as exc:
            ap.error(str(exc))
        result = result.model_copy(update={"contacts": events})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.model_dump_json(), encoding="utf-8")
        print(f"{len(events)} candidate contact events; wrote {args.out}")


if __name__ == "__main__":
    main()
