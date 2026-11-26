from __future__ import annotations

import argparse
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
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        print(f"Writing {args.out}...", file=sys.stderr, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.model_dump_json())
        n = sum(len(f.players) for f in result.frames)
        ids = {p.track_id for f in result.frames for p in f.players}
        print(f"{len(result.frames)} frames, {n} player detections, {len(ids)} unique track IDs")
    else:
        from . import overlay

        if args.out.suffix.lower() != ".mp4":
            ap.error("overlay output must end in .mp4")
        try:
            result = ClipResult.model_validate_json(args.json.read_text(encoding="utf-8"))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=args.out.parent) as folder:
                temp = Path(folder) / "overlay.mp4"
                overlay.render(args.video, result, temp)
                temp.rename(args.out)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
