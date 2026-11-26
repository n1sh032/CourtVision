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
