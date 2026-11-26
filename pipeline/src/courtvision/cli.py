from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .schema import ClipResult


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="courtvision")
    sub = ap.add_subparsers(dest="cmd", required=True)
