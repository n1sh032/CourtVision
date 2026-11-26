"""Detection + tracking + pose via Ultralytics (needs the `ml` extra)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from .schema import ClipResult, Frame, PlayerFrame, VideoMeta
