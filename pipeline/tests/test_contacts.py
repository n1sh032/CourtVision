"""Synthetic geometry and event tests; no footage or ML dependencies required."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from courtvision.cli import main
from courtvision.contact import BODY_SEGMENTS, detect, point_segment_distance
from courtvision.schema import ClipResult, ContactEvent, Frame, PlayerFrame, VideoMeta
from pydantic import ValidationError
