"""JSON contract between the ML pipeline and the web app.

Keypoints use the COCO-17 layout. Coordinates are source-video pixels.
Bump SCHEMA_VERSION on any breaking change.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"

Keypoint = tuple[float, float, float]  # x, y, confidence
