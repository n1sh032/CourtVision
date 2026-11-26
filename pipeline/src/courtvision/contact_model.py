"""Extract event features and train a small classifier from reviewed clips."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat
