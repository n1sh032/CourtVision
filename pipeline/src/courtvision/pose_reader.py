from collections.abc import Iterator
from pathlib import Path

import ijson

from .schema import SCHEMA_VERSION, Frame, VideoMeta


def read_metadata(path: Path) -> VideoMeta:
    with path.open("rb") as file:
        version = next(ijson.items(file, "schema_version"), SCHEMA_VERSION)
        file.seek(0)
