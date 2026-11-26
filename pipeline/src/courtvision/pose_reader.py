from collections.abc import Iterator
from pathlib import Path

import ijson

from .schema import SCHEMA_VERSION, Frame, VideoMeta


def read_metadata(path: Path) -> VideoMeta:
    with path.open("rb") as file:
        version = next(ijson.items(file, "schema_version"), SCHEMA_VERSION)
        file.seek(0)
        metadata = next(ijson.items(file, "video"), None)
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported pose schema: {version}")
    return VideoMeta.model_validate(metadata)


