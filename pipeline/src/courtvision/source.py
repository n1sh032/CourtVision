from hashlib import sha256
from pathlib import Path


def fingerprint(path: Path) -> str:
    digest = sha256()
