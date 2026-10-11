"""Small file helpers shared by the scripts and the app."""
import json
import os
from pathlib import Path


def write_json(path: Path, data, indent: int | None = 2) -> None:
    """Write atomically so a crash mid-save never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
