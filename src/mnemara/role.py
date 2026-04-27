"""Role doc loader. Re-read every API call so swaps are transparent."""
from __future__ import annotations

from pathlib import Path

from .logging_util import warn


def load_role_doc(path: str) -> str:
    if not path:
        return ""
    p = Path(path).expanduser()
    if not p.exists():
        warn(f"role doc path not found: {p}")
        return ""
    try:
        return p.read_text()
    except OSError as e:
        warn(f"failed to read role doc {p}: {e}")
        return ""
