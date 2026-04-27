"""Append-only JSONL debug log + stderr warnings."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log_path: Path | None = None


def set_log_path(path: Path) -> None:
    global _log_path
    _log_path = path
    path.parent.mkdir(parents=True, exist_ok=True)


def log(event: str, **kwargs: Any) -> None:
    if _log_path is None:
        return
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    try:
        with _log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def warn(msg: str) -> None:
    print(f"[mnemara] warning: {msg}", file=sys.stderr)
    log("warning", message=msg)
