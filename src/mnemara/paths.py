"""Filesystem layout for an instance."""
from __future__ import annotations

import re
from pathlib import Path

_INSTANCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def root() -> Path:
    return Path.home() / ".mnemara"


def _validate_instance_name(name: str) -> str:
    if not isinstance(name, str) or not _INSTANCE_NAME_RE.match(name):
        raise ValueError(
            f"invalid instance name: {name!r} "
            "(allowed: alnum, '_', '-', '.'; must start with alnum; max 64 chars)"
        )
    return name


def instance_dir(name: str) -> Path:
    return root() / _validate_instance_name(name)


def config_path(name: str) -> Path:
    return instance_dir(name) / "config.json"


def db_path(name: str) -> Path:
    return instance_dir(name) / "turns.sqlite"


def permissions_path(name: str) -> Path:
    return instance_dir(name) / "permissions.json"


def memory_dir(name: str) -> Path:
    return instance_dir(name) / "memory"


def role_proposals_dir(name: str) -> Path:
    return instance_dir(name) / "role_proposals"


def choices_path(name: str) -> Path:
    return instance_dir(name) / "choices.jsonl"


def stats_dir(name: str) -> Path:
    return instance_dir(name) / "stats"


def debug_log(name: str) -> Path:
    return instance_dir(name) / "debug.log"


def wiki_dir(name: str) -> Path:
    return instance_dir(name) / "wiki"


def rag_index_dir(name: str) -> Path:
    return instance_dir(name) / "index"


def graph_dir(name: str) -> Path:
    return instance_dir(name) / "graph"


def sleep_dir(name: str) -> Path:
    return instance_dir(name) / "sleep"


def wiki_proposals_dir(name: str) -> Path:
    return instance_dir(name) / "wiki_proposals"


def memory_archive_dir(name: str) -> Path:
    return instance_dir(name) / "memory" / "archive"


def role_proposals_count(name: str) -> int:
    d = role_proposals_dir(name)
    if not d.exists():
        return 0
    return len(list(d.glob("*.md")))


def list_instances() -> list[str]:
    r = root()
    if not r.exists():
        return []
    return sorted(p.name for p in r.iterdir() if p.is_dir() and (p / "config.json").exists())
