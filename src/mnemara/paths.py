"""Filesystem layout for an instance."""
from __future__ import annotations

from pathlib import Path


def root() -> Path:
    return Path.home() / ".mnemara"


def instance_dir(name: str) -> Path:
    return root() / name


def config_path(name: str) -> Path:
    return instance_dir(name) / "config.json"


def db_path(name: str) -> Path:
    return instance_dir(name) / "turns.sqlite"


def permissions_path(name: str) -> Path:
    return instance_dir(name) / "permissions.json"


def memory_dir(name: str) -> Path:
    return instance_dir(name) / "memory"


def debug_log(name: str) -> Path:
    return instance_dir(name) / "debug.log"


def list_instances() -> list[str]:
    r = root()
    if not r.exists():
        return []
    return sorted(p.name for p in r.iterdir() if p.is_dir() and (p / "config.json").exists())
