"""Persisted per-instance permission decisions."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import paths
from .config import Config, ToolPolicy


class PermissionStore:
    """Wraps permissions.json — patterns user has approved with 'always'."""

    def __init__(self, instance: str):
        self.path = paths.permissions_path(instance)
        if self.path.exists():
            try:
                self.data: dict[str, Any] = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}
        else:
            self.data = {}

    def patterns(self, tool: str) -> list[str]:
        return list(self.data.get(tool, {}).get("allowed_patterns", []))

    def add_pattern(self, tool: str, pattern: str) -> None:
        bucket = self.data.setdefault(tool, {"allowed_patterns": []})
        if pattern not in bucket["allowed_patterns"]:
            bucket["allowed_patterns"].append(pattern)
        self._save()

    def session_allow(self, tool: str) -> None:
        self.data.setdefault(tool, {"allowed_patterns": []})["session_allow"] = True
        # session-only — we do NOT persist this flag
        # but we keep it in memory for the session
        # (we still write the file to ensure the bucket exists)

    def is_session_allowed(self, tool: str) -> bool:
        return bool(self.data.get(tool, {}).get("session_allow"))

    def _save(self) -> None:
        # Strip session-only flags before persisting.
        clean = {}
        for k, v in self.data.items():
            v2 = {kk: vv for kk, vv in v.items() if kk != "session_allow"}
            clean[k] = v2
        self.path.write_text(json.dumps(clean, indent=2) + "\n")


def matches_any(text: str, patterns: list[str]) -> bool:
    for pat in patterns:
        try:
            if re.search(pat, text):
                return True
        except re.error:
            continue
    return False


def decide(
    cfg: Config,
    perms: PermissionStore,
    tool: str,
    target: str,
) -> str:
    """Return 'allow' | 'ask' | 'deny' for this invocation.

    target is the bash command (for Bash) or absolute path (for file tools).
    """
    policy = cfg.policy_for(tool)
    if policy.mode == "deny":
        return "deny"
    if policy.mode == "allow":
        return "allow"
    # mode == ask
    if perms.is_session_allowed(tool):
        return "allow"
    config_patterns = list(policy.allowed_patterns)
    persisted = perms.patterns(tool)
    if matches_any(target, config_patterns + persisted):
        return "allow"
    return "ask"
