"""Native tool implementations: Bash, Read, Edit, Write, WriteMemory."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import paths
from .config import Config
from .logging_util import log
from .permissions import PermissionStore, decide


# JSONSchema-style tool definitions sent to the API.
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "Bash",
        "description": "Run a shell command and return stdout+stderr. Subject to user permission policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "timeout": {"type": "integer", "description": "Optional timeout seconds."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read the contents of a file at an absolute path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": "Write contents to a file (creates or overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace exact occurrences of old_string with new_string in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "WriteMemory",
        "description": "Append a timestamped note to the instance's memory file. Use to record insights that should survive rolling-window eviction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["text"],
        },
    },
]


# Permission prompt callback: (tool, target) -> 'allow' | 'allow_always' | 'deny'
PromptFn = Callable[[str, str], str]


class ToolRunner:
    def __init__(self, instance: str, cfg: Config, perms: PermissionStore, prompt: PromptFn):
        self.instance = instance
        self.cfg = cfg
        self.perms = perms
        self.prompt = prompt

    def dispatch(self, name: str, params: dict[str, Any]) -> tuple[str, bool]:
        """Returns (text_result, is_error)."""
        try:
            if name == "Bash":
                return self._bash(params)
            if name == "Read":
                return self._read(params)
            if name == "Write":
                return self._write(params)
            if name == "Edit":
                return self._edit(params)
            if name == "WriteMemory":
                return self._write_memory(params)
            return (f"Unknown tool: {name}", True)
        except Exception as e:
            log("tool_error", tool=name, error=str(e))
            return (f"Tool error: {e}", True)

    # ------------------------------------------------------------------ helpers

    def _check_perm(self, tool: str, target: str) -> tuple[bool, str | None]:
        verdict = decide(self.cfg, self.perms, tool, target)
        if verdict == "allow":
            return True, None
        if verdict == "deny":
            return False, "Permission denied by config policy."
        # ask
        choice = self.prompt(tool, target)
        if choice == "deny":
            return False, "Permission denied by user."
        if choice == "allow_always":
            pattern = "^" + re.escape(target) + "$"
            self.perms.add_pattern(tool, pattern)
            return True, None
        if choice == "allow_session":
            self.perms.session_allow(tool)
            return True, None
        if choice == "allow":
            return True, None
        return False, "Permission denied."

    def _enforce_path(self, p: str) -> Path:
        path = Path(p).expanduser().resolve()
        if self.cfg.file_tool_home_only:
            home = Path.home().resolve()
            if home not in path.parents and path != home:
                raise PermissionError(f"path outside home not allowed: {path}")
        return path

    # ------------------------------------------------------------------ tools

    def _bash(self, params: dict[str, Any]) -> tuple[str, bool]:
        cmd = params["command"]
        timeout = int(params.get("timeout") or self.cfg.bash_timeout_seconds)
        ok, err = self._check_perm("Bash", cmd)
        if not ok:
            return (err or "denied", True)
        log("bash_run", command=cmd)
        try:
            res = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            out = res.stdout
            if res.stderr:
                out = (out + ("\n" if out and not out.endswith("\n") else "")
                       + "[stderr]\n" + res.stderr)
            out += f"\n[exit {res.returncode}]"
            return (out, res.returncode != 0)
        except subprocess.TimeoutExpired:
            return (f"Command timed out after {timeout}s", True)

    def _read(self, params: dict[str, Any]) -> tuple[str, bool]:
        ok, err = self._check_perm("Read", params["path"])
        if not ok:
            return (err or "denied", True)
        path = self._enforce_path(params["path"])
        if not path.exists():
            return (f"File not found: {path}", True)
        text = path.read_text()
        offset = int(params.get("offset") or 0)
        limit = params.get("limit")
        lines = text.splitlines()
        if offset or limit:
            end = offset + int(limit) if limit else len(lines)
            lines = lines[offset:end]
        return ("\n".join(lines), False)

    def _write(self, params: dict[str, Any]) -> tuple[str, bool]:
        ok, err = self._check_perm("Write", params["path"])
        if not ok:
            return (err or "denied", True)
        path = self._enforce_path(params["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params["content"])
        return (f"Wrote {len(params['content'])} bytes to {path}", False)

    def _edit(self, params: dict[str, Any]) -> tuple[str, bool]:
        ok, err = self._check_perm("Edit", params["path"])
        if not ok:
            return (err or "denied", True)
        path = self._enforce_path(params["path"])
        if not path.exists():
            return (f"File not found: {path}", True)
        text = path.read_text()
        old = params["old_string"]
        new = params["new_string"]
        if old not in text:
            return ("old_string not found in file", True)
        if params.get("replace_all"):
            text2 = text.replace(old, new)
            count = text.count(old)
        else:
            if text.count(old) > 1:
                return ("old_string is not unique; pass replace_all=true or add context", True)
            text2 = text.replace(old, new, 1)
            count = 1
        path.write_text(text2)
        return (f"Replaced {count} occurrence(s) in {path}", False)

    def _write_memory(self, params: dict[str, Any]) -> tuple[str, bool]:
        text = params["text"]
        category = params.get("category", "note")
        write_memory(self.instance, text, category)
        return ("Memory note appended.", False)


def write_memory(instance: str, text: str, category: str = "note") -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = paths.memory_dir(instance)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{today}.md"
    ts = datetime.now(timezone.utc).isoformat()
    block = f"\n## [{ts}] {category}\n\n{text}\n"
    with f.open("a") as fh:
        fh.write(block)
    return f
