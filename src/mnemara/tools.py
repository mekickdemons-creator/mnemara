"""Native tool implementations: Bash, Read, Edit, Write, WriteMemory."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import paths
from .config import Config
from .logging_util import log
from .permissions import PermissionStore, decide
from .skeleton import extract_python_skeleton


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
    {
        "name": "read_skeleton",
        "description": "Return Python function and class signatures with docstrings only — no bodies. Use for dependency files where you only need the API surface. ~90% smaller than a full Read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the Python file"},
            },
            "required": ["file_path"],
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
            if name == "read_skeleton":
                return (_read_skeleton(params), False)
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
        write_memory(self.instance, text, category, cfg=self.cfg)
        return ("Memory note appended.", False)


def _read_skeleton(inp: dict) -> str:
    """Read a Python file and return its skeleton (signatures + docstrings).

    Returns an error string for non-Python files or missing files.
    """
    file_path = inp.get("file_path", "")
    p = Path(file_path)
    if p.suffix != ".py":
        return (
            f"read_skeleton supports Python only; use Read for {p.suffix} files."
        )
    if not p.exists():
        return f"File not found: {file_path}"
    try:
        source = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    return extract_python_skeleton(source)


def write_memory(
    instance: str,
    text: str,
    category: str = "note",
    payload: Optional[dict] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Append a memory note. With cfg provided, also performs consolidation:
    auto-RAG-index, and if category starts with 'wiki/', also write to the
    wiki layer at the slug after the prefix. Backend failures are swallowed
    so the primary memory write always succeeds."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = paths.memory_dir(instance)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{today}.md"
    ts = datetime.now(timezone.utc).isoformat()
    if payload is not None:
        obs = payload.get("observation", "")
        ev = payload.get("evidence", "")
        pred = payload.get("prediction", "")
        applies = payload.get("applies_to", "")
        conf = payload.get("confidence", "")
        block = (
            f"\n## [{ts}] observation\n\n{obs}\n\n"
            f"**evidence:** {ev}\n\n"
            f"**prediction:** {pred}\n\n"
            f"**applies_to:** {applies}\n\n"
            f"**confidence:** {conf}\n"
        )
        applies_str = (
            ", ".join(str(x) for x in applies)
            if isinstance(applies, list)
            else str(applies or "")
        )
        rag_text = "\n".join(
            x for x in (str(obs), str(ev), str(pred), applies_str) if x
        ) or text
    else:
        block = f"\n## [{ts}] {category}\n\n{text}\n"
        rag_text = text
    with f.open("a") as fh:
        fh.write(block)

    if cfg is not None:
        # wiki/ category routing
        if category and category.startswith("wiki/"):
            slug = category[len("wiki/"):]
            try:
                from . import wiki as wiki_mod
                wiki_mod.write_page(instance, slug, rag_text, mode="replace")
            except Exception as e:
                log("memory_to_wiki_error", error=str(e))
        # Auto RAG indexing
        if getattr(cfg, "rag_auto_index_memory", True) and getattr(cfg, "rag_enabled", True):
            try:
                from . import rag as rag_mod
                rag_mod.store_for(instance, cfg).index(
                    rag_text,
                    kind="memory",
                    source_path=str(f),
                    category=category or "",
                )
            except Exception as e:
                log("memory_rag_index_error", error=str(e))
        # Auto graph edges from structured payload
        if getattr(cfg, "graph_enabled", True):
            try:
                from . import graph as graph_mod
                graph_mod.auto_edges_from_memory(
                    instance, cfg, rag_text, payload, str(f)
                )
            except Exception as e:
                log("memory_graph_edge_error", error=str(e))
    return f


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:max_words]
    slug = "-".join(words)
    return slug or "proposal"


def propose_role_amendment(
    instance: str, text: str, rationale: str, severity: str
) -> Path:
    d = paths.role_proposals_dir(instance)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = _slugify(text)
    f = d / f"{ts}_{slug}.md"
    body = (
        f"---\n"
        f"date: {datetime.now(timezone.utc).isoformat()}\n"
        f"severity: {severity}\n"
        f"rationale: {json.dumps(rationale)}\n"
        f"---\n\n"
        f"{text}\n"
    )
    f.write_text(body)
    return f


def parse_proposal_file(path: "Path") -> tuple[str, str]:
    """Return (severity, body_preview) from a proposal .md file.

    Parses YAML-like frontmatter between leading/trailing '---' lines.
    Defaults severity to 'unknown' on any parse failure. Preview is the
    first 80 chars of the body (content after the closing '---').
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "unknown", ""
    lines = text.splitlines()
    severity = "unknown"
    body_start = 0
    in_front = False
    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_front = True
            continue
        if in_front:
            if line.strip() == "---":
                in_front = False
                body_start = i + 1
                break
            if line.startswith("severity:"):
                val = line.split(":", 1)[1].strip().strip('"').strip("'")
                if val:
                    severity = val
    body = " ".join(l.strip() for l in lines[body_start:] if l.strip())
    preview = body[:80] + ("..." if len(body) > 80 else "")
    return severity, preview


def log_choice(
    instance: str,
    decision_type: str,
    decision: str,
    rationale: str,
    context_summary: str = "",
    turn_id: Optional[int] = None,
    tokens_at_choice: Optional[int] = None,
) -> Path:
    f = paths.choices_path(instance)
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "decision_type": decision_type,
        "decision": decision,
        "rationale": rationale,
        "context_summary": context_summary,
        "turn_id": turn_id,
        "tokens_at_choice": tokens_at_choice,
    }
    with f.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return f
