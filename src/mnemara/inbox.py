"""Read-only inbox peek against the Architect returns queue.

Queries muninn.db (Architect SQLite) without writing — never modifies the DB.
Gracefully degrades if the DB path is unset, doesn't exist, or is unreachable.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: str) -> "sqlite3.Connection | None":
    """Open a read-only connection to muninn.db. Returns None on any error."""
    try:
        p = Path(db_path).expanduser()
        if not p.exists():
            return None
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def peek_pending_pings(
    architect_db_path: "str | None",
    peer_roles: list[str],
    exclude_role: "str | None" = None,
) -> list[dict[str, Any]]:
    """Return all pending pings from peer_roles, oldest-first.

    Reads `returns` table with status='pending' and agent_role IN peer_roles.
    exclude_role is removed from the query set (caller's own instance/role).
    Returns [] on any error or if the DB is unconfigured. Never writes.
    """
    if not architect_db_path:
        return []
    roles = [r for r in peer_roles if r != exclude_role]
    if not roles:
        return []
    conn = _connect(architect_db_path)
    if conn is None:
        return []
    try:
        placeholders = ",".join("?" * len(roles))
        rows = conn.execute(
            f"SELECT id, agent_role, task_id, payload_json, submitted_at "
            f"FROM returns "
            f"WHERE status='pending' AND agent_role IN ({placeholders}) "
            f"ORDER BY submitted_at ASC",
            roles,
        ).fetchall()
        now = datetime.now(timezone.utc)
        result = []
        for row in rows:
            payload: dict[str, Any] = {}
            try:
                raw = row["payload_json"]
                if raw:
                    payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                pass
            body_preview = str(payload.get("body", "")).replace("\n", " ")[:80]
            result.append({
                "id": row["id"],
                "agent_role": row["agent_role"],
                "task_id": row["task_id"] or "",
                "submitted_at": row["submitted_at"] or "",
                "age": _humanize_age(row["submitted_at"] or "", now),
                "payload_type": payload.get("type", ""),
                "payload_subject": payload.get("subject", ""),
                "body_preview": body_preview,
            })
        return result
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def count_pending(
    architect_db_path: "str | None",
    peer_roles: list[str],
    exclude_role: "str | None" = None,
) -> int:
    """Return the count of pending pings from peer_roles. Returns 0 on any error."""
    try:
        return len(peek_pending_pings(architect_db_path, peer_roles, exclude_role))
    except Exception:
        return 0


def format_inbox(pings: list[dict[str, Any]]) -> str:
    """Format a list of pings into a human-readable inbox listing."""
    if not pings:
        return "Inbox empty."
    lines = [f"[b]{len(pings)} pending ping{'s' if len(pings) != 1 else ''}:[/b]"]
    for p in pings:
        sender = p["agent_role"]
        task = p["task_id"]
        age = p["age"]
        ptype = p["payload_type"]
        subject = p["payload_subject"]
        preview = p["body_preview"]
        parts = [f"[{p['id']}]", f"{age}", f"{sender}"]
        if task:
            parts.append(f"-> {task}")
        if ptype:
            parts.append(f"| {ptype}")
        if subject:
            parts.append(f"| {subject}")
        if preview:
            parts.append(f"| {preview}")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


def _humanize_age(submitted_at: str, now: datetime) -> str:
    if not submitted_at:
        return "?"
    try:
        ts = submitted_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        secs = int(delta.total_seconds())
        if secs < 0:
            return "just now"
        if secs < 60:
            return f"{secs}s ago"
        mins = secs // 60
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return "?"
