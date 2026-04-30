"""SQLite-backed rolling-window store."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_uses TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ts ON turns(ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, instance: str):
        self.instance = instance
        self.path = paths.db_path(instance)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def append_turn(
        self,
        role: str,
        content: list[dict] | str,
        tool_uses: list[dict] | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> int:
        content_json = content if isinstance(content, str) else json.dumps(content)
        tu_json = json.dumps(tool_uses) if tool_uses else None
        cur = self.conn.execute(
            "INSERT INTO turns (ts, role, content, tool_uses, tokens_in, tokens_out) VALUES (?,?,?,?,?,?)",
            (_now(), role, content_json, tu_json, tokens_in, tokens_out),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def evict(self, max_turns: int, max_tokens: int | None = None) -> int:
        """Evict oldest turns until both caps are satisfied. Returns number deleted.

        max_turns: hard cap on row count.
        max_tokens: optional cap on estimated stored context size in tokens.
            Estimated as sum(LENGTH(content)) // 4 across all rows — the API's
            tokens_in column compounds the entire prior window per call so
            summing it would N-count the same content. The content-length /4
            heuristic is rough but bounded and additive.
        """
        deleted = 0
        # First pass: enforce turn-count cap.
        cur = self.conn.execute("SELECT COUNT(*) FROM turns")
        n = cur.fetchone()[0]
        if n > max_turns:
            to_delete = n - max_turns
            self.conn.execute(
                "DELETE FROM turns WHERE id IN (SELECT id FROM turns ORDER BY id ASC LIMIT ?)",
                (to_delete,),
            )
            self.conn.commit()
            deleted += to_delete
        # Second pass: enforce token cap (content-size estimate, not API counts).
        if max_tokens is not None and max_tokens > 0:
            while True:
                cur = self.conn.execute(
                    "SELECT COALESCE(SUM(LENGTH(content)) / 4, 0) FROM turns"
                )
                total = int(cur.fetchone()[0])
                if total <= max_tokens:
                    break
                cur = self.conn.execute("SELECT id FROM turns ORDER BY id ASC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    break
                self.conn.execute("DELETE FROM turns WHERE id = ?", (row[0],))
                self.conn.commit()
                deleted += 1
        return deleted

    def window(self, limit: int | None = None) -> list[dict[str, Any]]:
        q = "SELECT id, ts, role, content, tool_uses, tokens_in, tokens_out FROM turns ORDER BY id ASC"
        if limit is not None:
            q = f"SELECT * FROM (SELECT id, ts, role, content, tool_uses, tokens_in, tokens_out FROM turns ORDER BY id DESC LIMIT {int(limit)}) ORDER BY id ASC"
        rows = []
        for row in self.conn.execute(q):
            rows.append(
                {
                    "id": row[0],
                    "ts": row[1],
                    "role": row[2],
                    "content": _maybe_json(row[3]),
                    "tool_uses": json.loads(row[4]) if row[4] else None,
                    "tokens_in": row[5],
                    "tokens_out": row[6],
                }
            )
        return rows

    def messages_for_api(self) -> list[dict]:
        """Build the messages= list for the Anthropic API from stored turns.

        Only user/assistant rows are emitted. Marker rows (role='marker')
        and any future non-conversational rows are filtered out.
        """
        out: list[dict] = []
        for t in self.window():
            if t["role"] not in ("user", "assistant"):
                continue
            content = t["content"]
            # Always pass list-of-blocks form; if string, wrap as text block.
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            out.append({"role": t["role"], "content": content})
        return out

    def clear(self) -> None:
        self.conn.execute("DELETE FROM turns")
        self.conn.commit()

    # ------------------------------------------------------------------ eviction
    # Manual-eviction primitives. Distinct from evict() above which is the
    # cap-driven FIFO eviction the agent loop calls every turn. These are the
    # *active forgetting* surface: producer/agent picks specific rows to drop.

    def evict_last(self, n: int) -> int:
        """Delete the N most-recent rows. Returns count actually deleted.

        Counts ALL rows including markers; if you want to drop the last N
        user/assistant turns specifically, filter first via window() and
        use evict_ids.
        """
        if n <= 0:
            return 0
        cur = self.conn.execute(
            "SELECT id FROM turns ORDER BY id DESC LIMIT ?", (int(n),)
        )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM turns WHERE id IN ({placeholders})", ids
        )
        self.conn.commit()
        return len(ids)

    def evict_ids(self, ids: list[int]) -> int:
        """Delete specific row ids. Returns count actually deleted.

        Silently ignores ids that don't exist; the return is the actual
        delete count, so callers can detect partial hits.
        """
        if not ids:
            return 0
        clean = [int(i) for i in ids if isinstance(i, (int, str)) and str(i).lstrip("-").isdigit()]
        if not clean:
            return 0
        placeholders = ",".join("?" * len(clean))
        cur = self.conn.execute(
            f"SELECT COUNT(*) FROM turns WHERE id IN ({placeholders})", clean
        )
        existing = int(cur.fetchone()[0])
        if existing == 0:
            return 0
        self.conn.execute(
            f"DELETE FROM turns WHERE id IN ({placeholders})", clean
        )
        self.conn.commit()
        return existing

    def mark_segment(self, name: str) -> int:
        """Insert a named segment marker row. Returns the marker's row id.

        Markers are stored as role='marker' with the name in the content
        field. They DO NOT appear in messages_for_api (filtered there) so
        the model never sees them, but they DO appear in window() and
        list_markers() so the producer/agent can target them. If the name
        already exists, the older marker is left in place and a new one is
        appended; evict_since looks up by name + most-recent id.
        """
        if not name or not isinstance(name, str):
            raise ValueError("marker name required")
        cur = self.conn.execute(
            "INSERT INTO turns (ts, role, content, tool_uses, tokens_in, tokens_out) VALUES (?,?,?,?,?,?)",
            (_now(), "marker", json.dumps(name), None, None, None),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_markers(self) -> list[dict[str, Any]]:
        """Return all markers in id-ascending order: [{id, ts, name}, ...]."""
        rows = []
        for row in self.conn.execute(
            "SELECT id, ts, content FROM turns WHERE role='marker' ORDER BY id ASC"
        ):
            try:
                name = json.loads(row[2])
            except Exception:
                name = row[2]
            rows.append({"id": row[0], "ts": row[1], "name": name})
        return rows

    def evict_since(self, marker_name: str) -> int:
        """Delete the named marker AND every row appended after it.

        Resolves the marker by NAME picking the most recent (highest id)
        marker with that name. Deletes that marker plus all rows with
        id > marker_id. Returns count deleted, or 0 if no marker matched.
        """
        cur = self.conn.execute(
            "SELECT id FROM turns WHERE role='marker' AND content=? ORDER BY id DESC LIMIT 1",
            (json.dumps(marker_name),),
        )
        row = cur.fetchone()
        if not row:
            return 0
        marker_id = row[0]
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM turns WHERE id >= ?", (marker_id,)
        )
        n = int(cur.fetchone()[0])
        self.conn.execute("DELETE FROM turns WHERE id >= ?", (marker_id,))
        self.conn.commit()
        return n

    # ---------------------------------------------------- block-level surgery
    # The row-level eviction primitives above (evict_last/evict_ids/
    # evict_since) drop entire turns. Block-level surgery operates
    # *inside* a row: strip specific block types out of the content list
    # while preserving the rest. Used to free context budget for content
    # types the model doesn't reference back (thinking) without losing
    # the durable signal in the same turn (text + tool_use).

    def evict_thinking_blocks(
        self,
        *,
        ids: list[int] | None = None,
        keep_recent: int | None = None,
        all_rows: bool = False,
    ) -> dict[str, int]:
        """Strip 'thinking' blocks from selected rolling-window rows.

        Block-level surgery: unlike evict_ids/evict_last/evict_since which
        delete entire rows, this strips ONE block type out of rows while
        preserving text + tool_use + tool_result blocks. Designed for the
        common case where you want to free context budget by dropping the
        model's past thinking scratch (which it doesn't reference back
        across turns) while keeping the durable signal in text + tool_use
        blocks of the same turn.

        Selection (exactly one of these required):

          ids          explicit row id list
          keep_recent  strip from every row EXCEPT the most-recent N
                       (preserves recent reasoning chains the model may
                       still reference; 0 = strip from every row, same
                       semantic as all_rows)
          all_rows     strip from every row in the store

        Rows whose stripping would leave 0 blocks are skipped without
        modification — the agent can `evict_ids` such rows separately
        if a full delete is desired.

        Rows with non-list content (legacy string-encoded turns, marker
        rows, etc.) are also skipped — they have no blocks to strip.

        Returns:
          {
            "rows_scanned":   rows examined (the selector's resolved set)
            "rows_modified":  rows whose content was rewritten
            "blocks_evicted": total thinking blocks removed
            "bytes_freed":    sum of (old_content_len - new_content_len)
                              across modified rows; rough estimate of
                              context-budget savings (eviction enforces
                              against bytes/4 per total_tokens()).
          }
        """
        # Selector validation: exactly one path. `all_rows` defaults to
        # False so the explicit-True check is unambiguous; `keep_recent`
        # uses None as the unset sentinel so 0 is a legal value.
        have_ids = ids is not None
        have_keep = keep_recent is not None
        have_all = all_rows is True
        n_selected = sum([have_ids, have_keep, have_all])
        if n_selected != 1:
            raise ValueError(
                "exactly one of ids, keep_recent, all_rows required "
                f"(got ids={have_ids}, keep_recent={have_keep}, all_rows={have_all})"
            )

        # Resolve the target id set.
        target_ids: list[int]
        if have_ids:
            try:
                target_ids = [int(i) for i in ids]  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ids must be integers: {exc}") from exc
        elif have_keep:
            n = max(0, int(keep_recent))  # type: ignore[arg-type]
            cur = self.conn.execute("SELECT id FROM turns ORDER BY id DESC")
            all_ids_desc = [r[0] for r in cur.fetchall()]
            # Skip the first N (most recent); strip the rest.
            target_ids = all_ids_desc[n:]
        else:  # have_all
            cur = self.conn.execute("SELECT id FROM turns ORDER BY id ASC")
            target_ids = [r[0] for r in cur.fetchall()]

        if not target_ids:
            return {
                "rows_scanned": 0,
                "rows_modified": 0,
                "blocks_evicted": 0,
                "bytes_freed": 0,
            }

        placeholders = ",".join("?" * len(target_ids))
        rows = list(self.conn.execute(
            f"SELECT id, content FROM turns WHERE id IN ({placeholders})",
            target_ids,
        ))

        rows_modified = 0
        blocks_evicted = 0
        bytes_freed = 0
        for row_id, content_str in rows:
            try:
                blocks = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                # Non-JSON content (legacy string turns) — nothing to strip.
                continue
            if not isinstance(blocks, list):
                # JSON value but not a block list (e.g. marker rows store
                # the marker name as a JSON string). Skip.
                continue
            new_blocks = [
                b for b in blocks
                if not (isinstance(b, dict) and b.get("type") == "thinking")
            ]
            n_evicted = len(blocks) - len(new_blocks)
            if n_evicted == 0:
                continue
            if not new_blocks:
                # Don't leave a row with empty content — Anthropic's API
                # rejects messages with empty content lists. Agent can
                # evict_ids this row separately if they want it gone.
                continue
            new_str = json.dumps(new_blocks)
            delta = len(content_str) - len(new_str)
            self.conn.execute(
                "UPDATE turns SET content=? WHERE id=?",
                (new_str, row_id),
            )
            rows_modified += 1
            blocks_evicted += n_evicted
            bytes_freed += delta

        if rows_modified:
            self.conn.commit()

        return {
            "rows_scanned": len(rows),
            "rows_modified": rows_modified,
            "blocks_evicted": blocks_evicted,
            "bytes_freed": bytes_freed,
        }

    def total_tokens(self) -> tuple[int, int]:
        """Return (estimated_stored_tokens, sum_output_tokens).

        First value is the content-length-based estimate used for eviction —
        roughly bytes/4 across all rows. This is what consumes context budget.
        Second value is the cumulative API output_tokens (kept for stats; do
        NOT use for eviction since it doesn't include input).
        """
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)) / 4, 0), COALESCE(SUM(tokens_out),0) FROM turns"
        )
        a, b = cur.fetchone()
        return int(a), int(b)


def _maybe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return s
