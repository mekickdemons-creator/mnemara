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
    tokens_out INTEGER,
    pin_label TEXT
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
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Idempotent column adds for older DBs that pre-date a column.

        Mirrors the architect-side ensure_*_schema PRAGMA pattern. Running on
        every Store() construction is cheap (PRAGMA + dict membership). The
        index creation MUST run after the ALTER TABLE on legacy DBs because
        the column doesn't exist when SCHEMA's executescript runs the first
        time on a pre-existing turns table.
        """
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(turns)")}
        for col, defn in [
            # 2026-04-30 phase 1: pin_label preserves narrative-bearing rows
            # against proactive eviction (time-based, bulk-mode block surgery,
            # future auto-decay). NULL = unpinned (default). Free-form string
            # value = pin category, e.g. 'commit', 'finding', 'decision' —
            # query as `pin_label IS NOT NULL` for "all pinned" or filter by
            # the specific category.
            ("pin_label", "TEXT"),
        ]:
            if col not in cols:
                try:
                    self.conn.execute(f"ALTER TABLE turns ADD COLUMN {col} {defn}")
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass
        # Index on pin_label MUST run after the ALTER TABLE — on a legacy DB
        # the column didn't exist when SCHEMA ran. Idempotent CREATE INDEX
        # IF NOT EXISTS handles re-runs.
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pin_label ON turns(pin_label)"
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

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

    # ------------------------------------------------------------------ pinning
    # Pinning preserves narrative-bearing rows against proactive eviction.
    # The mental model: pinning marks a row as "load-bearing — don't evict
    # this in bulk operations". Time-based eviction, bulk-mode thinking
    # surgery, and future auto-decay timers all skip pinned rows by default.
    # Explicit-target eviction (evict_ids) ignores pin status because the
    # caller is making an explicit choice — the pin is advisory, not a lock.

    def pin_row(self, row_id: int, label: str = "pinned") -> bool:
        """Pin a row with a free-form category label. Returns True if matched.

        Label is free-form: 'commit', 'finding', 'decision', 'summary',
        'directive', etc. Use `pin_label IS NOT NULL` for "all pinned" and
        filter by specific label for "all commits", etc. An existing pin is
        overwritten (idempotent re-pin with a new label is allowed).

        Returns False if the row id doesn't exist.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        cur = self.conn.execute(
            "UPDATE turns SET pin_label=? WHERE id=?",
            (label.strip(), int(row_id)),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def unpin_row(self, row_id: int) -> bool:
        """Remove the pin from a row. Returns True if a previously-pinned row matched.

        Returns False if the row didn't exist OR the row existed but was not
        pinned. Idempotent: unpinning an unpinned row is a no-op.
        """
        cur = self.conn.execute(
            "UPDATE turns SET pin_label=NULL WHERE id=? AND pin_label IS NOT NULL",
            (int(row_id),),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def list_pinned(self, label: str | None = None) -> list[dict[str, Any]]:
        """Return all pinned rows ordered by id ascending.

        If `label` is provided, filters to rows with that exact pin_label.
        Each row has the same shape as window() entries.
        """
        if label is not None:
            q = (
                "SELECT id, ts, role, content, tool_uses, tokens_in, tokens_out, pin_label "
                "FROM turns WHERE pin_label=? ORDER BY id ASC"
            )
            params: tuple = (label,)
        else:
            q = (
                "SELECT id, ts, role, content, tool_uses, tokens_in, tokens_out, pin_label "
                "FROM turns WHERE pin_label IS NOT NULL ORDER BY id ASC"
            )
            params = ()
        rows = []
        for row in self.conn.execute(q, params):
            rows.append({
                "id": row[0],
                "ts": row[1],
                "role": row[2],
                "content": _maybe_json(row[3]),
                "tool_uses": json.loads(row[4]) if row[4] else None,
                "tokens_in": row[5],
                "tokens_out": row[6],
                "pin_label": row[7],
            })
        return rows

    # ------------------------------------------------------------------ eviction
    # Manual-eviction primitives. Distinct from evict() above which is the
    # cap-driven FIFO eviction the agent loop calls every turn. These are the
    # *active forgetting* surface: producer/agent picks specific rows to drop.

    def evict_last(self, n: int, *, skip_pinned: bool = True) -> int:
        """Delete the N most-recent rows. Returns count actually deleted.

        Counts ALL rows including markers; if you want to drop the last N
        user/assistant turns specifically, filter first via window() and
        use evict_ids.

        skip_pinned: if True (default) pinned rows are skipped over when
            counting and selecting. Pass skip_pinned=False to ignore pins
            and drop the most-recent N regardless of pin status.
        """
        if n <= 0:
            return 0
        if skip_pinned:
            cur = self.conn.execute(
                "SELECT id FROM turns WHERE pin_label IS NULL "
                "ORDER BY id DESC LIMIT ?",
                (int(n),),
            )
        else:
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

    def evict_since(self, marker_name: str, *, skip_pinned: bool = True) -> int:
        """Delete the named marker AND every row appended after it.

        Resolves the marker by NAME picking the most recent (highest id)
        marker with that name. Deletes that marker plus all rows with
        id > marker_id. Returns count deleted, or 0 if no marker matched.

        skip_pinned: if True (default) pinned rows in [marker_id, tail] are
            preserved (the marker itself is also preserved if pinned, which
            is unusual but consistent). Pass skip_pinned=False to drop
            everything from the marker forward including pins.
        """
        cur = self.conn.execute(
            "SELECT id FROM turns WHERE role='marker' AND content=? ORDER BY id DESC LIMIT 1",
            (json.dumps(marker_name),),
        )
        row = cur.fetchone()
        if not row:
            return 0
        marker_id = row[0]
        if skip_pinned:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM turns WHERE id >= ? AND pin_label IS NULL",
                (marker_id,),
            )
            n = int(cur.fetchone()[0])
            self.conn.execute(
                "DELETE FROM turns WHERE id >= ? AND pin_label IS NULL",
                (marker_id,),
            )
        else:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM turns WHERE id >= ?", (marker_id,)
            )
            n = int(cur.fetchone()[0])
            self.conn.execute("DELETE FROM turns WHERE id >= ?", (marker_id,))
        self.conn.commit()
        return n

    def evict_older_than(
        self,
        seconds: int,
        *,
        skip_pinned: bool = True,
    ) -> dict[str, int]:
        """Delete rows whose ts is older than `seconds` ago. Row-level deletion.

        Time-based proactive eviction. Designed for autonomous decay: an
        agent can call `evict_older_than(600)` to drop everything from
        more than 10 minutes ago, while pinned rows (commits, decisions,
        load-bearing summaries) survive.

        seconds: cutoff in seconds. Rows whose ts is older than now-seconds
            are eligible. 0 or negative = nothing eligible.
        skip_pinned: if True (default) pinned rows are preserved regardless
            of age. Pass skip_pinned=False for hard time-based purges.

        Returns:
          {
            "rows_evicted": rows actually deleted,
            "rows_skipped_pinned": rows that matched the time cutoff but
                                   were skipped because they were pinned
                                   (only meaningful when skip_pinned=True),
            "cutoff_ts":   the ISO timestamp used as cutoff.
          }
        """
        if seconds <= 0:
            return {
                "rows_evicted": 0,
                "rows_skipped_pinned": 0,
                "cutoff_ts": _now(),
            }
        # Compute cutoff timestamp. Rows are stored with _now() ISO format
        # in UTC, so string comparison on ts is monotonic and correct.
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=int(seconds))
        cutoff_ts = cutoff_dt.isoformat()

        # Count eligible rows by time + pin status for the report.
        if skip_pinned:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM turns WHERE ts < ? AND pin_label IS NULL",
                (cutoff_ts,),
            )
            n_evict = int(cur.fetchone()[0])
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM turns WHERE ts < ? AND pin_label IS NOT NULL",
                (cutoff_ts,),
            )
            n_skipped = int(cur.fetchone()[0])
            self.conn.execute(
                "DELETE FROM turns WHERE ts < ? AND pin_label IS NULL",
                (cutoff_ts,),
            )
        else:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM turns WHERE ts < ?", (cutoff_ts,)
            )
            n_evict = int(cur.fetchone()[0])
            n_skipped = 0
            self.conn.execute(
                "DELETE FROM turns WHERE ts < ?", (cutoff_ts,)
            )
        if n_evict:
            self.conn.commit()
        return {
            "rows_evicted": n_evict,
            "rows_skipped_pinned": n_skipped,
            "cutoff_ts": cutoff_ts,
        }

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
        older_than_seconds: int | None = None,
        skip_pinned: bool = True,
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

          ids                 explicit row id list
          keep_recent         strip from every row EXCEPT the most-recent N
                              (preserves recent reasoning chains the model
                              may still reference; 0 = strip from every row,
                              same semantic as all_rows)
          all_rows            strip from every row in the store
          older_than_seconds  strip from every row whose ts is older than
                              now-seconds (proactive time-based decay)

        skip_pinned: if True (default) rows with pin_label IS NOT NULL are
            excluded from the resolved target set after selector resolution.
            Pass skip_pinned=False on explicit ids to override pin
            protection (rare; agent making a deliberate choice). For bulk
            modes (keep_recent, all_rows, older_than_seconds) the default
            should almost always stay True.

        Rows whose stripping would leave 0 blocks are skipped without
        modification — the agent can `evict_ids` such rows separately
        if a full delete is desired.

        Rows with non-list content (legacy string-encoded turns, marker
        rows, etc.) are also skipped — they have no blocks to strip.

        Returns:
          {
            "rows_scanned":         rows examined after selector + pin filter
            "rows_modified":        rows whose content was rewritten
            "blocks_evicted":       total thinking blocks removed
            "bytes_freed":          sum of (old_content_len - new_content_len)
                                    across modified rows; rough estimate of
                                    context-budget savings.
            "rows_skipped_pinned":  rows that matched the selector but were
                                    skipped because they were pinned (only
                                    meaningful when skip_pinned=True).
          }
        """
        # Selector validation: exactly one path. `all_rows` defaults to
        # False so the explicit-True check is unambiguous; `keep_recent`
        # and `older_than_seconds` use None as the unset sentinel so 0 is
        # a legal value (semantically a no-op for both).
        have_ids = ids is not None
        have_keep = keep_recent is not None
        have_all = all_rows is True
        have_older = older_than_seconds is not None
        n_selected = sum([have_ids, have_keep, have_all, have_older])
        if n_selected != 1:
            raise ValueError(
                "exactly one of ids, keep_recent, all_rows, older_than_seconds required "
                f"(got ids={have_ids}, keep_recent={have_keep}, "
                f"all_rows={have_all}, older_than_seconds={have_older})"
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
        elif have_older:
            secs = int(older_than_seconds)  # type: ignore[arg-type]
            if secs <= 0:
                target_ids = []
            else:
                from datetime import timedelta
                cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=secs)
                cutoff_ts = cutoff_dt.isoformat()
                cur = self.conn.execute(
                    "SELECT id FROM turns WHERE ts < ? ORDER BY id ASC",
                    (cutoff_ts,),
                )
                target_ids = [r[0] for r in cur.fetchall()]
        else:  # have_all
            cur = self.conn.execute("SELECT id FROM turns ORDER BY id ASC")
            target_ids = [r[0] for r in cur.fetchall()]

        # Apply pin filter to the resolved id set.
        rows_skipped_pinned = 0
        if skip_pinned and target_ids:
            placeholders = ",".join("?" * len(target_ids))
            cur = self.conn.execute(
                f"SELECT id FROM turns "
                f"WHERE id IN ({placeholders}) AND pin_label IS NOT NULL",
                target_ids,
            )
            pinned_ids = {r[0] for r in cur.fetchall()}
            if pinned_ids:
                target_ids = [i for i in target_ids if i not in pinned_ids]
                rows_skipped_pinned = len(pinned_ids)

        if not target_ids:
            return {
                "rows_scanned": 0,
                "rows_modified": 0,
                "blocks_evicted": 0,
                "bytes_freed": 0,
                "rows_skipped_pinned": rows_skipped_pinned,
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
            "rows_skipped_pinned": rows_skipped_pinned,
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


def parse_duration_seconds(raw: str) -> int:
    """Parse a duration string into seconds. Accepts 's', 'm', 'h', 'd' suffixes.

    Examples:
      '600'  -> 600  (bare integer = seconds)
      '10s'  -> 10
      '10m'  -> 600
      '2h'   -> 7200
      '1d'   -> 86400

    Raises ValueError on unparseable input. Used by the /evict slash
    command and the evict_older_than / evict_thinking_blocks MCP tools
    so the parsing stays consistent across surfaces.
    """
    if raw is None:
        raise ValueError("duration required")
    s = str(raw).strip().lower()
    if not s:
        raise ValueError("duration required")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in multipliers:
        try:
            n = float(s[:-1])
        except ValueError as exc:
            raise ValueError(f"invalid duration: {raw!r}") from exc
        return int(n * multipliers[s[-1]])
    try:
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"invalid duration: {raw!r}") from exc
