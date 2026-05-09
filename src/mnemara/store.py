"""SQLite-backed rolling-window store."""
from __future__ import annotations

import difflib
import hashlib
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
        # Session-scoped eviction stats. Bumped by every eviction path
        # (cap-FIFO via evict(), manual via evict_last/evict_ids/evict_since/
        # evict_older_than, block surgery via _strip_blocks_by_type, and the
        # auto-evict-after-write pairing primitive). NOT persisted — resets
        # on each Store() construction. Read via get_eviction_stats() for
        # status-bar display, reporting, etc.
        self._eviction_stats: dict[str, int] = {
            "rows_evicted": 0,
            "blocks_evicted": 0,
            "bytes_freed": 0,
            "pinned_rows_force_evicted": 0,  # last-resort cap-FIFO evictions of pinned rows
        }

    def _migrate_schema(self) -> None:
        """Idempotent column adds for older DBs that pre-date a column.

        Running on every Store() construction is cheap (PRAGMA + dict
        membership lookup). The
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
            # 2026-05-07 v0.6.0: compressed_read_stub marks rows whose
            # tool_result content was replaced by a diff-based stub by
            # compress_repeated_reads(). When preserve_compressed_reads is
            # True, these rows are excluded from cap-FIFO eviction (same soft-
            # protect semantics as pin_label). 0 = normal, 1 = stub row.
            ("compressed_read_stub", "INTEGER DEFAULT 0"),
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

    # ---------------------------------------------- eviction stats counter
    # Single funnel for all eviction paths so the status bar (and any other
    # reporter) sees a consistent live picture. Counters are session-scoped:
    # not persisted, reset on Store() construction. The semantics:
    #   rows_evicted   — full rows deleted (cap-FIFO + manual row deletion)
    #   blocks_evicted — block-surgery blocks stripped (thinking + tool_use
    #                    + write-pair stubs)
    #   bytes_freed    — sum of (old_content_len - new_content_len) across
    #                    block surgery and pair eviction; rough estimate of
    #                    context-budget savings.

    def _bump_eviction_stats(
        self,
        *,
        rows: int = 0,
        blocks: int = 0,
        bytes_: int = 0,
        pinned_force: int = 0,
    ) -> None:
        if rows:
            self._eviction_stats["rows_evicted"] += int(rows)
        if blocks:
            self._eviction_stats["blocks_evicted"] += int(blocks)
        if bytes_:
            self._eviction_stats["bytes_freed"] += int(bytes_)
        if pinned_force:
            self._eviction_stats["pinned_rows_force_evicted"] += int(pinned_force)

    def get_eviction_stats(self) -> dict[str, int]:
        """Return a snapshot of session-scoped eviction counters.

        Snapshot semantics: returned dict is a copy; callers can stash it
        without worrying about mutation. Reset on Store() construction
        (i.e. once per panel session).
        """
        return dict(self._eviction_stats)

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

    # Headroom threshold for row-cap slack. When current token usage is below
    # this fraction of max_tokens, the row cap can be exceeded by up to
    # row_cap_slack rows. Hardcoded at 0.5 (50%) for the v0.3.3 ship; if
    # operators need to tune the threshold separately from the slack count,
    # promote this to a Config field. The single-knob brief was deliberate —
    # one config field (row_cap_slack_when_token_headroom) tunes the magnitude
    # of the slack; the threshold below tunes when it engages and stays a
    # constant unless we get evidence otherwise.
    HEADROOM_RATIO: float = 0.5

    def evict(
        self,
        max_turns: int,
        max_tokens: int | None = None,
        *,
        row_cap_slack: int = 0,
        preserve_compressed_reads: bool = False,
    ) -> int:
        """Evict oldest turns until both caps are satisfied. Returns number deleted.

        max_turns: hard cap on row count (the strict ceiling).
        max_tokens: optional cap on estimated stored context size in tokens.
            Estimated as sum(LENGTH(content)) // 4 across all rows — the API's
            tokens_in column compounds the entire prior window per call so
            summing it would N-count the same content. The content-length /4
            heuristic is rough but bounded and additive — and crucially, it's
            BYTE-AWARE, so block surgery's bytes_freed naturally registers
            here without any extra accounting.
        row_cap_slack: when > 0 AND max_tokens is set AND current estimated
            tokens are below max_tokens * HEADROOM_RATIO, the effective row
            cap is (max_turns + row_cap_slack) instead of max_turns. Default
            0 = no slack (backward-compat). The token cap remains the hard
            ceiling regardless of slack.

        Pin semantics (Option B — soft protection):
            Both the row-cap and token-cap passes try unpinned rows first.
            If the cap cannot be satisfied by evicting unpinned rows alone
            (all remaining rows are pinned), pinned rows are evicted oldest-
            first as a last resort, and the count is recorded separately in
            _eviction_stats["pinned_rows_force_evicted"] so the status bar
            can warn. "Pin protects proactively, not absolutely" — pins resist
            the cap until the window is otherwise empty, then yield.

        Pin coverage by eviction path:
            evict()           → pin-aware (this function; Option B semantics)
            evict_last()      → pin-aware (skip_pinned=True by default)
            evict_ids()       → NOT pin-aware (explicit IDs = explicit choice)
            evict_since()     → pin-aware (skip_pinned=True by default)
            evict_older_than()→ pin-aware (skip_pinned=True by default)
            _strip_blocks_by_type() → pin-aware (skip_pinned kwarg)
            evict_write_pairs() → pin-aware (skip_pinned=True by default)

        The dynamic-tightening guarantee: when token usage rises back above
        the headroom threshold, slack disappears immediately. Rows above
        max_turns get FIFO-trimmed on the next turn that triggers evict().
        """
        deleted = 0
        pinned_force = 0

        # Compute effective row cap. Slack engages only when there's
        # measurable headroom against max_tokens — without max_tokens we
        # can't define headroom and slack stays disengaged (defensive).
        effective_max_turns = max_turns
        if row_cap_slack > 0 and max_tokens is not None and max_tokens > 0:
            cur = self.conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)) / 4, 0) FROM turns"
            )
            current_tokens = int(cur.fetchone()[0])
            if current_tokens < int(max_tokens * self.HEADROOM_RATIO):
                effective_max_turns = max_turns + int(row_cap_slack)

        # First pass: enforce effective turn-count cap.
        # Strategy: delete unpinned rows oldest-first until either the cap is
        # satisfied or no unpinned rows remain; then, if the cap still isn't
        # met, delete pinned rows oldest-first (last resort).
        # When preserve_compressed_reads=True, rows with compressed_read_stub=1
        # are also protected (same soft-protect semantics as pin_label).
        _stub_guard = (
            "AND (compressed_read_stub IS NULL OR compressed_read_stub = 0)"
            if preserve_compressed_reads
            else ""
        )
        cur = self.conn.execute("SELECT COUNT(*) FROM turns")
        n = cur.fetchone()[0]
        if n > effective_max_turns:
            to_delete = n - effective_max_turns
            # Phase A: try unpinned (and non-stub) rows first.
            cur = self.conn.execute(
                f"SELECT id FROM turns WHERE pin_label IS NULL {_stub_guard} ORDER BY id ASC LIMIT ?",
                (to_delete,),
            )
            unpinned_ids = [r[0] for r in cur.fetchall()]
            if unpinned_ids:
                placeholders = ",".join("?" * len(unpinned_ids))
                self.conn.execute(
                    f"DELETE FROM turns WHERE id IN ({placeholders})", unpinned_ids
                )
                self.conn.commit()
                deleted += len(unpinned_ids)
                to_delete -= len(unpinned_ids)
            # Phase B: last resort — evict pinned rows if cap still not met.
            if to_delete > 0:
                cur = self.conn.execute(
                    "SELECT id FROM turns WHERE pin_label IS NOT NULL ORDER BY id ASC LIMIT ?",
                    (to_delete,),
                )
                pinned_ids = [r[0] for r in cur.fetchall()]
                if pinned_ids:
                    placeholders = ",".join("?" * len(pinned_ids))
                    self.conn.execute(
                        f"DELETE FROM turns WHERE id IN ({placeholders})", pinned_ids
                    )
                    self.conn.commit()
                    deleted += len(pinned_ids)
                    pinned_force += len(pinned_ids)

        # Second pass: enforce token cap (content-size estimate, not API counts).
        # Unaffected by slack: this is the hard byte ceiling.
        # Same unpinned-first, pinned-last-resort ordering as the row-cap pass.
        if max_tokens is not None and max_tokens > 0:
            while True:
                cur = self.conn.execute(
                    "SELECT COALESCE(SUM(LENGTH(content)) / 4, 0) FROM turns"
                )
                total = int(cur.fetchone()[0])
                if total <= max_tokens:
                    break
                # Try oldest unpinned (and non-stub) row first.
                cur = self.conn.execute(
                    f"SELECT id FROM turns WHERE pin_label IS NULL {_stub_guard} ORDER BY id ASC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    self.conn.execute("DELETE FROM turns WHERE id = ?", (row[0],))
                    self.conn.commit()
                    deleted += 1
                    continue
                # No unpinned rows left — last resort: oldest pinned row.
                cur = self.conn.execute(
                    "SELECT id FROM turns WHERE pin_label IS NOT NULL ORDER BY id ASC LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    break  # store is empty, nothing more to do
                self.conn.execute("DELETE FROM turns WHERE id = ?", (row[0],))
                self.conn.commit()
                deleted += 1
                pinned_force += 1

        if deleted:
            self._bump_eviction_stats(rows=deleted, pinned_force=pinned_force)
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
        """Build a generic messages-style list from stored turns.

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
        n = len(ids)
        if n:
            self._bump_eviction_stats(rows=n)
        return n

    def evict_oldest(self, n: int, *, skip_pinned: bool = True) -> int:
        """Delete the N oldest (lowest-id) rows. Returns count actually deleted.

        This is the budget-reclaim primitive: stale history is removed while
        keeping the most-recent context intact — the opposite of evict_last.

        skip_pinned: if True (default) pinned rows are skipped when counting
            and selecting. Pass skip_pinned=False to include pinned rows.
        """
        if n <= 0:
            return 0
        if skip_pinned:
            cur = self.conn.execute(
                "SELECT id FROM turns WHERE pin_label IS NULL "
                "ORDER BY id ASC LIMIT ?",
                (int(n),),
            )
        else:
            cur = self.conn.execute(
                "SELECT id FROM turns ORDER BY id ASC LIMIT ?", (int(n),)
            )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM turns WHERE id IN ({placeholders})", ids
        )
        self.conn.commit()
        n = len(ids)
        if n:
            self._bump_eviction_stats(rows=n)
        return n

    def evict_by_role(self, role: str, *, skip_pinned: bool = True) -> int:
        """Delete ALL rows with a given role ('user' or 'assistant').

        Useful for bulk-clearing user prompts while keeping assistant responses
        (or vice-versa). Pinned rows are skipped by default.

        Returns count of rows actually deleted.
        """
        role = role.lower()
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        if skip_pinned:
            cur = self.conn.execute(
                "SELECT id FROM turns WHERE role = ? AND pin_label IS NULL",
                (role,),
            )
        else:
            cur = self.conn.execute(
                "SELECT id FROM turns WHERE role = ?", (role,)
            )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM turns WHERE id IN ({placeholders})", ids
        )
        self.conn.commit()
        n = len(ids)
        if n:
            self._bump_eviction_stats(rows=n)
        return n

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
        if existing:
            self._bump_eviction_stats(rows=existing)
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
        if n:
            self._bump_eviction_stats(rows=n)
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
            self._bump_eviction_stats(rows=n_evict)
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

    def _resolve_strip_target_ids(
        self,
        *,
        ids: list[int] | None,
        keep_recent: int | None,
        all_rows: bool,
        older_than_seconds: int | None,
        skip_pinned: bool,
    ) -> tuple[list[int], int]:
        """Resolve the target id set for a block-surgery operation.

        Validates the selector (exactly-one-of-four), resolves to a list of
        candidate row ids, then applies the pin filter. Returns
        (target_ids, rows_skipped_pinned). Shared by all evict_*_blocks
        public methods so the selector + pin semantics stay consistent.

        Raises ValueError on selector validation failure or non-integer ids.
        """
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

        return target_ids, rows_skipped_pinned

    def _strip_blocks_by_type(
        self,
        target_ids: list[int],
        block_type: str,
        rows_skipped_pinned: int,
    ) -> dict[str, int]:
        """Strip all blocks of the given type from the target rows.

        Caller is responsible for resolving target_ids via
        _resolve_strip_target_ids (which also produces rows_skipped_pinned
        passed through here so the return dict stays complete).

        Common rules across all block surgery:
          - non-list content (legacy strings, marker JSON strings) is
            scanned but skipped (no blocks to strip)
          - rows whose strip would leave 0 blocks are skipped without
            modification (empty content lists are not useful context)
          - returns {rows_scanned, rows_modified, blocks_evicted,
            bytes_freed, rows_skipped_pinned}
        """
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
                continue
            if not isinstance(blocks, list):
                continue
            new_blocks = [
                b for b in blocks
                if not (isinstance(b, dict) and b.get("type") == block_type)
            ]
            n_evicted = len(blocks) - len(new_blocks)
            if n_evicted == 0:
                continue
            if not new_blocks:
                # Don't leave a row with empty content. Agent can
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
            self._bump_eviction_stats(blocks=blocks_evicted, bytes_=bytes_freed)

        return {
            "rows_scanned": len(rows),
            "rows_modified": rows_modified,
            "blocks_evicted": blocks_evicted,
            "bytes_freed": bytes_freed,
            "rows_skipped_pinned": rows_skipped_pinned,
        }

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
        target_ids, rows_skipped_pinned = self._resolve_strip_target_ids(
            ids=ids,
            keep_recent=keep_recent,
            all_rows=all_rows,
            older_than_seconds=older_than_seconds,
            skip_pinned=skip_pinned,
        )
        return self._strip_blocks_by_type(target_ids, "thinking", rows_skipped_pinned)

    def evict_tool_use_blocks(
        self,
        *,
        ids: list[int] | None = None,
        keep_recent: int | None = None,
        all_rows: bool = False,
        older_than_seconds: int | None = None,
        skip_pinned: bool = True,
    ) -> dict[str, int]:
        """Strip 'tool_use' blocks from selected rolling-window rows.

        Block-level surgery for the largest bloat category in long sessions.
        In long agent sessions, tool_use specs (file paths, command strings,
        payload JSONs, edit before/after strings) typically dominate stored
        bytes — far more than thinking. Stripping them is the highest-
        impact context budget intervention available.

        Pairing safety note: Mnemara persists assistant-facing blocks as an
        audit trail, while tool results are surfaced live and summarized by
        the model's final text. Historical tool_use blocks are therefore
        safe to strip when they become context bloat. The skip-empty-row rule
        prevents accidentally leaving an empty content list on a turn that
        consisted entirely of tool_use blocks.

        Selectors and behavior identical to evict_thinking_blocks.

        Trade-offs to be aware of:
          - High byte savings: tool_use blocks average ~870 bytes each
            (vs. ~32 for thinking signature stubs).
          - Audit trail loss: stripping a tool_use block removes the
            model's record of what it called. The actual EFFECT (the
            commit, the file change, etc.) lives in git or wherever the
            tool wrote; only the CALL itself is gone from the rolling
            window. For most work this is fine; for sessions where the
            agent needs to remember "did I already call X?" within the
            window, prefer keep_recent or pinning specific rows.
          - The agent may want to write_memory(category='tool_audit')
            with a one-line summary of significant tool calls before
            evicting, the same opt-in pattern as thought_summary for
            thinking-block surgery.
        """
        target_ids, rows_skipped_pinned = self._resolve_strip_target_ids(
            ids=ids,
            keep_recent=keep_recent,
            all_rows=all_rows,
            older_than_seconds=older_than_seconds,
            skip_pinned=skip_pinned,
        )
        return self._strip_blocks_by_type(target_ids, "tool_use", rows_skipped_pinned)

    # ------------------------------------------- read/write pair eviction
    # Distinct surgery shape from evict_thinking_blocks / evict_tool_use_blocks:
    # those strip ENTIRE blocks of a given type. evict_write_pairs preserves
    # the block but stubs only its `input` field down to a small audit-trail
    # ({file_path, _evicted: true}), keeping the model's record that "I
    # called Edit on /foo/bar.py" while dropping the bulky old_string /
    # new_string / content payloads. Designed to run automatically after
    # each turn that did writes (cfg.auto_evict_after_write).
    #
    # The "pair" semantics: when an Edit/Write happens for /foo/bar.py,
    # any prior Read tool_use for the same file is also stubbed. Read
    # specs are usually small (just `file_path`), so this mostly just
    # marks them as superseded for audit clarity. The major savings come
    # from the Edit/Write side.

    # Tool-name conventions. Built-in file tools use these names;
    # mnemara's MCP-spawned tools never collide because they're
    # prefixed with mcp__. NotebookEdit uses 'edits' or 'cell_source'
    # depending on cell type; we strip both. MultiEdit stores 'edits'
    # (a list of dicts with old_string/new_string per edit).
    _WRITE_TOOL_NAMES: frozenset[str] = frozenset({
        "Edit", "Write", "MultiEdit", "NotebookEdit",
    })
    _READ_TOOL_NAMES: frozenset[str] = frozenset({"Read"})
    # Field names within tool_use.input that hold bulky body content.
    # Stripping these (replacing with `_evicted: true`) preserves the
    # block structure + the audit trail (file_path stays) while freeing
    # the kilobytes-per-block bloat.
    _BULKY_INPUT_FIELDS: frozenset[str] = frozenset({
        "content",      # Write
        "old_string",   # Edit, MultiEdit (per-edit)
        "new_string",   # Edit, MultiEdit (per-edit)
        "edits",        # MultiEdit (entire list)
        "cell_source",  # NotebookEdit
        "new_source",   # NotebookEdit
    })

    def evict_write_pairs(
        self,
        *,
        only_in_rows: list[int] | None = None,
        skip_pinned: bool = True,
    ) -> dict[str, int]:
        """Stub bulky body content from Edit/Write tool_use blocks + matching prior Reads.

        For each tool_use block in scope whose `name` is one of
        Edit/Write/MultiEdit/NotebookEdit, strip the bulky body fields
        (old_string, new_string, content, edits, cell_source, new_source)
        from its `input` dict, replacing them with `_evicted: true`.
        The `file_path` is preserved so the audit trail says "I edited
        /foo/bar.py" without holding the kilobytes of before/after.

        For each unique file_path collected from the writes, scan rows
        OLDER than the writing row for prior Read tool_use blocks
        targeting the same file_path, and stub them the same way (the
        stale read content is no longer current after the edit; the
        audit trail "I read /foo/bar.py" stays intact).

        Args:
          only_in_rows: if provided, restrict the WRITE scan to these
            rows. Read scan still searches ALL rows older than each
            write. Used by auto-evict-after-write to scope to the
            current-turn assistant row. None = scan all rows for writes.
          skip_pinned: if True (default), pinned rows are excluded from
            both write and read scans. Pin protection covers the
            advisory-not-locked semantics consistent with other eviction.

        Returns:
          {
            "writes_stubbed":   tool_use blocks stubbed in write tools,
            "reads_stubbed":    tool_use blocks stubbed in read tools,
            "rows_modified":    distinct rows whose content was rewritten,
            "bytes_freed":      sum of (old_content_len - new_content_len)
                                 across modified rows,
            "files_seen":       distinct file_paths processed,
            "rows_skipped_pinned": rows that matched the scope but were
                                 skipped because pinned,
          }

        Idempotent: re-running stubs nothing new (already-stubbed inputs
        keep their `_evicted: true` marker; the bulky fields are gone).
        """
        # Resolve the row set to scan for WRITES (the trigger source).
        if only_in_rows is not None:
            try:
                write_scan_ids = [int(i) for i in only_in_rows]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"only_in_rows must be integers: {exc}") from exc
            if not write_scan_ids:
                return {
                    "writes_stubbed": 0, "reads_stubbed": 0, "rows_modified": 0,
                    "bytes_freed": 0, "files_seen": 0, "rows_skipped_pinned": 0,
                }
            placeholders = ",".join("?" * len(write_scan_ids))
            cur = self.conn.execute(
                f"SELECT id, content, pin_label FROM turns "
                f"WHERE id IN ({placeholders}) AND role='assistant' "
                f"ORDER BY id ASC",
                write_scan_ids,
            )
        else:
            cur = self.conn.execute(
                "SELECT id, content, pin_label FROM turns "
                "WHERE role='assistant' ORDER BY id ASC"
            )

        # Per-row mutation accumulator: row_id -> (orig_content_str, new_blocks)
        # We rewrite each row at most once, accumulating both write-side and
        # read-side stubs into a single UPDATE so bytes_freed is calculated
        # against the ORIGINAL content. This keeps the row-modification
        # count honest (a row that has both a Write and a stale Read
        # counts as one row_modified, not two).
        pending: dict[int, dict[str, Any]] = {}
        # (file_path, max_write_row_id) so we know how far back to scan
        # for stale Reads. A Read at row 50 paired with a Write at row 80
        # gets stubbed; a Read at row 90 (after the Write) doesn't.
        files_max_write_id: dict[str, int] = {}
        rows_skipped_pinned = 0

        for row_id, content_str, pin_label in cur.fetchall():
            if skip_pinned and pin_label is not None:
                rows_skipped_pinned += 1
                continue
            try:
                blocks = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(blocks, list):
                continue
            new_blocks, n_stubbed, files_seen_in_row = self._stub_write_blocks(blocks)
            if n_stubbed > 0:
                pending[row_id] = {
                    "orig": content_str,
                    "blocks": new_blocks,
                    "writes": n_stubbed,
                    "reads": 0,
                }
                for fp in files_seen_in_row:
                    prior = files_max_write_id.get(fp, -1)
                    if row_id > prior:
                        files_max_write_id[fp] = row_id

        # If no writes found in scope, nothing to do.
        if not pending:
            return {
                "writes_stubbed": 0, "reads_stubbed": 0, "rows_modified": 0,
                "bytes_freed": 0, "files_seen": 0,
                "rows_skipped_pinned": rows_skipped_pinned,
            }

        # Read scan: for each (file_path, max_write_id), find prior Read
        # tool_use blocks for that file_path in rows older than the write.
        # Read scan covers ALL rows (not just only_in_rows), because stale
        # Reads naturally live in earlier turns.
        for file_path, max_write_id in files_max_write_id.items():
            cur = self.conn.execute(
                "SELECT id, content, pin_label FROM turns "
                "WHERE id < ? AND role='assistant' ORDER BY id ASC",
                (max_write_id,),
            )
            for row_id, content_str, pin_label in cur.fetchall():
                if skip_pinned and pin_label is not None:
                    # Counted in rows_skipped_pinned only if we WOULD have
                    # otherwise scanned + modified the row. We don't double-
                    # count rows already counted in the write scan.
                    continue
                # Use the already-pending blocks if this row has them; else
                # parse fresh from disk.
                if row_id in pending:
                    blocks = pending[row_id]["blocks"]
                    orig = pending[row_id]["orig"]
                else:
                    try:
                        blocks = json.loads(content_str)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(blocks, list):
                        continue
                    orig = content_str
                new_blocks, n_stubbed = self._stub_read_blocks_for_file(
                    blocks, file_path
                )
                if n_stubbed > 0:
                    if row_id in pending:
                        pending[row_id]["blocks"] = new_blocks
                        pending[row_id]["reads"] += n_stubbed
                    else:
                        pending[row_id] = {
                            "orig": orig,
                            "blocks": new_blocks,
                            "writes": 0,
                            "reads": n_stubbed,
                        }

        # Single UPDATE per row, calculating bytes_freed against the
        # ORIGINAL content string captured at the start of pending entry.
        rows_modified = 0
        writes_stubbed = 0
        reads_stubbed = 0
        bytes_freed = 0
        for row_id, entry in pending.items():
            new_str = json.dumps(entry["blocks"])
            delta = len(entry["orig"]) - len(new_str)
            self.conn.execute(
                "UPDATE turns SET content=? WHERE id=?",
                (new_str, row_id),
            )
            rows_modified += 1
            writes_stubbed += entry["writes"]
            reads_stubbed += entry["reads"]
            bytes_freed += delta

        if rows_modified:
            self.conn.commit()
            self._bump_eviction_stats(
                blocks=writes_stubbed + reads_stubbed,
                bytes_=bytes_freed,
            )

        return {
            "writes_stubbed": writes_stubbed,
            "reads_stubbed": reads_stubbed,
            "rows_modified": rows_modified,
            "bytes_freed": bytes_freed,
            "files_seen": len(files_max_write_id),
            "rows_skipped_pinned": rows_skipped_pinned,
        }

    def _stub_write_blocks(
        self, blocks: list[Any]
    ) -> tuple[list[Any], int, set[str]]:
        """Stub Edit/Write/MultiEdit/NotebookEdit tool_use blocks in-place.

        Returns (new_blocks, count_stubbed, files_seen).
        Already-stubbed blocks (those whose input has '_evicted': True)
        are skipped to keep the operation idempotent.
        """
        new_blocks: list[Any] = []
        n_stubbed = 0
        files_seen: set[str] = set()
        for b in blocks:
            if not (
                isinstance(b, dict)
                and b.get("type") == "tool_use"
                and b.get("name") in self._WRITE_TOOL_NAMES
            ):
                new_blocks.append(b)
                continue
            inp = b.get("input")
            if not isinstance(inp, dict):
                new_blocks.append(b)
                continue
            if inp.get("_evicted") is True:
                # Already stubbed; idempotent skip. Still record the
                # file_path for read-pair scan even on re-runs.
                fp = inp.get("file_path") or inp.get("notebook_path")
                if isinstance(fp, str) and fp:
                    files_seen.add(fp)
                new_blocks.append(b)
                continue
            # Build a stubbed input preserving file_path + audit marker.
            fp = inp.get("file_path") or inp.get("notebook_path")
            stub_input: dict[str, Any] = {"_evicted": True}
            if isinstance(fp, str) and fp:
                stub_input["file_path"] = fp
                files_seen.add(fp)
            new_b = dict(b)
            new_b["input"] = stub_input
            new_blocks.append(new_b)
            n_stubbed += 1
        return new_blocks, n_stubbed, files_seen

    def _stub_read_blocks_for_file(
        self, blocks: list[Any], file_path: str
    ) -> tuple[list[Any], int]:
        """Stub Read tool_use blocks targeting `file_path` in-place.

        Returns (new_blocks, count_stubbed). Reads for OTHER files in
        the same row are left untouched. Already-stubbed reads are
        skipped (idempotent).
        """
        new_blocks: list[Any] = []
        n_stubbed = 0
        for b in blocks:
            if not (
                isinstance(b, dict)
                and b.get("type") == "tool_use"
                and b.get("name") in self._READ_TOOL_NAMES
            ):
                new_blocks.append(b)
                continue
            inp = b.get("input")
            if not isinstance(inp, dict):
                new_blocks.append(b)
                continue
            if inp.get("file_path") != file_path:
                new_blocks.append(b)
                continue
            if inp.get("_evicted") is True:
                new_blocks.append(b)
                continue
            stub_input: dict[str, Any] = {
                "file_path": file_path,
                "_evicted": True,
            }
            new_b = dict(b)
            new_b["input"] = stub_input
            new_blocks.append(new_b)
            n_stubbed += 1
        return new_blocks, n_stubbed

    def stamp_read_cache(
        self,
        file_path: str,
        content: str,
        content_hash: str,
    ) -> bool:
        """Stamp _cached_content and _cached_hash into the most-recent persisted
        tool_use block for the given file_path.

        Called from the UserMessage callback in _run_turn when a Read
        tool_result arrives.  The tool_result itself is NOT persisted (by
        design); instead we back-patch the already-persisted tool_use block
        so downstream analysis (compress_repeated_reads, collect_read_stats)
        can walk tool_use rows and find the content without needing
        tool_result rows.

        Returns True if a matching row was found and updated, False otherwise.
        """
        # Find the most-recently inserted tool_use block with name=="Read"
        # and input.file_path == file_path.
        rows = self.conn.execute(
            "SELECT id, content FROM turns ORDER BY id DESC"
        ).fetchall()

        for row_id, content_str in rows:
            try:
                blocks = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(blocks, list):
                continue
            # Scan blocks in reverse to find the latest matching tool_use
            for i in range(len(blocks) - 1, -1, -1):
                blk = blocks[i]
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") != "tool_use":
                    continue
                if blk.get("name") != "Read":
                    continue
                inp = blk.get("input")
                if not isinstance(inp, dict):
                    continue
                fp = inp.get("file_path") or inp.get("path")
                if fp != file_path:
                    continue
                # Found the matching block — stamp it
                new_inp = dict(inp)
                new_inp["_cached_content"] = content
                new_inp["_cached_hash"] = content_hash
                blocks[i] = dict(blk)
                blocks[i]["input"] = new_inp
                new_str = json.dumps(blocks)
                self.conn.execute(
                    "UPDATE turns SET content=? WHERE id=?",
                    (new_str, row_id),
                )
                self.conn.commit()
                return True
        return False

    def compress_repeated_reads(
        self,
        skip_pinned: bool = True,
        min_bytes: int = 200,
        preserve_compressed_reads: bool = False,
    ) -> dict:
        """Compress historical Read tool_use blocks for files read multiple times.

        Walks stored tool_use blocks that have _cached_content stamped by
        stamp_read_cache().  For each file that has been Read 2+ times in the
        rolling window, keep the LAST Read at full fidelity and replace earlier
        Reads with:
          - an "unchanged" pointer if the content hash matches the last Read
          - a unified diff against the last Read otherwise

        This frees context budget without breaking Claude's pre-write
        old_string exact-match (which references the most-recent Read).

        Args:
          skip_pinned: if True (default), rows with pin_label IS NOT NULL
            are skipped over and their Read tool_use blocks are preserved.
          min_bytes: minimum content length to compress. Reads shorter than
            this are kept full (compression overhead exceeds savings).
          preserve_compressed_reads: if True, mark modified rows with
            compressed_read_stub=1 so evict() will skip them when called
            with preserve_compressed_reads=True.

        Returns:
          {"reads_compressed": N, "bytes_freed": M}
        """
        # Walk all stored turns chronologically; collect tool_use blocks
        # with name=="Read" and _cached_content in their input dict.
        # These were stamped by stamp_read_cache() when the tool_result arrived.

        all_rows = list(self.conn.execute(
            "SELECT id, content, pin_label, compressed_read_stub FROM turns ORDER BY id ASC"
        ))

        # For each file_path, accumulate entries in chronological order.
        # Each entry: {turn_id, block_index, content_hash, full_content, pin_label, compressed}
        file_reads: dict[str, list[dict]] = {}

        for row_id, content_str, pin_label, compressed_stub in all_rows:
            try:
                blocks = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(blocks, list):
                continue

            for block_idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                # Case-sensitive "Read" — excludes "read_skeleton"
                if block.get("name") != "Read":
                    continue
                inp = block.get("input")
                if not isinstance(inp, dict):
                    continue
                # Only process blocks stamped with _cached_content
                cached_content = inp.get("_cached_content")
                if cached_content is None:
                    continue
                cached_hash = inp.get("_cached_hash", "")
                fp = inp.get("file_path") or inp.get("path")
                if not fp or not isinstance(fp, str):
                    continue

                entry = {
                    "turn_id": row_id,
                    "block_index": block_idx,
                    "content_hash": cached_hash,
                    "full_content": cached_content,
                    "pin_label": pin_label,
                    "compressed_stub": compressed_stub,
                }
                file_reads.setdefault(fp, []).append(entry)

        reads_compressed = 0
        bytes_freed = 0

        for file_path, reads in file_reads.items():
            if len(reads) < 2:
                continue

            # Find the last non-stubbed entry as the canonical reference
            last_entry = None
            for entry in reversed(reads):
                c = entry["full_content"]
                if c.startswith("(see turn") or c.startswith("(historical state"):
                    continue
                last_entry = entry
                break

            if last_entry is None:
                continue

            last_turn_id = last_entry["turn_id"]
            last_hash = last_entry["content_hash"]
            last_content = last_entry["full_content"]

            # Compress all earlier entries
            for entry in reads:
                if entry is last_entry:
                    continue

                turn_id = entry["turn_id"]
                content = entry["full_content"]
                block_index = entry["block_index"]
                entry_pin = entry["pin_label"]

                # Pin check
                if skip_pinned and entry_pin is not None:
                    continue

                # Already stubbed — idempotent skip
                if content.startswith("(see turn") or content.startswith("(historical state"):
                    continue

                # Binary check: skip if \x00 bytes present
                if "\x00" in content:
                    continue

                # Size check
                if len(content) < min_bytes:
                    continue

                # Build the replacement stub
                if entry["content_hash"] == last_hash:
                    stub = f"(see turn {last_turn_id} — content unchanged at this point)"
                else:
                    diff_lines = list(difflib.unified_diff(
                        content.splitlines(keepends=True),
                        last_content.splitlines(keepends=True),
                        fromfile=f"turn {turn_id}",
                        tofile=f"turn {last_turn_id}",
                    ))
                    stub = (
                        f"(historical state at turn {turn_id}; "
                        f"diff vs latest read at turn {last_turn_id}:\n"
                        + "".join(diff_lines)
                        + ")"
                    )

                # Load the row, update the specific tool_use block's input
                row_data = self.conn.execute(
                    "SELECT content FROM turns WHERE id = ?", (turn_id,)
                ).fetchone()
                if not row_data:
                    continue

                try:
                    row_blocks = json.loads(row_data[0])
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(row_blocks, list):
                    continue

                orig_str = row_data[0]
                modified = False
                new_blocks = list(row_blocks)
                # Update the tool_use block at block_index
                if block_index < len(new_blocks):
                    target_blk = new_blocks[block_index]
                    if (
                        isinstance(target_blk, dict)
                        and target_blk.get("type") == "tool_use"
                        and target_blk.get("name") == "Read"
                    ):
                        new_blk = dict(target_blk)
                        new_inp = dict(target_blk.get("input") or {})
                        new_inp["_cached_content"] = stub
                        new_blk["input"] = new_inp
                        new_blocks[block_index] = new_blk
                        modified = True

                if not modified:
                    continue

                new_str = json.dumps(new_blocks)
                delta = len(orig_str) - len(new_str)
                self.conn.execute(
                    "UPDATE turns SET content=? WHERE id=?",
                    (new_str, turn_id),
                )
                if preserve_compressed_reads:
                    self.conn.execute(
                        "UPDATE turns SET compressed_read_stub=1 WHERE id=?",
                        (turn_id,),
                    )
                reads_compressed += 1
                bytes_freed += delta

        if reads_compressed:
            self.conn.commit()
            self._bump_eviction_stats(blocks=reads_compressed, bytes_=bytes_freed)

        return {"reads_compressed": reads_compressed, "bytes_freed": bytes_freed}

    def collect_read_stats(self) -> dict[str, dict]:
        """Walk all turns chronologically and collect stats about Read tool calls.

        For each file_path that was Read, returns the MOST RECENT read's:
          last_read_turn (int): the turn id of the most recent Read result
          last_hash (str):      sha256[:8] of the content at that read (from _cached_hash)
          last_size (int):      len(_cached_content) at that read

        Walks tool_use blocks that have been stamped with _cached_content by
        stamp_read_cache().  tool_result blocks are NOT persisted, so this
        method operates on the enriched tool_use input dicts.

        Returns an empty dict if no stamped Read turns exist.
        """
        all_rows = list(self.conn.execute(
            "SELECT id, content FROM turns ORDER BY id ASC"
        ))

        # file_path -> {last_read_turn, last_hash, last_size}
        result: dict[str, dict] = {}

        for row_id, content_str in all_rows:
            try:
                blocks = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(blocks, list):
                continue

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                if block.get("name") != "Read":
                    continue
                inp = block.get("input")
                if not isinstance(inp, dict):
                    continue
                # Only count stamped blocks
                cached_content = inp.get("_cached_content")
                if cached_content is None:
                    continue
                fp = inp.get("file_path") or inp.get("path")
                if not fp or not isinstance(fp, str):
                    continue
                cached_hash = inp.get("_cached_hash", "")
                # Overwrite on each pass — keeps only the MOST RECENT read
                result[fp] = {
                    "last_read_turn": row_id,
                    "last_hash": cached_hash,
                    "last_size": len(str(cached_content)),
                }

        return result

    def list_window(
        self,
        limit: int = 50,
        offset: int = 0,
        role: str = "",
    ) -> dict:
        """Return rolling-window rows with lightweight summaries.

        Args:
          limit:  max rows to return (capped at 200).
          offset: skip this many rows (for pagination).
          role:   if non-empty, filter to 'user' or 'assistant' only.

        Returns:
          {
            "ok": True,
            "rows": [{"row_id": int, "timestamp": str, "role": str, "summary": str}, ...],
            "total": int,  # total matching rows (before limit/offset)
          }

        Sort: most recent first (highest row_id first).
        Summary: first 80 chars of flattened text content.
        For assistant rows with structured block content, extracts the first
        text block's text, or returns "[tool_use <name>]" for tool_use blocks.
        """
        limit = min(int(limit), 200)
        offset = max(int(offset), 0)

        if role:
            count_cur = self.conn.execute(
                "SELECT COUNT(*) FROM turns WHERE role=?", (role,)
            )
        else:
            count_cur = self.conn.execute("SELECT COUNT(*) FROM turns")
        total = int(count_cur.fetchone()[0])

        if role:
            rows_cur = self.conn.execute(
                "SELECT id, ts, role, content, pin_label FROM turns "
                "WHERE role=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (role, limit, offset),
            )
        else:
            rows_cur = self.conn.execute(
                "SELECT id, ts, role, content, pin_label FROM turns "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )

        rows = []
        for row_id, ts, row_role, content_str, pin_label in rows_cur.fetchall():
            summary = _window_summary(row_role, content_str)
            rows.append({
                "row_id": row_id,
                "timestamp": ts,
                "role": row_role,
                "summary": summary,
                "pin_label": pin_label,
            })

        return {"ok": True, "rows": rows, "total": total}

    def get_turn(self, row_id: int) -> dict | None:
        """Return a single turn row by id, or None if not found.

        Returns the same shape as window() entries plus pin_label.
        """
        cur = self.conn.execute(
            "SELECT id, ts, role, content, tool_uses, tokens_in, tokens_out, pin_label "
            "FROM turns WHERE id=?",
            (int(row_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "ts": row[1],
            "role": row[2],
            "content": _maybe_json(row[3]),
            "tool_uses": json.loads(row[4]) if row[4] else None,
            "tokens_in": row[5],
            "tokens_out": row[6],
            "pin_label": row[7],
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


def _window_summary(role: str, content_str: str) -> str:
    """Build an 80-char summary from a raw content string.

    For assistant rows with JSON-encoded block lists, extracts the first
    text block's text, or returns "[tool_use <name>]" for tool_use blocks.
    Falls back to the first 80 chars of the raw string on any failure.
    """
    if role == "assistant":
        try:
            blocks = json.loads(content_str)
            if isinstance(blocks, list) and blocks:
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type")
                    if btype == "text":
                        return (b.get("text") or "")[:80]
                    if btype == "tool_use":
                        name = b.get("name") or "?"
                        return f"[tool_use {name}]"
                # No matching block found; fall through to raw
        except (json.JSONDecodeError, TypeError):
            pass
    return content_str[:80]


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
