"""v0.11.0 — autonomous peer-poll tests (v0.12.0 detect/process split).

Detection is now token-free (pure SQLite → _peer_pending_rows).
Processing is lazy — all pending rows batched into ONE LLM turn when idle.

Test coverage:
- Config defaults (interval now 30 s)
- Detection fills _peer_pending_rows without LLM call when busy
- Detection fires _process_peer_messages immediately when idle
- _process_peer_messages batches N rows into 1 _handle_user_input call
- _process_peer_messages is a no-op when _peer_pending_rows is empty
- ⚡ N badge appears in status text when rows are pending
- Non-watched roles, non-pending status, duplicate rows all skipped
- Missing muninn.db handled gracefully
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_muninn_db(tmp_path: Path) -> Path:
    """Create a minimal muninn.db with the returns table."""
    db = tmp_path / "muninn.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE returns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL,
            agent_role    TEXT NOT NULL,
            task_id       TEXT,
            submitted_at  TEXT NOT NULL,
            payload_json  TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'pending',
            processed_at  TEXT,
            completed_at  TEXT,
            recipient_role TEXT
        )
    """)
    conn.execute("CREATE INDEX idx_returns_status ON returns(status)")
    conn.execute("CREATE INDEX idx_returns_recipient ON returns(recipient_role)")
    conn.commit()
    conn.close()
    return db


def _seed_return(db: Path, *, agent_role: str, task_id: str, payload: dict,
                 status: str = "pending", recipient_role: str | None = None) -> int:
    """Insert a row into the returns table and return its id."""
    conn = sqlite3.connect(str(db))
    cur = conn.execute(
        "INSERT INTO returns (session_id, agent_role, task_id, submitted_at, "
        "payload_json, status, recipient_role) VALUES (?,?,?,datetime('now'),?,?,?)",
        ("test-session", agent_role, task_id, json.dumps(payload), status, recipient_role),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _make_tui(tmp_path: Path, db_path: str, peer_roles: str = "theseus",
              interval: int = 30, busy: bool = False) -> MagicMock:
    """Build a minimal MnemaraTUI-like mock that exercises the poll/process methods."""
    from mnemara import config as config_mod

    cfg = config_mod.Config.from_dict({
        "peer_poll_enabled": True,
        "peer_poll_interval_seconds": interval,
        "architect_db_path": db_path,
        "peer_poll_roles": peer_roles,
        "model": "claude-sonnet-4-6",
    })

    tui = MagicMock()
    tui.cfg = cfg
    tui.instance = "substrate"
    tui._busy = busy
    tui._delivered_peer_row_ids = set()
    tui._peer_pending_rows = []
    tui._chat.return_value = MagicMock()
    tui._handle_user_input = AsyncMock()
    tui._refresh_status = MagicMock()
    return tui


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_peer_poll_config_defaults() -> None:
    """Default interval is now 30 s (was 90); other defaults unchanged."""
    from mnemara import config as config_mod
    cfg = config_mod.Config.from_dict({})
    assert cfg.peer_poll_enabled is False
    assert cfg.peer_poll_interval_seconds == 30   # changed from 90
    assert cfg.architect_db_path == ""
    assert cfg.peer_poll_roles == "theseus"


def test_peer_poll_config_roundtrip() -> None:
    """Config serialises and deserialises all peer-poll fields cleanly."""
    from mnemara import config as config_mod
    original = config_mod.Config.from_dict({
        "peer_poll_enabled": True,
        "peer_poll_interval_seconds": 60,
        "architect_db_path": "/tmp/muninn.db",
        "peer_poll_roles": "theseus,majordomo",
    })
    raw = original.to_dict()
    restored = config_mod.Config.from_dict(raw)
    assert restored.peer_poll_enabled is True
    assert restored.peer_poll_interval_seconds == 60
    assert restored.architect_db_path == "/tmp/muninn.db"
    assert restored.peer_poll_roles == "theseus,majordomo"


# ---------------------------------------------------------------------------
# Detection tests (_poll_peer_messages — no LLM, pure SQLite)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_detect_fills_pending_no_lm_when_busy(tmp_path: Path) -> None:
    """When busy, detection claims the row into _peer_pending_rows but does NOT
    call _handle_user_input — processing defers until the turn completes."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(db, agent_role="theseus", task_id="t1",
                          payload={"msg": "queued"})

    tui = _make_tui(tmp_path, str(db), busy=True)
    # _process_peer_messages is a real async method; mock it on the tui object
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    # Row must be claimed (will not be re-detected next tick).
    assert row_id in tui._delivered_peer_row_ids
    # Row must be in the pending list waiting for next idle window.
    assert any(r["row_id"] == row_id for r in tui._peer_pending_rows)
    # _process_peer_messages must NOT have fired (agent was busy).
    tui._process_peer_messages.assert_not_awaited()
    # _handle_user_input must NOT have fired.
    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_detect_triggers_process_when_idle(tmp_path: Path) -> None:
    """When idle, detection immediately calls _process_peer_messages."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(db, agent_role="theseus", task_id="t1",
                          payload={"msg": "hello"})

    tui = _make_tui(tmp_path, str(db), busy=False)
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    assert row_id in tui._delivered_peer_row_ids
    tui._process_peer_messages.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_skips_non_watched_roles(tmp_path: Path) -> None:
    """Rows from roles not in peer_poll_roles are ignored."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    _seed_return(db, agent_role="designer", task_id="t1",
                 payload={"msg": "irrelevant"})

    tui = _make_tui(tmp_path, str(db), peer_roles="theseus")
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    assert len(tui._peer_pending_rows) == 0
    tui._process_peer_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_skips_non_pending_rows(tmp_path: Path) -> None:
    """Rows with status != 'pending' are not detected."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    _seed_return(db, agent_role="theseus", task_id="done-row",
                 payload={"msg": "already done"}, status="done")

    tui = _make_tui(tmp_path, str(db))
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    assert len(tui._peer_pending_rows) == 0
    tui._process_peer_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_skips_already_delivered(tmp_path: Path) -> None:
    """A row already in _delivered_peer_row_ids is not re-claimed."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(db, agent_role="theseus", task_id="t1",
                          payload={"msg": "hi"})

    tui = _make_tui(tmp_path, str(db))
    tui._delivered_peer_row_ids.add(row_id)
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    assert len(tui._peer_pending_rows) == 0
    tui._process_peer_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_batches_multiple_rows_in_one_detection(tmp_path: Path) -> None:
    """Multiple pending rows are all claimed in a single detection pass."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id_1 = _seed_return(db, agent_role="theseus", task_id="t1", payload={"n": 1})
    row_id_2 = _seed_return(db, agent_role="theseus", task_id="t2", payload={"n": 2})
    row_id_3 = _seed_return(db, agent_role="majordomo", task_id="t3", payload={"n": 3})

    tui = _make_tui(tmp_path, str(db), peer_roles="theseus,majordomo")
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    # All three rows claimed in one detection pass.
    assert {row_id_1, row_id_2, row_id_3} == tui._delivered_peer_row_ids
    assert len(tui._peer_pending_rows) == 3
    # _process_peer_messages called exactly once (not once per row).
    tui._process_peer_messages.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_multi_role_filter(tmp_path: Path) -> None:
    """peer_poll_roles='theseus,majordomo' watches both roles, ignores others."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_theseus = _seed_return(db, agent_role="theseus", task_id="t",
                               payload={"x": 1})
    _seed_return(db, agent_role="cognition-researcher", task_id="c",
                 payload={"x": 2})  # not watched

    tui = _make_tui(tmp_path, str(db), peer_roles="theseus,majordomo")
    tui._process_peer_messages = AsyncMock()

    await MnemaraTUI._poll_peer_messages(tui)

    assert row_theseus in tui._delivered_peer_row_ids
    assert len(tui._peer_pending_rows) == 1


@pytest.mark.asyncio
async def test_poll_handles_missing_db_gracefully(tmp_path: Path) -> None:
    """Non-existent muninn.db: poll returns silently (no crash)."""
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, str(tmp_path / "nonexistent.db"))
    tui._process_peer_messages = AsyncMock()

    # Should not raise.
    await MnemaraTUI._poll_peer_messages(tui)
    assert len(tui._peer_pending_rows) == 0
    tui._process_peer_messages.assert_not_awaited()


# ---------------------------------------------------------------------------
# Processing tests (_process_peer_messages — batched LLM turn)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_batches_n_rows_into_one_turn(tmp_path: Path) -> None:
    """_process_peer_messages batches 3 pending rows into exactly 1 LLM call."""
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, "/unused.db")
    tui._peer_pending_rows = [
        {"row_id": 10, "sender_role": "theseus", "task_id": "topic-A",
         "payload": {"type": "directive"}, "submitted_at": "2026-05-08"},
        {"row_id": 11, "sender_role": "majordomo", "task_id": "topic-B",
         "payload": {"type": "status"}, "submitted_at": "2026-05-08"},
        {"row_id": 12, "sender_role": "cognition-researcher", "task_id": "topic-C",
         "payload": {"type": "observation"}, "submitted_at": "2026-05-08"},
    ]

    await MnemaraTUI._process_peer_messages(tui)

    # Exactly ONE LLM call regardless of row count.
    tui._handle_user_input.assert_awaited_once()
    # Pending list is drained.
    assert tui._peer_pending_rows == []

    msg = tui._handle_user_input.call_args[0][0]
    # All three row_ids present in one message.
    assert "row_id=10" in msg
    assert "row_id=11" in msg
    assert "row_id=12" in msg
    # All three senders present.
    assert "theseus" in msg
    assert "majordomo" in msg
    assert "cognition-researcher" in msg
    # Must tell agent to ack and reply.
    assert "mcp__architect__ack_return" in msg
    assert "mcp__architect__submit_return" in msg
    # Header shows total count.
    assert "3 pending" in msg


@pytest.mark.asyncio
async def test_process_noop_when_no_pending(tmp_path: Path) -> None:
    """_process_peer_messages is a no-op when _peer_pending_rows is empty."""
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, "/unused.db")
    tui._peer_pending_rows = []

    await MnemaraTUI._process_peer_messages(tui)

    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_noop_when_busy(tmp_path: Path) -> None:
    """_process_peer_messages does nothing when agent is busy."""
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, "/unused.db", busy=True)
    tui._peer_pending_rows = [
        {"row_id": 5, "sender_role": "theseus", "task_id": "t",
         "payload": {}, "submitted_at": "2026-05-08"},
    ]

    await MnemaraTUI._process_peer_messages(tui)

    # Rows must remain for when busy clears.
    assert len(tui._peer_pending_rows) == 1
    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_turn_by_turn_consumes_one_row(tmp_path: Path) -> None:
    """peer_poll_batch=False: _process_peer_messages fires ONE turn per call."""
    from mnemara import config as config_mod
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, "/unused.db")
    tui.cfg = config_mod.Config.from_dict({
        "peer_poll_batch": False,
        "peer_poll_enabled": True,
        "architect_db_path": "/unused.db",
        "model": "claude-sonnet-4-6",
    })
    tui._peer_pending_rows = [
        {"row_id": 20, "sender_role": "theseus", "task_id": "first",
         "payload": {"n": 1}, "submitted_at": "2026-05-08"},
        {"row_id": 21, "sender_role": "theseus", "task_id": "second",
         "payload": {"n": 2}, "submitted_at": "2026-05-08"},
        {"row_id": 22, "sender_role": "theseus", "task_id": "third",
         "payload": {"n": 3}, "submitted_at": "2026-05-08"},
    ]

    await MnemaraTUI._process_peer_messages(tui)

    # Exactly one LLM call fired.
    tui._handle_user_input.assert_awaited_once()
    # Only the first row consumed; two remain.
    assert len(tui._peer_pending_rows) == 2
    assert tui._peer_pending_rows[0]["row_id"] == 21
    # The message targets only the first row.
    msg = tui._handle_user_input.call_args[0][0]
    assert "row_id=20" in msg
    assert "row_id=21" not in msg
    assert "row_id=22" not in msg


@pytest.mark.asyncio
async def test_process_turn_by_turn_drains_sequentially(tmp_path: Path) -> None:
    """peer_poll_batch=False: three calls drain three rows one at a time."""
    from mnemara import config as config_mod
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, "/unused.db")
    tui.cfg = config_mod.Config.from_dict({
        "peer_poll_batch": False,
        "peer_poll_enabled": True,
        "architect_db_path": "/unused.db",
        "model": "claude-sonnet-4-6",
    })
    tui._peer_pending_rows = [
        {"row_id": 30, "sender_role": "theseus", "task_id": "a",
         "payload": {}, "submitted_at": "2026-05-08"},
        {"row_id": 31, "sender_role": "theseus", "task_id": "b",
         "payload": {}, "submitted_at": "2026-05-08"},
        {"row_id": 32, "sender_role": "theseus", "task_id": "c",
         "payload": {}, "submitted_at": "2026-05-08"},
    ]

    # Simulate three idle windows (e.g. _send_turn finally block fires 3x).
    await MnemaraTUI._process_peer_messages(tui)
    assert len(tui._peer_pending_rows) == 2

    await MnemaraTUI._process_peer_messages(tui)
    assert len(tui._peer_pending_rows) == 1

    await MnemaraTUI._process_peer_messages(tui)
    assert tui._peer_pending_rows == []

    # Three turns fired, one per call.
    assert tui._handle_user_input.await_count == 3


# ---------------------------------------------------------------------------
# Status bar ⚡ badge test
# ---------------------------------------------------------------------------

def _make_status_tui(pending_count: int) -> MagicMock:
    """Build a minimal tui mock for _compute_status_text calls."""
    from mnemara.tui import MnemaraTUI
    tui = MagicMock()
    tui._queued_input = None
    tui._peer_pending_rows = [{"row_id": i} for i in range(pending_count)]
    tui.store.total_tokens.return_value = (1000, 500)
    tui.store.window.return_value = [{}] * 5
    tui.store.get_eviction_stats.return_value = {"rows_evicted": 0, "blocks_evicted": 0}
    tui.cfg.max_window_tokens = 150_000
    tui.cfg.model = "claude-sonnet-4-6"
    return tui


def test_status_bar_shows_inbox_count() -> None:
    """_compute_status_text includes ⚡ N when _peer_pending_rows is non-empty."""
    from mnemara.tui import MnemaraTUI

    tui = _make_status_tui(2)
    status = MnemaraTUI._compute_status_text(tui)
    assert "⚡ 2" in status


def test_status_bar_no_badge_when_empty() -> None:
    """_compute_status_text shows no ⚡ when inbox is empty."""
    from mnemara.tui import MnemaraTUI

    tui = _make_status_tui(0)
    status = MnemaraTUI._compute_status_text(tui)
    assert "⚡" not in status
