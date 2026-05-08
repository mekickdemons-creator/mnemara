"""v0.11.0 — autonomous peer-poll tests.

Tests for _poll_peer_messages: background timer that reads muninn.db
and injects peer panel messages as agent turns without producer relay.

Uses sqlite3 directly to seed a test muninn.db, then calls the
poll method on a minimal MnemaraTUI instance.  No real Textual
pilot needed — the method is async and directly testable.
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
              interval: int = 90) -> MagicMock:
    """Build a minimal MnemaraTUI-like mock that exercises _poll_peer_messages.

    We instantiate the real method via an unbound call, injecting only the
    attributes the method actually touches.
    """
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
    tui._busy = False
    tui._delivered_peer_row_ids = set()
    tui._chat.return_value = MagicMock()
    tui._handle_user_input = AsyncMock()
    return tui


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_peer_poll_config_defaults() -> None:
    """New config fields load with sensible defaults when absent from dict."""
    from mnemara import config as config_mod
    cfg = config_mod.Config.from_dict({})
    assert cfg.peer_poll_enabled is False
    assert cfg.peer_poll_interval_seconds == 90
    assert cfg.architect_db_path == ""
    assert cfg.peer_poll_roles == "theseus"


def test_peer_poll_config_roundtrip() -> None:
    """Config serialises and deserialises all four new fields cleanly."""
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


@pytest.mark.asyncio
async def test_poll_delivers_pending_row(tmp_path: Path) -> None:
    """A pending row from a watched role is delivered as a user turn."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(db, agent_role="theseus", task_id="ping-42",
                          payload={"type": "directive", "msg": "hello substrate"})

    tui = _make_tui(tmp_path, str(db))
    await MnemaraTUI._poll_peer_messages(tui)

    assert row_id in tui._delivered_peer_row_ids
    tui._handle_user_input.assert_awaited_once()
    injected = tui._handle_user_input.call_args[0][0]
    assert "theseus" in injected
    assert f"row_id={row_id}" in injected
    assert "ping-42" in injected
    assert "ack_return" in injected
    assert "submit_return" in injected


@pytest.mark.asyncio
async def test_poll_skips_non_watched_roles(tmp_path: Path) -> None:
    """Rows from roles not in peer_poll_roles are ignored."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    _seed_return(db, agent_role="designer", task_id="t1",
                 payload={"msg": "irrelevant"})

    tui = _make_tui(tmp_path, str(db), peer_roles="theseus")
    await MnemaraTUI._poll_peer_messages(tui)

    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_skips_non_pending_rows(tmp_path: Path) -> None:
    """Rows with status != 'pending' are not delivered."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    _seed_return(db, agent_role="theseus", task_id="done-row",
                 payload={"msg": "already done"}, status="done")

    tui = _make_tui(tmp_path, str(db))
    await MnemaraTUI._poll_peer_messages(tui)

    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_skips_already_delivered(tmp_path: Path) -> None:
    """A row already in _delivered_peer_row_ids is not re-delivered."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(db, agent_role="theseus", task_id="t1",
                          payload={"msg": "hi"})

    tui = _make_tui(tmp_path, str(db))
    tui._delivered_peer_row_ids.add(row_id)
    await MnemaraTUI._poll_peer_messages(tui)

    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_skips_when_busy(tmp_path: Path) -> None:
    """When _busy is True, no row is delivered (retry next tick)."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(db, agent_role="theseus", task_id="t1",
                          payload={"msg": "queued"})

    tui = _make_tui(tmp_path, str(db))
    tui._busy = True
    await MnemaraTUI._poll_peer_messages(tui)

    # Row must NOT be in delivered set — it should be retried next tick.
    assert row_id not in tui._delivered_peer_row_ids
    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_delivers_one_per_tick(tmp_path: Path) -> None:
    """Multiple pending rows deliver one per tick (break after first delivery)."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id_1 = _seed_return(db, agent_role="theseus", task_id="t1",
                             payload={"n": 1})
    row_id_2 = _seed_return(db, agent_role="theseus", task_id="t2",
                             payload={"n": 2})

    tui = _make_tui(tmp_path, str(db))
    await MnemaraTUI._poll_peer_messages(tui)

    # Only one delivery per tick.
    assert tui._handle_user_input.await_count == 1
    # The first (lowest id) row was delivered.
    assert row_id_1 in tui._delivered_peer_row_ids
    # The second row is pending — will be delivered on the next tick.
    assert row_id_2 not in tui._delivered_peer_row_ids


@pytest.mark.asyncio
async def test_poll_multi_tick_drains_queue(tmp_path: Path) -> None:
    """Running the poll twice drains two pending rows across two ticks."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id_1 = _seed_return(db, agent_role="theseus", task_id="t1",
                             payload={"n": 1})
    row_id_2 = _seed_return(db, agent_role="theseus", task_id="t2",
                             payload={"n": 2})

    tui = _make_tui(tmp_path, str(db))
    await MnemaraTUI._poll_peer_messages(tui)  # tick 1 → delivers row 1
    await MnemaraTUI._poll_peer_messages(tui)  # tick 2 → delivers row 2

    assert tui._handle_user_input.await_count == 2
    assert row_id_1 in tui._delivered_peer_row_ids
    assert row_id_2 in tui._delivered_peer_row_ids


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
    await MnemaraTUI._poll_peer_messages(tui)

    assert row_theseus in tui._delivered_peer_row_ids
    assert tui._handle_user_input.await_count == 1


@pytest.mark.asyncio
async def test_poll_handles_missing_db_gracefully(tmp_path: Path) -> None:
    """Non-existent muninn.db: poll returns silently (no crash)."""
    from mnemara.tui import MnemaraTUI

    tui = _make_tui(tmp_path, str(tmp_path / "nonexistent.db"))
    # Should not raise.
    await MnemaraTUI._poll_peer_messages(tui)
    tui._handle_user_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_message_format_includes_required_instructions(
    tmp_path: Path,
) -> None:
    """Injected message contains ack and submit_return instructions for the agent."""
    from mnemara.tui import MnemaraTUI

    db = _make_muninn_db(tmp_path)
    row_id = _seed_return(
        db,
        agent_role="theseus",
        task_id="research-project-alpha",
        payload={"type": "directive", "content": "start the research project"},
    )

    tui = _make_tui(tmp_path, str(db))
    await MnemaraTUI._poll_peer_messages(tui)

    msg = tui._handle_user_input.call_args[0][0]
    # Must tell the agent its row_id for acking
    assert f"row_id={row_id}" in msg
    # Must contain explicit ack instruction
    assert "mcp__architect__ack_return" in msg
    # Must contain explicit submit_return instruction with substrate identity
    assert 'role="substrate"' in msg
    # Must contain the topic name
    assert "research-project-alpha" in msg
    # Must contain the payload content
    assert "start the research project" in msg
