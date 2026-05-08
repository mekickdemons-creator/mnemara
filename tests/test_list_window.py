"""Tests for list_window store method and MCP tool, plus inspect_context include_rows."""
from __future__ import annotations

# Skip entire module when claude_agent_sdk is not installed (gemma package).
# _make_session() patches agent_mod.query which only exists when the SDK is present.
import pytest
pytest.importorskip("claude_agent_sdk")

import asyncio
import json
from pathlib import Path

import pytest


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Store-level tests — exercise store.list_window() directly
# ---------------------------------------------------------------------------


def test_list_window_returns_row_ids_and_summaries(home):
    """Insert 3 user + 2 assistant turns; list_window returns all 5 sorted most-recent first."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t1")
    s = Store("lw_t1")

    # Insert 3 user turns
    ids = []
    ids.append(s.append_turn("user", [{"type": "text", "text": "user message one"}]))
    ids.append(s.append_turn("user", [{"type": "text", "text": "user message two"}]))
    ids.append(s.append_turn("user", [{"type": "text", "text": "user message three"}]))
    # Insert 2 assistant turns
    ids.append(s.append_turn("assistant", [{"type": "text", "text": "assistant reply one"}]))
    ids.append(s.append_turn("assistant", [{"type": "text", "text": "assistant reply two"}]))

    result = s.list_window(limit=50)

    assert result["ok"] is True
    assert result["total"] == 5
    rows = result["rows"]
    assert len(rows) == 5

    # All row_ids are real ints
    for r in rows:
        assert isinstance(r["row_id"], int)
        assert r["row_id"] > 0

    # All summaries are non-empty strings
    for r in rows:
        assert isinstance(r["summary"], str)
        assert len(r["summary"]) > 0

    # Sorted most-recent first (highest row_id first)
    returned_ids = [r["row_id"] for r in rows]
    assert returned_ids == sorted(returned_ids, reverse=True)

    s.close()


def test_list_window_role_filter(home):
    """Filter to role='user' returns only user rows."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t2")
    s = Store("lw_t2")

    s.append_turn("user", [{"type": "text", "text": "user turn A"}])
    s.append_turn("assistant", [{"type": "text", "text": "assistant turn A"}])
    s.append_turn("user", [{"type": "text", "text": "user turn B"}])
    s.append_turn("assistant", [{"type": "text", "text": "assistant turn B"}])
    s.append_turn("user", [{"type": "text", "text": "user turn C"}])

    result = s.list_window(limit=50, role="user")

    assert result["ok"] is True
    assert result["total"] == 3
    rows = result["rows"]
    assert len(rows) == 3
    for r in rows:
        assert r["role"] == "user"

    result_asst = s.list_window(limit=50, role="assistant")
    assert result_asst["total"] == 2
    for r in result_asst["rows"]:
        assert r["role"] == "assistant"

    s.close()


def test_list_window_offset_pagination(home):
    """Paginate with limit+offset — no overlap between pages."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t3")
    s = Store("lw_t3")

    for i in range(10):
        s.append_turn("user", [{"type": "text", "text": f"turn {i}"}])

    page1 = s.list_window(limit=3, offset=0)
    page2 = s.list_window(limit=3, offset=3)

    assert page1["ok"] is True
    assert page2["ok"] is True
    assert len(page1["rows"]) == 3
    assert len(page2["rows"]) == 3

    ids1 = {r["row_id"] for r in page1["rows"]}
    ids2 = {r["row_id"] for r in page2["rows"]}
    assert ids1.isdisjoint(ids2), "Pages must not overlap"

    # total is the same regardless of pagination
    assert page1["total"] == 10
    assert page2["total"] == 10

    s.close()


def test_list_window_empty_store(home):
    """list_window on an empty store returns ok with empty rows and total=0."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t4")
    s = Store("lw_t4")

    result = s.list_window(limit=50)
    assert result["ok"] is True
    assert result["total"] == 0
    assert result["rows"] == []

    s.close()


def test_list_window_limit_capped_at_200(home):
    """list_window caps limit at 200 even when caller passes a higher value."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t5")
    s = Store("lw_t5")

    for i in range(5):
        s.append_turn("user", [{"type": "text", "text": f"msg {i}"}])

    # Passing limit=9999 should still only return all 5 rows (capped internally)
    result = s.list_window(limit=9999)
    assert result["ok"] is True
    assert len(result["rows"]) == 5

    s.close()


def test_list_window_assistant_tool_use_summary(home):
    """Assistant rows with tool_use first block return '[tool_use <name>]' summary."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t6")
    s = Store("lw_t6")

    # Structured assistant turn starting with tool_use
    s.append_turn("assistant", [
        {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file_path": "/foo.py"}},
        {"type": "text", "text": "Here is the content."},
    ])

    result = s.list_window(limit=50)
    assert result["ok"] is True
    assert len(result["rows"]) == 1
    summary = result["rows"][0]["summary"]
    assert summary == "[tool_use Read]"

    s.close()


def test_list_window_assistant_text_block_summary(home):
    """Assistant rows with text first block return first 80 chars of text."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("lw_t7")
    s = Store("lw_t7")

    long_text = "A" * 200
    s.append_turn("assistant", [{"type": "text", "text": long_text}])

    result = s.list_window(limit=50)
    assert result["ok"] is True
    summary = result["rows"][0]["summary"]
    assert len(summary) <= 80
    assert summary == "A" * 80

    s.close()


# ---------------------------------------------------------------------------
# MCP tool tests — exercise the list_window tool via _registered_tools
# ---------------------------------------------------------------------------


def _make_session(home, instance_name, monkeypatch):
    """Helper: create an AgentSession with a fake query and return (session, store)."""
    from mnemara import agent as agent_mod
    from mnemara import config
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance(instance_name)
    cfg = config.load(instance_name)
    store = Store(instance_name)
    perms = PermissionStore(instance_name)
    runner = ToolRunner(instance_name, cfg, perms, prompt=lambda t, x: "deny")

    async def _fake_query(*, prompt, options, transport=None):
        async for _ in prompt:
            break
        if False:
            yield None
        return

    monkeypatch.setattr(agent_mod, "query", _fake_query)

    session = agent_mod.AgentSession(cfg, store, runner)
    session.turn("init")  # trigger _build_options so _registered_tools is populated
    return session, store


def test_list_window_mcp_tool_returns_rows(home, monkeypatch):
    """list_window MCP tool returns rows with row_id, timestamp, role, summary."""
    from mnemara import agent as agent_mod

    session, store = _make_session(home, "lw_mcp_t1", monkeypatch)

    store.append_turn("user", [{"type": "text", "text": "hello from user"}])
    store.append_turn("assistant", [{"type": "text", "text": "hello from assistant"}])

    handlers = session._registered_tools
    assert "list_window" in handlers, "list_window must be registered"

    fn = handlers["list_window"]
    result = asyncio.run(fn({}))
    text = result["content"][0]["text"]
    data = json.loads(text)

    assert data["ok"] is True
    rows = data["rows"]
    # At least our 2 manually inserted rows are there (init turn also added rows)
    assert len(rows) >= 2
    for r in rows:
        assert "row_id" in r
        assert "timestamp" in r
        assert "role" in r
        assert "summary" in r
        assert isinstance(r["row_id"], int)

    store.close()


def test_list_window_mcp_tool_role_filter(home, monkeypatch):
    """list_window MCP tool role filter works."""
    from mnemara import agent as agent_mod

    session, store = _make_session(home, "lw_mcp_t2", monkeypatch)

    store.append_turn("user", [{"type": "text", "text": "user only msg"}])
    store.append_turn("assistant", [{"type": "text", "text": "asst only msg"}])

    fn = session._registered_tools["list_window"]
    result = asyncio.run(fn({"role": "user"}))
    data = json.loads(result["content"][0]["text"])

    assert data["ok"] is True
    for r in data["rows"]:
        assert r["role"] == "user"

    store.close()


# ---------------------------------------------------------------------------
# inspect_context include_rows tests
# ---------------------------------------------------------------------------


def test_inspect_context_include_rows_returns_row_ids(home, monkeypatch):
    """inspect_context with include_rows=True returns aggregate AND rows list."""
    from mnemara import agent as agent_mod

    session, store = _make_session(home, "ic_ir_t1", monkeypatch)

    store.append_turn("user", [{"type": "text", "text": "msg A"}])
    store.append_turn("assistant", [{"type": "text", "text": "reply A"}])
    store.append_turn("user", [{"type": "text", "text": "msg B"}])

    fn = session._registered_tools["inspect_context"]
    result = asyncio.run(fn({"include_rows": True}))
    text = result["content"][0]["text"]
    info = json.loads(text)

    # Aggregate stats must still be present
    assert "current_turn_count" in info
    assert "total_input_tokens" in info
    assert "total_output_tokens" in info

    # rows key must be present and non-empty
    assert "rows" in info
    rows = info["rows"]
    assert len(rows) > 0

    # Each row must have a non-empty row_id
    for r in rows:
        assert "row_id" in r
        assert isinstance(r["row_id"], int)
        assert r["row_id"] > 0
        assert "summary" in r
        assert "role" in r
        assert "timestamp" in r

    store.close()


def test_inspect_context_without_include_rows_unchanged(home, monkeypatch):
    """inspect_context without include_rows has no 'rows' key (backward compat)."""
    from mnemara import agent as agent_mod

    session, store = _make_session(home, "ic_ir_t2", monkeypatch)
    store.append_turn("user", [{"type": "text", "text": "hi"}])

    fn = session._registered_tools["inspect_context"]
    result = asyncio.run(fn({}))
    info = json.loads(result["content"][0]["text"])

    assert "rows" not in info
    assert "current_turn_count" in info

    store.close()


def test_inspect_context_include_rows_false_no_rows(home, monkeypatch):
    """inspect_context with include_rows=False also omits rows."""
    from mnemara import agent as agent_mod

    session, store = _make_session(home, "ic_ir_t3", monkeypatch)
    store.append_turn("user", [{"type": "text", "text": "ping"}])

    fn = session._registered_tools["inspect_context"]
    result = asyncio.run(fn({"include_rows": False}))
    info = json.loads(result["content"][0]["text"])

    assert "rows" not in info

    store.close()
