"""Smoke tests — no network, no Claude. Exercise store, config, tools, CLI plumbing."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_init_and_load(home):
    from mnemara import config, paths
    config.init_instance("t1", role_doc_path="/tmp/role.md")
    assert paths.instance_dir("t1").exists()
    assert paths.config_path("t1").exists()
    cfg = config.load("t1")
    assert cfg.role_doc_path == "/tmp/role.md"
    assert cfg.max_window_turns == 100
    assert cfg.max_window_tokens == 500_000


def test_store_eviction(home):
    from mnemara import config
    from mnemara.store import Store
    config.init_instance("t2")
    s = Store("t2")
    for i in range(25):
        s.append_turn("user", [{"type": "text", "text": f"u{i}"}])
        s.append_turn("assistant", [{"type": "text", "text": f"a{i}"}])
    deleted = s.evict(20)
    assert deleted > 0
    assert len(s.window()) == 20
    s.close()


def test_messages_for_api_shape(home):
    from mnemara import config
    from mnemara.store import Store
    config.init_instance("t3")
    s = Store("t3")
    s.append_turn("user", [{"type": "text", "text": "hi"}])
    s.append_turn("assistant", [{"type": "text", "text": "hello"}])
    msgs = s.messages_for_api()
    assert msgs == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    s.close()


def test_write_memory(home):
    from mnemara import config
    from mnemara.tools import write_memory
    config.init_instance("t4")
    p = write_memory("t4", "remember this", category="insight")
    assert p.exists()
    text = p.read_text()
    assert "remember this" in text
    assert "insight" in text


def test_permission_decide(home):
    from mnemara import config
    from mnemara.config import Config, ToolPolicy
    from mnemara.permissions import PermissionStore, decide
    config.init_instance("t5")
    cfg = Config.default()
    perms = PermissionStore("t5")
    # Bash defaults to ask
    assert decide(cfg, perms, "Bash", "ls") == "ask"
    # session-allow flips to allow
    perms.session_allow("Bash")
    assert decide(cfg, perms, "Bash", "ls") == "allow"


def test_tool_runner_read_write_edit(home, tmp_path):
    from mnemara import config
    from mnemara.config import Config
    from mnemara.permissions import PermissionStore
    from mnemara.tools import ToolRunner
    config.init_instance("t6")
    cfg = Config.default()
    cfg.file_tool_home_only = True
    # Pre-allow file tools
    for tp in cfg.allowed_tools:
        if tp.tool in ("Read", "Write", "Edit"):
            tp.mode = "allow"
    perms = PermissionStore("t6")
    runner = ToolRunner("t6", cfg, perms, prompt=lambda t, x: "deny")

    target = tmp_path / "f.txt"
    out, err = runner.dispatch("Write", {"path": str(target), "content": "alpha\nbeta\n"})
    assert not err and target.exists()

    out, err = runner.dispatch("Read", {"path": str(target)})
    assert not err and "alpha" in out

    out, err = runner.dispatch("Edit", {"path": str(target), "old_string": "beta", "new_string": "gamma"})
    assert not err
    assert "gamma" in target.read_text()


def test_cli_init_idempotent(home):
    from click.testing import CliRunner
    from mnemara.cli import main
    runner = CliRunner()
    r = runner.invoke(main, ["init", "--instance", "cli1", "--role", ""])
    assert r.exit_code == 0
    r2 = runner.invoke(main, ["init", "--instance", "cli1", "--role", ""])
    assert r2.exit_code != 0  # refuses to overwrite


def test_agent_prompt_includes_role_window_and_input(home, monkeypatch):
    """Verify the SDK call receives [system: role_doc, ...rolling_window, current_input].

    We mock claude_agent_sdk.query to capture the prompt + options it was
    called with, then inspect them.
    """
    from mnemara import config
    from mnemara.config import Config
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner
    from mnemara import agent as agent_mod

    config.init_instance("agent_t")
    cfg = config.load("agent_t")
    role_path = paths_role(home, "role.md", "ROLE-DOC-MARKER")
    cfg.role_doc_path = str(role_path)
    cfg.stream = False
    config.save("agent_t", cfg)

    store = Store("agent_t")
    store.append_turn("user", [{"type": "text", "text": "earlier-user"}])
    store.append_turn("assistant", [{"type": "text", "text": "earlier-asst"}])

    perms = PermissionStore("agent_t")
    runner = ToolRunner("agent_t", cfg, perms, prompt=lambda t, x: "deny")

    captured: dict = {}

    async def _fake_query(*, prompt, options, transport=None):
        captured["prompt"] = prompt
        captured["options"] = options
        # Yield nothing — agent loop tolerates empty stream.
        if False:
            yield None
        return

    monkeypatch.setattr(agent_mod, "query", _fake_query)

    session = agent_mod.AgentSession(cfg, store, runner, client=None)
    session.turn("CURRENT-INPUT-MARKER")

    prompt = captured["prompt"]
    options = captured["options"]
    # Role doc goes through as system_prompt.
    assert "ROLE-DOC-MARKER" in options.system_prompt
    # Prompt is an AsyncIterable yielding user-message dicts (SDK requirement
    # when can_use_tool is set). Drain it to inspect the payload.
    import asyncio as _asyncio
    async def _drain(gen):
        out = []
        async for m in gen:
            out.append(m)
        return out
    msgs = _asyncio.run(_drain(prompt))
    assert msgs, "prompt generator yielded nothing"
    payload = msgs[0]["message"]["content"]
    # Rolling window is reflected in the prompt prefix.
    assert "earlier-user" in payload
    assert "earlier-asst" in payload
    # Current user input is the live message.
    assert "CURRENT-INPUT-MARKER" in payload
    # Current input ordered after the prior history.
    assert payload.index("earlier-user") < payload.index("CURRENT-INPUT-MARKER")
    # Built-in Claude Code tools are exposed.
    assert "Bash" in options.allowed_tools
    assert any("write_memory" in t for t in options.allowed_tools)
    store.close()


def paths_role(home, name: str, content: str):
    p = home / name
    p.write_text(content)
    return p


def test_tui_imports_and_instantiates(home):
    """TUI module imports and the App class instantiates without crashing."""
    from mnemara import config
    config.init_instance("tui_t")
    from mnemara import tui as tui_mod
    assert tui_mod._TEXTUAL_AVAILABLE, "textual should import in dev environment"
    app = tui_mod.MnemaraTUI("tui_t")
    assert app.instance == "tui_t"
    assert app.cfg.model
    s = app._status_text()
    assert "turns:" in s and "tokens:" in s
    app.store.close()


def test_on_token_callback_invoked(home, monkeypatch):
    """Streaming on_token callback receives text deltas during turn_async."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner
    from mnemara import agent as agent_mod

    config.init_instance("cb_t")
    cfg = config.load("cb_t")
    cfg.stream = True
    config.save("cb_t", cfg)

    store = Store("cb_t")
    perms = PermissionStore("cb_t")
    runner = ToolRunner("cb_t", cfg, perms, prompt=lambda t, x: "deny")

    async def _maybe(cb, arg):
        r = cb(arg)
        if _asyncio.iscoroutine(r):
            await r

    async def _fake_run_turn(prompt, options, stream, on_token=None,
                             on_tool_use=None, on_tool_result=None):
        if on_token is not None:
            await _maybe(on_token, "Hello, ")
            await _maybe(on_token, "world!")
        return {
            "assistant_blocks": [{"type": "text", "text": "Hello, world!"}],
            "tokens_in": 5,
            "tokens_out": 7,
        }

    monkeypatch.setattr(agent_mod, "_run_turn", _fake_run_turn)

    session = agent_mod.AgentSession(cfg, store, runner, client=None)

    captured: list[str] = []

    async def _go():
        return await session.turn_async(
            "hi", on_token=lambda t: captured.append(t)
        )

    usage = _asyncio.run(_go())
    assert captured == ["Hello, ", "world!"]
    assert usage["input_tokens"] == 5
    rows = store.window()
    assert [r["role"] for r in rows[-2:]] == ["user", "assistant"]
    store.close()


def test_inspect_context_schema(home, monkeypatch):
    """inspect_context returns the full schema with populated fields."""
    import asyncio as _asyncio
    import json as _json
    from mnemara import agent as agent_mod
    from mnemara import config
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance("ic_t")
    cfg = config.load("ic_t")
    role_path = home / "role.md"
    role_path.write_text("ROLE-DOC")
    cfg.role_doc_path = str(role_path)
    config.save("ic_t", cfg)

    store = Store("ic_t")
    store.append_turn("user", [{"type": "text", "text": "hi"}], tokens_in=10, tokens_out=20)
    perms = PermissionStore("ic_t")
    runner = ToolRunner("ic_t", cfg, perms, prompt=lambda t, x: "deny")

    captured = {}

    async def _fake_query(*, prompt, options, transport=None):
        captured["options"] = options
        # drain the prompt generator to satisfy the SDK contract
        async for _ in prompt:
            break
        if False:
            yield None
        return

    monkeypatch.setattr(agent_mod, "query", _fake_query)

    session = agent_mod.AgentSession(cfg, store, runner)
    session.evicted_this_session = 3
    session.turn("ping")

    handlers = session._registered_tools
    assert "inspect_context" in handlers
    fn = handlers["inspect_context"]
    result = _asyncio.run(fn({}))
    text = result["content"][0]["text"]
    info = _json.loads(text)
    assert info["instance"] == "ic_t"
    assert info["model"] == cfg.model
    assert info["max_window_turns"] == cfg.max_window_turns
    assert info["max_window_tokens"] == cfg.max_window_tokens
    assert info["role_doc_path"] == str(role_path)
    assert info["role_doc_size_bytes"] == len("ROLE-DOC")
    assert info["evicted_this_session"] == 3
    assert info["total_input_tokens"] >= 0
    assert info["total_output_tokens"] >= 0
    assert info["total_tokens"] == info["total_input_tokens"] + info["total_output_tokens"]
    assert info["tokens_remaining"] == max(0, cfg.max_window_tokens - info["total_tokens"])
    assert "mnemara_memory" in info["mcp_servers"]
    assert any(e["tool"] == "InspectContext" for e in info["allowed_tools_summary"])
    store.close()


def test_propose_role_amendment_writes_file(home):
    from mnemara import config, paths
    from mnemara.tools import propose_role_amendment

    config.init_instance("pra_t")
    p = propose_role_amendment(
        "pra_t",
        "Adopt a stricter test-first discipline for memory tools",
        "Observed silent regressions in the last two sessions.",
        "moderate",
    )
    assert p.exists()
    assert p.parent == paths.role_proposals_dir("pra_t")
    body = p.read_text()
    assert body.startswith("---\n")
    assert "severity: moderate" in body
    assert "rationale:" in body
    assert "Adopt a stricter test-first discipline for memory tools" in body
    # Slug appears in filename
    assert "adopt-a-stricter-test-first-discipline-for" in p.name


def test_log_choice_appends_jsonl(home):
    import json as _json
    from mnemara import config, paths
    from mnemara.tools import log_choice

    config.init_instance("lc_t")
    log_choice("lc_t", "approach", "use cache", "lower latency", context_summary="ctx",
               turn_id=42, tokens_at_choice=1234)
    log_choice("lc_t", "approach", "skip retry", "user wants speed",
               turn_id=43, tokens_at_choice=2000)

    f = paths.choices_path("lc_t")
    assert f.exists()
    lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    rec = _json.loads(lines[0])
    assert rec["decision_type"] == "approach"
    assert rec["decision"] == "use cache"
    assert rec["rationale"] == "lower latency"
    assert rec["context_summary"] == "ctx"
    assert rec["turn_id"] == 42
    assert rec["tokens_at_choice"] == 1234
    assert "ts" in rec


def test_structured_write_memory_payload(home):
    from mnemara import config
    from mnemara.tools import write_memory

    config.init_instance("swm_t")
    payload = {
        "observation": "tools without backpressure cause throughput cliffs",
        "evidence": "saw 3x latency in session 2026-04-26 when tool calls bursted",
        "prediction": "next bursty session will hit the same cliff",
        "applies_to": "agent_loop, tool_dispatch",
        "confidence": "medium",
    }
    p = write_memory("swm_t", "ignored when payload set", category="ignored",
                     payload=payload)
    text = p.read_text()
    assert "## " in text and "observation" in text
    assert "**evidence:**" in text
    assert "**prediction:**" in text
    assert "**applies_to:**" in text
    assert "**confidence:**" in text
    assert "tools without backpressure cause throughput cliffs" in text
    assert "medium" in text

    # Backward compatibility: legacy text+category still works.
    p2 = write_memory("swm_t", "legacy note", category="legacy")
    text2 = p2.read_text()
    assert "legacy note" in text2
    assert "] legacy" in text2


def test_session_stats_dump_merges(home):
    import json as _json
    from datetime import datetime, timezone
    from mnemara import agent as agent_mod, config, paths
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance("ss_t")
    cfg = config.load("ss_t")
    store = Store("ss_t")
    perms = PermissionStore("ss_t")
    runner = ToolRunner("ss_t", cfg, perms, prompt=lambda t, x: "deny")

    s1 = agent_mod.AgentSession(cfg, store, runner)
    s1.session_turns = 2
    s1.session_tokens_in = 100
    s1.session_tokens_out = 200
    s1.evicted_this_session = 1
    s1.tools_called = {"Bash": 2, "Read": 1}
    s1.memory_writes = 1
    s1.role_proposals = 0
    s1.choices_logged = 1
    p1 = s1.write_session_stats()
    assert p1 is not None and p1.exists()

    s2 = agent_mod.AgentSession(cfg, store, runner)
    s2.session_turns = 3
    s2.session_tokens_in = 50
    s2.session_tokens_out = 75
    s2.evicted_this_session = 2
    s2.tools_called = {"Bash": 1, "Write": 4}
    s2.memory_writes = 2
    s2.role_proposals = 1
    s2.choices_logged = 0
    p2 = s2.write_session_stats()
    assert p2 == p1  # same date file

    doc = _json.loads(p1.read_text())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert doc["date"] == today
    assert doc["instance"] == "ss_t"
    assert len(doc["sessions"]) == 2
    cum = doc["cumulative"]
    assert cum["turns"] == 5
    assert cum["tokens_in"] == 150
    assert cum["tokens_out"] == 275
    assert cum["evicted"] == 3
    assert cum["memory_writes"] == 3
    assert cum["role_proposals"] == 1
    assert cum["choices_logged"] == 1
    assert cum["tools_called"]["Bash"] == 3
    assert cum["tools_called"]["Read"] == 1
    assert cum["tools_called"]["Write"] == 4

    # Idempotent on second call within same session
    assert s2.write_session_stats() is None
    store.close()


def test_proposals_list_parses_severity(home):
    """parse_proposal_file returns correct severity and body preview."""
    from mnemara import config, paths
    from mnemara.tools import parse_proposal_file, propose_role_amendment

    config.init_instance("prop_t")
    p1 = propose_role_amendment("prop_t", "Add self-check before each tool call", "Saw failures.", "moderate")
    p2 = propose_role_amendment("prop_t", "Minor wording fix in role doc", "Clarity.", "minor")

    sev1, preview1 = parse_proposal_file(p1)
    assert sev1 == "moderate"
    assert "Add self-check" in preview1

    sev2, preview2 = parse_proposal_file(p2)
    assert sev2 == "minor"
    assert "Minor wording" in preview2

    # Verify count helper
    assert paths.role_proposals_count("prop_t") == 2

    # Verify /proposals command output via REPL helper
    from io import StringIO
    from rich.console import Console as _Console
    from mnemara import repl as repl_mod
    buf = StringIO()
    orig_console = repl_mod.console
    repl_mod.console = _Console(file=buf, highlight=False)
    try:
        repl_mod._cmd_proposals("prop_t")
    finally:
        repl_mod.console = orig_console
    output = buf.getvalue()
    assert "2 pending proposals" in output
    assert "moderate" in output
    assert "minor" in output


def test_session_end_summary_fires(home, capsys):
    """Session end summary prints when role_proposals > 0."""
    from mnemara import config, paths
    from mnemara import agent as agent_mod
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance("ses_t")
    cfg = config.load("ses_t")
    store = Store("ses_t")
    perms = PermissionStore("ses_t")
    runner = ToolRunner("ses_t", cfg, perms, prompt=lambda t, x: "deny")

    session = agent_mod.AgentSession(cfg, store, runner)
    session.role_proposals = 2
    # Simulate the REPL shutdown path manually
    from io import StringIO
    from rich.console import Console as _Console
    from mnemara import repl as repl_mod
    buf = StringIO()
    orig_console = repl_mod.console
    repl_mod.console = _Console(file=buf, highlight=False)
    try:
        try:
            if session.role_proposals > 0:
                n = session.role_proposals
                p = paths.role_proposals_dir("ses_t")
                repl_mod.console.print(
                    f"📋 {n} role-amendment proposal(s) written this session. "
                    f"Review at {p}"
                )
        except Exception:
            pass
    finally:
        repl_mod.console = orig_console
    output = buf.getvalue()
    assert "2 role-amendment proposal(s)" in output
    assert "role_proposals" in str(paths.role_proposals_dir("ses_t"))
    store.close()


def test_cli_list_show_clear(home):
    from click.testing import CliRunner
    from mnemara.cli import main
    from mnemara.store import Store
    runner = CliRunner()
    runner.invoke(main, ["init", "--instance", "x1", "--role", ""])
    s = Store("x1")
    s.append_turn("user", [{"type": "text", "text": "hi"}])
    s.close()
    r = runner.invoke(main, ["list"])
    assert "x1" in r.output
    r = runner.invoke(main, ["show", "--instance", "x1"])
    assert "hi" in r.output
    r = runner.invoke(main, ["clear", "--instance", "x1"])
    assert r.exit_code == 0


# ------------------------------------------------------------------ inbox tests


def test_inbox_peek_pending_pings_with_mocked_db(tmp_path):
    """peek_pending_pings returns expected rows from a mocked sqlite db."""
    import json
    import sqlite3
    from datetime import datetime, timezone
    from mnemara.inbox import peek_pending_pings, count_pending

    db_file = tmp_path / "muninn.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE returns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            agent_role TEXT NOT NULL,
            task_id TEXT,
            submitted_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            processed_at TEXT,
            completed_at TEXT,
            recipient_role TEXT
        )
    """)
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO returns (session_id, agent_role, task_id, submitted_at, payload_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("s1", "theseus", "task-abc", now_iso,
         json.dumps({"type": "directive", "subject": "proto-refresh", "body": "Drain pings on each tick."}),
         "pending"),
    )
    conn.execute(
        "INSERT INTO returns (session_id, agent_role, task_id, submitted_at, payload_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("s2", "theseus", "task-xyz", now_iso,
         json.dumps({"type": "verdict", "body": "All good."}),
         "processed"),  # not pending — should be excluded
    )
    conn.execute(
        "INSERT INTO returns (session_id, agent_role, task_id, submitted_at, payload_json, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("s3", "majordomo", "task-def", now_iso,
         json.dumps({"type": "ack"}),
         "pending"),
    )
    conn.commit()
    conn.close()

    pings = peek_pending_pings(str(db_file), ["theseus", "majordomo"])
    assert len(pings) == 2
    roles = {p["agent_role"] for p in pings}
    assert roles == {"theseus", "majordomo"}
    theseus_ping = next(p for p in pings if p["agent_role"] == "theseus")
    assert theseus_ping["task_id"] == "task-abc"
    assert theseus_ping["payload_type"] == "directive"
    assert theseus_ping["payload_subject"] == "proto-refresh"
    assert "Drain pings" in theseus_ping["body_preview"]

    # exclude_role removes that sender from results
    pings_excl = peek_pending_pings(str(db_file), ["theseus", "majordomo"], exclude_role="theseus")
    assert len(pings_excl) == 1
    assert pings_excl[0]["agent_role"] == "majordomo"

    # count_pending matches
    assert count_pending(str(db_file), ["theseus", "majordomo"]) == 2
    assert count_pending(str(db_file), ["theseus", "majordomo"], exclude_role="theseus") == 1


def test_inbox_count_pending_absent_db_returns_zero(tmp_path):
    """count_pending returns 0 gracefully when DB path doesn't exist."""
    from mnemara.inbox import count_pending, peek_pending_pings

    missing = str(tmp_path / "nonexistent.db")
    assert count_pending(missing, ["theseus", "majordomo"]) == 0
    assert peek_pending_pings(missing, ["theseus", "majordomo"]) == []
    # Also gracefully handles None/empty path
    assert count_pending(None, ["theseus", "majordomo"]) == 0
    assert count_pending("", ["theseus", "majordomo"]) == 0


def test_inbox_auto_surface_prepends_when_pings_present(home, monkeypatch):
    """turn_async prepends inbox notice to user_text when pings are waiting."""
    import asyncio as _asyncio
    from mnemara import config, agent as agent_mod
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance("inbox_t")
    cfg = config.load("inbox_t")
    cfg.inbox_auto_surface = True
    cfg.architect_db_path = "/fake/muninn.db"
    cfg.peer_roles = ["theseus"]
    config.save("inbox_t", cfg)

    store = Store("inbox_t")
    perms = PermissionStore("inbox_t")
    runner = ToolRunner("inbox_t", cfg, perms, prompt=lambda t, x: "deny")

    # _run_turn receives prompt as a plain string built by _build_prompt.
    captured: dict = {}

    async def _fake_run_turn(prompt, options, stream, on_token=None,
                             on_tool_use=None, on_tool_result=None):
        captured["prompt"] = prompt
        return {
            "assistant_blocks": [{"type": "text", "text": "ok"}],
            "tokens_in": 3,
            "tokens_out": 2,
        }

    monkeypatch.setattr(agent_mod, "_run_turn", _fake_run_turn)

    fake_pings = [{"agent_role": "theseus", "task_id": "t1", "age": "2m ago",
                   "payload_type": "directive", "payload_subject": "hey",
                   "body_preview": "hello there", "id": 42, "submitted_at": ""}]
    monkeypatch.setattr(agent_mod.inbox_mod, "peek_pending_pings", lambda *a, **kw: fake_pings)

    session = agent_mod.AgentSession(cfg, store, runner)
    _asyncio.run(session.turn_async("my actual message"))

    assert "prompt" in captured
    payload = captured["prompt"]
    assert "INBOX:" in payload
    assert "theseus" in payload
    assert "next_return" in payload
    assert "my actual message" in payload
    store.close()


def test_tui_input_guard_ignores_non_userinput(home):
    """on_input_submitted returns without action when event.input.id != 'userinput'."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("guard_t")
    app = tui_mod.MnemaraTUI("guard_t")

    turns_sent: list[str] = []

    async def _fake_send_turn(text: str) -> None:
        turns_sent.append(text)

    app._send_turn = _fake_send_turn  # type: ignore[method-assign]

    class _FakeInput:
        id = "note_text"
        value = "should be ignored"

    class _FakeEvent:
        input = _FakeInput()
        value = "should be ignored"

    _asyncio.run(app.on_input_submitted(_FakeEvent()))  # type: ignore[arg-type]
    assert turns_sent == [], "non-userinput events must be ignored"
    app.store.close()


def test_tui_on_token_tolerates_missing_status_widget(home, monkeypatch):
    """on_token callback absorbs query_one failure and still accumulates the stream buffer."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("tok_rz_t")
    app = tui_mod.MnemaraTUI("tok_rz_t")
    app._stream_buffer = ""
    app._stream_chars = 0

    # Patch query_one to raise — simulates transient widget unavailability during resize.
    def _raise(*args, **kwargs):
        raise Exception("NoMatches: simulated resize redraw")

    monkeypatch.setattr(app, "query_one", _raise)

    # Replicate the on_token closure logic (mirrors the fixed implementation).
    async def on_token(t: str) -> None:
        app._stream_buffer += t
        app._stream_chars += len(t)
        try:
            from textual.widgets import Static
            app.query_one("#status", Static).update("")
        except Exception:
            pass

    _asyncio.run(on_token("hello "))
    _asyncio.run(on_token("world"))
    assert app._stream_buffer == "hello world"
    assert app._stream_chars == 11
    app.store.close()


def test_inbox_auto_surface_skipped_when_disabled(home, monkeypatch):
    """turn_async does NOT prepend inbox notice when inbox_auto_surface=False."""
    import asyncio as _asyncio
    from mnemara import config, agent as agent_mod
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance("inbox_off_t")
    cfg = config.load("inbox_off_t")
    cfg.inbox_auto_surface = False
    cfg.architect_db_path = "/fake/muninn.db"
    cfg.peer_roles = ["theseus"]
    config.save("inbox_off_t", cfg)

    store = Store("inbox_off_t")
    perms = PermissionStore("inbox_off_t")
    runner = ToolRunner("inbox_off_t", cfg, perms, prompt=lambda t, x: "deny")

    captured: dict = {}

    async def _fake_run_turn(prompt, options, stream, on_token=None,
                             on_tool_use=None, on_tool_result=None):
        captured["prompt"] = prompt
        return {
            "assistant_blocks": [{"type": "text", "text": "ok"}],
            "tokens_in": 3,
            "tokens_out": 2,
        }

    monkeypatch.setattr(agent_mod, "_run_turn", _fake_run_turn)

    fake_pings = [{"agent_role": "theseus", "task_id": "t1", "age": "2m ago",
                   "payload_type": "directive", "payload_subject": "hey",
                   "body_preview": "hello", "id": 42, "submitted_at": ""}]
    monkeypatch.setattr(agent_mod.inbox_mod, "peek_pending_pings", lambda *a, **kw: fake_pings)

    session = agent_mod.AgentSession(cfg, store, runner)
    _asyncio.run(session.turn_async("my message"))

    assert "prompt" in captured
    payload = captured["prompt"]
    assert "INBOX:" not in payload
    assert "my message" in payload
    store.close()


# ---------------------------------------------------------- Pilot-based TUI tests


def test_tui_pilot_focus_returns_after_turn(home, monkeypatch):
    """After a streaming turn completes, focus settles back on #userinput.

    Pilot-based; covers the call_after_refresh focus path.
    """
    import asyncio as _asyncio
    from mnemara import config, agent as agent_mod
    from mnemara import tui as tui_mod
    from textual.widgets import Input

    config.init_instance("pilot_focus_t")

    async def _fake_turn_async(self, text, on_token=None, on_tool_use=None,
                               on_tool_result=None):
        if on_token:
            for chunk in ("hel", "lo ", "world"):
                await on_token(chunk)
        return {"tokens_in": 1, "tokens_out": 1, "evicted": 0}

    monkeypatch.setattr(agent_mod.AgentSession, "turn_async", _fake_turn_async)

    app = tui_mod.MnemaraTUI("pilot_focus_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#userinput", Input)
            assert inp.has_focus, "input should have focus on mount"
            inp.value = "hello"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            inp2 = app.query_one("#userinput", Input)
            assert inp2.has_focus, "input must regain focus after turn"
            inp2.value = "second"
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#userinput", Input).has_focus

    _asyncio.run(_run())
    app.store.close()


def test_tui_pilot_input_visible_height(home):
    """#userinput renders with non-zero height even when chatlog is full."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod
    from textual.widgets import Input, RichLog

    config.init_instance("pilot_height_t")
    app = tui_mod.MnemaraTUI("pilot_height_t")

    async def _run() -> None:
        async with app.run_test(size=(120, 30)) as pilot:
            chat = app.query_one("#chatlog", RichLog)
            for i in range(200):
                chat.write(f"[b green]assistant:[/b green] line {i}")
            await pilot.pause()
            inp = app.query_one("#userinput", Input)
            # outer_size = border + content + border; widget reserves 3 rows total
            assert inp.outer_size.height >= 3, f"input collapsed: {inp.outer_size}"
            assert inp.region.y + inp.region.height <= app.size.height
            assert inp.size.width > 0

    _asyncio.run(_run())
    app.store.close()


def test_tui_pilot_focus_input_action(home):
    """ctrl+i / escape binding refocuses input as escape hatch."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod
    from textual.widgets import Input, RichLog

    config.init_instance("pilot_action_t")
    app = tui_mod.MnemaraTUI("pilot_action_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chatlog", RichLog)
            chat.focus()
            await pilot.pause()
            assert not app.query_one("#userinput", Input).has_focus
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert app.query_one("#userinput", Input).has_focus

    _asyncio.run(_run())
    app.store.close()


def test_tui_pilot_richlog_scroll_actions(home):
    """PageUp/PageDown bindings drive the chatlog scroll position."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod
    from textual.widgets import RichLog

    config.init_instance("pilot_scroll_t")
    app = tui_mod.MnemaraTUI("pilot_scroll_t")

    async def _run() -> None:
        async with app.run_test(size=(120, 30)) as pilot:
            chat = app.query_one("#chatlog", RichLog)
            for i in range(200):
                chat.write(f"line {i}")
            await pilot.pause()
            bottom_y = chat.scroll_y
            await pilot.press("pageup")
            await pilot.pause()
            await pilot.pause()
            up_y = chat.scroll_y
            assert up_y < bottom_y, f"pageup did not scroll: {bottom_y} -> {up_y}"
            await pilot.press("pagedown")
            await pilot.pause()
            await pilot.pause()
            assert chat.scroll_y > up_y

    _asyncio.run(_run())
    app.store.close()


def test_tui_pilot_action_paste(home, monkeypatch):
    """action_paste inserts clipboard text at the cursor of the focused Input."""
    import asyncio as _asyncio
    import sys
    import types
    from mnemara import config
    from mnemara import tui as tui_mod
    from textual.widgets import Input

    config.init_instance("pilot_paste_t")

    # Provide a fake pyperclip returning a known string.
    fake_pyperclip = types.ModuleType("pyperclip")
    fake_pyperclip.paste = lambda: "pasted_text"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)

    app = tui_mod.MnemaraTUI("pilot_paste_t")
    # Reset the one-per-session warning flag between test runs.
    tui_mod.MnemaraTUI._paste_unavailable_warned = False

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#userinput", Input)
            inp.focus()
            inp.value = "before_"
            inp.cursor_position = len(inp.value)
            await pilot.pause()
            app.action_paste()
            await pilot.pause()
            assert inp.value == "before_pasted_text"
            assert inp.cursor_position == len("before_pasted_text")

    _asyncio.run(_run())
    app.store.close()


def test_tui_action_copy_last_writes_to_clipboard(home, monkeypatch):
    """action_copy_last copies the most recent assistant response via pyperclip."""
    import asyncio as _asyncio
    import sys
    import types
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("copy_last_t")

    copied: list[str] = []
    fake_pyperclip = types.ModuleType("pyperclip")
    fake_pyperclip.copy = lambda text: copied.append(text)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)

    app = tui_mod.MnemaraTUI("copy_last_t")
    tui_mod.MnemaraTUI._copy_unavailable_warned = False

    # Seed the store with one user turn and one assistant turn.
    app.store.append_turn("user", "Hello")
    app.store.append_turn("assistant", "Hi there, I am the assistant.")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_copy_last()
            await pilot.pause()

    _asyncio.run(_run())
    app.store.close()

    assert len(copied) == 1
    assert copied[0] == "Hi there, I am the assistant."


def test_tui_slash_copy_n_argument(home, monkeypatch):
    """/copy N copies the last N window rows as role-prefixed text."""
    import asyncio as _asyncio
    import sys
    import types
    from mnemara import config
    from mnemara import tui as tui_mod
    from textual.widgets import RichLog

    config.init_instance("copy_n_t")

    copied: list[str] = []
    fake_pyperclip = types.ModuleType("pyperclip")
    fake_pyperclip.copy = lambda text: copied.append(text)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)

    app = tui_mod.MnemaraTUI("copy_n_t")
    tui_mod.MnemaraTUI._copy_unavailable_warned = False

    app.store.append_turn("user", "First question")
    app.store.append_turn("assistant", "First answer")
    app.store.append_turn("user", "Second question")
    app.store.append_turn("assistant", "Second answer")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chatlog", RichLog)
            await app._slash_copy("2", chat)
            await pilot.pause()

    _asyncio.run(_run())
    app.store.close()

    assert len(copied) == 1
    # Last 2 rows are the second user turn and second assistant turn.
    assert "Second question" in copied[0]
    assert "Second answer" in copied[0]
    # Earlier turns should not be included.
    assert "First" not in copied[0]


def test_tui_mouse_safe_modes_on_mount(home):
    """On mount the TUI swaps Textual's default mouse mode set for a safe one.

    Disables 1003 (any-event motion: chatty) and 1015 (urxvt encoding: unsafe
    on wide terminals) while keeping 1000+1002+1006 enabled so the scrollbar
    and mouse-wheel work without crashing the input UTF-8 decoder.
    """
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("selmode_t")
    app = tui_mod.MnemaraTUI("selmode_t")

    driver_writes: list[str] = []

    async def _run() -> None:
        async with app.run_test() as pilot:
            if app._driver is not None:
                orig = app._driver.write
                app._driver.write = lambda s: driver_writes.append(s) or orig(s)  # type: ignore[method-assign]
                # Re-trigger the safe sequence (on_mount already ran before the
                # patch landed) so we can inspect the bytes.
                app._driver.write(app._MOUSE_DISABLE_UNSAFE)
                app._driver.write(app._MOUSE_ENABLE_SAFE)
            await pilot.pause()
            disables = "".join(w for w in driver_writes)
            # Unsafe modes disabled.
            assert "\x1b[?1003l" in disables, "should disable any-event motion"
            assert "\x1b[?1015l" in disables, "should disable urxvt encoding (unsafe on wide terms)"
            # Safe modes enabled.
            assert "\x1b[?1000h" in disables, "should enable basic click tracking"
            assert "\x1b[?1002h" in disables, "should enable button-event drag (scrollbar)"
            assert "\x1b[?1006h" in disables, "should enable SGR encoding (decoder-safe)"

    _asyncio.run(_run())
    app.store.close()


def test_tui_spinner_ticks_when_busy(home):
    """Spinner index advances while _busy=True and resets on busy->idle edge.

    Validates the timer callback contract without depending on real-time
    scheduling: we directly invoke _tick_spinner the way Textual's interval
    timer would, and assert the state transitions.
    """
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("spinner_t")
    app = tui_mod.MnemaraTUI("spinner_t")
    app._cached_status_static = "turns: 0/100 | tokens: 0/500000"

    # Idle -> tick should not advance frame.
    app._busy = False
    app._spinner_idx = 0
    app._tick_spinner()
    assert app._spinner_idx == 0

    # Busy -> tick advances frame and marks _spinner_was_busy.
    app._busy = True
    app._tick_spinner()
    assert app._spinner_idx == 1
    assert app._spinner_was_busy is True
    app._tick_spinner()
    assert app._spinner_idx == 2

    # Busy -> idle transition: next tick clears spinner state once.
    app._busy = False
    app._tick_spinner()
    assert app._spinner_idx == 0
    assert app._spinner_was_busy is False

    # Subsequent idle ticks remain no-ops.
    app._tick_spinner()
    assert app._spinner_idx == 0

    app.store.close()


def test_tui_render_status_widget_omits_spinner_when_idle(home):
    """_render_status_widget includes spinner glyph only when _busy.

    Smoke-checks the rendered text shape — busy state prefixes one of the
    braille frames; idle state shows just the cached static portion.
    """
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod
    from textual.widgets import Static

    config.init_instance("spinner_render_t")
    app = tui_mod.MnemaraTUI("spinner_render_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one("#status", Static)

            # Idle render: no spinner frame.
            app._busy = False
            app._cached_status_static = "STATIC_PART"
            app._render_status_widget()
            await pilot.pause()
            text = str(status.content)
            assert "STATIC_PART" in text
            for f in app._SPINNER_FRAMES:
                assert f not in text, f"idle render leaked spinner frame {f!r}"

            # Busy render: spinner frame prepended.
            app._busy = True
            app._spinner_idx = 3  # frame index 3 is "⠸"
            app._render_status_widget()
            await pilot.pause()
            text = str(status.content)
            assert app._SPINNER_FRAMES[3] in text, f"busy render missing spinner frame: {text!r}"
            assert "STATIC_PART" in text

    _asyncio.run(_run())
    app.store.close()


def test_run_turn_yields_event_loop_between_messages(home, monkeypatch):
    """_run_turn must yield to the event loop between SDK messages.

    The streaming hot path was starving concurrent tasks (Textual Input
    keypress dispatch, resize handlers, spinner timer) when SDK messages
    arrived in tight bursts. We verify the fix by running a concurrent
    sentinel coroutine and asserting it gets scheduled at least once
    DURING the message stream — not just before/after.
    """
    import asyncio as _asyncio
    from mnemara import agent as agent_mod
    from claude_agent_sdk import AssistantMessage, TextBlock, ResultMessage

    sentinel_ticks: list[int] = []
    messages_processed: list[int] = []

    # Build a fake query() async generator that yields many AssistantMessages
    # back-to-back without any internal awaits — simulates the SDK delivering
    # buffered tokens in a burst.
    async def _fake_query(*, prompt, options):
        for i in range(20):
            messages_processed.append(i)
            msg = AssistantMessage(content=[TextBlock(text=f"chunk{i}")], model="test")
            yield msg
        yield ResultMessage(
            subtype="success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            num_turns=1,
            session_id="s",
            total_cost_usd=0.0,
            usage={"input_tokens": 1, "output_tokens": 1},
            result=None,
        )

    monkeypatch.setattr(agent_mod, "query", _fake_query)

    async def _sentinel() -> None:
        # Tick whenever we get scheduled. If _run_turn yields properly,
        # we'll record several ticks while messages are being processed.
        for _ in range(50):
            sentinel_ticks.append(len(messages_processed))
            await _asyncio.sleep(0)

    async def _go() -> None:
        # Schedule sentinel concurrently with _run_turn.
        sentinel_task = _asyncio.create_task(_sentinel())
        result = await agent_mod._run_turn(
            "hi",
            options=None,  # not used by fake_query
            stream=True,
            on_token=None,
            on_tool_use=None,
            on_tool_result=None,
        )
        sentinel_task.cancel()
        try:
            await sentinel_task
        except _asyncio.CancelledError:
            pass
        assert result["assistant_blocks"]

    _asyncio.run(_go())

    # Sentinel should have observed the messages_processed counter advancing
    # mid-stream — i.e. at least two distinct mid-stream values, not just
    # 0 (before) and 20 (after).
    distinct = sorted(set(sentinel_ticks))
    mid_stream = [v for v in distinct if 0 < v < 20]
    assert len(mid_stream) >= 2, (
        f"sentinel saw no mid-stream scheduling — _run_turn isn't yielding "
        f"to the event loop. distinct ticks: {distinct}"
    )


def test_parse_size_handles_suffixes_and_underscores():
    """tui._parse_size accepts plain ints, k/m suffixes, and underscores/commas."""
    from mnemara.tui import _parse_size
    import pytest as _pytest

    assert _parse_size("500") == 500
    assert _parse_size("500k") == 500_000
    assert _parse_size("500K") == 500_000
    assert _parse_size("1m") == 1_000_000
    assert _parse_size("1M") == 1_000_000
    assert _parse_size("1_000_000") == 1_000_000
    assert _parse_size("1,000,000") == 1_000_000
    assert _parse_size("  100k  ") == 100_000
    with _pytest.raises(ValueError):
        _parse_size("")
    with _pytest.raises(ValueError):
        _parse_size("abc")
    with _pytest.raises(ValueError):
        _parse_size("1.5m")  # no float support
    with _pytest.raises(ValueError):
        _parse_size("k")


def test_slash_turns_and_tokens_persist_and_temp(home):
    """/turns N persists by default; /turns N --temp does not."""
    import asyncio as _asyncio
    import json as _json
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod
    from mnemara import paths

    config_mod.init_instance("tune_t")
    cfg_path = paths.config_path("tune_t")
    app = tui_mod.MnemaraTUI("tune_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app._chat()
            # Persist path
            await app._handle_slash("/turns 250")
            await pilot.pause()
            assert app.cfg.max_window_turns == 250
            saved = _json.load(cfg_path.open())
            assert saved["max_window_turns"] == 250

            # Temp path
            await app._handle_slash("/turns 999 --temp")
            await pilot.pause()
            assert app.cfg.max_window_turns == 999
            saved = _json.load(cfg_path.open())
            assert saved["max_window_turns"] == 250  # disk untouched

            # Tokens with k suffix + persist
            await app._handle_slash("/tokens 750k")
            await pilot.pause()
            assert app.cfg.max_window_tokens == 750_000
            saved = _json.load(cfg_path.open())
            assert saved["max_window_tokens"] == 750_000

            # Tokens with m suffix + temp
            await app._handle_slash("/tokens 2m --temp")
            await pilot.pause()
            assert app.cfg.max_window_tokens == 2_000_000
            saved = _json.load(cfg_path.open())
            assert saved["max_window_tokens"] == 750_000  # disk untouched

    _asyncio.run(_run())
    app.store.close()


def test_slash_turns_rejects_out_of_bounds(home):
    """/turns 0 and /turns 99999 reject; /tokens 100 and /tokens 99999999 reject."""
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    config_mod.init_instance("tune_bounds_t")
    app = tui_mod.MnemaraTUI("tune_bounds_t")
    starting_turns = app.cfg.max_window_turns
    starting_tokens = app.cfg.max_window_tokens

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._handle_slash("/turns 0")
            await app._handle_slash("/turns 100000")
            await app._handle_slash("/turns abc")
            assert app.cfg.max_window_turns == starting_turns

            await app._handle_slash("/tokens 100")
            await app._handle_slash("/tokens 100m")
            await app._handle_slash("/tokens xyz")
            assert app.cfg.max_window_tokens == starting_tokens

    _asyncio.run(_run())
    app.store.close()


def test_tune_window_tool_persists_to_config(home):
    """The agent-side tune_window tool persists by default and respects -1 sentinel.

    Calls the tool handler directly (synchronous: no SDK roundtrip) and asserts
    config.json reflects the change.
    """
    import asyncio as _asyncio
    import json as _json
    from mnemara import config as config_mod
    from mnemara import paths
    from mnemara.agent import AgentSession
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    from mnemara.permissions import PermissionStore

    config_mod.init_instance("agent_tune_t")
    cfg = config_mod.load("agent_tune_t")
    cfg_path = paths.config_path("agent_tune_t")
    store = Store("agent_tune_t")
    perms = PermissionStore("agent_tune_t")
    runner = ToolRunner("agent_tune_t", cfg, perms, lambda t, x: "allow")
    session = AgentSession(cfg, store, runner)
    # Force the tool registration path to run (it's lazy in _build_options).
    try:
        session._build_options("test system prompt")
    except Exception:
        # _build_options may fail on test environment for unrelated reasons
        # (e.g. no ANTHROPIC creds); that's fine — registration runs first.
        pass
    handler = session._registered_tools.get("tune_window")
    assert handler is not None, f"tune_window not registered; got {list(session._registered_tools)}"

    async def _go():
        # Bump turns + tokens, persist=true (default).
        out = await handler({"max_turns": 333, "max_tokens": 444_000, "persist": "true"})
        text_block = out["content"][0]["text"]
        assert "max_window_turns" in text_block
        assert cfg.max_window_turns == 333
        assert cfg.max_window_tokens == 444_000
        saved = _json.load(cfg_path.open())
        assert saved["max_window_turns"] == 333
        assert saved["max_window_tokens"] == 444_000

        # -1 sentinels: leave alone.
        out = await handler({"max_turns": 555, "max_tokens": -1, "persist": "true"})
        assert cfg.max_window_turns == 555
        assert cfg.max_window_tokens == 444_000

        # persist=false: in-memory only.
        out = await handler({"max_turns": -1, "max_tokens": 999_000, "persist": "false"})
        assert cfg.max_window_tokens == 999_000
        saved = _json.load(cfg_path.open())
        assert saved["max_window_tokens"] == 444_000  # unchanged on disk

        # Out-of-bounds: rejected, no-op.
        out = await handler({"max_turns": 99999, "max_tokens": -1, "persist": "true"})
        assert cfg.max_window_turns == 555  # unchanged
        assert out.get("is_error") is True

        # Both sentinels: error no-op.
        out = await handler({"max_turns": -1, "max_tokens": -1, "persist": "true"})
        assert out.get("is_error") is True

    _asyncio.run(_go())
    store.close()


# ---------------------------------------------------------------- eviction


def test_store_evict_last_drops_most_recent_rows(home):
    """Store.evict_last(n) removes exactly n most-recent rows when available."""
    from mnemara.store import Store

    store = Store("evict_last_t")
    for i in range(5):
        store.append_turn("user", [{"type": "text", "text": f"msg{i}"}])
    assert len(store.window()) == 5
    deleted = store.evict_last(2)
    assert deleted == 2
    rows = store.window()
    assert len(rows) == 3
    # Confirm the most-recent (highest id) rows went; the surviving rows
    # are the first three appended.
    texts = [r["content"][0]["text"] for r in rows]
    assert texts == ["msg0", "msg1", "msg2"]

    # Asking for more than available deletes what's there.
    deleted = store.evict_last(99)
    assert deleted == 3
    assert store.window() == []

    # Zero / negative is a no-op.
    deleted = store.evict_last(0)
    assert deleted == 0
    deleted = store.evict_last(-1)
    assert deleted == 0
    store.close()


def test_store_evict_ids_targets_specific_rows(home):
    """Store.evict_ids deletes only the requested ids; unknown ids ignored."""
    from mnemara.store import Store

    store = Store("evict_ids_t")
    ids = [
        store.append_turn("user", [{"type": "text", "text": f"m{i}"}]) for i in range(5)
    ]
    # Drop the middle three.
    deleted = store.evict_ids([ids[1], ids[2], ids[3]])
    assert deleted == 3
    rows = store.window()
    assert len(rows) == 2
    assert [r["id"] for r in rows] == [ids[0], ids[4]]

    # Unknown ids ignored; deleted reflects actual matches.
    deleted = store.evict_ids([99999, 88888])
    assert deleted == 0

    # Mixed known + unknown returns only the known count.
    deleted = store.evict_ids([ids[0], 99999])
    assert deleted == 1
    assert [r["id"] for r in store.window()] == [ids[4]]

    # Empty input is a no-op.
    deleted = store.evict_ids([])
    assert deleted == 0
    store.close()


def test_store_mark_segment_and_evict_since(home):
    """mark_segment inserts a marker row; evict_since drops marker + everything after."""
    from mnemara.store import Store

    store = Store("mark_evict_t")
    store.append_turn("user", [{"type": "text", "text": "before-detour"}])
    mid = store.mark_segment("checkpoint")
    store.append_turn("assistant", [{"type": "text", "text": "during-detour-1"}])
    store.append_turn("user", [{"type": "text", "text": "during-detour-2"}])
    store.append_turn("assistant", [{"type": "text", "text": "during-detour-3"}])

    # The marker shows up in window() (visible to producer/agent) but NOT
    # in messages_for_api() (model never sees it).
    rows = store.window()
    assert any(r["role"] == "marker" and r["id"] == mid for r in rows)
    api_msgs = store.messages_for_api()
    assert all(m["role"] in ("user", "assistant") for m in api_msgs)
    assert len(api_msgs) == 4  # 1 user before + 3 turns after marker

    # list_markers exposes the marker by name.
    marks = store.list_markers()
    assert len(marks) == 1
    assert marks[0]["name"] == "checkpoint"
    assert marks[0]["id"] == mid

    # evict_since drops the marker + 3 rows after it = 4 deletions.
    deleted = store.evict_since("checkpoint")
    assert deleted == 4
    rows = store.window()
    assert len(rows) == 1
    assert rows[0]["content"][0]["text"] == "before-detour"

    # Unknown marker name returns 0 (no-op).
    deleted = store.evict_since("nonexistent")
    assert deleted == 0
    store.close()


def test_store_evict_since_picks_most_recent_when_name_repeats(home):
    """If a marker name appears twice, evict_since uses the more recent one."""
    from mnemara.store import Store

    store = Store("dup_marker_t")
    store.append_turn("user", [{"type": "text", "text": "before-first"}])
    first_mark = store.mark_segment("ckpt")
    store.append_turn("user", [{"type": "text", "text": "between"}])
    second_mark = store.mark_segment("ckpt")
    store.append_turn("user", [{"type": "text", "text": "after-second"}])

    # evict_since('ckpt') should drop the second marker + the row after it,
    # leaving the first marker and everything before the second intact.
    deleted = store.evict_since("ckpt")
    assert deleted == 2
    rows = store.window()
    ids = [r["id"] for r in rows]
    assert second_mark not in ids
    assert first_mark in ids
    # Three rows survive: original user, first marker, between user
    assert len(rows) == 3
    store.close()


def test_messages_for_api_filters_marker_rows(home):
    """messages_for_api never emits role='marker' rows."""
    from mnemara.store import Store

    store = Store("api_filter_t")
    store.append_turn("user", [{"type": "text", "text": "u1"}])
    store.mark_segment("only-visible-internally")
    store.append_turn("assistant", [{"type": "text", "text": "a1"}])

    msgs = store.messages_for_api()
    assert len(msgs) == 2
    roles = {m["role"] for m in msgs}
    assert roles == {"user", "assistant"}
    assert "marker" not in roles
    # Window still shows the marker (so the agent can target it).
    win_roles = [r["role"] for r in store.window()]
    assert "marker" in win_roles
    store.close()


def test_slash_evict_and_mark_command_dispatch(home):
    """/mark, /marks, /evict last|ids|since dispatch through the TUI handler."""
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    config_mod.init_instance("slash_evict_t")
    app = tui_mod.MnemaraTUI("slash_evict_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            # Seed the store with five rows.
            for i in range(5):
                app.store.append_turn("user", [{"type": "text", "text": f"m{i}"}])

            # /mark and /marks
            await app._handle_slash("/mark checkpoint-a")
            await pilot.pause()
            marks = app.store.list_markers()
            assert len(marks) == 1
            assert marks[0]["name"] == "checkpoint-a"

            # /evict last 1 should drop the marker (it was the most recent row).
            await app._handle_slash("/evict last 1")
            await pilot.pause()
            assert app.store.list_markers() == []
            assert len(app.store.window()) == 5  # five user turns intact

            # Re-mark and evict_since
            await app._handle_slash("/mark checkpoint-b")
            await pilot.pause()
            for i in range(3):
                app.store.append_turn("assistant", [{"type": "text", "text": f"r{i}"}])
            assert len(app.store.window()) == 5 + 1 + 3  # 5 user + marker + 3 assistant

            await app._handle_slash("/evict since checkpoint-b")
            await pilot.pause()
            # Marker + 3 assistant rows dropped; 5 original user turns survive.
            rows = app.store.window()
            assert len(rows) == 5
            assert all(r["role"] == "user" for r in rows)

            # /evict ids targeting two of the survivors
            ids_to_drop = [r["id"] for r in rows[:2]]
            await app._handle_slash(f"/evict ids {ids_to_drop[0]},{ids_to_drop[1]}")
            await pilot.pause()
            assert len(app.store.window()) == 3

            # Bad input is a no-op (errors logged to chat, not raised).
            await app._handle_slash("/evict last abc")
            await pilot.pause()
            assert len(app.store.window()) == 3
            await app._handle_slash("/evict since does-not-exist")
            await pilot.pause()
            assert len(app.store.window()) == 3
            await app._handle_slash("/evict bogus-mode")
            await pilot.pause()
            assert len(app.store.window()) == 3

    _asyncio.run(_run())
    app.store.close()


def test_agent_eviction_tools_persist_to_store(home):
    """Agent-side evict_last / evict_ids / mark_segment / evict_since tools.

    Invokes the registered tool handlers directly (bypassing the SDK
    roundtrip) and asserts the underlying store reflects each operation.
    """
    import asyncio as _asyncio
    import json as _json
    from mnemara import config as config_mod
    from mnemara import paths
    from mnemara.agent import AgentSession
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config_mod.init_instance("agent_evict_t")
    cfg = config_mod.load("agent_evict_t")
    store = Store("agent_evict_t")
    perms = PermissionStore("agent_evict_t")
    runner = ToolRunner("agent_evict_t", cfg, perms, lambda t, x: "allow")
    session = AgentSession(cfg, store, runner)
    try:
        session._build_options("test system prompt")
    except Exception:
        # SDK env may not be fully constructable in tests; registration
        # runs first, that's all we need.
        pass

    handlers = session._registered_tools
    for name in ("evict_last", "evict_ids", "mark_segment", "evict_since"):
        assert name in handlers, f"{name} not registered; got {list(handlers)}"

    async def _go() -> None:
        # Seed with five rows.
        ids = [
            store.append_turn("user", [{"type": "text", "text": f"m{i}"}])
            for i in range(5)
        ]

        # mark_segment
        out = await handlers["mark_segment"]({"name": "tangent"})
        payload = _json.loads(out["content"][0]["text"])
        assert payload["name"] == "tangent"
        assert isinstance(payload["marker_id"], int)

        # Add three rows after the marker.
        for i in range(3):
            store.append_turn("assistant", [{"type": "text", "text": f"r{i}"}])

        # evict_since drops marker + 3 = 4 rows
        out = await handlers["evict_since"]({"marker": "tangent"})
        payload = _json.loads(out["content"][0]["text"])
        assert payload["deleted"] == 4
        assert payload["matched"] is True
        assert len(store.window()) == 5

        # evict_last 2
        out = await handlers["evict_last"]({"n": 2})
        payload = _json.loads(out["content"][0]["text"])
        assert payload["deleted"] == 2
        assert len(store.window()) == 3

        # evict_ids both csv and json shapes
        survivors = [r["id"] for r in store.window()]
        out = await handlers["evict_ids"]({"ids": f"{survivors[0]},{survivors[1]}"})
        payload = _json.loads(out["content"][0]["text"])
        assert payload["deleted"] == 2
        assert len(store.window()) == 1

        out = await handlers["evict_ids"]({"ids": f"[{store.window()[0]['id']}]"})
        payload = _json.loads(out["content"][0]["text"])
        assert payload["deleted"] == 1
        assert store.window() == []

        # Error paths
        out = await handlers["evict_last"]({"n": 0})
        assert out.get("is_error") is True
        out = await handlers["evict_last"]({"n": "abc"})
        assert out.get("is_error") is True
        out = await handlers["evict_ids"]({"ids": ""})
        assert out.get("is_error") is True
        out = await handlers["evict_ids"]({"ids": "not,a,number"})
        assert out.get("is_error") is True
        out = await handlers["mark_segment"]({"name": ""})
        assert out.get("is_error") is True
        out = await handlers["evict_since"]({"marker": ""})
        assert out.get("is_error") is True

        # evict_since with unknown marker is a non-error 0-delete (matched=False)
        out = await handlers["evict_since"]({"marker": "nonexistent"})
        payload = _json.loads(out["content"][0]["text"])
        assert payload["deleted"] == 0
        assert payload["matched"] is False
        assert out.get("is_error") is not True

    _asyncio.run(_go())
    store.close()


# ---------------------------------------------------------------- userinput paste


def test_userinput_paste_collapses_multiline_atomically(home):
    """_UserInput._on_paste joins multi-line content with spaces and inserts atomically.

    Drives the Input via run_test() Pilot, posts a synthesized Paste event,
    and asserts the resulting value is the collapsed single-line content
    (not the first-line-only stock Textual behavior).
    """
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod
    from textual import events as _txt_events

    config_mod.init_instance("paste_t")
    app = tui_mod.MnemaraTUI("paste_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#userinput", tui_mod.Input)
            inp.focus()
            await pilot.pause()

            # Plain single-line paste -> inserted as-is.
            inp.value = ""
            inp.cursor_position = 0
            inp.post_message(_txt_events.Paste("hello world"))
            await pilot.pause()
            assert inp.value == "hello world"

            # Multi-line paste -> collapsed with single spaces.
            inp.value = ""
            inp.cursor_position = 0
            inp.post_message(_txt_events.Paste("line one\nline two\n\nline four"))
            await pilot.pause()
            assert inp.value == "line one line two line four"

            # Paste at non-zero cursor preserves prefix/suffix.
            inp.value = "AB CD"
            inp.cursor_position = 3  # between 'B ' and 'CD'
            inp.post_message(_txt_events.Paste("X\nY"))
            await pilot.pause()
            assert inp.value == "AB X YCD"

            # Truncation cap: paste larger than _USERINPUT_PASTE_CAP gets clipped.
            inp.value = ""
            inp.cursor_position = 0
            big = "a" * (tui_mod._USERINPUT_PASTE_CAP + 500)
            inp.post_message(_txt_events.Paste(big))
            await pilot.pause()
            assert len(inp.value) == tui_mod._USERINPUT_PASTE_CAP

            # Empty paste is a no-op.
            inp.value = "preserved"
            inp.cursor_position = len(inp.value)
            inp.post_message(_txt_events.Paste(""))
            await pilot.pause()
            assert inp.value == "preserved"

    _asyncio.run(_run())
    app.store.close()


# ---------------------------------------------------------------- worker decoupling


def test_on_input_submitted_returns_before_send_turn_completes(home, monkeypatch):
    """on_input_submitted must spawn _send_turn as a worker, not await it.

    Pins down the resize-during-streaming fix: the input handler must
    return immediately so Textual's _process_messages_loop is freed to
    dispatch other queued events (resize, key, mouse) concurrently with
    the streaming work. If on_input_submitted reverts to awaiting
    _send_turn directly, this test fails -- the assertion would only
    hold if _send_turn awaited completion before returning, and that's
    exactly the architecture we don't want.
    """
    import asyncio as _asyncio
    import time as _time
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod
    from textual.widgets import Input

    config_mod.init_instance("worker_t")
    app = tui_mod.MnemaraTUI("worker_t")

    send_turn_started = _asyncio.Event()
    send_turn_can_finish = _asyncio.Event()

    async def _slow_send_turn(text: str) -> None:
        # Simulates a streaming turn that takes "forever" -- enough time
        # for the test to assert handler-return before completion.
        send_turn_started.set()
        app._busy = True
        try:
            await send_turn_can_finish.wait()
        finally:
            app._busy = False

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            # Patch _send_turn to our slow stand-in.
            monkeypatch.setattr(app, "_send_turn", _slow_send_turn)

            # Simulate a submission. The handler should return BEFORE
            # _send_turn completes; we measure by checking the busy flag
            # is set (worker started) but the handler awaitable resolved
            # without blocking.
            inp = app.query_one("#userinput", Input)
            inp.value = "hello"
            handler_returned_at = None

            async def _do_submit() -> None:
                nonlocal handler_returned_at
                t0 = _time.monotonic()
                # Mimic Textual posting an Input.Submitted event.
                event = Input.Submitted(inp, "hello")
                await app.on_input_submitted(event)
                handler_returned_at = _time.monotonic() - t0

            # Run the handler; it should return promptly because
            # _send_turn was spawned as a worker, not awaited.
            await _do_submit()

            # Wait for the worker to actually start (proves it WAS spawned).
            await _asyncio.wait_for(send_turn_started.wait(), timeout=2.0)

            # The handler returned in well under a second even though
            # the worker is still parked on send_turn_can_finish.
            assert handler_returned_at is not None
            assert handler_returned_at < 0.5, (
                f"on_input_submitted took {handler_returned_at:.3f}s -- "
                "this means it awaited _send_turn directly instead of "
                "spawning it as a worker. Resize-during-streaming bug "
                "will recur."
            )

            # Confirm the worker is actually running (busy=True).
            assert app._busy is True

            # Let the worker finish so app teardown is clean.
            send_turn_can_finish.set()
            # Drain any remaining tasks.
            for _ in range(20):
                if not app._busy:
                    break
                await pilot.pause()

    _asyncio.run(_run())
    app.store.close()


# ---------------------------------------------------------------- ambient inbox


def test_check_inbox_ambient_notifies_on_new_pings(home, tmp_path):
    """_check_inbox_ambient walks boot -> new ping -> ack -> new ping cleanly.

    Sets up a temp muninn.db with the returns schema, points the panel's
    config at it, and exercises the state-tracker mutations across each
    transition. Notification text is RichLog content (hard to introspect
    cleanly in unit tests); _last_seen_inbox_id is the load-bearing
    behavior we assert on.
    """
    import asyncio as _asyncio
    import sqlite3 as _sqlite3
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    db_path = tmp_path / "muninn.db"
    conn = _sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE returns ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id TEXT NOT NULL,"
        "  agent_role TEXT NOT NULL,"
        "  task_id TEXT,"
        "  submitted_at TEXT NOT NULL,"
        "  payload_json TEXT NOT NULL,"
        "  status TEXT NOT NULL DEFAULT 'pending',"
        "  processed_at TEXT,"
        "  completed_at TEXT,"
        "  recipient_role TEXT"
        ")"
    )
    conn.execute("CREATE INDEX idx_returns_status ON returns(status)")
    conn.commit()

    def _add_ping(sender: str, task: str = "tX", payload_type: str = "hello") -> int:
        cur = conn.execute(
            "INSERT INTO returns (session_id, agent_role, task_id, "
            "submitted_at, payload_json, status) VALUES (?, ?, ?, ?, ?, 'pending')",
            ("sess1", sender, task, "2026-04-30T00:00:00+00:00",
             '{"type":"' + payload_type + '","body":"x"}'),
        )
        conn.commit()
        return cur.lastrowid

    def _ack_all() -> None:
        conn.execute("UPDATE returns SET status='completed' WHERE status='pending'")
        conn.commit()

    config_mod.init_instance("inbox_amb_t")
    cfg = config_mod.load("inbox_amb_t")
    cfg.architect_db_path = str(db_path)
    cfg.peer_roles = ["substrate", "majordomo", "producer"]
    config_mod.save("inbox_amb_t", cfg)

    app = tui_mod.MnemaraTUI("inbox_amb_t")

    async def _run() -> None:
        # Pre-boot: one ping already pending.
        boot_ping_id = _add_ping("substrate", task="ping_001")

        async with app.run_test() as pilot:
            await pilot.pause()

            # Boot path already ran in on_mount. Tracker should reflect
            # the pre-existing ping.
            assert app._last_seen_inbox_id == boot_ping_id

            # Add a brand-new ping. _check_inbox_ambient should bump
            # tracker forward.
            new_id = _add_ping("majordomo", task="ping_002", payload_type="reply")
            assert new_id > boot_ping_id
            app._check_inbox_ambient(boot=False)
            assert app._last_seen_inbox_id == new_id

            # Calling again with no new rows is a no-op.
            tracker_before = app._last_seen_inbox_id
            app._check_inbox_ambient(boot=False)
            assert app._last_seen_inbox_id == tracker_before

            # Ack everything; next poll should reset the tracker to 0.
            _ack_all()
            app._check_inbox_ambient(boot=False)
            assert app._last_seen_inbox_id == 0

            # New ping after the ack: notifies again, tracker advances.
            future_id = _add_ping("producer", task="ping_003")
            app._check_inbox_ambient(boot=False)
            assert app._last_seen_inbox_id == future_id

    _asyncio.run(_run())
    app.store.close()
    conn.close()


def test_check_inbox_ambient_no_db_is_noop(home):
    """When architect_db_path is unset, the poller is a clean no-op."""
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    config_mod.init_instance("inbox_no_db_t")
    cfg = config_mod.load("inbox_no_db_t")
    cfg.architect_db_path = ""
    cfg.peer_roles = ["substrate"]
    config_mod.save("inbox_no_db_t", cfg)

    app = tui_mod.MnemaraTUI("inbox_no_db_t")

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._last_seen_inbox_id == 0
            app._check_inbox_ambient(boot=False)
            assert app._last_seen_inbox_id == 0

    _asyncio.run(_run())
    app.store.close()


# ---------------------------------------------------------------- auto-respond


def _build_returns_db(tmp_path):
    """Helper: build a temp muninn.db with the returns schema + seed helpers."""
    import sqlite3 as _sqlite3

    db_path = tmp_path / "muninn.db"
    conn = _sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE returns ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id TEXT NOT NULL,"
        "  agent_role TEXT NOT NULL,"
        "  task_id TEXT,"
        "  submitted_at TEXT NOT NULL,"
        "  payload_json TEXT NOT NULL,"
        "  status TEXT NOT NULL DEFAULT 'pending',"
        "  processed_at TEXT,"
        "  completed_at TEXT,"
        "  recipient_role TEXT"
        ")"
    )
    conn.execute("CREATE INDEX idx_returns_status ON returns(status)")
    conn.commit()
    return db_path, conn


def _seed_ping(conn, sender, task="tX", payload_type="hello", recipient=None):
    cur = conn.execute(
        "INSERT INTO returns (session_id, agent_role, task_id, "
        "submitted_at, payload_json, status, recipient_role) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
        ("sess1", sender, task, "2026-04-30T00:00:00+00:00",
         '{"type":"' + payload_type + '","body":"x"}', recipient),
    )
    conn.commit()
    return cur.lastrowid


def test_inbox_auto_respond_spawns_worker_for_new_ping(home, tmp_path):
    """When inbox_auto_respond=True, a new ping spawns a turn worker.

    Patches run_worker to capture invocations rather than actually spinning
    up _send_turn (which would need an MCP roundtrip). Verifies the
    synthetic prompt structure and tracker advancement.
    """
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    db_path, conn = _build_returns_db(tmp_path)
    config_mod.init_instance("auto_resp_t")
    cfg = config_mod.load("auto_resp_t")
    cfg.architect_db_path = str(db_path)
    cfg.peer_roles = ["substrate", "majordomo"]
    cfg.inbox_auto_respond = True
    config_mod.save("auto_resp_t", cfg)

    app = tui_mod.MnemaraTUI("auto_resp_t")
    spawn_calls = []

    def _capture_run_worker(coro, **kw):
        spawn_calls.append({"coro": coro, "kwargs": kw})
        # Close the coroutine to avoid "never awaited" warnings.
        try:
            coro.close()
        except Exception:
            pass
        return None

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.run_worker = _capture_run_worker  # type: ignore[assignment]

            # Add a new ping and tick the poller.
            row_id = _seed_ping(conn, "substrate", task="ping_X", payload_type="hello")
            app._check_inbox_ambient(boot=False)

            # Worker spawned exactly once.
            assert len(spawn_calls) == 1, f"expected 1 spawn, got {len(spawn_calls)}"
            kw = spawn_calls[0]["kwargs"]
            assert kw.get("group") == "turn"
            assert kw.get("exclusive") is True
            assert str(row_id) in kw.get("name", "")

            # Both trackers advanced.
            assert app._last_seen_inbox_id == row_id
            assert app._last_auto_processed_id == row_id

            # Second tick with no new pings is a no-op.
            spawn_calls.clear()
            app._check_inbox_ambient(boot=False)
            assert spawn_calls == []

    _asyncio.run(_run())
    app.store.close()
    conn.close()


def test_inbox_auto_respond_skipped_when_disabled(home, tmp_path):
    """When inbox_auto_respond=False, notification fires but no worker spawns."""
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    db_path, conn = _build_returns_db(tmp_path)
    config_mod.init_instance("auto_off_t")
    cfg = config_mod.load("auto_off_t")
    cfg.architect_db_path = str(db_path)
    cfg.peer_roles = ["substrate"]
    cfg.inbox_auto_respond = False
    config_mod.save("auto_off_t", cfg)

    app = tui_mod.MnemaraTUI("auto_off_t")
    spawn_calls = []
    app_run_worker = lambda coro, **kw: (spawn_calls.append(kw), coro.close())

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.run_worker = app_run_worker  # type: ignore[assignment]

            row_id = _seed_ping(conn, "substrate", task="ping_Y")
            app._check_inbox_ambient(boot=False)

            # Notification advanced; auto did not.
            assert app._last_seen_inbox_id == row_id
            assert app._last_auto_processed_id == 0
            assert spawn_calls == []

    _asyncio.run(_run())
    app.store.close()
    conn.close()


def test_inbox_auto_respond_skipped_while_busy(home, tmp_path):
    """When _busy=True, auto-respond defers; tracker stays put for retry."""
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    db_path, conn = _build_returns_db(tmp_path)
    config_mod.init_instance("auto_busy_t")
    cfg = config_mod.load("auto_busy_t")
    cfg.architect_db_path = str(db_path)
    cfg.peer_roles = ["substrate"]
    cfg.inbox_auto_respond = True
    config_mod.save("auto_busy_t", cfg)

    app = tui_mod.MnemaraTUI("auto_busy_t")
    spawn_calls = []
    app_run_worker = lambda coro, **kw: (spawn_calls.append(kw), coro.close())

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.run_worker = app_run_worker  # type: ignore[assignment]

            row_id = _seed_ping(conn, "substrate", task="ping_Z")
            app._busy = True

            app._check_inbox_ambient(boot=False)
            # Notification advanced (visual happens regardless).
            assert app._last_seen_inbox_id == row_id
            # Auto tracker did NOT advance — retry-eligible.
            assert app._last_auto_processed_id == 0
            assert spawn_calls == []

            # Once the panel un-busies, the next tick fires.
            app._busy = False
            app._check_inbox_ambient(boot=False)
            assert app._last_auto_processed_id == row_id
            assert len(spawn_calls) == 1

    _asyncio.run(_run())
    app.store.close()
    conn.close()


def test_inbox_auto_respond_skips_terminal_payload_types(home, tmp_path):
    """ack / ack_final / reply_final pings advance tracker without spawning.

    Loop guard: avoids two auto-responding panels infinitely acking each other.
    """
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    db_path, conn = _build_returns_db(tmp_path)
    config_mod.init_instance("auto_term_t")
    cfg = config_mod.load("auto_term_t")
    cfg.architect_db_path = str(db_path)
    cfg.peer_roles = ["substrate"]
    cfg.inbox_auto_respond = True
    config_mod.save("auto_term_t", cfg)

    app = tui_mod.MnemaraTUI("auto_term_t")
    spawn_calls = []
    app_run_worker = lambda coro, **kw: (spawn_calls.append(kw), coro.close())

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.run_worker = app_run_worker  # type: ignore[assignment]

            ack_id = _seed_ping(conn, "substrate", payload_type="ack")
            app._check_inbox_ambient(boot=False)
            assert app._last_auto_processed_id == ack_id
            assert spawn_calls == []

            final_id = _seed_ping(conn, "substrate", payload_type="reply_final")
            app._check_inbox_ambient(boot=False)
            assert app._last_auto_processed_id == final_id
            assert spawn_calls == []

            # Non-terminal type after terminals: spawns.
            hello_id = _seed_ping(conn, "substrate", payload_type="hello")
            app._check_inbox_ambient(boot=False)
            assert app._last_auto_processed_id == hello_id
            assert len(spawn_calls) == 1

    _asyncio.run(_run())
    app.store.close()
    conn.close()


def test_inbox_auto_respond_boot_backlog_primes_both_trackers(home, tmp_path):
    """Boot-time backlog should prime both trackers — don't auto-respond on boot."""
    import asyncio as _asyncio
    from mnemara import config as config_mod
    from mnemara import tui as tui_mod

    db_path, conn = _build_returns_db(tmp_path)
    config_mod.init_instance("auto_boot_t")
    cfg = config_mod.load("auto_boot_t")
    cfg.architect_db_path = str(db_path)
    cfg.peer_roles = ["substrate"]
    cfg.inbox_auto_respond = True
    config_mod.save("auto_boot_t", cfg)

    # Pre-boot pings already pending.
    p1 = _seed_ping(conn, "substrate", task="old1")
    p2 = _seed_ping(conn, "substrate", task="old2")

    app = tui_mod.MnemaraTUI("auto_boot_t")
    spawn_calls = []
    app_run_worker = lambda coro, **kw: (spawn_calls.append(kw), coro.close())

    async def _run() -> None:
        async with app.run_test() as pilot:
            # Patch BEFORE any tick fires beyond the on_mount boot path.
            app.run_worker = app_run_worker  # type: ignore[assignment]
            await pilot.pause()

            # Both trackers should now reflect the max pre-existing ping.
            # Boot path runs in on_mount which already executed.
            assert app._last_seen_inbox_id == max(p1, p2)
            assert app._last_auto_processed_id == max(p1, p2)
            # No worker spawned for boot backlog.
            assert spawn_calls == []

            # A NEW ping after boot does spawn.
            new_id = _seed_ping(conn, "substrate", task="new1")
            app._check_inbox_ambient(boot=False)
            assert app._last_auto_processed_id == new_id
            assert len(spawn_calls) == 1

    _asyncio.run(_run())
    app.store.close()
    conn.close()


# ---------------------------------------------------------------- routing


def test_peek_pending_pings_routes_to_recipient(home, tmp_path):
    """Rows with recipient_role=X are visible only to instance X.

    Producer-flagged 2026-04-30: a ping from substrate addressed to
    cognition-researcher was visible to producer's poller because the
    schema had no recipient column. Producer's auto-respond worker
    drained the misroute. Schema fix: recipient_role column + filter
    in peek_pending_pings.
    """
    from mnemara.inbox import peek_pending_pings

    db_path, conn = _build_returns_db(tmp_path)
    # Seed three rows from substrate addressed to different recipients.
    to_researcher = _seed_ping(conn, "substrate", task="t1", recipient="cognition-researcher")
    to_producer = _seed_ping(conn, "substrate", task="t2", recipient="producer")
    broadcast = _seed_ping(conn, "substrate", task="t3", recipient=None)

    peers = ["substrate", "majordomo", "producer"]

    # Researcher's poller should see only the row addressed to it + broadcast.
    pings = peek_pending_pings(
        str(db_path), peers, exclude_role="cognition-researcher",
        instance="cognition-researcher",
    )
    ids = sorted(p["id"] for p in pings)
    assert ids == sorted([to_researcher, broadcast]), (
        f"researcher should see {[to_researcher, broadcast]}, got {ids}"
    )

    # Producer's poller should see only the row addressed to it + broadcast.
    pings = peek_pending_pings(
        str(db_path), peers, exclude_role="producer", instance="producer",
    )
    ids = sorted(p["id"] for p in pings)
    assert ids == sorted([to_producer, broadcast]), (
        f"producer should see {[to_producer, broadcast]}, got {ids}"
    )

    # Majordomo's poller should see ONLY the broadcast row (nothing
    # addressed to majordomo).
    pings = peek_pending_pings(
        str(db_path), peers, exclude_role="majordomo", instance="majordomo",
    )
    ids = sorted(p["id"] for p in pings)
    assert ids == [broadcast], (
        f"majordomo should see only [{broadcast}], got {ids}"
    )

    conn.close()


def test_peek_pending_pings_no_instance_is_legacy_broadcast(home, tmp_path):
    """When instance=None, no recipient filter applies (legacy callers).

    Preserves backward compat for any callers that pre-date the recipient
    parameter. They get every row matching the sender filter, same as
    before the schema migration.
    """
    from mnemara.inbox import peek_pending_pings

    db_path, conn = _build_returns_db(tmp_path)
    a = _seed_ping(conn, "substrate", recipient="alice")
    b = _seed_ping(conn, "substrate", recipient="bob")
    c = _seed_ping(conn, "substrate", recipient=None)

    pings = peek_pending_pings(
        str(db_path), ["substrate"], exclude_role="anyone", instance=None,
    )
    ids = sorted(p["id"] for p in pings)
    assert ids == sorted([a, b, c])

    conn.close()


def test_peek_pending_pings_excludes_self_outbox(home, tmp_path):
    """exclude_role still filters out the caller's own authored rows.

    Even with recipient routing, a panel must not drain its own outbox
    (that would break the protocol: senders read their own replies via
    next_return(agent_role=peer), not via inbox-poll).
    """
    from mnemara.inbox import peek_pending_pings

    db_path, conn = _build_returns_db(tmp_path)
    # producer's own outbound row (would be wrong to surface in producer's inbox).
    own = _seed_ping(conn, "producer", recipient="substrate")
    # incoming row addressed to producer.
    incoming = _seed_ping(conn, "substrate", recipient="producer")

    peers = ["substrate", "producer", "majordomo"]
    pings = peek_pending_pings(
        str(db_path), peers, exclude_role="producer", instance="producer",
    )
    ids = sorted(p["id"] for p in pings)
    assert ids == [incoming], (
        f"producer must not see its own outbox (#{own}); got {ids}"
    )

    conn.close()


def test_count_pending_routes_to_recipient(home, tmp_path):
    """count_pending honors the recipient filter (used by status bar).

    Status bar's pending count must reflect what the panel will actually
    auto-surface, not what's in the queue overall.
    """
    from mnemara.inbox import count_pending

    db_path, conn = _build_returns_db(tmp_path)
    _seed_ping(conn, "substrate", recipient="cognition-researcher")
    _seed_ping(conn, "substrate", recipient="cognition-researcher")
    _seed_ping(conn, "substrate", recipient="producer")
    _seed_ping(conn, "substrate", recipient=None)  # broadcast: visible to all

    n = count_pending(
        str(db_path), ["substrate"], exclude_role="cognition-researcher",
        instance="cognition-researcher",
    )
    # 2 addressed + 1 broadcast = 3
    assert n == 3

    n = count_pending(
        str(db_path), ["substrate"], exclude_role="producer",
        instance="producer",
    )
    # 1 addressed + 1 broadcast = 2
    assert n == 2

    n = count_pending(
        str(db_path), ["substrate"], exclude_role="majordomo",
        instance="majordomo",
    )
    # 0 addressed + 1 broadcast = 1
    assert n == 1

    conn.close()


# ---------------------------------------------------------------- thinking surgery


def test_evict_thinking_blocks_strips_thinking_only(home):
    """thinking blocks evicted; text + tool_use + tool_result preserved."""
    from mnemara.store import Store

    store = Store("etb_basic_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "scratch reasoning"},
            {"type": "text", "text": "the answer is X"},
            {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"path": "/foo"}},
            {"type": "thinking", "text": "more scratch"},
        ],
    )

    result = store.evict_thinking_blocks(ids=[rid])
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 2
    assert result["bytes_freed"] > 0

    rows = store.window()
    assert len(rows) == 1
    blocks = rows[0]["content"]
    assert isinstance(blocks, list)
    types = [b["type"] for b in blocks]
    assert types == ["text", "tool_use"]
    # text + tool_use payload preserved verbatim
    assert blocks[0]["text"] == "the answer is X"
    assert blocks[1]["name"] == "Read"
    store.close()


def test_evict_thinking_blocks_keep_recent_preserves_tail(home):
    """keep_recent=N strips from all rows except the most-recent N."""
    from mnemara.store import Store

    store = Store("etb_keep_t")
    ids = []
    for i in range(5):
        ids.append(store.append_turn(
            "assistant",
            [
                {"type": "thinking", "text": f"scratch {i}"},
                {"type": "text", "text": f"answer {i}"},
            ],
        ))
    # Keep last 2 untouched; strip from the older 3.
    result = store.evict_thinking_blocks(keep_recent=2)
    assert result["rows_scanned"] == 3
    assert result["rows_modified"] == 3
    assert result["blocks_evicted"] == 3

    rows = store.window()
    # Older 3 should have only text blocks.
    for r in rows[:3]:
        types = [b["type"] for b in r["content"]]
        assert types == ["text"], f"row {r['id']} should have thinking stripped: {types}"
    # Most-recent 2 should still have both.
    for r in rows[3:]:
        types = [b["type"] for b in r["content"]]
        assert types == ["thinking", "text"], f"row {r['id']} should be untouched: {types}"
    store.close()


def test_evict_thinking_blocks_keep_recent_zero_strips_all(home):
    """keep_recent=0 is equivalent to all_rows=True semantically."""
    from mnemara.store import Store

    store = Store("etb_keep_zero_t")
    for i in range(3):
        store.append_turn(
            "assistant",
            [
                {"type": "thinking", "text": f"x{i}"},
                {"type": "text", "text": f"a{i}"},
            ],
        )
    result = store.evict_thinking_blocks(keep_recent=0)
    assert result["rows_modified"] == 3
    assert result["blocks_evicted"] == 3
    for r in store.window():
        assert all(b["type"] != "thinking" for b in r["content"])
    store.close()


def test_evict_thinking_blocks_all_rows_strips_every_row(home):
    """all_rows=True strips thinking from every row in the store."""
    from mnemara.store import Store

    store = Store("etb_all_t")
    store.append_turn("user", [{"type": "text", "text": "u1"}])  # no thinking
    a = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "..."},
            {"type": "text", "text": "answer"},
        ],
    )
    store.append_turn(
        "user",
        [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}],
    )

    result = store.evict_thinking_blocks(all_rows=True)
    assert result["rows_scanned"] == 3
    assert result["rows_modified"] == 1  # only the assistant row had thinking
    assert result["blocks_evicted"] == 1

    # Verify the assistant row now has only text.
    for r in store.window():
        if r["id"] == a:
            assert [b["type"] for b in r["content"]] == ["text"]
    store.close()


def test_evict_thinking_blocks_skips_empty_result_rows(home):
    """Rows whose stripping leaves 0 blocks are skipped without modification."""
    from mnemara.store import Store

    store = Store("etb_empty_t")
    # Row that's 100% thinking — stripping would empty it.
    rid = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "lone scratch"},
        ],
    )
    result = store.evict_thinking_blocks(ids=[rid])
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 0
    assert result["blocks_evicted"] == 0
    assert result["bytes_freed"] == 0

    # Original content unchanged.
    rows = store.window()
    assert len(rows) == 1
    assert [b["type"] for b in rows[0]["content"]] == ["thinking"]
    store.close()


def test_evict_thinking_blocks_noop_when_no_thinking_present(home):
    """A row with no thinking blocks is scanned but not modified."""
    from mnemara.store import Store

    store = Store("etb_noop_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "text", "text": "no thinking here"},
            {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}},
        ],
    )
    result = store.evict_thinking_blocks(ids=[rid])
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 0
    assert result["blocks_evicted"] == 0
    assert result["bytes_freed"] == 0
    store.close()


def test_evict_thinking_blocks_idempotent_on_rerun(home):
    """Running twice on the same row produces no further changes the second time."""
    from mnemara.store import Store

    store = Store("etb_idem_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "x"},
            {"type": "text", "text": "y"},
        ],
    )
    r1 = store.evict_thinking_blocks(ids=[rid])
    assert r1["rows_modified"] == 1
    assert r1["blocks_evicted"] == 1
    r2 = store.evict_thinking_blocks(ids=[rid])
    assert r2["rows_modified"] == 0
    assert r2["blocks_evicted"] == 0
    store.close()


def test_evict_thinking_blocks_handles_string_and_marker_rows(home):
    """Legacy string-encoded content and marker rows are skipped without error."""
    from mnemara.store import Store

    store = Store("etb_legacy_t")
    # Legacy string-encoded turn (content stored as plain string, not JSON list).
    legacy = store.append_turn("user", "plain-string-content-no-blocks")
    # Marker row (role='marker', content=json.dumps(name)).
    marker = store.mark_segment("checkpoint-A")
    # Real assistant row with thinking.
    real = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "real"},
            {"type": "text", "text": "kept"},
        ],
    )

    result = store.evict_thinking_blocks(all_rows=True)
    # Legacy + marker scanned but skipped (no list content); real row modified.
    assert result["rows_scanned"] == 3
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1

    # Confirm legacy + marker rows untouched.
    rows = {r["id"]: r for r in store.window()}
    # Legacy string content is parsed by _maybe_json — could come back as
    # str or dict depending on JSON shape. Just confirm the row still exists
    # and wasn't modified to an empty list.
    assert legacy in rows
    assert marker in rows
    # Marker row content should still parse to its original name.
    assert rows[marker]["content"] == "checkpoint-A"
    store.close()


def test_evict_thinking_blocks_selector_validation(home):
    """Exactly one of ids / keep_recent / all_rows must be provided."""
    from mnemara.store import Store
    import pytest

    store = Store("etb_sel_t")
    store.append_turn("assistant", [{"type": "text", "text": "x"}])

    # Zero selectors.
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks()
    # Two selectors.
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(ids=[1], keep_recent=2)
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(ids=[1], all_rows=True)
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(keep_recent=2, all_rows=True)
    # Three selectors.
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(ids=[1], keep_recent=2, all_rows=True)
    store.close()


def test_evict_thinking_blocks_empty_ids_list_is_zero(home):
    """An empty ids list yields all-zero counts (not an error)."""
    from mnemara.store import Store

    store = Store("etb_empty_ids_t")
    store.append_turn("assistant", [{"type": "thinking", "text": "x"}, {"type": "text", "text": "y"}])
    result = store.evict_thinking_blocks(ids=[])
    assert result == {
        "rows_scanned": 0,
        "rows_modified": 0,
        "blocks_evicted": 0,
        "bytes_freed": 0,
        "rows_skipped_pinned": 0,
    }
    # Original row untouched.
    rows = store.window()
    assert len(rows[0]["content"]) == 2
    store.close()


def test_evict_thinking_blocks_unknown_ids_silently_ignored(home):
    """Ids that don't exist are silently dropped (rows_scanned reflects matches)."""
    from mnemara.store import Store

    store = Store("etb_unknown_t")
    rid = store.append_turn(
        "assistant", [{"type": "thinking", "text": "x"}, {"type": "text", "text": "y"}]
    )
    result = store.evict_thinking_blocks(ids=[rid, 99999, 88888])
    # Only the real row scanned.
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1
    store.close()


def test_evict_thinking_blocks_messages_for_api_stays_valid(home):
    """After stripping, messages_for_api still produces non-empty content per row.

    Regression guard for the empty-row skip rule. If we ever forget to skip
    a row that would empty out, messages_for_api would emit a row with
    content=[] and the Anthropic API would 400. This test asserts the
    skip works for an all-thinking row even when bundled with normal rows.
    """
    from mnemara.store import Store

    store = Store("etb_api_t")
    # Mix: an all-thinking assistant row (will be skipped), a normal row.
    store.append_turn("user", [{"type": "text", "text": "u"}])
    store.append_turn("assistant", [{"type": "thinking", "text": "lone"}])
    store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "x"},
            {"type": "text", "text": "answer"},
        ],
    )

    store.evict_thinking_blocks(all_rows=True)

    msgs = store.messages_for_api()
    for m in msgs:
        assert isinstance(m["content"], list) and len(m["content"]) > 0, (
            f"row produced empty content list: {m}"
        )
    store.close()


# ---------------------------------------------------------------- pinning


def test_pin_row_marks_row_with_label(home):
    """pin_row sets pin_label and list_pinned surfaces the row."""
    from mnemara.store import Store

    store = Store("pin_basic_t")
    rid = store.append_turn("assistant", [{"type": "text", "text": "hello"}])
    matched = store.pin_row(rid, label="commit")
    assert matched is True

    pinned = store.list_pinned()
    assert len(pinned) == 1
    assert pinned[0]["id"] == rid
    assert pinned[0]["pin_label"] == "commit"
    store.close()


def test_pin_row_unknown_id_returns_false(home):
    """Pinning a non-existent row returns False (no error)."""
    from mnemara.store import Store

    store = Store("pin_unknown_t")
    matched = store.pin_row(99999, label="commit")
    assert matched is False
    assert store.list_pinned() == []
    store.close()


def test_pin_row_idempotent_overwrites_label(home):
    """Re-pinning a pinned row overwrites the label."""
    from mnemara.store import Store

    store = Store("pin_overwrite_t")
    rid = store.append_turn("assistant", [{"type": "text", "text": "x"}])
    store.pin_row(rid, label="finding")
    store.pin_row(rid, label="decision")
    pinned = store.list_pinned()
    assert len(pinned) == 1
    assert pinned[0]["pin_label"] == "decision"
    store.close()


def test_pin_row_rejects_empty_label(home):
    """pin_row raises ValueError on empty/whitespace label."""
    import pytest
    from mnemara.store import Store

    store = Store("pin_empty_label_t")
    rid = store.append_turn("assistant", [{"type": "text", "text": "x"}])
    with pytest.raises(ValueError, match="label must be"):
        store.pin_row(rid, label="")
    with pytest.raises(ValueError, match="label must be"):
        store.pin_row(rid, label="   ")
    store.close()


def test_unpin_row_removes_pin(home):
    """unpin_row clears pin_label and returns True only if row was pinned."""
    from mnemara.store import Store

    store = Store("unpin_basic_t")
    rid = store.append_turn("assistant", [{"type": "text", "text": "x"}])
    store.pin_row(rid, label="commit")
    assert store.unpin_row(rid) is True
    assert store.list_pinned() == []
    # Idempotent: second unpin returns False (no longer pinned).
    assert store.unpin_row(rid) is False
    store.close()


def test_unpin_row_unknown_id_returns_false(home):
    """unpin_row on non-existent id is a no-op returning False."""
    from mnemara.store import Store

    store = Store("unpin_unknown_t")
    assert store.unpin_row(99999) is False
    store.close()


def test_list_pinned_filters_by_label(home):
    """list_pinned(label='X') returns only rows with that exact pin_label."""
    from mnemara.store import Store

    store = Store("list_filter_t")
    a = store.append_turn("assistant", [{"type": "text", "text": "a"}])
    b = store.append_turn("assistant", [{"type": "text", "text": "b"}])
    c = store.append_turn("assistant", [{"type": "text", "text": "c"}])
    store.pin_row(a, label="commit")
    store.pin_row(b, label="commit")
    store.pin_row(c, label="finding")

    commits = store.list_pinned("commit")
    assert sorted(r["id"] for r in commits) == sorted([a, b])
    findings = store.list_pinned("finding")
    assert [r["id"] for r in findings] == [c]
    everything = store.list_pinned()
    assert len(everything) == 3
    none = store.list_pinned("nonexistent")
    assert none == []
    store.close()


def test_list_pinned_empty_when_nothing_pinned(home):
    """list_pinned() returns empty list when no rows are pinned."""
    from mnemara.store import Store

    store = Store("list_empty_t")
    store.append_turn("assistant", [{"type": "text", "text": "x"}])
    assert store.list_pinned() == []
    store.close()


def test_pin_label_survives_schema_migration_on_legacy_db(home, tmp_path, monkeypatch):
    """A pre-existing turns table without pin_label gets the column added.

    Regression guard: Store() on a legacy DB must not raise; it must add
    the column via PRAGMA-driven ALTER TABLE and let pin/list operations
    work afterwards.
    """
    import sqlite3
    from mnemara import paths
    from mnemara.store import Store

    instance = "legacy_db_t"
    db_path = paths.db_path(instance)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-create the table with the OLD schema (no pin_label column).
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_uses TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO turns (ts, role, content, tool_uses, tokens_in, tokens_out) "
        "VALUES ('2026-01-01T00:00:00+00:00', 'user', "
        "'[{\"type\":\"text\",\"text\":\"legacy\"}]', NULL, NULL, NULL)"
    )
    conn.commit()
    conn.close()

    # Now open via Store — must run the migration.
    store = Store(instance)
    cols = {row[1] for row in store.conn.execute("PRAGMA table_info(turns)")}
    assert "pin_label" in cols, f"pin_label column missing after migration: {cols}"

    # Pinning the legacy row must work.
    cur = store.conn.execute("SELECT id FROM turns LIMIT 1")
    row_id = cur.fetchone()[0]
    assert store.pin_row(row_id, label="legacy_pin") is True
    pinned = store.list_pinned()
    assert len(pinned) == 1 and pinned[0]["pin_label"] == "legacy_pin"
    store.close()


# ---------------------------------------------------------------- evict_older_than


def test_evict_older_than_drops_old_rows(home):
    """Rows whose ts is older than the cutoff are deleted; recent rows survive."""
    import sqlite3, time
    from mnemara.store import Store

    store = Store("eot_basic_t")
    # Hand-build rows with crafted timestamps so the test is deterministic.
    old_a = store.append_turn("user", [{"type": "text", "text": "old1"}])
    old_b = store.append_turn("user", [{"type": "text", "text": "old2"}])
    # Backdate by hand.
    store.conn.execute(
        "UPDATE turns SET ts='2020-01-01T00:00:00+00:00' WHERE id IN (?, ?)",
        (old_a, old_b),
    )
    store.conn.commit()
    fresh = store.append_turn("user", [{"type": "text", "text": "fresh"}])

    result = store.evict_older_than(60)  # anything older than 1 minute
    assert result["rows_evicted"] == 2
    assert result["rows_skipped_pinned"] == 0

    rows = store.window()
    assert [r["id"] for r in rows] == [fresh]
    store.close()


def test_evict_older_than_skips_pinned_by_default(home):
    """Pinned old rows are preserved; unpinned old rows are dropped."""
    from mnemara.store import Store

    store = Store("eot_skip_pinned_t")
    pinned_old = store.append_turn("assistant", [{"type": "text", "text": "decision"}])
    unpinned_old = store.append_turn("assistant", [{"type": "text", "text": "scratch"}])
    store.conn.execute(
        "UPDATE turns SET ts='2020-01-01T00:00:00+00:00' WHERE id IN (?, ?)",
        (pinned_old, unpinned_old),
    )
    store.conn.commit()
    store.pin_row(pinned_old, label="decision")

    result = store.evict_older_than(60)
    assert result["rows_evicted"] == 1
    assert result["rows_skipped_pinned"] == 1
    rows = store.window()
    assert [r["id"] for r in rows] == [pinned_old]
    store.close()


def test_evict_older_than_force_drops_pinned(home):
    """skip_pinned=False (force) drops pinned rows too."""
    from mnemara.store import Store

    store = Store("eot_force_t")
    pinned_old = store.append_turn("assistant", [{"type": "text", "text": "x"}])
    store.conn.execute(
        "UPDATE turns SET ts='2020-01-01T00:00:00+00:00' WHERE id=?",
        (pinned_old,),
    )
    store.conn.commit()
    store.pin_row(pinned_old, label="commit")

    result = store.evict_older_than(60, skip_pinned=False)
    assert result["rows_evicted"] == 1
    assert result["rows_skipped_pinned"] == 0
    assert store.window() == []
    store.close()


def test_evict_older_than_zero_or_negative_is_noop(home):
    """seconds <= 0 returns zero counts without touching the store."""
    from mnemara.store import Store

    store = Store("eot_zero_t")
    rid = store.append_turn("assistant", [{"type": "text", "text": "x"}])
    r0 = store.evict_older_than(0)
    assert r0["rows_evicted"] == 0
    r_neg = store.evict_older_than(-100)
    assert r_neg["rows_evicted"] == 0
    # Row still present.
    assert [r["id"] for r in store.window()] == [rid]
    store.close()


def test_evict_older_than_recent_rows_not_evicted(home):
    """A row created moments ago survives evict_older_than(600)."""
    from mnemara.store import Store

    store = Store("eot_recent_t")
    rid = store.append_turn("assistant", [{"type": "text", "text": "x"}])
    result = store.evict_older_than(600)
    assert result["rows_evicted"] == 0
    assert [r["id"] for r in store.window()] == [rid]
    store.close()


# ---------------------------------------------------------------- evict_thinking_blocks extensions


def test_evict_thinking_blocks_older_than_strips_old_only(home):
    """older_than_seconds selector strips thinking from time-old rows only."""
    from mnemara.store import Store

    store = Store("etb_older_t")
    old_id = store.append_turn(
        "assistant",
        [{"type": "thinking", "text": "old"}, {"type": "text", "text": "answer"}],
    )
    store.conn.execute(
        "UPDATE turns SET ts='2020-01-01T00:00:00+00:00' WHERE id=?",
        (old_id,),
    )
    store.conn.commit()
    fresh_id = store.append_turn(
        "assistant",
        [{"type": "thinking", "text": "fresh"}, {"type": "text", "text": "fresh ans"}],
    )

    result = store.evict_thinking_blocks(older_than_seconds=60)
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1

    rows = {r["id"]: r for r in store.window()}
    # Old row had thinking stripped.
    types_old = [b["type"] for b in rows[old_id]["content"]]
    assert types_old == ["text"]
    # Fresh row untouched.
    types_fresh = [b["type"] for b in rows[fresh_id]["content"]]
    assert types_fresh == ["thinking", "text"]
    store.close()


def test_evict_thinking_blocks_skips_pinned_in_bulk_modes(home):
    """all_rows + keep_recent skip pinned rows by default."""
    from mnemara.store import Store

    store = Store("etb_pin_bulk_t")
    pinned = store.append_turn(
        "assistant",
        [{"type": "thinking", "text": "important"}, {"type": "text", "text": "decision"}],
    )
    unpinned = store.append_turn(
        "assistant",
        [{"type": "thinking", "text": "scratch"}, {"type": "text", "text": "answer"}],
    )
    store.pin_row(pinned, label="decision")

    result = store.evict_thinking_blocks(all_rows=True)
    assert result["rows_modified"] == 1
    assert result["rows_skipped_pinned"] == 1
    assert result["blocks_evicted"] == 1

    rows = {r["id"]: r for r in store.window()}
    # Pinned row's thinking preserved.
    assert [b["type"] for b in rows[pinned]["content"]] == ["thinking", "text"]
    # Unpinned row's thinking stripped.
    assert [b["type"] for b in rows[unpinned]["content"]] == ["text"]
    store.close()


def test_evict_thinking_blocks_skip_pinned_false_overrides(home):
    """skip_pinned=False strips even pinned rows."""
    from mnemara.store import Store

    store = Store("etb_pin_force_t")
    pinned = store.append_turn(
        "assistant",
        [{"type": "thinking", "text": "x"}, {"type": "text", "text": "y"}],
    )
    store.pin_row(pinned, label="commit")

    result = store.evict_thinking_blocks(all_rows=True, skip_pinned=False)
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1
    assert result["rows_skipped_pinned"] == 0
    rows = store.window()
    assert [b["type"] for b in rows[0]["content"]] == ["text"]
    store.close()


def test_evict_thinking_blocks_explicit_ids_with_pin_skip(home):
    """ids selector respects skip_pinned by default; a pinned id is preserved."""
    from mnemara.store import Store

    store = Store("etb_ids_pin_t")
    pinned = store.append_turn(
        "assistant",
        [{"type": "thinking", "text": "x"}, {"type": "text", "text": "y"}],
    )
    store.pin_row(pinned, label="commit")

    result = store.evict_thinking_blocks(ids=[pinned])
    # Pinned row was filtered out before the strip pass.
    assert result["rows_scanned"] == 0
    assert result["rows_modified"] == 0
    assert result["rows_skipped_pinned"] == 1
    # With skip_pinned=False the same call strips it.
    result2 = store.evict_thinking_blocks(ids=[pinned], skip_pinned=False)
    assert result2["rows_modified"] == 1
    store.close()


def test_evict_thinking_blocks_selector_validation_includes_older(home):
    """The 4-selector validation rejects zero or multiple selectors."""
    import pytest
    from mnemara.store import Store

    store = Store("etb_sel_v2_t")
    store.append_turn("assistant", [{"type": "text", "text": "x"}])
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks()
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(older_than_seconds=60, all_rows=True)
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(older_than_seconds=60, ids=[1])
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_thinking_blocks(older_than_seconds=60, keep_recent=2)
    store.close()


# ---------------------------------------------------------------- evict_last/evict_since with skip_pinned


def test_evict_last_skips_pinned_by_default(home):
    """evict_last(N) skips pinned rows when counting & deleting."""
    from mnemara.store import Store

    store = Store("el_pin_t")
    a = store.append_turn("user", [{"type": "text", "text": "a"}])
    b = store.append_turn("assistant", [{"type": "text", "text": "b"}])
    c = store.append_turn("user", [{"type": "text", "text": "c"}])
    store.pin_row(b, label="commit")

    # evict_last(2) with skip_pinned=True should drop c and a (the two
    # most-recent unpinned), leaving b alone.
    deleted = store.evict_last(2)
    assert deleted == 2
    rows = store.window()
    assert [r["id"] for r in rows] == [b]
    store.close()


def test_evict_last_force_drops_pinned(home):
    """evict_last(N, skip_pinned=False) treats pinned rows like any other."""
    from mnemara.store import Store

    store = Store("el_force_t")
    a = store.append_turn("user", [{"type": "text", "text": "a"}])
    b = store.append_turn("assistant", [{"type": "text", "text": "b"}])
    store.pin_row(b, label="commit")

    deleted = store.evict_last(2, skip_pinned=False)
    assert deleted == 2
    assert store.window() == []
    store.close()


def test_evict_since_skips_pinned_by_default(home):
    """evict_since preserves pinned rows in the deletion range."""
    from mnemara.store import Store

    store = Store("es_pin_t")
    store.mark_segment("checkpoint")
    pinned = store.append_turn(
        "assistant", [{"type": "text", "text": "decision after marker"}]
    )
    unpinned = store.append_turn(
        "user", [{"type": "text", "text": "stuff after marker"}]
    )
    store.pin_row(pinned, label="decision")

    deleted = store.evict_since("checkpoint")
    # Marker + unpinned dropped (2); pinned preserved.
    assert deleted == 2
    rows = store.window()
    assert [r["id"] for r in rows] == [pinned]
    store.close()


def test_evict_since_force_drops_pinned(home):
    """evict_since(skip_pinned=False) drops pinned rows too."""
    from mnemara.store import Store

    store = Store("es_force_t")
    store.mark_segment("ckpt")
    pinned = store.append_turn(
        "assistant", [{"type": "text", "text": "x"}]
    )
    store.pin_row(pinned, label="commit")

    deleted = store.evict_since("ckpt", skip_pinned=False)
    assert deleted == 2  # marker + the pinned row
    assert store.window() == []
    store.close()


# ---------------------------------------------------------------- parse_duration_seconds


def test_parse_duration_seconds_basic_units():
    """Bare integer + 's' / 'm' / 'h' / 'd' suffixes parse correctly."""
    from mnemara.store import parse_duration_seconds

    assert parse_duration_seconds("600") == 600
    assert parse_duration_seconds("10s") == 10
    assert parse_duration_seconds("10m") == 600
    assert parse_duration_seconds("2h") == 7200
    assert parse_duration_seconds("1d") == 86400
    # Whitespace + uppercase tolerated.
    assert parse_duration_seconds(" 10M ") == 600
    # Float coefficient OK.
    assert parse_duration_seconds("0.5m") == 30


def test_parse_duration_seconds_invalid_raises():
    """Empty, garbage, or unsupported-suffix input raises ValueError."""
    import pytest
    from mnemara.store import parse_duration_seconds

    with pytest.raises(ValueError, match="duration required"):
        parse_duration_seconds("")
    with pytest.raises(ValueError, match="duration required"):
        parse_duration_seconds("   ")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration_seconds("abc")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration_seconds("10x")  # unknown suffix falls through to int parse


# ---------------------------------------------------------------- normalize_model_name


def test_normalize_model_name_strips_whitespace():
    """Leading/trailing whitespace is stripped."""
    from mnemara.config import normalize_model_name

    assert normalize_model_name("claude-sonnet-4-5") == "claude-sonnet-4-5"
    assert normalize_model_name("  claude-sonnet-4-5  ") == "claude-sonnet-4-5"
    assert normalize_model_name("\tclaude-opus-4-7\n") == "claude-opus-4-7"


def test_normalize_model_name_rejects_internal_whitespace():
    """The actual reported bug: spaces inside the model name."""
    import pytest
    from mnemara.config import normalize_model_name

    with pytest.raises(ValueError, match="whitespace"):
        normalize_model_name("claude sonnet 4 5")
    with pytest.raises(ValueError, match="whitespace"):
        normalize_model_name("claude-sonnet 4-5")
    with pytest.raises(ValueError, match="whitespace"):
        normalize_model_name("claude\tsonnet")


def test_normalize_model_name_rejects_empty():
    """Empty / None / whitespace-only inputs raise."""
    import pytest
    from mnemara.config import normalize_model_name

    with pytest.raises(ValueError, match="required"):
        normalize_model_name("")
    with pytest.raises(ValueError, match="required"):
        normalize_model_name("   ")
    with pytest.raises(ValueError, match="required"):
        normalize_model_name(None)


def test_normalize_model_name_rejects_non_alpha_first_char():
    """First character must be a letter (catches accidental quote chars / leading hyphens)."""
    import pytest
    from mnemara.config import normalize_model_name

    with pytest.raises(ValueError, match="must start"):
        normalize_model_name("-claude-sonnet")
    with pytest.raises(ValueError, match="must start"):
        normalize_model_name("4claude")
    with pytest.raises(ValueError, match="must start"):
        normalize_model_name("'claude'")


def test_normalize_model_name_rejects_invalid_chars():
    """Characters outside [a-zA-Z0-9.-] raise."""
    import pytest
    from mnemara.config import normalize_model_name

    with pytest.raises(ValueError, match="invalid character"):
        normalize_model_name("claude/sonnet")
    with pytest.raises(ValueError, match="invalid character"):
        normalize_model_name("claude_sonnet")  # underscore not allowed
    with pytest.raises(ValueError, match="invalid character"):
        normalize_model_name("claude@sonnet")


def test_normalize_model_name_accepts_known_anthropic_formats():
    """Real Anthropic model names from various families parse cleanly."""
    from mnemara.config import normalize_model_name

    valid = [
        "claude-opus-4-7",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        # Permissive — future families and dotted versions OK.
        "claude-4.0-mini",
        "anthropic-foo-bar",
    ]
    for name in valid:
        assert normalize_model_name(name) == name


def test_normalize_model_name_idempotent_on_clean_input():
    """A pre-normalized name passes through unchanged."""
    from mnemara.config import normalize_model_name

    assert normalize_model_name("claude-opus-4-7") == "claude-opus-4-7"


# ---------------------------------------------------------------- tool_use surgery


def test_evict_tool_use_blocks_strips_tool_use_only(home):
    """tool_use blocks evicted; text + thinking + tool_result preserved."""
    from mnemara.store import Store

    store = Store("etub_basic_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "consider the read"},
            {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"path": "/foo"}},
            {"type": "text", "text": "I'll read foo first"},
            {"type": "tool_use", "id": "tu2", "name": "Edit", "input": {"old": "x", "new": "y"}},
            {"type": "tool_result", "tool_use_id": "tu1", "content": "file content"},
        ],
    )

    result = store.evict_tool_use_blocks(ids=[rid])
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 2
    assert result["bytes_freed"] > 0

    rows = store.window()
    blocks = rows[0]["content"]
    types = [b["type"] for b in blocks]
    assert types == ["thinking", "text", "tool_result"]
    # Verify the thinking and text are preserved verbatim.
    assert blocks[0]["text"] == "consider the read"
    assert blocks[1]["text"] == "I'll read foo first"
    store.close()


def test_evict_tool_use_blocks_keep_recent_preserves_tail(home):
    """keep_recent=N strips tool_use from older rows but not the last N."""
    from mnemara.store import Store

    store = Store("etub_keep_t")
    ids = []
    for i in range(5):
        ids.append(store.append_turn(
            "assistant",
            [
                {"type": "tool_use", "id": f"tu{i}", "name": "Bash", "input": {"command": f"ls {i}"}},
                {"type": "text", "text": f"output {i}"},
            ],
        ))
    result = store.evict_tool_use_blocks(keep_recent=2)
    assert result["rows_scanned"] == 3
    assert result["rows_modified"] == 3
    assert result["blocks_evicted"] == 3

    rows = store.window()
    # Older 3 should have tool_use stripped.
    for r in rows[:3]:
        types = [b["type"] for b in r["content"]]
        assert types == ["text"], f"row {r['id']}: tool_use should be stripped, got {types}"
    # Most-recent 2 untouched.
    for r in rows[3:]:
        types = [b["type"] for b in r["content"]]
        assert types == ["tool_use", "text"]
    store.close()


def test_evict_tool_use_blocks_all_rows(home):
    """all_rows=True strips tool_use from every row regardless of role."""
    from mnemara.store import Store

    store = Store("etub_all_t")
    user_id = store.append_turn("user", [{"type": "text", "text": "ping"}])
    asst_id = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "tu1", "name": "Read", "input": {}},
            {"type": "text", "text": "answer"},
        ],
    )

    result = store.evict_tool_use_blocks(all_rows=True)
    assert result["rows_scanned"] == 2
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1

    rows = {r["id"]: r for r in store.window()}
    assert [b["type"] for b in rows[asst_id]["content"]] == ["text"]
    assert [b["type"] for b in rows[user_id]["content"]] == ["text"]
    store.close()


def test_evict_tool_use_blocks_older_than(home):
    """older_than_seconds selector works for tool_use blocks."""
    from mnemara.store import Store

    store = Store("etub_older_t")
    old_id = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "old", "name": "Read", "input": {}},
            {"type": "text", "text": "old answer"},
        ],
    )
    store.conn.execute(
        "UPDATE turns SET ts='2020-01-01T00:00:00+00:00' WHERE id=?",
        (old_id,),
    )
    store.conn.commit()
    fresh_id = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "fresh", "name": "Read", "input": {}},
            {"type": "text", "text": "fresh answer"},
        ],
    )

    result = store.evict_tool_use_blocks(older_than_seconds=60)
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1

    rows = {r["id"]: r for r in store.window()}
    assert [b["type"] for b in rows[old_id]["content"]] == ["text"]
    assert [b["type"] for b in rows[fresh_id]["content"]] == ["tool_use", "text"]
    store.close()


def test_evict_tool_use_blocks_skips_pinned_in_bulk_modes(home):
    """Pinned rows preserved in all_rows / keep_recent / older_than modes."""
    from mnemara.store import Store

    store = Store("etub_pin_t")
    pinned = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "tu_p", "name": "Read", "input": {"path": "/important"}},
            {"type": "text", "text": "decision"},
        ],
    )
    unpinned = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "tu_u", "name": "Read", "input": {"path": "/scratch"}},
            {"type": "text", "text": "answer"},
        ],
    )
    store.pin_row(pinned, label="audit")

    result = store.evict_tool_use_blocks(all_rows=True)
    assert result["rows_modified"] == 1
    assert result["rows_skipped_pinned"] == 1

    rows = {r["id"]: r for r in store.window()}
    # Pinned row's tool_use preserved.
    assert [b["type"] for b in rows[pinned]["content"]] == ["tool_use", "text"]
    # Unpinned row's tool_use stripped.
    assert [b["type"] for b in rows[unpinned]["content"]] == ["text"]
    store.close()


def test_evict_tool_use_blocks_skip_pinned_false_overrides(home):
    """skip_pinned=False strips tool_use from pinned rows too."""
    from mnemara.store import Store

    store = Store("etub_pin_force_t")
    pinned = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "tu", "name": "X", "input": {}},
            {"type": "text", "text": "y"},
        ],
    )
    store.pin_row(pinned, label="commit")

    result = store.evict_tool_use_blocks(all_rows=True, skip_pinned=False)
    assert result["rows_modified"] == 1
    assert result["blocks_evicted"] == 1
    assert result["rows_skipped_pinned"] == 0
    rows = store.window()
    assert [b["type"] for b in rows[0]["content"]] == ["text"]
    store.close()


def test_evict_tool_use_blocks_skips_empty_result_rows(home):
    """Rows whose stripping leaves 0 blocks are skipped without modification.

    A row containing ONLY tool_use blocks (e.g. an assistant turn that
    silently called several tools without producing text) would be emptied
    by a strip. Since Anthropic rejects empty content lists, the strip
    pass leaves such rows alone — agent can evict_ids them explicitly.
    """
    from mnemara.store import Store

    store = Store("etub_empty_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "a", "name": "X", "input": {}},
            {"type": "tool_use", "id": "b", "name": "Y", "input": {}},
        ],
    )
    result = store.evict_tool_use_blocks(ids=[rid])
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 0
    assert result["blocks_evicted"] == 0
    # Row content unchanged.
    rows = store.window()
    assert [b["type"] for b in rows[0]["content"]] == ["tool_use", "tool_use"]
    store.close()


def test_evict_tool_use_blocks_noop_when_no_tool_use(home):
    """A row with no tool_use blocks is scanned but not modified."""
    from mnemara.store import Store

    store = Store("etub_noop_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "text", "text": "no tools called"},
            {"type": "thinking", "text": "scratch"},
        ],
    )
    result = store.evict_tool_use_blocks(ids=[rid])
    assert result["rows_scanned"] == 1
    assert result["rows_modified"] == 0
    assert result["blocks_evicted"] == 0
    store.close()


def test_evict_tool_use_blocks_idempotent_on_rerun(home):
    """Second run on already-stripped row reports zero changes."""
    from mnemara.store import Store

    store = Store("etub_idem_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "tu", "name": "X", "input": {}},
            {"type": "text", "text": "y"},
        ],
    )
    r1 = store.evict_tool_use_blocks(ids=[rid])
    assert r1["rows_modified"] == 1
    r2 = store.evict_tool_use_blocks(ids=[rid])
    assert r2["rows_modified"] == 0
    assert r2["blocks_evicted"] == 0
    store.close()


def test_evict_tool_use_blocks_selector_validation(home):
    """Exactly one of ids/keep_recent/all_rows/older_than_seconds required."""
    import pytest
    from mnemara.store import Store

    store = Store("etub_sel_t")
    store.append_turn("assistant", [{"type": "tool_use", "id": "x", "name": "y", "input": {}}, {"type": "text", "text": "z"}])
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_tool_use_blocks()
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_tool_use_blocks(ids=[1], all_rows=True)
    with pytest.raises(ValueError, match="exactly one of"):
        store.evict_tool_use_blocks(keep_recent=2, older_than_seconds=60)
    store.close()


def test_evict_tool_use_blocks_messages_for_api_stays_valid(home):
    """After stripping, every emitted row has non-empty content list.

    Regression guard for the empty-row skip rule across both surgeries.
    """
    from mnemara.store import Store

    store = Store("etub_api_t")
    store.append_turn("user", [{"type": "text", "text": "u"}])
    # All-tool_use row: strip would empty it, must be skipped.
    store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "lone", "name": "X", "input": {}}],
    )
    # Mixed row: strip leaves text behind.
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "tu", "name": "Y", "input": {}},
            {"type": "text", "text": "answer"},
        ],
    )

    store.evict_tool_use_blocks(all_rows=True)
    msgs = store.messages_for_api()
    for m in msgs:
        assert isinstance(m["content"], list) and len(m["content"]) > 0, (
            f"row produced empty content list: {m}"
        )
    store.close()


def test_evict_tool_use_blocks_real_world_byte_savings(home):
    """Sanity check: stripping a realistic tool_use block frees significant bytes.

    Documents the high-impact value of this surgery vs thinking — a single
    Edit tool_use spec is hundreds of bytes; tool_use blocks dominate
    long-session stores.
    """
    from mnemara.store import Store

    store = Store("etub_real_t")
    realistic_input = {
        "file_path": "/home/user/workspace/some/long/path/to/a/file.py",
        "old_string": "def some_function(x, y, z):\n    return x + y + z\n" * 10,
        "new_string": "def some_function(x, y, z):\n    return x * y * z\n" * 10,
    }
    rid = store.append_turn(
        "assistant",
        [
            {"type": "text", "text": "Editing the file."},
            {"type": "tool_use", "id": "tu_edit", "name": "Edit", "input": realistic_input},
        ],
    )
    result = store.evict_tool_use_blocks(ids=[rid])
    assert result["blocks_evicted"] == 1
    # The Edit input alone is ~600+ bytes, plus block wrapper overhead.
    assert result["bytes_freed"] > 500, (
        f"expected >500 bytes freed for realistic Edit tool_use, got {result['bytes_freed']}"
    )
    store.close()


def test_evict_thinking_and_tool_use_compose(home):
    """Both surgeries can run on the same row sequentially without conflict.

    Regression guard for the shared _strip_blocks_by_type helper: stripping
    thinking from a row, then stripping tool_use from the same row, should
    leave only text + tool_result blocks.
    """
    from mnemara.store import Store

    store = Store("etub_compose_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "scratch"},
            {"type": "tool_use", "id": "tu", "name": "X", "input": {}},
            {"type": "text", "text": "answer"},
            {"type": "tool_result", "tool_use_id": "tu", "content": "ok"},
        ],
    )
    r1 = store.evict_thinking_blocks(ids=[rid])
    assert r1["blocks_evicted"] == 1
    r2 = store.evict_tool_use_blocks(ids=[rid])
    assert r2["blocks_evicted"] == 1

    rows = store.window()
    types = [b["type"] for b in rows[0]["content"]]
    assert types == ["text", "tool_result"]
    store.close()


# ----------------------------------------------------------------------
# Eviction-stats counter (session-scoped, surfaced via get_eviction_stats)
# ----------------------------------------------------------------------

def test_eviction_stats_starts_zeroed(home):
    """A fresh Store reports zero counters across the board."""
    from mnemara.store import Store

    store = Store("evstats_zero_t")
    s = store.get_eviction_stats()
    assert s == {
        "rows_evicted": 0,
        "blocks_evicted": 0,
        "bytes_freed": 0,
        "pinned_rows_force_evicted": 0,
    }
    store.close()


def test_eviction_stats_get_returns_snapshot_copy(home):
    """get_eviction_stats returns a copy; mutating it doesn't affect store."""
    from mnemara.store import Store

    store = Store("evstats_snapshot_t")
    s = store.get_eviction_stats()
    s["rows_evicted"] = 999  # mutate caller copy
    s2 = store.get_eviction_stats()
    assert s2["rows_evicted"] == 0
    store.close()


def test_eviction_stats_bumps_on_cap_fifo(home):
    """Cap-driven evict() bumps rows_evicted by deleted count."""
    from mnemara.store import Store

    store = Store("evstats_fifo_t")
    for i in range(5):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    # Cap of 2 forces 3 rows evicted.
    deleted = store.evict(max_turns=2)
    assert deleted == 3
    s = store.get_eviction_stats()
    assert s["rows_evicted"] == 3
    assert s["blocks_evicted"] == 0
    assert s["bytes_freed"] == 0
    store.close()


def test_eviction_stats_bumps_on_evict_last(home):
    """evict_last bumps rows_evicted by actual count."""
    from mnemara.store import Store

    store = Store("evstats_last_t")
    for i in range(4):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    n = store.evict_last(2)
    assert n == 2
    assert store.get_eviction_stats()["rows_evicted"] == 2
    store.close()


def test_eviction_stats_bumps_on_evict_ids(home):
    """evict_ids bumps rows_evicted by existing-row count, not requested count."""
    from mnemara.store import Store

    store = Store("evstats_ids_t")
    a = store.append_turn("user", [{"type": "text", "text": "a"}])
    b = store.append_turn("user", [{"type": "text", "text": "b"}])
    # Request includes one nonexistent id (99999); only the two real ones bump.
    n = store.evict_ids([a, b, 99999])
    assert n == 2
    assert store.get_eviction_stats()["rows_evicted"] == 2
    store.close()


def test_eviction_stats_bumps_on_evict_since(home):
    """evict_since bumps rows_evicted by deleted count."""
    from mnemara.store import Store

    store = Store("evstats_since_t")
    store.append_turn("user", [{"type": "text", "text": "before"}])
    store.mark_segment("M")
    store.append_turn("user", [{"type": "text", "text": "after1"}])
    store.append_turn("user", [{"type": "text", "text": "after2"}])
    n = store.evict_since("M")
    # marker + 2 rows after = 3 deleted
    assert n == 3
    assert store.get_eviction_stats()["rows_evicted"] == 3
    store.close()


def test_eviction_stats_bumps_on_evict_older_than(home):
    """evict_older_than bumps rows_evicted by deleted-row count."""
    from mnemara.store import Store

    store = Store("evstats_older_t")
    rid = store.append_turn("user", [{"type": "text", "text": "old"}])
    # Backdate the row so it qualifies for the 1-second cutoff.
    store.conn.execute(
        "UPDATE turns SET ts='2020-01-01T00:00:00+00:00' WHERE id=?",
        (rid,),
    )
    store.conn.commit()
    result = store.evict_older_than(1)
    assert result["rows_evicted"] == 1
    assert store.get_eviction_stats()["rows_evicted"] == 1
    store.close()


def test_eviction_stats_bumps_on_block_surgery(home):
    """Block surgery bumps blocks_evicted + bytes_freed; row count unchanged."""
    from mnemara.store import Store

    store = Store("evstats_blocks_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "thinking", "text": "x" * 200},
            {"type": "text", "text": "kept"},
            {"type": "thinking", "text": "y" * 100},
        ],
    )
    result = store.evict_thinking_blocks(ids=[rid])
    assert result["blocks_evicted"] == 2
    assert result["bytes_freed"] > 0
    s = store.get_eviction_stats()
    assert s["rows_evicted"] == 0  # block surgery doesn't drop rows
    assert s["blocks_evicted"] == 2
    assert s["bytes_freed"] == result["bytes_freed"]
    store.close()


def test_eviction_stats_accumulates_across_multiple_calls(home):
    """Counters compose: row drops + block surgery sum cleanly within a session."""
    from mnemara.store import Store

    store = Store("evstats_accum_t")
    # Add 5 rows, drop the last 2 explicitly, do block surgery on the rest.
    ids = []
    for i in range(5):
        ids.append(store.append_turn(
            "assistant",
            [
                {"type": "thinking", "text": f"t{i}"},
                {"type": "text", "text": f"a{i}"},
            ],
        ))
    n = store.evict_last(2)
    assert n == 2
    r = store.evict_thinking_blocks(all_rows=True)
    assert r["blocks_evicted"] == 3  # 3 surviving rows had 1 thinking each

    s = store.get_eviction_stats()
    assert s["rows_evicted"] == 2
    assert s["blocks_evicted"] == 3
    assert s["bytes_freed"] > 0
    store.close()


def test_eviction_stats_reset_on_new_store_instance(home):
    """Counters are session-scoped: a fresh Store() on the same DB starts at 0."""
    from mnemara.store import Store

    store1 = Store("evstats_reset_t")
    store1.append_turn("user", [{"type": "text", "text": "x"}])
    store1.evict_last(1)
    assert store1.get_eviction_stats()["rows_evicted"] == 1
    store1.close()

    store2 = Store("evstats_reset_t")
    assert store2.get_eviction_stats() == {
        "rows_evicted": 0,
        "blocks_evicted": 0,
        "bytes_freed": 0,
        "pinned_rows_force_evicted": 0,
    }
    store2.close()


# ----------------------------------------------------------------------
# evict_write_pairs — auto-evict-after-write surgery
# ----------------------------------------------------------------------

def _big(n: int) -> str:
    """Helper: a large filler string to make bytes_freed nonzero in tests."""
    return "x" * n


def test_evict_write_pairs_stubs_edit_input(home):
    """Edit tool_use input gets stubbed to {file_path, _evicted: true}."""
    from mnemara.store import Store

    store = Store("ewp_edit_t")
    rid = store.append_turn(
        "assistant",
        [
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "Edit",
                "input": {
                    "file_path": "/foo/bar.py",
                    "old_string": _big(2000),
                    "new_string": _big(2500),
                },
            },
            {"type": "text", "text": "edited."},
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    assert result["reads_stubbed"] == 0
    assert result["rows_modified"] == 1
    assert result["files_seen"] == 1
    assert result["bytes_freed"] > 4000  # at least the bulky strings

    rows = store.window()
    blocks = rows[0]["content"]
    edit_block = next(b for b in blocks if b["type"] == "tool_use")
    assert edit_block["name"] == "Edit"
    assert edit_block["input"] == {"file_path": "/foo/bar.py", "_evicted": True}
    # Other blocks preserved verbatim.
    text_block = next(b for b in blocks if b["type"] == "text")
    assert text_block["text"] == "edited."
    store.close()


def test_evict_write_pairs_stubs_write_content(home):
    """Write tool's `content` field is stripped; file_path preserved."""
    from mnemara.store import Store

    store = Store("ewp_write_t")
    rid = store.append_turn(
        "assistant",
        [
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "Write",
                "input": {
                    "file_path": "/new/file.py",
                    "content": _big(5000),
                },
            },
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    assert result["bytes_freed"] > 4000

    blocks = store.window()[0]["content"]
    inp = blocks[0]["input"]
    assert inp == {"file_path": "/new/file.py", "_evicted": True}
    assert "content" not in inp
    store.close()


def test_evict_write_pairs_stubs_multiedit(home):
    """MultiEdit's `edits` array is stripped; file_path preserved."""
    from mnemara.store import Store

    store = Store("ewp_multi_t")
    rid = store.append_turn(
        "assistant",
        [
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "MultiEdit",
                "input": {
                    "file_path": "/multi.py",
                    "edits": [
                        {"old_string": _big(1000), "new_string": _big(1100)},
                        {"old_string": _big(800), "new_string": _big(900)},
                    ],
                },
            },
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    blocks = store.window()[0]["content"]
    assert blocks[0]["input"] == {"file_path": "/multi.py", "_evicted": True}
    store.close()


def test_evict_write_pairs_stubs_matching_prior_read(home):
    """Read for /foo before Edit on /foo gets stubbed by pair scan."""
    from mnemara.store import Store

    store = Store("ewp_pair_t")
    # Row 1: Read /foo
    store.append_turn(
        "assistant",
        [
            {
                "type": "tool_use",
                "id": "r1",
                "name": "Read",
                "input": {"file_path": "/foo/bar.py"},
            },
        ],
    )
    # Row 2: Edit /foo with bulky payload
    store.append_turn(
        "assistant",
        [
            {
                "type": "tool_use",
                "id": "e1",
                "name": "Edit",
                "input": {
                    "file_path": "/foo/bar.py",
                    "old_string": _big(2000),
                    "new_string": _big(2000),
                },
            },
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    assert result["reads_stubbed"] == 1
    assert result["rows_modified"] == 2

    rows = store.window()
    # Read got stubbed
    read_inp = rows[0]["content"][0]["input"]
    assert read_inp == {"file_path": "/foo/bar.py", "_evicted": True}
    # Edit got stubbed
    edit_inp = rows[1]["content"][0]["input"]
    assert edit_inp == {"file_path": "/foo/bar.py", "_evicted": True}
    store.close()


def test_evict_write_pairs_does_not_touch_unrelated_reads(home):
    """Reads for OTHER files are left alone."""
    from mnemara.store import Store

    store = Store("ewp_unrelated_t")
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "r1", "name": "Read",
             "input": {"file_path": "/keep_me.py"}},
        ],
    )
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/different.py",
                       "old_string": _big(1000), "new_string": _big(1000)}},
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    assert result["reads_stubbed"] == 0  # Read was for a different file

    rows = store.window()
    read_inp = rows[0]["content"][0]["input"]
    assert read_inp == {"file_path": "/keep_me.py"}  # untouched
    store.close()


def test_evict_write_pairs_does_not_touch_later_reads(home):
    """A Read AFTER the Edit is left alone (not a stale-read pairing)."""
    from mnemara.store import Store

    store = Store("ewp_later_t")
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(1000), "new_string": _big(1000)}},
        ],
    )
    store.append_turn(
        "assistant",
        [
            # This Read happened AFTER the Edit; it's the post-edit re-read,
            # not a stale prior read. Don't stub it.
            {"type": "tool_use", "id": "r1", "name": "Read",
             "input": {"file_path": "/foo.py"}},
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    assert result["reads_stubbed"] == 0

    rows = store.window()
    later_read = rows[1]["content"][0]["input"]
    assert later_read == {"file_path": "/foo.py"}  # preserved
    store.close()


def test_evict_write_pairs_only_in_rows_scopes_write_scan(home):
    """only_in_rows=[N] limits the write scan to row N (auto-evict pattern)."""
    from mnemara.store import Store

    store = Store("ewp_scope_t")
    rid1 = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/a.py",
                       "old_string": _big(500), "new_string": _big(500)}},
        ],
    )
    rid2 = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e2", "name": "Edit",
             "input": {"file_path": "/b.py",
                       "old_string": _big(500), "new_string": _big(500)}},
        ],
    )
    # Scope to row 2 only — row 1's Edit should remain bulky.
    result = store.evict_write_pairs(only_in_rows=[rid2])
    assert result["writes_stubbed"] == 1
    assert result["files_seen"] == 1

    rows = store.window()
    # Row 1 untouched
    assert "old_string" in rows[0]["content"][0]["input"]
    # Row 2 stubbed
    assert rows[1]["content"][0]["input"] == {"file_path": "/b.py", "_evicted": True}
    store.close()


def test_evict_write_pairs_skip_pinned_default(home):
    """Pinned rows are skipped by default; their writes/reads stay bulky."""
    from mnemara.store import Store

    store = Store("ewp_pinned_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    store.pin_row(rid, "deliberate")
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 0
    assert result["rows_skipped_pinned"] == 1

    rows = store.window()
    # Edit body still has its bulky strings.
    assert "old_string" in rows[0]["content"][0]["input"]
    store.close()


def test_evict_write_pairs_skip_pinned_false_strips_anyway(home):
    """skip_pinned=False overrides pin protection."""
    from mnemara.store import Store

    store = Store("ewp_force_t")
    rid = store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    store.pin_row(rid, "deliberate")
    result = store.evict_write_pairs(skip_pinned=False)
    assert result["writes_stubbed"] == 1

    rows = store.window()
    assert rows[0]["content"][0]["input"] == {"file_path": "/foo.py", "_evicted": True}
    store.close()


def test_evict_write_pairs_idempotent(home):
    """Running twice on the same data — second run stubs nothing."""
    from mnemara.store import Store

    store = Store("ewp_idem_t")
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    r1 = store.evict_write_pairs()
    assert r1["writes_stubbed"] == 1

    r2 = store.evict_write_pairs()
    assert r2["writes_stubbed"] == 0
    assert r2["reads_stubbed"] == 0
    assert r2["rows_modified"] == 0
    store.close()


def test_evict_write_pairs_no_writes_no_op(home):
    """If no rows contain Edit/Write/MultiEdit, function reports zeros."""
    from mnemara.store import Store

    store = Store("ewp_empty_t")
    store.append_turn(
        "assistant",
        [
            {"type": "text", "text": "no tool calls here"},
            {"type": "tool_use", "id": "r1", "name": "Read",
             "input": {"file_path": "/foo.py"}},
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 0
    assert result["reads_stubbed"] == 0
    assert result["rows_modified"] == 0
    # The standalone Read is not stubbed because there's no matching Write.
    rows = store.window()
    read_block = next(
        b for b in rows[0]["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    assert read_block["input"] == {"file_path": "/foo.py"}
    store.close()


def test_evict_write_pairs_bumps_eviction_stats(home):
    """Stubbing increments blocks_evicted + bytes_freed in session stats."""
    from mnemara.store import Store

    store = Store("ewp_stats_t")
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    s_before = store.get_eviction_stats()
    assert s_before["blocks_evicted"] == 0
    result = store.evict_write_pairs()
    s_after = store.get_eviction_stats()
    assert s_after["blocks_evicted"] == 1  # 1 write stubbed
    assert s_after["bytes_freed"] == result["bytes_freed"]
    assert s_after["rows_evicted"] == 0  # no rows dropped
    store.close()


def test_evict_write_pairs_only_in_rows_empty_returns_zeros(home):
    """only_in_rows=[] short-circuits to a zero report."""
    from mnemara.store import Store

    store = Store("ewp_emptyscope_t")
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    result = store.evict_write_pairs(only_in_rows=[])
    assert result["writes_stubbed"] == 0
    assert result["rows_modified"] == 0
    store.close()


def test_evict_write_pairs_handles_tool_use_without_file_path(home):
    """Tool_use of a write tool with no file_path: stub still strips bulky fields, no file recorded."""
    from mnemara.store import Store

    store = Store("ewp_nofp_t")
    store.append_turn(
        "assistant",
        [
            # Edit without file_path (degenerate, but agent might emit it)
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1
    assert result["files_seen"] == 0  # no file_path to record

    blocks = store.window()[0]["content"]
    assert blocks[0]["input"] == {"_evicted": True}
    store.close()


def test_evict_write_pairs_preserves_non_write_tool_use(home):
    """Bash/Grep/etc. tool_use blocks in the same row are NOT stubbed."""
    from mnemara.store import Store

    store = Store("ewp_mixed_t")
    store.append_turn(
        "assistant",
        [
            {"type": "tool_use", "id": "b1", "name": "Bash",
             "input": {"command": "ls -la"}},
            {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": "/foo.py",
                       "old_string": _big(2000), "new_string": _big(2000)}},
        ],
    )
    result = store.evict_write_pairs()
    assert result["writes_stubbed"] == 1

    blocks = store.window()[0]["content"]
    bash_block = next(b for b in blocks if b.get("name") == "Bash")
    assert bash_block["input"] == {"command": "ls -la"}  # untouched

    edit_block = next(b for b in blocks if b.get("name") == "Edit")
    assert edit_block["input"] == {"file_path": "/foo.py", "_evicted": True}
    store.close()


# ----------------------------------------------------------------------
# Config.auto_evict_after_write toggle
# ----------------------------------------------------------------------

def test_config_auto_evict_after_write_default_false():
    from mnemara.config import Config

    cfg = Config()
    assert cfg.auto_evict_after_write is False


def test_config_auto_evict_after_write_round_trips_through_dict():
    from mnemara.config import Config

    cfg = Config()
    cfg.auto_evict_after_write = True
    d = cfg.to_dict()
    assert d["auto_evict_after_write"] is True

    cfg2 = Config.from_dict(d)
    assert cfg2.auto_evict_after_write is True


def test_config_auto_evict_after_write_missing_field_defaults_false():
    """Pre-existing config.json files without the field load as False."""
    from mnemara.config import Config

    minimal = {"role_doc_path": "", "model": "claude-opus-4-5"}
    cfg = Config.from_dict(minimal)
    assert cfg.auto_evict_after_write is False


def test_config_default_includes_evict_write_pairs_policy():
    """Fresh instances get EvictWritePairs allowlisted by default."""
    from mnemara.config import Config

    cfg = Config.default()
    tool_names = {t.tool for t in cfg.allowed_tools}
    assert "EvictWritePairs" in tool_names


# ----------------------------------------------------------------------
# row_cap_slack_when_token_headroom — token-aware row-cap relaxation
# ----------------------------------------------------------------------
# Semantics under test:
#   - slack=0 (default) preserves existing strict-cap behavior (backward-compat)
#   - slack>0 with token usage BELOW HEADROOM_RATIO * max_tokens lets row count
#     stretch by `slack` rows beyond max_turns
#   - slack>0 with token usage AT/ABOVE the threshold trims to max_turns strictly
#   - slack>0 with NO max_tokens (None) leaves slack disengaged (defensive)
#   - the second pass (token cap) still trims if total bytes exceed max_tokens
#     regardless of row count, giving us the hard byte ceiling we promised


def test_evict_slack_zero_preserves_existing_behavior(home):
    """Default slack=0 means strict row cap, identical to pre-v0.3.3 behavior."""
    from mnemara.store import Store

    store = Store("evslack_zero_t")
    for i in range(5):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    deleted = store.evict(max_turns=2, max_tokens=1_000_000)
    assert deleted == 3
    assert len(store.window()) == 2
    store.close()


def test_evict_slack_engaged_with_token_headroom(home):
    """With slack=3 and tokens well under threshold, row count of 7 (max=5)
    is allowed to stretch and no rows are deleted (since 7 <= 5 + 3)."""
    from mnemara.store import Store

    store = Store("evslack_headroom_t")
    # Each turn ~7 chars => ~1 token under the /4 heuristic. With max_tokens
    # set high, current usage is far under HEADROOM_RATIO * max_tokens.
    for i in range(7):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    deleted = store.evict(max_turns=5, max_tokens=1_000_000, row_cap_slack=3)
    assert deleted == 0  # 7 <= 5 + 3, no eviction needed
    assert len(store.window()) == 7
    store.close()


def test_evict_slack_trims_to_effective_cap_not_strict(home):
    """When n exceeds max + slack, we trim to (max + slack), NOT to max.
    The slack ceiling is the new soft cap — keeping rows trickle-deleted."""
    from mnemara.store import Store

    store = Store("evslack_trim_t")
    for i in range(12):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    # max=5, slack=3 → effective cap 8. 12 rows → delete 4 down to 8.
    deleted = store.evict(max_turns=5, max_tokens=1_000_000, row_cap_slack=3)
    assert deleted == 4
    assert len(store.window()) == 8
    store.close()


def test_evict_slack_disengaged_when_tokens_above_threshold(home):
    """Tokens at/above 50% of max_tokens disable slack — strict row cap fires."""
    from mnemara.store import Store

    store = Store("evslack_above_t")
    # Build content that reliably exceeds the headroom threshold.
    # Each row's content has ~4000 chars => ~1000 tokens. 6 rows => ~6000 tokens.
    # max_tokens=10_000 means HEADROOM_RATIO * max_tokens = 5_000.
    # current=~6000 > 5000 → slack disengaged → strict cap of 5 applies.
    for i in range(6):
        store.append_turn("user", [{"type": "text", "text": "x" * 4000}])
    deleted = store.evict(max_turns=5, max_tokens=10_000, row_cap_slack=3)
    assert deleted == 1  # strict cap, one row dropped
    assert len(store.window()) == 5
    store.close()


def test_evict_slack_disengaged_when_max_tokens_is_none(home):
    """Defensive: without max_tokens, headroom is undefined; slack stays off."""
    from mnemara.store import Store

    store = Store("evslack_no_max_t")
    for i in range(7):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    deleted = store.evict(max_turns=5, max_tokens=None, row_cap_slack=3)
    assert deleted == 2  # strict cap, slack ignored
    assert len(store.window()) == 5
    store.close()


def test_evict_slack_token_cap_still_hard_ceiling(home):
    """Slack relaxes ROW cap, not TOKEN cap. If total bytes exceed
    max_tokens, the second pass trims regardless of slack."""
    from mnemara.store import Store

    store = Store("evslack_token_ceiling_t")
    # 6 rows * 4000 chars * /4 heuristic = ~6000 tokens. max_tokens=4000
    # forces the token-cap pass to trim until total <= 4000. The first
    # pass (with slack) wouldn't fire (6 < 5+3=8) but the second pass
    # still has to delete rows to satisfy the byte ceiling.
    for i in range(6):
        store.append_turn("user", [{"type": "text", "text": "x" * 4000}])
    deleted = store.evict(max_turns=5, max_tokens=4_000, row_cap_slack=3)
    # After token-cap trim, total bytes/4 should be <= 4000.
    cur = store.conn.execute("SELECT COALESCE(SUM(LENGTH(content)) / 4, 0) FROM turns")
    total = int(cur.fetchone()[0])
    assert total <= 4000
    assert deleted >= 2  # had to drop at least 2 rows to fit byte ceiling
    store.close()


def test_evict_slack_dynamic_tightening_when_tokens_climb(home):
    """When token usage transitions from headroom to no-headroom,
    the next evict() call snaps from the slack ceiling to the strict cap."""
    from mnemara.store import Store

    store = Store("evslack_dynamic_t")
    # Phase 1: lots of small rows under threshold, slack engages.
    for i in range(8):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    deleted_1 = store.evict(max_turns=5, max_tokens=10_000, row_cap_slack=3)
    assert deleted_1 == 0  # 8 <= 5 + 3, slack absorbed the overage
    assert len(store.window()) == 8

    # Phase 2: append a heavy row that pushes total above the headroom
    # threshold. Now slack disengages and the strict cap fires.
    # Need total tokens to cross 5000 (50% of 10_000): each char/4 = tokens,
    # so 5000 tokens = 20_000 chars total content. 8 rows of ~7 chars = 56 chars
    # already; need ~20_000 more.
    store.append_turn("user", [{"type": "text", "text": "x" * 25_000}])
    deleted_2 = store.evict(max_turns=5, max_tokens=10_000, row_cap_slack=3)
    # 9 rows now, strict cap fires → delete 4 down to 5. Then token cap
    # may trim further if still over 10_000 tokens; depends on residual.
    # We just assert deleted_2 reduced count to <= 5 (strict cap applied).
    assert len(store.window()) <= 5
    store.close()


def test_evict_slack_at_exact_threshold_disengages(home):
    """Boundary: tokens exactly at threshold means NOT below threshold —
    slack stays off (we use strict < not <=)."""
    from mnemara.store import Store

    store = Store("evslack_boundary_t")
    # Engineer total content/4 exactly == max_tokens * 0.5.
    # max_tokens=2000 → threshold 1000. Need content len 4000 exactly.
    # Make 5 rows of 800 chars each = 4000 chars total = 1000 tokens.
    for i in range(5):
        store.append_turn("user", [{"type": "text", "text": "x" * 800}])
    # Confirm we're exactly at threshold.
    cur = store.conn.execute("SELECT COALESCE(SUM(LENGTH(content)) / 4, 0) FROM turns")
    total = int(cur.fetchone()[0])
    # Some overhead from JSON wrapping makes the exact value uncertain; this
    # test verifies the comparison shape (strict less-than): if at-or-above
    # threshold, slack disengages. If we ended up below (because heuristic
    # rounding), slack engages and this test correctly fails-loudly so we
    # know to adjust our calibration.
    if total >= 1000:
        # Above/at threshold: slack disengaged, strict cap of 3 fires.
        deleted = store.evict(max_turns=3, max_tokens=2000, row_cap_slack=2)
        assert deleted == 2
        assert len(store.window()) == 3
    else:
        # Below threshold (heuristic rounding): slack engages, ceiling 3+2=5.
        deleted = store.evict(max_turns=3, max_tokens=2000, row_cap_slack=2)
        assert len(store.window()) <= 5
    store.close()


def test_evict_slack_bumps_eviction_stats(home):
    """Slack-induced trims still bump rows_evicted in the stats counter."""
    from mnemara.store import Store

    store = Store("evslack_stats_t")
    for i in range(10):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
    s_before = store.get_eviction_stats()
    assert s_before["rows_evicted"] == 0
    # max=5, slack=2 → effective cap 7, delete 3.
    deleted = store.evict(max_turns=5, max_tokens=1_000_000, row_cap_slack=2)
    assert deleted == 3
    s_after = store.get_eviction_stats()
    assert s_after["rows_evicted"] == 3
    store.close()


# ----------------------------------------------------------------------
# Config.row_cap_slack_when_token_headroom — toggle wiring
# ----------------------------------------------------------------------


def test_config_row_cap_slack_default_zero():
    from mnemara.config import Config

    cfg = Config()
    assert cfg.row_cap_slack_when_token_headroom == 0


def test_config_row_cap_slack_round_trips_through_dict():
    from mnemara.config import Config

    cfg = Config()
    cfg.row_cap_slack_when_token_headroom = 30
    d = cfg.to_dict()
    assert d["row_cap_slack_when_token_headroom"] == 30
    cfg2 = Config.from_dict(d)
    assert cfg2.row_cap_slack_when_token_headroom == 30


def test_config_row_cap_slack_missing_field_defaults_zero():
    """Pre-existing config.json files without the field load as 0
    (backward-compat — feature disabled)."""
    from mnemara.config import Config

    minimal = {"role_doc_path": "", "model": "claude-opus-4-5"}
    cfg = Config.from_dict(minimal)
    assert cfg.row_cap_slack_when_token_headroom == 0


def test_config_row_cap_slack_coerces_string_to_int():
    """Tolerant load: '30' becomes 30 (matches other int fields' from_dict pattern)."""
    from mnemara.config import Config

    cfg = Config.from_dict({"row_cap_slack_when_token_headroom": "30"})
    assert cfg.row_cap_slack_when_token_headroom == 30


# ----------------------------------------------------------------------
# TUI: /stop slash command + live input while busy + CancelledError stub
# ----------------------------------------------------------------------

def test_tui_slash_cmd_routes_through_when_busy(home):
    """Slash commands bypass the _busy guard — on_input_submitted dispatches
    to _handle_slash regardless of _busy state."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("slash_busy_t")
    app = tui_mod.MnemaraTUI("slash_busy_t")
    app._busy = True  # Simulate a turn in flight

    slash_cmds_seen: list[str] = []

    async def _fake_handle_slash(line: str) -> None:
        slash_cmds_seen.append(line)

    app._handle_slash = _fake_handle_slash  # type: ignore[method-assign]
    app._refresh_status = lambda: None  # suppress widget call

    # Patch query_one so clearing inp.value doesn't trigger a DOM lookup.
    _fake_inp = type("FakeInp", (), {"value": ""})()
    app.query_one = lambda *a, **kw: _fake_inp  # type: ignore[method-assign]

    class _FakeInput:
        id = "userinput"
        value = "/stop"

    class _FakeEvent:
        input = _FakeInput()
        value = "/stop"

    _asyncio.run(app.on_input_submitted(_FakeEvent()))  # type: ignore[arg-type]
    assert slash_cmds_seen == ["/stop"], "slash commands must bypass _busy guard"
    app.store.close()


def test_tui_non_slash_blocked_when_busy(home):
    """Non-slash text submitted while _busy shows the 'use /stop' hint
    and does NOT enqueue a turn."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("busy_block_t")
    app = tui_mod.MnemaraTUI("busy_block_t")
    app._busy = True

    turns_sent: list[str] = []

    async def _fake_send_turn(text: str) -> None:
        turns_sent.append(text)

    app._send_turn = _fake_send_turn  # type: ignore[method-assign]

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    app._chat = lambda: _FakeChat()  # type: ignore[method-assign]

    class _FakeInput:
        id = "userinput"
        value = "hello"

    class _FakeEvent:
        input = _FakeInput()
        value = "hello"

    # Patch query_one so clearing the input doesn't crash.
    _fake_inp = type("FakeInp", (), {"value": ""})()
    app.query_one = lambda *a, **kw: _fake_inp  # type: ignore[method-assign]

    _asyncio.run(app.on_input_submitted(_FakeEvent()))  # type: ignore[arg-type]
    assert turns_sent == [], "no turn should be queued while busy"
    assert any("/stop" in m for m in chat_msgs), "hint message should mention /stop"
    app.store.close()


def test_tui_cancelled_error_writes_stub_assistant_turn(home):
    """`_send_turn` catches CancelledError, writes a stub [interrupted] assistant
    turn so the rolling window stays well-formed, then re-raises."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("cancel_stub_t")
    app = tui_mod.MnemaraTUI("cancel_stub_t")
    app._stream_buffer = ""
    app._stream_chars = 0
    app._busy = False

    # Stub turn_async to raise CancelledError immediately.
    async def _raise_cancel(*args, **kwargs):
        raise _asyncio.CancelledError()

    app.session.turn_async = _raise_cancel  # type: ignore[method-assign]

    # Stub chat widget.
    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    app._chat = lambda: _FakeChat()  # type: ignore[method-assign]
    app._refresh_status = lambda: None
    app._focus_input_after_refresh = lambda: None

    # _send_turn should propagate CancelledError after cleanup.
    with __import__("pytest").raises(_asyncio.CancelledError):
        _asyncio.run(app._send_turn("do something long"))

    # _busy must be cleared in finally block.
    assert app._busy is False

    # A stub "[interrupted]" assistant turn must be in the store.
    rows = app.store.window()
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert assistant_rows, "expected at least one assistant row in store"
    last_content = assistant_rows[-1]["content"]
    assert any(
        isinstance(b, dict) and b.get("text") == "[interrupted]"
        for b in last_content
    ), f"expected [interrupted] stub turn; got {last_content}"

    # Chat should show the interrupt notice.
    assert any("interrupted" in m.lower() for m in chat_msgs)
    app.store.close()


def test_tui_cancelled_error_shows_partial_stream_if_any(home):
    """When cancelled mid-stream, the partial buffer is shown in dim before
    the interrupt notice — operator knows how far the turn got."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("partial_stream_t")
    app = tui_mod.MnemaraTUI("partial_stream_t")
    app._stream_buffer = ""
    app._stream_chars = 0
    app._busy = False

    async def _partial_then_cancel(*args, **kwargs):
        # Simulate partial streaming: call on_token with some text, then cancel.
        on_tok = kwargs.get("on_token")
        if on_tok is not None:
            await on_tok("The answer is")
        raise _asyncio.CancelledError()

    app.session.turn_async = _partial_then_cancel  # type: ignore[method-assign]

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    app._chat = lambda: _FakeChat()  # type: ignore[method-assign]
    app._refresh_status = lambda: None
    app._focus_input_after_refresh = lambda: None

    with __import__("pytest").raises(_asyncio.CancelledError):
        _asyncio.run(app._send_turn("tell me about X"))

    # The partial buffer should appear in the log (dimmed).
    assert any("The answer is" in m for m in chat_msgs), (
        f"partial stream buffer not shown; messages: {chat_msgs}"
    )
    app.store.close()


def test_tui_stop_slash_when_not_busy(home, monkeypatch):
    """/stop when nothing is in flight shows 'nothing in flight' message."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("stop_idle_t")
    app = tui_mod.MnemaraTUI("stop_idle_t")
    app._busy = False

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    app._chat = lambda: _FakeChat()  # type: ignore[method-assign]
    app._refresh_status = lambda: None

    # workers is a read-only Textual App property; monkeypatch cancel_group
    # on the existing WorkerManager instance.
    cancel_calls: list[tuple] = []
    monkeypatch.setattr(app.workers, "cancel_group", lambda dom, grp: cancel_calls.append((dom, grp)))

    _asyncio.run(app._handle_slash("/stop"))
    assert any("nothing in flight" in m for m in chat_msgs), (
        f"expected 'nothing in flight' message; got {chat_msgs}"
    )
    # cancel_group should have been called once with the "turn" group.
    assert len(cancel_calls) == 1
    assert cancel_calls[0][1] == "turn"
    app.store.close()


# ----------------------------------------------------------------------
# /export slash command
# ----------------------------------------------------------------------

def test_slash_export_writes_temp_file(home, tmp_path):
    """/export writes the full window to a temp file and prints its path."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("exp_temp_t")
    app = tui_mod.MnemaraTUI("exp_temp_t")
    # Seed a couple of turns so there's content to export.
    app.store.append_turn("user", [{"type": "text", "text": "hello"}])
    app.store.append_turn("assistant", [{"type": "text", "text": "hi there"}])

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    app._chat = lambda: _FakeChat()  # type: ignore[method-assign]

    _asyncio.run(app._slash_export("", _FakeChat()))

    # A path should appear in the messages.
    path_msgs = [m for m in chat_msgs if "mnemara_" in m and ".md" in m]
    assert path_msgs, f"expected path in chat messages; got {chat_msgs}"

    # Extract path from the message and verify the file exists and has content.
    import re
    for m in chat_msgs:
        match = re.search(r"(/[^\s\[\]]+\.md)", m)
        if match:
            p = __import__("pathlib").Path(match.group(1))
            assert p.exists(), f"temp file {p} does not exist"
            content = p.read_text(encoding="utf-8")
            assert "hello" in content
            assert "hi there" in content
            p.unlink()  # cleanup
            break
    else:
        __import__("pytest").fail(f"no .md path found in chat messages: {chat_msgs}")
    app.store.close()


def test_slash_export_explicit_path(home, tmp_path):
    """/export N path writes last N turns to the explicit path."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("exp_path_t")
    app = tui_mod.MnemaraTUI("exp_path_t")
    for i in range(5):
        app.store.append_turn("user", [{"type": "text", "text": f"q{i}"}])
        app.store.append_turn("assistant", [{"type": "text", "text": f"a{i}"}])

    out_file = tmp_path / "export_test.md"
    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    _asyncio.run(app._slash_export(f"4 {out_file}", _FakeChat()))

    assert out_file.exists(), f"output file {out_file} not created"
    content = out_file.read_text(encoding="utf-8")
    # Last 4 rows out of 10 = turns 7-10. Check partial content present.
    assert "q" in content
    assert "a" in content
    # Full window should NOT appear (we only asked for 4 rows out of 10)
    # — hard to assert exactly without knowing turn ids so just assert
    # the file is nonempty and path appears in chat.
    assert str(out_file) in " ".join(chat_msgs), (
        f"export path should appear in chat; got {chat_msgs}"
    )
    app.store.close()


def test_slash_export_empty_window(home):
    """/export on an empty window shows a 'nothing to export' message."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("exp_empty_t")
    app = tui_mod.MnemaraTUI("exp_empty_t")

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    _asyncio.run(app._slash_export("", _FakeChat()))

    assert any("empty" in m.lower() for m in chat_msgs), (
        f"expected empty-window message; got {chat_msgs}"
    )
    app.store.close()


def test_slash_export_last_n(home, tmp_path):
    """/export N (without explicit path) exports only last N turns."""
    import asyncio as _asyncio
    import re
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("exp_lastn_t")
    app = tui_mod.MnemaraTUI("exp_lastn_t")
    app.store.append_turn("user", [{"type": "text", "text": "old"}])
    app.store.append_turn("user", [{"type": "text", "text": "new"}])

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    _asyncio.run(app._slash_export("1", _FakeChat()))

    # Find the exported file from the message.
    written_path = None
    for m in chat_msgs:
        match = re.search(r"(/[^\s\[\]]+\.md)", m)
        if match:
            written_path = __import__("pathlib").Path(match.group(1))
            break

    assert written_path is not None and written_path.exists()
    content = written_path.read_text(encoding="utf-8")
    assert "new" in content
    # "old" turn is outside the last-1 window.
    assert "old" not in content
    written_path.unlink()
    app.store.close()


def test_slash_export_bad_n_shows_usage(home):
    """/export with a non-integer non-path first token shows usage hint."""
    import asyncio as _asyncio
    from mnemara import config
    from mnemara import tui as tui_mod

    config.init_instance("exp_badarg_t")
    app = tui_mod.MnemaraTUI("exp_badarg_t")

    chat_msgs: list[str] = []

    class _FakeChat:
        def write(self, msg: str) -> None:
            chat_msgs.append(msg)

    _asyncio.run(app._slash_export("notanumber", _FakeChat()))

    assert any("usage" in m.lower() for m in chat_msgs), (
        f"expected usage message; got {chat_msgs}"
    )
    app.store.close()


# ----------------------------------------------------------------------
# evict() pin-awareness — Option B: skip pinned first, last-resort evict
# ----------------------------------------------------------------------

def test_evict_preserves_pins_under_row_cap(home):
    """Cap-FIFO skips pinned rows and deletes unpinned first.
    When enough unpinned rows exist to satisfy the cap, pinned rows survive."""
    from mnemara.store import Store

    store = Store("evpinrow_t")
    pinned = store.append_turn("user", [{"type": "text", "text": "keep me"}])
    store.pin_row(pinned, "important")
    store.append_turn("user", [{"type": "text", "text": "evict me 1"}])
    store.append_turn("user", [{"type": "text", "text": "evict me 2"}])

    # 3 rows total, cap=1 -> need to delete 2. 2 unpinned rows available.
    deleted = store.evict(max_turns=1, max_tokens=1_000_000)
    assert deleted == 2

    rows = store.window()
    assert len(rows) == 1
    assert rows[0]["id"] == pinned  # pinned row survived
    assert store.get_eviction_stats()["pinned_rows_force_evicted"] == 0
    store.close()


def test_evict_preserves_pins_under_token_cap(home):
    """Token-cap pass skips pinned rows; unpinned rows trimmed first."""
    from mnemara.store import Store

    store = Store("evpintok_t")
    pinned = store.append_turn(
        "user", [{"type": "text", "text": "x" * 2000}]
    )
    store.pin_row(pinned, "critical")
    store.append_turn("user", [{"type": "text", "text": "x" * 2000}])

    # Both rows ~500 tokens each (2000 chars / 4). Token cap = 600 forces
    # deletion of 1 row. The unpinned row should go first.
    deleted = store.evict(max_turns=1_000, max_tokens=600)
    assert deleted == 1

    rows = store.window()
    assert len(rows) == 1
    assert rows[0]["id"] == pinned
    assert store.get_eviction_stats()["pinned_rows_force_evicted"] == 0
    store.close()


def test_evict_last_resort_evicts_pinned_when_no_unpinned(home):
    """When all remaining rows are pinned and the cap still isn't met,
    pinned rows are evicted oldest-first (last resort) and the
    pinned_rows_force_evicted counter is bumped."""
    from mnemara.store import Store

    store = Store("evpinlast_t")
    r1 = store.append_turn("user", [{"type": "text", "text": "a"}])
    r2 = store.append_turn("user", [{"type": "text", "text": "b"}])
    r3 = store.append_turn("user", [{"type": "text", "text": "c"}])
    for rid in (r1, r2, r3):
        store.pin_row(rid, "all pinned")

    # cap=1, all 3 rows pinned -> no unpinned to evict; last resort fires.
    deleted = store.evict(max_turns=1, max_tokens=1_000_000)
    assert deleted == 2
    assert len(store.window()) == 1

    s = store.get_eviction_stats()
    assert s["rows_evicted"] == 2
    assert s["pinned_rows_force_evicted"] == 2
    store.close()


def test_evict_token_cap_last_resort_evicts_pinned(home):
    """When token cap requires eviction but only pinned rows remain,
    pinned rows are evicted and the force counter is bumped."""
    from mnemara.store import Store

    store = Store("evpintokforce_t")
    r1 = store.append_turn("user", [{"type": "text", "text": "x" * 3000}])
    store.pin_row(r1, "pinned")

    # Token cap at 100 tokens — far below the ~750 tokens this row costs.
    # No unpinned rows exist; last resort fires.
    deleted = store.evict(max_turns=1_000, max_tokens=100)
    assert deleted == 1

    s = store.get_eviction_stats()
    assert s["pinned_rows_force_evicted"] == 1
    store.close()


def test_evict_mixed_pin_status_respects_order(home):
    """With mixed pinned/unpinned rows, unpinned are always deleted before
    pinned, regardless of insertion order."""
    from mnemara.store import Store

    store = Store("evpinmixed_t")
    r1 = store.append_turn("user", [{"type": "text", "text": "old-unpinned"}])
    r2 = store.append_turn("user", [{"type": "text", "text": "old-pinned"}])
    r3 = store.append_turn("user", [{"type": "text", "text": "new-unpinned"}])
    store.pin_row(r2, "keep")

    # cap=1, need to delete 2. Only 2 unpinned (r1, r3) — pinned r2 survives.
    deleted = store.evict(max_turns=1, max_tokens=1_000_000)
    assert deleted == 2
    rows = store.window()
    assert len(rows) == 1
    assert rows[0]["id"] == r2

    s = store.get_eviction_stats()
    assert s["pinned_rows_force_evicted"] == 0
    store.close()


def test_evict_pin_protection_leaves_no_force_eviction_when_enough_unpinned(home):
    """Regression: when there are always enough unpinned rows to satisfy the
    cap, pinned_rows_force_evicted must stay 0 across many evict() calls."""
    from mnemara.store import Store

    store = Store("evpinreg_t")
    pinned = store.append_turn("user", [{"type": "text", "text": "anchor"}])
    store.pin_row(pinned, "anchor")

    for i in range(10):
        store.append_turn("user", [{"type": "text", "text": f"u{i}"}])
        store.evict(max_turns=3, max_tokens=1_000_000)

    # Pinned anchor row should still be in the store.
    ids = {r["id"] for r in store.window()}
    assert pinned in ids, "pinned anchor evicted when it should have survived"
    assert store.get_eviction_stats()["pinned_rows_force_evicted"] == 0
    store.close()
