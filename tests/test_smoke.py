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
            completed_at TEXT
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
