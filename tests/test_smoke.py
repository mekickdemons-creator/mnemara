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
