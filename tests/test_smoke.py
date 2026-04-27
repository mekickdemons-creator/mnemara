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
    assert cfg.max_window_tokens == 800_000


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
