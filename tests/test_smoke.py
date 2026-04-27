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
    assert cfg.max_window_tokens == 200_000


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
