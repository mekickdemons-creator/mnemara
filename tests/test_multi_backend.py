"""v0.2.1 — wiki layer + RAG (LanceDB + nomic-embed-text) tests.

No real network: Ollama embed calls are monkeypatched to deterministic
fake vectors. RAG graceful-degradation tested by forcing the embed call
to raise ConnectionError.

NOTE: Skipped on gemma package (no claude_agent_sdk dep).
"""
from __future__ import annotations

# Skip entire module when claude_agent_sdk is not installed (gemma package).
# AgentSession construction in multi-backend tests requires the SDK.
import pytest
pytest.importorskip("claude_agent_sdk")

import json
from pathlib import Path


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Reset RAG store singletons between tests so each instance gets a fresh
    # LanceDB connection rooted under the test home.
    from mnemara import rag as rag_mod
    rag_mod.reset_stores()
    return tmp_path


def _fake_embed(text: str, dim: int = 768) -> list[float]:
    """Deterministic toy embedding: bag-of-words frequency over fixed vocab."""
    vec = [0.0] * dim
    for i, ch in enumerate(text.lower()):
        vec[(ord(ch) * 31 + i) % dim] += 1.0
    # L2 normalise
    s = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / s for v in vec]


def test_wiki_round_trip(home):
    from mnemara import config, wiki
    config.init_instance("w1")
    f = wiki.write_page("w1", "patterns/loader_traps", "# Traps\n\nstale modules.\n")
    assert f.exists()
    body = wiki.read_page("w1", "patterns/loader_traps")
    assert "stale modules" in body
    # append mode adds to existing
    wiki.write_page("w1", "patterns/loader_traps", "more text\n", mode="append")
    body2 = wiki.read_page("w1", "patterns/loader_traps")
    assert "stale modules" in body2 and "more text" in body2
    # list_pages
    wiki.write_page("w1", "replay_policy", "policy body\n")
    pages = wiki.list_pages("w1")
    paths = sorted(p["path"] for p in pages)
    assert paths == ["patterns/loader_traps", "replay_policy"]
    for p in pages:
        assert p["size_bytes"] > 0
        assert "T" in p["last_modified"]
    # prefix filter
    only = wiki.list_pages("w1", prefix="patterns")
    assert len(only) == 1 and only[0]["path"] == "patterns/loader_traps"
    # missing page
    assert wiki.read_page("w1", "nope") is None


def test_wiki_path_safety(home):
    from mnemara import config, wiki
    config.init_instance("w2")
    with pytest.raises(ValueError):
        wiki.write_page("w2", "../escape", "bad")
    with pytest.raises(ValueError):
        wiki.write_page("w2", "", "empty")


def test_write_memory_routes_to_wiki_and_rag(home, monkeypatch):
    """write_memory(category='wiki/topic', cfg=...) writes to memory file,
    wiki page, and RAG index."""
    from mnemara import config, paths, rag as rag_mod, tools, wiki
    config.init_instance("m1")
    cfg = config.load("m1")

    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    p = tools.write_memory(
        "m1",
        "loader patterns: never re-import after sys.modules edits",
        category="wiki/patterns/loader_traps",
        cfg=cfg,
    )
    assert p.exists()
    assert "loader patterns" in p.read_text()

    body = wiki.read_page("m1", "patterns/loader_traps")
    assert body is not None and "loader patterns" in body

    res = rag_mod.store_for("m1", cfg).query("loader sys.modules", k=3)
    assert res["ok"], res
    assert any("loader patterns" in r["text"] for r in res["results"])


def test_rag_index_query_round_trip(home, monkeypatch):
    from mnemara import config, rag as rag_mod
    config.init_instance("r1")
    cfg = config.load("r1")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))
    store = rag_mod.store_for("r1", cfg)
    a = store.index("alpha foxtrot golf hotel", kind="manual")
    b = store.index("kilo lima mike november", kind="manual")
    assert a["ok"] and b["ok"]
    res = store.query("alpha foxtrot", k=2)
    assert res["ok"]
    assert res["results"][0]["text"].startswith("alpha")


def test_rag_graceful_degradation_when_ollama_unreachable(home, monkeypatch):
    from mnemara import config, rag as rag_mod
    config.init_instance("r2")
    cfg = config.load("r2")

    def _boom(url, model, text, timeout=30.0):
        raise ConnectionError("Ollama not reachable")

    monkeypatch.setattr(rag_mod, "embed_text", _boom)
    store = rag_mod.store_for("r2", cfg)
    res = store.index("anything", kind="manual")
    assert not res["ok"]
    assert "RAG backend unavailable" in res["error"]
    res2 = store.query("anything", k=3)
    assert not res2["ok"]
    assert "RAG backend unavailable" in res2["error"]


def test_rag_disabled_returns_inactive(home, monkeypatch):
    from mnemara import config, rag as rag_mod
    config.init_instance("r3")
    cfg = config.load("r3")
    cfg.rag_enabled = False
    res = rag_mod.store_for("r3", cfg).index("text", kind="manual")
    assert not res["ok"]
    assert "disabled" in res["error"]


def test_config_backward_compat(home):
    """A config.json missing v0.2.1 fields still loads with defaults."""
    from mnemara import config, paths
    config.init_instance("bc1")
    p = paths.config_path("bc1")
    raw = json.loads(p.read_text())
    for k in ("rag_enabled", "rag_embed_url", "rag_embed_model",
              "rag_auto_index_memory", "rag_auto_index_wiki"):
        raw.pop(k, None)
    p.write_text(json.dumps(raw))
    cfg = config.load("bc1")
    assert cfg.rag_enabled is True
    assert cfg.rag_embed_url.startswith("http://localhost:11434")
    assert cfg.rag_embed_model == "nomic-embed-text"
    assert cfg.rag_auto_index_memory is True


def test_agent_registers_new_tools(home, monkeypatch):
    """The five new in-process tools register and the wiki_write handler
    routes through to wiki + RAG."""
    import asyncio as _asyncio
    from mnemara import agent as agent_mod, config, rag as rag_mod, wiki
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    config.init_instance("ag1")
    cfg = config.load("ag1")
    store = Store("ag1")
    perms = PermissionStore("ag1")
    runner = ToolRunner("ag1", cfg, perms, prompt=lambda t, x: "deny")

    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    captured = {}

    async def _fake_query(*, prompt, options, transport=None):
        captured["options"] = options
        async for _ in prompt:
            break
        if False:
            yield None
        return

    monkeypatch.setattr(agent_mod, "query", _fake_query)

    session = agent_mod.AgentSession(cfg, store, runner)
    session.turn("ping")

    handlers = session._registered_tools
    for name in ("wiki_read", "wiki_write", "wiki_list", "rag_index", "rag_query"):
        assert name in handlers

    # wiki_write should write the file and (because rag_auto_index_wiki) index.
    res = _asyncio.run(handlers["wiki_write"]({
        "path": "replay_policy",
        "content": "Always replay from offset 0.\n",
        "mode": "replace",
    }))
    assert "Wrote wiki page" in res["content"][0]["text"]
    assert wiki.read_page("ag1", "replay_policy") is not None
    assert session.wiki_writes == 1
    assert session.rag_indexes >= 1

    # rag_query returns the indexed wiki content.
    qres = _asyncio.run(handlers["rag_query"]({"question": "replay offset", "k": 3}))
    payload = qres["content"][0]["text"]
    assert "replay" in payload.lower()
    assert session.rag_queries == 1

    # wiki_read for missing page
    miss = _asyncio.run(handlers["wiki_read"]({"path": "nonexistent"}))
    assert miss["content"][0]["text"] == "no such page"

    # allowed_tools includes the five new MCP tool names
    allowed = captured["options"].allowed_tools
    for n in ("mcp__mnemara_memory__wiki_read",
              "mcp__mnemara_memory__wiki_write",
              "mcp__mnemara_memory__wiki_list",
              "mcp__mnemara_memory__rag_index",
              "mcp__mnemara_memory__rag_query"):
        assert n in allowed
    store.close()
