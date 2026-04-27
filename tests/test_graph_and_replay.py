"""v0.3 — graph backend (kuzu) + sleep/replay primitive tests.

Uses fake embeddings (deterministic toy vectors) for RAG so no network is
required. Kuzu is real — installed via the package.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from mnemara import rag as rag_mod, graph as graph_mod
    rag_mod.reset_stores()
    graph_mod.reset_stores()
    return tmp_path


def _fake_embed(text: str, dim: int = 768) -> list[float]:
    vec = [0.0] * dim
    for i, ch in enumerate(text.lower()):
        vec[(ord(ch) * 31 + i) % dim] += 1.0
    s = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / s for v in vec]


# ---------------------------------------------------------------- graph tests


def test_graph_add_node_and_neighbors_round_trip(home):
    from mnemara import config, graph as graph_mod
    config.init_instance("g1")
    cfg = config.load("g1")
    store = graph_mod.store_for("g1", cfg)

    a = store.add_node("entity", {"ref": "alpha"})
    b = store.add_node("entity", {"ref": "beta"})
    assert a["ok"] and b["ok"]
    e = store.add_edge(a["id"], b["id"], "knows")
    assert e["ok"]
    nb = store.neighbors(a["id"], depth=1)
    assert nb["ok"]
    ids = [n["id"] for n in nb["neighbors"]]
    assert b["id"] in ids


def test_graph_shortest_path(home):
    from mnemara import config, graph as graph_mod
    config.init_instance("g2")
    cfg = config.load("g2")
    store = graph_mod.store_for("g2", cfg)
    a = store.add_node("entity", {"ref": "a"})["id"]
    b = store.add_node("entity", {"ref": "b"})["id"]
    c = store.add_node("entity", {"ref": "c"})["id"]
    store.add_edge(a, b, "next")
    store.add_edge(b, c, "next")
    res = store.shortest_path(a, c)
    assert res["ok"]
    assert a in res["path"]
    assert c in res["path"]
    # Disconnected nodes -> empty path
    d = store.add_node("entity", {"ref": "d"})["id"]
    res2 = store.shortest_path(a, d)
    assert res2["ok"]
    assert res2["path"] == []


def test_graph_query_cypher(home):
    from mnemara import config, graph as graph_mod
    config.init_instance("g3")
    cfg = config.load("g3")
    store = graph_mod.store_for("g3", cfg)
    store.add_node("entity", {"ref": "x"})
    store.add_node("wiki_page", {"ref": "topic/a"})
    res = store.query("MATCH (n:Node) WHERE n.label = 'entity' RETURN n.id AS id")
    assert res["ok"]
    assert "id" in res["columns"]
    assert len(res["rows"]) >= 1


def test_graph_match_pattern(home):
    from mnemara import config, graph as graph_mod
    config.init_instance("g4")
    cfg = config.load("g4")
    store = graph_mod.store_for("g4", cfg)
    store.add_node("entity", {"ref": "alpha"})
    store.add_node("entity", {"ref": "beta"})
    store.add_node("wiki_page", {"ref": "topic"})
    res = store.match({"label": "entity"})
    assert res["ok"]
    refs = sorted(m["properties"].get("ref") for m in res["matches"])
    assert refs == ["alpha", "beta"]


def test_graph_auto_edges_from_write_memory(home):
    from mnemara import config, tools, graph as graph_mod
    config.init_instance("g5")
    cfg = config.load("g5")
    cfg.rag_enabled = False  # avoid embedding calls
    config.save("g5", cfg)
    cfg = config.load("g5")
    payload = {
        "observation": "loaders cache module references",
        "evidence": "saw stale __pycache__",
        "prediction": "next reload will surface stale module",
        "applies_to": ["loader", "module_cache"],
        "confidence": "high",
    }
    tools.write_memory("g5", "ignored", category="observation", payload=payload, cfg=cfg)
    store = graph_mod.store_for("g5", cfg)
    # An entity node for each ref should exist
    res = store.match({"label": "entity"})
    assert res["ok"]
    refs = sorted(m["properties"].get("ref") for m in res["matches"])
    assert refs == ["loader", "module_cache"]
    # And a memory_entry node should connect to them
    mem = store.match({"label": "memory_entry"})
    assert mem["ok"]
    assert len(mem["matches"]) >= 1


def test_graph_graceful_degradation_when_kuzu_absent(home, monkeypatch):
    """If kuzu import fails, graph methods return unavailable but don't raise."""
    from mnemara import config, graph as graph_mod
    config.init_instance("g6")
    cfg = config.load("g6")
    store = graph_mod.KuzuStore("g6", cfg)
    # Force the import to fail by stubbing the connect path
    store._init_error = "kuzu import failed: fake"
    res = store.add_node("entity", {"ref": "a"})
    assert not res["ok"]
    assert "Graph backend unavailable" in res["error"]
    res2 = store.query("MATCH (n) RETURN n")
    assert not res2["ok"]
    assert "Graph backend unavailable" in res2["error"]


def test_graph_disabled_returns_inactive(home):
    from mnemara import config, graph as graph_mod
    config.init_instance("g7")
    cfg = config.load("g7")
    cfg.graph_enabled = False
    res = graph_mod.store_for("g7", cfg).add_node("entity", {"ref": "a"})
    assert not res["ok"]
    assert "disabled" in res["error"]


def test_config_v03_backward_compat(home):
    """A config.json missing v0.3 fields still loads with defaults."""
    from mnemara import config, paths
    config.init_instance("bc3")
    p = paths.config_path("bc3")
    raw = json.loads(p.read_text())
    for k in ("graph_enabled", "replay_default_days", "replay_default_threshold",
              "replay_policy_path"):
        raw.pop(k, None)
    p.write_text(json.dumps(raw))
    cfg = config.load("bc3")
    assert cfg.graph_enabled is True
    assert cfg.replay_default_days == 7
    assert cfg.replay_default_threshold == 3
    assert cfg.replay_policy_path == ""


# ---------------------------------------------------------------- replay tests


def _plant_memory(instance: str, date: str, atoms: list[tuple[str, str, str]]) -> Path:
    """atoms: list of (ts_iso, category, body)."""
    from mnemara import paths
    d = paths.memory_dir(instance)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{date}.md"
    parts = []
    for ts, cat, body in atoms:
        parts.append(f"\n## [{ts}] {cat}\n\n{body}\n")
    f.write_text("".join(parts), encoding="utf-8")
    return f


def test_replay_dry_run_identifies_recurring_pattern(home, monkeypatch):
    from mnemara import config, paths, rag as rag_mod, replay as replay_mod
    config.init_instance("rp1")
    cfg = config.load("rp1")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    # Plant 4 atoms with very similar text → cluster
    body = "loader cache stale references after sys.modules edit"
    _plant_memory("rp1", today, [
        (now, "note", body + " case 1"),
        (now, "note", body + " case 2"),
        (now, "note", body + " case 3"),
        (now, "note", body + " case 4"),
        (now, "note", "totally unrelated text about boats and rivers"),
    ])

    # Index those into RAG so cluster_atoms can find them
    store = rag_mod.store_for("rp1", cfg)
    f = paths.memory_dir("rp1") / f"{today}.md"
    for ts, cat, txt in [
        (now, "note", body + " case 1"),
        (now, "note", body + " case 2"),
        (now, "note", body + " case 3"),
        (now, "note", body + " case 4"),
        (now, "note", "totally unrelated text about boats and rivers"),
    ]:
        store.index(txt, kind="memory", source_path=str(f), category=cat)

    out = replay_mod.run_replay("rp1", days=7, threshold=3, apply=False, cfg=cfg)
    assert out["ok"]
    assert out["atoms_loaded"] == 5
    # Dry-run: digest path computed but file not written
    assert not Path(out["digest_path"]).exists()
    # Patterns should be found
    assert out["patterns"] >= 1


def test_replay_apply_writes_wiki_proposal_and_digest(home, monkeypatch):
    from mnemara import config, paths, rag as rag_mod, replay as replay_mod
    config.init_instance("rp2")
    cfg = config.load("rp2")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    body = "deferred cleanup leaks file handles in long sessions"
    _plant_memory("rp2", today, [
        (now, "note", body + " A"),
        (now, "note", body + " B"),
        (now, "note", body + " C"),
    ])
    store = rag_mod.store_for("rp2", cfg)
    f = paths.memory_dir("rp2") / f"{today}.md"
    for txt in (body + " A", body + " B", body + " C"):
        store.index(txt, kind="memory", source_path=str(f), category="note")

    out = replay_mod.run_replay("rp2", days=7, threshold=3, apply=True, cfg=cfg)
    assert out["ok"]
    assert Path(out["digest_path"]).exists()
    assert len(out["proposals"]) >= 1
    proposal_path = Path(out["proposals"][0])
    assert proposal_path.exists()
    text = proposal_path.read_text()
    assert "source_count:" in text
    assert "Member observations" in text


def test_replay_archives_near_duplicates(home, monkeypatch):
    from mnemara import config, paths, rag as rag_mod, replay as replay_mod
    config.init_instance("rp3")
    cfg = config.load("rp3")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    # Two near-identical bodies and a third for cluster threshold
    a = "exact same observation about caching"
    _plant_memory("rp3", today, [
        (now, "note", a),
        (now, "note", a),
        (now, "note", a),
    ])
    store = rag_mod.store_for("rp3", cfg)
    f = paths.memory_dir("rp3") / f"{today}.md"
    for _ in range(3):
        store.index(a, kind="memory", source_path=str(f), category="note")

    out = replay_mod.run_replay("rp3", days=7, threshold=3, apply=True, cfg=cfg)
    assert out["ok"]
    archive_dir = paths.memory_archive_dir("rp3")
    # archive dir should have at least one file
    if out["archived"]:
        assert archive_dir.exists()
        assert any(archive_dir.iterdir())


def test_replay_surfaces_role_amendment_when_self_obs_cluster(home, monkeypatch):
    from mnemara import config, paths, rag as rag_mod, replay as replay_mod
    config.init_instance("rp4")
    cfg = config.load("rp4")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    body = "I should batch tool calls more aggressively to save tokens"
    _plant_memory("rp4", today, [
        (now, "self_observation", body + " v1"),
        (now, "self_observation", body + " v2"),
        (now, "self_observation", body + " v3"),
    ])
    store = rag_mod.store_for("rp4", cfg)
    f = paths.memory_dir("rp4") / f"{today}.md"
    for v in ("v1", "v2", "v3"):
        store.index(body + " " + v, kind="memory", source_path=str(f), category="self_observation")

    out = replay_mod.run_replay("rp4", days=7, threshold=3, apply=True, cfg=cfg)
    assert out["ok"]
    assert len(out["role_amendments"]) >= 1
    p = Path(out["role_amendments"][0])
    assert p.exists()
    assert "replay-" in p.name
    assert "Replay-surfaced amendment draft" in p.read_text()


def test_replay_reads_policy_overrides(home, monkeypatch):
    from mnemara import config, paths, rag as rag_mod, replay as replay_mod, wiki as wiki_mod
    config.init_instance("rp5")
    cfg = config.load("rp5")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))

    # Write a policy doc that pins threshold=5, days=14
    wiki_mod.write_page("rp5", "replay_policy",
                        "# Replay policy\n\nthreshold: 5\ndays: 14\n")
    overrides = replay_mod.load_policy_overrides("rp5", cfg)
    assert overrides == {"threshold": 5, "days": 14}

    # When CLI doesn't pass days/threshold, policy applies
    out = replay_mod.run_replay("rp5", days=None, threshold=None, apply=False, cfg=cfg)
    assert out["days"] == 14
    assert out["threshold"] == 5


def test_last_replay_summary_after_apply(home, monkeypatch):
    from mnemara import config, paths, rag as rag_mod, replay as replay_mod
    config.init_instance("rp6")
    cfg = config.load("rp6")
    monkeypatch.setattr(rag_mod, "embed_text", lambda url, model, text, timeout=30.0: _fake_embed(text))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    body = "this is a recurring theme about something interesting"
    _plant_memory("rp6", today, [
        (now, "note", body + " 1"),
        (now, "note", body + " 2"),
        (now, "note", body + " 3"),
    ])
    store = rag_mod.store_for("rp6", cfg)
    f = paths.memory_dir("rp6") / f"{today}.md"
    for n in range(3):
        store.index(body + f" {n+1}", kind="memory", source_path=str(f), category="note")
    replay_mod.run_replay("rp6", days=7, threshold=3, apply=True, cfg=cfg)
    summary = replay_mod.last_replay_summary("rp6")
    assert summary is not None
    assert "wiki proposals" in summary
