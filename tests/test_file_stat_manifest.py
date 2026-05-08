"""Tests for the file stat manifest injection in agent.py."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from mnemara.agent import _inject_file_stat_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store_stub(reads: list[dict]):
    """Return a minimal store-like object with collect_read_stats()."""

    class _StoreStub:
        def collect_read_stats(self):
            result = {}
            for r in reads:
                fp = r["file_path"]
                result[fp] = {
                    "last_read_turn": r["turn"],
                    "last_hash": r["hash"],
                    "last_size": r["size"],
                }
            return result

    return _StoreStub()


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Empty state — no manifest rendered
# ---------------------------------------------------------------------------

def test_empty_store_returns_prompt_unchanged():
    store = _make_store_stub([])
    prompt = "## Role doc\nYou are a coder."
    result = _inject_file_stat_manifest(prompt, store)
    assert result == prompt


# ---------------------------------------------------------------------------
# Manifest rendered when reads exist
# ---------------------------------------------------------------------------

def test_manifest_appears_when_reads_exist(tmp_path):
    content = "hello world\n" * 100
    fp = tmp_path / "foo.py"
    fp.write_text(content)

    current_hash = _sha8(content)
    store = _make_store_stub([
        {"file_path": str(fp), "turn": 5, "hash": current_hash, "size": len(content)},
    ])

    result = _inject_file_stat_manifest("## Role\n", store)
    assert "File state manifest" in result
    assert str(fp) in result
    assert "turn 5" in result


# ---------------------------------------------------------------------------
# Fresh vs STALE detection
# ---------------------------------------------------------------------------

def test_fresh_file_shows_fresh(tmp_path):
    content = "x = 1\n"
    fp = tmp_path / "a.py"
    fp.write_text(content)

    store = _make_store_stub([
        {"file_path": str(fp), "turn": 1, "hash": _sha8(content), "size": len(content)},
    ])
    result = _inject_file_stat_manifest("", store)
    assert "fresh" in result


def test_stale_file_shows_stale(tmp_path):
    original = "x = 1\n"
    modified = "x = 99\n"
    fp = tmp_path / "b.py"
    fp.write_text(modified)  # disk has modified version

    store = _make_store_stub([
        # stored hash is from original content
        {"file_path": str(fp), "turn": 2, "hash": _sha8(original), "size": len(original)},
    ])
    result = _inject_file_stat_manifest("", store)
    assert "STALE" in result


# ---------------------------------------------------------------------------
# Missing file shows 'gone'
# ---------------------------------------------------------------------------

def test_missing_file_shows_gone(tmp_path):
    missing = str(tmp_path / "does_not_exist.py")
    store = _make_store_stub([
        {"file_path": missing, "turn": 3, "hash": "deadbeef", "size": 500},
    ])
    result = _inject_file_stat_manifest("", store)
    assert "gone" in result


# ---------------------------------------------------------------------------
# Flag off — no manifest injected
# ---------------------------------------------------------------------------

def test_flag_off_means_no_manifest(tmp_path):
    """When file_stat_manifest_enabled is False, _inject_file_stat_manifest
    should never be called by _build_options.  This test verifies the function
    itself returns unchanged prompt when the store is empty — the flag guard
    lives in agent._build_options, tested via integration."""
    # We test the function's own empty-guard here:
    store = _make_store_stub([])
    original = "## Role\nDo stuff."
    assert _inject_file_stat_manifest(original, store) == original


# ---------------------------------------------------------------------------
# Multiple files — all appear in table
# ---------------------------------------------------------------------------

def test_multiple_files_all_appear(tmp_path):
    files = []
    for name in ("alpha.py", "beta.py", "gamma.py"):
        fp = tmp_path / name
        fp.write_text(f"# {name}\n")
        files.append({"file_path": str(fp), "turn": 1, "hash": _sha8(f"# {name}\n"), "size": 10})

    store = _make_store_stub(files)
    result = _inject_file_stat_manifest("", store)
    for name in ("alpha.py", "beta.py", "gamma.py"):
        assert name in result


# ---------------------------------------------------------------------------
# Manifest appended AFTER existing prompt content
# ---------------------------------------------------------------------------

def test_manifest_appended_not_prepended(tmp_path):
    content = "# src\n"
    fp = tmp_path / "src.py"
    fp.write_text(content)

    prompt = "PROMPT_SENTINEL"
    store = _make_store_stub([
        {"file_path": str(fp), "turn": 1, "hash": _sha8(content), "size": len(content)},
    ])
    result = _inject_file_stat_manifest(prompt, store)
    assert result.index("PROMPT_SENTINEL") < result.index("File state manifest")
