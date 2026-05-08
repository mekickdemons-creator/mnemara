"""v0.8.0 tests: stamp_read_cache, rewritten compress_repeated_reads,
rewritten collect_read_stats, and end-to-end turn_async pipeline.

These tests exercise the REAL data paths that v0.6.0/v0.7.0 tests missed
because they relied on fabricated tool_result rows (which are never persisted
in production).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Step 1 unit tests — stamp_read_cache
# ---------------------------------------------------------------------------


def test_stamp_read_cache_updates_input_json(home):
    """stamp_read_cache inserts _cached_content and _cached_hash into the
    most-recent matching tool_use block's input dict and persists to SQLite."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("stamp_t1")
    store = Store("stamp_t1")

    file_path = "/tmp/test_e2e.py"
    content = "def foo():\n    pass\n"
    content_hash = _sha8(content)

    # Insert an assistant row with a Read tool_use block (unstamped)
    row_id = store.append_turn(
        "assistant",
        [
            {
                "type": "tool_use",
                "id": "tu-abc",
                "name": "Read",
                "input": {"file_path": file_path},
            }
        ],
    )

    # Stamp it
    result = store.stamp_read_cache(file_path, content, content_hash)
    assert result is True, "stamp_read_cache should return True on success"

    # Verify the input JSON was updated in the DB
    row = store.conn.execute(
        "SELECT content FROM turns WHERE id=?", (row_id,)
    ).fetchone()
    blocks = json.loads(row[0])
    inp = blocks[0]["input"]
    assert inp["_cached_content"] == content
    assert inp["_cached_hash"] == content_hash
    assert inp["file_path"] == file_path  # original field preserved

    store.close()


def test_stamp_read_cache_returns_false_when_no_match(home):
    """stamp_read_cache returns False if there is no matching Read tool_use."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("stamp_t2")
    store = Store("stamp_t2")

    # No rows at all
    result = store.stamp_read_cache("/nonexistent.py", "content", "deadbeef")
    assert result is False

    store.close()


def test_stamp_read_cache_matches_most_recent(home):
    """stamp_read_cache stamps the MOST RECENT matching tool_use block."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("stamp_t3")
    store = Store("stamp_t3")

    fp = "/tmp/myfile.py"
    # Two rows reading the same file
    row1 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file_path": fp}}],
    )
    row2 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "tu2", "name": "Read", "input": {"file_path": fp}}],
    )

    content = "hello"
    store.stamp_read_cache(fp, content, _sha8(content))

    # row2 (most recent) should be stamped; row1 should not
    def _get_inp(rid):
        r = store.conn.execute("SELECT content FROM turns WHERE id=?", (rid,)).fetchone()
        return json.loads(r[0])[0]["input"]

    assert _get_inp(row2).get("_cached_content") == content
    assert "_cached_content" not in _get_inp(row1)

    store.close()


# ---------------------------------------------------------------------------
# Step 2 unit tests — compress_repeated_reads on stamped tool_use blocks
# ---------------------------------------------------------------------------


def test_compress_repeated_reads_operates_on_tool_use_blocks(home):
    """compress_repeated_reads must walk tool_use blocks (not tool_result rows).

    Two stamped Reads of the same file → first gets stubbed, second stays full.
    """
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crr_tu_t")
    store = Store("crr_tu_t")

    fp = "/tmp/target.py"
    v1 = "def bar():\n    return 1\n" + "x" * 300
    v2 = "def bar():\n    return 2\n" + "x" * 300

    row1 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp, "_cached_content": v1, "_cached_hash": _sha8(v1)}}],
    )
    row2 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t2", "name": "Read",
          "input": {"file_path": fp, "_cached_content": v2, "_cached_hash": _sha8(v2)}}],
    )

    result = store.compress_repeated_reads()
    assert result["reads_compressed"] == 1

    rows = {r["id"]: r for r in store.window() if r["role"] == "assistant"}

    # row2 stays full
    assert rows[row2]["content"][0]["input"]["_cached_content"] == v2
    # row1 is stubbed
    stub = rows[row1]["content"][0]["input"]["_cached_content"]
    assert stub.startswith("(historical state"), f"Expected diff stub, got: {stub[:80]}"

    store.close()


def test_compress_unchanged_uses_pointer_stub(home):
    """Two stamped Reads with same content → older gets 'content unchanged' pointer stub."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crr_ptr_t")
    store = Store("crr_ptr_t")

    fp = "/tmp/stable.py"
    content = "def stable():\n    pass\n" + "s" * 300
    h = _sha8(content)

    row1 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp, "_cached_content": content, "_cached_hash": h}}],
    )
    row2 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t2", "name": "Read",
          "input": {"file_path": fp, "_cached_content": content, "_cached_hash": h}}],
    )

    result = store.compress_repeated_reads()
    assert result["reads_compressed"] == 1

    rows = {r["id"]: r for r in store.window() if r["role"] == "assistant"}
    stub = rows[row1]["content"][0]["input"]["_cached_content"]
    assert "content unchanged" in stub, f"Expected 'content unchanged', got: {stub[:120]}"
    assert str(row2) in stub, "Pointer stub should reference the last turn id"

    # row2 untouched
    assert rows[row2]["content"][0]["input"]["_cached_content"] == content

    store.close()


def test_compress_changed_uses_diff(home):
    """Two stamped Reads with different content → older gets unified diff stub."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crr_diff2_t")
    store = Store("crr_diff2_t")

    fp = "/tmp/changing.py"
    v1 = "def foo():\n    return 1\n" + "a" * 250
    v2 = "def foo():\n    return 99\n" + "a" * 250

    row1 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp, "_cached_content": v1, "_cached_hash": _sha8(v1)}}],
    )
    row2 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t2", "name": "Read",
          "input": {"file_path": fp, "_cached_content": v2, "_cached_hash": _sha8(v2)}}],
    )

    result = store.compress_repeated_reads()
    assert result["reads_compressed"] == 1

    rows = {r["id"]: r for r in store.window() if r["role"] == "assistant"}
    stub = rows[row1]["content"][0]["input"]["_cached_content"]
    assert stub.startswith("(historical state"), f"Diff stub should start with header, got: {stub[:80]}"
    # Diff lines should appear (the changed line)
    assert "return 1" in stub or "@@" in stub, f"Expected diff content in stub: {stub[:200]}"

    store.close()


def test_no_lineterm_in_diff_headers(home):
    """Regression guard: without lineterm='', diff headers are on separate lines.

    The v0.6.0 bug used lineterm='' which caused '---' / '+++' / '@@' lines
    to be concatenated with adjacent content lines. The fix removes lineterm=''
    so standard newline-terminated lines are emitted.
    """
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crr_lineterm_t")
    store = Store("crr_lineterm_t")

    fp = "/tmp/lineterm_test.py"
    v1 = "line one\nline two\n" + "z" * 250
    v2 = "line one\nline TWO\n" + "z" * 250  # changed second line

    store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp, "_cached_content": v1, "_cached_hash": _sha8(v1)}}],
    )
    store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t2", "name": "Read",
          "input": {"file_path": fp, "_cached_content": v2, "_cached_hash": _sha8(v2)}}],
    )

    store.compress_repeated_reads()

    rows = store.window()
    asst = [r for r in rows if r["role"] == "assistant"]
    # row1 is the compressed one
    stub = asst[0]["content"][0]["input"]["_cached_content"]

    # The stub contains a diff — verify header lines are separated by newlines
    # (not concatenated, which would be the lineterm='' bug)
    lines = stub.splitlines()
    diff_lines = [l for l in lines if l.startswith("---") or l.startswith("+++") or l.startswith("@@")]
    assert len(diff_lines) >= 1, f"Expected diff header lines in stub, got: {stub[:200]}"
    # Each header line must be its own line (not part of a longer concatenated line)
    for dl in diff_lines:
        # If lineterm='' was used, header content would bleed into next line without \n
        # so the line would contain the header AND the next line's content.
        # With correct behavior, the line is just the header.
        assert len(dl) < 200, f"Diff header line suspiciously long (lineterm bug?): {dl!r}"

    store.close()


# ---------------------------------------------------------------------------
# Step 3 unit tests — collect_read_stats walks tool_use blocks
# ---------------------------------------------------------------------------


def test_collect_read_stats_walks_tool_use_blocks(home):
    """collect_read_stats returns stats from stamped tool_use blocks (not tool_result rows)."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crs_tu_t")
    store = Store("crs_tu_t")

    fp1 = "/tmp/alpha.py"
    fp2 = "/tmp/beta.py"
    c1 = "# alpha\n" + "a" * 100
    c2 = "# beta\n" + "b" * 200

    row1 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp1, "_cached_content": c1, "_cached_hash": _sha8(c1)}}],
    )
    row2 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t2", "name": "Read",
          "input": {"file_path": fp2, "_cached_content": c2, "_cached_hash": _sha8(c2)}}],
    )

    stats = store.collect_read_stats()

    assert fp1 in stats
    assert stats[fp1]["last_read_turn"] == row1
    assert stats[fp1]["last_hash"] == _sha8(c1)
    assert stats[fp1]["last_size"] == len(c1)

    assert fp2 in stats
    assert stats[fp2]["last_read_turn"] == row2
    assert stats[fp2]["last_hash"] == _sha8(c2)

    store.close()


def test_collect_read_stats_returns_most_recent(home):
    """collect_read_stats returns data from the MOST RECENT read per file."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crs_recent_t")
    store = Store("crs_recent_t")

    fp = "/tmp/multi.py"
    c1 = "v1\n" + "a" * 100
    c2 = "v2\n" + "b" * 100

    store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp, "_cached_content": c1, "_cached_hash": _sha8(c1)}}],
    )
    row2 = store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t2", "name": "Read",
          "input": {"file_path": fp, "_cached_content": c2, "_cached_hash": _sha8(c2)}}],
    )

    stats = store.collect_read_stats()
    assert stats[fp]["last_read_turn"] == row2
    assert stats[fp]["last_hash"] == _sha8(c2)
    assert stats[fp]["last_size"] == len(c2)

    store.close()


def test_collect_read_stats_ignores_unstamped_tool_use(home):
    """collect_read_stats ignores Read tool_use blocks without _cached_content."""
    from mnemara import config
    from mnemara.store import Store

    config.init_instance("crs_unstamped_t")
    store = Store("crs_unstamped_t")

    fp = "/tmp/unstamped.py"
    # Unstamped Read (as if stamp_read_cache was never called)
    store.append_turn(
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "Read",
          "input": {"file_path": fp}}],
    )

    stats = store.collect_read_stats()
    assert fp not in stats, "Unstamped tool_use should not appear in collect_read_stats"

    store.close()


# ---------------------------------------------------------------------------
# End-to-end test — exercises real turn_async pipeline
# ---------------------------------------------------------------------------


def test_e2e_stamp_and_compress_via_turn_async(home, monkeypatch):
    """End-to-end: turn_async pipeline stamps _cached_content into the
    persisted tool_use block when a Read tool_result arrives in UserMessage.
    A second mocked turn reading the same file triggers compression.

    This exercises the REAL turn_async code path (not the fabricated
    _make_read_pair substrate that bypasses the agent loop).
    """
    from mnemara import config, agent as agent_mod
    from mnemara.config import Config
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner
    from claude_agent_sdk import (
        AssistantMessage, UserMessage, ResultMessage,
        ToolUseBlock, ToolResultBlock, TextBlock,
    )

    config.init_instance("e2e_stamp_t")
    cfg = config.load("e2e_stamp_t")
    cfg.stream = False
    cfg.compress_repeated_reads = True
    config.save("e2e_stamp_t", cfg)

    store = Store("e2e_stamp_t")
    perms = PermissionStore("e2e_stamp_t")
    runner = ToolRunner("e2e_stamp_t", cfg, perms, prompt=lambda t, x: "deny")

    file_path = "/tmp/test_e2e_stamp.py"
    file_content_v1 = "def foo():\n    pass\n" + "x" * 300
    file_content_v2 = "def foo():\n    return 42\n" + "x" * 300

    # ------------------------------------------------------------------
    # Turn 1: model issues Read, SDK returns tool_result with v1 content
    # ------------------------------------------------------------------

    tu_id_1 = "tool-use-id-read-1"

    async def _fake_query_turn1(*, prompt, options, **kw):
        # AssistantMessage with Read tool_use block
        yield AssistantMessage(
            content=[ToolUseBlock(
                id=tu_id_1,
                name="Read",
                input={"file_path": file_path},
            )],
            model="test",
        )
        # UserMessage with tool_result (content stamped by SDK/tool handler)
        yield UserMessage(
            content=[ToolResultBlock(
                tool_use_id=tu_id_1,
                content=file_content_v1,
                is_error=False,
            )],
        )
        # Final result
        yield ResultMessage(
            subtype="success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            num_turns=1,
            session_id="test",
            total_cost_usd=0.0,
            usage={"input_tokens": 10, "output_tokens": 5},
            result=None,
        )

    monkeypatch.setattr(agent_mod, "query", _fake_query_turn1)

    session = agent_mod.AgentSession(cfg, store, runner)
    asyncio.run(session.turn_async("please read the file"))

    # After turn 1: verify _cached_content was stamped into the tool_use block
    rows = store.window()
    asst_rows = [r for r in rows if r["role"] == "assistant"]
    read_block = None
    for row in asst_rows:
        for blk in row["content"]:
            if (isinstance(blk, dict)
                    and blk.get("type") == "tool_use"
                    and blk.get("name") == "Read"):
                read_block = blk
    assert read_block is not None, "No Read tool_use block found in assistant rows"
    assert "_cached_content" in read_block["input"], (
        f"_cached_content not stamped into Read block input: {read_block['input']}"
    )
    assert read_block["input"]["_cached_content"] == file_content_v1
    assert read_block["input"]["_cached_hash"] == _sha8(file_content_v1)

    # ------------------------------------------------------------------
    # Turn 2: model reads the same file again (v2 content)
    # compress_repeated_reads fires automatically (cfg.compress_repeated_reads=True)
    # ------------------------------------------------------------------

    tu_id_2 = "tool-use-id-read-2"

    async def _fake_query_turn2(*, prompt, options, **kw):
        yield AssistantMessage(
            content=[ToolUseBlock(
                id=tu_id_2,
                name="Read",
                input={"file_path": file_path},
            )],
            model="test",
        )
        yield UserMessage(
            content=[ToolResultBlock(
                tool_use_id=tu_id_2,
                content=file_content_v2,
                is_error=False,
            )],
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            num_turns=1,
            session_id="test",
            total_cost_usd=0.0,
            usage={"input_tokens": 12, "output_tokens": 6},
            result=None,
        )

    monkeypatch.setattr(agent_mod, "query", _fake_query_turn2)
    asyncio.run(session.turn_async("read the file again"))

    # After turn 2: verify compression happened on the first Read block
    rows = store.window()
    asst_rows = [r for r in rows if r["role"] == "assistant"]

    # Collect all Read tool_use blocks in order
    read_blocks = []
    for row in sorted(asst_rows, key=lambda r: r["id"]):
        for blk in row["content"]:
            if (isinstance(blk, dict)
                    and blk.get("type") == "tool_use"
                    and blk.get("name") == "Read"):
                read_blocks.append(blk)

    assert len(read_blocks) == 2, f"Expected 2 Read blocks, got {len(read_blocks)}"

    first_cached = read_blocks[0]["input"].get("_cached_content", "")
    second_cached = read_blocks[1]["input"].get("_cached_content", "")

    # First block should be compressed (stubbed)
    assert (
        first_cached.startswith("(see turn") or first_cached.startswith("(historical state")
    ), f"First Read block should be stubbed after compression, got: {first_cached[:120]}"

    # Second (latest) block should be full content
    assert second_cached == file_content_v2, (
        f"Latest Read block should remain full, got: {second_cached[:80]}"
    )

    store.close()
