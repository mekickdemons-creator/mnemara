"""Tests for RuntimeSentinel — polling detection via SDK hook events.

All tests are offline (no SDK imports, no network). We craft synthetic event
dicts that mirror what HookEventMessage.data looks like for PreToolUse events
and feed them directly to the sentinel.
"""
from __future__ import annotations

import pytest

from mnemara.runtime_sentinel import RuntimeSentinel, _THRESHOLD, _WINDOW


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _pre_tool_use(tool_name: str, tool_input: dict | None = None) -> dict:
    """Return a minimal PreToolUse event payload dict."""
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input or {},
    }


def _post_tool_use(tool_name: str) -> dict:
    """Return a minimal PostToolUse event payload dict (non-PreToolUse)."""
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {},
        "tool_response": "ok",
    }


class _FakeHookEventMessage:
    """Stand-in for HookEventMessage that RuntimeSentinel handles via attrs."""

    def __init__(self, hook_event_name: str, tool_name: str = "", tool_input: dict | None = None):
        self.hook_event_name = hook_event_name
        self.data = {
            "hook_event_name": hook_event_name,
            "tool_name": tool_name,
            "tool_input": tool_input or {},
        }


# ---------------------------------------------------------------------------
# Basic polling detection — dict payload interface
# ---------------------------------------------------------------------------


def test_no_halt_below_threshold():
    """Fewer than THRESHOLD identical calls should not trigger halt."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD - 1):
        s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    assert s.should_halt() is None


def test_halt_at_threshold():
    """Exactly THRESHOLD identical calls within the window fires the halt."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    reason = s.should_halt()
    assert reason is not None
    assert "Bash" in reason
    assert str(_THRESHOLD) in reason
    assert "polling" in reason


def test_halt_above_threshold():
    """More than THRESHOLD identical calls also fires (count >= threshold)."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD + 2):
        s.observe(_pre_tool_use("Read", {"file_path": "/etc/hosts"}))
    reason = s.should_halt()
    assert reason is not None
    assert "Read" in reason


def test_no_halt_with_varying_args():
    """Different args on the same tool should never trigger the rule."""
    s = RuntimeSentinel()
    for i in range(_THRESHOLD * 2):
        s.observe(_pre_tool_use("Bash", {"command": f"ls /dir{i}"}))
    assert s.should_halt() is None


def test_no_halt_with_varying_tools():
    """Same args but different tool names — each (tool, args) pair is distinct."""
    s = RuntimeSentinel()
    tools = ["Bash", "Read", "Write", "Edit", "Bash"]
    for t in tools:
        s.observe(_pre_tool_use(t, {"command": "ls"}))
    # Only 2 Bash+ls occurrences in window, under threshold.
    assert s.should_halt() is None


def test_sliding_window_evicts_old_events():
    """Events older than the window do not contribute to the count.

    Strategy: fill the window with WINDOW unique-arg Bash calls, then add
    THRESHOLD identical calls. The window is WINDOW events; the identical
    calls must all fit within WINDOW.
    """
    s = RuntimeSentinel()
    # Push WINDOW - THRESHOLD events with distinct args to fill the buffer.
    for i in range(_WINDOW - _THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": f"unique {i}"}))
    assert s.should_halt() is None  # No pattern yet.

    # Now push exactly THRESHOLD identical events; they fill the tail of
    # the window and should trigger the halt.
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": "polling"}))
    assert s.should_halt() is not None


def test_old_identical_events_scroll_out_of_window():
    """Identical events that fall outside the window don't cause a false alarm.

    Push THRESHOLD - 1 identical events, then WINDOW identical *different*
    events to flush the old ones out, then just 1 more of the original.
    The original (threshold-1) events should be gone from the window.
    """
    s = RuntimeSentinel()
    # Push THRESHOLD-1 "target" events.
    for _ in range(_THRESHOLD - 1):
        s.observe(_pre_tool_use("Bash", {"command": "old"}))
    # Flush with distinct events to push the old ones out of the window.
    for i in range(_WINDOW):
        s.observe(_pre_tool_use("Read", {"file_path": f"/tmp/{i}"}))
    # Add one more "old" event — only 1 in the window now, not THRESHOLD.
    s.observe(_pre_tool_use("Bash", {"command": "old"}))
    assert s.should_halt() is None


# ---------------------------------------------------------------------------
# Non-PreToolUse events are silently ignored
# ---------------------------------------------------------------------------


def test_non_pre_tool_use_events_ignored():
    """PostToolUse and other event types must not affect the window."""
    s = RuntimeSentinel()
    # Fill window with PostToolUse events that mention the same tool.
    for _ in range(_THRESHOLD * 2):
        s.observe(_post_tool_use("Bash"))
    assert s.should_halt() is None


def test_mixed_event_types_only_pre_tool_use_counts():
    """Only PreToolUse events count toward the polling window."""
    s = RuntimeSentinel()
    # Interleave PostToolUse (ignored) with PreToolUse (counts).
    for i in range(_THRESHOLD * 2):
        s.observe(_post_tool_use("Bash"))
        if i < _THRESHOLD - 1:
            # Add fewer than threshold PreToolUse identical events.
            s.observe(_pre_tool_use("Bash", {"command": "polling"}))
    assert s.should_halt() is None


# ---------------------------------------------------------------------------
# Object (HookEventMessage-style) interface
# ---------------------------------------------------------------------------


def test_object_interface_triggers_halt():
    """RuntimeSentinel handles object events (HookEventMessage-like) correctly."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_FakeHookEventMessage("PreToolUse", "Bash", {"command": "ls"}))
    reason = s.should_halt()
    assert reason is not None
    assert "Bash" in reason


def test_object_interface_non_pre_ignored():
    """Non-PreToolUse HookEventMessage objects are silently ignored."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD * 2):
        s.observe(_FakeHookEventMessage("PostToolUse", "Bash", {"command": "ls"}))
    assert s.should_halt() is None


# ---------------------------------------------------------------------------
# Idempotence after halt
# ---------------------------------------------------------------------------


def test_observe_after_halt_is_noop():
    """Once halted, further observe() calls don't change the reason."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    reason1 = s.should_halt()
    assert reason1 is not None

    # Keep feeding events — reason must stay the same.
    for _ in range(10):
        s.observe(_pre_tool_use("Read", {"file_path": "/etc/hosts"}))
    reason2 = s.should_halt()
    assert reason1 == reason2


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_halt():
    """reset() should clear both the window and the halt reason."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    assert s.should_halt() is not None
    s.reset()
    assert s.should_halt() is None


def test_reset_allows_fresh_detection():
    """After reset(), detection works from scratch."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    s.reset()
    # Two more (below threshold) — no halt.
    for _ in range(_THRESHOLD - 1):
        s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    assert s.should_halt() is None
    # One more to hit threshold again.
    s.observe(_pre_tool_use("Bash", {"command": "ls"}))
    assert s.should_halt() is not None


# ---------------------------------------------------------------------------
# Halt reason content
# ---------------------------------------------------------------------------


def test_halt_reason_includes_count():
    """The halt reason must mention the repetition count at the time of halt.

    The sentinel halts as soon as it reaches _THRESHOLD identical events, so
    the count in the reason is _THRESHOLD (not _THRESHOLD+1, because
    subsequent observe() calls are no-ops after halt fires).
    """
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Read", {"file_path": "/etc/passwd"}))
    reason = s.should_halt()
    assert reason is not None
    assert str(_THRESHOLD) in reason, f"Expected count {_THRESHOLD} in: {reason!r}"


def test_halt_reason_format():
    """Smoke-test the expected reason string format."""
    s = RuntimeSentinel()
    for _ in range(_THRESHOLD):
        s.observe(_pre_tool_use("Bash", {"command": "echo hi"}))
    reason = s.should_halt()
    assert reason is not None
    # Format: "polling: <tool> called <N> times with identical args"
    assert reason.startswith("polling:")
    assert "called" in reason
    assert "identical args" in reason
