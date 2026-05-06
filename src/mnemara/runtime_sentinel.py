"""Runtime-enforced sentinel: polling detection via SDK hook events.

Instances of RuntimeSentinel are per-AgentSession. They consume SDK hook
events (PreToolUseHookInput dicts from HookEventMessage.data) and flag
repeating-poll patterns before the model wastes another round-trip.

Polling rule (v0.3.4):
  If the same (tool_name, hash(args)) appears 3 or more times in the
  sliding window of the last 5 PreToolUse events, should_halt() returns
  a human-readable halt reason. The check uses only the five most-recent
  PreToolUse events — older events outside the window don't contribute.

Stateless across sessions: instantiate once per AgentSession; do not
reuse across sessions. This keeps the detection window per-session and
avoids cross-session false positives on legitimately identical tool calls
(e.g. `rag_query("what is X")` being equally useful in separate sessions).
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any, Optional


def _stable_hash(args: Any) -> int:
    """Return a stable hash of an args dict (or any JSON-serialisable value).

    We hash the JSON-serialised representation so that dicts with the same
    keys/values always produce the same fingerprint, regardless of insertion
    order, across Python runs within a single process.  Cross-process
    stability is not required — the sentinel's window is per-session.
    """
    try:
        serialised = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialised = repr(args)
    return hash(serialised)


# How many recent PreToolUse events we keep in the sliding window.
_WINDOW = 5
# How many identical (tool, args_hash) occurrences within the window triggers
# a halt.
_THRESHOLD = 3


class RuntimeSentinel:
    """Per-AgentSession polling detector.

    Feed SDK hook event payloads to ``observe(event)`` as they arrive in the
    stream.  Call ``should_halt()`` after each event to check whether a halt
    reason has accumulated.  If it has, inject a synthetic system notice and
    stop dispatching tools for the current turn.

    The detector resets its window automatically at session boundaries because
    each AgentSession gets its own RuntimeSentinel instance (see agent.py).
    """

    def __init__(self) -> None:
        # Sliding window of (tool_name, args_hash) tuples for the last
        # _WINDOW PreToolUse events.
        self._window: deque[tuple[str, int]] = deque(maxlen=_WINDOW)
        # Accumulated halt reason, if any.
        self._halt_reason: Optional[str] = None

    # ---------------------------------------------------------------- public

    def observe(self, event: Any) -> None:
        """Feed one hook event payload.

        ``event`` is the value of ``HookEventMessage.data`` (a plain dict)
        or the full ``HookEventMessage`` object — we handle both.  Only
        ``PreToolUse`` events update the window; all other event types are
        silently ignored.

        Calling ``observe`` after ``should_halt()`` has returned a reason is
        safe (it is a no-op once halted).
        """
        if self._halt_reason is not None:
            # Already halted this turn — no point accumulating further.
            return

        # Accept either a HookEventMessage object or a raw dict payload.
        hook_event_name: str = ""
        tool_name: str = ""
        tool_input: Any = {}

        if isinstance(event, dict):
            hook_event_name = str(event.get("hook_event_name", ""))
            tool_name = str(event.get("tool_name", ""))
            tool_input = event.get("tool_input", {})
        else:
            # HookEventMessage or subclass; inspect attributes.
            hook_event_name = getattr(event, "hook_event_name", "")
            data = getattr(event, "data", {}) or {}
            tool_name = str(data.get("tool_name", ""))
            tool_input = data.get("tool_input", {})

        if hook_event_name != "PreToolUse":
            return

        args_hash = _stable_hash(tool_input)
        self._window.append((tool_name, args_hash))
        self._check_window(tool_name, args_hash)

    def should_halt(self) -> Optional[str]:
        """Return a halt-reason string if polling was detected, else None.

        The returned string is intended for direct injection as a synthetic
        system notice so the model can see why tool dispatch was stopped.
        """
        return self._halt_reason

    def reset(self) -> None:
        """Clear window and halt reason.  Call at the start of each turn if
        you want per-turn windows (the current agent.py wires one sentinel
        per session, which gives a session-scoped window).
        """
        self._window.clear()
        self._halt_reason = None

    # --------------------------------------------------------------- internal

    def _check_window(self, tool_name: str, args_hash: int) -> None:
        """Scan the current window for a polling pattern and set halt reason."""
        count = sum(
            1 for (tn, ah) in self._window if tn == tool_name and ah == args_hash
        )
        if count >= _THRESHOLD:
            self._halt_reason = (
                f"polling: {tool_name} called {count} times with identical args"
            )
