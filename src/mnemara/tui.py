"""Mnemara STABLE TUI — role doc + token-cap eviction + spinner.

Three features, nothing else:
  1. Role doc at slot 0 of every API call  (AgentSession's identity layer)
  2. Token-count eviction only             (row cap is not enforced here)
  3. Spinner during streaming turns

Slash commands: /quit, /exit, /models, /swap, /tokens, /export, /import, and /stop.
No block surgery, no pin system, no copy.
"""
from __future__ import annotations

import asyncio
import atexit
import dataclasses
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from textual import events as _txt_events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.screen import ModalScreen
    from textual.widgets import Button, Footer, Header, Input, ListItem, ListView, RichLog, Static, TextArea
    from textual.containers import Horizontal, Vertical
    _TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TEXTUAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Escape sequences for all mouse-tracking modes Mnemara may enable.
# Written to /dev/tty at process exit as a last-resort safety net in case
# Textual's driver cleanup is skipped (true SIGINT, crash in shutdown path).
# All sequences are idempotent — disabling a mode never enabled is a no-op.
_MOUSE_RESET_BYTES = (
    b"\x1b[?1000l"  # basic click tracking
    b"\x1b[?1002l"  # button-event motion  (we enable; Textual's disabler omits)
    b"\x1b[?1003l"  # any-event motion
    b"\x1b[?1005l"  # UTF-8 extended encoding
    b"\x1b[?1006l"  # SGR extended encoding
    b"\x1b[?1015l"  # urxvt extended encoding
)


def _tty_mouse_reset() -> None:
    """Atexit hook: reset all mouse-tracking modes by writing to /dev/tty."""
    try:
        fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, _MOUSE_RESET_BYTES)
        finally:
            os.close(fd)
    except Exception:
        pass


atexit.register(_tty_mouse_reset)

# ---------------------------------------------------------------------------
# Deferred imports (after _TEXTUAL_AVAILABLE guard)
# ---------------------------------------------------------------------------

from . import config as config_mod
from . import paths
from .agent import AgentSession
from .config import Config
from .logging_util import log, set_log_path
from .permissions import PermissionStore
from .store import Store
from .tools import ToolRunner, _read_skeleton


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    background: #1d2330;
    color: #e6e6e6;
}

#chatlog {
    height: 1fr;
    min-height: 5;
    border: round #3a4256;
    background: #1a1f2b;
    padding: 1 2;
    overflow-y: scroll;
    scrollbar-background: #1a1f2b;
    scrollbar-background-hover: #1a1f2b;
    scrollbar-background-active: #1a1f2b;
    scrollbar-color: #4d6fa3;
    scrollbar-color-hover: #6f9ad9;
    scrollbar-color-active: #8fb6e6;
    scrollbar-size-vertical: 2;
}

#status {
    height: 1;
    background: #11151e;
    color: #8aa1c1;
    padding: 0 1;
}

#keyhint {
    height: 1;
    background: #11151e;
    color: #636b7a;
    padding: 0 1;
}

#userinput {
    height: auto;
    min-height: 4;
    max-height: 10;
    border: round #4d6fa3;
    background: #11151e;
    color: #ffffff;
    padding: 0 1;
}

#userinput:focus {
    border: round #6f9ad9;
}

#userinput > .text-area--cursor {
    background: #6f9ad9;
    color: #11151e;
}

#userinput > .text-area--selection {
    background: #2a3f5f;
}

#btn-row {
    height: 3;
    background: #11151e;
    align: right middle;
}

#btn-send {
    background: #1e3a5f;
    color: #6f9ad9;
    border: tall #4d6fa3;
    min-width: 14;
}

#btn-send:hover {
    background: #2a4f7a;
    color: #8fb6e6;
}

#btn-send:focus {
    border: tall #6f9ad9;
}

#btn-quit {
    background: #2a1f1f;
    color: #8a6a6a;
    border: tall #5a3a3a;
    min-width: 14;
    margin-left: 1;
}

#btn-quit:hover {
    background: #3a2a2a;
    color: #c08080;
}

#btn-quit:focus {
    border: tall #a05050;
}

#btn-inbox {
    background: #1e3a5f;
    color: #6f9ad9;
    border: tall #4d6fa3;
    min-width: 16;
    margin-left: 1;
}

#btn-inbox:hover {
    background: #2a4f7a;
    color: #8fb6e6;
}

#btn-inbox:focus {
    border: tall #6f9ad9;
}

.inbox-off {
    background: #3a3a3a;
    color: #888888;
    border: tall #555555;
}

.inbox-off:hover {
    background: #444444;
    color: #aaaaaa;
}

/* ---- Role-doc editor modal ---- */
RoleDocEditorModal {
    align: center middle;
}

#role-editor-dialog {
    width: 90%;
    height: 90%;
    border: round #6f9ad9;
    background: #1d2330;
    padding: 1 2;
}

#role-editor-title {
    height: 1;
    color: #8fb6e6;
    text-align: center;
    margin-bottom: 1;
}

#role-editor-ta {
    height: 1fr;
    border: round #4d6fa3;
    background: #11151e;
    color: #ffffff;
    padding: 0 1;
}

#role-editor-ta:focus {
    border: round #6f9ad9;
}

#role-editor-btn-row {
    height: 3;
    background: #1d2330;
    align: right middle;
    margin-top: 1;
}

#btn-role-save {
    background: #1e3a5f;
    color: #6f9ad9;
    border: tall #4d6fa3;
    min-width: 14;
}

#btn-role-save:hover {
    background: #2a4f7a;
    color: #8fb6e6;
}

#btn-role-cancel {
    background: #2a1f1f;
    color: #8a6a6a;
    border: tall #5a3a3a;
    min-width: 14;
    margin-left: 1;
}

#btn-role-cancel:hover {
    background: #3a2a2a;
    color: #c08080;
}

#btn-role-paste, #btn-role-copy {
    background: #1e2a1e;
    color: #6a9a6a;
    border: tall #3a5a3a;
    min-width: 12;
    margin-right: 1;
}

#btn-role-paste:hover, #btn-role-copy:hover {
    background: #2a3a2a;
    color: #8aba8a;
}

#btn-role {
    background: #1e3a5f;
    color: #6f9ad9;
    border: tall #4d6fa3;
    min-width: 12;
    margin-left: 1;
}

#btn-role:hover {
    background: #2a4f7a;
    color: #8fb6e6;
}

#btn-role:focus {
    border: tall #6f9ad9;
}

/* ---- Context viewer modal ---- */
ContextViewerModal {
    align: center middle;
}

#ctx-dialog {
    width: 95%;
    height: 95%;
    border: round #6f9ad9;
    background: #1d2330;
    padding: 1 2;
}

#ctx-header {
    height: 1;
    color: #8fb6e6;
    text-align: center;
    margin-bottom: 1;
}

#ctx-filter-row {
    height: 3;
    background: #1d2330;
    margin-bottom: 1;
}

#ctx-filter-input {
    width: 1fr;
    border: tall #4d6fa3;
    background: #11151e;
    color: #ffffff;
}

#ctx-panels {
    height: 1fr;
}

#ctx-list-panel {
    width: 2fr;
    border: round #4d6fa3;
    margin-right: 1;
}

#ctx-list {
    height: 1fr;
    background: #11151e;
}

#ctx-detail-panel {
    width: 3fr;
    border: round #4d6fa3;
}

#ctx-detail-ta {
    height: 1fr;
    background: #11151e;
    color: #d0d8e8;
    padding: 0 1;
}

#ctx-action-row {
    height: 3;
    background: #1d2330;
    align: left middle;
    margin-top: 1;
}

#btn-ctx-evict {
    background: #2a1f1f;
    color: #c08080;
    border: tall #5a3a3a;
    min-width: 10;
}

#btn-ctx-pin {
    background: #1e3a5f;
    color: #6f9ad9;
    border: tall #4d6fa3;
    min-width: 10;
    margin-left: 1;
}

#btn-ctx-unpin {
    background: #1a2a1a;
    color: #6fad6f;
    border: tall #3a6f3a;
    min-width: 10;
    margin-left: 1;
}

#btn-ctx-close {
    background: #2a2a2a;
    color: #888888;
    border: tall #444444;
    min-width: 14;
    margin-left: 1;
}

#btn-ctx-evict:hover { background: #3a2a2a; }
#btn-ctx-pin:hover   { background: #2a4f7a; }
#btn-ctx-unpin:hover { background: #2a3a2a; }
#btn-ctx-close:hover { background: #333333; }

#ctx-detail-btn-row {
    height: 3;
    background: #11151e;
    align: left middle;
    margin-top: 0;
}

#btn-ctx-paste, #btn-ctx-copy {
    background: #1e2a1e;
    color: #6a9a6a;
    border: tall #3a5a3a;
    min-width: 10;
    margin-right: 1;
}

#btn-ctx-paste:hover, #btn-ctx-copy:hover {
    background: #2a3a2a;
    color: #8aba8a;
}

#btn-ctx-save-edit {
    background: #1e3a1e;
    color: #6fad6f;
    border: tall #3a6f3a;
    min-width: 14;
}

#btn-ctx-save-edit:hover {
    background: #2a4f2a;
    color: #8fc88f;
}

#btn-context {
    background: #1a2a1a;
    color: #6fad6f;
    border: tall #3a6f3a;
    min-width: 14;
    margin-left: 1;
}

#btn-context:hover {
    background: #2a3a2a;
    color: #8fc88f;
}

#btn-context:focus {
    border: tall #6fad6f;
}

#btn-compress {
    background: #2a1f35;
    color: #a080d0;
    border: tall #5a3a80;
    min-width: 14;
    margin-left: 1;
}

#btn-compress:hover {
    background: #3a2a4a;
    color: #c0a0f0;
}

#btn-compress:focus {
    border: tall #a080d0;
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_size(s: str) -> int:
    """Parse a human-readable integer size: '500', '500k', '1m', '1_000_000'.

    Used by /tokens. Returns the parsed int. Raises ValueError on bad input.
    Suffixes are case-insensitive: k=1_000, m=1_000_000.
    Underscores and commas in the digit portion are ignored.
    """
    if not s:
        raise ValueError("empty value")
    raw = s.strip().lower().replace("_", "").replace(",", "")
    if not raw:
        raise ValueError("empty value")
    mult = 1
    if raw.endswith("k"):
        mult = 1_000
        raw = raw[:-1]
    elif raw.endswith("m"):
        mult = 1_000_000
        raw = raw[:-1]
    if not raw or not raw.lstrip("-").isdigit():
        raise ValueError("not a number")
    return int(raw) * mult


def _is_tty() -> bool:
    import sys
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _short(d: Any, n: int = 80) -> str:
    import json as _json
    s = _json.dumps(d, default=str) if not isinstance(d, str) else d
    return s if len(s) <= n else s[: n - 1] + "…"


def _flatten_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(p for p in parts if p)


def _render_full_export(
    rows: list[dict[str, Any]],
    instance: str,
    cfg_json: str = "",
    role_doc_text: str = "",
) -> str:
    """Render a full round-trippable export with config, role_doc, and turns.

    Sections are delimited by <!-- mnemara:begin:NAME --> / <!-- mnemara:end:NAME -->
    markers so _parse_export_sections() can extract them unambiguously.
    """
    ts = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        f"# Mnemara Full Export: {instance}",
        "",
        "- mnemara-export-version: 1",
        f"- exported_at: {ts}",
        f"- instance: {instance}",
        f"- turns: {len(rows)}",
        "",
    ]

    if cfg_json:
        lines += [
            "<!-- mnemara:begin:config -->",
            "```json",
            cfg_json,
            "```",
            "<!-- mnemara:end:config -->",
            "",
        ]

    if role_doc_text:
        lines += [
            "<!-- mnemara:begin:role_doc -->",
            role_doc_text.rstrip(),
            "",
            "<!-- mnemara:end:role_doc -->",
            "",
        ]

    lines.append("<!-- mnemara:begin:turns -->")
    for row in rows:
        role = str(row.get("role", "unknown"))
        row_ts = str(row.get("ts", ""))
        lines.append(f"## {role} {row.get('id', '')}")
        if row_ts:
            lines.append(f"_ts: {row_ts}_")
            lines.append("")
        text = _flatten_text_blocks(row.get("content", ""))
        lines.append(text or "(no text content)")
        lines.append("")
    lines.append("<!-- mnemara:end:turns -->")

    return "\n".join(lines).rstrip() + "\n"


# Compiled once at import time; used by both export and import paths.
_SECTION_RE = re.compile(
    r"<!-- mnemara:begin:(\w+) -->\n(.*?)<!-- mnemara:end:\1 -->",
    re.DOTALL,
)

# Matches "## role id" turn headers in the turns section.
_TURN_HEADER_RE = re.compile(r"^## (user|assistant|system|tool)\s+(\d+)", re.MULTILINE)


def _parse_export_sections(text: str) -> dict[str, str]:
    """Return {section_name: raw_content} for every section in an export file."""
    return {m.group(1): m.group(2) for m in _SECTION_RE.finditer(text)}


def _parse_export_turns(turns_section: str) -> list[dict[str, Any]]:
    """Parse the turns section back into [{role, content}] dicts.

    Original store IDs are discarded; the store assigns new IDs on import.
    Timestamps from the export are preserved in the content header comment
    but store.append_turn() stamps a new ts at insertion time.
    """
    turns: list[dict[str, Any]] = []
    # split() with 2 capture groups gives:
    #   [preamble, role0, id0, body0, role1, id1, body1, ...]
    blocks = _TURN_HEADER_RE.split(turns_section)
    i = 1  # skip preamble at blocks[0]
    while i + 2 < len(blocks):
        role = blocks[i].strip()
        # blocks[i+1] is the original row id — ignored on import
        body = blocks[i + 2].strip()
        # Strip leading _ts: ..._ line if present (metadata, not content)
        body = re.sub(r"^_ts:[^\n]*_\n?", "", body).strip()
        if not body:
            body = "(no text content)"
        turns.append({"role": role, "content": body})
        i += 3
    return turns


# ---------------------------------------------------------------------------
# Input widget — multi-line TextArea subclass
# ---------------------------------------------------------------------------


class _UserTextArea(TextArea):
    """Multi-line plain-text prompt area for the Mnemara panel.

    Enter inserts a newline.  Ctrl+S submits.  Multi-line paste works
    natively — no collapsing needed since TextArea handles newlines correctly.

    Tab moves focus out of the input rather than inserting whitespace
    (tab_behavior="focus"); use Shift+Tab to move backwards.
    """

    BINDINGS = [
        Binding("ctrl+s",     "app.submit_prompt",  "Send",        show=True),
        # Escape declared as a Binding for documentation, but the actual
        # interception happens in _on_key below — TextArea swallows Escape
        # at the key-event level before any Binding lookup runs.
        Binding("escape",     "app.clear_input",    "Clear input", show=False, priority=True),
        # Delegate page scroll to App so the chatlog scrolls even when input
        # has focus.  Without these, TextArea's own scroll handling intercepts
        # the keys and the chatlog never receives them.
        Binding("pageup",   "app.scroll_log_up",   show=False),
        Binding("pagedown", "app.scroll_log_down",  show=False),
    ]

    async def _on_key(self, event) -> None:  # type: ignore[override]
        """Intercept submit and clear keys at the key-event level.

        TextArea's internal key handler runs before the Binding system,
        so bindings with app.* actions are unreliable — the widget may
        consume the key first.  We catch Ctrl+S and Escape here instead
        and forward them to App actions manually.
        """
        if event.key == "ctrl+s":
            event.prevent_default()
            event.stop()
            await self.app.run_action("submit_prompt")
            return
        if event.key == "escape":
            event.prevent_default()
            event.stop()
            await self.app.run_action("clear_input")
            return


# ---------------------------------------------------------------------------
# Role-doc editor modal
# ---------------------------------------------------------------------------


class RoleDocEditorModal(ModalScreen):  # type: ignore[misc]
    """Full-screen modal overlay for editing the instance role doc.

    Pushed onto the screen stack by MnemaraTUI when the user runs
    /role_doc or clicks the [📄 Role] button.  The TextArea is pre-populated
    with the current on-disk role doc content.  Ctrl+S / Save writes it back
    and dismisses; Escape / Cancel dismisses without writing.

    The role doc is re-read from disk at every API call, so changes take
    effect immediately on the next turn — no restart required.
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, role_doc_path: str, initial_content: str) -> None:
        super().__init__()
        self._role_doc_path = role_doc_path
        self._initial_content = initial_content

    def compose(self) -> "ComposeResult":
        short_path = self._role_doc_path or "(no role doc configured)"
        with Vertical(id="role-editor-dialog"):
            yield Static(
                f"[bold]Role doc editor[/bold] — [dim]{short_path}[/dim]  "
                "[dim]⌃C copy · ⌃X cut · ⌃V paste[/dim]",
                id="role-editor-title",
            )
            yield TextArea(
                self._initial_content,
                language=None,
                show_line_numbers=True,
                tab_behavior="indent",
                soft_wrap=True,
                id="role-editor-ta",
            )
            with Horizontal(id="role-editor-btn-row"):
                yield Button("📋 Paste", id="btn-role-paste")
                yield Button("⎘ Copy", id="btn-role-copy")
                yield Button("Save  ⌃S", id="btn-role-save")
                yield Button("Cancel  Esc", id="btn-role-cancel")

    def on_mount(self) -> None:
        self.query_one("#role-editor-ta", TextArea).focus()

    async def _on_key(self, event: "Key") -> None:  # type: ignore[name-defined]
        """Intercept clipboard keys before app-level priority bindings fire.

        The app has priority=True on ctrl+c (quit) and ctrl+v (paste via
        pyperclip).  Inside the role doc editor those keys must behave as
        clipboard operations within the TextArea instead.
        """
        ta = self.query_one("#role-editor-ta", TextArea)
        if event.key == "ctrl+c":
            event.prevent_default()
            event.stop()
            try:
                import pyperclip  # type: ignore[import]
                if ta.selected_text:
                    pyperclip.copy(ta.selected_text)
            except Exception:
                pass
        elif event.key == "ctrl+x":
            event.prevent_default()
            event.stop()
            try:
                import pyperclip  # type: ignore[import]
                if ta.selected_text:
                    pyperclip.copy(ta.selected_text)
                    ta.insert("")  # replaces selection with empty → cut
            except Exception:
                pass
        elif event.key == "ctrl+v":
            event.prevent_default()
            event.stop()
            try:
                import pyperclip  # type: ignore[import]
                text = pyperclip.paste()
                if text:
                    ta.insert(text)
            except Exception:
                pass

    def on_button_pressed(self, event: "Button.Pressed") -> None:
        if event.button.id == "btn-role-save":
            self.action_save()
        elif event.button.id == "btn-role-cancel":
            self.action_cancel()
        elif event.button.id == "btn-role-paste":
            try:
                import pyperclip  # type: ignore[import]
                text = pyperclip.paste()
                if text:
                    self.query_one("#role-editor-ta", TextArea).insert(text)
            except Exception:
                pass
        elif event.button.id == "btn-role-copy":
            try:
                import pyperclip  # type: ignore[import]
                ta = self.query_one("#role-editor-ta", TextArea)
                pyperclip.copy(ta.selected_text or ta.text)
            except Exception:
                pass

    def action_save(self) -> None:
        """Write edited content to disk and dismiss."""
        ta = self.query_one("#role-editor-ta", TextArea)
        new_text = ta.text
        if not self._role_doc_path:
            self.dismiss({"saved": False, "error": "no role_doc_path configured"})
            return
        try:
            Path(self._role_doc_path).write_text(new_text, encoding="utf-8")
            self.dismiss({"saved": True, "path": self._role_doc_path, "dirty": new_text != self._initial_content})
        except Exception as exc:
            self.dismiss({"saved": False, "error": str(exc)})

    def action_cancel(self) -> None:
        """Dismiss without saving."""
        ta = self.query_one("#role-editor-ta", TextArea)
        dirty = ta.text != self._initial_content
        self.dismiss({"saved": False, "cancelled": True, "dirty": dirty})


class ContextViewerModal(ModalScreen):  # type: ignore[misc]
    """Full-screen modal overlay for viewing and managing the rolling-window context.

    Pushed onto the screen stack by MnemaraTUI when the user runs /context or
    clicks the [💬 Context] button.

    Left panel: scrollable list of turns (most recent first) with row_id, role,
    timestamp, and summary. Pin marker (📌) shown for pinned rows.

    Right panel: read-only TextArea showing the full content of the selected turn.

    Actions: Evict (removes from window), Pin / Unpin, Close.
    Filter: live text filter + role filter buttons (All / User / Asst).
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
    ]

    def __init__(self, store: "Store", instance: str) -> None:  # type: ignore[name-defined]
        super().__init__()
        self._store = store
        self._instance = instance
        self._all_rows: list[dict] = []
        self._filtered_rows: list[dict] = []
        self._selected_idx: int = -1
        self._role_filter: str = ""

    def compose(self) -> "ComposeResult":
        result = self._store.list_window(limit=200)
        self._all_rows = result.get("rows", [])
        self._filtered_rows = list(self._all_rows)
        total = result.get("total", 0)

        with Vertical(id="ctx-dialog"):
            yield Static(
                f"[bold]Context viewer[/bold] — [dim]{self._instance}[/dim] "
                f"([dim]{total} turns total[/dim])",
                id="ctx-header",
            )
            with Horizontal(id="ctx-filter-row"):
                yield Input(placeholder="filter turns…", id="ctx-filter-input")
                yield Button("All", id="btn-ctx-role-all")
                yield Button("User", id="btn-ctx-role-user")
                yield Button("Asst", id="btn-ctx-role-asst")
            with Horizontal(id="ctx-panels"):
                with Vertical(id="ctx-list-panel"):
                    yield ListView(id="ctx-list")
                with Vertical(id="ctx-detail-panel"):
                    yield TextArea(
                        "← select a turn to view its full content",
                        id="ctx-detail-ta",
                    )
                    with Horizontal(id="ctx-detail-btn-row"):
                        yield Button("📋 Paste", id="btn-ctx-paste")
                        yield Button("⎘ Copy", id="btn-ctx-copy")
                        yield Button("💾 Save edit", id="btn-ctx-save-edit")
            with Horizontal(id="ctx-action-row"):
                yield Button("Evict", id="btn-ctx-evict")
                yield Button("📌 Pin", id="btn-ctx-pin")
                yield Button("Unpin", id="btn-ctx-unpin")
                yield Button("Close  Esc", id="btn-ctx-close")

    def on_mount(self) -> None:
        self._rebuild_list()
        self.query_one("#ctx-list", ListView).focus()

    async def _on_key(self, event: "Key") -> None:  # type: ignore[name-defined]
        """Intercept ctrl+c to copy selected text from the detail panel.

        The app-level ctrl+c binding has priority=True and would trigger quit.
        In the context viewer ctrl+c should copy whatever text is selected in
        the read-only detail TextArea instead.
        """
        if event.key == "ctrl+c":
            event.prevent_default()
            event.stop()
            try:
                import pyperclip  # type: ignore[import]
                ta = self.query_one("#ctx-detail-ta", TextArea)
                if ta.selected_text:
                    pyperclip.copy(ta.selected_text)
            except Exception:
                pass

    # ---------------------------------------------------------------- list helpers

    def _row_label(self, row: dict) -> str:
        pin = "📌 " if row.get("pin_label") else "   "
        ts = str(row.get("timestamp", ""))[:16].replace("T", " ")
        role = (row.get("role") or "?")[:4]
        summary = (row.get("summary") or "")[:38]
        return f"{pin}{row['row_id']:>5} · {role} · {ts} · {summary}"

    def _rebuild_list(self) -> None:
        try:
            lv = self.query_one("#ctx-list", ListView)
            lv.clear()
            for row in self._filtered_rows:
                lv.append(ListItem(Static(self._row_label(row))))
            self._selected_idx = -1
            ta = self.query_one("#ctx-detail-ta", TextArea)
            ta.load_text("← select a turn to view its full content")
        except Exception:
            # Modal not yet mounted — skip UI update; data is already in _filtered_rows.
            pass

    def _apply_filters(self) -> None:
        try:
            text_filter = self.query_one("#ctx-filter-input", Input).value.lower()
        except Exception:
            text_filter = ""
        rows = self._all_rows
        if self._role_filter:
            rows = [r for r in rows if r.get("role", "") == self._role_filter]
        if text_filter:
            rows = [r for r in rows if text_filter in r.get("summary", "").lower()
                    or text_filter in str(r.get("row_id", ""))]
        self._filtered_rows = rows
        self._rebuild_list()

    def _fmt_full_content(self, full: dict) -> str:
        """Render a full turn dict as readable text for the detail panel."""
        hdr = (
            f"Turn {full['id']} · {full.get('role', '?')} · {full.get('ts', '')}"
        )
        pin = full.get("pin_label")
        if pin:
            hdr += f"  [pinned: {pin}]"
        sep = "─" * 50
        content = full.get("content", "")
        if isinstance(content, str):
            body = content
        elif isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type", "")
                if btype == "text":
                    parts.append(b.get("text", ""))
                elif btype == "tool_use":
                    inp = b.get("input", {})
                    import json as _j
                    inp_str = _j.dumps(inp, indent=2) if isinstance(inp, dict) else str(inp)
                    parts.append(f"[tool_use: {b.get('name', '?')}]\n{inp_str}")
                elif btype == "thinking":
                    snip = (b.get("thinking") or "")[:300]
                    parts.append(f"<thinking>\n{snip}{'…' if len(b.get('thinking',''))>300 else ''}\n</thinking>")
            body = "\n\n".join(p for p in parts if p)
        else:
            body = str(content)
        return f"{hdr}\n{sep}\n{body}"

    # ---------------------------------------------------------------- events

    def on_input_changed(self, event: "Input.Changed") -> None:  # type: ignore[name-defined]
        if event.input.id == "ctx-filter-input":
            self._apply_filters()

    def on_list_view_highlighted(self, event: "ListView.Highlighted") -> None:  # type: ignore[name-defined]
        if event.list_view.id != "ctx-list":
            return
        idx = event.list_view.index
        if idx is None or idx < 0 or idx >= len(self._filtered_rows):
            return
        self._selected_idx = idx
        row = self._filtered_rows[idx]
        full = self._store.get_turn(row["row_id"])
        if full:
            ta = self.query_one("#ctx-detail-ta", TextArea)
            ta.load_text(self._fmt_full_content(full))

    def on_button_pressed(self, event: "Button.Pressed") -> None:  # type: ignore[name-defined]
        bid = event.button.id
        if bid == "btn-ctx-close":
            self.action_close()
        elif bid == "btn-ctx-role-all":
            self._role_filter = ""
            self._apply_filters()
        elif bid == "btn-ctx-role-user":
            self._role_filter = "user"
            self._apply_filters()
        elif bid == "btn-ctx-role-asst":
            self._role_filter = "assistant"
            self._apply_filters()
        elif bid == "btn-ctx-evict":
            self._do_evict()
        elif bid == "btn-ctx-pin":
            self._do_pin()
        elif bid == "btn-ctx-unpin":
            self._do_unpin()
        elif bid == "btn-ctx-paste":
            try:
                import pyperclip  # type: ignore[import]
                text = pyperclip.paste()
                if text:
                    self.query_one("#ctx-detail-ta", TextArea).insert(text)
            except Exception:
                pass
        elif bid == "btn-ctx-copy":
            try:
                import pyperclip  # type: ignore[import]
                ta = self.query_one("#ctx-detail-ta", TextArea)
                pyperclip.copy(ta.selected_text or ta.text)
            except Exception:
                pass
        elif bid == "btn-ctx-save-edit":
            self._do_save_edit()

    # ---------------------------------------------------------------- actions

    def _selected_row_id(self) -> "int | None":
        if self._selected_idx < 0 or self._selected_idx >= len(self._filtered_rows):
            return None
        return self._filtered_rows[self._selected_idx]["row_id"]

    def _do_evict(self) -> None:
        row_id = self._selected_row_id()
        if row_id is None:
            return
        self._store.evict_ids([row_id])
        self._all_rows = [r for r in self._all_rows if r["row_id"] != row_id]
        self._apply_filters()

    def _do_pin(self) -> None:
        row_id = self._selected_row_id()
        if row_id is None:
            return
        self._store.pin_row(row_id, "pinned")
        # Refresh pin_label in _all_rows
        for r in self._all_rows:
            if r["row_id"] == row_id:
                r["pin_label"] = "pinned"
                break
        self._apply_filters()

    def _do_unpin(self) -> None:
        row_id = self._selected_row_id()
        if row_id is None:
            return
        self._store.unpin_row(row_id)
        for r in self._all_rows:
            if r["row_id"] == row_id:
                r["pin_label"] = None
                break
        self._apply_filters()

    def _do_save_edit(self) -> None:
        """Write the detail panel's current text back to the selected turn row."""
        row_id = self._selected_row_id()
        if row_id is None:
            return
        try:
            ta = self.query_one("#ctx-detail-ta", TextArea)
            new_text = ta.text
            ok = self._store.update_turn_content(row_id, new_text)
            # Refresh summary in the list so it reflects any changes.
            if ok:
                for r in self._all_rows:
                    if r["row_id"] == row_id:
                        r["summary"] = (new_text.strip()[:80] or "")
                        break
                self._apply_filters()
        except Exception:
            pass

    def action_close(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class MnemaraTUI(App):  # type: ignore[misc]
    """Stable Mnemara TUI: role doc + token-cap eviction + spinner."""

    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+v", "paste", "Paste", priority=True, show=False),
        Binding("pageup",   "scroll_log_up",   "Scroll up",   show=False),
        Binding("pagedown", "scroll_log_down",  "Scroll down", show=False),
    ]

    # Safe mouse-enable sequence: basic click (1000) + button-event drag
    # (1002, needed for scrollbar) + SGR encoding (1006, safe extended coords).
    # Replaces Textual's default which includes 1003 (any-event motion: noisy)
    # and 1015 (urxvt: emits raw high bytes that crash the UTF-8 decoder on
    # wide terminals).
    _MOUSE_DISABLE_UNSAFE = "\x1b[?1003l\x1b[?1015l\x1b[?1005l"
    _MOUSE_ENABLE_SAFE    = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"

    def __init__(self, instance: str) -> None:
        super().__init__()
        self.instance = instance
        self.cfg: Config = config_mod.load(instance)
        set_log_path(paths.debug_log(instance))
        self.store = Store(instance)
        self.perms = PermissionStore(instance)
        self.runner = ToolRunner(
            instance,
            self.cfg,
            self.perms,
            prompt=self._sync_permission_prompt,
        )
        self.session = AgentSession(self.cfg, self.store, self.runner, client=None)

        self._busy = False
        self._stream_buffer = ""
        self._queued_input: str | None = None  # input received while busy; fires after turn

        # Peer-poll state (v0.11.0 autonomous inter-panel messaging).
        # _peer_poll_watermark: persisted high-watermark (max row_id seen so far).
        #   Only rows with id > watermark are delivered.  Persists across restarts so
        #   historical backlog is never re-delivered.  Loaded from disk in on_mount.
        # _peer_pending_rows: detected but not yet processed; batched into one LLM turn.
        self._peer_poll_watermark: int = 0
        self._peer_pending_rows: list[dict] = []
        self._peer_poll_timer = None  # set in on_mount when peer_poll_enabled

        # Spinner state.
        _SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        self._SPINNER_FRAMES = _SPINNER_FRAMES
        self._spinner_idx = 0
        self._spinner_was_busy = False
        self._cached_status_static = ""
        self._spinner_timer = None  # set in on_mount

    # ---------------------------------------------------------------- compose

    def compose(self) -> "ComposeResult":
        role = self.cfg.role_doc_path or "(none)"
        self.title = f"mnemara: {self.instance}"
        self.sub_title = f"model={self.cfg.model}  role={role}"
        yield Header(show_clock=False)
        yield RichLog(
            id="chatlog",
            wrap=True,
            markup=True,
            highlight=False,
            auto_scroll=True,
        )
        yield Static(self._status_text(), id="status")
        yield Static(
            "[dim]Enter for newline · Esc to clear[/dim]",
            id="keyhint",
        )
        yield _UserTextArea(
            "",
            language=None,
            show_line_numbers=False,
            tab_behavior="focus",
            soft_wrap=True,
            id="userinput",
        )
        with Horizontal(id="btn-row"):
            yield Button("Send  ⌃S", id="btn-send")
            yield Button(self._inbox_button_label(), id="btn-inbox")
            yield Button("📄 Role", id="btn-role")
            yield Button("💬 Context", id="btn-context")
            yield Button("🗜 Compress", id="btn-compress")
            yield Button("Quit  ⌃C", id="btn-quit")
        yield Footer()

    # ---------------------------------------------------------------- mount

    def on_mount(self) -> None:
        log("tui_start", instance=self.instance, model=self.cfg.model)

        # Replace Textual's mouse-enable sequence with our safe set.
        # See full rationale in _MOUSE_DISABLE_UNSAFE / _MOUSE_ENABLE_SAFE.
        try:
            drv = self._driver
            if drv is not None:
                drv.write(self._MOUSE_DISABLE_UNSAFE)
                drv.write(self._MOUSE_ENABLE_SAFE)
                drv.flush()
                # Re-enable our safe set on SIGCONT (iTerm resume, etc.)
                if hasattr(drv, "_enable_mouse_support"):
                    drv_ref = drv
                    def _safe_reenable() -> None:
                        try:
                            drv_ref.write(self._MOUSE_DISABLE_UNSAFE)
                            drv_ref.write(self._MOUSE_ENABLE_SAFE)
                            drv_ref.flush()
                        except Exception:
                            pass
                    drv._enable_mouse_support = _safe_reenable  # type: ignore[method-assign]
                # Patch the disable path to also cover mode 1002 which we add
                # but Textual's built-in disabler omits.  Without this patch,
                # mode 1002 survives every exit and the shell renders raw X10
                # mouse bytes as garbage characters on click ("T6#T6" issue).
                if hasattr(drv, "_disable_mouse_support"):
                    drv_ref_d = drv
                    def _safe_disable() -> None:
                        try:
                            drv_ref_d.write("\x1b[?1000l")
                            drv_ref_d.write("\x1b[?1002l")
                            drv_ref_d.write("\x1b[?1003l")
                            drv_ref_d.write("\x1b[?1005l")
                            drv_ref_d.write("\x1b[?1006l")
                            drv_ref_d.write("\x1b[?1015l")
                            drv_ref_d.flush()
                        except Exception:
                            pass
                    drv._disable_mouse_support = _safe_disable  # type: ignore[method-assign]
        except Exception:
            pass

        self._cached_status_static = self._compute_status_text()
        try:
            self._spinner_timer = self.set_interval(0.15, self._tick_spinner)
        except Exception:
            self._spinner_timer = None

        # Sync inbox button CSS class to the loaded config state.
        self._update_inbox_button()

        self._render_history()
        self._warn_if_context_near_limit()
        self._focus_input_after_refresh()

        # Start autonomous peer-message poller if configured.
        if self.cfg.peer_poll_enabled:
            try:
                self._load_peer_poll_watermark()
                interval = max(30, self.cfg.peer_poll_interval_seconds)
                self._peer_poll_timer = self.set_interval(
                    interval, self._poll_peer_messages
                )
                log("peer_poll_started", interval=interval, roles=self.cfg.peer_poll_roles,
                    watermark=self._peer_poll_watermark)
            except Exception:
                self._peer_poll_timer = None

    def _focus_input_after_refresh(self) -> None:
        def _do_focus() -> None:
            try:
                self.query_one("#userinput", _UserTextArea).focus()
            except Exception:
                pass
        try:
            self.call_after_refresh(_do_focus)
        except Exception:
            _do_focus()

    # ---------------------------------------------------------------- startup checks

    _CONTEXT_WARN_RATIO = 0.80        # trigger auto-evict when >= 80% of max_window_tokens
    _CONTEXT_AUTO_EVICT_TARGET = 0.60  # trim to 60% on startup auto-evict

    # Peer messages whose payload.type matches these strings are auto-acked
    # silently — no LLM turn, just a debug.log entry.  Covers protocol-close
    # noise (ack, thread-close, pong_ack, etc.).  Overridable via config.
    _DEFAULT_SILENT_PEER_TYPES: frozenset[str] = frozenset({
        "ack", "ack_close", "ack_close_final", "ack_thread_terminal",
        "ack_close_2", "pong_ack", "thread_close", "thread_close_final",
    })

    def _warn_if_context_near_limit(self) -> None:
        """Auto-evict on startup if the rolling window is already near capacity.

        When estimated tokens >= _CONTEXT_WARN_RATIO * max_window_tokens,
        evict oldest turns down to _CONTEXT_AUTO_EVICT_TARGET and post a
        single inline notice showing what was dropped.  Fires silently when
        below the threshold.
        """
        try:
            max_tok = self.cfg.max_window_tokens
            if max_tok <= 0:
                return
            estimated, _ = self.store.total_tokens()
            if estimated < max_tok * self._CONTEXT_WARN_RATIO:
                return
            pct_before = int(100 * estimated / max_tok)
            target_tokens = int(max_tok * self._CONTEXT_AUTO_EVICT_TARGET)
            rows_dropped = self.store.evict(
                max_turns=self.cfg.max_window_turns,
                max_tokens=target_tokens,
            )
            new_estimated, _ = self.store.total_tokens()
            pct_after = int(100 * new_estimated / max_tok)
            log(
                "context_auto_evict_startup",
                pct_before=pct_before,
                pct_after=pct_after,
                rows_dropped=rows_dropped,
                max=max_tok,
            )
            msg = (
                f"[bold yellow]⚠ rolling window was at {pct_before}%[/bold yellow] — "
                f"auto-evicted [bold]{rows_dropped}[/bold] turn(s), "
                f"now at {pct_after}% (~{new_estimated:,} / {max_tok:,} tokens)"
            )
            self._chat().write(msg)
            self._refresh_status()
        except Exception:
            pass  # never crash startup over a warning

    # ---------------------------------------------------------------- peer poll

    def _inbox_button_label(self) -> str:
        """Return the inbox button label matching current peer_poll_enabled state."""
        if self.cfg.peer_poll_enabled:
            return "⚡ Inbox: ON"
        return "📭 Inbox: OFF"

    def _update_inbox_button(self) -> None:
        """Sync the inbox button label and CSS class to current peer_poll_enabled."""
        try:
            btn = self.query_one("#btn-inbox", Button)
            btn.label = self._inbox_button_label()
            if self.cfg.peer_poll_enabled:
                btn.remove_class("inbox-off")
            else:
                btn.add_class("inbox-off")
        except Exception:
            pass  # widget may not be mounted yet (e.g. during compose)

    def _watermark_path(self) -> Path:
        return Path.home() / ".mnemara" / self.instance / "peer_poll_watermark"

    def _peer_db_path(self) -> str:
        """Return the configured path to the peer message database.

        Returns an empty string when not configured.  Callers that require
        the database (polling, watermark init) must check for empty and skip
        gracefully rather than trying to open a default path.
        """
        return self.cfg.peer_db_path

    def _load_peer_poll_watermark(self) -> None:
        """Load the persisted high-watermark from disk.

        If no watermark file exists (first run with peer poll enabled), initialize
        to the current max row_id in the peer message database so that all
        pre-existing rows are silently skipped — only messages submitted AFTER this
        startup will be delivered.  This prevents the 'stale message flood' on first
        enable.
        """
        import sqlite3 as _sqlite3

        wm_path = self._watermark_path()
        if wm_path.exists():
            try:
                self._peer_poll_watermark = int(wm_path.read_text().strip())
                return
            except Exception:
                pass  # corrupted — re-initialize below

        # No watermark file: bootstrap to current max row_id so backlog is skipped.
        db_path = self._peer_db_path()
        if not db_path:
            self._peer_poll_watermark = 0
            return
        try:
            conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM returns").fetchone()
                self._peer_poll_watermark = row[0] if row else 0
            finally:
                conn.close()
        except Exception:
            self._peer_poll_watermark = 0

        self._save_peer_poll_watermark()
        log("peer_poll_watermark_init", watermark=self._peer_poll_watermark)

    def _save_peer_poll_watermark(self) -> None:
        """Persist the current watermark to disk."""
        try:
            wm_path = self._watermark_path()
            wm_path.write_text(str(self._peer_poll_watermark))
        except Exception:
            pass  # never crash over watermark persistence

    async def _poll_peer_messages(self) -> None:
        """Background timer: detect new peer messages in the peer database (zero LLM cost).

        Fires every peer_poll_interval_seconds (default 30 s).  This method
        only reads SQLite and updates _peer_pending_rows — it never starts an
        LLM turn.  Actual processing (one batched LLM turn) is deferred to
        _process_peer_messages(), which fires when the agent is next idle.

        Detection/processing split keeps token cost at zero on empty polls and
        batches N arriving messages into exactly 1 API call instead of N.

        When peer_poll_enabled is False the method returns immediately — no
        SQLite read, no badge update, no LLM turn.  The ⚡ N badge in the
        status bar is NOT suppressed: messages that were already pending before
        the toggle stay visible so the user can see they are accumulating.

        When peer_db_path is empty the method returns immediately with a
        warning log entry — callers should not enable polling without configuring
        the database path.
        """
        # Guard: skip all detection work when polling is toggled OFF.
        if not self.cfg.peer_poll_enabled:
            return

        import sqlite3 as _sqlite3

        db_path = self._peer_db_path()
        if not db_path:
            log("peer_poll_skip_no_db_path")
            return

        peer_roles = [
            r.strip()
            for r in self.cfg.peer_poll_roles.split(",")
            if r.strip()
        ]
        if not peer_roles:
            return

        try:
            conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except Exception:
            return

        new_rows: list[dict] = []
        try:
            placeholders = ",".join("?" * len(peer_roles))
            # Only fetch rows ABOVE the persisted watermark — prevents backlog re-delivery
            # on every restart.  The watermark is advanced + saved after each detection
            # batch so future polls and restarts see only genuinely new messages.
            #
            # Two delivery paths (OR-joined):
            #   1. Sender is in peer_poll_roles AND row is addressed to us or broadcast.
            #      This is the normal peer-to-peer path (known senders).
            #   2. Row is explicitly addressed to this panel (recipient_role = self.instance)
            #      regardless of who sent it.
            #      This is the "new sender" path — any sender not yet in peer_poll_roles
            #      can reach a panel without a config update by using explicit addressing.
            #      recipient_role-targeted delivery is always intentional, so sender-list
            #      gating would only create false negatives.
            cur = conn.execute(
                f"SELECT id, agent_role, task_id, payload_json, submitted_at "
                f"FROM returns "
                f"WHERE status='pending' AND id > ? AND ("
                f"  (agent_role IN ({placeholders}) AND (recipient_role IS NULL OR recipient_role = ?))"
                f"  OR recipient_role = ?"
                f") "
                f"ORDER BY id ASC LIMIT 20",
                [self._peer_poll_watermark] + peer_roles + [self.instance, self.instance],
            )
            for row_id, sender_role, task_id, payload_json, submitted_at in cur.fetchall():
                try:
                    payload = json.loads(payload_json) if payload_json else {}
                except Exception:
                    payload = {"raw": payload_json}
                new_rows.append({
                    "row_id": row_id,
                    "sender_role": sender_role,
                    "task_id": task_id or "(no topic)",
                    "payload": payload,
                    "submitted_at": submitted_at,
                })
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if new_rows:
            # Advance watermark to highest id seen; persist immediately so a crash
            # or restart won't re-deliver these rows.
            max_id = max(r["row_id"] for r in new_rows)
            if max_id > self._peer_poll_watermark:
                self._peer_poll_watermark = max_id
                self._save_peer_poll_watermark()

            # Split into silent (auto-ack, no LLM) vs active (needs LLM turn).
            # Silent types: ack, thread-close, pong_ack, etc. — protocol noise
            # that doesn't need the agent's attention.  We ack them here, log
            # them, and skip them.  Active rows go to _peer_pending_rows as before.
            _default_silent = MnemaraTUI._DEFAULT_SILENT_PEER_TYPES
            silent_types: frozenset[str] = (
                frozenset(self.cfg.peer_poll_silent_types)
                if self.cfg.peer_poll_silent_types is not None
                else _default_silent
            )
            active_rows: list[dict] = []
            for r in new_rows:
                msg_type = r["payload"].get("type", "")
                if msg_type in silent_types:
                    # Auto-ack silently: the MCP ack tool is not available in a
                    # sync SQLite poll context, so we mark the row done directly
                    # in the peer database.
                    try:
                        ack_conn = _sqlite3.connect(self._peer_db_path())
                        ack_conn.execute(
                            "UPDATE returns SET status='done', completed_at=? WHERE id=?",
                            (datetime.now(timezone.utc).isoformat(), r["row_id"])
                        )
                        ack_conn.commit()
                        ack_conn.close()
                    except Exception:
                        pass
                    log("peer_silent_ack",
                        sender=r["sender_role"], row_id=r["row_id"],
                        msg_type=msg_type, topic=r["task_id"])
                else:
                    active_rows.append(r)
                    log("peer_ping_detected",
                        sender=r["sender_role"], row_id=r["row_id"], topic=r["task_id"])

            self._peer_pending_rows.extend(active_rows)
            # Show ⚡ badge in status bar immediately (no LLM cost).
            self._refresh_status()
            # If there are active rows and we're idle, process now; otherwise
            # the _send_turn finally block drains them after the current turn ends.
            if active_rows and not self._busy:
                await self._process_peer_messages()

    async def _process_peer_messages(self) -> None:
        """Batch ALL pending peer rows into ONE LLM turn (called when idle).

        N messages → 1 API call.  Called from:
        - _poll_peer_messages() when not busy at detection time
        - _send_turn()'s finally block after each turn completes
        - action_check_inbox() (Inbox button / /inbox command)
        """
        if not self._peer_pending_rows or self._busy:
            return

        # Batch mode (default): consume all pending rows in one LLM turn.
        # Turn-by-turn mode (peer_poll_batch=False): consume only the first
        # row; the remainder stay in _peer_pending_rows and are drained by
        # _send_turn's finally block after this turn completes.
        if self.cfg.peer_poll_batch:
            rows = list(self._peer_pending_rows)
            self._peer_pending_rows.clear()
        else:
            rows = [self._peer_pending_rows.pop(0)]
        self._refresh_status()

        count = len(rows)
        header = f"[PEER MESSAGES — {count} pending]\n\n"
        parts: list[str] = []
        for i, r in enumerate(rows, 1):
            row_id = r["row_id"]
            sender = r["sender_role"]
            topic = r["task_id"]
            payload_pretty = json.dumps(r["payload"], indent=2)
            parts.append(
                f"--- Message {i}/{count}: from {sender} "
                f"(row_id={row_id}, topic={topic}) ---\n{payload_pretty}"
            )

        ack_tool = self.cfg.peer_poll_ack_tool
        submit_tool = self.cfg.peer_poll_submit_tool
        if ack_tool:
            ack_lines = "\n".join(
                f"  {ack_tool}(row_id={r['row_id']})" for r in rows
            )
            ack_instruction = f"1. Ack each row:\n{ack_lines}"
        else:
            row_ids = ", ".join(str(r["row_id"]) for r in rows)
            ack_instruction = (
                f"1. Ack each row (row_id in [{row_ids}]) via whatever ack tool "
                f"your peer-message system provides."
            )
        if submit_tool:
            reply_instruction = (
                f"2. Submit your responses via {submit_tool}("
                f'role="{self.instance}", task_id=<topic>, payload={{...}})'
            )
        else:
            reply_instruction = (
                f"2. Submit your responses via whatever reply tool your peer-message "
                f'system provides, identifying yourself as "{self.instance}".'
            )
        footer = (
            f"\nFor each message, per your role doc protocol:\n"
            f"{ack_instruction}\n"
            f"{reply_instruction}"
        )

        message = header + "\n\n".join(parts) + footer

        # Show a single chat notice for all batched messages.
        senders = ", ".join(r["sender_role"] for r in rows)
        self._chat().write(
            f"[dim]📨 [bold]{count}[/bold] peer message(s) from "
            f"[bold]{senders}[/bold] — processing now[/dim]"
        )
        log("peer_messages_processing", count=count,
            row_ids=[r["row_id"] for r in rows])

        await self._handle_user_input(message)

    # ---------------------------------------------------------------- chat log

    def _chat(self) -> "RichLog":
        return self.query_one("#chatlog", RichLog)

    def _render_history(self) -> None:
        log_widget = self._chat()
        rows = self.store.window()
        if not rows:
            log_widget.write(
                "[dim]Tip: Enter inserts a newline. Ctrl+S sends.[/dim]"
            )
            log_widget.write("[dim](empty window — start chatting below)[/dim]")
            return
        for row in rows:
            self._render_turn(row["role"], row["content"], row.get("ts", ""))

    def _render_turn(self, role: str, content: Any, ts: str = "") -> None:
        log_widget = self._chat()
        if role == "user":
            text = _flatten_text_blocks(content)
            log_widget.write(f"[b cyan]you:[/b cyan] {text}")
        elif role == "assistant":
            label = self.cfg.display_name or "assistant"
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    t = b.get("type")
                    if t == "text" and b.get("text"):
                        log_widget.write(f"[b green]{label}:[/b green] {b['text']}")
                    elif t == "tool_use":
                        name = b.get("name", "?")
                        inp = b.get("input") or {}
                        log_widget.write(
                            f"[b magenta]> tool:[/b magenta] [magenta]{name}[/magenta]({_short(inp)})"
                        )
                    elif t == "tool_result":
                        c = b.get("content")
                        log_widget.write(f"[dim]  result: {str(c)[:200]}[/dim]")
            else:
                log_widget.write(f"[b green]{label}:[/b green] {content}")

    # ---------------------------------------------------------------- status / spinner

    def _compute_status_text(self) -> str:
        """Compute status line. Called only on turn boundaries — not on every tick."""
        try:
            tin, tout = self.store.total_tokens()
        except Exception:
            tin, tout = 0, 0
        try:
            nturns = len(self.store.window())
        except Exception:
            nturns = 0
        try:
            ev = self.store.get_eviction_stats()
            ev_str = f"{ev['rows_evicted']}r"
            blocks = ev.get("blocks_evicted", 0)
            if blocks > 0:
                ev_str += f" {blocks}b"
        except Exception:
            ev_str = "0r"
        queue_str = " [yellow]⏸ queued[/yellow]" if self._queued_input is not None else ""
        inbox_count = len(self._peer_pending_rows)
        inbox_str = f" [bold yellow]⚡ {inbox_count}[/bold yellow]" if inbox_count > 0 else ""
        return (
            f"turns: {nturns} | tokens: {tin}/{self.cfg.max_window_tokens} (out: {tout} cum) | "
            f"model: {self.cfg.model} | evicted: {ev_str}{queue_str}{inbox_str}"
        )

    def _status_text(self) -> str:
        return self._compute_status_text()

    def _render_status_widget(self) -> None:
        if self._busy:
            frame = self._SPINNER_FRAMES[self._spinner_idx % len(self._SPINNER_FRAMES)]
            text = f"[#6f9ad9]{frame}[/#6f9ad9] {self._cached_status_static}"
        else:
            text = self._cached_status_static
        try:
            self.query_one("#status", Static).update(text)
        except Exception:
            pass

    def _refresh_status(self) -> None:
        self._cached_status_static = self._compute_status_text()
        self._render_status_widget()

    def _tick_spinner(self) -> None:
        if self._busy:
            self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER_FRAMES)
            self._spinner_was_busy = True
            self._render_status_widget()
            return
        if self._spinner_was_busy:
            self._spinner_was_busy = False
            self._spinner_idx = 0
            self._render_status_widget()

    # ---------------------------------------------------------------- events

    async def action_submit_prompt(self) -> None:
        """Ctrl+S handler: pull text from the TextArea and dispatch."""
        try:
            ta = self.query_one("#userinput", _UserTextArea)
        except Exception:
            return
        text = (ta.text or "").strip()
        if not text:
            return
        ta.clear()
        await self._handle_user_input(text)

    def action_clear_input(self) -> None:
        """Escape handler: discard whatever is in the input bar and refocus it."""
        try:
            ta = self.query_one("#userinput", _UserTextArea)
        except Exception:
            return
        ta.clear()
        ta.focus()

    async def on_button_pressed(self, event: "Button.Pressed") -> None:
        """Handle Send, Inbox, Role, Context, Compress, and Quit button clicks."""
        if event.button.id == "btn-send":
            await self.action_submit_prompt()
        elif event.button.id == "btn-inbox":
            await self.action_check_inbox()
        elif event.button.id == "btn-role":
            await self.action_open_role_editor()
        elif event.button.id == "btn-context":
            await self.action_open_context_viewer()
        elif event.button.id == "btn-compress":
            self._compress_to_tokens(target_tokens=None, chat=self._chat())
        elif event.button.id == "btn-quit":
            await self.action_quit()

    async def action_check_inbox(self) -> None:
        """Toggle peer message delivery on/off (Inbox button / /inbox).

        Flips cfg.peer_poll_enabled, persists to config.json, updates the
        button label + style live, and logs the new state to debug.log.
        When toggled ON, any already-pending rows are processed immediately
        if the agent is idle.
        """
        self.cfg.peer_poll_enabled = not self.cfg.peer_poll_enabled
        new_state = self.cfg.peer_poll_enabled

        try:
            config_mod.save(self.instance, self.cfg)
        except Exception as exc:
            self._chat().write(f"[red]inbox: config save failed: {exc}[/red]")

        self._update_inbox_button()
        log("peer_poll_toggled", enabled=new_state, instance=self.instance)

        state_label = "ON" if new_state else "OFF"
        self._chat().write(
            f"[dim]inbox: peer message delivery toggled "
            f"[bold]{state_label}[/bold] (persisted)[/dim]"
        )

        # When turned ON, immediately drain any messages that accumulated while OFF.
        if new_state and self._peer_pending_rows and not self._busy:
            await self._process_peer_messages()

    async def action_open_role_editor(self) -> None:
        """Open the role-doc editor modal (📄 Role button / /role_doc).

        Reads the current role doc from disk, presents it in a full-screen
        TextArea overlay.  On Save (Ctrl+S), writes the edited content back
        to disk.  Changes take effect on the next turn — no restart needed
        because the role doc is re-read from disk at every API call.
        """
        chat = self._chat()
        path = self.cfg.role_doc_path
        if not path:
            chat.write(
                "[dim]role_doc: no role_doc_path configured — "
                "set it in config.json and restart[/dim]"
            )
            return
        try:
            content = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            content = ""
        except Exception as exc:
            chat.write(f"[red]role_doc: could not read {path}: {exc}[/red]")
            return

        def _on_dismiss(result: dict) -> None:
            if result.get("saved"):
                chat.write(
                    f"[green]role_doc: saved[/green] [dim]{path}[/dim] "
                    f"— changes take effect on next turn"
                )
                log("role_doc_saved", path=path, instance=self.instance)
            elif result.get("cancelled") and result.get("dirty"):
                chat.write("[dim]role_doc: cancelled (unsaved changes discarded)[/dim]")
            elif result.get("error"):
                chat.write(f"[red]role_doc: save failed: {result['error']}[/red]")

        await self.push_screen(RoleDocEditorModal(path, content), _on_dismiss)

    async def action_open_context_viewer(self) -> None:
        """Open the context viewer modal (💬 Context button / /context).

        Displays the rolling window as a searchable, filterable two-panel view:
        left = turn list (most recent first), right = full content of selected turn.
        Actions: Evict (remove from window), Pin, Unpin.
        Changes (evict/pin) take effect immediately in the store.
        """
        await self.push_screen(ContextViewerModal(self.store, self.instance))

    async def _handle_user_input(self, text: str) -> None:
        """Process a submitted prompt: slash commands, queuing, or turn dispatch."""
        if text.startswith("/"):
            await self._handle_slash(text)
            self._refresh_status()
            return

        if self._busy:
            if self._queued_input is None:
                self._queued_input = text
                self._chat().write(
                    f"[dim]⏸ queued (1): [i]{text[:80]}{'…' if len(text) > 80 else ''}[/i] "
                    f"— will fire when current turn completes[/dim]"
                )
            else:
                # Already one item queued — replace it and tell the user.
                self._queued_input = text
                self._chat().write(
                    f"[dim]⏸ queue replaced: [i]{text[:80]}{'…' if len(text) > 80 else ''}[/i][/dim]"
                )
            self._refresh_status()
            return

        self.run_worker(
            self._send_turn(text),
            name="mnemara_turn",
            group="turn",
            exclusive=True,
        )

    async def _send_turn(self, text: str) -> None:
        self._busy = True
        chat = self._chat()
        chat.write(f"[b cyan]you:[/b cyan] {text}")
        self._stream_buffer = ""

        async def on_token(t: str) -> None:
            self._stream_buffer += t

        async def on_tool_use(name: str, inp: dict) -> None:
            chat.write(f"[b magenta]> tool:[/b magenta] [magenta]{name}[/magenta]({_short(inp)})")

        async def on_tool_result(tid: str, content: Any, is_error: bool) -> None:
            tag = "[red]error[/red]" if is_error else "[dim]result[/dim]"
            chat.write(f"  {tag}: {str(content)[:300]}")

        try:
            await self.session.turn_async(
                text,
                on_token=on_token,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
            )
            if self._stream_buffer:
                label = self.cfg.display_name or "assistant"
                chat.write(f"[b green]{label}:[/b green] {self._stream_buffer}")
        except asyncio.CancelledError:
            try:
                self.store.append_turn(
                    "assistant",
                    [{"type": "text", "text": "[interrupted]"}],
                )
            except Exception:
                pass
            if self._stream_buffer:
                label = self.cfg.display_name or "assistant"
                chat.write(f"[b green]{label}:[/b green] [dim]{self._stream_buffer}[/dim]")
            chat.write("[dim]⏹ turn interrupted[/dim]")
            raise
        except Exception as exc:
            log("tui_turn_error", error=str(exc))
            chat.write(f"[red]error:[/red] {exc}")
            # Surface a recovery hint for context-length overflows.
            exc_str = str(exc).lower()
            if "too long" in exc_str or "context" in exc_str or "/evict" in exc_str:
                chat.write(
                    "[dim]hint:[/dim] run [bold]/evict N[/bold] to drop the oldest N turns,"
                    " or [bold]/clear[/bold] to reset the window"
                )
        finally:
            self._busy = False
            self._refresh_status()
            self._focus_input_after_refresh()
            # Drain the user input queue — fire the next message automatically.
            if self._queued_input is not None:
                queued = self._queued_input
                self._queued_input = None
                self._refresh_status()
                self.run_worker(
                    self._send_turn(queued),
                    name="mnemara_turn",
                    group="turn",
                    exclusive=True,
                )
            # Drain any peer messages that arrived while we were busy.
            elif self._peer_pending_rows:
                self.run_worker(
                    self._process_peer_messages(),
                    name="mnemara_turn",
                    group="turn",
                    exclusive=True,
                )

    # ---------------------------------------------------------------- slash commands

    async def _handle_slash(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        chat = self._chat()

        if cmd in ("/quit", "/exit"):
            self.exit()
            return

        if cmd == "/tokens":
            await self._slash_set_window(arg, chat)
            return

        if cmd == "/models":
            self._slash_models(chat)
            return

        if cmd == "/swap":
            await self._slash_swap_model(arg, chat)
            return

        if cmd == "/export":
            await self._slash_export(arg, chat)
            return

        if cmd == "/import":
            await self._slash_import(arg, chat)
            return

        if cmd == "/stop":
            await self._slash_stop(chat)
            return

        if cmd == "/clear":
            self._slash_clear(chat)
            return

        if cmd == "/evict":
            self._slash_evict(arg, chat)
            return

        if cmd in ("/help", "/?"):
            self._slash_help(chat)
            return

        if cmd == "/compress":
            stripped_arg = arg.strip().lower()
            if stripped_arg == "reads":
                self._slash_compress_reads(chat)
            elif stripped_arg == "smart":
                await self._slash_compress_smart(chat)
            else:
                # /compress <N>  or  /compress (bare = 25% default)
                target: int | None = None
                if stripped_arg:
                    try:
                        target = _parse_size(stripped_arg)
                    except ValueError:
                        chat.write(
                            "[dim]usage: /compress [reads | smart | <token-count>] "
                            "— e.g. /compress 20000 or /compress 500k or /compress smart[/dim]"
                        )
                        return
                self._compress_to_tokens(target_tokens=target, chat=chat)
            return

        if cmd == "/skeleton":
            await self._slash_skeleton(arg)
            return

        if cmd == "/name":
            self._slash_name(arg, chat)
            return

        if cmd == "/inbox":
            await self.action_check_inbox()
            return

        if cmd in ("/role_doc", "/role-doc", "/roledoc"):
            await self.action_open_role_editor()
            return

        if cmd in ("/context", "/ctx"):
            await self.action_open_context_viewer()
            return

        chat.write(
            f"[dim]unknown command: {cmd} — try /help for a full list[/dim]"
        )

    def _slash_models(self, chat: "RichLog") -> None:
        lines = ["[bold]available models[/bold]"]
        for i, model in enumerate(config_mod.AVAILABLE_MODELS, start=1):
            marker = " [green](current)[/green]" if model == self.cfg.model else ""
            lines.append(f"  {i}. {model}{marker}")
        aliases = ", ".join(
            f"{k}={v}" for k, v in sorted(config_mod.MODEL_ALIASES.items())
        )
        lines.append(f"[dim]aliases: {aliases}[/dim]")
        lines.append("[dim]usage: /swap 1  or  /swap claude-opus-4-7[/dim]")
        chat.write("\n".join(lines))

    async def _slash_swap_model(self, arg: str, chat: "RichLog") -> None:
        parts = arg.split()
        if not parts:
            chat.write("[red]usage: /swap MODEL|NUMBER  (try /models)[/red]")
            return
        raw = parts[0]
        temp = len(parts) > 1 and parts[1].lower() in ("--temp", "-t", "temp")
        try:
            model = config_mod.resolve_model_choice(raw)
        except ValueError as exc:
            chat.write(f"[red]{exc}[/red]")
            return
        old = self.cfg.model
        self.cfg.model = model
        if not temp:
            try:
                config_mod.save(self.instance, self.cfg)
                persist_note = "(persisted to config.json)"
            except Exception as exc:
                persist_note = f"[red](persist failed: {exc})[/red]"
        else:
            persist_note = "(in-memory only — reverts on restart)"
        chat.write(f"[green]model: {old} -> {model}[/green]  [dim]{persist_note}[/dim]")
        self.sub_title = f"model={self.cfg.model}  role={self.cfg.role_doc_path or '(none)'}"
        self._refresh_status()

    async def _slash_set_window(self, arg: str, chat: "RichLog") -> None:
        """/tokens N [--temp] — set max_window_tokens.

        Accepts plain integers or shorthand: 500k, 1m, 200_000.
        Persists to config.json by default; pass --temp to keep in-memory only.
        Bounds: 1000 – 10_000_000.
        """
        parts = arg.split()
        if not parts:
            chat.write(
                "[red]usage: /tokens N [--temp]  "
                "(N accepts 500k, 1m, 200000, etc.)[/red]"
            )
            return
        raw = parts[0]
        temp = len(parts) > 1 and parts[1].lower() in ("--temp", "-t", "temp")
        try:
            n = _parse_size(raw)
        except ValueError as e:
            chat.write(f"[red]invalid value '{raw}': {e}[/red]")
            return
        if not 1000 <= n <= 10_000_000:
            chat.write("[red]tokens must be between 1000 and 10000000 (10M)[/red]")
            return
        old = self.cfg.max_window_tokens
        self.cfg.max_window_tokens = n
        if not temp:
            try:
                config_mod.save(self.instance, self.cfg)
                persist_note = "(persisted to config.json)"
            except Exception as exc:
                persist_note = f"[red](persist failed: {exc})[/red]"
        else:
            persist_note = "(in-memory only — reverts on restart)"
        chat.write(f"[green]tokens: {old} → {n}[/green]  [dim]{persist_note}[/dim]")
        self._refresh_status()

    async def _slash_stop(self, chat: "RichLog") -> None:
        """Cancel the active turn worker, if any."""
        # Copied from mainline behavior: cancel the in-flight streaming turn
        # and let _send_turn's CancelledError handler write the interrupt stub
        # and clear _busy in its finally block.
        self.workers.cancel_group(self, "turn")
        if self._busy:
            chat.write("[dim]⏹ stop signal sent — finishing current operation…[/dim]")
        else:
            chat.write("[dim]nothing in flight[/dim]")

    def _slash_clear(self, chat: "RichLog") -> None:
        """/clear — fresh-start wipe: strips tool_use blocks, thinking blocks,
        and deletes ALL turn rows (user + assistant) from storage, then clears
        the chat display.

        After /clear only pinned rows and the role doc remain. Status bar and
        token count update immediately to reflect the emptied store.
        Pinned rows are always preserved.
        """
        try:
            before, _ = self.store.total_tokens()
            tools_result = self.store.evict_tool_use_blocks(all_rows=True, skip_pinned=True)
            think_result = self.store.evict_thinking_blocks(all_rows=True, skip_pinned=True)
            user_rows = self.store.evict_by_role("user", skip_pinned=True)
            asst_rows = self.store.evict_by_role("assistant", skip_pinned=True)
            after, _ = self.store.total_tokens()
            tools_freed = tools_result.get("blocks_evicted", 0)
            think_freed = think_result.get("blocks_evicted", 0)
            freed = max(0, before - after)
        except Exception:
            tools_freed = think_freed = user_rows = asst_rows = freed = 0

        chat.clear()
        chat.write(
            f"[dim]clear: stripped {tools_freed} tool block(s), "
            f"{think_freed} thinking block(s), "
            f"{user_rows} user turn(s), {asst_rows} assistant turn(s) — "
            f"~{freed:,} tokens freed[/dim]"
        )
        self._refresh_status()

    def _slash_evict(self, arg: str, chat: "RichLog") -> None:
        """/evict [tools|thinking|user|assistant|N|last N] — free context budget.

        /evict tools      — strip tool_use blocks from all stored rows
        /evict thinking   — strip thinking blocks from all stored rows
        /evict user       — drop all user-turn rows (keeps assistant responses)
        /evict assistant  — drop all assistant-turn rows (keeps user inputs)
        /evict N          — drop the N oldest rows (budget reclaim, keeps recent context)
        /evict last N     — drop the N most-recent rows (rollback a bad paste/turn)
        /evict            — show eviction stats
        """
        arg = arg.strip().lower()
        ev = self.store.get_eviction_stats()
        if not arg:
            chat.write(
                f"[bold]eviction stats[/bold]  "
                f"rows: {ev['rows_evicted']} | blocks: {ev.get('blocks_evicted', 0)} | "
                f"bytes freed: {ev.get('bytes_freed', 0):,}"
            )
            return

        try:
            if arg == "tools":
                result = self.store.evict_tool_use_blocks(all_rows=True)
                freed = result.get("blocks_evicted", result.get("rows_modified", 0))
                chat.write(f"[green]/evict tools:[/green] {freed} tool_use block(s) stripped")
            elif arg == "thinking":
                result = self.store.evict_thinking_blocks(all_rows=True)
                freed = result.get("blocks_evicted", result.get("rows_modified", 0))
                chat.write(f"[green]/evict thinking:[/green] {freed} thinking block(s) stripped")
            elif arg in ("user", "assistant"):
                dropped = self.store.evict_by_role(arg)
                label = "user input(s)" if arg == "user" else "assistant response(s)"
                chat.write(f"[green]/evict {arg}:[/green] {dropped} {label} evicted")
            elif arg.isdigit():
                n = int(arg)
                dropped = self.store.evict_oldest(n)
                chat.write(f"[green]/evict {n}:[/green] {dropped} oldest row(s) evicted")
            elif arg.startswith("last ") and arg[5:].strip().isdigit():
                n = int(arg[5:].strip())
                dropped = self.store.evict_last(n)
                chat.write(f"[green]/evict last {n}:[/green] {dropped} most-recent row(s) evicted")
            else:
                chat.write("[dim]/evict [tools|thinking|user|assistant|N|last N] — see /help[/dim]")
                return
        except Exception as exc:
            chat.write(f"[red]evict error: {exc}[/red]")
        self._refresh_status()

    def _slash_compress_reads(self, chat: "RichLog") -> None:
        """/compress reads — stub earlier Read tool_results with diffs vs latest."""
        try:
            result = self.store.compress_repeated_reads(skip_pinned=True)
            chat.write(
                f"[green]compressed reads:[/green] {result['reads_compressed']} read(s) stubbed, "
                f"{result['bytes_freed']:,} bytes freed"
            )
        except Exception as exc:
            chat.write(f"[red]compress reads error: {exc}[/red]")
        self._refresh_status()

    async def _slash_compress_smart(self, chat: "RichLog") -> None:
        """/compress smart — fire a judgment-driven audit turn.

        Injects a structured audit prompt as a real user turn so the agent
        reviews its own rolling window, decides what is stale or redundant,
        and calls evict_ids / evict_thinking_blocks / evict_write_pairs with
        deliberate intent.  Unlike /compress <N> (mechanical FIFO), this path
        lets the agent weigh content before evicting.

        Posts a brief notice to chat, then hands off to _handle_user_input so
        the full turn lifecycle (streaming, tool-use display, status bar update)
        runs as normal.  The agent's reply is the compression report.
        """
        chat.write(
            "[dim]🧠 compress smart: injecting audit turn — "
            "agent will review context and evict by judgment…[/dim]"
        )
        _SMART_COMPRESS_PROMPT = (
            "Please audit your rolling window and compress it using your own judgment. "
            "Steps:\n"
            "1. Call list_window() to read summaries of all current turns.\n"
            "2. Identify which turns are stale or low-value: resolved feature threads "
            "where work is committed, superseded information, trivial one-liners, "
            "duplicate summaries, and old coordination acks.\n"
            "3. Call evict_thinking_blocks(all_rows=True, skip_pinned=True) and "
            "evict_write_pairs(skip_pinned=True) to strip bulk content first.\n"
            "4. Call evict_ids() on the rows you have judged as removable. "
            "Prefer evicting oldest resolved threads over recent active context.\n"
            "5. Never evict pinned rows (pin_label is set) or the role doc.\n"
            "6. After evicting, report: how many rows were removed, estimated tokens "
            "freed (before vs after), and a brief note on what categories you dropped "
            "and what you kept. One concise paragraph is enough."
        )
        await self._handle_user_input(_SMART_COMPRESS_PROMPT)

    # Compression to a token target — used by /compress <N> and 🗜 Compress button
    _COMPRESS_DEFAULT_RATIO = 0.25  # button: compress to 25% of current tokens

    def _compress_to_tokens(
        self, target_tokens: int | None, chat: "RichLog"
    ) -> None:
        """Compress the rolling window to target_tokens, skipping pinned turns.

        Pass target_tokens=None to compress to 25% of the current estimated
        token count (the default used by the 🗜 Compress button).

        Three-pass strategy (same order every time):
          1. evict_thinking_blocks    — strip thinking blocks (all rows, skip pinned)
          2. evict_write_pairs        — stub Edit/Write/Read body content (skip pinned)
          3. evict_tool_use_blocks    — strip full tool_use blocks (skip pinned)
          4. Bulk-evict oldest rows   — FIFO until total_tokens() <= target

        Pinned turns and role docs are never touched.
        Reports what each pass freed, then the final token count.
        """
        try:
            before, _ = self.store.total_tokens()
            if target_tokens is None:
                target_tokens = max(0, int(before * self._COMPRESS_DEFAULT_RATIO))

            think_result = self.store.evict_thinking_blocks(
                all_rows=True, skip_pinned=True
            )
            think_freed = think_result.get("blocks_evicted", 0)

            write_result = self.store.evict_write_pairs(skip_pinned=True)
            write_freed = write_result.get("blocks_evicted", write_result.get("rows_modified", 0))

            tool_result = self.store.evict_tool_use_blocks(
                all_rows=True, skip_pinned=True
            )
            tool_freed = tool_result.get("blocks_evicted", 0)

            # FIFO evict oldest rows until at or below target
            rows_dropped = 0
            current, _ = self.store.total_tokens()
            while current > target_tokens:
                dropped = self.store.evict_oldest(10, skip_pinned=True)
                if dropped == 0:
                    break  # nothing left to evict (all pinned or empty)
                rows_dropped += dropped
                current, _ = self.store.total_tokens()

            after, _ = self.store.total_tokens()
            freed = max(0, before - after)

            chat.write(
                f"[green]compress:[/green] "
                f"{think_freed} thinking block(s), "
                f"{write_freed} write pair(s), "
                f"{tool_freed} tool block(s) stripped; "
                f"{rows_dropped} oldest row(s) evicted — "
                f"~{freed:,} tokens freed "
                f"(~{before:,} → ~{after:,})"
            )
        except Exception as exc:
            chat.write(f"[red]compress error: {exc}[/red]")
        self._refresh_status()

    async def _slash_skeleton(self, arg: str) -> None:
        """/skeleton <path> — display Python skeleton (signatures + docstrings only)."""
        chat = self.query_one("#chatlog", RichLog)
        if not arg:
            chat.write("usage: /skeleton <path>")
            return
        result = _read_skeleton({"file_path": arg.strip()})
        if len(result) > 2000:
            result = result[:2000] + "\n[dim]... (truncated)[/dim]"
        chat.write(result)

    def _slash_name(self, arg: str, chat: "RichLog") -> None:
        """/name [label] — set or clear the display name shown on responses."""
        label = arg.strip()
        self.cfg.display_name = label
        try:
            config_mod.save(self.instance, self.cfg)
            if label:
                chat.write(f'[green]display name set to "{label}" and saved to config[/green]')
            else:
                chat.write('[dim]display name cleared — responses will show "assistant"[/dim]')
        except Exception as exc:
            chat.write(f"[red]name updated in memory but config save failed: {exc}[/red]")

    def _slash_help(self, chat: "RichLog") -> None:
        """/help — list available slash commands."""
        lines = [
            "[bold]slash commands[/bold]",
            "  /help, /?               — show this list",
            "  /clear                  — strip tools, thinking, user turns + clear display",
            "  /models                 — list available models",
            "  /swap MODEL             — switch to MODEL (in-session only)",
            "  /tokens N [--temp]      — set max context window",
            "  /evict                  — show eviction stats",
            "  /evict tools            — strip tool_use blocks",
            "  /evict thinking         — strip thinking blocks",
            "  /evict user             — drop all user turns (keep assistant responses)",
            "  /evict assistant        — drop all assistant turns (keep user inputs)",
            "  /evict N                — drop N oldest rows (budget reclaim)",
            "  /evict last N           — drop N most-recent rows (rollback)",
            "  /compress               — compress to 25% of current tokens (strips thinking/tools/write pairs, evicts oldest)",
            "  /compress <N>           — compress to N tokens (e.g. /compress 20000 or /compress 500k)",
            "  /compress reads         — stub repeated Read results with diffs",
            "  /compress smart         — agent reviews context and evicts by judgment",
            "  /skeleton <path>        — show Python skeleton (signatures/docstrings only)",
            "  /inbox                  — toggle peer message delivery on/off",
            "  /role_doc               — open role doc editor (📄 Role button)",
            "  /context                — open context viewer: browse, evict, pin turns (💬 Context button)",
            "  /name <label>           — set response label (e.g. /name coordinator)",
            "  /name                   — clear label, revert to \"assistant\"",
            "  /export [N] [path]      — export turns + config + role_doc to markdown",
            "  /import <path>          — restore turns from a full export file",
            "  /stop                   — cancel active streaming turn",
            "  /quit, /exit            — exit",
        ]
        for line in lines:
            chat.write(line)

    async def _slash_export(self, arg: str, chat: "RichLog") -> None:
        """/export [N] [path] — write full round-trippable markdown export."""
        parts = arg.split()
        n: int | None = None
        out_path: Path | None = None
        if parts:
            first = parts[0]
            if first.isdigit():
                n = int(first)
                if n <= 0:
                    chat.write("[red]usage: /export [N] [path]  (N must be > 0)[/red]")
                    return
                if len(parts) > 1:
                    out_path = Path(" ".join(parts[1:])).expanduser()
            elif len(parts) == 1 and (
                first.startswith("/")
                or first.startswith("~")
                or first.endswith(".md")
                or "/" in first
            ):
                out_path = Path(first).expanduser()
            else:
                chat.write("[red]usage: /export [N] [path][/red]")
                return
        rows = self.store.window()
        if n is not None:
            rows = rows[-n:]
        if not rows:
            chat.write("[dim]window is empty; nothing to export[/dim]")
            return
        if out_path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_instance = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in self.instance
            )
            out_path = Path(tempfile.gettempdir()) / f"mnemara_{safe_instance}_{stamp}.md"

        # Gather config JSON (safe — dataclasses.asdict handles nested dataclasses)
        try:
            cfg_json = json.dumps(dataclasses.asdict(self.cfg), indent=2)
        except Exception:
            cfg_json = ""

        # Read role doc text if a path is configured
        role_doc_text = ""
        if self.cfg.role_doc_path:
            try:
                role_doc_text = Path(self.cfg.role_doc_path).read_text(encoding="utf-8")
            except Exception:
                pass

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                _render_full_export(rows, self.instance, cfg_json, role_doc_text),
                encoding="utf-8",
            )
        except Exception as exc:
            chat.write(f"[red]export failed: {exc}[/red]")
            return
        sections_written = ["turns"]
        if cfg_json:
            sections_written.append("config")
        if role_doc_text:
            sections_written.append("role_doc")
        chat.write(
            f"[green]exported {len(rows)} turn(s):[/green] {out_path}  "
            f"[dim](sections: {', '.join(sections_written)})[/dim]"
        )

    async def _slash_import(self, arg: str, chat: "RichLog") -> None:
        """/import <path> — restore turns from an export file (turns section only)."""
        arg = arg.strip()
        if not arg:
            chat.write("[red]usage: /import <path>[/red]")
            return
        path = Path(arg).expanduser()
        if not path.exists():
            chat.write(f"[red]file not found: {path}[/red]")
            return
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            chat.write(f"[red]read failed: {exc}[/red]")
            return

        sections = _parse_export_sections(text)
        if "turns" not in sections:
            chat.write("[red]no turns section found — is this a full mnemara export?[/red]")
            return

        turns = _parse_export_turns(sections["turns"])
        old_count = len(self.store.window())
        self.store.clear()
        for t in turns:
            self.store.append_turn(role=t["role"], content=t["content"])

        chat.write(
            f"[green]imported {len(turns)} turn(s)[/green] "
            f"(replaced {old_count} previous turn(s) — {path.name})"
        )
        self._refresh_status()

    # ---------------------------------------------------------------- actions

    def action_paste(self) -> None:
        """Ctrl+V paste: read clipboard via pyperclip and insert at cursor.

        TextArea handles multi-line content natively, so no collapsing is
        needed.  The insert() call places text at the current cursor position,
        replacing any active selection.
        """
        try:
            import pyperclip
            raw = pyperclip.paste()
        except Exception:
            return
        if not raw:
            return
        try:
            ta = self.query_one("#userinput", _UserTextArea)
            ta.insert(raw)
        except Exception:
            pass

    def action_scroll_log_up(self) -> None:
        try:
            self._chat().scroll_page_up(animate=False)
        except Exception:
            pass

    def action_scroll_log_down(self) -> None:
        try:
            self._chat().scroll_page_down(animate=False)
        except Exception:
            pass

    async def action_quit(self) -> None:
        self.exit()

    # ---------------------------------------------------------------- shutdown

    async def on_unmount(self) -> None:
        # Step 1: cancel timers BEFORE touching the store.
        # Timers that fire after store.close() cause exceptions that put
        # Textual's shutdown into an error path, skipping driver cleanup
        # and leaving the terminal in raw mouse-tracking mode.
        try:
            if self._spinner_timer is not None:
                self._spinner_timer.stop()
        except Exception:
            pass

        # Cancel in-flight streaming workers and AWAIT their completion.
        # cancel_group() is fire-and-forget; the workers' finally blocks
        # (including _query_gen.aclose() → subprocess cleanup) need the event
        # loop still alive.  Awaiting keeps the loop live until cleanup is done.
        try:
            cancelled = self.workers.cancel_group(self, "turn")
            if cancelled:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[w.wait() for w in cancelled],
                            return_exceptions=True,
                        ),
                        timeout=5.0,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
        except Exception:
            pass

        # Step 2: flush session stats.
        try:
            self.session.write_session_stats()
        except Exception as e:
            log("tui_stats_error", error=str(e))

        # Step 3: close the store (safe — no timers or workers are live now).
        try:
            self.store.close()
        except Exception:
            pass
        log("tui_stop", instance=self.instance)

    # ---------------------------------------------------------------- permission hook

    def _sync_permission_prompt(self, tool: str, target: str) -> str:
        return "deny"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(instance: str) -> bool:
    """Launch the STABLE TUI for an instance.

    Returns True on clean exit, False if Textual is unavailable or stdin/stdout
    is not a TTY (caller should fall back to the bare REPL).
    """
    if not _TEXTUAL_AVAILABLE:
        return False
    if not _is_tty():
        return False
    app = MnemaraTUI(instance)
    app.run()
    return True
