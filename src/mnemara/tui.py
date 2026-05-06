"""Mnemara STABLE TUI — role doc + token-cap eviction + spinner.

Three features, nothing else:
  1. Role doc at slot 0 of every API call  (AgentSession's identity layer)
  2. Token-count eviction only             (row cap is not enforced here)
  3. Spinner during streaming turns

Slash commands: /quit, /exit, /models, /swap, /tokens, /export, and /stop.
No block surgery, no pin system, no copy.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from textual import events as _txt_events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Input, RichLog, Static
    _TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TEXTUAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERINPUT_PASTE_CAP = 16_000

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
from .gemma_agent import GemmaSession as AgentSession
from .config import Config
from .logging_util import log, set_log_path
from .permissions import PermissionStore
from .store import Store
from .tools import ToolRunner


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

#userinput {
    height: 3;
    min-height: 3;
    border: round #4d6fa3;
    background: #11151e;
    color: #ffffff;
}

#userinput:focus {
    border: round #6f9ad9;
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


def _render_export_rows(rows: list[dict[str, Any]], instance: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# Mnemara Export: {instance}",
        "",
        f"- exported_at: {ts}",
        f"- turns: {len(rows)}",
        "",
    ]
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
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Input widget — paste-safe subclass
# ---------------------------------------------------------------------------


class _UserInput(Input):
    """Input with paste behaviour tuned for the Mnemara panel.

    Textual's stock Input._on_paste inserts char-by-char after taking only
    the first line.  We collapse multi-line pastes to one line (join with
    spaces), truncate at _USERINPUT_PASTE_CAP, and do a single atomic value
    assignment so there's one render instead of one per character.
    """

    def _on_paste(self, event: "_txt_events.Paste") -> None:  # type: ignore[name-defined]
        event.prevent_default()
        event.stop()
        text = event.text or ""
        if not text:
            return
        lines = text.splitlines()
        collapsed = " ".join(part for part in (s.strip() for s in lines) if part)
        if len(collapsed) > _USERINPUT_PASTE_CAP:
            collapsed = collapsed[:_USERINPUT_PASTE_CAP]
        cur = self.cursor_position
        old = self.value
        self.value = old[:cur] + collapsed + old[cur:]
        self.cursor_position = cur + len(collapsed)


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
        yield _UserInput(
            placeholder="message  (Enter to send, Ctrl+C to quit)",
            id="userinput",
        )
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

        self._render_history()
        self._focus_input_after_refresh()

    def _focus_input_after_refresh(self) -> None:
        def _do_focus() -> None:
            try:
                self.query_one("#userinput", Input).focus()
            except Exception:
                pass
        try:
            self.call_after_refresh(_do_focus)
        except Exception:
            _do_focus()

    # ---------------------------------------------------------------- chat log

    def _chat(self) -> "RichLog":
        return self.query_one("#chatlog", RichLog)

    def _render_history(self) -> None:
        log_widget = self._chat()
        rows = self.store.window()
        if not rows:
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
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    t = b.get("type")
                    if t == "text" and b.get("text"):
                        log_widget.write(f"[b green]assistant:[/b green] {b['text']}")
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
                log_widget.write(f"[b green]assistant:[/b green] {content}")

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
        except Exception:
            ev_str = "0r"
        queue_str = " [yellow]⏸ queued[/yellow]" if self._queued_input is not None else ""
        return (
            f"turns: {nturns} | tokens: {tin}/{self.cfg.max_window_tokens} (out: {tout} cum) | "
            f"model: {self.cfg.model} | evicted: {ev_str}{queue_str}"
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

    async def on_input_submitted(self, event: "Input.Submitted") -> None:
        if event.input.id != "userinput":
            return
        text = (event.value or "").strip()
        if not text:
            return
        inp = self.query_one("#userinput", Input)
        inp.value = ""

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
                chat.write(f"[b green]assistant:[/b green] {self._stream_buffer}")
        except asyncio.CancelledError:
            try:
                self.store.append_turn(
                    "assistant",
                    [{"type": "text", "text": "[interrupted]"}],
                )
            except Exception:
                pass
            if self._stream_buffer:
                chat.write(f"[b green]assistant:[/b green] [dim]{self._stream_buffer}[/dim]")
            chat.write("[dim]⏹ turn interrupted[/dim]")
            raise
        except Exception as exc:
            log("tui_turn_error", error=str(exc))
            chat.write(f"[red]error:[/red] {exc}")
        finally:
            self._busy = False
            self._refresh_status()
            self._focus_input_after_refresh()
            # Drain the input queue — fire the next message automatically.
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

        if cmd == "/stop":
            await self._slash_stop(chat)
            return

        if cmd == "/evict":
            self._slash_evict(arg, chat)
            return

        if cmd == "/clear":
            self._slash_clear(chat)
            return

        if cmd == "/help":
            self._slash_help(chat)
            return

        chat.write(
            f"[dim]unknown command: {cmd} — type /help for the full list[/dim]"
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

    def _slash_evict(self, arg: str, chat: "RichLog") -> None:
        """/evict [tools|N] — context surgery.

        /evict          → show current eviction stats
        /evict tools    → strip tool_use blocks from all rows (free up bloat)
        /evict N        → drop the oldest N rows from the rolling window
        """
        arg = arg.strip()
        if not arg:
            ev = self.store.get_eviction_stats()
            chat.write(
                f"[bold]eviction stats[/bold]\n"
                f"  rows evicted   : {ev.get('rows_evicted', 0)}\n"
                f"  blocks evicted : {ev.get('blocks_evicted', 0)}\n"
                f"  bytes freed    : {ev.get('bytes_freed', 0)}"
            )
            return

        if arg == "tools":
            result = self.store.evict_tool_use_blocks(all_rows=True)
            rows_hit = result.get("rows_modified", 0)
            blocks_rm = result.get("blocks_evicted", 0)
            chat.write(
                f"[green]✓ tool_use blocks stripped[/green] — "
                f"{blocks_rm} block(s) removed from {rows_hit} row(s)"
            )
            return

        if arg.isdigit():
            n = int(arg)
            if n <= 0:
                chat.write("[red]N must be > 0[/red]")
                return
            removed = self.store.evict_last(n)
            chat.write(f"[green]✓ evicted[/green] {removed} oldest row(s)")
            return

        chat.write("[red]usage: /evict  |  /evict tools  |  /evict N[/red]")

    def _slash_clear(self, chat: "RichLog") -> None:
        """/clear — wipe the entire rolling window."""
        n = len(self.store.window())
        if n == 0:
            chat.write("[dim]rolling window is already empty[/dim]")
            return
        removed = self.store.evict_last(n)
        chat.write(f"[green]✓ cleared[/green] {removed} row(s) — rolling window wiped")

    def _slash_help(self, chat: "RichLog") -> None:
        """/help — print all available slash commands."""
        chat.write(
            "[bold]slash commands[/bold]\n"
            "  /quit, /exit          — close the panel\n"
            "  /stop                 — interrupt the current streaming turn\n"
            "  /models               — list available Gemma models\n"
            "  /swap MODEL           — switch model by index, alias, or name\n"
            "  /tokens N             — set rolling-window token cap\n"
            "  /export [N] [path]    — export chat turns to a markdown file\n"
            "  /evict                — show eviction stats\n"
            "  /evict tools          — strip tool_use blocks from all rows\n"
            "  /evict N              — drop oldest N rows from rolling window\n"
            "  /clear                — wipe the entire rolling window\n"
            "  /help                 — show this message"
        )

    async def _slash_export(self, arg: str, chat: "RichLog") -> None:
        """/export [N] [path] — write rolling-window text to markdown."""
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
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                _render_export_rows(rows, self.instance),
                encoding="utf-8",
            )
        except Exception as exc:
            chat.write(f"[red]export failed: {exc}[/red]")
            return
        chat.write(f"[green]exported {len(rows)} turn(s):[/green] {out_path}")

    # ---------------------------------------------------------------- actions

    def action_paste(self) -> None:
        try:
            import pyperclip
            text = pyperclip.paste()
            inp = self.query_one("#userinput", Input)
            cur = inp.cursor_position
            old = inp.value
            inp.value = old[:cur] + text + old[cur:]
            inp.cursor_position = cur + len(text)
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
