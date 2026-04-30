"""Textual TUI chat panel — primary UI for `mnemara run`.

A panel-style chat interface on top of AgentSession. The bare prompt-toolkit
REPL (mnemara/repl.py) remains as a scriptable / non-TTY fallback.

Layout (top to bottom):

    +--------------------------------------------------------------+
    | header: instance | model | role doc                          |
    +--------------------------------------------------------------+
    |                                                              |
    |   chat log (RichLog) — user right-aligned, assistant left,   |
    |   tool_use cards inline, scrollable                          |
    |                                                              |
    +--------------------------------------------------------------+
    | status: turns N/MAX | tokens K/MAX | model | evicted N       |
    +--------------------------------------------------------------+
    | input box (multi-line; Enter submits, Shift+Enter newline)   |
    +--------------------------------------------------------------+

Slash commands route to the same handlers as the bare REPL. Tool permissions
are mediated through AgentSession's existing can_use_tool callback (which
goes via repl-style permission_prompt by default; the TUI shows a modal).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from textual import events as _txt_events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static
    _TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TEXTUAL_AVAILABLE = False


# Ceiling on a single paste applied to the user input box. Beyond this, we
# truncate and warn — keeps a runaway paste from stalling render or breaking
# the rolling-window math when the next turn fires. Can be tuned later if
# real-use pastes ever come close.
_USERINPUT_PASTE_CAP = 16_000


class _UserInput(Input):
    """Input subclass with paste behavior tuned for Mnemara's userinput.

    Stock Textual `Input._on_paste` (.venv/.../widgets/_input.py:746) discards
    everything after the first line and inserts char-by-char via
    `insert_text_at_cursor`. Two surprises producers reported on this widget:

    1. **Ghosting + apparent freeze on multi-line paste.** Hypothesis: the
       per-line / per-cursor-position write path during a paste burst races
       with concurrent renders elsewhere on screen (status spinner ticking
       at 150ms, chatlog auto-scroll). Visual signature: stale characters
       linger near the input, terminal seems frozen until paste completes.
    2. **Silent line drop.** `splitlines()[0]` means a 5-line paste shows
       only the first line; producer assumes paste was incomplete. Single-
       line input is by design (Enter submits) so we collapse the multi-
       line content to one line by joining with spaces rather than dropping.

    This subclass overrides `_on_paste` to:
      - Join all lines with single spaces (collapse paragraphs, preserve word
        boundaries).
      - Truncate to `_USERINPUT_PASTE_CAP` chars to bound a runaway paste.
      - Atomically reassign `self.value` (one assignment, one render) instead
        of inserting char-by-char.
      - Stop the event from propagating.
      - Log the path taken so a future ghosting report can be diagnosed
        from the debug log without instrumenting again.
    """

    def _on_paste(self, event: "_txt_events.Paste") -> None:  # type: ignore[name-defined]
        # event.prevent_default() blocks the MRO walk in MessagePump from
        # ever reaching the parent Input._on_paste; without it both run and
        # the paste content gets inserted twice. event.stop() only stops
        # bubbling up the widget tree, not the same-widget MRO chain.
        event.prevent_default()
        event.stop()
        text = event.text or ""
        if not text:
            return
        lines = text.splitlines()
        # Multi-line collapse: join with single spaces, drop empty fragments
        # so a "abc\n\ndef" paste becomes "abc def" not "abc  def".
        collapsed = " ".join(part for part in (s.strip() for s in lines) if part)
        truncated = False
        if len(collapsed) > _USERINPUT_PASTE_CAP:
            collapsed = collapsed[:_USERINPUT_PASTE_CAP]
            truncated = True
        # Atomic value replacement at cursor.
        cur = self.cursor_position
        old = self.value
        self.value = old[:cur] + collapsed + old[cur:]
        self.cursor_position = cur + len(collapsed)
        try:
            log(
                "tui_paste",
                bytes_in=len(text),
                lines=len(lines),
                bytes_inserted=len(collapsed),
                truncated=truncated,
            )
        except Exception:
            pass

from . import config as config_mod
from . import inbox as inbox_mod
from . import paths
from .agent import AgentSession
from .config import Config
from .logging_util import log, set_log_path
from .permissions import PermissionStore
from .store import Store
from .tools import ToolRunner, parse_proposal_file, write_memory


def _parse_size(s: str) -> int:
    """Parse a human-readable integer size: '500', '500k', '1m', '1_000_000'.

    Used by /turns and /tokens. Returns the parsed int. Raises ValueError on
    malformed input. Suffixes are case-insensitive: k=1_000, m=1_000_000.
    Underscores in the digit portion are ignored (Python literal style).
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


DEFAULT_CSS = """
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
    /* Always reserve the scrollbar gutter so the bar is visible at startup
       (not only after content overflows) — gives a consistent click target. */
    overflow-y: scroll;
    scrollbar-background: #1a1f2b;
    scrollbar-background-hover: #1a1f2b;
    scrollbar-background-active: #1a1f2b;
    scrollbar-color: #4d6fa3;
    scrollbar-color-hover: #6f9ad9;
    scrollbar-color-active: #8fb6e6;
    /* Two-cell-wide bar so it's easier to grab with the mouse. */
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

PermissionModal {
    align: center middle;
}

#permbox {
    width: 70;
    height: auto;
    padding: 1 2;
    border: thick #6f9ad9;
    background: #1a1f2b;
}

#permbuttons {
    height: 3;
    align-horizontal: center;
}

NoteModal {
    align: center middle;
}

#notebox {
    width: 80;
    height: auto;
    padding: 1 2;
    border: thick #6f9ad9;
    background: #1a1f2b;
}
"""


def _is_tty() -> bool:
    import sys
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


# ---------------------------------------------------------------- modal screens


class PermissionModal(ModalScreen):  # type: ignore[misc]
    """Modal: ask the user whether to allow a tool invocation."""

    def __init__(self, tool: str, target: str) -> None:
        super().__init__()
        self.tool = tool
        self.target = target

    def compose(self) -> "ComposeResult":
        with Container(id="permbox"):
            yield Label(f"[b]Permission requested:[/b] {self.tool}")
            yield Static(f"[dim]target:[/dim] {self.target}")
            with Horizontal(id="permbuttons"):
                yield Button("Allow", id="allow", variant="primary")
                yield Button("Always", id="always", variant="success")
                yield Button("Session", id="session")
                yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: "Button.Pressed") -> None:
        bid = event.button.id or "deny"
        mapping = {
            "allow": "allow",
            "always": "allow_always",
            "session": "allow_session",
            "deny": "deny",
        }
        self.dismiss(mapping.get(bid, "deny"))


class NoteModal(ModalScreen):  # type: ignore[misc]
    """Modal: capture a memory note + optional category."""

    def compose(self) -> "ComposeResult":
        with Container(id="notebox"):
            yield Label("[b]Memory note[/b]")
            yield Input(placeholder="text", id="note_text")
            yield Input(placeholder="category (default: user_note)", id="note_category")
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: "Button.Pressed") -> None:
        if event.button.id == "save":
            text = self.query_one("#note_text", Input).value.strip()
            cat = self.query_one("#note_category", Input).value.strip() or "user_note"
            self.dismiss((text, cat))
        else:
            self.dismiss(None)


# --------------------------------------------------------------------- main app


class MnemaraTUI(App):  # type: ignore[misc]
    CSS = DEFAULT_CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear log"),
        Binding("ctrl+i", "focus_input", "Focus input", priority=True, show=False),
        Binding("escape", "focus_input", "Focus input", show=False),
        Binding("ctrl+v", "paste", "Paste", priority=True, show=False),
        Binding("ctrl+y", "copy_last", "Copy last response", priority=True, show=False),
        Binding("pageup", "scroll_log_up", "Scroll up", show=False),
        Binding("pagedown", "scroll_log_down", "Scroll down", show=False),
    ]

    def __init__(self, instance: str) -> None:
        super().__init__()
        self.instance = instance
        self.cfg: Config = config_mod.load(instance)
        set_log_path(paths.debug_log(instance))
        self.store = Store(instance)
        self.perms = PermissionStore(instance)
        self._pending_perm: dict[str, Any] = {}

        # Permission prompt routed through a Textual modal. SDK's can_use_tool
        # is async, but ToolRunner._check_perm calls our prompt synchronously.
        # We expose a sync prompt that calls back into the running event loop
        # via call_from_thread; the SDK callback runs on the app's loop, so
        # we use a sync wrapper that returns "deny" and rely on the modal-based
        # flow handled via the on_tool_use stream cards instead.
        self.runner = ToolRunner(
            instance,
            self.cfg,
            self.perms,
            prompt=self._sync_permission_prompt,
        )
        self.session = AgentSession(self.cfg, self.store, self.runner, client=None)

        self._busy = False
        self._stream_buffer = ""
        self._stream_chars = 0
        self._evicted_total = 0
        self._copy_flash: str = ""

        # Spinner state. The spinner ticks via a Textual interval timer at
        # 150ms — fast enough to feel alive, slow enough to not flood the
        # render queue or contend with streaming. The cached static portion
        # of the status text avoids re-running DB queries on every tick.
        self._spinner_idx = 0
        self._spinner_was_busy = False
        self._cached_status_static = ""
        self._spinner_timer = None  # set in on_mount

    # ------------------------------------------------------------- composition

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
            placeholder="message  (Enter to send, /help for commands, Ctrl+C to quit)",
            id="userinput",
        )
        yield Footer()

    def _on_resize(self, event: "_txt_events.Resize") -> None:  # type: ignore[name-defined]
        """Logs each resize event for diagnosis of resize-during-streaming.

        Resize-during-streaming root cause (resolved 2026-04-30): until the
        worker-pattern fix in on_input_submitted, this handler effectively
        could not run during a stream. on_input_submitted was awaiting
        _send_turn directly, which blocked Textual's _process_messages_loop
        on dispatch_message for the entire stream. Queued resize events
        sat in the message queue but couldn't be dispatched until our
        handler returned (~50s post-stream). Producer-confirmed via this
        very log: a mid-stream resize emitted no state="streaming" entry,
        only a state="idle" one ~50s later when the stream had ended.

        After the run_worker fix in on_input_submitted, mid-stream resize
        events emit state="streaming" entries here in real time. If the
        bug recurs (visuals break + log shows state="streaming" entries
        at the breakage timestamps), that's a different problem: dispatch
        is happening but layout isn't sticking. Branch B fixes apply:
        call_after_refresh second-pass refresh, manual child walks, or
        _compositor.full_map invalidation, scoped to self._busy.

        IMPORTANT: do NOT call event.prevent_default() here. Textual's
        App._on_resize must still run AFTER us in the MRO walk so the
        actual layout reflow happens.
        See ~/.mnemara/substrate/wiki/textual_subclass_mro_dispatch.md.
        """
        try:
            log(
                "tui_resize",
                state="streaming" if self._busy else "idle",
                w=event.size.width,
                h=event.size.height,
                stream_chars=self._stream_chars if self._busy else 0,
            )
        except Exception:
            pass
        # Intentionally do not call event.stop() either; let resize
        # bubble normally through the framework.

    def on_mount(self) -> None:
        log("tui_start", instance=self.instance, model=self.cfg.model)
        # Replace Textual's mouse-enable sequence with one that uses only
        # safe modes. Textual's default enables 1000 (basic click) +
        # 1003 (any-event motion: chatty + battery drain) + 1015 (urxvt
        # encoding: emits raw high bytes when coords > 223, crashes the
        # input UTF-8 decoder on wide terminals) + 1006 (SGR safe).
        #
        # We replace it with: 1000 (click) + 1002 (button-event drag,
        # needed for scrollbar drag interaction) + 1006 (SGR safe). No
        # 1003 (motion noise) and crucially no 1015 (the byte-0xd5
        # crasher).
        #
        # Trade-off: native click-and-drag text selection no longer works
        # without modifier — most terminals honor shift+drag to bypass
        # mouse capture and use native selection (iTerm2, Terminal.app,
        # GNOME Terminal, kitty, Windows Terminal, Alacritty all support
        # this). Mouse-wheel scrolling and scrollbar interaction now work.
        try:
            drv = self._driver
            if drv is not None:
                # Disable any unsafe modes that may have been enabled by
                # Textual's driver before we got here.
                drv.write(self._MOUSE_DISABLE_UNSAFE)
                # Enable our safe set immediately.
                drv.write(self._MOUSE_ENABLE_SAFE)
                drv.flush()
                # Replace the driver's enabler so the SIGCONT / iTerm
                # workaround (linux_driver.py:288) re-enables with our
                # safe sequence instead of Textual's default.
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
        except Exception:
            pass
        # Prime the cached status text so the first paint has full content.
        self._cached_status_static = self._compute_status_text()
        # Start the spinner ticker. 150ms is fast enough to feel alive but
        # slow enough that even 100% packed 8h sessions cost trivial CPU.
        # The callback bails out cheaply when not busy.
        try:
            self._spinner_timer = self.set_interval(0.15, self._tick_spinner)
        except Exception:
            self._spinner_timer = None
        self._render_history()
        self._focus_input_after_refresh()

    def _focus_input_after_refresh(self) -> None:
        """Schedule input focus after the next paint.

        Calling .focus() inline can race redraws — Textual may settle focus
        elsewhere on the next paint (e.g. when RichLog grows during streaming).
        call_after_refresh defers until the screen has settled.
        """
        def _do_focus() -> None:
            try:
                self.query_one("#userinput", Input).focus()
            except Exception:
                pass
        try:
            self.call_after_refresh(_do_focus)
        except Exception:
            _do_focus()

    # -------------------------------------------------------------- rendering

    def _chat(self) -> "RichLog":
        return self.query_one("#chatlog", RichLog)

    # Braille-pattern spinner frames. 10 frames at 150ms = full rotation in 1.5s.
    _SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def _compute_status_text(self) -> str:
        """Compute the static portion of the status line.

        Runs the DB queries (window, total_tokens, role_proposals_count,
        inbox count_pending). Called only on turn boundaries / explicit
        refresh — NOT on every spinner tick. Result is cached in
        self._cached_status_static.
        """
        rows = self.store.window()
        n_turns = len(rows)
        # `tin` is the content-length/4 estimate of the rolling window's
        # stored size in tokens. This is what `Store.evict()` enforces
        # against `max_window_tokens`. `tout` is cumulative API output
        # tokens across the session -- informational, NOT a cap input.
        # Showing `tin + tout` against the cap (the previous behavior)
        # was misleading: the displayed total could cross the cap while
        # the enforced metric stayed well under, so eviction correctly
        # didn't fire but the producer saw "over cap, no evictions" as
        # a bug. Display the enforced metric against the cap and surface
        # cumulative output tokens separately.
        tin, tout = self.store.total_tokens()
        base = (
            f"turns: {n_turns}/{self.cfg.max_window_turns} | "
            f"tokens: {tin}/{self.cfg.max_window_tokens} (out: {tout} cum) | "
            f"model: {self.cfg.model} | evicted: {self._evicted_total}"
        )
        try:
            n_prop = paths.role_proposals_count(self.instance)
            if n_prop > 0:
                base += f" | [yellow]📋 {n_prop} proposal{'s' if n_prop != 1 else ''}[/yellow]"
        except Exception:
            pass
        try:
            db = getattr(self.cfg, "architect_db_path", "") or ""
            peers = getattr(self.cfg, "peer_roles", ["theseus", "majordomo"])
            if db:
                n_inbox = inbox_mod.count_pending(db, peers, exclude_role=self.instance)
                if n_inbox > 0:
                    base += f" | [yellow]I {n_inbox} pending[/yellow]"
        except Exception:
            pass
        if self._copy_flash:
            base += f" | [green]{self._copy_flash}[/green]"
        return base

    def _status_text(self) -> str:
        """Backward-compat: callers that want the full status line.

        Internally uses _compute_status_text; kept for tests.
        """
        return self._compute_status_text()

    def _render_status_widget(self) -> None:
        """Push current cached status (with spinner if busy) to the widget.

        Called on every spinner tick AND on every _refresh_status() call.
        Cheap — no DB queries; just a string format and Static.update().
        """
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
        """Recompute the cached static portion + push to widget.

        Called on turn boundaries, copy flash transitions, and similar
        state changes. Spinner ticks DO NOT call this — they only call
        _render_status_widget().
        """
        self._cached_status_static = self._compute_status_text()
        self._render_status_widget()

    def _tick_spinner(self) -> None:
        """Interval-timer callback. Updates spinner cell only.

        Triggered every 150ms regardless of busy state. When busy:
        advance the frame and re-render. When idle: clear the spinner
        once on the busy→idle transition, then no-op until the next
        turn starts.
        """
        if self._busy:
            self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER_FRAMES)
            self._spinner_was_busy = True
            self._render_status_widget()
            return
        # Idle. Only push an update on the busy→idle edge so we don't
        # spam Static.update() at 7Hz when nothing's happening.
        if self._spinner_was_busy:
            self._spinner_was_busy = False
            self._spinner_idx = 0
            self._render_status_widget()

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
                        log_widget.write(
                            f"[dim]  result: {str(c)[:200]}[/dim]"
                        )
            else:
                log_widget.write(f"[b green]assistant:[/b green] {content}")

    # ----------------------------------------------------------------- events

    async def on_input_submitted(self, event: "Input.Submitted") -> None:
        if event.input.id != "userinput":
            return
        text = (event.value or "").strip()
        if not text or self._busy:
            return
        inp = self.query_one("#userinput", Input)
        inp.value = ""

        if text.startswith("/"):
            await self._handle_slash(text)
            self._refresh_status()
            return

        # Spawn streaming as a Textual worker so this handler returns
        # immediately. CRITICAL for the resize-during-streaming bug:
        # Textual's _process_messages_loop (textual/message_pump.py:634)
        # awaits _dispatch_message on each event, which means while we
        # await _send_turn here, the pump is suspended on US for the
        # entire stream duration. Queued resize / key / mouse events
        # can't be dispatched until our handler returns -- no amount
        # of asyncio.sleep(0) inside the SDK iterator helps because
        # the pump's *task* is parked on dispatch, not waiting on the
        # queue. Producer-confirmed 2026-04-30 via debug.log: a resize
        # mid-stream emitted no state=streaming entry; the queued
        # event only fired ~50s later, post-stream, with state=idle.
        # Worker pattern decouples streaming from event dispatch so
        # the pump runs concurrently and processes events at normal
        # cadence. exclusive=True cancels any prior in-flight stream
        # if the producer submits a new prompt before the prior turn
        # finishes.
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
        self._stream_chars = 0

        async def on_token(t: str) -> None:
            # Buffer streamed text; flush the complete message once the turn
            # finishes. Don't call widget.update() per-token — that floods the
            # Textual render queue and races with resize events, which was the
            # source of the input-box duplication during streaming + resize.
            self._stream_buffer += t
            self._stream_chars += len(t)

        async def on_tool_use(name: str, inp: dict) -> None:
            chat.write(f"[b magenta]> tool:[/b magenta] [magenta]{name}[/magenta]({_short(inp)})")

        async def on_tool_result(tid: str, content: Any, is_error: bool) -> None:
            tag = "[red]error[/red]" if is_error else "[dim]result[/dim]"
            chat.write(f"  {tag}: {str(content)[:300]}")

        try:
            usage = await self.session.turn_async(
                text,
                on_token=on_token,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
            )
            self._evicted_total += int(usage.get("evicted", 0) or 0)
            if self._stream_buffer:
                chat.write(f"[b green]assistant:[/b green] {self._stream_buffer}")
        except Exception as exc:
            log("tui_turn_error", error=str(exc))
            chat.write(f"[red]error:[/red] {exc}")
        finally:
            self._busy = False
            self._refresh_status()
            self._focus_input_after_refresh()

    # ------------------------------------------------------------ slash cmds

    async def _handle_slash(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        chat = self._chat()

        if cmd in ("/quit", "/exit"):
            self.exit()
            return

        if cmd == "/help":
            chat.write(
                "[b]Slash commands:[/b]\n"
                "  /role <path>     swap role doc (also persists to config)\n"
                "  /show            print the rolling window\n"
                "  /clear           wipe the rolling window\n"
                "  /swap <model>    switch model for this and future sessions\n"
                "  /note [text]     append to today's memory file (modal if no text)\n"
                "  /proposals       list pending role-amendment proposals\n"
                "  /inbox           list pending pings from peer panels\n"
                "  /copy [all|N]    copy to clipboard: last response (default), all turns, or last N turns\n"
                "  /window          print current max_window_turns and max_window_tokens\n"
                "  /turns N [--temp]   set max_window_turns (persists unless --temp)\n"
                "  /tokens N [--temp]  set max_window_tokens (accepts 500k, 1m, 1000000; persists unless --temp)\n"
                "  /show ids        print rolling window with row ids (target evict)\n"
                "  /mark <name>     insert a named segment marker at this point\n"
                "  /marks           list all segment markers in the window\n"
                "  /evict last N           drop the N most-recent rows\n"
                "  /evict ids 4,7,9        drop specific rows by id (see /show ids)\n"
                "  /evict since <name>     drop the named marker and everything after it\n"
                "  /quit, /exit     exit\n"
                "[b]Key bindings:[/b]\n"
                "  Ctrl+Y           copy last assistant response to clipboard\n"
                "  Ctrl+V           paste from clipboard into input"
            )
            return

        if cmd == "/role":
            if not arg:
                chat.write("[red]usage: /role <path>[/red]")
                return
            self.cfg.role_doc_path = str(Path(arg).expanduser())
            config_mod.save(self.instance, self.cfg)
            self.sub_title = f"model={self.cfg.model}  role={self.cfg.role_doc_path}"
            chat.write(f"[green]role doc set to[/green] {self.cfg.role_doc_path}")
            return

        if cmd == "/show":
            if arg.strip() == "ids":
                self._slash_show_ids(chat)
            else:
                self._render_history()
            return

        if cmd == "/clear":
            self.store.clear()
            chat.clear()
            chat.write("[green]window cleared[/green]")
            return

        if cmd == "/swap":
            if not arg:
                chat.write("[red]usage: /swap <model>[/red]")
                return
            self.cfg.model = arg
            config_mod.save(self.instance, self.cfg)
            self.sub_title = f"model={self.cfg.model}  role={self.cfg.role_doc_path or '(none)'}"
            chat.write(f"[green]model set to[/green] {self.cfg.model}")
            return

        if cmd == "/note":
            if arg:
                p = write_memory(self.instance, arg, category="user_note")
                chat.write(f"[green]appended to[/green] {p}")
                return
            result = await self.push_screen_wait(NoteModal())
            if result:
                text, cat = result
                if text:
                    p = write_memory(self.instance, text, category=cat)
                    chat.write(f"[green]appended to[/green] {p} [dim]({cat})[/dim]")
            return

        if cmd == "/proposals":
            prop_dir = paths.role_proposals_dir(self.instance)
            if not prop_dir.exists() or not list(prop_dir.glob("*.md")):
                chat.write("No pending proposals.")
                return
            files = sorted(prop_dir.glob("*.md"), reverse=True)
            chat.write(f"[b]📋 {len(files)} pending proposal{'s' if len(files) != 1 else ''}:[/b]")
            for f in files:
                severity, preview = parse_proposal_file(f)
                chat.write(f"  [[yellow]{severity}[/yellow]] {f.name} — {preview}")
            return

        if cmd == "/copy":
            await self._slash_copy(arg.strip(), chat)
            return

        if cmd == "/inbox":
            db = getattr(self.cfg, "architect_db_path", "") or ""
            if not db:
                chat.write("[dim]inbox: not configured (set architect_db_path in config)[/dim]")
                return
            peers = getattr(self.cfg, "peer_roles", ["theseus", "majordomo"])
            pings = inbox_mod.peek_pending_pings(db, peers, exclude_role=self.instance)
            chat.write(inbox_mod.format_inbox(pings))
            return

        if cmd == "/window":
            tin, tout = self.store.total_tokens()
            n_turns = len(self.store.window())
            chat.write(
                f"[b]window caps[/b]\n"
                f"  turns:   {n_turns} / {self.cfg.max_window_turns}\n"
                f"  tokens:  {tin} / {self.cfg.max_window_tokens}  "
                f"[dim](enforced metric: rolling-window stored size)[/dim]\n"
                f"  out:     {tout} cumulative  "
                f"[dim](session-total API output tokens, not a cap input)[/dim]"
            )
            return

        if cmd == "/turns":
            await self._slash_set_window(arg, chat, field="turns")
            return

        if cmd == "/tokens":
            await self._slash_set_window(arg, chat, field="tokens")
            return

        if cmd == "/marks":
            self._slash_marks(chat)
            return

        if cmd == "/mark":
            self._slash_mark(arg, chat)
            return

        if cmd == "/evict":
            self._slash_evict(arg, chat)
            return

        chat.write(f"[red]unknown command:[/red] {cmd}  (try /help)")

    # ---------------------------------------------------------------- actions

    def action_clear_log(self) -> None:
        self._chat().clear()

    def action_focus_input(self) -> None:
        self._focus_input_after_refresh()

    def action_scroll_log_up(self) -> None:
        try:
            self._chat().scroll_page_up()
        except Exception:
            pass

    def action_scroll_log_down(self) -> None:
        try:
            self._chat().scroll_page_down()
        except Exception:
            pass

    # ANSI mouse sequences. Only the *unsafe* modes are disabled — we keep
    # mouse tracking on so the scrollbar and mouse-wheel work, but use
    # encoding modes that don't crash Textual's UTF-8 input decoder.
    #
    # Disabled: 1003 (any-event motion: chatty + battery drain),
    #           1005 (UTF-8 ext encoding: high bytes break decoder),
    #           1015 (urxvt encoding: high bytes break decoder when
    #                 terminal width > 223 chars; the original 0xd5 crash).
    _MOUSE_DISABLE_UNSAFE = "\x1b[?1003l\x1b[?1005l\x1b[?1015l"
    # Enabled: 1000 (basic click tracking),
    #          1002 (button-event motion: needed for scrollbar drag),
    #          1006 (SGR ext encoding: text-only chars, decoder-safe).
    _MOUSE_ENABLE_SAFE = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"

    _paste_unavailable_warned: bool = False

    def action_paste(self) -> None:
        """Insert clipboard text at the cursor of the focused Input widget.

        Behavior matches `_UserInput._on_paste` (terminal-bracketed-paste path)
        when targeting the userinput widget: multi-line content collapses to
        a single space-joined line, capped at `_USERINPUT_PASTE_CAP` chars.
        For other Input widgets (modal note-text input) the raw text is
        inserted as-is so multi-paragraph notes still work.
        """
        try:
            import pyperclip  # type: ignore[import]
            paste_text: str = pyperclip.paste()
        except Exception as exc:
            log("tui_paste_unavailable", error=str(exc))
            if not MnemaraTUI._paste_unavailable_warned:
                MnemaraTUI._paste_unavailable_warned = True
                self._chat().write(
                    "[dim][paste unavailable: install pyperclip or set up clipboard backend][/dim]"
                )
            return

        if not paste_text:
            return

        focused = self.focused
        if not isinstance(focused, Input):
            return

        # Userinput-targeted paste mirrors the terminal-bracketed-paste path
        # so producer experience is identical regardless of paste mechanism.
        if getattr(focused, "id", None) == "userinput":
            lines = paste_text.splitlines()
            collapsed = " ".join(part for part in (s.strip() for s in lines) if part)
            if len(collapsed) > _USERINPUT_PASTE_CAP:
                collapsed = collapsed[:_USERINPUT_PASTE_CAP]
            paste_text = collapsed
            if not paste_text:
                return

        cur = focused.cursor_position
        focused.value = focused.value[:cur] + paste_text + focused.value[cur:]
        focused.cursor_position = cur + len(paste_text)

    _copy_unavailable_warned: bool = False

    def _extract_last_assistant(self) -> str:
        """Return the text of the most recent assistant turn, or ''."""
        rows = self.store.window()
        for row in reversed(rows):
            if row.get("role") == "assistant":
                return _flatten_assistant_content(row["content"])
        return ""

    def _extract_window_as_text(self, last_n: int | None = None) -> str:
        """Return the conversation window (or last N rows) as plain text."""
        rows = self.store.window()
        if last_n is not None:
            rows = rows[-last_n:]
        lines: list[str] = []
        for row in rows:
            role = row.get("role", "?")
            if role == "user":
                lines.append(f"you: {_flatten_text_blocks(row['content'])}")
            elif role == "assistant":
                lines.append(f"assistant: {_flatten_assistant_content(row['content'])}")
        return "\n\n".join(lines)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard via pyperclip. Returns True on success."""
        try:
            import pyperclip  # type: ignore[import]
            pyperclip.copy(text)
            return True
        except Exception as exc:
            log("tui_copy_unavailable", error=str(exc))
            if not MnemaraTUI._copy_unavailable_warned:
                MnemaraTUI._copy_unavailable_warned = True
                self._chat().write(
                    "[dim][copy unavailable: install pyperclip or set up clipboard backend][/dim]"
                )
            return False

    def _flash_copy(self, n_chars: int) -> None:
        """Show a brief 'copied N chars' indicator in the status bar for 2s."""
        self._copy_flash = f"copied {n_chars} chars"
        self._refresh_status()
        self.set_timer(2.0, self._clear_copy_flash)

    def _clear_copy_flash(self) -> None:
        self._copy_flash = ""
        self._refresh_status()

    def action_copy_last(self) -> None:
        """Ctrl+Y — copy the last assistant response to the system clipboard."""
        text = self._extract_last_assistant()
        if not text:
            self._chat().write("[dim](no assistant response to copy)[/dim]")
            return
        if self._copy_to_clipboard(text):
            self._flash_copy(len(text))

    async def _slash_set_window(
        self, arg: str, chat: "RichLog", *, field: str
    ) -> None:
        """/turns N [--temp] and /tokens N [--temp] handler.

        field='turns'  -> mutates cfg.max_window_turns  (bounds [1, 10000])
        field='tokens' -> mutates cfg.max_window_tokens (bounds [1000, 10_000_000];
                          accepts shorthand like '500k' / '1m')

        Persists to config.json by default. Pass --temp as second arg for
        in-memory only (revert on restart).
        """
        parts = arg.split()
        if not parts:
            chat.write(
                f"[red]usage: /{field} N [--temp]  "
                f"(N is a positive integer"
                f"{'; tokens accepts 500k, 1m, etc.' if field == 'tokens' else ''}"
                f")[/red]"
            )
            return
        raw = parts[0]
        temp = len(parts) > 1 and parts[1].lower() in ("--temp", "-t", "temp")
        try:
            n = _parse_size(raw)
        except ValueError as e:
            chat.write(f"[red]invalid value '{raw}': {e}[/red]")
            return
        if field == "turns":
            if not 1 <= n <= 10000:
                chat.write("[red]turns must be between 1 and 10000[/red]")
                return
            old = self.cfg.max_window_turns
            self.cfg.max_window_turns = n
            label = "turns"
        else:
            if not 1000 <= n <= 10_000_000:
                chat.write(
                    "[red]tokens must be between 1000 and 10000000 (10M)[/red]"
                )
                return
            old = self.cfg.max_window_tokens
            self.cfg.max_window_tokens = n
            label = "tokens"
        if not temp:
            try:
                config_mod.save(self.instance, self.cfg)
                persist_note = "(persisted to config.json)"
            except Exception as exc:
                persist_note = f"[red](persist failed: {exc})[/red]"
        else:
            persist_note = "(in-memory only — reverts on restart)"
        chat.write(
            f"[green]{label}: {old} → {n}[/green]  [dim]{persist_note}[/dim]"
        )
        self._refresh_status()

    def _slash_show_ids(self, chat: "RichLog") -> None:
        """`/show ids` — print the rolling window with row ids prepended.

        Useful before targeting `/evict ids ...` so the producer can see
        exactly which rows hold which content. Marker rows get a
        distinct prefix so they're easy to spot.
        """
        rows = self.store.window()
        if not rows:
            chat.write("[dim](empty window)[/dim]")
            return
        chat.write("[b]rolling window (id | role | preview):[/b]")
        for row in rows:
            rid = row["id"]
            role = row["role"]
            content = row["content"]
            if role == "marker":
                # content is the marker name (json-encoded scalar)
                name = content if isinstance(content, str) else str(content)
                chat.write(
                    f"  [dim]#{rid}[/dim] [yellow]⚑ marker[/yellow] [b]{name}[/b]"
                )
                continue
            # User/assistant rows: short preview of the content
            text = _flatten_text_blocks(content) if role == "user" else _flatten_assistant_content(content)
            preview = text.replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            color = "cyan" if role == "user" else "green"
            chat.write(f"  [dim]#{rid}[/dim] [{color}]{role:>9}[/{color}]  {preview}")

    def _slash_marks(self, chat: "RichLog") -> None:
        """`/marks` — list every segment marker in the current window."""
        marks = self.store.list_markers()
        if not marks:
            chat.write("[dim]no segment markers in window[/dim]")
            return
        chat.write(f"[b]⚑ {len(marks)} segment marker{'s' if len(marks) != 1 else ''}:[/b]")
        for m in marks:
            chat.write(f"  [dim]#{m['id']}[/dim] [b yellow]{m['name']}[/b yellow]  [dim]{m['ts']}[/dim]")

    def _slash_mark(self, arg: str, chat: "RichLog") -> None:
        """`/mark <name>` — insert a segment marker at the current tail."""
        name = arg.strip()
        if not name:
            chat.write("[red]usage: /mark <name>  (use a short token like 'pre-aethon-detour')[/red]")
            return
        try:
            mid = self.store.mark_segment(name)
        except Exception as exc:
            chat.write(f"[red]marker insert failed: {exc}[/red]")
            return
        chat.write(f"[green]⚑ marker '{name}' inserted at #{mid}[/green]")
        self._refresh_status()

    def _slash_evict(self, arg: str, chat: "RichLog") -> None:
        """`/evict last N` | `/evict ids 4,7,9` | `/evict since <name>`."""
        parts = arg.split(None, 1)
        if not parts:
            chat.write(
                "[red]usage:[/red]\n"
                "  /evict last N            drop the N most-recent rows\n"
                "  /evict ids 4,7,9         drop specific rows by id\n"
                "  /evict since <name>      drop named marker + everything after"
            )
            return
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "last":
            try:
                n = int(rest.strip())
            except ValueError:
                chat.write("[red]usage: /evict last N  (N must be a positive integer)[/red]")
                return
            if n <= 0:
                chat.write("[red]N must be > 0[/red]")
                return
            deleted = self.store.evict_last(n)
            chat.write(f"[green]evicted {deleted} row{'s' if deleted != 1 else ''}[/green]")
            self._refresh_status()
            return
        if sub == "ids":
            try:
                ids = [int(x.strip()) for x in rest.replace(",", " ").split() if x.strip()]
            except ValueError:
                chat.write("[red]usage: /evict ids 4,7,9  (comma- or space-separated row ids)[/red]")
                return
            if not ids:
                chat.write("[red]no ids provided[/red]")
                return
            deleted = self.store.evict_ids(ids)
            chat.write(
                f"[green]evicted {deleted}/{len(ids)} row{'s' if len(ids) != 1 else ''}[/green]"
                + (f"  [dim](missing: {len(ids) - deleted})[/dim]" if deleted != len(ids) else "")
            )
            self._refresh_status()
            return
        if sub == "since":
            name = rest.strip()
            if not name:
                chat.write("[red]usage: /evict since <name>[/red]")
                return
            deleted = self.store.evict_since(name)
            if deleted == 0:
                chat.write(f"[yellow]no marker named '{name}' — use /marks to list[/yellow]")
            else:
                chat.write(
                    f"[green]evicted {deleted} row{'s' if deleted != 1 else ''} from marker '{name}' onward[/green]"
                )
                self._refresh_status()
            return
        chat.write(f"[red]unknown evict mode '{sub}'  (use last|ids|since)[/red]")

    async def _slash_copy(self, arg: str, chat: "RichLog") -> None:
        """/copy [all|N] — copy turns to clipboard."""
        if not arg or arg == "last":
            text = self._extract_last_assistant()
            if not text:
                chat.write("[dim](no assistant response to copy)[/dim]")
                return
        elif arg == "all":
            text = self._extract_window_as_text()
            if not text:
                chat.write("[dim](window is empty)[/dim]")
                return
        else:
            try:
                n = int(arg)
                if n <= 0:
                    raise ValueError
            except ValueError:
                chat.write(f"[red]usage: /copy [all|N]  (N must be a positive integer)[/red]")
                return
            rows = self.store.window()
            n = min(n, len(rows))
            text = self._extract_window_as_text(last_n=n)
            if not text:
                chat.write("[dim](window is empty)[/dim]")
                return

        if self._copy_to_clipboard(text):
            self._flash_copy(len(text))

    async def action_quit(self) -> None:
        self.exit()

    def on_unmount(self) -> None:
        try:
            self.session.write_session_stats()
        except Exception as e:
            log("tui_stats_error", error=str(e))
        try:
            if self.session.role_proposals > 0:
                n = self.session.role_proposals
                p = paths.role_proposals_dir(self.instance)
                print(
                    f"📋 {n} role-amendment proposal(s) written this session. "
                    f"Review at {p}"
                )
        except Exception:
            pass
        try:
            self.store.close()
        except Exception:
            pass
        log("tui_stop", instance=self.instance)

    # -------------------------------------------------------------- perm hook

    def _sync_permission_prompt(self, tool: str, target: str) -> str:
        # Called from sync ToolRunner contexts (test paths). The SDK path does
        # not actually go through ToolRunner.dispatch — it goes through
        # can_use_tool which calls runner._check_perm directly, and the policy
        # decides without prompting unless mode == "ask". For "ask" inside the
        # TUI we'd need a sync->async bridge; not exercised by the SDK loop in
        # current Mnemara, since policies default to allow/deny pre-set modes.
        return "deny"


# --------------------------------------------------------------------- helpers


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


def _flatten_assistant_content(content: Any) -> str:
    """Extract plain text from an assistant content value (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            text = b.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------- entry point


def run(instance: str) -> bool:
    """Launch the Textual TUI for an instance.

    Returns True on clean exit. Returns False (and prints a warning) if
    Textual is unavailable or the terminal is not a TTY — caller should
    fall back to the bare REPL.
    """
    if not _TEXTUAL_AVAILABLE:
        return False
    if not _is_tty():
        return False
    app = MnemaraTUI(instance)
    app.run()
    return True
