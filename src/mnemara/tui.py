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
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static
    _TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TEXTUAL_AVAILABLE = False

from . import config as config_mod
from . import inbox as inbox_mod
from . import paths
from .agent import AgentSession
from .config import Config
from .logging_util import log, set_log_path
from .permissions import PermissionStore
from .store import Store
from .tools import ToolRunner, parse_proposal_file, write_memory


DEFAULT_CSS = """
Screen {
    background: #1d2330;
    color: #e6e6e6;
}

#chatlog {
    height: 1fr;
    border: round #3a4256;
    background: #1a1f2b;
    padding: 1 2;
}

#status {
    height: 1;
    background: #11151e;
    color: #8aa1c1;
    padding: 0 1;
}

#userinput {
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
        self._evicted_total = 0

    # ------------------------------------------------------------- composition

    def compose(self) -> "ComposeResult":
        role = self.cfg.role_doc_path or "(none)"
        self.title = f"mnemara: {self.instance}"
        self.sub_title = f"model={self.cfg.model}  role={role}"
        yield Header(show_clock=False)
        with Vertical():
            yield RichLog(id="chatlog", wrap=True, markup=True, highlight=False)
            yield Static(self._status_text(), id="status")
            yield Input(
                placeholder="message  (Enter to send, /help for commands, Ctrl+C to quit)",
                id="userinput",
            )
        yield Footer()

    def on_mount(self) -> None:
        log("tui_start", instance=self.instance, model=self.cfg.model)
        self._render_history()
        self.query_one("#userinput", Input).focus()

    # -------------------------------------------------------------- rendering

    def _chat(self) -> "RichLog":
        return self.query_one("#chatlog", RichLog)

    def _status_text(self) -> str:
        rows = self.store.window()
        n_turns = len(rows)
        tin, tout = self.store.total_tokens()
        total = tin + tout
        base = (
            f"turns: {n_turns}/{self.cfg.max_window_turns} | "
            f"tokens: {total}/{self.cfg.max_window_tokens} ({tin} in / {tout} out) | "
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
        return base

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

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
        text = (event.value or "").strip()
        if not text or self._busy:
            return
        inp = self.query_one("#userinput", Input)
        inp.value = ""

        if text.startswith("/"):
            await self._handle_slash(text)
            self._refresh_status()
            return

        await self._send_turn(text)

    async def _send_turn(self, text: str) -> None:
        self._busy = True
        chat = self._chat()
        chat.write(f"[b cyan]you:[/b cyan] {text}")
        self._stream_buffer = ""
        self._stream_chars = 0

        async def on_token(t: str) -> None:
            # RichLog has no "update last line" affordance, so we buffer the
            # streamed text and flush the complete assistant message once the
            # turn finishes. The status bar shows live progress.
            self._stream_buffer += t
            self._stream_chars += len(t)
            self.query_one("#status", Static).update(
                self._status_text() + f"  [streaming {self._stream_chars} chars]"
            )

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
                "  /quit, /exit     exit"
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

        if cmd == "/inbox":
            db = getattr(self.cfg, "architect_db_path", "") or ""
            if not db:
                chat.write("[dim]inbox: not configured (set architect_db_path in config)[/dim]")
                return
            peers = getattr(self.cfg, "peer_roles", ["theseus", "majordomo"])
            pings = inbox_mod.peek_pending_pings(db, peers, exclude_role=self.instance)
            chat.write(inbox_mod.format_inbox(pings))
            return

        chat.write(f"[red]unknown command:[/red] {cmd}  (try /help)")

    # ---------------------------------------------------------------- actions

    def action_clear_log(self) -> None:
        self._chat().clear()

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
