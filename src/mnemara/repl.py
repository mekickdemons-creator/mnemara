"""Interactive REPL with slash commands."""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

from . import config as config_mod
from . import inbox as inbox_mod
from . import paths
from .agent import AgentSession
from .config import Config
from .logging_util import log, set_log_path
from .permissions import PermissionStore
from .store import Store
from .tools import ToolRunner, parse_proposal_file, write_memory

console = Console()


def _make_prompt(instance: str):
    history_file = paths.instance_dir(instance) / ".prompt_history"
    return PromptSession(history=FileHistory(str(history_file)))


def permission_prompt(tool: str, target: str) -> str:
    console.print(f"\n[yellow]Permission requested: {tool}[/yellow]")
    console.print(f"[dim]target:[/dim] {target}")
    console.print("[y]es / [n]o / [a]lways (allow this exact target forever) / [s]ession (allow this tool for the session)")
    while True:
        try:
            ans = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        if ans in ("y", "yes"):
            return "allow"
        if ans in ("n", "no", ""):
            return "deny"
        if ans in ("a", "always"):
            return "allow_always"
        if ans in ("s", "session"):
            return "allow_session"


def run(instance: str) -> None:
    cfg = config_mod.load(instance)
    set_log_path(paths.debug_log(instance))
    store = Store(instance)
    perms = PermissionStore(instance)
    log("repl_start", instance=instance, model=cfg.model)

    if shutil.which("codex") is None:
        console.print("[red]codex CLI not found. Install Codex and authenticate first.[/red]")
        sys.exit(1)

    runner = ToolRunner(instance, cfg, perms, permission_prompt)
    session = AgentSession(cfg, store, runner, client=None)

    psession = _make_prompt(instance)
    console.print(f"[bold green]mnemara[/bold green] instance=[cyan]{instance}[/cyan] model=[cyan]{cfg.model}[/cyan] window={cfg.max_window_turns}")
    role_path = cfg.role_doc_path or "(none)"
    console.print(f"[dim]role:[/dim] {role_path}")
    console.print("[dim]Type /help for commands, /quit to exit.[/dim]\n")

    while True:
        try:
            text = psession.prompt("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]exit[/dim]")
            break
        if not text:
            continue
        if text.startswith("/"):
            if not _handle_slash(text, instance, cfg, store):
                break
            continue
        try:
            usage = session.turn(text)
            tin, tout = store.total_tokens()
            console.print(
                f"[dim]({usage['input_tokens']} in / {usage['output_tokens']} out — total {tin}/{tout}; evicted {usage['evicted']})[/dim]"
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/yellow]")
        except Exception as e:
            log("repl_error", error=str(e))
            console.print(f"[red]error:[/red] {e}")

    try:
        session.write_session_stats()
    except Exception as e:  # pragma: no cover
        log("repl_stats_error", error=str(e))
    try:
        if session.role_proposals > 0:
            n = session.role_proposals
            p = paths.role_proposals_dir(instance)
            console.print(
                f"📋 {n} role-amendment proposal(s) written this session. "
                f"Review at {p}"
            )
    except Exception:
        pass
    store.close()
    log("repl_stop", instance=instance)


# ----------------------------------------------------------------- slash cmds


def _handle_slash(line: str, instance: str, cfg: Config, store: Store) -> bool:
    """Returns False to exit the REPL."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return False

    if cmd == "/help":
        console.print(
            "[bold]Slash commands:[/bold]\n"
            "  /role <path>     swap role doc (also persists to config)\n"
            "  /show            print the rolling window\n"
            "  /clear           wipe the rolling window (with confirm)\n"
            "  /swap <model>    switch model for this and future sessions\n"
            "  /note <text>     append to today's memory file\n"
            "  /proposals       list pending role-amendment proposals\n"
            "  /inbox           list pending pings from peer panels\n"
            "  /quit, /exit     exit\n"
        )
        return True

    if cmd == "/role":
        if not arg:
            console.print("[red]usage: /role <path>[/red]")
            return True
        cfg.role_doc_path = str(Path(arg).expanduser())
        config_mod.save(instance, cfg)
        console.print(f"[green]role doc set to[/green] {cfg.role_doc_path}")
        return True

    if cmd == "/show":
        rows = store.window()
        if not rows:
            console.print("[dim](window is empty)[/dim]")
            return True
        for row in rows:
            console.print(f"[bold]{row['role']}[/bold] [dim]{row['ts']}[/dim]")
            content = row["content"]
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            console.print(Markdown(b.get("text", "")))
                        elif b.get("type") == "tool_use":
                            console.print(f"  [magenta]tool_use[/magenta] {b.get('name')} {b.get('input')}")
                        elif b.get("type") == "tool_result":
                            console.print(f"  [magenta]tool_result[/magenta] {str(b.get('content',''))[:200]}")
                        else:
                            console.print(f"  {b}")
            else:
                console.print(str(content))
        tin, tout = store.total_tokens()
        console.print(f"[dim]totals: {tin} in / {tout} out across {len(rows)} turns[/dim]")
        return True

    if cmd == "/clear":
        try:
            ans = input("clear rolling window? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("y", "yes"):
            store.clear()
            console.print("[green]window cleared[/green]")
        return True

    if cmd == "/swap":
        if not arg:
            console.print("[red]usage: /swap <model>  e.g. /swap gpt-5.3-codex[/red]")
            return True
        try:
            normalized = config_mod.normalize_model_name(arg)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return True
        cfg.model = normalized
        config_mod.save(instance, cfg)
        console.print(f"[green]model set to[/green] {cfg.model}")
        return True

    if cmd == "/note":
        if not arg:
            console.print("[red]usage: /note <text>[/red]")
            return True
        path = write_memory(instance, arg, category="user_note")
        console.print(f"[green]appended to[/green] {path}")
        return True

    if cmd == "/proposals":
        _cmd_proposals(instance)
        return True

    if cmd == "/inbox":
        db = getattr(cfg, "architect_db_path", "") or ""
        if not db:
            console.print("[dim]inbox: not configured (set architect_db_path in config)[/dim]")
            return True
        peers = getattr(cfg, "peer_roles", ["theseus", "majordomo"])
        pings = inbox_mod.peek_pending_pings(db, peers, exclude_role=instance)
        console.print(inbox_mod.format_inbox(pings))
        return True

    console.print(f"[red]unknown command:[/red] {cmd}  (try /help)")
    return True


def _cmd_proposals(instance: str) -> None:
    prop_dir = paths.role_proposals_dir(instance)
    if not prop_dir.exists():
        console.print("No pending proposals.")
        return
    files = sorted(prop_dir.glob("*.md"), reverse=True)
    if not files:
        console.print("No pending proposals.")
        return
    console.print(f"[bold]📋 {len(files)} pending proposal{'s' if len(files) != 1 else ''}:[/bold]")
    for f in files:
        severity, preview = parse_proposal_file(f)
        console.print(f"  [[yellow]{severity}[/yellow]] {f.name} — {preview}")
