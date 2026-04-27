"""Click-based CLI entry points."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import click
from rich.console import Console

from . import config as config_mod
from . import paths
from . import repl as repl_mod
from .store import Store
from .tools import write_memory

console = Console()


@click.group()
def main() -> None:
    """mnemara — controlled rolling-context conversation runtime."""


@main.command("init")
@click.option("--instance", required=True, help="Instance name.")
@click.option("--role", "role_doc", default=None, help="Path to role doc (optional).")
def init_cmd(instance: str, role_doc: str | None) -> None:
    """Create ~/.mnemara/<instance>/ with a default config."""
    if paths.instance_dir(instance).exists():
        console.print(f"[red]Instance already exists:[/red] {paths.instance_dir(instance)}")
        sys.exit(1)
    if not role_doc:
        try:
            role_doc = click.prompt("Role doc path (blank for none)", default="", show_default=False).strip()
        except click.Abort:
            role_doc = ""
    role_doc = role_doc or ""
    if role_doc:
        role_doc = str(Path(role_doc).expanduser())
    d = config_mod.init_instance(instance, role_doc_path=role_doc)
    console.print(f"[green]created[/green] {d}")
    console.print(f"  config:   {paths.config_path(instance)}")
    console.print(f"  memory:   {paths.memory_dir(instance)}")
    console.print(f"  role:     {role_doc or '(none)'}")
    console.print(f"\nNext: [bold]mnemara run --instance {instance}[/bold]")


@main.command("run")
@click.option("--instance", required=True)
@click.option("--no-tui", is_flag=True, help="Force the bare prompt-toolkit REPL.")
def run_cmd(instance: str, no_tui: bool) -> None:
    """Open the interactive chat panel for an instance.

    Launches the Textual TUI by default; falls back to the bare REPL when
    --no-tui is set, MNEMARA_NO_TUI=1, textual is not installed, or stdout
    is not a TTY.
    """
    if not paths.config_path(instance).exists():
        console.print(f"[red]No instance:[/red] {instance}")
        sys.exit(1)

    force_repl = no_tui or os.environ.get("MNEMARA_NO_TUI") == "1"
    if not force_repl:
        try:
            from . import tui as tui_mod
        except ImportError as exc:
            console.print(f"[yellow]textual unavailable ({exc}); using bare REPL.[/yellow]")
            tui_mod = None  # type: ignore[assignment]
        if tui_mod is not None:
            launched = tui_mod.run(instance)
            if launched:
                return
            console.print("[yellow]TUI not launched (no TTY?); falling back to REPL.[/yellow]")

    repl_mod.run(instance)


@main.command("list")
def list_cmd() -> None:
    """List all instances."""
    insts = paths.list_instances()
    if not insts:
        console.print("[dim](no instances)[/dim]")
        return
    for name in insts:
        cfg = config_mod.load(name)
        console.print(f"  [cyan]{name}[/cyan]  model={cfg.model}  role={cfg.role_doc_path or '(none)'}")


@main.command("show")
@click.option("--instance", required=True)
@click.option("-n", "limit", default=None, type=int)
def show_cmd(instance: str, limit: int | None) -> None:
    """Print the rolling window for an instance (read-only)."""
    if not paths.config_path(instance).exists():
        console.print(f"[red]No instance:[/red] {instance}")
        sys.exit(1)
    store = Store(instance)
    rows = store.window(limit=limit)
    for row in rows:
        console.print(f"[bold]{row['role']}[/bold] [dim]{row['ts']}[/dim]")
        content = row["content"]
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        console.print(b.get("text", ""))
                    else:
                        console.print(f"  [{b.get('type')}] {b}")
        else:
            console.print(str(content))
        console.print("")
    tin, tout = store.total_tokens()
    console.print(f"[dim]totals: {tin} in / {tout} out  ({len(rows)} turns)[/dim]")
    store.close()


@main.command("clear")
@click.option("--instance", required=True)
def clear_cmd(instance: str) -> None:
    """Wipe the rolling window (preserves config and memory files)."""
    if not paths.config_path(instance).exists():
        console.print(f"[red]No instance:[/red] {instance}")
        sys.exit(1)
    store = Store(instance)
    store.clear()
    store.close()
    console.print(f"[green]cleared[/green] {instance}")


@main.command("delete")
@click.option("--instance", required=True)
@click.option("--force", is_flag=True)
def delete_cmd(instance: str, force: bool) -> None:
    """Remove an instance directory entirely. Requires --force."""
    d = paths.instance_dir(instance)
    if not d.exists():
        console.print(f"[red]No such instance:[/red] {d}")
        sys.exit(1)
    if not force:
        console.print("[red]--force required to delete[/red]")
        sys.exit(1)
    shutil.rmtree(d)
    console.print(f"[green]deleted[/green] {d}")


@main.command("role")
@click.option("--instance", required=True)
@click.option("--set", "role_path", required=True)
def role_cmd(instance: str, role_path: str) -> None:
    """Set role_doc_path for an instance without entering the REPL."""
    cfg = config_mod.load(instance)
    cfg.role_doc_path = str(Path(role_path).expanduser())
    config_mod.save(instance, cfg)
    console.print(f"[green]role doc set to[/green] {cfg.role_doc_path}")


@main.command("note")
@click.option("--instance", required=True)
@click.argument("text", nargs=-1, required=True)
def note_cmd(instance: str, text: tuple[str, ...]) -> None:
    """Append a timestamped note to today's memory file."""
    if not paths.config_path(instance).exists():
        console.print(f"[red]No instance:[/red] {instance}")
        sys.exit(1)
    body = " ".join(text)
    p = write_memory(instance, body, category="user_note")
    console.print(f"[green]appended to[/green] {p}")


if __name__ == "__main__":
    main()
