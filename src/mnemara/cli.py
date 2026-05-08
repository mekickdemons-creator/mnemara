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

    try:
        from . import replay as replay_mod
        summary = replay_mod.last_replay_summary(instance)
        if summary:
            console.print(f"[dim]{summary}[/dim]")
    except Exception:
        pass

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
@click.option("--set", "role_path", default=None, help="Path to a local role doc.")
@click.option(
    "--set-from-url",
    "role_url",
    default=None,
    help="Download a role doc from an https:// URL into the instance dir.",
)
def role_cmd(instance: str, role_path: str | None, role_url: str | None) -> None:
    """Set role_doc_path for an instance without entering the REPL.

    Pass either --set <path> for a local file, or --set-from-url <https://...>
    to download once into the instance dir. Downloaded role docs are stored
    locally; Mnemara never re-fetches the URL at runtime.
    """
    if not role_path and not role_url:
        raise click.UsageError("provide either --set or --set-from-url")
    if role_path and role_url:
        raise click.UsageError("--set and --set-from-url are mutually exclusive")

    cfg = config_mod.load(instance)

    if role_url:
        if not role_url.startswith("https://"):
            raise click.UsageError("--set-from-url requires an https:// URL")
        import urllib.request
        dest = paths.instance_dir(instance) / "role.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(role_url, timeout=30) as resp:
                data = resp.read()
        except Exception as exc:
            raise click.ClickException(f"failed to download role doc: {exc}")
        if len(data) > 1_000_000:
            raise click.ClickException(
                f"role doc too large ({len(data)} bytes; cap is 1 MB)"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise click.ClickException("role doc is not valid UTF-8")
        dest.write_text(text)
        cfg.role_doc_path = str(dest)
        console.print(
            f"[green]downloaded[/green] {role_url}\n"
            f"[green]role doc set to[/green] {cfg.role_doc_path}"
        )
    else:
        cfg.role_doc_path = str(Path(role_path).expanduser())
        console.print(f"[green]role doc set to[/green] {cfg.role_doc_path}")

    config_mod.save(instance, cfg)


@main.command("replay")
@click.option("--instance", required=True)
@click.option("--days", default=None, type=int, help="Days of memory to scan (overrides config/policy).")
@click.option("--threshold", default=None, type=int, help="Min cluster size to count as a recurring pattern.")
@click.option("--apply", "apply_mode", is_flag=True, default=False, help="Write proposals/archives. Default is dry-run.")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Explicit dry-run (default).")
def replay_cmd(instance: str, days: int | None, threshold: int | None, apply_mode: bool, dry_run: bool) -> None:
    """Run the consolidation pass over recent memory atoms."""
    if not paths.config_path(instance).exists():
        console.print(f"[red]No instance:[/red] {instance}")
        sys.exit(1)
    if apply_mode and dry_run:
        console.print("[red]--apply and --dry-run are mutually exclusive[/red]")
        sys.exit(1)
    cfg = config_mod.load(instance)
    from . import replay as replay_mod
    out = replay_mod.run_replay(
        instance, days=days, threshold=threshold, apply=apply_mode, cfg=cfg
    )
    console.print(
        f"[cyan]replay[/cyan] mode={'apply' if apply_mode else 'dry-run'} "
        f"days={out['days']} threshold={out['threshold']}"
    )
    console.print(
        f"  atoms_loaded:        {out['atoms_loaded']}\n"
        f"  patterns_identified: {out['patterns']}\n"
        f"  wiki_proposals:      {len(out['proposals'])}\n"
        f"  duplicates_archived: {len(out['archived'])}\n"
        f"  role_amendments:     {len(out['role_amendments'])}\n"
        f"  digest:              {out['digest_path']}"
    )
    if out["proposals"]:
        console.print("\n[dim]proposals:[/dim]")
        for p in out["proposals"][:5]:
            console.print(f"  {p}")
    if not apply_mode:
        console.print("\n[yellow]dry-run — pass --apply to write files[/yellow]")


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


@main.command("migrate")
@click.option("--all", "all_instances", is_flag=True, default=False,
              help="Migrate every instance under ~/.mnemara/.")
@click.option("--instance", "instance_name", default=None,
              help="Migrate a single named instance.")
def migrate_cmd(all_instances: bool, instance_name: str | None) -> None:
    """Run schema migrations on one or all instances.

    Each Store() construction runs _migrate_schema() idempotently — this
    command forces that pass on panels that haven't been started since a
    schema-version bump (e.g. the v0.6.0 compressed_read_stub column).

    Examples:
        mnemara migrate --all
        mnemara migrate --instance majordomo
    """
    if not all_instances and not instance_name:
        console.print("[red]provide --all or --instance <name>[/red]")
        sys.exit(1)
    if all_instances and instance_name:
        console.print("[red]--all and --instance are mutually exclusive[/red]")
        sys.exit(1)

    targets: list[str]
    if instance_name:
        if not paths.config_path(instance_name).exists():
            console.print(f"[red]No instance:[/red] {instance_name}")
            sys.exit(1)
        targets = [instance_name]
    else:
        targets = paths.list_instances()
        if not targets:
            console.print("[dim](no instances found)[/dim]")
            return

    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for name in targets:
        try:
            store = Store(name)
            store.close()
            ok.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, str(exc)))

    for name in ok:
        console.print(f"[green]migrated[/green] {name}")
    for name, err in failed:
        console.print(f"[red]failed[/red] {name}: {err}")

    console.print(
        f"\n[dim]{len(ok)} migrated, {len(failed)} failed[/dim]"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
