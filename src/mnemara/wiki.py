"""Wiki layer — slash-allowed slug -> markdown page under <instance>/wiki/.

Plain-markdown pages with optional frontmatter. No schema beyond that.
Paths are normalised: leading/trailing slashes stripped, '..' rejected.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths


def _resolve(instance: str, page_path: str) -> Path:
    base = paths.wiki_dir(instance).resolve()
    rel = (page_path or "").strip().strip("/")
    if not rel:
        raise ValueError("page path is empty")
    if any(part in ("..", "") for part in rel.split("/")):
        raise ValueError(f"invalid wiki path: {page_path!r}")
    target = (base / f"{rel}.md").resolve()
    # Ensure target stays within wiki_dir.
    if base != target and base not in target.parents:
        raise ValueError(f"wiki path escapes wiki dir: {page_path!r}")
    return target


def read_page(instance: str, page_path: str) -> str | None:
    """Return page content, or None if no such page."""
    try:
        f = _resolve(instance, page_path)
    except ValueError:
        return None
    if not f.exists():
        return None
    return f.read_text(encoding="utf-8")


def write_page(
    instance: str,
    page_path: str,
    content: str,
    mode: str = "replace",
) -> Path:
    """Write or append to a wiki page. Creates parent dirs as needed."""
    f = _resolve(instance, page_path)
    f.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append" and f.exists():
        with f.open("a", encoding="utf-8") as fh:
            if not content.startswith("\n"):
                fh.write("\n")
            fh.write(content)
            if not content.endswith("\n"):
                fh.write("\n")
    else:
        f.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    return f


def list_pages(instance: str, prefix: str = "") -> list[dict[str, Any]]:
    """List wiki pages under prefix (slash-allowed). Each entry has
    {path, size_bytes, last_modified} (last_modified ISO-8601 UTC)."""
    base = paths.wiki_dir(instance)
    if not base.exists():
        return []
    base_resolved = base.resolve()
    p = (prefix or "").strip().strip("/")
    out: list[dict[str, Any]] = []
    for f in base.rglob("*.md"):
        rel = f.resolve().relative_to(base_resolved)
        rel_path = str(rel.with_suffix(""))
        if p and not rel_path.startswith(p):
            continue
        st = f.stat()
        out.append(
            {
                "path": rel_path,
                "size_bytes": st.st_size,
                "last_modified": datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    out.sort(key=lambda d: d["path"])
    return out
