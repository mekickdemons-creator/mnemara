"""Sleep/replay primitive — consolidation pass over recent memory atoms.

Phase 1: load atoms from <instance>/memory/*.md within last --days days
Phase 2: identify recurring patterns via RAG similarity clustering
Phase 3: augment patterns with graph structure (related entities)
Phase 4: draft wiki proposals for novel patterns
Phase 5: archive near-duplicate atoms (preserve, never delete)
Phase 6: surface role-amendment drafts when self-observations cluster
Phase 7: write a sleep digest

Default behavior is dry-run: prints planned actions without writing.
--apply must be explicit.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from . import paths
from . import wiki as wiki_mod
from .config import Config
from .logging_util import log


# Distance thresholds (cosine via LanceDB).
# These match the calibration used elsewhere in the project.
DUP_DISTANCE = 0.10  # below = near-duplicate (archive candidate)
CLUSTER_DISTANCE = 0.35  # below = same-cluster (count toward recurrence)


_HEADER_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.MULTILINE)


@dataclass
class Atom:
    ts: str
    category: str
    text: str
    source_file: str
    structured_payload: Optional[dict[str, Any]] = None


@dataclass
class Pattern:
    centroid_text: str
    member_atoms: list[Atom] = field(default_factory=list)
    count: int = 0
    kind_distribution: dict[str, int] = field(default_factory=dict)
    related_entities: list[str] = field(default_factory=list)
    causal_chains: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Phase 1: load atoms
# -----------------------------------------------------------------------------


def parse_memory_file(path: Path) -> list[Atom]:
    """Parse a memory markdown file into atoms split on '## [ts] category' headers."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    atoms: list[Atom] = []
    matches = list(_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        ts = m.group(1).strip()
        cat = m.group(2).strip() or "note"
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        struct = _parse_structured_block(body) if cat == "observation" else None
        atoms.append(
            Atom(
                ts=ts,
                category=cat,
                text=body,
                source_file=str(path),
                structured_payload=struct,
            )
        )
    return atoms


def _parse_structured_block(body: str) -> Optional[dict[str, Any]]:
    """Pull `**evidence:** ...` style fields if present."""
    fields = ("evidence", "prediction", "applies_to", "confidence")
    out: dict[str, Any] = {}
    for f in fields:
        m = re.search(rf"\*\*{f}:\*\*\s*(.+?)(?:\n\n|\n\*\*|$)", body, re.DOTALL)
        if m:
            out[f] = m.group(1).strip()
    if not out:
        return None
    # observation = first paragraph before first ** field
    head = re.split(r"\n\*\*", body, maxsplit=1)[0].strip()
    out["observation"] = head
    return out


def load_recent_atoms(instance: str, days: int) -> list[Atom]:
    d = paths.memory_dir(instance)
    if not d.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    atoms: list[Atom] = []
    for f in sorted(d.glob("*.md")):
        # filename = YYYY-MM-DD.md; quick prefilter
        try:
            file_date = datetime.strptime(f.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            file_date = None
        if file_date and file_date < cutoff - timedelta(days=1):
            continue
        for a in parse_memory_file(f):
            try:
                ats = datetime.fromisoformat(a.ts.replace("Z", "+00:00"))
                if ats.tzinfo is None:
                    ats = ats.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                ats = None
            if ats is None or ats >= cutoff:
                atoms.append(a)
    return atoms


# -----------------------------------------------------------------------------
# Phase 2: cluster via RAG
# -----------------------------------------------------------------------------


def _atom_key(a: Atom) -> str:
    return f"{a.source_file}::{a.ts}"


def cluster_atoms(
    atoms: list[Atom],
    cfg: Config,
    instance: str,
    threshold: int,
) -> tuple[list[Pattern], list[tuple[Atom, Atom, float]]]:
    """Return (patterns, near_duplicate_pairs).

    Patterns are clusters of >= threshold atoms within CLUSTER_DISTANCE.
    Near-duplicate pairs are atom pairs within DUP_DISTANCE.
    """
    if not atoms:
        return [], []
    from . import rag as rag_mod
    store = rag_mod.store_for(instance, cfg)

    seen_in_cluster: set[str] = set()
    patterns: list[Pattern] = []
    dup_pairs: list[tuple[Atom, Atom, float]] = []
    by_key = {_atom_key(a): a for a in atoms}

    for atom in atoms:
        key = _atom_key(atom)
        if key in seen_in_cluster:
            continue
        # Query RAG for similar atoms
        q = atom.text or atom.category
        if not q.strip():
            continue
        res = store.query(q, k=max(threshold * 2, 8), kind="memory")
        if not res.get("ok"):
            continue
        members: list[Atom] = [atom]
        for r in res.get("results", []):
            sp = r.get("source_path") or ""
            r_text = r.get("text") or ""
            dist = float(r.get("distance", 1.0))
            # Match the result back to one of our atoms (by source_path + text prefix).
            matched: Optional[Atom] = None
            for a in atoms:
                if a is atom:
                    continue
                if a.source_file == sp and (
                    r_text.strip()[:80] in a.text or a.text.strip()[:80] in r_text
                ):
                    matched = a
                    break
            if matched is None:
                continue
            if dist < DUP_DISTANCE:
                dup_pairs.append((atom, matched, dist))
            # Category gate: when replay_cluster_within_category is set,
            # only accumulate cluster members whose category matches the
            # centroid atom.  Near-dup detection (above) stays category-blind
            # intentionally — a duplicate is a duplicate regardless of tag.
            within_cat = not getattr(cfg, "replay_cluster_within_category", False) or (
                matched.category == atom.category
            )
            if dist < CLUSTER_DISTANCE and matched not in members and within_cat:
                members.append(matched)
        if len(members) >= threshold:
            kinds: dict[str, int] = {}
            for m in members:
                kinds[m.category] = kinds.get(m.category, 0) + 1
            patterns.append(
                Pattern(
                    centroid_text=atom.text[:200],
                    member_atoms=list(members),
                    count=len(members),
                    kind_distribution=kinds,
                )
            )
            for m in members:
                seen_in_cluster.add(_atom_key(m))
    return patterns, dup_pairs


# -----------------------------------------------------------------------------
# Phase 3: graph augmentation
# -----------------------------------------------------------------------------


def augment_with_graph(
    patterns: list[Pattern], cfg: Config, instance: str
) -> None:
    if not patterns:
        return
    if not getattr(cfg, "graph_enabled", True):
        return
    try:
        from . import graph as graph_mod
        store = graph_mod.store_for(instance, cfg)
    except Exception:
        return
    for pat in patterns:
        # Pull entity refs from member atoms' applies_to
        refs: list[str] = []
        for a in pat.member_atoms:
            sp = a.structured_payload or {}
            applies = sp.get("applies_to")
            if isinstance(applies, str):
                refs.extend([r.strip() for r in re.split(r"[,\s]+", applies) if r.strip()])
            elif isinstance(applies, list):
                refs.extend([str(r).strip() for r in applies if str(r).strip()])
        # Frequency count
        freq: dict[str, int] = {}
        for r in refs:
            freq[r] = freq.get(r, 0) + 1
        pat.related_entities = sorted(freq, key=lambda k: -freq[k])[:5]
        # Causal phrasing
        chains: list[str] = []
        for a in pat.member_atoms:
            for marker in ("causes", "leads to", "results in", "because"):
                if marker in a.text.lower():
                    snippet = a.text[:160].replace("\n", " ")
                    chains.append(snippet)
                    break
        pat.causal_chains = chains[:3]


# -----------------------------------------------------------------------------
# Phase 4: draft wiki proposals
# -----------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:max_words]
    return "-".join(words) or "pattern"


def existing_wiki_covers(instance: str, slug: str) -> bool:
    pages = wiki_mod.list_pages(instance)
    for p in pages:
        if slug in p["path"]:
            return True
    return False


def draft_wiki_proposal(
    instance: str, pattern: Pattern, apply: bool
) -> Optional[Path]:
    slug = _slugify(pattern.centroid_text)
    if existing_wiki_covers(instance, slug):
        return None
    proposals = paths.wiki_proposals_dir(instance)
    f = proposals / f"{slug}.md"
    member_ids = [_atom_key(a) for a in pattern.member_atoms]
    body_lines = [
        "---",
        f"source_count: {pattern.count}",
        f"member_atom_ids: {member_ids}",
        f"drafted_at: {datetime.now(timezone.utc).isoformat()}",
        "status: proposed",
        f"related_entities: {pattern.related_entities}",
        "---",
        "",
        f"# Proposed pattern: {slug}",
        "",
        "## Member observations",
        "",
    ]
    for a in pattern.member_atoms:
        snippet = a.text.replace("\n", " ").strip()[:200]
        body_lines.append(f"- [{a.ts}] ({a.category}) {snippet}")
    body_lines.append("")
    body_lines.append("## Hypothesis")
    body_lines.append("")
    body_lines.append(
        f"These {pattern.count} atoms cluster around the same theme. "
        "Possible synthesis: review member observations and decide whether "
        "this is a stable pattern worth promoting to the wiki."
    )
    if pattern.related_entities:
        body_lines.append("")
        body_lines.append("## Related entities")
        body_lines.append("")
        for e in pattern.related_entities:
            body_lines.append(f"- {e}")
    if pattern.causal_chains:
        body_lines.append("")
        body_lines.append("## Causal phrasing observed")
        body_lines.append("")
        for c in pattern.causal_chains:
            body_lines.append(f"- {c}")
    body = "\n".join(body_lines) + "\n"
    if apply:
        proposals.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return f


# -----------------------------------------------------------------------------
# Phase 5: archive near-duplicates
# -----------------------------------------------------------------------------


def archive_duplicates(
    instance: str,
    dup_pairs: list[tuple[Atom, Atom, float]],
    apply: bool,
) -> list[tuple[Atom, Atom]]:
    """Pick canonical (longer/older) of each pair, archive the other.
    Never delete — copy original into archive/ with a header noting
    which atom subsumed it."""
    archived: list[tuple[Atom, Atom]] = []
    seen: set[str] = set()
    archive_dir = paths.memory_archive_dir(instance)
    for a, b, dist in dup_pairs:
        # canonical = the one to keep
        ka, kb = _atom_key(a), _atom_key(b)
        if len(a.text) >= len(b.text):
            canonical, redundant = a, b
        else:
            canonical, redundant = b, a
        if _atom_key(redundant) in seen or _atom_key(canonical) == _atom_key(redundant):
            continue
        seen.add(_atom_key(redundant))
        if apply:
            archive_dir.mkdir(parents=True, exist_ok=True)
            src = Path(redundant.source_file)
            ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dst = archive_dir / f"{src.name}.archived.{ts_stamp}.md"
            header = (
                f"<!-- archived by replay {ts_stamp} — subsumed by atom "
                f"{_atom_key(canonical)} (distance {dist:.3f}) -->\n"
            )
            try:
                content = src.read_text(encoding="utf-8") if src.exists() else ""
                dst.write_text(header + content, encoding="utf-8")
            except OSError as e:
                log("replay_archive_error", error=str(e))
                continue
        archived.append((canonical, redundant))
    return archived


# -----------------------------------------------------------------------------
# Phase 6: surface role amendments
# -----------------------------------------------------------------------------


def surface_role_amendments(
    instance: str, patterns: list[Pattern], threshold: int, apply: bool
) -> list[Path]:
    """When N+ self-observation atoms cluster, draft a role amendment proposal."""
    out: list[Path] = []
    for pat in patterns:
        self_obs = [
            a for a in pat.member_atoms
            if a.category == "self_observation"
            or (a.structured_payload or {}).get("role_proposal")
        ]
        if len(self_obs) < threshold:
            continue
        slug = _slugify(pat.centroid_text)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        d = paths.role_proposals_dir(instance)
        f = d / f"{ts}_replay-{slug}.md"
        body_lines = [
            "---",
            f"date: {datetime.now(timezone.utc).isoformat()}",
            "severity: minor",
            f'rationale: "Surfaced by replay — {len(self_obs)} self-observations clustered."',
            "source: replay",
            "---",
            "",
            f"## Replay-surfaced amendment draft: {slug}",
            "",
            f"The replay primitive observed {len(self_obs)} self-observations "
            "clustering on the same theme over the recent window. Suggested "
            "role amendment topic:",
            "",
            f"> {pat.centroid_text}",
            "",
            "## Member observations",
            "",
        ]
        for a in self_obs:
            body_lines.append(
                f"- [{a.ts}] {a.text.replace(chr(10), ' ').strip()[:200]}"
            )
        body = "\n".join(body_lines) + "\n"
        if apply:
            d.mkdir(parents=True, exist_ok=True)
            f.write_text(body, encoding="utf-8")
        out.append(f)
    return out


# -----------------------------------------------------------------------------
# Phase 7: sleep digest
# -----------------------------------------------------------------------------


def write_sleep_digest(
    instance: str,
    days: int,
    atoms: list[Atom],
    patterns: list[Pattern],
    proposals: list[Path],
    archived: list[tuple[Atom, Atom]],
    role_amendments: list[Path],
    apply: bool,
) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = paths.sleep_dir(instance)
    f = d / f"{today}.md"
    lines = [
        f"# Sleep digest — {today}",
        "",
        f"- days_scanned: {days}",
        f"- atoms_loaded: {len(atoms)}",
        f"- patterns_identified: {len(patterns)}",
        f"- wiki_proposals: {len(proposals)}",
        f"- duplicates_archived: {len(archived)}",
        f"- role_amendments_drafted: {len(role_amendments)}",
        f"- mode: {'apply' if apply else 'dry-run'}",
        "",
        "## Patterns",
        "",
    ]
    for i, pat in enumerate(patterns, 1):
        lines.append(
            f"{i}. count={pat.count}, kinds={pat.kind_distribution}, "
            f"related_entities={pat.related_entities}, "
            f'centroid="{pat.centroid_text[:120]}"'
        )
    lines.append("")
    lines.append("## Wiki proposals")
    lines.append("")
    for p in proposals:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Archived duplicates")
    lines.append("")
    for canonical, redundant in archived:
        lines.append(
            f"- {_atom_key(redundant)} subsumed by {_atom_key(canonical)}"
        )
    lines.append("")
    lines.append("## Role amendments")
    lines.append("")
    for p in role_amendments:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Agent reflection")
    lines.append("")
    lines.append("(Reserved — write a memory entry post-replay reflecting on this digest's quality.)")
    body = "\n".join(lines) + "\n"
    if apply:
        d.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return f


# -----------------------------------------------------------------------------
# Policy doc
# -----------------------------------------------------------------------------


_POLICY_KV = re.compile(r"^([a-z_]+)\s*[:=]\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def load_policy_overrides(instance: str, cfg: Config) -> dict[str, Any]:
    p = (cfg.replay_policy_path or "").strip()
    if p:
        path = Path(p).expanduser()
    else:
        path = paths.wiki_dir(instance) / "replay_policy.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    for m in _POLICY_KV.finditer(text):
        k = m.group(1).lower()
        v = m.group(2).strip()
        if k in ("threshold", "days"):
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def run_replay(
    instance: str,
    days: Optional[int] = None,
    threshold: Optional[int] = None,
    apply: bool = False,
    cfg: Optional[Config] = None,
) -> dict[str, Any]:
    if cfg is None:
        from . import config as config_mod
        cfg = config_mod.load(instance)
    overrides = load_policy_overrides(instance, cfg)
    if days is None:
        days = overrides.get("days", cfg.replay_default_days)
    if threshold is None:
        threshold = overrides.get("threshold", cfg.replay_default_threshold)
    days = int(days)
    threshold = int(threshold)

    atoms = load_recent_atoms(instance, days)
    patterns, dup_pairs = cluster_atoms(atoms, cfg, instance, threshold)
    augment_with_graph(patterns, cfg, instance)

    proposals: list[Path] = []
    for pat in patterns:
        p = draft_wiki_proposal(instance, pat, apply)
        if p is not None:
            proposals.append(p)
    archived = archive_duplicates(instance, dup_pairs, apply)
    role_amendments = surface_role_amendments(instance, patterns, threshold, apply)
    digest = write_sleep_digest(
        instance, days, atoms, patterns, proposals, archived, role_amendments, apply
    )
    return {
        "ok": True,
        "instance": instance,
        "days": days,
        "threshold": threshold,
        "apply": apply,
        "atoms_loaded": len(atoms),
        "patterns": len(patterns),
        "proposals": [str(p) for p in proposals],
        "archived": [
            {"canonical": _atom_key(c), "redundant": _atom_key(r)}
            for c, r in archived
        ],
        "role_amendments": [str(p) for p in role_amendments],
        "digest_path": str(digest),
        "policy_overrides": overrides,
    }


# -----------------------------------------------------------------------------
# Last-replay surfacing — used at session start.
# -----------------------------------------------------------------------------


def last_replay_summary(instance: str) -> Optional[str]:
    """If a sleep digest was written today or yesterday, return a one-line
    summary string. Otherwise None."""
    d = paths.sleep_dir(instance)
    if not d.exists():
        return None
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    for date in (today, yesterday):
        f = d / f"{date.strftime('%Y-%m-%d')}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        proposals = _grep_int(text, "wiki_proposals")
        archived = _grep_int(text, "duplicates_archived")
        amendments = _grep_int(text, "role_amendments_drafted")
        return (
            f"Last replay ({date.strftime('%Y-%m-%d')}): "
            f"{proposals} wiki proposals, {amendments} role amendments, "
            f"{archived} archived."
        )
    return None


def _grep_int(text: str, key: str) -> int:
    m = re.search(rf"-\s*{re.escape(key)}:\s*(\d+)", text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0
