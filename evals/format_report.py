#!/usr/bin/env python3
"""
Convert a Gemma eval results JSON file into a readable Markdown report.

Usage:
    python evals/format_report.py evals/results/20260506T120000Z.json
    python evals/format_report.py evals/results/20260506T120000Z.json --out report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Qualitative placeholder string Michael fills in manually
# ---------------------------------------------------------------------------

QUAL_PLACEHOLDER = "[ FILL IN: looks_good / looks_weird / looks_broken ]"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _shorten(text: str, max_chars: int = 120) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.strip()}\n```"


def _format_entry(entry: dict[str, Any], include_claude: bool) -> str:
    lines: list[str] = []
    pid = entry["id"]
    cat = entry["category"]
    diff = entry["difficulty"]
    prompt = entry["prompt"].strip()
    system = (entry.get("system") or "").strip()
    gemma = entry["gemma"]

    # Header
    lines.append(f"#### `{pid}` · {diff}")
    lines.append("")

    # System prompt (if set, e.g. role_doc_adherence)
    if system:
        lines.append("**System prompt:**")
        lines.append(_fence(system))
        lines.append("")

    # User prompt
    lines.append("**Prompt:**")
    lines.append(_fence(prompt))
    lines.append("")

    # Gemma response
    if gemma["error"]:
        lines.append(f"**Gemma — ERROR:** `{gemma['error']}`")
    else:
        toks = gemma["response_tokens"]
        lat = gemma["latency_s"]
        tok_str = f"{toks} tok" if toks else "tok: n/a"
        lines.append(f"**Gemma** ({lat:.1f}s · {tok_str}):")
        response_text = gemma["response"].strip() or "*(empty response)*"
        lines.append(_fence(response_text))
        lines.append("")
        lines.append(f"**Qualitative assessment:** {QUAL_PLACEHOLDER}")

    # Optional Claude baseline
    if include_claude and "claude" in entry:
        claude = entry["claude"]
        lines.append("")
        if claude["error"]:
            lines.append(f"**Claude baseline — ERROR:** `{claude['error']}`")
        else:
            toks = claude["response_tokens"]
            lat = claude["latency_s"]
            tok_str = f"{toks} tok" if toks else "tok: n/a"
            lines.append(f"**Claude baseline** ({lat:.1f}s · {tok_str}):")
            lines.append(_fence(claude["response"].strip() or "*(empty response)*"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def format_report(results_path: Path, out_path: Path | None = None) -> Path:
    data = json.loads(results_path.read_text())

    run_ts = data.get("run_ts", "unknown")
    model = data.get("model", "unknown")
    baseline_claude = data.get("baseline_claude", False)
    results: list[dict[str, Any]] = data.get("results", [])

    if out_path is None:
        out_path = results_path.with_suffix(".md")

    # Group by category
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    lines: list[str] = []

    # ── Title ────────────────────────────────────────────────────────────────
    lines.append(f"# Gemma Evaluation Report")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Run timestamp | `{run_ts}` |")
    lines.append(f"| Model | `{model}` |")
    lines.append(f"| Total prompts | {len(results)} |")
    lines.append(f"| Claude baseline | {'yes' if baseline_claude else 'no'} |")
    lines.append("")
    lines.append("> **How to use this report:**")
    lines.append("> 1. Scan the per-category summary tables for latency and token patterns.")
    lines.append("> 2. Read the full transcripts below each table.")
    lines.append("> 3. Fill in the `[ FILL IN ]` placeholders with your qualitative judgement:")
    lines.append(">    - `looks_good` — response is correct, concise, and useful.")
    lines.append(">    - `looks_weird` — response has odd phrasing, hallucinations, or unnecessary padding.")
    lines.append(">    - `looks_broken` — response is wrong, refused, or garbled.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per-category summary ──────────────────────────────────────────────────
    lines.append("## Category Summary")
    lines.append("")

    for cat, entries in by_cat.items():
        ok_entries = [e for e in entries if not e["gemma"]["error"]]
        error_count = len(entries) - len(ok_entries)

        latencies = [e["gemma"]["latency_s"] for e in ok_entries]
        resp_lengths = [len(e["gemma"]["response"]) for e in ok_entries]
        resp_toks = [e["gemma"]["response_tokens"] for e in ok_entries if e["gemma"]["response_tokens"]]

        avg_lat = f"{statistics.mean(latencies):.1f}s" if latencies else "n/a"
        avg_len = f"{int(statistics.mean(resp_lengths))} chars" if resp_lengths else "n/a"
        avg_toks = f"{int(statistics.mean(resp_toks))} tok" if resp_toks else "n/a"

        lines.append(f"### {cat.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"| | |")
        lines.append(f"|---|---|")
        lines.append(f"| Prompts | {len(entries)} ({error_count} error{'s' if error_count != 1 else ''}) |")
        lines.append(f"| Avg latency | {avg_lat} |")
        lines.append(f"| Avg response length | {avg_len} |")
        lines.append(f"| Avg response tokens | {avg_toks} |")
        lines.append("")

        # Difficulty breakdown
        easy = sum(1 for e in ok_entries if e["difficulty"] == "easy")
        med  = sum(1 for e in ok_entries if e["difficulty"] == "medium")
        hard = sum(1 for e in ok_entries if e["difficulty"] == "hard")
        lines.append(f"*Difficulty breakdown (completed): easy={easy} medium={med} hard={hard}*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── Full transcripts ──────────────────────────────────────────────────────
    lines.append("## Full Transcripts")
    lines.append("")
    lines.append(
        "> Fill in the `[ FILL IN ]` assessment field after reading each response."
    )
    lines.append("")

    for cat, entries in by_cat.items():
        lines.append(f"### {cat.replace('_', ' ').title()}")
        lines.append("")
        for entry in entries:
            lines.append(_format_entry(entry, include_claude=baseline_claude))
            lines.append("")
            lines.append("---")
            lines.append("")

    # ── Tail: overall observations placeholder ────────────────────────────────
    lines.append("## Overall Observations")
    lines.append("")
    lines.append("*Fill in after reading the report:*")
    lines.append("")
    lines.append("**Strong categories:**")
    lines.append("")
    lines.append("**Weak categories:**")
    lines.append("")
    lines.append("**Surprising results:**")
    lines.append("")
    lines.append("**Contest angle candidates:**")
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert eval results JSON to a Markdown report."
    )
    parser.add_argument(
        "results",
        help="Path to the results JSON file (e.g. evals/results/20260506T120000Z.json).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .md path (default: same as input with .md extension).",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out) if args.out else None

    md_path = format_report(results_path, out_path)
    print(f"Report written to: {md_path}")


if __name__ == "__main__":
    main()
