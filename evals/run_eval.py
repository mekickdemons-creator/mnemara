#!/usr/bin/env python3
"""
Gemma 4 evaluation runner.

Usage:
    python evals/run_eval.py
    python evals/run_eval.py --baseline-claude
    python evals/run_eval.py --prompts evals/prompts.yaml --out evals/results/

Outputs a timestamped JSON file in evals/results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:26b"
OLLAMA_TIMEOUT = 120.0  # seconds per prompt

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> list[dict[str, Any]]:
    """
    Load prompts from YAML and flatten into a list of prompt dicts.

    Each dict has keys:
        category    str
        id          str
        difficulty  str ("easy" | "medium" | "hard")
        prompt      str
        system      str | None   (only for role_doc_adherence category)
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    items: list[dict[str, Any]] = []
    categories = data.get("categories", {})

    for cat_name, cat_data in categories.items():
        if cat_name == "role_doc_adherence":
            system_prompt = cat_data.get("system_prompt", "").strip()
            for p in cat_data.get("prompts", []):
                items.append({
                    "category": cat_name,
                    "id": p["id"],
                    "difficulty": p.get("difficulty", "medium"),
                    "prompt": p["prompt"].strip(),
                    "system": system_prompt,
                })
        else:
            for p in cat_data:
                items.append({
                    "category": cat_name,
                    "id": p["id"],
                    "difficulty": p.get("difficulty", "medium"),
                    "prompt": p["prompt"].strip(),
                    "system": None,
                })

    return items


# ---------------------------------------------------------------------------
# Gemma via Ollama
# ---------------------------------------------------------------------------

def run_gemma(
    prompt: str,
    system: str | None = None,
    model: str = OLLAMA_MODEL,
    timeout: float = OLLAMA_TIMEOUT,
) -> dict[str, Any]:
    """
    Send a single prompt to Ollama and return a result dict:
        response        str
        latency_s       float
        prompt_tokens   int
        response_tokens int
        error           str | None
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    start = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
        elapsed = time.monotonic() - start
        body = r.json()
        response_text = body.get("message", {}).get("content", "")
        usage = body.get("prompt_eval_count", 0), body.get("eval_count", 0)
        return {
            "response": response_text,
            "latency_s": round(elapsed, 3),
            "prompt_tokens": usage[0],
            "response_tokens": usage[1],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        return {
            "response": "",
            "latency_s": round(elapsed, 3),
            "prompt_tokens": 0,
            "response_tokens": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Claude baseline (optional)
# ---------------------------------------------------------------------------

def run_claude(
    prompt: str,
    system: str | None = None,
) -> dict[str, Any]:
    """
    Run a prompt through Claude as a baseline comparison.
    Uses claude-agent-sdk under the hood; requires ANTHROPIC_API_KEY.

    Returns the same shape as run_gemma().
    """
    try:
        from claude_agent_sdk import query, ClaudeOptions  # type: ignore

        options = ClaudeOptions(system_prompt=system or "You are a helpful assistant.")
        start = time.monotonic()
        full_text = []
        for event in query(prompt=prompt, options=options):
            if hasattr(event, "content"):
                for block in (event.content or []):
                    if hasattr(block, "text"):
                        full_text.append(block.text)
        elapsed = time.monotonic() - start
        return {
            "response": "".join(full_text),
            "latency_s": round(elapsed, 3),
            "prompt_tokens": 0,   # SDK doesn't expose token counts easily
            "response_tokens": 0,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "response": "",
            "latency_s": 0.0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(
    prompts_path: Path,
    out_dir: Path,
    baseline_claude: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Run the full evaluation battery.

    Returns the path to the JSON results file.
    """
    prompts = load_prompts(prompts_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"

    results: list[dict[str, Any]] = []

    total = len(prompts)
    for i, item in enumerate(prompts, 1):
        cat = item["category"]
        pid = item["id"]
        if verbose:
            print(f"[{i}/{total}] {cat}/{pid} ({item['difficulty']}) ... ", end="", flush=True)

        gemma_result = run_gemma(item["prompt"], system=item.get("system"))
        if verbose:
            status = "ERROR" if gemma_result["error"] else f"{gemma_result['latency_s']:.1f}s"
            print(status)

        entry: dict[str, Any] = {
            "id": pid,
            "category": cat,
            "difficulty": item["difficulty"],
            "prompt": item["prompt"],
            "system": item.get("system"),
            "gemma": gemma_result,
        }

        if baseline_claude:
            if verbose:
                print(f"           (claude baseline) ... ", end="", flush=True)
            claude_result = run_claude(item["prompt"], system=item.get("system"))
            entry["claude"] = claude_result
            if verbose:
                status = "ERROR" if claude_result["error"] else f"{claude_result['latency_s']:.1f}s"
                print(status)

        results.append(entry)

    payload = {
        "run_ts": ts,
        "model": OLLAMA_MODEL,
        "baseline_claude": baseline_claude,
        "prompt_count": total,
        "results": results,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if verbose:
        print(f"\nResults written to: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 evaluation battery via Ollama."
    )
    parser.add_argument(
        "--prompts",
        default="evals/prompts.yaml",
        help="Path to prompts.yaml (default: evals/prompts.yaml)",
    )
    parser.add_argument(
        "--out",
        default="evals/results",
        help="Output directory for results JSON (default: evals/results/)",
    )
    parser.add_argument(
        "--baseline-claude",
        action="store_true",
        default=False,
        help="Also run each prompt through Claude for comparison (off by default).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-prompt progress output.",
    )
    args = parser.parse_args()

    out_path = run_eval(
        prompts_path=Path(args.prompts),
        out_dir=Path(args.out),
        baseline_claude=args.baseline_claude,
        verbose=not args.quiet,
    )
    print(f"Done. Format with: python evals/format_report.py {out_path}")


if __name__ == "__main__":
    main()
