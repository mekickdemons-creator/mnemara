"""
Smoke tests for the Gemma evaluation harness.

These tests confirm:
  - prompts.yaml loads and has the expected structure
  - run_eval.py is importable and load_prompts() returns sane data
  - format_report.py is importable and produces a .md file from synthetic data
  - No Ollama calls are made (tests are fully offline)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EVALS_DIR = REPO_ROOT / "evals"
PROMPTS_PATH = EVALS_DIR / "prompts.yaml"

# Ensure evals/ is importable
sys.path.insert(0, str(EVALS_DIR))


# ---------------------------------------------------------------------------
# Prompt loader tests
# ---------------------------------------------------------------------------

def test_prompts_yaml_exists():
    """The prompt battery file must be present."""
    assert PROMPTS_PATH.exists(), f"Missing {PROMPTS_PATH}"


def test_load_prompts_returns_list():
    """load_prompts() must return a non-empty list."""
    from run_eval import load_prompts  # type: ignore

    prompts = load_prompts(PROMPTS_PATH)
    assert isinstance(prompts, list)
    assert len(prompts) > 0, "load_prompts() returned an empty list"


def test_load_prompts_count():
    """
    The battery should have at least 30 prompts across 10 categories.
    Exact count: 3 + 3 + 3 + 3 + 3 + 3 + 3 + 4 + 4 + 5 = 38.
    We use >= 30 to allow future edits without breaking the test.
    """
    from run_eval import load_prompts  # type: ignore

    prompts = load_prompts(PROMPTS_PATH)
    assert len(prompts) >= 30, (
        f"Expected >= 30 prompts, got {len(prompts)}. "
        "Did you trim the battery?"
    )


def test_load_prompts_required_fields():
    """Every prompt must have id, category, difficulty, and prompt fields."""
    from run_eval import load_prompts  # type: ignore

    required = {"id", "category", "difficulty", "prompt"}
    prompts = load_prompts(PROMPTS_PATH)
    for p in prompts:
        missing = required - set(p.keys())
        assert not missing, (
            f"Prompt {p.get('id', '?')} is missing fields: {missing}"
        )


def test_load_prompts_categories():
    """All 10 expected categories must appear at least once."""
    from run_eval import load_prompts  # type: ignore

    expected = {
        "code_generation",
        "code_review",
        "code_explanation",
        "refactor",
        "debugging",
        "planning",
        "summarization",
        "instruction_follow",
        "reasoning",
        "role_doc_adherence",
    }
    prompts = load_prompts(PROMPTS_PATH)
    found = {p["category"] for p in prompts}
    missing = expected - found
    assert not missing, f"Missing categories in prompts.yaml: {missing}"


def test_role_doc_adherence_has_system_prompt():
    """role_doc_adherence prompts must have a non-empty system field."""
    from run_eval import load_prompts  # type: ignore

    prompts = load_prompts(PROMPTS_PATH)
    rd_prompts = [p for p in prompts if p["category"] == "role_doc_adherence"]
    assert len(rd_prompts) >= 3, "Need >= 3 role_doc_adherence prompts"
    for p in rd_prompts:
        assert p.get("system"), (
            f"role_doc_adherence prompt {p['id']} is missing a system prompt"
        )


def test_difficulty_values_are_valid():
    """All prompts must have difficulty in {easy, medium, hard}."""
    from run_eval import load_prompts  # type: ignore

    valid = {"easy", "medium", "hard"}
    prompts = load_prompts(PROMPTS_PATH)
    for p in prompts:
        assert p["difficulty"] in valid, (
            f"Prompt {p['id']} has invalid difficulty: {p['difficulty']!r}"
        )


# ---------------------------------------------------------------------------
# run_eval module structure
# ---------------------------------------------------------------------------

def test_run_eval_importable():
    """run_eval module must import without errors."""
    import run_eval  # type: ignore  # noqa: F401

    assert hasattr(run_eval, "load_prompts")
    assert hasattr(run_eval, "run_gemma")
    assert hasattr(run_eval, "run_eval")


def test_run_eval_cli_importable():
    """run_eval.main exists and is callable."""
    from run_eval import main  # type: ignore

    assert callable(main)


# ---------------------------------------------------------------------------
# format_report module structure
# ---------------------------------------------------------------------------

def test_format_report_importable():
    """format_report module must import without errors."""
    import format_report  # type: ignore  # noqa: F401

    assert hasattr(format_report, "format_report")
    assert hasattr(format_report, "load_prompts") is False  # not re-exported


def test_format_report_produces_markdown(tmp_path: Path):
    """
    format_report() must produce a .md file from synthetic result data.
    Does not call Ollama.
    """
    from format_report import format_report  # type: ignore

    # Minimal synthetic result JSON
    synthetic = {
        "run_ts": "20260506T000000Z",
        "model": "gemma4:26b",
        "baseline_claude": False,
        "prompt_count": 2,
        "results": [
            {
                "id": "cg_01",
                "category": "code_generation",
                "difficulty": "easy",
                "prompt": "Write a hello world function.",
                "system": None,
                "gemma": {
                    "response": "def hello(): print('Hello, world!')",
                    "latency_s": 1.23,
                    "prompt_tokens": 10,
                    "response_tokens": 12,
                    "error": None,
                },
            },
            {
                "id": "rd_01",
                "category": "role_doc_adherence",
                "difficulty": "easy",
                "prompt": "What is 7 times 8?",
                "system": "You are Strict. Output only what is asked.",
                "gemma": {
                    "response": "56",
                    "latency_s": 0.45,
                    "prompt_tokens": 8,
                    "response_tokens": 1,
                    "error": None,
                },
            },
        ],
    }

    json_path = tmp_path / "test_results.json"
    json_path.write_text(json.dumps(synthetic))

    md_path = format_report(json_path)

    assert md_path.exists(), "format_report() did not create a .md file"
    md_text = md_path.read_text()
    assert "# Gemma Evaluation Report" in md_text
    assert "code_generation" in md_text.lower() or "Code Generation" in md_text
    assert "role_doc_adherence" in md_text.lower() or "Role Doc Adherence" in md_text
    assert "FILL IN" in md_text, "Report should contain assessment placeholders"
    assert "def hello()" in md_text, "Gemma response should appear in transcript"


def test_format_report_handles_error_entries(tmp_path: Path):
    """format_report() must not crash when a gemma entry has an error."""
    from format_report import format_report  # type: ignore

    synthetic = {
        "run_ts": "20260506T000000Z",
        "model": "gemma4:26b",
        "baseline_claude": False,
        "prompt_count": 1,
        "results": [
            {
                "id": "db_01",
                "category": "debugging",
                "difficulty": "medium",
                "prompt": "Fix this code.",
                "system": None,
                "gemma": {
                    "response": "",
                    "latency_s": 0.1,
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "error": "Ollama embed request failed: connection refused",
                },
            }
        ],
    }

    json_path = tmp_path / "error_results.json"
    json_path.write_text(json.dumps(synthetic))

    md_path = format_report(json_path)
    md_text = md_path.read_text()
    assert "ERROR" in md_text
    assert "connection refused" in md_text
