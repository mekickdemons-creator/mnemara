# Gemma 4 Evaluation Harness

A lightweight tool for running categorized test prompts through `gemma4:26b` via Ollama
and producing a human-readable Markdown report.

## Prerequisites

- Ollama running locally with `gemma4:26b` pulled:
  ```bash
  ollama pull gemma4:26b
  ollama serve   # if not already running
  ```
- Python dependencies: `httpx`, `pyyaml` (already in `pyproject.toml`)

## Quick Start

```bash
# 1. Run the evaluation battery
python evals/run_eval.py

# 2. Format the results into a Markdown report
python evals/format_report.py evals/results/<timestamp>.json

# 3. Open the report
open evals/results/<timestamp>.md
```

## With Claude Baseline

To run the same prompts through Claude for side-by-side comparison:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/run_eval.py --baseline-claude
```

Results will include a `"claude"` key alongside `"gemma"` for each prompt.

## Files

| File | Purpose |
|---|---|
| `evals/prompts.yaml` | Test prompt battery — edit to add/change prompts |
| `evals/run_eval.py` | Runner — hits Ollama, writes `evals/results/<ts>.json` |
| `evals/format_report.py` | Formatter — reads JSON, writes `evals/results/<ts>.md` |
| `evals/results/` | Output directory — JSON + Markdown results |

## Prompt Categories

| Category | Description |
|---|---|
| `code_generation` | Write Python code from scratch |
| `code_review` | Review code for bugs, style, security |
| `code_explanation` | Explain what code does |
| `refactor` | Clean up messy code |
| `debugging` | Find and fix bugs |
| `planning` | Outline technical plans |
| `summarization` | Summarize technical text |
| `instruction_follow` | Follow strict output format instructions |
| `reasoning` | Logic puzzles and word problems |
| `role_doc_adherence` | Follow a system prompt under adversarial pressure |

## Filling In the Report

After generating the Markdown, search for `[ FILL IN ]` and replace each
placeholder with one of:

- `looks_good` — correct, concise, useful
- `looks_weird` — odd phrasing, hallucinations, unnecessary padding
- `looks_broken` — wrong, refused, or garbled

Use the completed assessments to identify Gemma's strongest categories for
the contest angle.

## Running Tests

```bash
cd /path/to/mnemara
.venv/bin/pytest tests/test_evals.py -v
```
