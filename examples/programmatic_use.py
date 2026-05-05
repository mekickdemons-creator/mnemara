"""Programmatic use of Mnemara — embed it in your own Python code.

The CLI (`mnemara run --instance ...`) is the primary user-facing
surface, but Mnemara is also a regular Python library. You can boot
an instance, drive turns, inspect the rolling window, and call
agent-side tools directly from your own code.

This script demonstrates the minimal embed:

  1. Initialize an instance (idempotent on re-runs).
  2. Configure model + role doc.
  3. Boot Store / PermissionStore / ToolRunner / AgentSession.
  4. Drive a single turn synchronously, printing the agent's reply.
  5. Print rolling-window stats.

Run:

  ANTHROPIC_API_KEY=... python examples/programmatic_use.py

Requires: `pip install mnemara` (or `pip install -e .` from a clone).
"""
from __future__ import annotations

import os
from pathlib import Path

from mnemara import config as config_mod
from mnemara.agent import AgentSession
from mnemara.config import Config
from mnemara.permissions import PermissionStore
from mnemara.store import Store
from mnemara.tools import ToolRunner


# ---- 1. Initialize an instance ---------------------------------------------

INSTANCE = "embedded-example"
ROLE_DOC_PATH = str(Path(__file__).resolve().parent / "roles" / "sentinel.md")

instance_dir = config_mod.paths.instance_dir(INSTANCE)
if not instance_dir.exists():
    config_mod.init_instance(INSTANCE, role_doc_path=ROLE_DOC_PATH)
else:
    cfg = config_mod.load(INSTANCE)
    if cfg.role_doc_path != ROLE_DOC_PATH:
        cfg.role_doc_path = ROLE_DOC_PATH
        config_mod.save(INSTANCE, cfg)

# ---- 2. Load config (model, window caps, tool policies, etc.) --------------

cfg: Config = config_mod.load(INSTANCE)
print(f"instance:  {INSTANCE}")
print(f"model:     {cfg.model}")
print(f"role doc:  {cfg.role_doc_path}")
print(f"max turns: {cfg.max_window_turns}")
print(f"max toks:  {cfg.max_window_tokens}")
print()

# ---- 3. Wire up the runtime objects ----------------------------------------

store = Store(INSTANCE)
perms = PermissionStore(INSTANCE)


def auto_deny(tool: str, target: str) -> str:
    """Headless permission prompt: refuse anything that wasn't pre-approved.

    For an interactive embed you'd surface a real prompt to the user.
    For a headless / automated embed, deny by default and pre-seed
    `allowed_patterns` in the config for the tools/targets you want
    to allow without a prompt.
    """
    print(f"[permission requested: {tool} on {target}] -> deny (headless)")
    return "deny"


runner = ToolRunner(INSTANCE, cfg, perms, auto_deny)
session = AgentSession(cfg, store, runner)

# ---- 4. Drive a turn -------------------------------------------------------

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not set — skipping the actual API call.")
    print("Set it and re-run to see the agent respond.")
else:
    user_input = "In one short sentence, what is your role?"
    print(f"you:    {user_input}")

    result = session.turn(user_input)

    # Pull the assistant's reply from the rolling-window store.
    rows = store.window()
    if rows:
        last = rows[-1]
        text_blocks = [
            b.get("text", "")
            for b in (last.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        reply = "\n".join(t for t in text_blocks if t)
        print(f"agent:  {reply}")

    print()
    print(f"tokens in:  {result.get('input_tokens', 0)}")
    print(f"tokens out: {result.get('output_tokens', 0)}")
    print(f"evicted:    {result.get('evicted', 0)} rows")

# ---- 5. Inspect the rolling window -----------------------------------------

rows = store.window()
total_in, total_out = store.total_tokens()
print()
print(f"window now holds {len(rows)} rows")
print(f"total tokens in window: {total_in} in / {total_out} out")

store.close()
