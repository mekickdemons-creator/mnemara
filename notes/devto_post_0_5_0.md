---
title: "Mnemara 0.5.0: the panel got real"
published: false
description: "Multi-line input. Self-healing context overflow. The version where Mnemara becomes a tool you'd actually live in."
tags: claude, llm, agents, opensource
---

I shipped [Mnemara 0.5.0](https://github.com/mekickdemons-creator/mnemara/releases/tag/v0.5.0) tonight. It's the version where the panel stops being a prototype and starts being a tool you'd actually live in. Two themes — input ergonomics and self-healing context overflow — both came out of the same observation: the runtime kept tripping over the *kinds of conversations real users have*.

## Multi-line input (yes, this was a missing feature)

Until 0.5.0, Mnemara's prompt area was a single-line input. Paste a code block and it collapsed to one space-joined line. Try to compose a multi-line prompt and you couldn't. This was always a weak spot — the kind of thing you justify with "well, it works" and quietly hope nobody pastes a stack trace.

0.5.0 makes the prompt area a real multi-line textarea. Type freely. Paste preserves newlines. Auto-grows up to 10 rows, then scrolls within itself.

**Keybinding change (the breaking part):**

| Key | Old | New |
|---|---|---|
| Enter | Submit | Insert newline |
| Ctrl+Enter | — | Submit |
| Ctrl+V | Paste (collapsed) | Paste (preserves newlines) |

This is the same convention modern chat UIs converged on (Slack, Discord, ChatGPT). For existing users: a persistent hint bar above the input now reads `Enter for newline · Ctrl+Enter to send · Ctrl+C to quit`. New users on an empty panel see a one-time tip in the chat log.

## Self-healing context overflow

The other 0.5.0 theme is what happens when a long-running conversation hits the model's hard context ceiling. Previously: cryptic `Command failed with exit code 1`, panel crashes, user runs `mnemara clear` and starts over.

Now, two layers of recovery:

**Startup** — if the rolling window is already saturated when the panel opens (say, after a long session ended near the cap), Mnemara auto-evicts to 60% before any message is sent. The status bar reports what happened:

```
⚠ rolling window was at 85% — auto-evicted 7 turn(s),
  now at 60% (~300,000 / 500,000 tokens)
```

**In-turn** — if a single turn would exceed the model's hard API ceiling mid-conversation, the runtime runs a recovery sequence:

1. Lift `max_window_tokens` to the model's hard ceiling (200K sonnet, 1M opus).
2. `evict_write_pairs` — stub Edit/Write/Read body content while preserving timestamps and IDs.
3. If still over, `evict_tool_use_blocks` — strip full tool_use blocks.
4. Retry the turn once.
5. Only if step 4 still fails does an actionable error message reach the user.

The runtime logs every step (`overflow_recovery_lift`, `overflow_recovery_evict`, etc.) so failures are debuggable. You can `tail -f ~/.mnemara/<instance>/debug.log` and watch the recovery happen.

## Why both at once

These look like unrelated features. They're not. They share a lineage: **Mnemara's job is to keep the conversation usable**. A panel that mangles paste isn't usable. A panel that crashes on context overflow isn't usable. Both failure modes blocked the same goal.

This is on-thesis with what I argued in [yesterday's post about context management being the next leap in intelligence](https://github.com/mekickdemons-creator/mnemara). Self-healing overflow is exactly that — the runtime managing context for you instead of putting the work on the user.

## Try it

```bash
pip install --upgrade mnemara
```

If you're new:

```bash
pip install mnemara
mnemara init --instance scratch
mnemara role --instance scratch --set-from-url \
  https://raw.githubusercontent.com/mekickdemons-creator/mnemara/main/examples/roles/sentinel.md
mnemara run --instance scratch
```

[Repo](https://github.com/mekickdemons-creator/mnemara). MIT. Issues and discussions open.
