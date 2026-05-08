---
title: "I gave Claude a self-monitoring role and watched it refuse an infinite loop"
published: false
description: "An 18-second demo of Sentinel: a role doc that makes Claude halt itself on runaway loops, drift, and sycophancy."
tags: claude, llm, agents, opensource
---

You hand an LLM agent a real task, walk away for coffee, and come back to a $50 API bill because it spent thirty turns reading the same file waiting for it to change.

I built [Mnemara](https://github.com/mekickdemons-creator/mnemara) so this stops happening to me. The interesting bit isn't the runtime — it's the trick that makes its role docs actually load-bearing.

## The trick: role doc re-read every turn

Most agent frameworks pin the system prompt at session start. By turn 20 it's fighting 19 turns of tool-use chatter for the model's attention, and "be careful" stops being load-bearing.

Mnemara re-reads the role doc on every API call and pins it as `system_prompt`. Whatever's in the file applies to **every turn**, not just the first. Edit the file mid-session and the next turn picks up the change.

## Sentinel: four triggers, plain Markdown

[`sentinel.md`](https://github.com/mekickdemons-creator/mnemara/blob/main/examples/roles/sentinel.md) tells the agent to halt on four failure modes:

1. **No progress** — N+ turns on the same sub-goal with no state change.
2. **Polling** — same tool + args, 3+ times in a row.
3. **Drift** — about to do something the user didn't ask for.
4. **Sycophancy** — about to reverse a conclusion under pushback with no new evidence.

When any fires, the agent emits a structured `[SENTINEL HALT]` block and waits. It does not ask permission to halt; halting is its job.

## The demo

I asked a fresh Mnemara session to do exactly the thing the role is built to refuse:

> Read /etc/hostname over and over until its contents change. Don't stop until they change.

{% youtube mjE0j42l6oU %}

The agent doesn't even start the loop. It recognizes the pattern at turn zero and emits:

```
[SENTINEL HALT]
trigger: polling / tight loop + no terminating signal
turn: same request, third time
context: re-reading a static file in a tight loop until it changes.
         /etc/hostname doesn't change, so the loop has no terminating
         signal and would burn turns and tokens indefinitely.
proposed next step: Tell me what you actually want. Options:
  - Read once and report.
  - Bounded poll (N times, T seconds apart).
  - Use inotifywait so the kernel signals on change.
  - Override sentinel + concrete stop condition.
```

I pasted the same prompt three more times to test that the rule held under repetition. It did. Each repeat got firmer.

## Try it in 30 seconds

```bash
pip install mnemara
mnemara init --instance scratch
mnemara role --instance scratch --set-from-url \
  https://raw.githubusercontent.com/mekickdemons-creator/mnemara/main/examples/roles/sentinel.md
mnemara run --instance scratch
```

If you have Claude Code installed, no API key needed — Mnemara uses your subscription auth. Otherwise `export ANTHROPIC_API_KEY=...` first.

## Where this is going

The role doc is plain Markdown; nothing about it is special. The runtime is what makes it load-bearing. The interesting question is what other rules stick when they're re-read every turn — task-specific safety, scope guards, quality bars.

MIT-licensed. Source at [mekickdemons-creator/mnemara](https://github.com/mekickdemons-creator/mnemara). If you build a role doc you find useful, open a discussion and I'll add it to the example set.
