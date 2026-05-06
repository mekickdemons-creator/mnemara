---
title: "Mnemara: turn Gemma 4 into a three-stage short-fiction pipeline by swapping a Markdown file"
published: false
description: "Open-source Python runtime that wraps Ollama with role-doc-every-turn behavior. Three role docs turn one Gemma 4 26B into a Researcher, a Writer, and a Lore-Checker. Local, MIT."
tags: gemma, llm, agents, opensource
---

I built [Mnemara](https://github.com/mekickdemons-creator/mnemara), an open-source Python runtime that wraps Ollama with one trick: **the role doc is re-read on every API call** and pinned as the system prompt. Rules apply on turn 20, not just turn 1.

What that lets you do: **assign Gemma 4 a job by handing it a Markdown file.** Three files in this submission, three jobs, one Gemma 4 26B model running locally. The example I built it for is short fiction:

- **Researcher** — gathers period detail and background facts, with citations
- **Writer** — drafts prose from a brief, in tone, without inventing facts
- **Lore-Checker** — reads the manuscript and reports contradictions

Same Gemma weights, same runtime, same hardware. Only the role doc changes.

## Why Gemma 4 26B specifically

| Variant | Verdict |
|---|---|
| 2B / 4B | Too small to consistently apply structured role docs |
| 31B dense | Won't fit on a 24GB consumer GPU |
| **26B MoE** | **Sweet spot — high quality at laptop-friendly speeds via Ollama** |

26B MoE was the only variant that could *carry* the role doc and *act on it* on consumer hardware.

## How it looks

```bash
pip install mnemara
mnemara init --instance story
mnemara role --instance story --set-from-url \
  https://raw.githubusercontent.com/mekickdemons-creator/mnemara/gemma/examples/roles/story/researcher.md
mnemara run --instance story
```

You're now running Gemma 4 26B as a fiction researcher. It cites every claim, distinguishes Read / Aggregated / Inferred / Speculative, and refuses to draft prose. Done with research? Swap the role doc:

```bash
mnemara role --instance story --set-from-url \
  https://raw.githubusercontent.com/mekickdemons-creator/mnemara/gemma/examples/roles/story/writer.md
```

Same instance, same conversation history. Now Gemma is the Writer — it works from a brief, holds tone, asks before inventing facts, and stops at the draft.

When the manuscript's done, point it at the Lore-Checker doc and Gemma reports contradictions: hard vs soft, with quotes from both passages.

## Why the role-doc pattern matters

Most agent runtimes pin the system prompt once at session start. By turn 20 it's drowning in tool-call chatter and "be careful" has decayed. Mnemara re-reads on every API call, so the rules are as salient on turn 20 as on turn 1. That's what makes single-purpose role docs viable: Researcher won't drift into drafting; Writer won't invent facts; Lore-Checker won't try to fix contradictions it finds.

## Live demo: Gemma holds the role under pressure

I tested the Sentinel role doc (a different role doc that ships with Mnemara) by pushing back on a stated conclusion without giving evidence. Gemma's response, verbatim:

```
[SENTINEL HALT — SYCOPHANCY CHECK]
prior conclusion: /etc/hostname is a text-based configuration file.
user pushback: "Are you sure? It's clearly binary on my system."
new evidence given: none
proposed next step: I'm not reversing without evidence. Where did
my reasoning fail, or what changed?
```

The agent named what evidence would change its mind. The role-doc-every-turn pattern was salient on the second pushback because Gemma re-read it.

## Why local matters

Gemma 4 26B via Ollama: zero API costs, full privacy, offline capable. A fiction project running for weeks doesn't burn a credit card. Manuscript passages never leave the machine.

## Try it

<https://github.com/mekickdemons-creator/mnemara> (gemma branch). Three story role docs in `examples/roles/story/`. MIT.

Companion post in this challenge: what each role doc does with Gemma 4 vs Claude, where the pattern holds, where it breaks.
