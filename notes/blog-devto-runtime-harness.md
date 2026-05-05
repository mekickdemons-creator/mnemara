# The Runtime Harness: why your CLAUDE.md is half the answer

*Tags: ai, claude, productivity, agents*

A few days ago [Louai Boumediene at Activepieces wrote a great
piece](https://dev.to/louaiboumediene/the-ai-harness-why-your-ai-coding-agent-is-only-as-smart-as-the-repo-you-put-it-in-l4o)
about the **AI harness**: the set of files, rules, and feature docs
inside a codebase that turns a frontier model into a productive
collaborator. `CLAUDE.md`, `.claude/rules/`, `.agents/features/`,
skills, scoped subagents. He's right. If you're using Claude Code or
the Agent SDK seriously, you should read his post and steal liberally
from it.

I want to add the layer he didn't talk about: the **runtime harness**.

Here's the part of his post I want to zoom in on:

> If you have corrected Claude twice on the same thing, stop correcting
> it. /clear the session, rewrite your prompt with what you just learned,
> and start over.

That's a great rule. It also requires *you*, the human, to notice that
you've corrected the agent twice on the same thing. In a real session,
when you're three rabbit holes deep and tired, you don't notice. You
correct it a third time. And a fourth. And the agent, drowning in
contradictory context, keeps getting it wrong faster than you can fix
it.

The static harness can't help you here. `CLAUDE.md` was applied at
session start. The rules in `.claude/rules/` were loaded once. They
can't *react* to what's happening on turn 47.

What you need is a runtime layer that re-checks its own rules on every
turn.

---

## The three failure modes that burn tokens

Long Claude Agent SDK sessions tend to fail the same three ways:

| Failure | What it looks like | Cost |
|---|---|---|
| **Looping** | Agent re-reads the same file 5 times waiting for output to change. Re-runs the same failing test with no code change. | Linear in turns |
| **Scope drift** | "Fix this bug" becomes a 3-hour refactor of unrelated code. | Quadratic — drift compounds |
| **Sycophantic reversal** | Agent states a correct conclusion. User says "are you sure?" Agent reverses without new evidence. | Catastrophic — wrong answer shipped |

All three are *role-doc-shaped problems*. They're not fixable by adding
more entity schemas to your feature docs or another skill to
`.claude/skills/`. They're fixable by giving the agent a rule it
applies on every turn:

> If you notice you've called the same tool 3 times in a row with no
> meaningful change in result, stop. State what you've tried, and ask
> the user before continuing.

The question is: where does that rule live so it's *guaranteed* to
apply on every turn?

---

## The system prompt is the only surface that always applies

Here's a fact about the Claude Agent SDK that's easy to miss: once a
session is running, **the only piece of context guaranteed to be in
every API call is the system prompt**.

- User messages get evicted as the conversation grows.
- Tool definitions sit in the request but are passive — the model only
  reads them when deciding to call a tool.
- Memory tools (your `WriteMemory`, your RAG index) are read on demand,
  not enforced.

If you put your "stop and ask the user when you're looping" rule in a
user message at turn 1, it's gone by turn 30. If you put it in a memory
tool the agent has to query, it's only consulted when the agent thinks
to consult it — exactly the thing a looping agent doesn't do.

The system prompt is the surface that sticks.

> 💡 The reframe: **The system prompt isn't an instruction. It's a
> guardrail you re-apply on every turn.** Treat it that way.

---

## Static role docs vs. live role docs

Most agent runtimes I've seen treat the system prompt as a one-shot
instruction set you write once, at startup, and then forget about. You
hand the agent a paragraph that says "you are a senior engineer working
on Project X" and that's it.

That's a static role doc. It's better than nothing. It's also a fixed
target — once the session starts, you can't change it without
restarting.

A **live role doc** is one the runtime re-reads from disk on every API
call. Two consequences:

1. **You can edit it mid-session.** If the agent is doing something
   wrong, append a rule to the file. The next turn picks it up. No
   restart, no `/clear`, no losing your work.
2. **You can encode rules that need to apply *every turn*.**
   "Self-check for looping" only works as a guardrail if the agent
   re-encounters it every turn. A live role doc is how you guarantee
   that.

This is the design decision behind Mnemara, the runtime I built on top
of the Claude Agent SDK. Every config has a `role_doc_path`. Every
turn, the runtime reads that file fresh and pins it as the system
prompt at slot 0.

```python
# simplified
system_prompt = open(cfg.role_doc_path).read()
options = ClaudeAgentOptions(system_prompt=system_prompt, ...)
result = await query(prompt, options)
```

That's it. The "harness" is just a Markdown file the runtime promises
to re-read.

---

## Sentinel: a role doc that detects its own failure modes

The Mnemara repo ships
[`examples/roles/sentinel.md`](https://github.com/mekickdemons-creator/mnemara/blob/main/examples/roles/sentinel.md),
a self-monitoring role doc you can drop in as your instance's role.
The agent uses it to watch its own execution and halt to ask the user
instead of spiraling.

The four trigger conditions:

| Trigger | What the agent watches for | Action |
|---|---|---|
| **Timeout / no progress** | N+ turns on a sub-goal with no state change | Halt, summarize what was tried, ask for direction |
| **Polling / tight loop** | Same tool call with same args 3+ times, no result change | Halt, state the polling pattern, ask user |
| **Semantic drift** | Next action's intent doesn't match user's original request | Halt, restate both, ask to confirm or redirect |
| **Sycophantic reversal** | About to flip a conclusion based on tone, not evidence | Hold the conclusion, ask what new evidence supports the reversal |

The trick with all four is the same: the rule is in the system prompt,
so the agent re-encounters it on *every* turn, including the turn where
it's about to make the mistake. That's the difference between "I told
the agent at turn 1 not to loop" and "the agent is currently being
asked, in real time, whether it's looping."

A sample from the file (the full doc is ~5KB):

```markdown
### POLLING / TIGHT LOOP

You have called the same tool with near-identical arguments 3+ times
in quick succession without a meaningful change in the result.

Examples that count:
- Re-reading the same file 3 times in a row.
- Running the same `grep` repeatedly waiting for output to change.
- Re-running a failing test with no code change between attempts.

**Action:**
1. Stop. The repeated call is not producing new information.
2. State plainly: "I'm polling — I've called {tool} with {args} {N}
   times and the result isn't changing."
3. Either:
   - Identify what signal you're actually waiting for, and ask the
     user whether that signal will arrive in this session, or
   - Abandon the wait and try a different approach.
```

> 💡 Pro tip: Sentinel is also a template. Copy the file, edit the
> trigger conditions to match the failure modes *you* hit most, and
> point your instance at your copy.

---

## How this composes with the static harness

Louai's static harness pattern and the runtime layer don't compete.
They stack:

| Layer | Where it lives | When it applies |
|---|---|---|
| Codebase rules / conventions | `CLAUDE.md`, `.agents/features/*` | Every session, session-static |
| Feature-specific knowledge | `.agents/features/*` | When agent explores a module |
| Workflow procedures | `.claude/skills/*` | When invoked as a slash command |
| **Self-monitoring rules** | **Live role doc, system prompt** | **Every turn, dynamic** |
| Tool integrations | MCP servers | When agent needs the tool |

The static layer answers *"how does this codebase work?"* The runtime
layer answers *"how should the agent behave when something goes
wrong?"* You want both.

---

## What I am NOT claiming

- **Sentinel doesn't fix every failure mode.** It catches the four
  patterns I described. Other failure modes (model just gets the
  reasoning wrong, tool returns garbage, dependency breaks) are not
  helped by a role doc. Use the static harness for those.
- **Re-reading on every call is not free.** Each turn pays the
  tokenization cost of the role doc. Mine is ~1.5KB and the cost is
  negligible. If your role doc is 50KB, reconsider.
- **The agent has to actually follow the rules.** The role doc is
  text. Claude is generally good at following clear, applied-every-turn
  instructions, but this is not a hard constraint — it's a strong
  steering signal. Pair it with `can_use_tool` permissions for anything
  that absolutely must not happen.
- **This isn't a replacement for code review.** Halting and asking the
  user is a guardrail, not a guarantee. Humans still review PRs.

---

## How to try it

```bash
pip install mnemara

mnemara init --instance scratch
# clone the repo to get examples/roles/sentinel.md, or write your own
mnemara role --instance scratch --set examples/roles/sentinel.md
mnemara run --instance scratch
```

Set `ANTHROPIC_API_KEY` first. The runtime is MIT-licensed and runs on
the official Claude Agent SDK.

The repo:
**<https://github.com/mekickdemons-creator/mnemara>**

---

## The thesis

Louai ends his post with: *"the harness is the moat."* I agree. I'd
extend it: the harness has two layers, and most teams have only built
the first.

The **static harness** encodes what your codebase is. It loads at
session start. It teaches the agent your conventions and your
gotchas.

The **runtime harness** encodes how the agent should behave on every
turn. It loads on every API call. It catches the agent before it
spirals.

Models are commoditizing. Frontier capability is converging. The
delta between teams shipping fast with AI and teams burning tokens
with nothing to show for it isn't model choice — it's how much of your
team's hard-won knowledge has made it into a guardrail the agent
re-encounters on every single turn.

That's a thing your team builds. It compounds. It doesn't get taken
away when a new model drops.

---

*Mnemara was built by Michael Anderson with Dave Moore. If you've been
hitting the same failure modes and have ideas for trigger conditions
I missed — or if you've encoded similar guardrails in your own role
docs — I'd genuinely like to hear about it. Issues welcome on the repo.*
