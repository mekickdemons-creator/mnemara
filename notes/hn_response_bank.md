# HN response bank — paste-ready replies for common comment patterns

Tone rules:
- No marketing language. No "great point!", no "thanks for asking!"
- Lead with the answer. Acknowledge the critique if it lands.
- One paragraph max unless the question demands depth.
- Link to a specific file/line when relevant.

---

## "Isn't this just a system prompt?"

```
Mostly yes — the difference is the re-read. Claude Agent SDK's default
pattern pins system_prompt at session start. By turn 20 the role doc
is fighting 19 turns of tool-use chatter for attention, and "be careful"
has decayed. Re-reading on every API call keeps it fresh. It costs ~the
size of the role doc in input tokens per turn, which on a 2KB doc is
nothing against typical context spend.
```

## "Why not just use [LangChain guards / Guardrails AI / Pydantic-AI]?"

```
Different layer. Those validate outputs against schemas — useful when
you have a known shape for the answer. Sentinel is procedural: it tells
the model when to halt, not how its output should look. They're
complementary; you could wire a Guardrails check into a Mnemara MCP
tool and have both. The thing Sentinel does that schema validators
don't is catch failure modes that don't produce malformed output —
a polling loop produces perfectly valid tool calls, just too many of them.
```

## "How is this different from Claude's stop conditions / max_tokens?"

```
max_tokens / max_turns are blunt budgets — they fire after the damage
is done. Sentinel fires *before* the next tool call, when the agent
notices it's about to do something it shouldn't. The agent decides;
the runtime just gives the role doc a vehicle. That said, max_turns
is a fine outer guardrail and Mnemara respects whatever you set on
the Agent SDK side.
```

## "Why Markdown? Why not structured rules / JSON / DSL?"

```
Tried structured first, threw it out. Models follow prose better than
schemas in this kind of soft-constraint role. The structure that
matters is the *output* (the [SENTINEL HALT] block) — that's what's
parseable. The input is just clear English about when to halt. If
someone wants a schema-driven version, the role doc is one Markdown
file in the repo; fork and convert it.
```

## "Does it work with non-Claude models?"

```
Mnemara's main branch is Claude-only via the Claude Agent SDK. There's
a `gemma` branch with an Ollama backend tuned for Gemma 4 — same idea
(role doc re-read every turn) but the role doc has to be rewritten in
a more explicit style because Gemma ignores the abstract phrasing
Claude responds to. examples/roles/gemma-sentinel.md is the reworked
version. OpenAI / Mistral / etc. would need their own backend; PRs
welcome.
```

## "What if the role doc itself gets prompt-injected?"

```
The role doc is a local file you write or pull from a URL you trust
(--set-from-url is one-shot fetch + save; nothing reads from the
internet at runtime). Threat model assumes the operator controls the
file. If a third party can write to ~/.mnemara/<instance>/role.md,
they own your agent regardless — that's a filesystem permission
problem, not a runtime problem. SECURITY.md spells out the boundary.
```

## "Why not just use Claude Code directly?"

```
You can. Claude Code is great for IDE-integrated coding work. Mnemara
is for the cases where you want a separate, scriptable, file-only
state directory per task — instance dirs under ~/.mnemara/<name>/ that
you can inspect, version-control, or delete. And the Sentinel role
doc travels: you can run it on Claude Code via slash commands too,
the trick is just the every-turn re-read. Mnemara's runtime is the
thing making the role doc load-bearing in long sessions.
```

## "Repo only has 0 stars / 162 installs — looks risky"

```
Fair signal but worth context: published to PyPI 2 days ago, no marketing.
Repo's MIT, 200+ tests, SECURITY.md with a private disclosure path. The
code is small enough to read in an afternoon — under 5K LOC. Don't trust
my star count, read src/mnemara/agent.py and src/mnemara/tools.py, those
are the load-bearing files.
```

## "What's the role doc actually doing under the hood?"

```
Two functions in src/mnemara/role.py — load_role_doc reads the file,
the agent loop in src/mnemara/agent.py passes its contents to the SDK
as system_prompt on every query() call. That's it. The "trick" is just
that we don't cache the read. The interesting work is in the role doc
itself, which is plain Markdown.
```

## "How do you handle the agent ignoring the role doc?"

```
You can't make a model obey rules it decides to override; that's a model
problem, not a runtime problem. What we can do is make the rules
maximally salient (re-read every turn, pinned as system_prompt) and
make the halt action zero-friction (the agent never has to ask permission
to halt). In practice with Claude Opus, sentinel.md fires reliably on
the four triggers. Smaller models or older Claudes — your mileage will
vary; that's why gemma-sentinel.md is explicit and verbose where
sentinel.md is terse.
```

---

## Generic "this is interesting, will try" reply

```
Thanks. If you hit a failure mode the role doc misses, open a discussion
on the repo — happy to add new triggers to the example set if there are
common patterns I haven't run into.
```

## When you don't know the answer

```
Honest answer: don't know. <one sentence on what you'd need to find out>.
If you try it and have data, drop an issue and I'll dig in.
```

## Hostile / dismissive comment

Don't engage. Reply with one neutral fact and stop.

```
Fair. <one specific concrete answer>. Repo's MIT if you want to read it
yourself.
```
