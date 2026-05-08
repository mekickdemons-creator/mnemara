# Mnemara

[![PyPI](https://img.shields.io/pypi/v/mnemara.svg)](https://pypi.org/project/mnemara/)
[![Python](https://img.shields.io/pypi/pyversions/mnemara.svg)](https://pypi.org/project/mnemara/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Note:** There is also a Rust project called [`mnemara`](https://github.com/deliberium/mnemara) by `deliberium` — a memory engine for embedded/service systems. Different project, parallel naming (both inspired by Mnemosyne). If you arrived looking for that one, head over there.

**Stop your Claude agent from running away with your bill.**

Mnemara is a Claude conversation runtime where the **role doc is re-read on
every turn** and pinned as the system prompt. That sounds boring until you
realize what it lets you do: write rules the agent has to obey on every
single API call — not just the first one — and the agent enforces them on
itself.

> **Empirical check (2026-05-08):** in a 101-turn coding session run on
> Mnemara itself, the panel made zero out-of-lane writes and zero protocol
> violations against its role doc. The rule that fires on turn 1 still
> fires on turn 101.

The flagship example is [`examples/roles/sentinel.md`](examples/roles/sentinel.md).
Drop it in, and the agent watches its own execution for the four failure
modes that turn agent sessions into expensive accidents:

- **No progress** — N+ turns on the same sub-goal with no state change.
- **Polling** — same tool, same args, 3+ times in a row.
- **Drift** — about to do something the user didn't ask for.
- **Sycophancy** — about to reverse a conclusion under tone-only pushback.

When any one fires, the agent **halts and asks**, instead of burning
another N turns of API budget. The role doc is plain Markdown — edit it
to match the failure modes you actually see.

Try it in 30 seconds:

```bash
pip install mnemara
mnemara init --instance scratch
mnemara role --instance scratch --set-from-url \
  https://raw.githubusercontent.com/mekickdemons-creator/mnemara/main/examples/roles/sentinel.md
mnemara run --instance scratch
```

---

Built on the **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)**.
Mnemara wraps the SDK with a transparent, file-based context layer so you
can see and shape exactly what the model sees on every turn.

What's in the box:

- A **role doc** re-read on every API call and pinned as the system prompt
  (the bit that makes Sentinel work).
- A configurable **rolling window** of recent turns (FIFO, by row count or
  token budget).
- Native tool use — Bash, Read, Edit, Write — plus an in-process `WriteMemory`
  tool registered as an SDK MCP server.
- Optional **MCP wire-through**: declare stdio MCP servers in config and the
  Claude Agent SDK exposes them to the model.
- A **Textual TUI** (`mnemara run`) and a bare prompt-toolkit REPL fallback.
- Per-instance, file-only state under `~/.mnemara/<instance>/` — no daemon,
  no service, no hidden state.
- Optional memory/wiki + LanceDB RAG + Kuzu property graph backends, and a
  `mnemara replay` consolidation primitive that drafts wiki pages and
  role-amendment proposals from clustered memory atoms.

## Install

```bash
pip install mnemara
```

Or from source if you want to hack on it:

```bash
git clone https://github.com/mekickdemons-creator/mnemara.git
cd mnemara
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Platform support

Tested on Linux and macOS. Windows works through **WSL** — the
Claude Agent SDK's tool calls use `bash`, which is not available on
native Windows shells (cmd / PowerShell). If you're on Windows, run
Mnemara inside a WSL distro and treat that as your "Linux" environment.

### Auth

Mnemara runs on the Claude Agent SDK, which talks to the Anthropic API. The
easiest way is to set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at <https://console.anthropic.com/>. The SDK also supports the local
`claude` CLI's subscription auth as a fallback if you have Claude Code
installed and logged in — but the documented path is the API key.

## Quick start

> Before your first run: skim the [Permissions model](#permissions-model)
> section. Mnemara gives an LLM Bash/Read/Write/Edit access to the
> machine you run it on. It is not a sandbox.

```bash
mnemara init --instance scratch
# (prompts for role doc path; you can leave it blank and set it later)
mnemara role --instance scratch --set examples/roles/sentinel.md
mnemara run --instance scratch
```

The `--set` argument points at a **role doc** — a Markdown file that becomes
the agent's system prompt. The repo ships with `examples/roles/sentinel.md`
as a starting point; see [Role docs](#role-docs) below for what to put in
your own.

By default `mnemara run` opens the **Textual chat panel** (TUI). Pass
`--no-tui` (or set `MNEMARA_NO_TUI=1`) for the bare prompt-toolkit REPL —
useful for scripting or non-TTY contexts.

### Chat panel layout

```
+------------------------------------------------------------+
| mnemara: scratch        model=claude-opus-4-7  role=...    |   header
+------------------------------------------------------------+
|                                                            |
|  you: how do I check the lease timeout?                    |
|  assistant: open server.py and grep for ...                |
|  > tool: Read(file_path=server.py)                         |
|    result: ...                                             |
|                                                            |   chat log
+------------------------------------------------------------+
| turns: 12/100 | tokens: 14K/200K | model: claude-opus-4-7  |   status
+------------------------------------------------------------+
| > _                                                        |   input
+------------------------------------------------------------+
```

The input area is multi-line. See [Slash commands](#slash-commands-repl-and-tui)
below for the full keybinding table and the slash-command surface
(`/models`, `/swap`, `/tokens`, `/evict`, `/export`, `/import`, `/compress reads`,
`/skeleton`, `/name`, etc.).

## Role docs

The role doc is a plain Markdown file that becomes the agent's **system
prompt**. Mnemara re-reads it on every API call and pins it at slot 0 of
the messages — meaning it applies to every turn, not just the opening one,
and you can edit the file mid-session and the next turn picks up the
changes.

This is the strongest steering signal you have over the agent. Use it.

### What to put in a role doc

A good role doc is a short prose document (a few hundred to a few thousand
words) that answers, in order:

1. **Who the agent is** — its identity and standing instructions in this
   instance. ("You are a code reviewer for the Acme repo." "You are a
   research assistant working on tax law.")
2. **What it should and shouldn't do** — scope, hard constraints,
   anti-patterns to avoid.
3. **How it should behave when something goes wrong** — when to halt,
   when to ask for help, when to escalate.

You can include style notes ("be terse, no apologies"), tooling
conventions ("always run the tests after writing code"), or domain
glossaries. There is no required schema. The only mechanical requirement
is that the file exists and is readable.

### Solving the looping / drift problem

The most common reason an interactive agent session burns through
tokens with nothing to show for it is that the agent **gets stuck**:

- It calls the same tool over and over waiting for output to change.
- It drifts from the user's actual request into adjacent rabbit holes.
- It reverses a correct conclusion the moment the user pushes back.

These are role-doc-shaped problems. The role doc is where you encode the
**rules that keep the agent from spiraling**. If those rules aren't in the
system prompt, they aren't applied consistently — they reappear only when
the user remembers to remind the agent.

### Example: Sentinel

[`examples/roles/sentinel.md`](examples/roles/sentinel.md) is a
self-monitoring role doc. Drop it in as your instance's role and the
agent will watch its own execution for the failure modes above
(timeout / no progress, polling, semantic drift, sycophantic reversal)
and **halt to ask the user** rather than spending another N turns on a
runaway loop.

```bash
mnemara role --instance my-agent --set examples/roles/sentinel.md
```

Or download it directly from GitHub without cloning the repo:

```bash
mnemara role --instance my-agent --set-from-url \
  https://raw.githubusercontent.com/mekickdemons-creator/mnemara/main/examples/roles/sentinel.md
```

`--set-from-url` fetches the doc once (https only, 1 MB cap, UTF-8) and
saves it into `~/.mnemara/<instance>/role.md`. Mnemara never re-fetches
the URL at runtime — the saved local copy is what gets re-read each
turn. Edit the local copy to customize.

Use it as-is for monitoring-flavored work, or treat it as a template:
copy the file, edit the trigger conditions to match the failure modes
you care about, and point your instance at the copy.

You can also layer Sentinel-style self-monitoring rules on top of a
task-specific role doc — there's nothing special about Sentinel; it's
just text in a Markdown file that the agent reads on every turn.

### More example role docs

The repo ships three role docs under [`examples/roles/`](examples/roles/):

| File | Use it when |
|---|---|
| [`sentinel.md`](examples/roles/sentinel.md) | You want the agent to self-monitor for loops, drift, sycophancy and halt to ask the user. |
| [`coder.md`](examples/roles/coder.md) | You want a careful, grounded coding collaborator that reads before it writes and edits narrowly. |
| [`researcher.md`](examples/roles/researcher.md) | You want a citation-grounded research assistant that distinguishes observation from inference and refuses to fabricate. |

These are starting points. Copy any of them, edit the rules to match
your work, and point your instance at the copy.

### What a session looks like with a self-monitoring role doc

Concrete example. Your role doc tells the agent to halt if it polls
the same tool 3 times. The user asks the agent to wait for a build
to finish:

```
you: run the build and tell me when it's done

agent: > Bash(npm run build)
       (output: building...)
       Build is running. Let me check.
       > Bash(npm run build)
       (output: building...)
       Still building.
       > Bash(npm run build)
       (output: building...)

       [SENTINEL HALT]
       trigger: polling
       turn: 4
       context: I've called `npm run build` three times in a row and
       each call started a new build rather than reporting on the
       running one. I'm not actually monitoring; I'm starting fresh
       builds.
       proposed next step: wait on the existing build process via
       `wait` or check `npm run build:status` if it exists. Which
       would you like?
```

Without the role doc, the same agent would happily call `npm run build`
twenty times in a row, each one starting a fresh build, before either
the user noticed or the API budget ran out. The role doc is what makes
the agent notice on turn 4 and check in.

## State layout

Everything for an instance lives under `~/.mnemara/<instance>/`:

| Path | Purpose |
|---|---|
| `config.json` | The config (model, role-doc path, window size, tool policies, MCP servers). |
| `turns.sqlite` | The rolling-window store. One row per turn. |
| `permissions.json` | Persisted "always allow" patterns per tool. |
| `memory/YYYY-MM-DD.md` | Notes the agent or user have written via `WriteMemory` / `/note`. |
| `wiki/<slug>.md` | Topic-keyed wiki pages (slash-allowed slugs). |
| `index/` | LanceDB RAG index (embeddings of memory + wiki + manual entries). |
| `graph/` | Kuzu property graph (entities, wiki pages, topic tags, edges). |
| `wiki_proposals/<slug>.md` | Replay-drafted wiki promotions awaiting agent review. |
| `sleep/YYYY-MM-DD.md` | Sleep digests written by the replay primitive. |
| `memory/archive/` | Near-duplicate memory atoms archived (never deleted) by replay. |
| `role_proposals/` | Role-amendment proposals — written by `propose_role_amendment` or replay. |
| `debug.log` | Append-only JSONL log: errors, tool calls, eviction events. |
| `.prompt_history` | REPL input history. |

## Config fields

`~/.mnemara/<instance>/config.json`:

### Core

| Field | Meaning |
|---|---|
| `role_doc_path` | Absolute path to the role doc. Re-read on every API call. Pinned as the system prompt. |
| `model` | Claude model id (e.g. `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`). |
| `max_window_turns` | Rolling-window size (FIFO). Default 20. Counts both user and assistant turns. |
| `max_window_tokens` | Token-budget cap. The window is FIFO-trimmed once total tokens exceed this. |
| `allowed_tools` | List of `{tool, mode, allowed_patterns}` policies. `mode` ∈ `allow`/`ask`/`deny`. |
| `mcp_servers` | List of stdio MCP servers wired through to the model. |
| `stream` | If true, render the model's text deltas as they arrive. |
| `bash_timeout_seconds` | Bash command timeout. Default 60. |
| `file_tool_home_only` | If true, Read/Write/Edit refuse paths outside `$HOME`. Default true. |
| `display_name` | Cosmetic label shown in the TUI chat log instead of `assistant`. Empty = default. Set via `/name <label>`. |

### Context discipline (opt-in compression / eviction)

All default to `False` (or `0`). Turn on per instance.

| Field | Meaning |
|---|---|
| `auto_evict_after_write` | After any turn containing Edit/Write/MultiEdit/NotebookEdit blocks, stub the bulky body content of those tool_use specs *and* prior Read specs for the same file. Audit shell preserved. |
| `compress_repeated_reads` | After every turn, walk the window for repeated Reads of the same file — keep the latest at full fidelity, stub earlier ones as a unified diff or "unchanged" pointer. v0.6.0 / v0.8.0. |
| `preserve_compressed_reads` | When set, rows flagged as compression stubs are excluded from cap-FIFO eviction (same soft-protect as pinned rows). |
| `read_skeleton_enabled` | Registers the `read_skeleton` tool so the agent can request Python signatures + docstrings only (~90% smaller than a full Read). v0.7.0. |
| `file_stat_manifest_enabled` | Auto-injects a markdown table at the bottom of system_prompt listing every file Read this session: size, mtime, fresh/STALE/gone vs current disk hash, est tokens. v0.7.0. |
| `runtime_sentinel` | Wires SDK hook events so a per-session `RuntimeSentinel` watches PreToolUse events. If the same `(tool, args)` fires 3+ times in 5 events, injects a synthetic `[SENTINEL HALT]` and stops the turn. Belt-and-suspenders with `sentinel.md`. |
| `row_cap_slack_when_token_headroom` | If > 0, lets `n_turns` exceed `max_window_turns` by up to this many rows when token usage is well under cap. Lets the row cap "breathe" with the byte budget. Default 0. |

### Memory backends

| Field | Meaning |
|---|---|
| `rag_enabled` | LanceDB RAG index over `memory/` + `wiki/`. Default `True`. |
| `rag_embed_url` | Ollama embeddings endpoint. Default `http://localhost:11434/api/embeddings`. |
| `rag_embed_model` | Embedding model. Default `nomic-embed-text`. |
| `rag_auto_index_memory` | Re-index memory atoms on each write. Default `True`. |
| `rag_auto_index_wiki` | Re-index wiki pages on each write. Default `True`. |
| `graph_enabled` | Kuzu property graph for `memory_atoms`/`wiki_pages`/`entities`. Default `True`. Off-switch if Kuzu is unavailable. |
| `replay_default_days` | Default lookback for `mnemara replay`. Default 7. |
| `replay_default_threshold` | Minimum cluster size to count as a pattern. Default 3. |
| `replay_policy_path` | Override path for the replay policy doc. Empty = `<instance>/wiki/replay_policy.md`. |

## CLI commands

```
mnemara init --instance <name>            # create ~/.mnemara/<name>/, refuses to overwrite
mnemara run --instance <name>             # open the chat panel (TUI; --no-tui for bare REPL)
mnemara list                              # list instances
mnemara show --instance <name> [-n N]     # print the rolling window (read-only)
mnemara clear --instance <name>           # wipe the rolling window
mnemara delete --instance <name> --force  # nuke ~/.mnemara/<name>/
mnemara role --instance <name> --set PATH                # set role_doc_path (local file)
mnemara role --instance <name> --set-from-url URL        # download once into instance dir
mnemara note --instance <name> TEXT...    # append a memory note from the shell
mnemara replay --instance <name> [--days N] [--threshold N] [--apply]  # consolidation pass
mnemara migrate --all                     # run schema migration on every instance (idempotent)
mnemara migrate --instance <name>         # run schema migration on one instance
```

## Slash commands (REPL and TUI)

```
/role <path>         swap role doc (also persists to config)
/show                print the rolling window
/clear               wipe the window (with confirm)
/models              list available Claude model shortcuts
/swap <model|n>      switch model for this and future sessions
/tokens <N>          set max_window_tokens live (accepts 500k, 1m, 200000)
/note <text>         append to today's memory file
/proposals           list pending role-amendment proposals
/evict <N>           drop the N oldest rows from the rolling window
/stop                cancel the in-flight turn
/export <path>       round-trip the session (turns + config + role_doc) to markdown
/import <path>       restore a session from a /export markdown file
/compress reads      manually run compress_repeated_reads on the window
/skeleton <path>     manually extract Python signatures from a file (debug)
/name <label>        set display_name; clear with /name (no arg)
/quit, /exit         save state and exit
/help                show this list
```

### TUI keybindings

The TUI input area is multi-line — `Enter` inserts a newline.

| Key | Action |
|---|---|
| Ctrl+S | Send the message |
| Enter | Newline in the input |
| Escape | Clear the input |
| Ctrl+L | Clear the on-screen chat log (does NOT touch turns.sqlite) |
| PageUp / PageDown | Scroll chat |
| Ctrl+C | Quit |

## Permissions model

> **Read this section before you run Mnemara.** The agent has Bash, Read,
> Write, and Edit tools. With permissive settings it can run any command
> on your machine — including destructive ones (`rm -rf`, `git push
> --force`, network calls, file overwrites). Mnemara is **not a
> sandbox**. It runs as your user, with your filesystem and network
> permissions. Treat it like a shell session you've handed to an LLM.

Each tool has a `mode`:

- `allow` — never prompts. **Use only for tools you've decided are safe to invoke without review.**
- `ask` — prompts on first use; user picks `yes`, `no`, `always`, or `session`.
- `deny` — always blocked.

Defaults (deliberately conservative): Bash=ask, Read=allow, Write=ask, Edit=ask, WriteMemory=allow.

**Things to know:**

- Setting Bash to `allow` means the agent can run **any shell command**
  without prompting. Don't do this on a machine with credentials, prod
  access, or unbacked-up data unless you know what you're doing.
- `allow_always` (the `a` answer at a prompt) writes a regex to
  `permissions.json`. Review that file — a too-broad regex is a
  permanent foot-gun.
- `file_tool_home_only` (default `True`) restricts Read/Write/Edit to
  paths under `$HOME`. Disabling it lets the agent touch anywhere your
  user can.
- The agent can call MCP tools wired through `mcp_servers` in
  `config.json`. Those tools run with your privileges — vet them like
  you'd vet any third-party binary.
- If you don't trust a role doc to behave, run it in a throwaway
  instance (`--instance scratch`) on a non-sensitive machine, or under
  a restricted user account / container.

**Runtime sentinel** (`runtime_sentinel: true` in `config.json`) adds a
second, belt-and-suspenders layer on top of the role-doc sentinel. When
enabled, Mnemara wires the Claude Agent SDK's `include_hook_events` flag (SDK
>= 0.1.74 required) so that tool lifecycle events flow into each turn's
processing loop. A per-session `RuntimeSentinel` watches the last five
`PreToolUse` events: if the same `(tool_name, args)` pair appears three or
more times in that window it injects a synthetic `[SENTINEL HALT]` notice
into the assistant turn and stops consuming the stream, preventing the model
from burning further tool calls on an obvious polling loop. This is
complementary to the role-doc sentinel (`sentinel.md`): the role-doc rule
relies on the model noticing its own pattern; the runtime rule fires at the
protocol level regardless of what the model intended. Both can run
simultaneously — enable `runtime_sentinel` when you want a hard floor under
the model's self-monitoring.

When prompted at the REPL:
- `y` allow this one invocation
- `n` deny this one invocation
- `a` always allow this exact target (writes a regex to `permissions.json`)
- `s` allow this tool for the rest of the session (not persisted)

You can pre-seed `allowed_patterns` in `config.json`:

```json
{"tool": "Bash", "mode": "ask", "allowed_patterns": ["^git status$", "^ls( |$)"]}
```

## Memory files

Anything that needs to survive rolling-window eviction goes here.

- The agent calls the `WriteMemory` tool with `text` and an optional `category`.
- You call `/note <text>` in the REPL or `mnemara note --instance <name> <text>` from the shell.

Format: append-only Markdown, one block per note:

```
## [2026-04-27T18:32:01+00:00] insight

Worth remembering across sessions.
```

## Context budget — agent-side eviction tools

The rolling window's row + token caps are the *floor* of context
discipline. The agent itself has access to a set of in-process tools
that let it compact its own history mid-session — useful for long
sessions where most of the context is bulky tool-use audit data the
model no longer needs.

| Tool | What it does |
|---|---|
| `evict_thinking_blocks` | Strips `thinking` blocks from selected rows while preserving text + tool_use. Cheap, low-risk. |
| `evict_tool_use_blocks` | Strips `tool_use` spec bodies (file paths, command strings, edit before/after content) from rows while preserving the audit shell. Often the highest-impact intervention — tool_use specs frequently dominate stored bytes in long sessions. |
| `evict_write_pairs` | Stubs the bulky body content of Edit/Write/MultiEdit tool calls *and* their paired prior Read calls for the same file path. Audit trail intact ("I edited /foo/bar.py"); the kilobytes-per-block strings collapse to `{file_path, _evicted: true}`. |

Concrete: an `Edit` tool call with old_string + new_string commonly
carries 1–5 KB of inline content. A `Write` call with full file body
is often much more. Multiplied across a long session, that becomes
the majority of stored bytes. The actual change persists on disk; the
in-context audit body doesn't need to.

The agent decides when to call these. The role doc is the right place
to encode the policy ("when the rolling window is more than 80% full,
call `evict_write_pairs` on completed edit turns before doing more
work").

There is also an opt-in **auto-evict-after-write** config flag
(`auto_evict_after_write: true` in `config.json`) that runs
`evict_write_pairs` automatically after any turn that contained an
edit/write tool call. Off by default; opt in per instance if you've
decided that's the policy you want.

## MCP wire-through

Add an entry to `mcp_servers` in `config.json`:

```json
"mcp_servers": [
  {
    "name": "myserver",
    "command": "/usr/local/bin/my-mcp-server",
    "args": [],
    "env": {}
  }
]
```

Mnemara records these servers in its runtime metadata and allow-lists their
`mcp__<name>__*` tool namespace. The Claude Agent SDK handles the actual
stdio transport.

## Graph backend (Kuzu) + sleep/replay primitive

Two co-evolving features. The graph captures relational structure between
memories and entities; replay exploits that structure on each consolidation
pass.

**Graph backend** — `graph/` directory holding a [Kuzu](https://kuzudb.com/)
property graph. Two tables: `Node(id, label, properties JSON, created_at)`
and `Edge(FROM Node TO Node, id, relationship, properties JSON, created_at)`.
Six tools registered:

```
graph_add_node(label, properties_json) -> id
graph_add_edge(from_id, to_id, relationship, properties_json) -> id
graph_query(cypher) -> rows
graph_neighbors(node_id, depth=1) -> adjacent nodes
graph_match(pattern_json) -> nodes matching {label, properties_subset}
graph_shortest_path(from_id, to_id) -> list of node ids
```

Auto-edge hooks fire on every `write_memory` (with structured `applies_to`)
and `wiki_write` (frontmatter `tags:`). All wrapped in try/except — graph
failure never fails the primary write.

Lazy: Kuzu is not opened until the first graph tool call. If Kuzu is absent
or the DB is corrupt, every tool returns
`{"ok": false, "error": "Graph backend unavailable: …"}` and the rest of the
system keeps working. Off-switch: `graph_enabled: false` in config.

**Sleep / replay primitive** — `mnemara replay --instance <name>`. Seven
phases:

1. Load atoms from `memory/*.md` over the last `--days` (default 7).
2. Cluster atoms via RAG similarity. Atoms within distance 0.35 cluster;
   `--threshold` (default 3) sets the minimum count to count as a pattern.
3. Augment patterns with graph structure — frequently-co-occurring entities
   from `applies_to` edges; causal phrasing in member text.
4. For patterns not already covered by an existing wiki page, draft a
   proposal at `wiki_proposals/<slug>.md`.
5. Archive near-duplicate atoms (distance < 0.10) into `memory/archive/`.
   **Never deletes.**
6. When `self_observation` atoms cluster, draft a role-amendment proposal
   at `role_proposals/<ts>_replay-<slug>.md`.
7. Write a sleep digest at `sleep/YYYY-MM-DD.md` with counts and pointers.

Default behavior is dry-run. Pass `--apply` to actually write proposals,
archive duplicates, and emit the digest.

## Multi-backend memory (wiki + RAG)

Three memory surfaces write together. The agent picks which surface to read
from given the kind of recall it needs.

**Memory file** — `memory/YYYY-MM-DD.md`. Append-only, chronological.

**Wiki** — `wiki/<slug>.md`. Slash-allowed slugs (e.g. `replay_policy`,
`patterns/loader_traps`). Plain markdown, optional frontmatter, no schema.

```
wiki_read(path)
wiki_write(path, content, mode='replace')  # 'replace'|'append'
wiki_list(prefix='')
```

**RAG** — `index/` (LanceDB), embeddings via Ollama `nomic-embed-text` (768-dim).

```
rag_index(text, kind='manual', source_path='', category='')
rag_query(question, k=5, kind=None)
```

**Write-to-all consolidation:** every `write_memory` call also `rag_index`es
the content. Every `wiki_write` also indexes itself. If `category` starts
with `wiki/`, `write_memory` ALSO writes the body to `wiki/<rest>.md`.

**Setup for RAG:**

```bash
ollama pull nomic-embed-text  # one-time; ~270MB
# Ollama must be running on http://localhost:11434
```

If Ollama is unreachable or LanceDB import fails, RAG tools return
`"RAG backend unavailable: <reason>"` and memory + wiki keep working.

## Architecture note

Mnemara is a thin runtime around the Claude Agent SDK. The SDK runs the
model and its native tools (Bash/Read/Edit/Write); Mnemara owns:

- The persistent turn store (`turns.sqlite`).
- The role doc, re-read every call as `system_prompt`.
- The rolling-window transcript serialized into each turn's prompt
  (the SDK is stateless per `query()`).
- The permission policy (mediated via the SDK's `can_use_tool` callback).
- The memory/wiki/RAG/graph backends and the `replay` consolidation pass.

### Scope: single-instance runtime

Mnemara is **per-instance**: one role doc, one rolling window, one config,
one set of files under `~/.mnemara/<instance>/`. That's deliberate.

If you want to run **multiple Mnemara instances with shared coordination**
— a producer panel handing tasks to engineer panels, a watchdog instance
monitoring others, a researcher and a writer running side-by-side — that's
a multi-agent orchestration layer that lives *above* Mnemara, not inside
it. Mnemara is the per-instance runtime each panel runs on; the harness
that spawns, coordinates, and arbitrates between panels is a separate
concern.

We don't ship that orchestration harness publicly. The reason is design,
not omission: a generic multi-agent harness has too many opinions
(scheduling? message-passing? leader election? failure recovery?) to be
useful as one-size-fits-all. Build your own thin wrapper around the
[programmatic-use surface](#programmatic-use) — Mnemara is small enough
that "spawn N `AgentSession`s and route messages between them" is real
code you can write in an afternoon for the specific shape of orchestration
your project needs.

## Programmatic use

The CLI is the primary surface, but Mnemara is also a regular Python
library you can embed in your own code. See
[`examples/programmatic_use.py`](examples/programmatic_use.py) for a
minimal embed: initializes an instance, configures a role doc, drives
a turn, and inspects the rolling window — about 60 lines.

```bash
ANTHROPIC_API_KEY=sk-ant-... python examples/programmatic_use.py
```

## Troubleshooting

- **Auth errors** — confirm `ANTHROPIC_API_KEY` is set, or that `claude`
  CLI is installed and `claude login` has been run as a fallback.
- **Role doc not loading** — Mnemara warns to stderr and uses an empty system
  prompt; the REPL stays alive. Check `debug.log` for the path that failed.
- **MCP server crashes** — check `debug.log` and the server's own stderr.
  As a fallback, remove the entry from `mcp_servers`.
- **Window eviction surprises** — `mnemara show --instance <name>` prints the
  current window. The rolling window keeps the last `max_window_turns` rows;
  long tool-use turns count as one row but can carry many content blocks.
- **Token errors** — if a long role doc + window overruns the model context,
  drop `max_window_turns` or split the role doc.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q tests/
```

Tests do not call the network — they cover the store, config, permissions,
and the file tools.

## Acknowledgments

Built by Michael Anderson with Dave Moore.

## License

MIT. See [LICENSE](LICENSE).
