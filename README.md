# Mnemara

A controlled rolling-context conversation runtime for Claude. Built on the
**[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)**:
Mnemara wraps the SDK with a transparent, file-based context layer so you can
see and shape exactly what the model sees on every turn.

What you get:

- A **role doc** re-read on every API call and pinned as the system prompt.
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

If you want a chat loop where you control the system prompt, control the
window, and can read every byte of state on disk, that's what this is.

## Install

```bash
git clone https://github.com/mekickdemons-creator/mnemara.git
cd mnemara
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

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

```bash
mnemara init --instance scratch
# (prompts for role doc path; you can leave it blank and set it later)
mnemara role --instance scratch --set ~/path/to/role.md
mnemara run --instance scratch
```

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

Keybindings:

| Key | Action |
|---|---|
| Enter | Send the message |
| Ctrl+L | Clear the on-screen chat log (does NOT touch turns.sqlite) |
| Ctrl+C | Quit |
| `/help` | Slash-command list (same as the REPL) |

The TUI accepts `/models`, `/swap`, `/tokens`, `/quit`, and `/exit`.
`/models` lists the available Claude model shortcuts; `/swap 1` or
`/swap claude-sonnet-4-6` switches the active model.

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

## CLI commands

```
mnemara init --instance <name>            # create ~/.mnemara/<name>/, refuses to overwrite
mnemara run --instance <name>             # open the chat panel (TUI; --no-tui for bare REPL)
mnemara list                              # list instances
mnemara show --instance <name> [-n N]     # print the rolling window (read-only)
mnemara clear --instance <name>           # wipe the rolling window
mnemara delete --instance <name> --force  # nuke ~/.mnemara/<name>/
mnemara role --instance <name> --set PATH # set role_doc_path
mnemara note --instance <name> TEXT...    # append a memory note from the shell
mnemara replay --instance <name> [--days N] [--threshold N] [--apply]  # consolidation pass
```

## Slash commands (REPL and TUI)

```
/role <path>     swap role doc (also persists to config)
/show            print the rolling window
/clear           wipe the window (with confirm)
/models          list available Claude model shortcuts
/swap <model|n>  switch model for this and future sessions
/note <text>     append to today's memory file
/proposals       list pending role-amendment proposals
/quit, /exit     save state and exit
/help            show this list
```

## Permissions model

Each tool has a `mode`:

- `allow` — never prompts.
- `ask` — prompts on first use; user picks `yes`, `no`, `always`, or `session`.
- `deny` — always blocked.

Defaults: Bash=ask, Read=allow, Write=ask, Edit=ask, WriteMemory=allow.

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

## License

MIT. See [LICENSE](LICENSE).
