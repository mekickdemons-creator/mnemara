# Mnemara

A controlled rolling-context conversation runtime for Claude. Runs an
interactive REPL with transparent context construction: a re-read-every-call
role doc as the system prompt, a configurable rolling window of recent turns,
native tool use (Bash, Read, Edit, Write, WriteMemory), and optional MCP
wire-through. Per-instance state under `~/.mnemara/<instance>/`.

Mnemara is the runtime under which Aethon orchestrators (Majordomo, Theseus,
future) operate when transparent and controlled context construction matters.
Each orchestrator panel runs its own Mnemara process.

## Install

```bash
cd ~/workspace/mnemara
python -m venv .venv
source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-...
```

## Quick start

```bash
mnemara init --instance majordomo
# (prompts for role doc path; you can leave blank and set later)
mnemara role --instance majordomo --set ~/workspace/architect/roles/majordomo.md
mnemara run --instance majordomo
```

In the REPL, type to chat. Slash commands (see below) manipulate state.

## State layout

Everything for an instance lives under `~/.mnemara/<instance>/`:

| Path | Purpose |
|---|---|
| `config.json` | The config (model, role-doc path, window size, tool policies, MCP servers). |
| `turns.sqlite` | The rolling-window store. One row per turn. |
| `permissions.json` | Persisted "always allow" patterns per tool. |
| `memory/YYYY-MM-DD.md` | Notes the agent or user have written via `WriteMemory` / `/note`. |
| `debug.log` | Append-only JSONL log: errors, tool calls, eviction events. |
| `.prompt_history` | REPL input history. |

## Config fields

`~/.mnemara/<instance>/config.json`:

| Field | Meaning |
|---|---|
| `role_doc_path` | Absolute path to the role doc. Re-read on every API call. Pinned as the system prompt. |
| `model` | Anthropic model id (e.g. `claude-opus-4-5`). |
| `max_window_turns` | Rolling-window size (FIFO). Default 20. Counts both user and assistant turns. |
| `allowed_tools` | List of `{tool, mode, allowed_patterns}` policies. `mode` ∈ `allow`/`ask`/`deny`. |
| `mcp_servers` | List of stdio MCP servers wired through to the model. |
| `stream` | If true, render the model's text deltas as they arrive. |
| `bash_timeout_seconds` | Bash command timeout. Default 60. |
| `file_tool_home_only` | If true, Read/Write/Edit refuse paths outside `$HOME`. Default true. |

## CLI commands

```
mnemara init --instance <name>            # create ~/.mnemara/<name>/, refuses to overwrite
mnemara run --instance <name>             # open the REPL
mnemara list                              # list instances
mnemara show --instance <name> [-n N]     # print the rolling window (read-only)
mnemara clear --instance <name>           # wipe the rolling window
mnemara delete --instance <name> --force  # nuke ~/.mnemara/<name>/
mnemara role --instance <name> --set PATH # set role_doc_path
mnemara note --instance <name> TEXT...    # append a memory note from the shell
```

## Slash commands (in REPL)

```
/role <path>     swap role doc (also persists to config)
/show            print the rolling window
/clear           wipe the window (with confirm)
/swap <model>    switch model for this and future sessions
/note <text>     append to today's memory file
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

The Majordomo refactor depends on the Phase-3 lease timeout patch landing first.
```

## MCP wire-through

Add an entry to `mcp_servers` in `config.json`:

```json
"mcp_servers": [
  {
    "name": "architect",
    "command": "/home/michael/workspace/architect/plugin/mcp_server.py",
    "args": [],
    "env": {}
  }
]
```

Mnemara passes this to the SDK's `mcp_servers=` parameter. If the SDK build
doesn't support it, Mnemara silently drops the parameter and continues with
native tools only — a warning is logged to `debug.log`.

## Where state lives

```
~/.mnemara/
  majordomo/
    config.json
    turns.sqlite
    permissions.json
    memory/
      2026-04-27.md
    debug.log
  theseus/
    ...
```

## Troubleshooting

- **`anthropic.AuthenticationError`** — set `ANTHROPIC_API_KEY` in your env.
- **Role doc not loading** — Mnemara warns to stderr and uses an empty system
  prompt; the REPL stays alive. Check `debug.log` for the path that failed.
- **MCP server crashes** — check `debug.log` and the server's own stderr; the
  SDK propagates the launch failure. As a fallback, remove the entry from
  `mcp_servers` and rely on native tools.
- **Window eviction surprises** — `mnemara show --instance <name>` prints the
  current window. The rolling window keeps the last `max_window_turns` rows;
  long tool-use turns count as one row but can carry many content blocks.
- **Token errors** — if a long role doc + window overruns the model context,
  drop `max_window_turns` or split the role doc.

## Running the smoke tests

```bash
pip install -e ".[dev]"
pytest -q tests/
```

Tests do not call the network — they cover store, config, permissions, and the
file tools.
