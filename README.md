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
```

### Auth

Mnemara uses the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk-overview),
which talks to your local `claude` CLI. It inherits your Claude Code
subscription auth — no API key required. If you have not already authed:

```bash
claude auth login
```

`ANTHROPIC_API_KEY` is **not** required. If it is set in your environment
the underlying `claude` CLI may use it (billed against API credits); leave
it unset to bill against your subscription.

## Quick start

```bash
mnemara init --instance majordomo
# (prompts for role doc path; you can leave blank and set later)
mnemara role --instance majordomo --set ~/workspace/architect/roles/majordomo.md
mnemara run --instance majordomo
```

By default `mnemara run` opens the **Textual chat panel** (TUI). Pass
`--no-tui` (or set `MNEMARA_NO_TUI=1`) to force the bare prompt-toolkit
REPL — useful for scripting or non-TTY contexts.

### Chat panel layout (v0.1.2)

```
+------------------------------------------------------------+
| mnemara: majordomo            model=opus-4-7  role=...     |   header
+------------------------------------------------------------+
|                                                            |
|  you: how do I check the lease timeout?                    |
|  assistant: open architect/orchestrator.py and grep ...    |
|  > tool: Read(file_path=architect/orchestrator.py)         |
|    result: ...                                             |
|                                                            |   chat log
+------------------------------------------------------------+
| turns: 12/100 | tokens: 14K/200K | model: opus-4-7 | ...   |   status
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

Slash commands (`/role`, `/show`, `/clear`, `/swap`, `/note`, `/quit`,
`/help`) are accepted in either UI. The `/note` command opens a small
modal in the TUI when invoked without text.

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
mnemara run --instance <name>             # open the chat panel (TUI; --no-tui for bare REPL)
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

Mnemara passes this to the Claude Agent SDK's `mcp_servers` option. The SDK
launches the stdio process and exposes its tools to the model under the
`mcp__<name>__*` namespace; Mnemara automatically allow-lists those.

### v0.1.2 — Textual chat panel

`mnemara run` now defaults to a Textual TUI: header (instance/model/role),
scrollable chat log with user/assistant/tool-use rendering, status bar
(turns / tokens / model / evicted), and a single-line input box. Streaming
tokens render live via new `on_token` / `on_tool_use` / `on_tool_result`
callbacks on `AgentSession.turn_async()`. The bare prompt-toolkit REPL
remains as the `--no-tui` / `MNEMARA_NO_TUI=1` fallback.

### Architecture note (v0.1.1)

Mnemara now uses the **Claude Agent SDK** (`claude-agent-sdk`) rather than
the raw Anthropic API SDK. The SDK is a higher-level wrapper around the
`claude` CLI: it does not accept a fabricated `messages=[...]` list with
synthetic assistant turns. Mnemara therefore serialises the rolling-window
transcript into a prefix prepended to each turn's user prompt, with the
role doc still pinned as `system_prompt`. Bash/Read/Edit/Write are
delegated to Claude Code's built-in tools (the SDK runs them in `cwd`);
WriteMemory is registered as an in-process SDK MCP tool. Permissions still
flow through Mnemara's `permissions.py` policy via the SDK's
`can_use_tool` callback.

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

- **Auth errors / "claude CLI not found"** — install Claude Code and run
  `claude auth login`. Mnemara delegates auth entirely to the `claude` CLI.
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
