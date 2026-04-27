"""Agent loop using the Claude Agent SDK.

Runs each user turn as a `query()` call against the Claude Agent SDK, which
talks to the local `claude` CLI under the user's existing subscription auth
(no ANTHROPIC_API_KEY needed). Mnemara still owns:

  * the rolling-window store (turns.sqlite),
  * the role doc (re-read every call as system_prompt),
  * the permission policy (mediated via the SDK's can_use_tool callback),
  * the WriteMemory tool (registered as an in-process SDK MCP server).

The SDK runs Bash/Read/Edit/Write itself (Claude Code's built-in tools).
Permissions still pass through Mnemara's tools.py policy via the
`can_use_tool` callback.

The SDK is unidirectional and stateless per `query()` — it does not accept a
prefab assistant-turn list. We work with that by serialising the rolling
window into a transcript prefix prepended to the current user input. The
role doc remains the system_prompt. Mnemara's per-turn rows in turns.sqlite
remain the source of truth for window/eviction.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console

from . import role as role_mod
from . import tools as tools_mod
from .config import Config
from .logging_util import log, warn
from .store import Store
from .tools import ToolRunner

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        PermissionResultAllow,
        PermissionResultDeny,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        create_sdk_mcp_server,
        query,
        tool,
    )
    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False

console = Console()

# Built-in Claude Code tool names we expose by default.
_BUILTIN_TOOLS = ("Bash", "Read", "Write", "Edit")
# MCP-prefixed name the SDK assigns to our in-process WriteMemory tool.
_WRITE_MEMORY_TOOL = "mcp__mnemara_memory__write_memory"


class AgentSession:
    def __init__(self, cfg: Config, store: Store, runner: ToolRunner, client: Any = None):
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                "claude_agent_sdk is not installed. Run: pip install claude-agent-sdk"
            )
        self.cfg = cfg
        self.store = store
        self.runner = runner
        # `client` retained in the signature for backward compatibility with
        # repl.py callers; the SDK manages its own transport, so we ignore it.
        self.client = client

    # ------------------------------------------------------------------ public

    def turn(self, user_text: str) -> dict[str, Any]:
        """Run one user->assistant turn. Returns usage + eviction info."""
        # Persist the user turn before contacting the model.
        user_blocks = [{"type": "text", "text": user_text}]
        self.store.append_turn("user", user_blocks)

        system_prompt = role_mod.load_role_doc(self.cfg.role_doc_path) or (
            "You are a helpful assistant."
        )

        # Build prompt: window transcript prefix + current user input.
        prompt = _build_prompt(self.store, user_text)

        options = self._build_options(system_prompt)

        result = asyncio.run(_run_turn(prompt, options, self.cfg.stream))

        assistant_blocks = result["assistant_blocks"]
        total_in = result["tokens_in"]
        total_out = result["tokens_out"]

        # Persist final assistant turn (full block list) and evict.
        self.store.append_turn(
            "assistant",
            assistant_blocks or [{"type": "text", "text": ""}],
            tokens_in=total_in,
            tokens_out=total_out,
        )
        evicted = self.store.evict(self.cfg.max_window_turns, self.cfg.max_window_tokens)
        if evicted:
            log("eviction", deleted=evicted)
        return {"input_tokens": total_in, "output_tokens": total_out, "evicted": evicted}

    # ------------------------------------------------------------------ internal

    def _build_options(self, system_prompt: str) -> "ClaudeAgentOptions":
        # In-process WriteMemory tool, exposed as an SDK MCP server.
        @tool(
            "write_memory",
            "Append a timestamped note to the instance's memory file. "
            "Use to record insights that should survive rolling-window eviction.",
            {"text": str, "category": str},
        )
        async def _write_memory_tool(args: dict[str, Any]) -> dict[str, Any]:
            tools_mod.write_memory(
                self.runner.instance,
                args["text"],
                args.get("category", "note") or "note",
            )
            return {"content": [{"type": "text", "text": "Memory note appended."}]}

        memory_server = create_sdk_mcp_server(
            name="mnemara_memory", tools=[_write_memory_tool]
        )

        mcp_servers: dict[str, Any] = {"mnemara_memory": memory_server}
        for s in self.cfg.mcp_servers:
            mcp_servers[s.name] = {
                "type": "stdio",
                "command": s.command,
                "args": list(s.args),
                "env": dict(s.env),
            }

        allowed_tools = list(_BUILTIN_TOOLS) + [_WRITE_MEMORY_TOOL]
        for s in self.cfg.mcp_servers:
            # Permit any tool exposed by configured MCP servers.
            allowed_tools.append(f"mcp__{s.name}__*")

        runner = self.runner
        cfg = self.cfg

        async def _can_use_tool(tool_name: str, tool_input: dict[str, Any], _ctx):
            # Map (tool, target) onto Mnemara's permission policy.
            mapped, target = _map_tool_target(tool_name, tool_input)
            if mapped is None:
                # Tool we don't policy (e.g. an MCP tool from an external server).
                return PermissionResultAllow(behavior="allow", updated_input=tool_input)
            ok, err = runner._check_perm(mapped, target)
            if ok:
                return PermissionResultAllow(behavior="allow", updated_input=tool_input)
            return PermissionResultDeny(
                behavior="deny",
                message=err or "Permission denied by Mnemara policy.",
                interrupt=False,
            )

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=cfg.model,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            can_use_tool=_can_use_tool,
            permission_mode="bypassPermissions",
            include_partial_messages=cfg.stream,
            setting_sources=[],
        )


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------


def _build_prompt(store: Store, current_user_text: str) -> str:
    """Flatten the rolling window into a transcript prefix + current input.

    The Claude Agent SDK does not accept a prefab message list with synthetic
    assistant turns, so we serialise prior turns into a single user-side text
    payload framed as the running transcript. The current user input follows
    the prefix under a clear separator. The model treats the prefix as
    context (consistent with the `system_prompt` instructing it to continue
    the conversation it sees).
    """
    msgs = store.messages_for_api()
    # The last row is the just-appended current user turn — drop it; we frame
    # the current input as the live message below.
    history = msgs[:-1] if msgs else []
    if not history:
        return current_user_text

    lines: list[str] = ["[CONVERSATION HISTORY — prior turns in this session]"]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", [])
        text = _flatten_blocks(content)
        if not text.strip():
            continue
        lines.append(f"\n--- {role.upper()} ---\n{text}")
    lines.append("\n[END HISTORY]\n\n[CURRENT USER MESSAGE]\n" + current_user_text)
    return "\n".join(lines)


def _flatten_blocks(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return str(blocks)
    parts = []
    for b in blocks:
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        btype = b.get("type")
        if btype == "text":
            parts.append(b.get("text", ""))
        elif btype == "tool_use":
            parts.append(f"[tool_use {b.get('name')} {json.dumps(b.get('input') or {})}]")
        elif btype == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = _flatten_blocks(c)
            parts.append(f"[tool_result {b.get('tool_use_id', '')}]\n{c}")
        else:
            parts.append(json.dumps(b))
    return "\n".join(p for p in parts if p)


# -----------------------------------------------------------------------------
# SDK driver
# -----------------------------------------------------------------------------


async def _run_turn(prompt: str, options: "ClaudeAgentOptions", stream: bool) -> dict[str, Any]:
    assistant_blocks: list[dict] = []
    tokens_in = 0
    tokens_out = 0
    printed_any = False

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                bd = _block_to_dict(block)
                if bd is None:
                    continue
                assistant_blocks.append(bd)
                if stream and bd.get("type") == "text" and bd.get("text"):
                    console.print(bd["text"], end="", soft_wrap=True, highlight=False)
                    printed_any = True
                elif bd.get("type") == "tool_use":
                    name = bd.get("name", "?")
                    inp = bd.get("input") or {}
                    console.print(f"\n[dim]> tool: {name}({_short(inp)})[/dim]")
                    log("tool_call", tool=name)
        elif isinstance(message, UserMessage):
            # tool_result blocks come back in a UserMessage. Surface failures
            # in debug log; the model already sees them via the SDK's loop.
            for block in (message.content or []):
                bd = _block_to_dict(block)
                if bd and bd.get("type") == "tool_result" and bd.get("is_error"):
                    log("tool_result_error", content=str(bd.get("content"))[:200])
        elif isinstance(message, ResultMessage):
            usage = message.usage or {}
            tokens_in = int(usage.get("input_tokens", 0) or 0) + int(
                usage.get("cache_read_input_tokens", 0) or 0
            ) + int(usage.get("cache_creation_input_tokens", 0) or 0)
            tokens_out = int(usage.get("output_tokens", 0) or 0)
            if message.is_error:
                log("agent_error", subtype=message.subtype, result=str(message.result)[:200])
                warn(f"agent_error subtype={message.subtype}")
        # SystemMessage: init / context info — ignore.

    if printed_any:
        console.print("")  # newline after streamed text

    return {
        "assistant_blocks": assistant_blocks,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def _block_to_dict(block: Any) -> dict[str, Any] | None:
    if isinstance(block, dict):
        return block
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "text": getattr(block, "thinking", "") or ""}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": getattr(block, "is_error", False) or False,
        }
    return None


# -----------------------------------------------------------------------------
# Permission mapping
# -----------------------------------------------------------------------------


def _map_tool_target(tool_name: str, tool_input: dict[str, Any]) -> tuple[str | None, str]:
    """Map an SDK tool invocation to (mnemara_policy_tool, target_string).

    Returns (None, "") for tools we don't have a Mnemara policy for (e.g.
    user MCP tools — those are governed by allowed_tools / the MCP server
    itself, not Mnemara's tool policy).
    """
    if tool_name == "Bash":
        return "Bash", str(tool_input.get("command", ""))
    if tool_name == "Read":
        return "Read", str(tool_input.get("file_path") or tool_input.get("path") or "")
    if tool_name == "Write":
        return "Write", str(tool_input.get("file_path") or tool_input.get("path") or "")
    if tool_name == "Edit":
        return "Edit", str(tool_input.get("file_path") or tool_input.get("path") or "")
    if tool_name == _WRITE_MEMORY_TOOL or tool_name.endswith("__write_memory"):
        return "WriteMemory", ""
    return None, ""


def _short(d: dict, n: int = 80) -> str:
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"
