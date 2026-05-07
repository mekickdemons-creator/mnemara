"""Gemma backend for Mnemara.

Drops the claude-agent-sdk entirely; drives Ollama's /api/chat per turn.
The public interface (GemmaSession) mirrors AgentSession closely enough
that tui.py and cli.py can swap the import and work unchanged.

MCP tool use is supported: configure ``mcp_servers`` in config.json and
Gemma can call them.  Each server is started as a stdio subprocess on the
first turn that needs tools, kept alive for the session, and torn down on
``aclose()``.  Ollama must support function-calling for the model in use
(gemma4:26b and gemma3:27b both do).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Optional, Union


async def _call_cb(cb: Callable, arg: Any) -> None:
    """Call a callback that may be sync or async."""
    result = cb(arg)
    if asyncio.iscoroutine(result):
        await result


import httpx
from rich.console import Console

from . import paths as paths_mod
from . import role as role_mod
from .config import Config, McpServer
from .logging_util import log
from .store import Store
from .tools import ToolRunner

console = Console()

# Type aliases matching agent.py's convention.
OnToken = Callable[[str], Awaitable[None]]
OnToolUse = Callable[[str, dict[str, Any]], Awaitable[None]]
OnToolResult = Callable[[str, Any], Awaitable[None]]

_DEFAULT_MODEL = "gemma4:26b"
_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
_OLLAMA_TIMEOUT = 300.0  # seconds — Gemma can be slow on first token
_MCP_REQUEST_TIMEOUT = 30.0  # seconds per JSON-RPC round trip
_MAX_TOOL_LOOPS = 10  # prevent runaway tool-call chains

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Flatten a stored block list into plain text for Ollama's message format."""
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "tool")
            inp = json.dumps(block.get("input", {}), ensure_ascii=False)[:200]
            parts.append(f"[tool_use: {name}({inp})]")
        elif btype == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            parts.append(f"[tool_result: {str(content)[:200]}]")
    return "\n".join(parts).strip()


def _mcp_tool_to_ollama(tool: dict[str, Any], server_name: str) -> dict[str, Any]:
    """Convert one MCP tool descriptor to Ollama function-calling format.

    Ollama expects::

        {"type": "function", "function": {"name": ..., "description": ...,
                                          "parameters": {...}}}

    MCP provides ``name``, ``description``, and ``inputSchema``.

    Tool names are namespaced as ``<server_name>__<tool_name>`` to avoid
    collisions when multiple MCP servers are configured.
    """
    name = tool.get("name", "")
    namespaced = f"{server_name}__{name}" if server_name else name
    return {
        "type": "function",
        "function": {
            "name": namespaced,
            "description": tool.get("description", ""),
            "parameters": tool.get(
                "inputSchema", {"type": "object", "properties": {}}
            ),
        },
    }


def _build_messages(
    store: Store,
    system_prompt: str,
    current_user_text: str,
    max_turns: int,
    max_tokens: int,
) -> list[dict[str, str]]:
    """Build the Ollama messages list from the rolling window.

    Format: [system, user, assistant, user, assistant, ..., user(current)]
    Omits the last stored user turn (it's the current input, sent fresh).
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    rows = store.window(limit=max_turns)
    # Rows are oldest-first; drop the last 'user' row since that's the current input.
    if rows and rows[-1].get("role") == "user":
        rows = rows[:-1]

    for row in rows:
        role = row.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        # store.window() returns content already deserialized as a list of dicts.
        content = row.get("content", [])
        if isinstance(content, str):
            # Legacy string storage: wrap as a text block.
            content = [{"type": "text", "text": content}]
        text = _extract_text_from_blocks(content)
        if text:
            messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": current_user_text})
    return messages


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


class McpClient:
    """Manages one MCP server subprocess over stdio JSON-RPC.

    Lifecycle::

        client = McpClient(server_cfg)
        ok = await client.start()          # launches process, handshakes
        result = await client.call_tool(name, args)
        await client.close()               # terminates process
    """

    def __init__(self, server: McpServer) -> None:
        self.server = server
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._req_counter: int = 0
        self.tools: list[dict[str, Any]] = []  # populated after start()
        self._ready: bool = False

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    async def start(self) -> bool:
        """Start subprocess and complete MCP initialization.

        Returns True on success, False if the server could not be started or
        the handshake failed (error is logged; caller should continue without
        this server's tools).
        """
        env = {**os.environ, **self.server.env} if self.server.env else None
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.server.command,
                *self.server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
        except Exception as exc:
            log("mcp_start_error", server=self.server.name, error=str(exc))
            return False

        try:
            # Initialize handshake
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "mnemara-gemma", "version": "0.4.1"},
                    },
                }
            )
            await asyncio.wait_for(self._recv(), timeout=_MCP_REQUEST_TIMEOUT)

            # initialized notification — no response expected
            await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

            # List available tools
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/list",
                }
            )
            resp = await asyncio.wait_for(self._recv(), timeout=_MCP_REQUEST_TIMEOUT)
            self.tools = resp.get("result", {}).get("tools", [])
            self._ready = True
            log(
                "mcp_ready",
                server=self.server.name,
                tools=[t.get("name") for t in self.tools],
            )
            return True

        except Exception as exc:
            log("mcp_init_error", server=self.server.name, error=str(exc))
            await self.close()
            return False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch one tool call and return result as a plain string."""
        if not self._ready or self._proc is None:
            return "[MCP error: server not ready]"
        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
            )
            resp = await asyncio.wait_for(self._recv(), timeout=_MCP_REQUEST_TIMEOUT)
        except Exception as exc:
            return f"[MCP error: {exc}]"

        if "error" in resp:
            err = resp["error"]
            return f"[MCP error: {err.get('message', str(err))}]"

        content = resp.get("result", {}).get("content", [])
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        parts.append(f"[image/{item.get('mimeType', 'unknown')}]")
            return "\n".join(parts)
        return str(content)

    async def _send(self, obj: dict[str, Any]) -> None:
        """Write one JSON-RPC message to the subprocess's stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("process not started")
        data = (json.dumps(obj) + "\n").encode()
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def _recv(self) -> dict[str, Any]:
        """Read the next JSON-RPC message from the subprocess's stdout.

        Skips blank lines and non-JSON noise (some servers emit startup banners).
        """
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("process not started")
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise RuntimeError("MCP server stdout closed unexpectedly")
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # skip noise / progress lines

    async def close(self) -> None:
        """Terminate the subprocess gracefully."""
        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.is_closing():
                    self._proc.stdin.close()
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._ready = False


# ---------------------------------------------------------------------------
# GemmaSession
# ---------------------------------------------------------------------------


class GemmaSession:
    """Ollama-backed conversation session.

    Drop-in replacement for AgentSession: same constructor signature,
    same turn_async() return schema, same write_session_stats() method.

    MCP tool use:
        Declare ``mcp_servers`` in config.json.  On the first turn that
        needs tools, GemmaSession starts each server, negotiates the MCP
        handshake, and advertises the available tools to Ollama via the
        ``tools`` request field.  When Gemma calls a tool, it is dispatched
        to the appropriate MCP server (permission gates from ``tools.py``
        apply).  Results are appended as ``role: "tool"`` messages and the
        model is re-invoked until it produces a plain text response (up to
        ``_MAX_TOOL_LOOPS`` iterations).

        Tool names are namespaced ``<server>__<tool>`` so multiple servers
        can coexist without collision.
    """

    def __init__(
        self,
        cfg: Config,
        store: Store,
        runner: ToolRunner,
        client: Any = None,
    ):
        self.cfg = cfg
        self.store = store
        self.runner = runner
        self.client = client  # ignored; kept for API compatibility

        # Session-scoped counters (mirrors AgentSession v0.2.0 instrumentation).
        self.session_started_at = datetime.now(timezone.utc).isoformat()
        self.session_ended_at: Optional[str] = None
        self.evicted_this_session: int = 0
        self.tools_called: dict[str, int] = {}
        self.memory_writes: int = 0
        self.role_proposals: int = 0
        self.choices_logged: int = 0
        self.wiki_writes: int = 0
        self._stats_written: bool = False

        # MCP state — initialized lazily on first turn with tool-enabled model.
        self._mcp_clients: dict[str, McpClient] = {}
        self._ollama_tools: list[dict[str, Any]] = []
        self._mcp_initialized: bool = False

    # ------------------------------------------------------------------ props

    @property
    def _model(self) -> str:
        return self.cfg.model if self.cfg.model else _DEFAULT_MODEL

    # ------------------------------------------------------------------ MCP init

    async def _init_mcp(self) -> None:
        """Lazily start all configured MCP servers.  Idempotent."""
        if self._mcp_initialized:
            return
        self._mcp_initialized = True  # set before awaiting so concurrent turns don't double-init

        for srv in self.cfg.mcp_servers:
            client = McpClient(srv)
            ok = await client.start()
            if ok:
                self._mcp_clients[srv.name] = client
                for tool in client.tools:
                    self._ollama_tools.append(_mcp_tool_to_ollama(tool, srv.name))
            else:
                log("mcp_server_skipped", server=srv.name)

    # ------------------------------------------------------------------ MCP dispatch

    async def _dispatch_mcp_call(
        self,
        namespaced_name: str,
        arguments: dict[str, Any],
        on_tool_use: OnToolUse | None,
        on_tool_result: OnToolResult | None,
    ) -> str:
        """Dispatch one MCP tool call with permission gate.

        Tool name format: ``<server_name>__<tool_name>``.
        Permission is checked against the full namespaced name so operators
        can grant blanket access with ``{"tool": "fetch__fetch", "mode": "allow"}``.
        """
        # Un-namespace
        if "__" in namespaced_name:
            server_name, tool_name = namespaced_name.split("__", 1)
        else:
            server_name = ""
            tool_name = namespaced_name

        # Permission gate
        ok, err = self.runner._check_perm(
            namespaced_name, json.dumps(arguments, ensure_ascii=False)[:120]
        )
        if not ok:
            result = f"[Permission denied: {err}]"
            if on_tool_result is not None:
                await _call_cb(on_tool_result, (namespaced_name, result))
            return result

        # Fire on_tool_use callback
        if on_tool_use is not None:
            await _call_cb(on_tool_use, (namespaced_name, arguments))

        # Resolve client
        client = self._mcp_clients.get(server_name)
        if client is None and not server_name:
            # No namespace — fall back to first available client
            client = next(iter(self._mcp_clients.values()), None)
            tool_name = namespaced_name
        if client is None:
            result = f"[MCP error: unknown server '{server_name}']"
            if on_tool_result is not None:
                await _call_cb(on_tool_result, (namespaced_name, result))
            return result

        result = await client.call_tool(tool_name, arguments)

        # Track usage
        self.tools_called[namespaced_name] = (
            self.tools_called.get(namespaced_name, 0) + 1
        )
        log("mcp_tool_called", name=namespaced_name, result_len=len(result))

        if on_tool_result is not None:
            await _call_cb(on_tool_result, (namespaced_name, result))

        return result

    # ------------------------------------------------------------------ turn

    async def turn_async(
        self,
        user_text: str,
        on_token: OnToken | None = None,
        on_tool_use: OnToolUse | None = None,
        on_tool_result: OnToolResult | None = None,
    ) -> dict[str, Any]:
        """Drive one conversation turn through Ollama.

        Persists the user turn, streams Gemma's response, executes any MCP
        tool calls (looping until the model produces a plain text response),
        persists the assistant turn, runs eviction, and returns a result dict
        that mirrors AgentSession's schema.
        """
        # 1. Persist user turn.
        user_blocks = [{"type": "text", "text": user_text}]
        self.store.append_turn("user", user_blocks)

        # 2. Build system prompt from role doc.
        system_prompt = role_mod.load_role_doc(self.cfg.role_doc_path) or (
            "You are a helpful assistant."
        )

        # 3. Build Ollama message list from the rolling window.
        messages = _build_messages(
            self.store,
            system_prompt,
            user_text,
            max_turns=self.cfg.max_window_turns,
            max_tokens=self.cfg.max_window_tokens,
        )

        # 4. Lazily initialize MCP servers (no-op after first turn).
        if self.cfg.mcp_servers:
            await self._init_mcp()

        # Tool-loop state
        full_text = ""
        tokens_in = 0
        tokens_out = 0
        # Accumulated tool_use / tool_result blocks for DB storage.
        tool_blocks: list[dict[str, Any]] = []

        for _loop_iter in range(_MAX_TOOL_LOOPS):
            # 5. Build request payload.
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "stream": True,
            }
            if self._ollama_tools:
                payload["tools"] = self._ollama_tools

            # 6. Stream one Ollama call.
            iteration_text = ""
            iteration_tool_calls: list[dict[str, Any]] = []
            iter_tokens_in = 0
            iter_tokens_out = 0

            try:
                async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as http:
                    async with http.stream(
                        "POST", _OLLAMA_CHAT_URL, json=payload
                    ) as resp:
                        resp.raise_for_status()
                        async for raw_line in resp.aiter_lines():
                            if not raw_line.strip():
                                continue
                            try:
                                chunk = json.loads(raw_line)
                            except json.JSONDecodeError:
                                continue
                            msg = chunk.get("message", {})
                            # Text delta
                            delta = msg.get("content", "")
                            if delta:
                                iteration_text += delta
                                if on_token is not None:
                                    await _call_cb(on_token, delta)
                            # Tool calls (accumulate; typically arrive in final chunk)
                            for tc in msg.get("tool_calls", []):
                                iteration_tool_calls.append(tc)
                            if chunk.get("done"):
                                iter_tokens_in = chunk.get("prompt_eval_count", 0)
                                iter_tokens_out = chunk.get("eval_count", 0)

            except httpx.ConnectError as exc:
                error_msg = (
                    f"[Gemma backend unavailable: {exc}. "
                    f"Is Ollama running? `ollama serve` then retry.]"
                )
                full_text = error_msg
                if on_token is not None:
                    await _call_cb(on_token, error_msg)
                break
            except Exception as exc:
                error_msg = f"[Gemma error: {exc}]"
                full_text += error_msg
                if on_token is not None:
                    await _call_cb(on_token, error_msg)
                break

            tokens_in += iter_tokens_in
            tokens_out += iter_tokens_out

            # No tool calls → this is the final response.
            if not iteration_tool_calls:
                full_text += iteration_text
                break

            # 7. Execute tool calls.
            # Append the assistant message (with tool_calls) to the loop context.
            loop_assistant: dict[str, Any] = {
                "role": "assistant",
                "content": iteration_text or "",
                "tool_calls": iteration_tool_calls,
            }
            messages.append(loop_assistant)

            for tc in iteration_tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
                # Ollama may return arguments as a JSON string; normalise.
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}

                result_text = await self._dispatch_mcp_call(
                    fn_name, fn_args, on_tool_use, on_tool_result
                )

                # Append tool result message so the model sees the outcome.
                messages.append({"role": "tool", "content": result_text})

                # Record for DB storage.
                tool_blocks.append({"type": "tool_use", "name": fn_name, "input": fn_args})
                tool_blocks.append({"type": "tool_result", "content": result_text})

            # Continue loop — model will now respond to the tool results.

        # 8. Build final assistant blocks: tool interactions then text response.
        assistant_blocks: list[dict[str, Any]] = list(tool_blocks)
        assistant_blocks.append({"type": "text", "text": full_text})

        # 9. Persist assistant turn.
        self.store.append_turn(
            "assistant",
            assistant_blocks,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # 10. Evict to stay within the rolling window.
        try:
            ev = self.store.evict(
                max_turns=self.cfg.max_window_turns,
                max_tokens=self.cfg.max_window_tokens,
            )
            evicted = ev.get("rows_evicted", 0)
            if evicted:
                self.evicted_this_session += evicted
        except Exception as exc:
            log("eviction_error", error=str(exc))

        return {
            "assistant_blocks": assistant_blocks,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "stop_reason": "end_turn",
        }

    # ------------------------------------------------------------------ cleanup

    async def aclose(self) -> None:
        """Terminate MCP subprocesses and write session stats."""
        for client in self._mcp_clients.values():
            await client.close()
        self._mcp_clients.clear()
        self.write_session_stats()

    def write_session_stats(self) -> None:
        """Write per-session stats to ~/.mnemara/<instance>/stats/YYYY-MM-DD.json.

        Mirrors AgentSession.write_session_stats() so tui.py's on_unmount
        hook works unchanged.  Idempotent — only writes once per GemmaSession.
        """
        if self._stats_written:
            return
        self._stats_written = True
        self.session_ended_at = datetime.now(timezone.utc).isoformat()
        try:
            instance = self.runner.instance
            stats_dir = paths_mod.stats_dir(instance)
            stats_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            stats_file = stats_dir / f"{today}.json"
            entry = {
                "session_started_at": self.session_started_at,
                "session_ended_at": self.session_ended_at,
                "model": self._model,
                "backend": "gemma_ollama",
                "evicted_this_session": self.evicted_this_session,
                "memory_writes": self.memory_writes,
                "role_proposals": self.role_proposals,
                "choices_logged": self.choices_logged,
                "wiki_writes": self.wiki_writes,
                "tools_called": self.tools_called,
            }
            # Merge with existing same-day stats if present.
            existing: list[dict[str, Any]] = []
            if stats_file.exists():
                try:
                    existing = json.loads(stats_file.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        existing = [existing]
                except Exception:
                    existing = []
            existing.append(entry)
            stats_file.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log("stats_write_error", error=str(exc))
