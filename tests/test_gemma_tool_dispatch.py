"""Tests for GemmaSession MCP tool dispatch.

All tests are offline — they mock httpx streams and asyncio subprocess so
Ollama and real MCP servers are never contacted.
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def home(monkeypatch, tmp_path):
    """Redirect all ~/.mnemara writes to a temp dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture
def gemma_session(home):
    """Return a GemmaSession wired to a fresh in-memory store."""
    from mnemara import config as cfg_mod
    from mnemara.config import Config, McpServer
    from mnemara.gemma_agent import GemmaSession
    from mnemara.permissions import PermissionStore
    from mnemara.store import Store
    from mnemara.tools import ToolRunner

    cfg_mod.init_instance("gtest")
    cfg = Config.default()
    cfg.model = "gemma4:26b"
    # Configure a single MCP server — tests will mock the subprocess.
    cfg.mcp_servers = [
        McpServer(name="fetch", command="uvx", args=["mcp-server-fetch"])
    ]
    # Auto-allow fetch tool to avoid interactive permission prompts.
    from mnemara.config import ToolPolicy
    cfg.allowed_tools.append(ToolPolicy(tool="fetch__fetch", mode="allow"))

    store = Store("gtest")
    perms = PermissionStore("gtest")
    runner = ToolRunner(
        instance="gtest",
        cfg=cfg,
        perms=perms,
        prompt=lambda tool, target: "allow",
    )
    session = GemmaSession(cfg=cfg, store=store, runner=runner)
    yield session
    store.close()


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def _ollama_text_chunk(text: str, done: bool = False) -> str:
    chunk: dict[str, Any] = {"message": {"role": "assistant", "content": text}}
    if done:
        chunk["done"] = True
        chunk["prompt_eval_count"] = 10
        chunk["eval_count"] = 5
    return json.dumps(chunk)


def _ollama_tool_chunk(tool_name: str, arguments: dict, done: bool = True) -> str:
    """Return a JSON chunk that carries a tool_call."""
    chunk: dict[str, Any] = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}],
        },
    }
    if done:
        chunk["done"] = True
        chunk["prompt_eval_count"] = 8
        chunk["eval_count"] = 3
    return json.dumps(chunk)


def _ollama_done_chunk() -> str:
    return json.dumps({"done": True, "prompt_eval_count": 0, "eval_count": 0})


# ---------------------------------------------------------------------------
# McpClient unit tests
# ---------------------------------------------------------------------------


class TestMcpToolToOllama:
    """Tests for _mcp_tool_to_ollama helper."""

    def test_basic_conversion(self):
        from mnemara.gemma_agent import _mcp_tool_to_ollama

        mcp_tool = {
            "name": "fetch",
            "description": "Fetch a URL",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        }
        result = _mcp_tool_to_ollama(mcp_tool, "myfetch")
        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "myfetch__fetch"
        assert fn["description"] == "Fetch a URL"
        assert fn["parameters"]["properties"]["url"]["type"] == "string"

    def test_no_server_name(self):
        from mnemara.gemma_agent import _mcp_tool_to_ollama

        result = _mcp_tool_to_ollama({"name": "echo", "description": ""}, "")
        assert result["function"]["name"] == "echo"

    def test_missing_input_schema(self):
        from mnemara.gemma_agent import _mcp_tool_to_ollama

        result = _mcp_tool_to_ollama({"name": "noop"}, "srv")
        assert result["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_namespace_separates_server_and_tool(self):
        from mnemara.gemma_agent import _mcp_tool_to_ollama

        result = _mcp_tool_to_ollama({"name": "search"}, "brave")
        assert result["function"]["name"] == "brave__search"


class TestMcpClientHandshake:
    """McpClient subprocess management — subprocess is mocked."""

    def _make_proc(self, responses: list[dict]) -> MagicMock:
        """Build a mock asyncio Process that emits the given JSON-RPC responses."""
        lines = [json.dumps(r).encode() + b"\n" for r in responses]
        # Append a sentinel empty bytes so readline() doesn't hang
        lines.append(b"")

        proc = MagicMock()
        # stdin.write is sync in asyncio.StreamWriter; only drain() is async.
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdin.is_closing.return_value = False
        proc.terminate = MagicMock()
        proc.wait = AsyncMock(return_value=0)

        async def _readline():
            if lines:
                return lines.pop(0)
            return b""

        proc.stdout = MagicMock()
        proc.stdout.readline = _readline
        return proc

    @pytest.mark.asyncio
    async def test_start_success(self, tmp_path):
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="fetch", command="uvx", args=["mcp-server-fetch"])
        client = McpClient(srv)

        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
        tools_resp = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "fetch",
                        "description": "Fetch a URL",
                        "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}},
                    }
                ]
            },
        }
        proc = self._make_proc([init_resp, tools_resp])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            ok = await client.start()

        assert ok is True
        assert client._ready is True
        assert len(client.tools) == 1
        assert client.tools[0]["name"] == "fetch"

    @pytest.mark.asyncio
    async def test_start_failure_on_spawn(self):
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="bad", command="nonexistent-cmd", args=[])
        client = McpClient(srv)

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("no cmd")):
            ok = await client.start()

        assert ok is False
        assert client._ready is False

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self):
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="fetch", command="uvx", args=["mcp-server-fetch"])
        client = McpClient(srv)

        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {}}
        tools_resp = {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}
        call_resp = {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"content": [{"type": "text", "text": "Hello from fetch!"}]},
        }
        proc = self._make_proc([init_resp, tools_resp, call_resp])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            await client.start()
            result = await client.call_tool("fetch", {"url": "https://example.com"})

        assert result == "Hello from fetch!"

    @pytest.mark.asyncio
    async def test_call_tool_mcp_error_response(self):
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="fetch", command="uvx", args=[])
        client = McpClient(srv)

        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {}}
        tools_resp = {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}
        error_resp = {
            "jsonrpc": "2.0",
            "id": 3,
            "error": {"code": -32600, "message": "Invalid URL"},
        }
        proc = self._make_proc([init_resp, tools_resp, error_resp])

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            await client.start()
            result = await client.call_tool("fetch", {"url": "bad"})

        assert "MCP error" in result
        assert "Invalid URL" in result

    @pytest.mark.asyncio
    async def test_call_tool_not_ready_returns_error(self):
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="x", command="x", args=[])
        client = McpClient(srv)
        # Do NOT call start() — client._ready stays False
        result = await client.call_tool("any", {})
        assert "not ready" in result

    @pytest.mark.asyncio
    async def test_recv_skips_noise_lines(self):
        """_recv should skip non-JSON lines and blank lines."""
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="noisy", command="x", args=[])
        client = McpClient(srv)

        target = {"jsonrpc": "2.0", "id": 1, "result": {}}
        noise_lines = [
            b"Starting server...\n",
            b"\n",
            b"  \n",
            (json.dumps(target) + "\n").encode(),
        ]

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdin.is_closing.return_value = False
        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(side_effect=noise_lines)
        client._proc = proc

        result = await client._recv()
        assert result == target

    @pytest.mark.asyncio
    async def test_close_terminates_process(self):
        from mnemara.config import McpServer
        from mnemara.gemma_agent import McpClient

        srv = McpServer(name="x", command="x", args=[])
        client = McpClient(srv)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdin.is_closing.return_value = False
        proc.terminate = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        client._proc = proc
        client._ready = True

        await client.close()
        proc.terminate.assert_called_once()
        assert client._proc is None
        assert client._ready is False


# ---------------------------------------------------------------------------
# GemmaSession.turn_async — text-only (no tools)
# ---------------------------------------------------------------------------


class TestGemmaSessionTextOnly:
    """Baseline: turn_async with no MCP servers configured."""

    def _make_session(self, home):
        from mnemara import config as cfg_mod
        from mnemara.config import Config
        from mnemara.gemma_agent import GemmaSession
        from mnemara.permissions import PermissionStore
        from mnemara.store import Store
        from mnemara.tools import ToolRunner

        cfg_mod.init_instance("text_only")
        cfg = Config.default()
        cfg.mcp_servers = []  # no tools
        store = Store("text_only")
        perms = PermissionStore("text_only")
        runner = ToolRunner("text_only", cfg, perms, lambda t, x: "allow")
        return GemmaSession(cfg=cfg, store=store, runner=runner), store

    @pytest.mark.asyncio
    async def test_plain_text_response(self, home):
        session, store = self._make_session(home)

        lines = [
            _ollama_text_chunk("Hello"),
            _ollama_text_chunk(", world!", done=True),
        ]

        with _patch_ollama_stream(lines):
            result = await session.turn_async("Hi")

        assert result["stop_reason"] == "end_turn"
        blocks = result["assistant_blocks"]
        text = next(b["text"] for b in blocks if b["type"] == "text")
        assert "Hello, world!" in text
        store.close()

    @pytest.mark.asyncio
    async def test_tokens_counted(self, home):
        session, store = self._make_session(home)

        lines = [_ollama_text_chunk("ok", done=True)]
        lines[-1] = json.dumps(
            {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 42, "eval_count": 7}
        )

        with _patch_ollama_stream(lines):
            result = await session.turn_async("count")

        assert result["tokens_in"] == 42
        assert result["tokens_out"] == 7
        store.close()

    @pytest.mark.asyncio
    async def test_connect_error_surfaces_as_block(self, home):
        session, store = self._make_session(home)

        with patch(
            "mnemara.gemma_agent.httpx.AsyncClient",
            side_effect=Exception("connect refused"),
        ):
            result = await session.turn_async("fail?")

        # Should not raise — error text in blocks
        blocks = result["assistant_blocks"]
        text = next(b["text"] for b in blocks if b["type"] == "text")
        assert "Gemma error" in text or "connect refused" in text
        store.close()


# ---------------------------------------------------------------------------
# GemmaSession.turn_async — tool dispatch path
# ---------------------------------------------------------------------------


class _FakeMcpClient:
    """Stands in for McpClient in dispatch tests."""

    def __init__(self, tools_list, call_result="<fetched content>"):
        self.tools = tools_list
        self._call_result = call_result
        self._ready = True
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self._call_result

    async def start(self):
        return True

    async def close(self):
        pass


def _patch_ollama_stream(lines: list[str]):
    """Context manager: replace httpx.AsyncClient.stream with a fake that
    yields the given newline-delimited JSON strings."""

    class _FakeStream:
        def __init__(self, lines):
            self._lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def stream(self, method, url, json=None):
            return _FakeStream(lines)

    return patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_FakeClient())


class TestGemmaSessionToolDispatch:

    @pytest.mark.asyncio
    async def test_single_tool_call_dispatched(self, gemma_session):
        """Model requests one tool call → dispatched → final text returned."""
        fake_client = _FakeMcpClient(
            tools_list=[{"name": "fetch", "description": "Fetch URL",
                         "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}}],
            call_result="Page content here",
        )
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__fetch", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        # Iteration 1: Ollama returns a tool call.
        # Iteration 2: Ollama returns plain text.
        iteration1 = _ollama_tool_chunk("fetch__fetch", {"url": "https://example.com"}, done=True)
        iteration2 = _ollama_text_chunk("The page says: Page content here", done=True)
        call_count = [0]

        class _TwoPassClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, method, url, json=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    return _FakeStreamFromLines([iteration1])
                else:
                    return _FakeStreamFromLines([iteration2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TwoPassClient()):
            result = await gemma_session.turn_async("Fetch example.com")

        assert len(fake_client.calls) == 1
        name, args = fake_client.calls[0]
        assert name == "fetch"
        assert args.get("url") == "https://example.com"

        blocks = result["assistant_blocks"]
        tool_use_blocks = [b for b in blocks if b["type"] == "tool_use"]
        tool_result_blocks = [b for b in blocks if b["type"] == "tool_result"]
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert len(tool_use_blocks) == 1
        assert len(tool_result_blocks) == 1
        assert "Page content here" in text_blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_tool_result_included_in_context(self, gemma_session):
        """Tool result appears in messages sent to Ollama on second pass."""
        fake_client = _FakeMcpClient([], call_result="result_text")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__echo", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        second_pass_messages: list[dict] = []
        iteration1 = _ollama_tool_chunk("fetch__echo", {}, done=True)
        iteration2 = _ollama_text_chunk("done", done=True)

        class _CapturingClient:
            def __init__(self):
                self._pass = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, method, url, json=None):
                self._pass += 1
                if self._pass == 1:
                    return _FakeStreamFromLines([iteration1])
                else:
                    second_pass_messages.extend(json.get("messages", []))
                    return _FakeStreamFromLines([iteration2])

        capturing = _CapturingClient()
        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=capturing):
            await gemma_session.turn_async("go")

        # The second-pass messages should include a tool result.
        roles = [m["role"] for m in second_pass_messages]
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_permission_denied_propagates(self, gemma_session):
        """When permission is denied, tool result contains denial message."""
        fake_client = _FakeMcpClient([], call_result="should not see this")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__secret", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True
        # Override runner.prompt to deny.
        gemma_session.runner.prompt = lambda t, x: "deny"

        iteration1 = _ollama_tool_chunk("fetch__secret", {}, done=True)
        iteration2 = _ollama_text_chunk("ok", done=True)

        class _TwoPass:
            def __init__(self):
                self._n = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                self._n += 1
                return _FakeStreamFromLines([iteration1 if self._n == 1 else iteration2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TwoPass()):
            result = await gemma_session.turn_async("secret fetch")

        # The tool_result block should mention permission denial.
        tool_result_blocks = [b for b in result["assistant_blocks"] if b["type"] == "tool_result"]
        assert any("Permission denied" in str(b.get("content", "")) for b in tool_result_blocks)

    @pytest.mark.asyncio
    async def test_tool_call_loop_limit(self, gemma_session):
        """Tool call loop is capped at _MAX_TOOL_LOOPS; does not hang."""
        from mnemara.gemma_agent import _MAX_TOOL_LOOPS

        fake_client = _FakeMcpClient([], call_result="result")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__loop", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        iteration = _ollama_tool_chunk("fetch__loop", {}, done=True)

        class _AlwaysToolCall:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                return _FakeStreamFromLines([iteration])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_AlwaysToolCall()):
            result = await gemma_session.turn_async("loop")

        # Should complete without error; tool call count bounded.
        assert len(fake_client.calls) <= _MAX_TOOL_LOOPS

    @pytest.mark.asyncio
    async def test_on_tool_use_callback_fired(self, gemma_session):
        """on_tool_use callback is called with (tool_name, args)."""
        fake_client = _FakeMcpClient([], call_result="r")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__fetch", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        fired: list[tuple] = []

        def on_use(pair):
            fired.append(pair)

        iter1 = _ollama_tool_chunk("fetch__fetch", {"url": "x"}, done=True)
        iter2 = _ollama_text_chunk("done", done=True)

        class _TP:
            def __init__(self):
                self._n = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                self._n += 1
                return _FakeStreamFromLines([iter1 if self._n == 1 else iter2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TP()):
            await gemma_session.turn_async("go", on_tool_use=on_use)

        assert len(fired) == 1
        assert fired[0][0] == "fetch__fetch"

    @pytest.mark.asyncio
    async def test_on_tool_result_callback_fired(self, gemma_session):
        """on_tool_result callback is called with (tool_name, result)."""
        fake_client = _FakeMcpClient([], call_result="the_result")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__fetch", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        results: list[tuple] = []

        def on_result(pair):
            results.append(pair)

        iter1 = _ollama_tool_chunk("fetch__fetch", {}, done=True)
        iter2 = _ollama_text_chunk("done", done=True)

        class _TP:
            def __init__(self):
                self._n = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                self._n += 1
                return _FakeStreamFromLines([iter1 if self._n == 1 else iter2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TP()):
            await gemma_session.turn_async("go", on_tool_result=on_result)

        assert len(results) == 1
        assert results[0][1] == "the_result"

    @pytest.mark.asyncio
    async def test_tools_counted_in_session_stats(self, gemma_session):
        """tools_called counter increments for each dispatched tool."""
        fake_client = _FakeMcpClient([], call_result="x")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__fetch", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        iter1 = _ollama_tool_chunk("fetch__fetch", {}, done=True)
        iter2 = _ollama_text_chunk("done", done=True)

        class _TP:
            def __init__(self):
                self._n = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                self._n += 1
                return _FakeStreamFromLines([iter1 if self._n == 1 else iter2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TP()):
            await gemma_session.turn_async("go")

        assert gemma_session.tools_called.get("fetch__fetch", 0) == 1

    @pytest.mark.asyncio
    async def test_unknown_server_returns_error_block(self, gemma_session):
        """Tool call for an unconfigured server → error in tool_result block."""
        gemma_session._mcp_clients = {}  # no servers
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "ghost__tool", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        iter1 = _ollama_tool_chunk("ghost__tool", {}, done=True)
        iter2 = _ollama_text_chunk("ok", done=True)

        class _TP:
            def __init__(self):
                self._n = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                self._n += 1
                return _FakeStreamFromLines([iter1 if self._n == 1 else iter2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TP()):
            result = await gemma_session.turn_async("go")

        tr_blocks = [b for b in result["assistant_blocks"] if b["type"] == "tool_result"]
        assert any("unknown server" in str(b.get("content", "")) for b in tr_blocks)

    @pytest.mark.asyncio
    async def test_arguments_as_json_string_normalised(self, gemma_session):
        """Ollama sometimes returns arguments as a JSON string; it should be parsed."""
        fake_client = _FakeMcpClient([], call_result="ok")
        gemma_session._mcp_clients["fetch"] = fake_client
        gemma_session._ollama_tools = [
            {"type": "function", "function": {"name": "fetch__fetch", "description": "", "parameters": {}}}
        ]
        gemma_session._mcp_initialized = True

        # Arguments as a JSON-encoded string (not a dict)
        raw_args = json.dumps({"url": "https://example.com"})
        iter1 = json.dumps({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fetch__fetch", "arguments": raw_args}}],
            },
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        })
        iter2 = _ollama_text_chunk("done", done=True)

        class _TP:
            def __init__(self):
                self._n = 0
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
            def stream(self, *a, **kw):
                self._n += 1
                return _FakeStreamFromLines([iter1 if self._n == 1 else iter2])

        with patch("mnemara.gemma_agent.httpx.AsyncClient", return_value=_TP()):
            await gemma_session.turn_async("fetch with string args")

        # Should have dispatched with parsed dict, not raw string
        assert len(fake_client.calls) == 1
        _, args = fake_client.calls[0]
        assert isinstance(args, dict)
        assert args.get("url") == "https://example.com"


# ---------------------------------------------------------------------------
# _init_mcp
# ---------------------------------------------------------------------------


class TestInitMcp:

    @pytest.mark.asyncio
    async def test_idempotent(self, gemma_session):
        """_init_mcp must not start servers twice."""
        started: list[str] = []

        async def _fake_start(client_self):
            started.append(client_self.server.name)
            client_self._ready = True
            client_self.tools = []
            return True

        with patch("mnemara.gemma_agent.McpClient.start", _fake_start):
            await gemma_session._init_mcp()
            await gemma_session._init_mcp()  # second call must be a no-op

        assert started.count("fetch") == 1

    @pytest.mark.asyncio
    async def test_failed_server_skipped(self, gemma_session):
        """If a server fails to start, tools remain empty for that server."""

        async def _fail(client_self):
            return False

        with patch("mnemara.gemma_agent.McpClient.start", _fail):
            await gemma_session._init_mcp()

        assert "fetch" not in gemma_session._mcp_clients
        assert gemma_session._ollama_tools == []


# ---------------------------------------------------------------------------
# aclose
# ---------------------------------------------------------------------------


class TestAclose:

    @pytest.mark.asyncio
    async def test_aclose_closes_all_clients(self, gemma_session):
        closed: list[str] = []

        class _FakeClient:
            def __init__(self, name):
                self.name = name
            async def close(self):
                closed.append(self.name)

        gemma_session._mcp_clients = {
            "a": _FakeClient("a"),
            "b": _FakeClient("b"),
        }
        gemma_session._stats_written = True  # skip stats write

        await gemma_session.aclose()

        assert set(closed) == {"a", "b"}
        assert gemma_session._mcp_clients == {}


# ---------------------------------------------------------------------------
# Stream helper used across tests
# ---------------------------------------------------------------------------


class _FakeStreamFromLines:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line
