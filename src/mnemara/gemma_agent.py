"""Gemma backend for Mnemara.

Drops the claude-agent-sdk entirely; drives one Ollama /api/chat streaming
call per turn.  The public interface (GemmaSession) mirrors AgentSession
closely enough that tui.py and cli.py can swap the import and work unchanged.

No tool use in this first pass — pure chat with role doc + rolling window.
Tool calls from Gemma are logged but not executed (Gemma may hallucinate
JSON tool calls; they appear in the transcript but are not acted on).
"""

from __future__ import annotations

import asyncio
import json
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
from .config import Config
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


def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Flatten a stored block list into plain text for Ollama's message format."""
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            # Represent tool calls as readable stubs in the context.
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


class GemmaSession:
    """Ollama-backed conversation session.

    Drop-in replacement for AgentSession: same constructor signature,
    same turn_async() return schema, same write_session_stats() method.
    """

    def __init__(self, cfg: Config, store: Store, runner: ToolRunner, client: Any = None):
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

    @property
    def _model(self) -> str:
        return self.cfg.model if self.cfg.model else _DEFAULT_MODEL

    async def turn_async(
        self,
        user_text: str,
        on_token: OnToken | None = None,
        on_tool_use: OnToolUse | None = None,
        on_tool_result: OnToolResult | None = None,
    ) -> dict[str, Any]:
        """Drive one conversation turn through Ollama.

        Persists the user turn, streams Gemma's response, persists the
        assistant turn, runs eviction, and returns a result dict that
        mirrors AgentSession's schema so tui.py works unchanged.
        """
        # 1. Persist the user turn.
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

        # 4. Stream from Ollama.
        full_text = ""
        tokens_in = 0
        tokens_out = 0

        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    _OLLAMA_CHAT_URL,
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        if not raw_line.strip():
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        # Token text
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            full_text += delta
                            if on_token is not None:
                                await _call_cb(on_token, delta)
                        # Usage (final chunk has done=True and eval_count etc.)
                        if chunk.get("done"):
                            tokens_in = chunk.get("prompt_eval_count", 0)
                            tokens_out = chunk.get("eval_count", 0)
        except httpx.ConnectError as exc:
            error_msg = (
                f"[Gemma backend unavailable: {exc}. "
                f"Is Ollama running? `ollama serve` then retry.]"
            )
            full_text = error_msg
            if on_token is not None:
                await _call_cb(on_token, error_msg)
        except Exception as exc:
            error_msg = f"[Gemma error: {exc}]"
            full_text = error_msg
            if on_token is not None:
                await _call_cb(on_token, error_msg)

        # 5. Persist the assistant turn.
        assistant_blocks = [{"type": "text", "text": full_text}]
        self.store.append_turn(
            "assistant",
            assistant_blocks,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # 6. Evict to stay within the rolling window.
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

    def write_session_stats(self) -> None:
        """Write per-session stats to ~/.mnemara/<instance>/stats/YYYY-MM-DD.json.

        Mirrors AgentSession.write_session_stats() so tui.py's on_unmount
        hook works unchanged. Idempotent — only writes once per GemmaSession.
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
