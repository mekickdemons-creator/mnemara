"""Agent loop: stream a response, run tools, loop until end_turn.

Persists a complete user turn (the user input as one DB row) and a complete
assistant turn (the full content-block list, possibly spanning multiple
sub-responses while tool use is in flight) as one DB row each.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from rich.console import Console

from . import role as role_mod
from .config import Config
from .logging_util import log
from .mcp import build_mcp_param
from .store import Store
from .tools import TOOL_DEFS, ToolRunner

console = Console()


class AgentSession:
    def __init__(self, cfg: Config, store: Store, runner: ToolRunner, client):
        self.cfg = cfg
        self.store = store
        self.runner = runner
        self.client = client

    def turn(self, user_text: str) -> dict[str, Any]:
        """Run a full user->assistant turn (with tool loops). Returns usage info."""
        # Persist user turn first.
        user_blocks = [{"type": "text", "text": user_text}]
        self.store.append_turn("user", user_blocks)

        system_prompt = role_mod.load_role_doc(self.cfg.role_doc_path)
        messages = self.store.messages_for_api()

        assistant_blocks: list[dict] = []
        total_in = 0
        total_out = 0

        while True:
            kwargs = dict(
                model=self.cfg.model,
                max_tokens=4096,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
                tools=TOOL_DEFS,
            )
            mcp = build_mcp_param(self.cfg)
            if mcp:
                kwargs["mcp_servers"] = mcp

            try:
                if self.cfg.stream:
                    final_message = self._stream_once(kwargs)
                else:
                    final_message = self.client.messages.create(**kwargs)
            except TypeError:
                # SDK doesn't know mcp_servers — drop and retry.
                kwargs.pop("mcp_servers", None)
                if self.cfg.stream:
                    final_message = self._stream_once(kwargs)
                else:
                    final_message = self.client.messages.create(**kwargs)

            usage = getattr(final_message, "usage", None)
            if usage is not None:
                total_in += getattr(usage, "input_tokens", 0) or 0
                total_out += getattr(usage, "output_tokens", 0) or 0

            content = _content_to_blocks(final_message.content)
            assistant_blocks.extend(content)

            stop_reason = getattr(final_message, "stop_reason", None)
            if stop_reason != "tool_use":
                break

            # Append assistant message with the tool_use blocks to the live conv.
            messages.append({"role": "assistant", "content": content})

            # Run all tool_use blocks in this response, in order.
            tool_results = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                name = block["name"]
                tool_id = block["id"]
                params = block.get("input") or {}
                console.print(f"[dim]> tool: {name}({_short(params)})[/dim]")
                result_text, is_error = self.runner.dispatch(name, params)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )
                log("tool_run", tool=name, error=is_error)

            messages.append({"role": "user", "content": tool_results})
            # Loop back into the model with results.

        # Persist final assistant turn (full block list) and evict.
        self.store.append_turn(
            "assistant",
            assistant_blocks,
            tokens_in=total_in,
            tokens_out=total_out,
        )
        evicted = self.store.evict(self.cfg.max_window_turns, self.cfg.max_window_tokens)
        if evicted:
            log("eviction", deleted=evicted)
        return {"input_tokens": total_in, "output_tokens": total_out, "evicted": evicted}

    def _stream_once(self, kwargs: dict[str, Any]):
        """Run one streaming request, render text deltas, return the final message."""
        printed_any = False
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is not None and getattr(delta, "type", None) == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            console.print(text, end="", soft_wrap=True, highlight=False)
                            printed_any = True
            final = stream.get_final_message()
        if printed_any:
            console.print("")  # newline after stream
        return final


def _content_to_blocks(content) -> list[dict]:
    out = []
    for b in content:
        if isinstance(b, dict):
            out.append(b)
            continue
        # SDK object — pull its dict representation.
        d: dict[str, Any] = {"type": getattr(b, "type", "unknown")}
        for attr in ("text", "name", "id", "input", "tool_use_id", "content", "is_error"):
            if hasattr(b, attr):
                v = getattr(b, attr)
                if v is not None:
                    d[attr] = v
        out.append(d)
    return out


def _short(d: dict, n: int = 80) -> str:
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"
