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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from rich.console import Console

# Streaming callback types — invoked during turn_async / turn when set.
# on_token(text):  called each time a text delta arrives from the model.
# on_tool_use(name, input):  called when the model issues a tool_use block.
# on_tool_result(tool_use_id, content, is_error):  called when a tool returns.
OnToken = Callable[[str], Optional[Awaitable[None]]]
OnToolUse = Callable[[str, dict], Optional[Awaitable[None]]]
OnToolResult = Callable[[str, Any, bool], Optional[Awaitable[None]]]

from . import paths as paths_mod
from . import role as role_mod
from . import tools as tools_mod
from .config import Config
from .logging_util import log, warn
from . import store as store_mod
from .store import Store
from .tools import ToolRunner
from .runtime_sentinel import RuntimeSentinel

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
    # HookEventMessage was added in SDK 0.1.74; import defensively so the
    # module loads on older SDKs too (the sentinel feature just won't fire).
    try:
        from claude_agent_sdk import HookEventMessage
    except ImportError:  # pragma: no cover
        HookEventMessage = None  # type: ignore[assignment,misc]
    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False
    HookEventMessage = None  # type: ignore[assignment]

console = Console()

# Built-in Claude Code tool names we expose by default.
_BUILTIN_TOOLS = ("Bash", "Read", "Write", "Edit")
# MCP-prefixed names the SDK assigns to our in-process tools.
_WRITE_MEMORY_TOOL = "mcp__mnemara_memory__write_memory"
_INSPECT_CONTEXT_TOOL = "mcp__mnemara_memory__inspect_context"
_PROPOSE_ROLE_AMENDMENT_TOOL = "mcp__mnemara_memory__propose_role_amendment"
_LOG_CHOICE_TOOL = "mcp__mnemara_memory__log_choice"
_WIKI_READ_TOOL = "mcp__mnemara_memory__wiki_read"
_WIKI_WRITE_TOOL = "mcp__mnemara_memory__wiki_write"
_WIKI_LIST_TOOL = "mcp__mnemara_memory__wiki_list"
_RAG_INDEX_TOOL = "mcp__mnemara_memory__rag_index"
_RAG_QUERY_TOOL = "mcp__mnemara_memory__rag_query"
_GRAPH_ADD_NODE_TOOL = "mcp__mnemara_memory__graph_add_node"
_GRAPH_ADD_EDGE_TOOL = "mcp__mnemara_memory__graph_add_edge"
_GRAPH_QUERY_TOOL = "mcp__mnemara_memory__graph_query"
_GRAPH_NEIGHBORS_TOOL = "mcp__mnemara_memory__graph_neighbors"
_GRAPH_MATCH_TOOL = "mcp__mnemara_memory__graph_match"
_GRAPH_SHORTEST_PATH_TOOL = "mcp__mnemara_memory__graph_shortest_path"
_TUNE_WINDOW_TOOL = "mcp__mnemara_memory__tune_window"
_EVICT_LAST_TOOL = "mcp__mnemara_memory__evict_last"
_EVICT_IDS_TOOL = "mcp__mnemara_memory__evict_ids"
_MARK_SEGMENT_TOOL = "mcp__mnemara_memory__mark_segment"
_EVICT_SINCE_TOOL = "mcp__mnemara_memory__evict_since"
_EVICT_THINKING_BLOCKS_TOOL = "mcp__mnemara_memory__evict_thinking_blocks"
_EVICT_TOOL_USE_BLOCKS_TOOL = "mcp__mnemara_memory__evict_tool_use_blocks"
_EVICT_WRITE_PAIRS_TOOL = "mcp__mnemara_memory__evict_write_pairs"
_EVICT_OLDER_THAN_TOOL = "mcp__mnemara_memory__evict_older_than"
_PIN_ROW_TOOL = "mcp__mnemara_memory__pin_row"
_UNPIN_ROW_TOOL = "mcp__mnemara_memory__unpin_row"
_LIST_PINNED_TOOL = "mcp__mnemara_memory__list_pinned"


# Hard context ceilings per model family — the API's absolute token limit.
# Used by _recover_from_overflow to gauge how aggressively to evict before
# retrying; unlike max_window_tokens these are not user-configurable.
_MODEL_CONTEXT_CEILING: dict[str, int] = {
    "opus": 1_000_000,
    "sonnet": 200_000,
    "haiku": 200_000,
}
_DEFAULT_CONTEXT_CEILING = 200_000


def _model_context_ceiling(model: str) -> int:
    """Return the hard context token ceiling for a given model identifier.

    Matches by substring — "claude-sonnet-4-6" → 200_000,
    "claude-opus-4-7" → 1_000_000.  Returns _DEFAULT_CONTEXT_CEILING for
    unknown models (safe underestimate, won't over-evict).
    """
    m = (model or "").lower()
    for family, ceiling in _MODEL_CONTEXT_CEILING.items():
        if family in m:
            return ceiling
    return _DEFAULT_CONTEXT_CEILING


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
        # Per-session RuntimeSentinel (v0.3.4) — created only when
        # cfg.runtime_sentinel is True; None otherwise.
        self._sentinel: Optional[RuntimeSentinel] = (
            RuntimeSentinel() if getattr(cfg, "runtime_sentinel", False) else None
        )
        # Session-scoped counters (self-instrumentation, v0.2.0).
        self.session_started_at = datetime.now(timezone.utc).isoformat()
        self.session_ended_at: Optional[str] = None
        self.evicted_this_session = 0
        self.tools_called: dict[str, int] = {}
        self.memory_writes = 0
        self.role_proposals = 0
        self.choices_logged = 0
        self.wiki_writes = 0
        self.rag_indexes = 0
        self.rag_queries = 0
        self.graph_nodes_added = 0
        self.graph_edges_added = 0
        self.graph_queries = 0
        self.replay_runs = 0
        self.session_turns = 0
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self._stats_written = False

    # ------------------------------------------------------------------ public

    def turn(
        self,
        user_text: str,
        on_token: OnToken | None = None,
        on_tool_use: OnToolUse | None = None,
        on_tool_result: OnToolResult | None = None,
    ) -> dict[str, Any]:
        """Run one user->assistant turn synchronously.

        Wraps turn_async via asyncio.run for the bare REPL.
        """
        return asyncio.run(
            self.turn_async(
                user_text,
                on_token=on_token,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
            )
        )

    async def turn_async(
        self,
        user_text: str,
        on_token: OnToken | None = None,
        on_tool_use: OnToolUse | None = None,
        on_tool_result: OnToolResult | None = None,
    ) -> dict[str, Any]:
        """Async variant — drives one turn inside an existing event loop.

        Used by the Textual TUI; the REPL uses turn() which wraps this.
        """
        # Persist the user turn before contacting the model.
        user_blocks = [{"type": "text", "text": user_text}]
        self.store.append_turn("user", user_blocks)

        system_prompt = role_mod.load_role_doc(self.cfg.role_doc_path) or (
            "You are a helpful assistant."
        )

        # Build prompt: window transcript prefix + current user input.
        prompt = _build_prompt(self.store, user_text)

        options = self._build_options(system_prompt)

        try:
            result = await _run_turn(
                prompt,
                options,
                self.cfg.stream,
                on_token=on_token,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
                sentinel=self._sentinel,
            )
        except RuntimeError as _overflow_exc:
            if "Prompt is too long" not in str(_overflow_exc):
                raise
            result = await self._recover_from_overflow(
                user_text=user_text,
                options=options,
                on_token=on_token,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
            )

        assistant_blocks = result["assistant_blocks"]
        total_in = result["tokens_in"]
        total_out = result["tokens_out"]

        # Persist final assistant turn (full block list) and evict.
        assistant_row_id = self.store.append_turn(
            "assistant",
            assistant_blocks or [{"type": "text", "text": ""}],
            tokens_in=total_in,
            tokens_out=total_out,
        )
        # Auto-evict-after-write: if the toggle is on AND the just-persisted
        # turn contains any Edit/Write/MultiEdit/NotebookEdit tool_use, stub
        # their bulky body content (and matching prior Read specs for the
        # same files). The block structure stays — audit trail intact, just
        # the kilobytes-per-block payloads collapse to {file_path, _evicted:
        # true}. Pinned rows skipped. Off by default; opt-in per panel.
        if getattr(self.cfg, "auto_evict_after_write", False):
            try:
                pair_result = self.store.evict_write_pairs(
                    only_in_rows=[assistant_row_id],
                    skip_pinned=True,
                )
                if pair_result.get("rows_modified", 0) > 0:
                    log(
                        "auto_evict_pairs",
                        writes=pair_result["writes_stubbed"],
                        reads=pair_result["reads_stubbed"],
                        rows=pair_result["rows_modified"],
                        bytes_freed=pair_result["bytes_freed"],
                        files=pair_result["files_seen"],
                    )
            except Exception as exc:
                # Eviction failures must never crash a turn; audit-trail
                # eviction is opportunistic.
                log("auto_evict_pairs_error", error=str(exc))
        # Compress repeated Read results after every turn when the flag is on.
        # Outside the auto_evict_after_write guard so it fires on read-heavy
        # turns too, not just write turns. compress_repeated_reads is
        # idempotent — already-stubbed rows are skipped in O(1).
        if getattr(self.cfg, "compress_repeated_reads", False):
            try:
                cr_result = self.store.compress_repeated_reads(
                    skip_pinned=True,
                    preserve_compressed_reads=getattr(
                        self.cfg, "preserve_compressed_reads", False
                    ),
                )
                if cr_result.get("reads_compressed", 0) > 0:
                    log(
                        "auto_compress_reads",
                        reads_compressed=cr_result["reads_compressed"],
                        bytes_freed=cr_result["bytes_freed"],
                    )
            except Exception as exc:
                log("auto_compress_reads_error", error=str(exc))
        # Pass row_cap_slack so the row cap can "breathe" with the byte
        # budget after heavy block surgery. The slack is configured per
        # panel via cfg.row_cap_slack_when_token_headroom (default 0 =
        # strict row cap). Slack only engages when current tokens are
        # under HEADROOM_RATIO * max_window_tokens; the token cap remains
        # the hard ceiling regardless.
        evicted = self.store.evict(
            self.cfg.max_window_turns,
            self.cfg.max_window_tokens,
            row_cap_slack=getattr(self.cfg, "row_cap_slack_when_token_headroom", 0),
            preserve_compressed_reads=getattr(self.cfg, "preserve_compressed_reads", False),
        )
        if evicted:
            log("eviction", deleted=evicted)

        # Update session-scoped counters.
        self.evicted_this_session += int(evicted or 0)
        self.session_turns += 1
        self.session_tokens_in += int(total_in or 0)
        self.session_tokens_out += int(total_out or 0)
        for blk in assistant_blocks or []:
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                name = blk.get("name") or "?"
                self.tools_called[name] = self.tools_called.get(name, 0) + 1

        return {"input_tokens": total_in, "output_tokens": total_out, "evicted": evicted}

    async def _recover_from_overflow(
        self,
        *,
        user_text: str,
        options: "ClaudeAgentOptions",
        on_token: "OnToken | None",
        on_tool_use: "OnToolUse | None",
        on_tool_result: "OnToolResult | None",
    ) -> dict[str, Any]:
        """Self-healing recovery when the API rejects the prompt as too long.

        Sequence:
          1. Log overflow_recovery_lift with model name and hard ceiling.
          2. evict_write_pairs(skip_pinned=True) — highest bytes freed per
             call, audit trail (file_path) preserved.
          3. If stored tokens are still above the *configured* cap, also run
             evict_tool_use_blocks(all_rows=True, skip_pinned=True).
          4. Rebuild the prompt from the trimmed store and retry once.
          5. If the retry still overflows, log overflow_recovery_failed and
             raise RuntimeError with the ceiling context for display.
        """
        model = getattr(self.cfg, "model", "") or ""
        ceiling = _model_context_ceiling(model)
        original_cap = self.cfg.max_window_tokens
        log("overflow_recovery_lift",
            model=model,
            original_cap=original_cap,
            ceiling=ceiling)

        # Step 1: stub Edit/Write body content — max bytes freed, min info lost.
        pair_result = self.store.evict_write_pairs(skip_pinned=True)
        log("overflow_recovery_evict_write_pairs",
            bytes_freed=pair_result.get("bytes_freed", 0),
            writes_stubbed=pair_result.get("writes_stubbed", 0),
            reads_stubbed=pair_result.get("reads_stubbed", 0))

        # Step 2: if still over the configured cap, strip full tool_use blocks.
        current_tokens, _ = self.store.total_tokens()
        if current_tokens > original_cap:
            tub_result = self.store.evict_tool_use_blocks(
                all_rows=True, skip_pinned=True
            )
            log("overflow_recovery_evict",
                rows_modified=tub_result.get("rows_modified", 0),
                bytes_freed=tub_result.get("bytes_freed", 0),
                blocks_stripped=tub_result.get("blocks_stripped", 0))

        # Step 3: rebuild the prompt from the now-trimmed store and retry once.
        prompt_retry = _build_prompt(self.store, user_text)
        log("overflow_recovery_retry", model=model, ceiling=ceiling)

        try:
            return await _run_turn(
                prompt_retry,
                options,
                self.cfg.stream,
                on_token=on_token,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
                sentinel=self._sentinel,
            )
        except RuntimeError as retry_exc:
            if "Prompt is too long" in str(retry_exc):
                tokens_now, _ = self.store.total_tokens()
                log("overflow_recovery_failed",
                    model=model,
                    ceiling=ceiling,
                    tokens_after_evict=tokens_now)
                raise RuntimeError(
                    f"Prompt is too long even after aggressive eviction "
                    f"({model} hard ceiling: {ceiling:,} tokens) — "
                    "use /evict N to free context or /clear to reset the window"
                ) from None
            raise

    def write_session_stats(self) -> Optional[Path]:
        """Dump session counters to ~/.mnemara/<instance>/stats/YYYY-MM-DD.json.

        Idempotent within a process — only writes once per AgentSession.
        On error, logs and returns None; never raises (caller may invoke from
        shutdown paths).
        """
        if self._stats_written:
            return None
        try:
            instance = self.runner.instance
            d = paths_mod.stats_dir(instance)
            d.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            f = d / f"{today}.json"
            ended_at = datetime.now(timezone.utc).isoformat()
            self.session_ended_at = ended_at
            session_summary = {
                "started_at": self.session_started_at,
                "ended_at": ended_at,
                "turns": self.session_turns,
                "evicted": self.evicted_this_session,
                "tokens_in": self.session_tokens_in,
                "tokens_out": self.session_tokens_out,
                "tools_called": dict(self.tools_called),
                "memory_writes": self.memory_writes,
                "role_proposals": self.role_proposals,
                "choices_logged": self.choices_logged,
                "wiki_writes": self.wiki_writes,
                "rag_indexes": self.rag_indexes,
                "rag_queries": self.rag_queries,
            }
            existing: dict[str, Any] = {}
            if f.exists():
                try:
                    existing = json.loads(f.read_text())
                except (ValueError, OSError):
                    existing = {}
            sessions = list(existing.get("sessions", []))
            sessions.append(session_summary)
            cum = existing.get("cumulative", {}) or {}
            cum_tools = dict(cum.get("tools_called", {}) or {})
            for k, v in self.tools_called.items():
                cum_tools[k] = cum_tools.get(k, 0) + int(v)
            cumulative = {
                "turns": int(cum.get("turns", 0) or 0) + self.session_turns,
                "tokens_in": int(cum.get("tokens_in", 0) or 0) + self.session_tokens_in,
                "tokens_out": int(cum.get("tokens_out", 0) or 0) + self.session_tokens_out,
                "evicted": int(cum.get("evicted", 0) or 0) + self.evicted_this_session,
                "memory_writes": int(cum.get("memory_writes", 0) or 0) + self.memory_writes,
                "role_proposals": int(cum.get("role_proposals", 0) or 0) + self.role_proposals,
                "choices_logged": int(cum.get("choices_logged", 0) or 0) + self.choices_logged,
                "wiki_writes": int(cum.get("wiki_writes", 0) or 0) + self.wiki_writes,
                "rag_indexes": int(cum.get("rag_indexes", 0) or 0) + self.rag_indexes,
                "rag_queries": int(cum.get("rag_queries", 0) or 0) + self.rag_queries,
                "tools_called": cum_tools,
            }
            doc = {
                "date": today,
                "instance": instance,
                "sessions": sessions,
                "cumulative": cumulative,
            }
            f.write_text(json.dumps(doc, indent=2) + "\n")
            self._stats_written = True
            return f
        except Exception as e:  # pragma: no cover
            log("session_stats_error", error=str(e))
            return None

    # ------------------------------------------------------------------ internal

    def _build_options(self, system_prompt: str) -> "ClaudeAgentOptions":
        session = self

        # In-process WriteMemory tool — supports legacy text+category and
        # structured payload mode (JSON-encoded `payload` field).
        @tool(
            "write_memory",
            "Append a timestamped note to the instance's memory file. "
            "Pass either text+category, OR a JSON-encoded `payload` with "
            "{observation, evidence, prediction, applies_to, confidence} "
            "for a structured stanza.",
            {"text": str, "category": str, "payload": str},
        )
        async def _write_memory_tool(args: dict[str, Any]) -> dict[str, Any]:
            payload_raw = args.get("payload")
            payload_dict: Optional[dict[str, Any]] = None
            if payload_raw:
                try:
                    parsed = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
                    if isinstance(parsed, dict):
                        payload_dict = parsed
                except (ValueError, TypeError):
                    payload_dict = None
            tools_mod.write_memory(
                session.runner.instance,
                args.get("text", "") or "",
                args.get("category", "note") or "note",
                payload=payload_dict,
                cfg=session.cfg,
            )
            session.memory_writes += 1
            cat = args.get("category", "") or ""
            if cat.startswith("wiki/"):
                session.wiki_writes += 1
            if getattr(session.cfg, "rag_auto_index_memory", True) and getattr(
                session.cfg, "rag_enabled", True
            ):
                session.rag_indexes += 1
            return {"content": [{"type": "text", "text": "Memory note appended."}]}

        @tool(
            "inspect_context",
            "Report Mnemara's view of the current session: turn count, token "
            "totals, eviction count, role doc info, configured MCP servers, "
            "and tool permission summary. Returns a JSON dict.",
            {},
        )
        async def _inspect_context_tool(_args: dict[str, Any]) -> dict[str, Any]:
            cfg = session.cfg
            tin, tout = session.store.total_tokens()
            n_turns = len(session.store.window())
            total = tin + tout
            role_size = 0
            try:
                if cfg.role_doc_path:
                    p = Path(cfg.role_doc_path).expanduser()
                    if p.exists():
                        role_size = p.stat().st_size
            except Exception:
                role_size = 0
            info = {
                "instance": session.runner.instance,
                "model": cfg.model,
                "role_doc_path": cfg.role_doc_path,
                "max_window_turns": cfg.max_window_turns,
                "max_window_tokens": cfg.max_window_tokens,
                "current_turn_count": n_turns,
                "total_input_tokens": tin,
                "total_output_tokens": tout,
                "total_tokens": total,
                "tokens_remaining": max(0, cfg.max_window_tokens - total),
                "evicted_this_session": session.evicted_this_session,
                "role_doc_size_bytes": role_size,
                "mcp_servers": ["mnemara_memory"] + [s.name for s in cfg.mcp_servers],
                "allowed_tools_summary": [
                    {"tool": t.tool, "mode": t.mode} for t in cfg.allowed_tools
                ],
            }
            return {
                "content": [
                    {"type": "text", "text": json.dumps(info, indent=2)}
                ]
            }

        @tool(
            "propose_role_amendment",
            "Append a role-amendment proposal as a markdown file under the "
            "instance's role_proposals/ directory. severity is one of "
            "minor|moderate|major.",
            {"text": str, "rationale": str, "severity": str},
        )
        async def _propose_role_amendment_tool(args: dict[str, Any]) -> dict[str, Any]:
            severity = args.get("severity", "minor") or "minor"
            if severity not in ("minor", "moderate", "major"):
                severity = "minor"
            p = tools_mod.propose_role_amendment(
                session.runner.instance,
                args.get("text", "") or "",
                args.get("rationale", "") or "",
                severity,
            )
            session.role_proposals += 1
            return {
                "content": [
                    {"type": "text", "text": f"Role amendment proposal written to {p}"}
                ]
            }

        @tool(
            "log_choice",
            "Append a JSONL line to choices.jsonl recording a decision the "
            "agent made — decision_type, decision, rationale, context_summary "
            "(pass empty string '' if no situational context to capture). "
            "Used for self-observation. All four fields are required by the "
            "tool schema; use empty strings for any you don't have.",
            {
                "decision_type": str,
                "decision": str,
                "rationale": str,
                "context_summary": str,
            },
        )
        async def _log_choice_tool(args: dict[str, Any]) -> dict[str, Any]:
            rows = session.store.window()
            turn_id = rows[-1]["id"] if rows else None
            tin, tout = session.store.total_tokens()
            tools_mod.log_choice(
                session.runner.instance,
                args.get("decision_type", "") or "",
                args.get("decision", "") or "",
                args.get("rationale", "") or "",
                args.get("context_summary", "") or "",
                turn_id=turn_id,
                tokens_at_choice=tin + tout,
            )
            session.choices_logged += 1
            return {"content": [{"type": "text", "text": "Choice logged."}]}

        @tool(
            "wiki_read",
            "Read a wiki page by slash-allowed slug (e.g. 'replay_policy' or "
            "'patterns/loader_traps'). Returns the page contents, or 'no such "
            "page' if missing.",
            {"path": str},
        )
        async def _wiki_read_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import wiki as wiki_mod
            path = args.get("path", "") or ""
            content = wiki_mod.read_page(session.runner.instance, path)
            if content is None:
                return {"content": [{"type": "text", "text": "no such page"}]}
            return {"content": [{"type": "text", "text": content}]}

        @tool(
            "wiki_write",
            "Write a wiki page. mode is 'replace' (default) or 'append'. "
            "Plain markdown body; optional frontmatter is your responsibility. "
            "Auto-indexed into RAG when rag_auto_index_wiki is enabled.",
            {"path": str, "content": str, "mode": str},
        )
        async def _wiki_write_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import wiki as wiki_mod
            path = args.get("path", "") or ""
            content = args.get("content", "") or ""
            mode = args.get("mode", "replace") or "replace"
            if mode not in ("replace", "append"):
                mode = "replace"
            try:
                f = wiki_mod.write_page(session.runner.instance, path, content, mode=mode)
            except ValueError as e:
                return {
                    "content": [{"type": "text", "text": f"wiki_write error: {e}"}],
                    "is_error": True,
                }
            session.wiki_writes += 1
            if getattr(session.cfg, "rag_auto_index_wiki", True) and getattr(
                session.cfg, "rag_enabled", True
            ):
                try:
                    from . import rag as rag_mod
                    rag_mod.store_for(session.runner.instance, session.cfg).index(
                        content, kind="wiki", source_path=str(f), category=path,
                    )
                    session.rag_indexes += 1
                except Exception as e:
                    log("wiki_rag_index_error", error=str(e))
            if getattr(session.cfg, "graph_enabled", True):
                try:
                    from . import graph as graph_mod
                    graph_mod.auto_edges_from_wiki(
                        session.runner.instance, session.cfg, path, content
                    )
                except Exception as e:
                    log("wiki_graph_edge_error", error=str(e))
            return {"content": [{"type": "text", "text": f"Wrote wiki page: {f}"}]}

        @tool(
            "wiki_list",
            "List wiki pages under an optional prefix. Returns a JSON list of "
            "{path, size_bytes, last_modified}.",
            {"prefix": str},
        )
        async def _wiki_list_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import wiki as wiki_mod
            prefix = args.get("prefix", "") or ""
            entries = wiki_mod.list_pages(session.runner.instance, prefix)
            return {
                "content": [{"type": "text", "text": json.dumps(entries, indent=2)}]
            }

        @tool(
            "rag_index",
            "Embed and index arbitrary text into the RAG store. kind is one of "
            "'manual' (default), 'memory', 'wiki'. Returns the new row id, or "
            "an unavailable error if the embedding backend is down.",
            {"text": str, "kind": str, "source_path": str, "category": str},
        )
        async def _rag_index_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import rag as rag_mod
            text = args.get("text", "") or ""
            kind = args.get("kind", "manual") or "manual"
            source_path = args.get("source_path", "") or ""
            category = args.get("category", "") or ""
            res = rag_mod.store_for(session.runner.instance, session.cfg).index(
                text, kind=kind, source_path=source_path, category=category
            )
            if res.get("ok"):
                session.rag_indexes += 1
                return {"content": [{"type": "text", "text": f"Indexed: {res['id']}"}]}
            return {
                "content": [{"type": "text", "text": res.get("error", "RAG unavailable")}],
                "is_error": True,
            }

        @tool(
            "rag_query",
            "Top-k semantic search over the RAG store. kind is an optional "
            "filter ('memory'|'wiki'|'manual'). Returns JSON results with "
            "distance scores, or an unavailable error if the embedding "
            "backend is down.",
            {"question": str, "k": int, "kind": str},
        )
        async def _rag_query_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import rag as rag_mod
            question = args.get("question", "") or ""
            k = int(args.get("k") or 5)
            kind = args.get("kind") or None
            res = rag_mod.store_for(session.runner.instance, session.cfg).query(
                question, k=k, kind=kind
            )
            if res.get("ok"):
                session.rag_queries += 1
                return {
                    "content": [
                        {"type": "text", "text": json.dumps(res["results"], indent=2)}
                    ]
                }
            return {
                "content": [{"type": "text", "text": res.get("error", "RAG unavailable")}],
                "is_error": True,
            }

        @tool(
            "graph_add_node",
            "Add a node to the property graph. label is a free string "
            "('entity', 'wiki_page', etc.). properties_json is a JSON-encoded "
            "object of arbitrary attributes. Returns the new node id, or an "
            "error if the graph backend is unavailable.",
            {"label": str, "properties_json": str},
        )
        async def _graph_add_node_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import graph as graph_mod
            label = args.get("label", "") or ""
            raw = args.get("properties_json", "") or ""
            try:
                props = json.loads(raw) if raw else {}
                if not isinstance(props, dict):
                    props = {}
            except (ValueError, TypeError):
                props = {}
            res = graph_mod.store_for(session.runner.instance, session.cfg).add_node(
                label, props
            )
            if res.get("ok"):
                session.graph_nodes_added += 1
                return {"content": [{"type": "text", "text": f"Added node: {res['id']}"}]}
            return {
                "content": [{"type": "text", "text": res.get("error", "graph unavailable")}],
                "is_error": True,
            }

        @tool(
            "graph_add_edge",
            "Add a directed edge between two existing nodes. relationship is a "
            "free string. properties_json is a JSON-encoded object. Returns "
            "the new edge id, or an error.",
            {"from_id": str, "to_id": str, "relationship": str, "properties_json": str},
        )
        async def _graph_add_edge_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import graph as graph_mod
            raw = args.get("properties_json", "") or ""
            try:
                props = json.loads(raw) if raw else {}
                if not isinstance(props, dict):
                    props = {}
            except (ValueError, TypeError):
                props = {}
            res = graph_mod.store_for(session.runner.instance, session.cfg).add_edge(
                args.get("from_id", "") or "",
                args.get("to_id", "") or "",
                args.get("relationship", "") or "",
                props,
            )
            if res.get("ok"):
                session.graph_edges_added += 1
                return {"content": [{"type": "text", "text": f"Added edge: {res['id']}"}]}
            return {
                "content": [{"type": "text", "text": res.get("error", "graph unavailable")}],
                "is_error": True,
            }

        @tool(
            "graph_query",
            "Run an arbitrary Cypher query against the graph. Returns rows as "
            "a JSON list of dicts. Errors return an unavailable message.",
            {"cypher": str},
        )
        async def _graph_query_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import graph as graph_mod
            res = graph_mod.store_for(session.runner.instance, session.cfg).query(
                args.get("cypher", "") or ""
            )
            if res.get("ok"):
                session.graph_queries += 1
                return {
                    "content": [
                        {"type": "text", "text": json.dumps(res.get("rows", []), indent=2, default=str)}
                    ]
                }
            return {
                "content": [{"type": "text", "text": res.get("error", "graph unavailable")}],
                "is_error": True,
            }

        @tool(
            "graph_neighbors",
            "Return adjacent nodes within a given depth (1..5). Returns a JSON "
            "list of {id, label, properties}.",
            {"node_id": str, "depth": int},
        )
        async def _graph_neighbors_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import graph as graph_mod
            res = graph_mod.store_for(session.runner.instance, session.cfg).neighbors(
                args.get("node_id", "") or "", int(args.get("depth") or 1)
            )
            if res.get("ok"):
                session.graph_queries += 1
                return {
                    "content": [
                        {"type": "text", "text": json.dumps(res.get("neighbors", []), indent=2, default=str)}
                    ]
                }
            return {
                "content": [{"type": "text", "text": res.get("error", "graph unavailable")}],
                "is_error": True,
            }

        @tool(
            "graph_match",
            "Convenience match: pattern_json is {label, properties_subset} where "
            "properties_subset is a dict of key/value substring filters. Returns "
            "matching nodes.",
            {"pattern_json": str},
        )
        async def _graph_match_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import graph as graph_mod
            raw = args.get("pattern_json", "") or "{}"
            try:
                pattern = json.loads(raw)
                if not isinstance(pattern, dict):
                    pattern = {}
            except (ValueError, TypeError):
                pattern = {}
            res = graph_mod.store_for(session.runner.instance, session.cfg).match(pattern)
            if res.get("ok"):
                session.graph_queries += 1
                return {
                    "content": [
                        {"type": "text", "text": json.dumps(res.get("matches", []), indent=2, default=str)}
                    ]
                }
            return {
                "content": [{"type": "text", "text": res.get("error", "graph unavailable")}],
                "is_error": True,
            }

        @tool(
            "graph_shortest_path",
            "Find shortest path between two nodes (undirected, max depth 6). "
            "Returns the list of node ids on the path, or an empty list.",
            {"from_id": str, "to_id": str},
        )
        async def _graph_shortest_path_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import graph as graph_mod
            res = graph_mod.store_for(
                session.runner.instance, session.cfg
            ).shortest_path(
                args.get("from_id", "") or "", args.get("to_id", "") or ""
            )
            if res.get("ok"):
                session.graph_queries += 1
                return {
                    "content": [{"type": "text", "text": json.dumps(res.get("path", []))}]
                }
            return {
                "content": [{"type": "text", "text": res.get("error", "graph unavailable")}],
                "is_error": True,
            }

        @tool(
            "tune_window",
            "Adjust this instance's rolling-context caps. Pass max_turns "
            "and/or max_tokens (use -1 to leave a field unchanged). "
            "Persists to config.json unless persist='false'. Bounds: "
            "turns in [1, 10000]; tokens in [1000, 10000000]. Returns "
            "the new effective values. Use this when you've judged the "
            "current cap is wrong for the work at hand — e.g. a deep "
            "investigation needs more tokens, or a fast iteration loop "
            "wants tighter turns.",
            {"max_turns": int, "max_tokens": int, "persist": str},
        )
        async def _tune_window_tool(args: dict[str, Any]) -> dict[str, Any]:
            from . import config as config_mod

            cfg = session.cfg
            mt = args.get("max_turns", -1)
            mtk = args.get("max_tokens", -1)
            persist_raw = (args.get("persist", "true") or "true").lower()
            persist = persist_raw not in ("false", "0", "no", "temp", "--temp")
            try:
                mt_i = int(mt) if mt is not None else -1
            except (TypeError, ValueError):
                mt_i = -1
            try:
                mtk_i = int(mtk) if mtk is not None else -1
            except (TypeError, ValueError):
                mtk_i = -1
            changes: list[str] = []
            errors: list[str] = []
            if mt_i != -1:
                if 1 <= mt_i <= 10000:
                    old = cfg.max_window_turns
                    cfg.max_window_turns = mt_i
                    changes.append(f"max_window_turns: {old} -> {mt_i}")
                else:
                    errors.append(
                        f"max_turns {mt_i} out of bounds [1, 10000]"
                    )
            if mtk_i != -1:
                if 1000 <= mtk_i <= 10_000_000:
                    old = cfg.max_window_tokens
                    cfg.max_window_tokens = mtk_i
                    changes.append(f"max_window_tokens: {old} -> {mtk_i}")
                else:
                    errors.append(
                        f"max_tokens {mtk_i} out of bounds [1000, 10000000]"
                    )
            if not changes and not errors:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "no-op: pass max_turns or max_tokens "
                                "(use -1 to leave unchanged)"
                            ),
                        }
                    ],
                    "is_error": True,
                }
            persist_note = ""
            if changes and persist:
                try:
                    config_mod.save(session.runner.instance, cfg)
                    persist_note = " (persisted to config.json)"
                except Exception as exc:
                    persist_note = f" (persist failed: {exc})"
            elif changes:
                persist_note = " (in-memory only)"
            payload = {
                "changes": changes,
                "errors": errors,
                "effective": {
                    "max_window_turns": cfg.max_window_turns,
                    "max_window_tokens": cfg.max_window_tokens,
                },
                "persisted": persist and not errors and bool(changes),
            }
            text = json.dumps(payload, indent=2) + persist_note
            return {
                "content": [{"type": "text", "text": text}],
                "is_error": bool(errors) and not changes,
            }

        @tool(
            "evict_last",
            "Drop the N most-recent rows from your rolling window. Use this "
            "to actively forget the immediate past — e.g. you read a file, "
            "decided it wasn't what you needed, want it out of your context "
            "so it doesn't compete for attention. NOTE: the user turn that "
            "triggered the current turn IS already in the store, so "
            "evict_last(1) called from a tool drops THAT turn (the prompt "
            "you're answering). For self-eviction prefer mark_segment + "
            "evict_since which is exact rather than positional.",
            {"n": int},
        )
        async def _evict_last_tool(args: dict[str, Any]) -> dict[str, Any]:
            try:
                n = int(args.get("n", 0))
            except (TypeError, ValueError):
                return {
                    "content": [{"type": "text", "text": "n must be a positive integer"}],
                    "is_error": True,
                }
            if n <= 0:
                return {
                    "content": [{"type": "text", "text": "n must be > 0"}],
                    "is_error": True,
                }
            deleted = session.store.evict_last(n)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"deleted": deleted, "requested": n}),
                    }
                ]
            }

        @tool(
            "evict_ids",
            "Drop specific rows from your rolling window by id. Pass `ids` "
            "as a comma-separated string (e.g. '4,7,9') or a JSON array "
            "string ('[4,7,9]'). Use inspect_context to surface ids first, "
            "or coordinate with the producer to learn which rows hold what. "
            "Silently ignores ids that don't exist; the response reports "
            "how many actually deleted.",
            {"ids": str},
        )
        async def _evict_ids_tool(args: dict[str, Any]) -> dict[str, Any]:
            raw = (args.get("ids") or "").strip()
            if not raw:
                return {
                    "content": [{"type": "text", "text": "ids required"}],
                    "is_error": True,
                }
            ids: list[int] = []
            try:
                if raw.startswith("["):
                    parsed = json.loads(raw)
                    ids = [int(x) for x in parsed]
                else:
                    ids = [int(x.strip()) for x in raw.replace(",", " ").split() if x.strip()]
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return {
                    "content": [{"type": "text", "text": f"could not parse ids: {exc}"}],
                    "is_error": True,
                }
            if not ids:
                return {
                    "content": [{"type": "text", "text": "no ids provided"}],
                    "is_error": True,
                }
            deleted = session.store.evict_ids(ids)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "deleted": deleted,
                            "requested": len(ids),
                            "missing": len(ids) - deleted,
                        }),
                    }
                ]
            }

        @tool(
            "mark_segment",
            "Insert a named segment marker at the current tail of your "
            "rolling window. Markers don't show up in the API messages "
            "the model sees — they're invisible bookmarks. Pair with "
            "evict_since to drop everything appended after the marker. "
            "Use this BEFORE you start an exploratory tangent so you can "
            "cleanly back out if it doesn't pan out.",
            {"name": str},
        )
        async def _mark_segment_tool(args: dict[str, Any]) -> dict[str, Any]:
            name = (args.get("name") or "").strip()
            if not name:
                return {
                    "content": [{"type": "text", "text": "name required"}],
                    "is_error": True,
                }
            try:
                mid = session.store.mark_segment(name)
            except Exception as exc:
                return {
                    "content": [{"type": "text", "text": f"mark insert failed: {exc}"}],
                    "is_error": True,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"marker_id": mid, "name": name}),
                    }
                ]
            }

        @tool(
            "evict_since",
            "Drop the named segment marker AND every row appended after "
            "it. If multiple markers share the name, the most recent one "
            "is used. Returns count deleted; 0 means no marker matched. "
            "This is the clean way to abandon an exploratory tangent: "
            "mark_segment('checkpoint') before, evict_since('checkpoint') "
            "after if you decide the tangent wasn't useful.",
            {"marker": str},
        )
        async def _evict_since_tool(args: dict[str, Any]) -> dict[str, Any]:
            name = (args.get("marker") or "").strip()
            if not name:
                return {
                    "content": [{"type": "text", "text": "marker name required"}],
                    "is_error": True,
                }
            deleted = session.store.evict_since(name)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "deleted": deleted,
                            "marker": name,
                            "matched": deleted > 0,
                        }),
                    }
                ]
            }

        @tool(
            "evict_thinking_blocks",
            "Strip 'thinking' blocks from rolling-window rows while preserving "
            "text/tool_use/tool_result blocks. Block-level surgery for context "
            "budget — thinking is the model's reasoning scratch which it doesn't "
            "reference back across turns; text and tool_use carry the durable "
            "outcome of the same turn. Selection (exactly one required): "
            "`ids='4,7,9'` (explicit list, comma- or JSON-array-encoded); "
            "`keep_recent='N'` (strip from every row EXCEPT the most-recent N — "
            "preserves recent reasoning chains the model may still want to "
            "reference; '0' is legal and equivalent to all_rows); "
            "`all_rows='true'` (strip every row in the store); "
            "`older_than='10m'` (strip from rows whose ts is older than the "
            "duration — accepts 'Ns', 'Nm', 'Nh', 'Nd' suffixes or bare integer "
            "seconds). Rows whose stripping would leave 0 blocks are skipped "
            "without modification. Pinned rows are preserved by default; pass "
            "`skip_pinned='false'` to override. If you want to remember a "
            "thinking chain before evicting it, call write_memory("
            "category='thought_summary', text=...) first — the primitive "
            "itself doesn't auto-summarize. Returns {rows_scanned, "
            "rows_modified, blocks_evicted, bytes_freed, rows_skipped_pinned}.",
            {
                "ids": str,
                "keep_recent": str,
                "all_rows": str,
                "older_than": str,
                "skip_pinned": str,
            },
        )
        async def _evict_thinking_blocks_tool(args: dict[str, Any]) -> dict[str, Any]:
            raw_ids = (args.get("ids") or "").strip()
            raw_keep = (args.get("keep_recent") or "").strip()
            raw_all = (args.get("all_rows") or "").strip().lower()
            raw_older = (args.get("older_than") or "").strip()
            raw_skip = (args.get("skip_pinned") or "true").strip().lower()

            have_ids = bool(raw_ids)
            have_keep = bool(raw_keep)
            have_all = raw_all in ("true", "1", "yes")
            have_older = bool(raw_older)

            if sum([have_ids, have_keep, have_all, have_older]) != 1:
                return {
                    "content": [{"type": "text", "text":
                        "exactly one of ids, keep_recent, all_rows, older_than required"}],
                    "is_error": True,
                }

            kw: dict[str, Any] = {
                "skip_pinned": raw_skip not in ("false", "0", "no", "off"),
            }
            try:
                if have_ids:
                    if raw_ids.startswith("["):
                        parsed = json.loads(raw_ids)
                        ids = [int(x) for x in parsed]
                    else:
                        ids = [int(x.strip()) for x in raw_ids.replace(",", " ").split() if x.strip()]
                    if not ids:
                        return {
                            "content": [{"type": "text", "text": "no ids provided"}],
                            "is_error": True,
                        }
                    kw["ids"] = ids
                elif have_keep:
                    kw["keep_recent"] = int(raw_keep)
                elif have_older:
                    kw["older_than_seconds"] = store_mod.parse_duration_seconds(raw_older)
                else:
                    kw["all_rows"] = True
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return {
                    "content": [{"type": "text", "text": f"parse error: {exc}"}],
                    "is_error": True,
                }

            try:
                result = session.store.evict_thinking_blocks(**kw)
            except (ValueError, TypeError) as exc:
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }

            return {
                "content": [{"type": "text", "text": json.dumps(result)}]
            }

        @tool(
            "evict_tool_use_blocks",
            "Strip 'tool_use' blocks from rolling-window rows while preserving "
            "text/thinking blocks. HIGHEST-IMPACT block surgery for context "
            "budget — tool_use spec blocks (file paths, command strings, "
            "payload JSONs, edit before/after content) average ~870 bytes each "
            "and dominate stored bytes in long sessions (often 70%+ of the "
            "store, vs. <1% for thinking signature stubs). Pairing-safe: in "
            "mnemara only assistant_blocks are persisted — tool_result blocks "
            "from the SDK come back via callback only and are never stored, "
            "so all tool_use blocks in our store are already orphaned by "
            "design and stripping them from historical rows is safe (the API "
            "tolerates orphaned tool_use in non-final assistant messages). "
            "Selection (exactly one required): "
            "`ids='4,7,9'` (explicit list, comma- or JSON-array-encoded); "
            "`keep_recent='N'` (strip from every row EXCEPT the most-recent N "
            "— preserves recent tool calls the agent may want to reference; "
            "'0' is legal and equivalent to all_rows); `all_rows='true'` "
            "(strip every row); `older_than='10m'` (strip from rows whose ts "
            "is older than the duration). Pinned rows preserved by default; "
            "pass `skip_pinned='false'` to override. AUDIT TRAIL CAVEAT: "
            "stripping a tool_use block removes the model's record of what "
            "it called. The actual EFFECT (commit, file change, etc.) lives "
            "in git or wherever the tool wrote; only the CALL is gone from "
            "the rolling window. For sessions where the agent needs to "
            "remember 'did I already call X?', prefer keep_recent or pinning "
            "the relevant rows. Consider write_memory(category='tool_audit') "
            "with a one-line summary of significant calls before evicting, "
            "the same opt-in breadcrumb pattern as thought_summary for "
            "thinking surgery. Returns {rows_scanned, rows_modified, "
            "blocks_evicted, bytes_freed, rows_skipped_pinned}.",
            {
                "ids": str,
                "keep_recent": str,
                "all_rows": str,
                "older_than": str,
                "skip_pinned": str,
            },
        )
        async def _evict_tool_use_blocks_tool(args: dict[str, Any]) -> dict[str, Any]:
            raw_ids = (args.get("ids") or "").strip()
            raw_keep = (args.get("keep_recent") or "").strip()
            raw_all = (args.get("all_rows") or "").strip().lower()
            raw_older = (args.get("older_than") or "").strip()
            raw_skip = (args.get("skip_pinned") or "true").strip().lower()

            have_ids = bool(raw_ids)
            have_keep = bool(raw_keep)
            have_all = raw_all in ("true", "1", "yes")
            have_older = bool(raw_older)

            if sum([have_ids, have_keep, have_all, have_older]) != 1:
                return {
                    "content": [{"type": "text", "text":
                        "exactly one of ids, keep_recent, all_rows, older_than required"}],
                    "is_error": True,
                }

            kw: dict[str, Any] = {
                "skip_pinned": raw_skip not in ("false", "0", "no", "off"),
            }
            try:
                if have_ids:
                    if raw_ids.startswith("["):
                        parsed = json.loads(raw_ids)
                        ids = [int(x) for x in parsed]
                    else:
                        ids = [int(x.strip()) for x in raw_ids.replace(",", " ").split() if x.strip()]
                    if not ids:
                        return {
                            "content": [{"type": "text", "text": "no ids provided"}],
                            "is_error": True,
                        }
                    kw["ids"] = ids
                elif have_keep:
                    kw["keep_recent"] = int(raw_keep)
                elif have_older:
                    kw["older_than_seconds"] = store_mod.parse_duration_seconds(raw_older)
                else:
                    kw["all_rows"] = True
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return {
                    "content": [{"type": "text", "text": f"parse error: {exc}"}],
                    "is_error": True,
                }

            try:
                result = session.store.evict_tool_use_blocks(**kw)
            except (ValueError, TypeError) as exc:
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }

            return {
                "content": [{"type": "text", "text": json.dumps(result)}]
            }

        @tool(
            "evict_write_pairs",
            "Stub bulky body content from Edit/Write/MultiEdit/NotebookEdit "
            "tool_use blocks (and matching prior Read tool_use blocks for "
            "the same file_path) in the rolling window. UNLIKE block-type "
            "surgery (evict_thinking_blocks, evict_tool_use_blocks), this "
            "PRESERVES the block — the model still sees 'I called Edit on "
            "/foo/bar.py' — but collapses the input dict to "
            "{file_path, _evicted: true}, dropping the kilobytes-per-block "
            "old_string/new_string/content/edits payloads. The actual "
            "change persists in git or wherever the tool wrote; only the "
            "in-context audit body goes. "
            "When cfg.auto_evict_after_write=true this runs automatically "
            "after each turn that did writes, scoped to that turn's "
            "assistant row. Calling this tool manually scans ALL rows by "
            "default (or restrict to specific rows via only_in_rows). "
            "Pinned rows are skipped. "
            "Idempotent: already-stubbed inputs keep their _evicted marker. "
            "Returns {writes_stubbed, reads_stubbed, rows_modified, "
            "bytes_freed, files_seen, rows_skipped_pinned}.",
            {
                "only_in_rows": str,
                "skip_pinned": str,
            },
        )
        async def _evict_write_pairs_tool(args: dict[str, Any]) -> dict[str, Any]:
            raw_only = (args.get("only_in_rows") or "").strip()
            raw_skip = (args.get("skip_pinned") or "true").strip().lower()
            kw: dict[str, Any] = {
                "skip_pinned": raw_skip not in ("false", "0", "no", "off"),
            }
            if raw_only:
                try:
                    if raw_only.startswith("["):
                        parsed = json.loads(raw_only)
                        only_ids = [int(x) for x in parsed]
                    else:
                        only_ids = [
                            int(x.strip())
                            for x in raw_only.replace(",", " ").split()
                            if x.strip()
                        ]
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    return {
                        "content": [{"type": "text", "text": f"parse error: {exc}"}],
                        "is_error": True,
                    }
                kw["only_in_rows"] = only_ids
            try:
                result = session.store.evict_write_pairs(**kw)
            except (ValueError, TypeError) as exc:
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }
            return {
                "content": [{"type": "text", "text": json.dumps(result)}]
            }

        @tool(
            "pin_row",
            "Pin a rolling-window row with a free-form category label. Pinned "
            "rows are preserved against PROACTIVE eviction (evict_older_than, "
            "bulk-mode thinking-block surgery, future auto-decay) but explicit "
            "evict_ids still drops them — the pin is advisory, not a lock. "
            "Use this to mark narrative-bearing turns: commits ('label=commit'), "
            "findings ('label=finding'), decisions ('label=decision'), summaries. "
            "Idempotent: re-pinning with a new label overwrites the previous "
            "label. Returns {row_id, label, matched} where matched=False means "
            "the row id didn't exist.",
            {"row_id": str, "label": str},
        )
        async def _pin_row_tool(args: dict[str, Any]) -> dict[str, Any]:
            try:
                row_id = int(str(args.get("row_id", "")).strip())
            except (TypeError, ValueError):
                return {
                    "content": [{"type": "text", "text": "row_id must be an integer"}],
                    "is_error": True,
                }
            label = (args.get("label") or "pinned").strip()
            if not label:
                label = "pinned"
            try:
                matched = session.store.pin_row(row_id, label)
            except (ValueError, TypeError) as exc:
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "row_id": row_id, "label": label, "matched": matched,
                })}]
            }

        @tool(
            "unpin_row",
            "Remove the pin from a rolling-window row. Returns {row_id, matched} "
            "where matched=False means the row didn't exist OR existed but was "
            "not pinned (idempotent unpin is a no-op).",
            {"row_id": str},
        )
        async def _unpin_row_tool(args: dict[str, Any]) -> dict[str, Any]:
            try:
                row_id = int(str(args.get("row_id", "")).strip())
            except (TypeError, ValueError):
                return {
                    "content": [{"type": "text", "text": "row_id must be an integer"}],
                    "is_error": True,
                }
            matched = session.store.unpin_row(row_id)
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "row_id": row_id, "matched": matched,
                })}]
            }

        @tool(
            "list_pinned",
            "List all currently-pinned rows in id-ascending order. Each entry "
            "has {id, ts, role, pin_label, content_preview}. Optional `label` "
            "argument filters to rows with that exact pin_label (e.g. "
            "label='commit' for all commits). Without a label, returns every "
            "pinned row regardless of category. Use to audit what survives "
            "proactive eviction.",
            {"label": str},
        )
        async def _list_pinned_tool(args: dict[str, Any]) -> dict[str, Any]:
            label_filter = (args.get("label") or "").strip() or None
            rows = session.store.list_pinned(label_filter)
            preview_rows = []
            for r in rows:
                content = r.get("content")
                if isinstance(content, list):
                    # Build a short preview from text + tool_use names.
                    bits = []
                    for b in content[:3]:
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "text":
                            txt = (b.get("text") or "")[:60]
                            bits.append(f"text:{txt!r}")
                        elif bt == "tool_use":
                            bits.append(f"tool:{b.get('name', '?')}")
                        elif bt == "tool_result":
                            bits.append("tool_result")
                        elif bt == "thinking":
                            bits.append("thinking")
                    preview = " | ".join(bits) if bits else f"({len(content)} blocks)"
                elif isinstance(content, str):
                    preview = content[:80]
                else:
                    preview = str(content)[:80]
                preview_rows.append({
                    "id": r["id"],
                    "ts": r["ts"],
                    "role": r["role"],
                    "pin_label": r["pin_label"],
                    "preview": preview,
                })
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "count": len(preview_rows),
                    "label_filter": label_filter,
                    "rows": preview_rows,
                })}]
            }

        @tool(
            "evict_older_than",
            "Time-based row-level eviction: drop rows whose ts is older than "
            "the given duration. Pass `duration` as a string with optional "
            "suffix — 'Ns' (seconds), 'Nm' (minutes), 'Nh' (hours), 'Nd' "
            "(days), or bare integer for seconds. Examples: '600' (10 minutes), "
            "'10m', '1h', '1d'. Pinned rows are preserved by default; pass "
            "`skip_pinned='false'` for hard time purges. Returns "
            "{rows_evicted, rows_skipped_pinned, cutoff_ts}. Designed for "
            "autonomous decay — an agent can call evict_older_than('10m') "
            "periodically while pin_row protecting load-bearing turns.",
            {"duration": str, "skip_pinned": str},
        )
        async def _evict_older_than_tool(args: dict[str, Any]) -> dict[str, Any]:
            raw_duration = (args.get("duration") or "").strip()
            if not raw_duration:
                return {
                    "content": [{"type": "text", "text": "duration required"}],
                    "is_error": True,
                }
            try:
                seconds = store_mod.parse_duration_seconds(raw_duration)
            except ValueError as exc:
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }
            raw_skip = (args.get("skip_pinned") or "true").strip().lower()
            skip_pinned = raw_skip not in ("false", "0", "no", "off")
            try:
                result = session.store.evict_older_than(
                    seconds, skip_pinned=skip_pinned
                )
            except (ValueError, TypeError) as exc:
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }
            return {
                "content": [{"type": "text", "text": json.dumps(result)}]
            }

        registered = [
            _write_memory_tool,
            _inspect_context_tool,
            _propose_role_amendment_tool,
            _log_choice_tool,
            _wiki_read_tool,
            _wiki_write_tool,
            _wiki_list_tool,
            _rag_index_tool,
            _rag_query_tool,
            _graph_add_node_tool,
            _graph_add_edge_tool,
            _graph_query_tool,
            _graph_neighbors_tool,
            _graph_match_tool,
            _graph_shortest_path_tool,
            _tune_window_tool,
            _evict_last_tool,
            _evict_ids_tool,
            _mark_segment_tool,
            _evict_since_tool,
            _evict_thinking_blocks_tool,
            _evict_tool_use_blocks_tool,
            _evict_write_pairs_tool,
            _pin_row_tool,
            _unpin_row_tool,
            _list_pinned_tool,
            _evict_older_than_tool,
        ]
        # Expose handlers by name for in-process testing / introspection.
        self._registered_tools = {
            getattr(t, "name", None) or "?": getattr(t, "handler", None)
            for t in registered
        }
        memory_server = create_sdk_mcp_server(
            name="mnemara_memory",
            tools=registered,
        )

        mcp_servers: dict[str, Any] = {"mnemara_memory": memory_server}
        for s in self.cfg.mcp_servers:
            mcp_servers[s.name] = {
                "type": "stdio",
                "command": s.command,
                "args": list(s.args),
                "env": dict(s.env),
            }

        allowed_tools = list(_BUILTIN_TOOLS) + [
            _WRITE_MEMORY_TOOL,
            _INSPECT_CONTEXT_TOOL,
            _PROPOSE_ROLE_AMENDMENT_TOOL,
            _LOG_CHOICE_TOOL,
            _WIKI_READ_TOOL,
            _WIKI_WRITE_TOOL,
            _WIKI_LIST_TOOL,
            _RAG_INDEX_TOOL,
            _RAG_QUERY_TOOL,
            _GRAPH_ADD_NODE_TOOL,
            _GRAPH_ADD_EDGE_TOOL,
            _GRAPH_QUERY_TOOL,
            _GRAPH_NEIGHBORS_TOOL,
            _GRAPH_MATCH_TOOL,
            _GRAPH_SHORTEST_PATH_TOOL,
            _TUNE_WINDOW_TOOL,
            _EVICT_LAST_TOOL,
            _EVICT_IDS_TOOL,
            _MARK_SEGMENT_TOOL,
            _EVICT_SINCE_TOOL,
            _EVICT_THINKING_BLOCKS_TOOL,
            _EVICT_TOOL_USE_BLOCKS_TOOL,
            _EVICT_WRITE_PAIRS_TOOL,
            _EVICT_OLDER_THAN_TOOL,
            _PIN_ROW_TOOL,
            _UNPIN_ROW_TOOL,
            _LIST_PINNED_TOOL,
        ]
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

        # Wire include_hook_events when the runtime sentinel is enabled so
        # that HookEventMessage entries arrive in the query() stream. This
        # requires SDK >= 0.1.74; the field is guarded at the type level by
        # the conditional import above and is only passed when the sentinel
        # is active.
        sentinel_active = self._sentinel is not None and HookEventMessage is not None
        extra_opts: dict[str, Any] = {}
        if sentinel_active:
            extra_opts["include_hook_events"] = True

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=cfg.model,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            can_use_tool=_can_use_tool,
            permission_mode="bypassPermissions",
            include_partial_messages=cfg.stream,
            setting_sources=[],
            stderr=lambda line: log("sdk_stderr", line=line),
            **extra_opts,
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


async def _run_turn(
    prompt: str,
    options: "ClaudeAgentOptions",
    stream: bool,
    on_token: OnToken | None = None,
    on_tool_use: OnToolUse | None = None,
    on_tool_result: OnToolResult | None = None,
    sentinel: "RuntimeSentinel | None" = None,
) -> dict[str, Any]:
    assistant_blocks: list[dict] = []
    tokens_in = 0
    tokens_out = 0
    printed_any = False
    # When any callback is wired the caller (TUI) owns presentation —
    # suppress the default console rendering to avoid double-output.
    callback_mode = any(c is not None for c in (on_token, on_tool_use, on_tool_result))
    # Whether we've already halted this turn due to sentinel firing.
    _sentinel_halted = False

    async def _maybe_await(result):
        if asyncio.iscoroutine(result):
            await result

    async def _prompt_stream():
        # The SDK requires an AsyncIterable prompt when can_use_tool is set
        # (it needs to keep the bidirectional channel open for permission
        # callbacks). Yield a single user message and complete.
        yield {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
            "session_id": "default",
        }

    # Capture the generator so we can explicitly aclose() it in the finally
    # block below.  Without this, Python defers cleanup to gc/__del__ which
    # fires after the event loop has already closed and produces:
    #   RuntimeError: Event loop is closed
    #       in asyncio.base_subprocess.BaseSubprocessTransport.__del__
    # Root cause: the SDK wraps anyio → asyncio.create_subprocess_exec, which
    # creates a BaseSubprocessTransport.  When _run_turn is cancelled (e.g.
    # via /stop or on_unmount's cancel_group), the async for exits via
    # CancelledError and the generator is abandoned.  Python tracks abandoned
    # async generators via sys.set_asyncgen_hooks and calls aclose() from
    # shutdown_asyncgens — but only AFTER the event loop starts its teardown
    # sequence, by which point close() has already been called and the
    # __del__ fails.  Calling aclose() here, while still inside a live
    # coroutine, ensures the transport is torn down cleanly.
    _query_gen = query(prompt=_prompt_stream(), options=options)
    try:
        async for message in _query_gen:
            # Explicitly yield control to the event loop on every SDK message.
            # The SDK's async iterator can deliver buffered messages back-to-back
            # without internally awaiting on I/O — when tokens arrive in bursts,
            # the loop body runs synchronously and starves concurrent tasks.
            # asyncio.sleep(0) is the standard way to let the scheduler dispatch
            # other ready tasks before continuing. Cost: negligible.
            #
            # NOTE 2026-04-30: an earlier diagnosis attempted to fix the
            # resize-during-streaming bug by escalating these yields to
            # sleep(0.001). That didn't work because the structural problem
            # was elsewhere: in tui.py, on_input_submitted was awaiting
            # _send_turn directly, blocking Textual's _process_messages_loop
            # on dispatch_message for the entire stream. Yields here had no
            # path to wake Textual's pump because the pump's task was parked
            # on US, not waiting on the queue. The real fix lives in tui.py
            # (run_worker pattern). These sleep(0) yields remain as cheap
            # hygiene against future task-starvation scenarios.
            await asyncio.sleep(0)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    bd = _block_to_dict(block)
                    if bd is None:
                        continue
                    assistant_blocks.append(bd)
                    if bd.get("type") == "text" and bd.get("text"):
                        if on_token is not None:
                            # Per-text-block yield (an AssistantMessage can
                            # carry many text blocks back-to-back).
                            await asyncio.sleep(0)
                            await _maybe_await(on_token(bd["text"]))
                        elif stream and not callback_mode:
                            console.print(bd["text"], end="", soft_wrap=True, highlight=False)
                            printed_any = True
                    elif bd.get("type") == "tool_use":
                        name = bd.get("name", "?")
                        inp = bd.get("input") or {}
                        if on_tool_use is not None:
                            await _maybe_await(on_tool_use(name, inp if isinstance(inp, dict) else {}))
                        elif not callback_mode:
                            console.print(f"\n[dim]> tool: {name}({_short(inp)})[/dim]")
                        log("tool_call", tool=name)
            elif isinstance(message, UserMessage):
                # tool_result blocks come back in a UserMessage. Surface failures
                # in debug log; the model already sees them via the SDK's loop.
                for block in (message.content or []):
                    bd = _block_to_dict(block)
                    if bd and bd.get("type") == "tool_result":
                        if bd.get("is_error"):
                            log("tool_result_error", content=str(bd.get("content"))[:200])
                        if on_tool_result is not None:
                            await _maybe_await(
                                on_tool_result(
                                    bd.get("tool_use_id", ""),
                                    bd.get("content"),
                                    bool(bd.get("is_error", False)),
                                )
                            )
            elif isinstance(message, ResultMessage):
                usage = message.usage or {}
                tokens_in = int(usage.get("input_tokens", 0) or 0) + int(
                    usage.get("cache_read_input_tokens", 0) or 0
                ) + int(usage.get("cache_creation_input_tokens", 0) or 0)
                tokens_out = int(usage.get("output_tokens", 0) or 0)
                if message.is_error:
                    result_str = str(message.result)
                    log("agent_error", subtype=message.subtype, result=result_str[:200])
                    warn(f"agent_error subtype={message.subtype}")
                    # Raise immediately so the TUI sees a descriptive error rather
                    # than the cryptic "Command failed with exit code 1" that the
                    # SDK emits when the subprocess exits after returning this
                    # ResultMessage.  The finally block below still runs to close
                    # the generator cleanly.
                    if any(kw in result_str.lower() for kw in ("too long", "context_length", "tokens")):
                        raise RuntimeError(
                            f"{result_str} — use /evict N to free context or /clear to reset the window"
                        )
                    raise RuntimeError(f"agent error: {result_str}")
            elif HookEventMessage is not None and isinstance(message, HookEventMessage):
                # SDK >= 0.1.74 emits hook lifecycle events into the stream
                # when include_hook_events=True. Feed PreToolUse events to the
                # sentinel so it can detect polling patterns.
                if sentinel is not None and not _sentinel_halted:
                    sentinel.observe(message)
                    halt_reason = sentinel.should_halt()
                    if halt_reason:
                        _sentinel_halted = True
                        notice = (
                            f"[SENTINEL HALT] {halt_reason}. "
                            "Stopping further tool dispatch for this turn. "
                            "Please check in with the user before continuing."
                        )
                        log("sentinel_halt", reason=halt_reason)
                        # Inject a synthetic text block so the notice lands
                        # in the persisted assistant turn.
                        assistant_blocks.append({"type": "text", "text": notice})
                        if on_token is not None:
                            await _maybe_await(on_token(notice))
                        elif stream and not callback_mode:
                            console.print(f"\n[bold red]{notice}[/bold red]")
                            printed_any = True
                        # Break out of the message stream to stop consuming
                        # further tool dispatches for this turn.
                        break
            # SystemMessage (non-hook): init / context info — ignore.
    finally:
        # Explicitly close the generator so SubprocessCLITransport.close()
        # runs while the event loop is still live.  The inner try/except
        # swallows any exception raised by aclose() itself (e.g. a second
        # CancelledError during cleanup) without disturbing whatever
        # exception is already propagating out of this function.
        try:
            await _query_gen.aclose()
        except BaseException:  # noqa: BLE001
            pass

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
    if tool_name == _INSPECT_CONTEXT_TOOL or tool_name.endswith("__inspect_context"):
        return "InspectContext", ""
    if tool_name == _PROPOSE_ROLE_AMENDMENT_TOOL or tool_name.endswith("__propose_role_amendment"):
        return "ProposeRoleAmendment", ""
    if tool_name == _LOG_CHOICE_TOOL or tool_name.endswith("__log_choice"):
        return "LogChoice", ""
    if tool_name == _WIKI_READ_TOOL or tool_name.endswith("__wiki_read"):
        return "WikiRead", ""
    if tool_name == _WIKI_WRITE_TOOL or tool_name.endswith("__wiki_write"):
        return "WikiWrite", ""
    if tool_name == _WIKI_LIST_TOOL or tool_name.endswith("__wiki_list"):
        return "WikiList", ""
    if tool_name == _RAG_INDEX_TOOL or tool_name.endswith("__rag_index"):
        return "RagIndex", ""
    if tool_name == _RAG_QUERY_TOOL or tool_name.endswith("__rag_query"):
        return "RagQuery", ""
    if tool_name == _GRAPH_ADD_NODE_TOOL or tool_name.endswith("__graph_add_node"):
        return "GraphAddNode", ""
    if tool_name == _GRAPH_ADD_EDGE_TOOL or tool_name.endswith("__graph_add_edge"):
        return "GraphAddEdge", ""
    if tool_name == _GRAPH_QUERY_TOOL or tool_name.endswith("__graph_query"):
        return "GraphQuery", ""
    if tool_name == _GRAPH_NEIGHBORS_TOOL or tool_name.endswith("__graph_neighbors"):
        return "GraphNeighbors", ""
    if tool_name == _GRAPH_MATCH_TOOL or tool_name.endswith("__graph_match"):
        return "GraphMatch", ""
    if tool_name == _GRAPH_SHORTEST_PATH_TOOL or tool_name.endswith("__graph_shortest_path"):
        return "GraphShortestPath", ""
    if tool_name == _TUNE_WINDOW_TOOL or tool_name.endswith("__tune_window"):
        return "TuneWindow", ""
    if tool_name == _EVICT_LAST_TOOL or tool_name.endswith("__evict_last"):
        return "EvictLast", ""
    if tool_name == _EVICT_IDS_TOOL or tool_name.endswith("__evict_ids"):
        return "EvictIds", ""
    if tool_name == _MARK_SEGMENT_TOOL or tool_name.endswith("__mark_segment"):
        return "MarkSegment", ""
    if tool_name == _EVICT_SINCE_TOOL or tool_name.endswith("__evict_since"):
        return "EvictSince", ""
    if tool_name == _EVICT_THINKING_BLOCKS_TOOL or tool_name.endswith("__evict_thinking_blocks"):
        return "EvictThinkingBlocks", ""
    if tool_name == _EVICT_TOOL_USE_BLOCKS_TOOL or tool_name.endswith("__evict_tool_use_blocks"):
        return "EvictToolUseBlocks", ""
    if tool_name == _EVICT_WRITE_PAIRS_TOOL or tool_name.endswith("__evict_write_pairs"):
        return "EvictWritePairs", ""
    if tool_name == _EVICT_OLDER_THAN_TOOL or tool_name.endswith("__evict_older_than"):
        return "EvictOlderThan", ""
    if tool_name == _PIN_ROW_TOOL or tool_name.endswith("__pin_row"):
        return "PinRow", ""
    if tool_name == _UNPIN_ROW_TOOL or tool_name.endswith("__unpin_row"):
        return "UnpinRow", ""
    if tool_name == _LIST_PINNED_TOOL or tool_name.endswith("__list_pinned"):
        return "ListPinned", ""
    return None, ""


def _short(d: dict, n: int = 80) -> str:
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"
