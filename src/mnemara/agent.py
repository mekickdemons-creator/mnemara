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

from . import inbox as inbox_mod
from . import paths as paths_mod
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
        # Inbox auto-surface: prepend a notice when peer pings are waiting.
        if getattr(self.cfg, "inbox_auto_surface", True):
            try:
                db = getattr(self.cfg, "architect_db_path", "") or ""
                peers = getattr(self.cfg, "peer_roles", ["theseus", "majordomo"])
                if db:
                    pings = inbox_mod.peek_pending_pings(
                        db, peers, exclude_role=self.runner.instance
                    )
                    if pings:
                        senders = sorted({p["agent_role"] for p in pings})
                        user_text = (
                            f"[INBOX: {len(pings)} ping(s) waiting from "
                            f"{', '.join(senders)} — call next_return to read]\n\n"
                            + user_text
                        )
            except Exception:
                pass

        # Persist the user turn before contacting the model.
        user_blocks = [{"type": "text", "text": user_text}]
        self.store.append_turn("user", user_blocks)

        system_prompt = role_mod.load_role_doc(self.cfg.role_doc_path) or (
            "You are a helpful assistant."
        )

        # Build prompt: window transcript prefix + current user input.
        prompt = _build_prompt(self.store, user_text)

        options = self._build_options(system_prompt)

        result = await _run_turn(
            prompt,
            options,
            self.cfg.stream,
            on_token=on_token,
            on_tool_use=on_tool_use,
            on_tool_result=on_tool_result,
        )

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
) -> dict[str, Any]:
    assistant_blocks: list[dict] = []
    tokens_in = 0
    tokens_out = 0
    printed_any = False
    # When any callback is wired the caller (TUI) owns presentation —
    # suppress the default console rendering to avoid double-output.
    callback_mode = any(c is not None for c in (on_token, on_tool_use, on_tool_result))

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

    async for message in query(prompt=_prompt_stream(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                bd = _block_to_dict(block)
                if bd is None:
                    continue
                assistant_blocks.append(bd)
                if bd.get("type") == "text" and bd.get("text"):
                    if on_token is not None:
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
    return None, ""


def _short(d: dict, n: int = 80) -> str:
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"
