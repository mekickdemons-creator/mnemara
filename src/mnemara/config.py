"""Per-instance config — load, save, defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import paths

DEFAULT_MODEL = "gpt-5.3-codex"
SENTINEL_DEFAULT_MODEL = "gpt-5.4-mini"
AVAILABLE_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.2",
]
MODEL_ALIASES = {
    "latest": "gpt-5.5",
    "frontier": "gpt-5.5",
    "default": DEFAULT_MODEL,
    "codex": "gpt-5.3-codex",
    "mini": "gpt-5.4-mini",
}
DEFAULT_MAX_TURNS = 100
DEFAULT_MAX_TOKENS = 500_000  # matches observed productive ceiling — natural compaction sets in around 600K, 500K leaves 100K safety buffer


def normalize_model_name(raw: str) -> str:
    """Validate and normalize a model name string. Returns the cleaned name.

    Accepts Codex/OpenAI-style model identifiers: an alphabetic first character
    followed by letters, digits, dots, or hyphens. Strips outer whitespace
    but rejects any internal whitespace (the bug producer reported
    2026-04-30 was `/swap gpt 5 codex` with spaces silently stored as
    a literal model="gpt 5 codex" string, which the transport then rejected
    on the next turn with an opaque error from deep in the transport).

    Permissive about the model FAMILY by design — OpenAI/Codex model names
    change over time ('gpt-5.3-codex', 'gpt-5.4-mini', future families), and
    a hardcoded allowlist would drift. Format
    validation catches the common typo classes (whitespace, control chars,
    accidental quotes) and lets the API itself reject genuinely-unknown
    model names on first use with a clear error.

    Raises ValueError on:
      - empty / whitespace-only input
      - internal whitespace
      - characters outside [a-zA-Z0-9.-]
      - first character not alphabetic
    """
    if raw is None:
        raise ValueError("model name required")
    s = str(raw).strip()
    if not s:
        raise ValueError("model name required")
    # Reject any internal whitespace explicitly so the error message
    # points at the actual bug rather than a generic format complaint.
    if any(c.isspace() for c in s):
        raise ValueError(
            f"model name contains whitespace: {raw!r} — "
            "did you mean it without spaces? (e.g. 'gpt-5.3-codex')"
        )
    if not s[0].isalpha():
        raise ValueError(
            f"model name must start with a letter, got: {raw!r}"
        )
    # Letters / digits / dots / hyphens only.
    for ch in s:
        if not (ch.isalnum() or ch in ".-"):
            raise ValueError(
                f"model name contains invalid character {ch!r} in {raw!r} "
                "(allowed: letters, digits, '.', '-')"
            )
    return s


def resolve_model_choice(raw: str) -> str:
    """Resolve /swap input to a model id.

    Accepts exact model ids, 1-based indexes from AVAILABLE_MODELS, and a
    small alias set for fast TUI use.
    """
    if raw is None:
        raise ValueError("model name required")
    s = str(raw).strip()
    if not s:
        raise ValueError("model name required")
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(AVAILABLE_MODELS):
            return AVAILABLE_MODELS[idx - 1]
        raise ValueError(
            f"model index out of range: {idx} "
            f"(choose 1-{len(AVAILABLE_MODELS)})"
        )
    alias = MODEL_ALIASES.get(s.lower())
    if alias:
        return alias
    return normalize_model_name(s)


@dataclass
class ToolPolicy:
    tool: str
    mode: str = "ask"  # allow | ask | deny
    allowed_patterns: list[str] = field(default_factory=list)


@dataclass
class McpServer:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    role_doc_path: str = ""
    model: str = DEFAULT_MODEL
    max_window_turns: int = DEFAULT_MAX_TURNS
    max_window_tokens: int = DEFAULT_MAX_TOKENS
    allowed_tools: list[ToolPolicy] = field(default_factory=list)
    mcp_servers: list[McpServer] = field(default_factory=list)
    stream: bool = True
    bash_timeout_seconds: int = 60
    file_tool_home_only: bool = True
    # v0.2.1 — multi-backend memory
    rag_enabled: bool = True
    rag_embed_url: str = "http://localhost:11434/api/embeddings"
    rag_embed_model: str = "nomic-embed-text"
    rag_auto_index_memory: bool = True
    rag_auto_index_wiki: bool = True
    # v0.2.1.1 — panel inbox
    peer_roles: list[str] = field(default_factory=lambda: ["theseus", "majordomo"])
    architect_db_path: str = ""  # empty = not configured; set to .../architect/muninn.db for Aethon
    inbox_auto_surface: bool = True
    # v0.3.1 — agent-level auto-respond to peer pings (off by default; opt-in
    # per panel). When True, the TUI's ambient inbox poller spawns a worker
    # turn the moment a new peer ping lands, prompting the agent to drain +
    # process + reply without waiting for a human-driven turn. Coordinator
    # panels (Theseus, Producer) typically want this on; engineer panels
    # (Substrate) typically leave it off so they only act when explicitly
    # prompted. See tui.py:_check_inbox_ambient for the loop guard
    # (skips payload_type in {ack, ack_final, reply_final}).
    inbox_auto_respond: bool = False
    # v0.3.2 — auto-evict-after-write context discipline
    # When True, after each turn that contained Edit/Write/MultiEdit/
    # NotebookEdit tool_use blocks, mnemara stubs the bulky body content
    # of those tool_use specs (old_string/new_string/content/edits arrays)
    # AND any prior Read tool_use spec for the same file_path. The block
    # itself is preserved (audit-trail intact: "I edited /foo/bar.py")
    # but the bulky strings — often 1-5KB per Edit, full file contents on
    # Write — collapse to a tiny stub like
    # {"file_path": "/foo/bar.py", "_evicted": true}.
    # Pinned rows are skipped. The actual change persists in git or
    # wherever the tool wrote; only the in-context audit body goes.
    # Off by default; opt-in per panel. Same agent-decides-primitive-
    # stays-clean pattern as inbox_auto_respond.
    auto_evict_after_write: bool = False
    # v0.3.3 — token-aware row-cap slack
    # When > 0, the cap-FIFO eviction loop allows n_turns to exceed
    # max_window_turns by up to this many rows, BUT only when current
    # token usage is well under max_window_tokens (under 50% — see
    # Store.HEADROOM_RATIO). When tokens climb back above the threshold
    # the slack disappears and FIFO trims down to max_window_turns.
    #
    # Motivation: heavy block surgery (evict_thinking_blocks, evict_
    # tool_use_blocks, evict_write_pairs) can free 80% of stored bytes
    # without dropping any rows. Without slack, the row cap then fires
    # too early — Major hits 100/100 turns with 50KB of actual context
    # and watches an old row get evicted despite massive token headroom.
    # Slack lets the row cap "breathe" with the byte budget: under
    # token pressure the row cap is strict; with token headroom it
    # gives extra room. The token cap (max_window_tokens) remains the
    # hard ceiling and trims regardless of slack.
    #
    # Default 0 = feature off (backward-compat). A reasonable starting
    # value is 30 (30% slack on a 100-turn cap). Set in config.json.
    row_cap_slack_when_token_headroom: int = 0
    # v0.3 — graph backend + sleep/replay
    graph_enabled: bool = True
    replay_default_days: int = 7
    replay_default_threshold: int = 3
    replay_policy_path: str = ""  # empty = use default at <instance>/wiki/replay_policy.md

    @classmethod
    def default(cls) -> "Config":
        return cls(
            allowed_tools=[
                ToolPolicy(tool="Bash", mode="ask"),
                ToolPolicy(tool="Read", mode="allow"),
                ToolPolicy(tool="Write", mode="ask"),
                ToolPolicy(tool="Edit", mode="ask"),
                ToolPolicy(tool="WriteMemory", mode="allow"),
                ToolPolicy(tool="InspectContext", mode="allow"),
                ToolPolicy(tool="ProposeRoleAmendment", mode="allow"),
                ToolPolicy(tool="LogChoice", mode="allow"),
                ToolPolicy(tool="WikiRead", mode="allow"),
                ToolPolicy(tool="WikiWrite", mode="allow"),
                ToolPolicy(tool="WikiList", mode="allow"),
                ToolPolicy(tool="RagIndex", mode="allow"),
                ToolPolicy(tool="RagQuery", mode="allow"),
                ToolPolicy(tool="GraphAddNode", mode="allow"),
                ToolPolicy(tool="GraphAddEdge", mode="allow"),
                ToolPolicy(tool="GraphQuery", mode="allow"),
                ToolPolicy(tool="GraphNeighbors", mode="allow"),
                ToolPolicy(tool="GraphMatch", mode="allow"),
                ToolPolicy(tool="GraphShortestPath", mode="allow"),
                ToolPolicy(tool="TuneWindow", mode="allow"),
                ToolPolicy(tool="EvictLast", mode="allow"),
                ToolPolicy(tool="EvictIds", mode="allow"),
                ToolPolicy(tool="MarkSegment", mode="allow"),
                ToolPolicy(tool="EvictSince", mode="allow"),
                ToolPolicy(tool="EvictThinkingBlocks", mode="allow"),
                ToolPolicy(tool="EvictToolUseBlocks", mode="allow"),
                ToolPolicy(tool="EvictOlderThan", mode="allow"),
                ToolPolicy(tool="EvictWritePairs", mode="allow"),
                ToolPolicy(tool="PinRow", mode="allow"),
                ToolPolicy(tool="UnpinRow", mode="allow"),
                ToolPolicy(tool="ListPinned", mode="allow"),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        tools = [ToolPolicy(**t) for t in d.get("allowed_tools", [])]
        servers = [
            McpServer(
                name=str(s.get("name", "")),
                command=str(s.get("command", "")),
                args=list(s.get("args", [])),
                env=dict(s.get("env", {})),
            )
            for s in d.get("mcp_servers", [])
            if isinstance(s, dict)
        ]
        return cls(
            role_doc_path=d.get("role_doc_path", ""),
            model=d.get("model", DEFAULT_MODEL),
            max_window_turns=int(d.get("max_window_turns", DEFAULT_MAX_TURNS)),
            max_window_tokens=int(d.get("max_window_tokens", DEFAULT_MAX_TOKENS)),
            allowed_tools=tools,
            mcp_servers=servers,
            stream=bool(d.get("stream", True)),
            bash_timeout_seconds=int(d.get("bash_timeout_seconds", 60)),
            file_tool_home_only=bool(d.get("file_tool_home_only", True)),
            rag_enabled=bool(d.get("rag_enabled", True)),
            rag_embed_url=str(d.get("rag_embed_url", "http://localhost:11434/api/embeddings")),
            rag_embed_model=str(d.get("rag_embed_model", "nomic-embed-text")),
            rag_auto_index_memory=bool(d.get("rag_auto_index_memory", True)),
            rag_auto_index_wiki=bool(d.get("rag_auto_index_wiki", True)),
            peer_roles=list(d.get("peer_roles", ["theseus", "majordomo"])),
            architect_db_path=str(d.get("architect_db_path", "") or ""),
            inbox_auto_surface=bool(d.get("inbox_auto_surface", True)),
            inbox_auto_respond=bool(d.get("inbox_auto_respond", False)),
            auto_evict_after_write=bool(d.get("auto_evict_after_write", False)),
            row_cap_slack_when_token_headroom=int(
                d.get("row_cap_slack_when_token_headroom", 0)
            ),
            graph_enabled=bool(d.get("graph_enabled", True)),
            replay_default_days=int(d.get("replay_default_days", 7)),
            replay_default_threshold=int(d.get("replay_default_threshold", 3)),
            replay_policy_path=str(d.get("replay_policy_path", "") or ""),
        )

    def policy_for(self, tool: str) -> ToolPolicy:
        for t in self.allowed_tools:
            if t.tool == tool:
                return t
        p = ToolPolicy(tool=tool, mode="ask")
        self.allowed_tools.append(p)
        return p


def load(instance: str) -> Config:
    path = paths.config_path(instance)
    if not path.exists():
        raise FileNotFoundError(f"No config at {path} — run `mnemara init --instance {instance}`")
    with path.open() as f:
        raw = json.load(f)
    cfg = Config.from_dict(raw)
    # Backward-compat: older configs may not carry an explicit model key.
    # Keep existing behavior for all instances except sentinel, which should
    # default to mini on stable per product policy.
    if "model" not in raw and instance == "sentinel":
        cfg.model = SENTINEL_DEFAULT_MODEL
    return cfg


def save(instance: str, cfg: Config) -> None:
    path = paths.config_path(instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def init_instance(instance: str, role_doc_path: str = "") -> Path:
    d = paths.instance_dir(instance)
    if d.exists():
        raise FileExistsError(f"Instance already exists at {d}")
    d.mkdir(parents=True)
    paths.memory_dir(instance).mkdir(exist_ok=True)
    paths.wiki_dir(instance).mkdir(exist_ok=True)
    paths.rag_index_dir(instance).mkdir(exist_ok=True)
    cfg = Config.default()
    cfg.role_doc_path = role_doc_path
    if instance == "sentinel":
        cfg.model = SENTINEL_DEFAULT_MODEL
    save(instance, cfg)
    # touch permissions file
    paths.permissions_path(instance).write_text("{}\n")
    return d
