"""Per-instance config — load, save, defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import paths

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TURNS = 100
DEFAULT_MAX_TOKENS = 500_000  # matches observed productive ceiling — natural compaction sets in around 600K, 500K leaves 100K safety buffer


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
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        tools = [ToolPolicy(**t) for t in d.get("allowed_tools", [])]
        servers = [McpServer(**s) for s in d.get("mcp_servers", [])]
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
        return Config.from_dict(json.load(f))


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
    save(instance, cfg)
    # touch permissions file
    paths.permissions_path(instance).write_text("{}\n")
    return d
