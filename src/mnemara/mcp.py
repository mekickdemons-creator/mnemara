"""MCP config helpers.

Mnemara stores local stdio MCP server descriptors in config.json and exposes
their names in runtime metadata. The Claude Agent SDK handles the actual MCP
wire-through; this helper keeps the descriptor conversion small and stable.
"""
from __future__ import annotations

from typing import Any

from .config import Config
from .logging_util import log, warn


def build_mcp_param(cfg: Config) -> list[dict[str, Any]]:
    """Build normalized stdio MCP server descriptors from config."""
    out = []
    for s in cfg.mcp_servers:
        out.append(
            {
                "type": "stdio",
                "name": s.name,
                "command": s.command,
                "args": s.args,
                "env": s.env,
            }
        )
    if out:
        log("mcp_configured", servers=[s["name"] for s in out])
    return out
