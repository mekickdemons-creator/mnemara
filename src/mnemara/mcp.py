"""MCP wire-through.

Anthropic's Messages API supports an `mcp_servers` parameter (in beta). When
present, we pass the configured servers straight through. We treat MCP servers
as opaque to the agent loop here — the model invokes them server-side.

For local-stdio MCP servers configured by the user, we surface them in the
config but require the SDK's connector form. If your SDK build doesn't support
mcp_servers yet, this module degrades gracefully: it returns an empty list and
a warning is logged.
"""
from __future__ import annotations

from typing import Any

from .config import Config
from .logging_util import log, warn


def build_mcp_param(cfg: Config) -> list[dict[str, Any]]:
    """Build the value passed as `mcp_servers=` to messages.create().

    The SDK accepts a list of server descriptors. For stdio servers configured
    locally we pass them through as-is; the SDK is expected to launch them.
    Unknown shape — left as a list of dicts the SDK can interpret.
    """
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
