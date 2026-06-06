"""MCP Server entry point for trade-bot.

Architecture
------------
Read-only tools  →  mcp_server/read_tools.py   (get_price, get_positions, …)
Execution tools  →  mcp_server/exec_tools.py   (open_trade, close_position, …)

The two modules share state only through BotContext (context.py).
They never import each other — the router below is the only place that
knows both exist.

Usage
-----
    python -m mcp_server.server          # stdio transport (default for Claude Desktop)
    python -m mcp_server.server --http   # SSE transport on port 8765

Claude Desktop config  (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "trade-bot": {
          "command": "/path/to/.venv/bin/python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/path/to/trade-bot"
        }
      }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .context import BotContext
from . import read_tools
from . import exec_tools
from trade_bot.core.logging import get_logger

logger = get_logger("mcp_server")

# ---------------------------------------------------------------------------
# Build tool registry at module level (fast lookup at call time)
# ---------------------------------------------------------------------------

_READ_NAMES:  frozenset[str] = frozenset(t.name for t in read_tools.READ_TOOLS)
_EXEC_NAMES:  frozenset[str] = frozenset(t.name for t in exec_tools.EXEC_TOOLS)
_ALL_TOOLS:   list[Tool]     = read_tools.READ_TOOLS + exec_tools.EXEC_TOOLS

assert not (_READ_NAMES & _EXEC_NAMES), (
    "Tool name collision between read and exec layers: "
    + str(_READ_NAMES & _EXEC_NAMES)
)

logger.info(
    "MCP tool registry: %d read-only tools, %d exec tools",
    len(read_tools.READ_TOOLS),
    len(exec_tools.EXEC_TOOLS),
)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(ctx: BotContext) -> Server:
    app = Server("trade-bot-mcp")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return _ALL_TOOLS

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        arguments = arguments or {}
        logger.info("MCP call_tool: %s  args=%s", name, arguments)

        if name in _READ_NAMES:
            return await read_tools.handle(name, arguments, ctx)

        if name in _EXEC_NAMES:
            return await exec_tools.handle(name, arguments, ctx)

        raise ValueError(f"Tool {name!r} is not registered in this server")

    return app


# ---------------------------------------------------------------------------
# Transport selection
# ---------------------------------------------------------------------------

async def run_stdio(ctx: BotContext) -> None:
    app = create_server(ctx)
    logger.info("MCP server starting — stdio transport")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def run_sse(ctx: BotContext, port: int = 8765) -> None:
    """HTTP + SSE transport — useful for local dashboards / testing."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn

    app = create_server(ctx)
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )

    logger.info("MCP server starting — SSE transport on http://0.0.0.0:%d/sse", port)
    config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trade-bot MCP server")
    parser.add_argument("--http", action="store_true", help="Use SSE/HTTP transport instead of stdio")
    parser.add_argument("--port", type=int, default=8765, help="Port for SSE transport (default 8765)")
    args = parser.parse_args()

    ctx = BotContext()

    if args.http:
        asyncio.run(run_sse(ctx, port=args.port))
    else:
        asyncio.run(run_stdio(ctx))


if __name__ == "__main__":
    main()
