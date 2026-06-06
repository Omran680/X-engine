"""READ-ONLY MCP tools — zero side effects, never mutate state.

Contract:
  - No calls to open_trade / close_position / stop_bot / enable_scalping
  - No writes to BotContext fields
  - Every function is safe to call at any frequency
"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import Tool, TextContent

from .context import BotContext
from trade_bot.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

READ_TOOLS: list[Tool] = [
    Tool(
        name="get_price",
        description=(
            "Fetch the current bid price for a given IG Markets epic. "
            "Uses the cache if a fresh value is available (max 5 s old)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "epic": {
                    "type": "string",
                    "description": "IG Markets epic identifier, e.g. CS.D.IN_GOLD.MFI.IP",
                }
            },
            "required": ["epic"],
        },
    ),
    Tool(
        name="get_positions",
        description="Return all currently open positions on the IG Markets account.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_account_balance",
        description="Return account balance and equity information.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_risk_metrics",
        description=(
            "Return live risk metrics: Sharpe ratio, win rate, profit factor, "
            "max drawdown, current capital, and consecutive losses."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_bot_status",
        description=(
            "Return the current state of the trading bot: running flag, step counter, "
            "last price, last action, open position flag, uptime, and scalping status."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    # ── Scalping read tools ──────────────────────────────────────────────
    Tool(
        name="get_scalp_signal",
        description=(
            "Evaluate the current scalping signal using the last N price bars. "
            "Returns action (BUY/SELL/HOLD), confidence, BB-squeeze flag, "
            "micro-momentum, RSI, and volume-spike flag. Read-only — does NOT open a trade."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Recent price history (at least 10 values).",
                },
                "volumes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Corresponding volume values (optional).",
                },
            },
            "required": ["prices"],
        },
    ),
    Tool(
        name="get_scalp_status",
        description=(
            "Return the scalping agent's full status: enabled flag, open position details, "
            "total/winning trades, win rate, cumulative PnL, trades-per-hour counter, "
            "current parameters, and last evaluated signal."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle(name: str, arguments: dict[str, Any], ctx: BotContext) -> list[TextContent]:
    """Dispatch to the correct read handler. Raises ValueError on unknown tool."""

    if name == "get_price":
        epic = arguments.get("epic", "CS.D.IN_GOLD.MFI.IP")
        try:
            price = ctx.get_price(epic)
            result = {"epic": epic, "price": price}
        except Exception as e:
            logger.error("get_price failed: %s", e)
            result = {"epic": epic, "error": str(e)}

    elif name == "get_positions":
        try:
            positions = ctx.get_positions()
            if hasattr(positions, "to_dict"):
                result = {"positions": positions.to_dict(orient="records")}
            elif hasattr(positions, "__iter__"):
                result = {"positions": [dict(p) if hasattr(p, "__dict__") else p
                                        for p in positions]}
            else:
                result = {"positions": str(positions)}
        except Exception as e:
            logger.error("get_positions failed: %s", e)
            result = {"error": str(e)}

    elif name == "get_account_balance":
        try:
            result = ctx.get_account_balance()
        except Exception as e:
            logger.error("get_account_balance failed: %s", e)
            result = {"error": str(e)}

    elif name == "get_risk_metrics":
        try:
            result = ctx.get_risk_metrics()
        except Exception as e:
            logger.error("get_risk_metrics failed: %s", e)
            result = {"error": str(e)}

    elif name == "get_bot_status":
        result = ctx.get_status()

    elif name == "get_scalp_signal":
        prices  = arguments.get("prices", [])
        volumes = arguments.get("volumes")
        if not prices:
            result = {"error": "prices list is required"}
        else:
            try:
                result = ctx.get_scalp_signal(prices, volumes)
            except Exception as e:
                logger.error("get_scalp_signal failed: %s", e)
                result = {"error": str(e)}

    elif name == "get_scalp_status":
        result = ctx.get_scalp_status()

    else:
        raise ValueError(f"Unknown read tool: {name!r}")

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
