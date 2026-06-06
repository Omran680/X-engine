"""EXECUTION MCP tools — these tools mutate state and/or call the exchange.

Contract:
  - Every tool that touches the exchange is logged at INFO level
  - Arguments are validated before any exchange call
  - All tools return a structured JSON result with a "status" field
  - Tools in this module must NEVER be called by read_tools
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mcp.types import Tool, TextContent

from .context import BotContext
from trade_bot.core.logging import get_logger

logger = get_logger(__name__)

_VALID_DIRECTIONS = {"BUY", "SELL"}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

EXEC_TOOLS: list[Tool] = [
    Tool(
        name="open_trade",
        description=(
            "Open a new market order on IG Markets. "
            "⚠️ This sends a real order to the exchange. "
            "Provide direction ('BUY' or 'SELL'), size (lots), "
            "and optional stop/limit distances in points."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "epic":           {"type": "string",  "description": "IG Markets epic."},
                "direction":      {"type": "string",  "enum": ["BUY", "SELL"]},
                "size":           {"type": "number",  "description": "Position size in lots."},
                "stop_distance":  {"type": "number",  "description": "SL distance in points."},
                "limit_distance": {"type": "number",  "description": "TP distance in points."},
            },
            "required": ["direction", "size"],
        },
    ),
    Tool(
        name="close_position",
        description="Close an open position by deal ID. ⚠️ Real exchange call.",
        inputSchema={
            "type": "object",
            "properties": {
                "deal_id": {"type": "string", "description": "Deal reference from open_trade."}
            },
            "required": ["deal_id"],
        },
    ),
    Tool(
        name="set_training_mode",
        description="Enable or disable online RL training during the live trading loop.",
        inputSchema={
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"}
            },
            "required": ["enabled"],
        },
    ),
    Tool(
        name="save_models",
        description="Immediately persist the current DQN + PPO model weights to disk.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="emergency_stop",
        description=(
            "⛔ Immediately halt the trading loop. "
            "Does NOT close open positions — use close_position for that."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    # ── Scalping exec tools ──────────────────────────────────────────────
    Tool(
        name="enable_scalping",
        description=(
            "Enable or disable the scalping agent. "
            "When enabled, the main trading loop evaluates BB-squeeze/RSI signals "
            "on every tick and may open short-term scalp positions independently "
            "of the main DQN/PPO agent."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "True = activate scalping; False = deactivate.",
                }
            },
            "required": ["enabled"],
        },
    ),
    Tool(
        name="set_scalp_params",
        description=(
            "Update scalping risk/signal parameters at runtime without restarting the bot. "
            "All fields are optional — only provided keys are updated."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sl_pct":             {"type": "number", "description": "Stop-loss % (e.g. 0.15)."},
                "tp_pct":             {"type": "number", "description": "Take-profit % (e.g. 0.25)."},
                "size":               {"type": "number", "description": "Lot size (e.g. 0.1)."},
                "lookback":           {"type": "integer","description": "Bars for signal (e.g. 10)."},
                "max_trades_hour":    {"type": "integer","description": "Max trades per 60 min."},
                "max_hold_seconds":   {"type": "integer","description": "Auto-exit timeout (s)."},
                "bb_squeeze_factor":  {"type": "number", "description": "BB width threshold."},
                "momentum_thresh":    {"type": "number", "description": "Min 3-bar ROC."},
                "rsi_low":            {"type": "number", "description": "RSI oversold threshold."},
                "rsi_high":           {"type": "number", "description": "RSI overbought threshold."},
                "volume_spike_mult":  {"type": "number", "description": "Volume spike multiplier."},
            },
        },
    ),
    Tool(
        name="scalp_force_exit",
        description=(
            "⚠️ Immediately close the current scalp position at market price. "
            "Useful when you want to manually exit before SL/TP/timeout triggers."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle(name: str, arguments: dict[str, Any], ctx: BotContext) -> list[TextContent]:

    # ── Main bot tools ────────────────────────────────────────────────────
    if name == "open_trade":
        direction = str(arguments.get("direction", "")).upper()
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be BUY or SELL, got {direction!r}")
        size = float(arguments.get("size", 0))
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        epic = arguments.get("epic") or os.getenv("EPIC", "CS.D.IN_GOLD.MFI.IP")
        sl   = arguments.get("stop_distance")
        tp   = arguments.get("limit_distance")
        logger.info("EXEC open_trade: %s %s size=%.3f sl=%s tp=%s", direction, epic, size, sl, tp)
        try:
            result  = ctx.open_trade(epic, direction, size,
                                     sl=float(sl) if sl is not None else None,
                                     tp=float(tp) if tp is not None else None)
            payload = {"status": "opened", "deal": result}
        except Exception as e:
            logger.error("open_trade failed: %s", e)
            payload = {"status": "error", "error": str(e)}

    elif name == "close_position":
        deal_id = str(arguments.get("deal_id", "")).strip()
        if not deal_id:
            raise ValueError("deal_id is required")
        logger.info("EXEC close_position: deal_id=%s", deal_id)
        try:
            result  = ctx.close_position(deal_id)
            payload = {"status": "closed", "result": result}
        except Exception as e:
            logger.error("close_position failed: %s", e)
            payload = {"status": "error", "error": str(e)}

    elif name == "set_training_mode":
        enabled = bool(arguments.get("enabled", True))
        ctx.set_training_mode(enabled)
        logger.info("EXEC set_training_mode: %s", enabled)
        payload = {"status": "ok", "training_mode": enabled}

    elif name == "save_models":
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from trade_bot.agents import HybridTradingAgent
            from trade_bot.core.config import STATE_SIZE, ACTION_SIZE
            agent = HybridTradingAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
            agent.load_models(models_dir)
            agent.save_models(models_dir)
            logger.info("EXEC save_models: saved to %s", models_dir)
            payload = {"status": "saved", "path": models_dir}
        except Exception as e:
            logger.error("save_models failed: %s", e)
            payload = {"status": "error", "error": str(e)}

    elif name == "emergency_stop":
        ctx.stop_bot()
        logger.warning("EXEC emergency_stop triggered via MCP")
        payload = {"status": "stopped", "warning": "open positions NOT closed automatically"}

    # ── Scalping exec tools ───────────────────────────────────────────────
    elif name == "enable_scalping":
        enabled = bool(arguments.get("enabled", True))
        ctx.enable_scalping(enabled)
        payload = {
            "status":   "ok",
            "scalping": "enabled" if enabled else "disabled",
        }

    elif name == "set_scalp_params":
        # Cast numeric args to float/int as required
        clean: dict = {}
        int_keys = {"lookback", "max_trades_hour", "max_hold_seconds"}
        for k, v in arguments.items():
            clean[k] = int(v) if k in int_keys else float(v)
        if not clean:
            payload = {"status": "noop", "message": "no parameters provided"}
        else:
            ctx.set_scalp_params(**clean)
            payload = {"status": "ok", "updated": clean}

    elif name == "scalp_force_exit":
        if not ctx.scalping.has_position():
            payload = {"status": "noop", "message": "no open scalp position"}
        else:
            try:
                result  = ctx.scalp_force_exit(dry_run=False)
                payload = {"status": "exited", "result": result}
            except Exception as e:
                logger.error("scalp_force_exit failed: %s", e)
                payload = {"status": "error", "error": str(e)}

    else:
        raise ValueError(f"Unknown exec tool: {name!r}")

    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]
