"""Unified entry point — trading bot + MCP server in a single process.

                     ┌─────────────────────────────────┐
                     │          BotContext              │
                     │  (shared, thread-safe Lock)      │
                     └────────────┬────────────────────┘
                                  │
              ┌───────────────────┴──────────────────────┐
              │                                          │
   ┌──────────▼──────────┐                  ┌───────────▼──────────┐
   │   Trading loop       │                  │    MCP server         │
   │   (main thread)      │                  │ (background thread)   │
   │                      │                  │                       │
   │  Writes to ctx:      │                  │  Reads from ctx:      │
   │    last_price        │                  │    get_price          │
   │    step              │◄────shared────►  │    get_bot_status     │
   │    has_position      │    BotContext    │    get_risk_metrics   │
   │    episode_reward    │                  │                       │
   │                      │                  │  Writes to ctx:       │
   │  Reads from ctx:     │                  │    open_trade         │
   │    bot_running       │                  │    close_position     │
   │    training_mode     │                  │    emergency_stop     │
   └──────────────────────┘                  └───────────────────────┘

Usage
-----
    python launch.py                        # live trading + MCP via stdio
    python launch.py --dry-run              # dry-run + MCP via stdio
    python launch.py --http                 # live trading + MCP via HTTP/SSE on :8765
    python launch.py --dry-run --http       # dry-run + MCP via HTTP
    python launch.py --forever --grok       # live, Grok enabled, no step limit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
import threading

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from mcp_server.context import BotContext
from mcp_server.server import run_stdio, run_sse
from main import XAUUSDHybridTrader
from trade_bot.core.logging import get_logger

logger = get_logger("launch")


# ---------------------------------------------------------------------------
# MCP background thread
# ---------------------------------------------------------------------------

def _start_mcp_thread(ctx: BotContext, use_http: bool, port: int) -> threading.Thread:
    """Start the MCP server in a daemon thread with its own asyncio event loop."""

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if use_http:
                loop.run_until_complete(run_sse(ctx, port=port))
            else:
                loop.run_until_complete(run_stdio(ctx))
        except Exception as e:
            logger.error("MCP server thread error: %s", e)
        finally:
            loop.close()

    thread = threading.Thread(target=_run, name="mcp-server", daemon=True)
    thread.start()
    logger.info(
        "MCP server started in background thread (transport=%s%s)",
        "http:" + str(port) if use_http else "stdio",
        "",
    )
    return thread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trade-bot + MCP server — single process launcher"
    )
    parser.add_argument("--forever",   action="store_true", help="Run trading loop indefinitely")
    parser.add_argument("--max-steps", type=int, default=50, help="Max trading steps (ignored with --forever)")
    parser.add_argument("--dry-run",   action="store_true", help="Simulate trades, never touch the exchange")
    parser.add_argument("--grok",      action="store_true", help="Enable Groq LLM 3-way voting")
    parser.add_argument("--scalping",  action="store_true", help="Enable scalping agent at startup")
    parser.add_argument("--http",      action="store_true", help="Use HTTP/SSE transport instead of stdio")
    parser.add_argument("--port",      type=int, default=8765, help="SSE port (default 8765)")
    args = parser.parse_args()

    # ── 1. Shared context ──────────────────────────────────────────────
    logger.info("Initialising shared BotContext...")
    ctx = BotContext()

    # ── 2. MCP server (background) ────────────────────────────────────
    _start_mcp_thread(ctx, use_http=args.http, port=args.port)

    # ── 3. Trading bot (foreground, reuses Trader from ctx) ───────────
    bot = XAUUSDHybridTrader(
        use_live_data=True,
        dry_run=args.dry_run,
        use_grok=args.grok,
        ctx=ctx,                 # ← inject shared context
    )

    if args.scalping:
        ctx.scalping.enabled = True
        logger.info("Scalping agent ENABLED via --scalping flag")

    if bot.trader:
        try:
            bot.trader.enable_streaming(bot.epic)
        except Exception as e:
            logger.warning("Streaming not enabled: %s", e)

    logger.info("Loading models...")
    bot.load_models()

    max_steps = None if args.forever else args.max_steps

    try:
        bot.live_trading_loop(max_steps=max_steps)
        bot.save_models()
    except KeyboardInterrupt:
        bot.save_models()
        logger.info("Stopped safely (KeyboardInterrupt)")
    except Exception as e:
        logger.exception("Unhandled error in trading loop: %s", e)
        bot.save_models()


if __name__ == "__main__":
    main()
