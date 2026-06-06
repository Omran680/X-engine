"""Shared bot context — single source of truth injected into all tool modules.

Both read_tools and exec_tools receive a BotContext instance so they never
import each other and never access global state directly.
"""

from __future__ import annotations

import sys
import os
import threading
import time
from typing import Optional

# Allow importing project modules from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trade_bot.execution import Trader
from trade_bot.execution.risk import PortfolioRiskManager
from trade_bot.agents.scalping import ScalpingAgent
from trade_bot.core.logging import get_logger

logger = get_logger(__name__)


class BotContext:
    """Thread-safe shared state exposed to MCP tools.

    Read tools may only call methods prefixed with `get_` or `check_`.
    Exec tools may call any method.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trader: Optional[Trader] = None
        self._risk_manager: Optional[PortfolioRiskManager] = None

        # Live state updated by the running bot (or by exec tools)
        self.last_price: float = 0.0
        self.last_action: int = 2          # HOLD
        self.step: int = 0
        self.episode_reward: float = 0.0
        self.has_position: bool = False
        self.position: Optional[dict] = None
        self.training_mode: bool = True
        self.bot_running: bool = False
        self.start_time: float = time.time()

        # Scalping agent — shared with the main trading loop
        self.scalping: ScalpingAgent = ScalpingAgent()

        self._init_components()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_components(self) -> None:
        try:
            self._trader = Trader()
            logger.info("BotContext: Trader connected to IG Markets")
        except Exception as e:
            logger.warning("BotContext: Trader init failed — read-only mode: %s", e)

        self._risk_manager = PortfolioRiskManager(initial_capital=10_000)

    # ------------------------------------------------------------------
    # READ helpers (no side effects)
    # ------------------------------------------------------------------
    def get_price(self, epic: str) -> float:
        if self._trader is None:
            return self.last_price
        return self._trader.get_price(epic)

    def get_positions(self) -> list:
        if self._trader is None:
            return []
        return self._trader.get_positions()

    def get_account_balance(self) -> dict:
        if self._trader is None:
            return {"error": "trader not connected"}
        try:
            raw = self._trader.get_account_balance()
            if hasattr(raw, "accounts"):
                return {"accounts": [dict(a) for a in raw.accounts]}
            return {"raw": str(raw)}
        except Exception as e:
            return {"error": str(e)}

    def get_risk_metrics(self) -> dict:
        if self._risk_manager is None:
            return {}
        return self._risk_manager.get_risk_metrics()

    def get_status(self) -> dict:
        uptime = int(time.time() - self.start_time)
        return {
            "bot_running":    self.bot_running,
            "step":           self.step,
            "last_price":     self.last_price,
            "last_action":    ["BUY", "SELL", "HOLD"][self.last_action],
            "has_position":   self.has_position,
            "episode_reward": round(self.episode_reward, 6),
            "training_mode":  self.training_mode,
            "uptime_seconds": uptime,
            "scalping_enabled": self.scalping.enabled,
        }

    def get_scalp_signal(self, prices: list, volumes: list | None = None) -> dict:
        """Evaluate and return the latest scalping signal (read-only)."""
        signal = self.scalping.evaluate(prices, volumes or [])
        return signal.to_dict()

    def get_scalp_status(self) -> dict:
        return self.scalping.get_status()

    # ------------------------------------------------------------------
    # EXEC helpers (mutate state / call exchange API)
    # ------------------------------------------------------------------
    def open_trade(self, epic: str, direction: str, size: float,
                   sl: float | None = None, tp: float | None = None) -> dict:
        if self._trader is None:
            raise RuntimeError("Trader not connected")
        with self._lock:
            result = self._trader.open_trade(epic, direction, size, sl, tp)
            self.has_position = True
            self.position = {
                "direction":   direction,
                "entry_price": self.last_price,
                "size":        size,
                "deal_id":     result.get("dealReference"),
            }
            return result

    def close_position(self, deal_id: str) -> dict:
        if self._trader is None:
            raise RuntimeError("Trader not connected")
        with self._lock:
            result = self._trader.close_position(deal_id)
            self.has_position = False
            self.position = None
            return result

    def set_training_mode(self, enabled: bool) -> None:
        with self._lock:
            self.training_mode = enabled

    def stop_bot(self) -> None:
        with self._lock:
            self.bot_running = False

    # ── Scalping exec ──────────────────────────────────────────────────
    def enable_scalping(self, enabled: bool) -> None:
        with self._lock:
            self.scalping.enabled = enabled
            logger.info("Scalping %s via MCP", "ENABLED" if enabled else "DISABLED")

    def set_scalp_params(self, **kwargs) -> None:
        with self._lock:
            self.scalping.set_params(**kwargs)

    def scalp_force_exit(self, dry_run: bool = False) -> Optional[dict]:
        with self._lock:
            if not self.scalping.has_position():
                return None
            result = self.scalping.force_exit(self.last_price)
            # Close on the exchange unless dry-run
            if not dry_run and result and result.get("deal_id") and result["deal_id"] != "dry_run":
                try:
                    self._trader.close_position(result["deal_id"])
                except Exception as e:
                    logger.error("scalp_force_exit exchange close failed: %s", e)
                    result["exchange_error"] = str(e)
            return result
