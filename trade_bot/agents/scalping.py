"""ScalpingAgent — high-frequency signal engine for XAU/USD.

Strategy
--------
1. **Bollinger-Band squeeze** — detects compression before a breakout.
2. **Micro-momentum** — 3-bar rate-of-change confirms direction.
3. **RSI filter** — avoids entering at extremes (overbought/oversold).
4. **Volume spike** — confirms the move has participation.

Risk controls
-------------
- Tight SL / TP (configured via config.py / MCP set_scalp_params)
- Max N trades per rolling 60-min window
- Auto-exit after max_hold_seconds (checked by the main loop)

Usage
-----
    agent = ScalpingAgent()
    signal = agent.evaluate(prices, volumes)   # → ScalpSignal
    if signal.action != "HOLD" and not agent.has_position():
        # execute trade, then call agent.open_position(...)
    agent.tick(current_price)                  # call every loop iteration
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from trade_bot.core.config import (
    SCALP_LOOKBACK,
    SCALP_SL_PCT,
    SCALP_TP_PCT,
    SCALP_SIZE,
    SCALP_MAX_TRADES_PER_HOUR,
    SCALP_MAX_HOLD_SECONDS,
    SCALP_BB_SQUEEZE_FACTOR,
    SCALP_MOMENTUM_THRESHOLD,
    SCALP_RSI_LOW,
    SCALP_RSI_HIGH,
    SCALP_VOLUME_SPIKE,
)
from trade_bot.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScalpSignal:
    action:     str   = "HOLD"    # "BUY" | "SELL" | "HOLD"
    confidence: float = 0.0
    reason:     str   = ""
    bb_squeeze:      bool  = False
    micro_momentum:  float = 0.0
    rsi:             float = 50.0
    volume_spike:    bool  = False
    bb_width:        float = 0.0

    def to_dict(self) -> dict:
        return {
            "action":         self.action,
            "confidence":     round(self.confidence, 4),
            "reason":         self.reason,
            "bb_squeeze":     self.bb_squeeze,
            "micro_momentum": round(self.micro_momentum, 6),
            "rsi":            round(self.rsi, 2),
            "volume_spike":   self.volume_spike,
            "bb_width":       round(self.bb_width, 6),
        }


@dataclass
class ScalpPosition:
    direction:    str
    entry_price:  float
    entry_time:   float
    size:         float
    sl_price:     float
    tp_price:     float
    deal_id:      Optional[str] = None

    def pnl_pct(self, current_price: float) -> float:
        if self.direction == "BUY":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price

    def sl_hit(self, price: float) -> bool:
        if self.direction == "BUY":
            return price <= self.sl_price
        return price >= self.sl_price

    def tp_hit(self, price: float) -> bool:
        if self.direction == "BUY":
            return price >= self.tp_price
        return price <= self.tp_price

    def timed_out(self) -> bool:
        return (time.time() - self.entry_time) >= SCALP_MAX_HOLD_SECONDS


# ---------------------------------------------------------------------------
# ScalpingAgent
# ---------------------------------------------------------------------------

class ScalpingAgent:
    """Stateful scalping agent — call evaluate() then tick() every loop."""

    def __init__(self) -> None:
        # Params (overridable via set_params)
        self.lookback:          int   = SCALP_LOOKBACK
        self.sl_pct:            float = SCALP_SL_PCT
        self.tp_pct:            float = SCALP_TP_PCT
        self.size:              float = SCALP_SIZE
        self.max_trades_hour:   int   = SCALP_MAX_TRADES_PER_HOUR
        self.max_hold_seconds:  int   = SCALP_MAX_HOLD_SECONDS
        self.bb_squeeze_factor: float = SCALP_BB_SQUEEZE_FACTOR
        self.momentum_thresh:   float = SCALP_MOMENTUM_THRESHOLD
        self.rsi_low:           float = SCALP_RSI_LOW
        self.rsi_high:          float = SCALP_RSI_HIGH
        self.volume_spike_mult: float = SCALP_VOLUME_SPIKE

        # State
        self.enabled:            bool                   = False
        self._position:          Optional[ScalpPosition] = None
        self._trade_timestamps:  deque                  = deque()  # rolling 1-hour window
        self._last_signal:       ScalpSignal            = ScalpSignal()

        # Metrics
        self.total_trades:       int   = 0
        self.winning_trades:     int   = 0
        self.total_pnl:          float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, prices: list[float], volumes: list[float] | None = None) -> ScalpSignal:
        """Compute the current scalping signal from recent price/volume data."""
        if not self.enabled:
            self._last_signal = ScalpSignal(reason="scalping disabled")
            return self._last_signal

        if len(prices) < self.lookback:
            self._last_signal = ScalpSignal(reason=f"insufficient data ({len(prices)}/{self.lookback})")
            return self._last_signal

        if self._position is not None:
            self._last_signal = ScalpSignal(action="HOLD", reason="position already open")
            return self._last_signal

        if not self._trade_allowed():
            self._last_signal = ScalpSignal(
                action="HOLD",
                reason=f"rate limit: {self._trades_last_hour()}/{self.max_trades_hour} trades/h",
            )
            return self._last_signal

        p = np.array(prices[-self.lookback:], dtype=np.float64)
        v = np.array(volumes[-self.lookback:], dtype=np.float64) if volumes else np.ones(self.lookback)

        signal = self._compute_signal(p, v)
        self._last_signal = signal
        return signal

    def tick(self, current_price: float) -> Optional[str]:
        """Check open position for SL / TP / timeout. Returns exit reason or None."""
        if self._position is None:
            return None

        reason: Optional[str] = None

        if self._position.sl_hit(current_price):
            reason = "stop_loss"
        elif self._position.tp_hit(current_price):
            reason = "take_profit"
        elif self._position.timed_out():
            reason = "timeout"

        if reason:
            pnl = self._position.pnl_pct(current_price)
            self._record_close(pnl)
            logger.info(
                "Scalp EXIT [%s] @ %.2f | pnl=%.4f%% | reason=%s",
                self._position.direction, current_price, pnl * 100, reason,
            )
            self._position = None

        return reason

    def open_position(self, direction: str, entry_price: float,
                      deal_id: str | None = None) -> ScalpPosition:
        """Record that a scalp position was opened."""
        sl = (entry_price * (1 - self.sl_pct / 100) if direction == "BUY"
              else entry_price * (1 + self.sl_pct / 100))
        tp = (entry_price * (1 + self.tp_pct / 100) if direction == "BUY"
              else entry_price * (1 - self.tp_pct / 100))

        self._position = ScalpPosition(
            direction=direction,
            entry_price=entry_price,
            entry_time=time.time(),
            size=self.size,
            sl_price=sl,
            tp_price=tp,
            deal_id=deal_id,
        )
        self._trade_timestamps.append(time.time())
        self.total_trades += 1

        logger.info(
            "Scalp OPEN %s @ %.2f | SL=%.2f TP=%.2f | deal=%s",
            direction, entry_price, sl, tp, deal_id,
        )
        return self._position

    def force_exit(self, current_price: float) -> Optional[dict]:
        """Force-close the current scalp position (MCP emergency)."""
        if self._position is None:
            return None
        pnl = self._position.pnl_pct(current_price)
        result = {
            "direction":   self._position.direction,
            "entry_price": self._position.entry_price,
            "exit_price":  current_price,
            "pnl_pct":     round(pnl * 100, 4),
            "deal_id":     self._position.deal_id,
        }
        self._record_close(pnl)
        logger.warning("Scalp FORCE EXIT @ %.2f | pnl=%.4f%%", current_price, pnl * 100)
        self._position = None
        return result

    def has_position(self) -> bool:
        return self._position is not None

    def get_position_info(self) -> Optional[dict]:
        if self._position is None:
            return None
        return {
            "direction":   self._position.direction,
            "entry_price": self._position.entry_price,
            "sl_price":    self._position.sl_price,
            "tp_price":    self._position.tp_price,
            "deal_id":     self._position.deal_id,
            "open_seconds": round(time.time() - self._position.entry_time, 1),
        }

    def get_status(self) -> dict:
        win_rate = (self.winning_trades / self.total_trades * 100
                    if self.total_trades > 0 else 0.0)
        return {
            "enabled":            self.enabled,
            "has_position":       self.has_position(),
            "position":           self.get_position_info(),
            "total_trades":       self.total_trades,
            "winning_trades":     self.winning_trades,
            "win_rate_pct":       round(win_rate, 1),
            "total_pnl_pct":      round(self.total_pnl * 100, 4),
            "trades_last_hour":   self._trades_last_hour(),
            "max_trades_per_hour":self.max_trades_hour,
            "last_signal":        self._last_signal.to_dict(),
            "params": {
                "sl_pct":  self.sl_pct,
                "tp_pct":  self.tp_pct,
                "size":    self.size,
                "lookback": self.lookback,
                "max_hold_seconds": self.max_hold_seconds,
            },
        }

    def set_params(self, **kwargs) -> None:
        """Update scalping parameters at runtime."""
        allowed = {
            "sl_pct", "tp_pct", "size", "lookback",
            "max_trades_hour", "max_hold_seconds",
            "bb_squeeze_factor", "momentum_thresh",
            "rsi_low", "rsi_high", "volume_spike_mult",
        }
        for k, v in kwargs.items():
            if k in allowed:
                setattr(self, k, v)
                logger.info("ScalpingAgent param updated: %s = %s", k, v)
            else:
                logger.warning("ScalpingAgent: unknown param %r ignored", k)

    # ------------------------------------------------------------------
    # Internal signal computation
    # ------------------------------------------------------------------

    def _compute_signal(self, prices: np.ndarray, volumes: np.ndarray) -> ScalpSignal:
        price = float(prices[-1])

        # ── Bollinger Bands ───────────────────────────────────────────
        sma = float(np.mean(prices))
        std = float(np.std(prices))
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_width = (upper - lower) / (sma + 1e-8)
        bb_squeeze = bb_width < self.bb_squeeze_factor

        # ── Micro-momentum (3-bar ROC) ────────────────────────────────
        if len(prices) >= 4:
            micro_momentum = float((prices[-1] - prices[-4]) / (prices[-4] + 1e-8))
        else:
            micro_momentum = 0.0

        # ── RSI (last N bars) ─────────────────────────────────────────
        rsi = self._rsi(prices)

        # ── Volume spike ──────────────────────────────────────────────
        avg_vol = float(np.mean(volumes[:-1])) if len(volumes) > 1 else 1.0
        current_vol = float(volumes[-1])
        volume_spike = current_vol > avg_vol * self.volume_spike_mult

        # ── Direction logic ───────────────────────────────────────────
        bullish = (
            bb_squeeze
            and micro_momentum > self.momentum_thresh
            and rsi < self.rsi_high
            and price > sma          # price breaking above mid-band
        )
        bearish = (
            bb_squeeze
            and micro_momentum < -self.momentum_thresh
            and rsi > self.rsi_low
            and price < sma          # price breaking below mid-band
        )

        # Volume confirmation boosts confidence
        vol_boost = 0.15 if volume_spike else 0.0

        if bullish:
            confidence = min(1.0, 0.5 + abs(micro_momentum) * 100 + vol_boost)
            action = "BUY"
            reason = (
                f"BB squeeze breakout ↑ | momentum={micro_momentum:.5f} "
                f"| RSI={rsi:.1f} | vol_spike={volume_spike}"
            )
        elif bearish:
            confidence = min(1.0, 0.5 + abs(micro_momentum) * 100 + vol_boost)
            action = "SELL"
            reason = (
                f"BB squeeze breakout ↓ | momentum={micro_momentum:.5f} "
                f"| RSI={rsi:.1f} | vol_spike={volume_spike}"
            )
        else:
            confidence = 0.0
            action = "HOLD"
            parts = []
            if not bb_squeeze:
                parts.append(f"no BB squeeze (width={bb_width:.4f})")
            if abs(micro_momentum) <= self.momentum_thresh:
                parts.append("momentum weak")
            if bullish is False and bearish is False and rsi >= self.rsi_high:
                parts.append("RSI overbought")
            if bullish is False and bearish is False and rsi <= self.rsi_low:
                parts.append("RSI oversold")
            reason = "; ".join(parts) or "no signal"

        return ScalpSignal(
            action=action,
            confidence=confidence,
            reason=reason,
            bb_squeeze=bb_squeeze,
            micro_momentum=micro_momentum,
            rsi=rsi,
            volume_spike=volume_spike,
            bb_width=bb_width,
        )

    @staticmethod
    def _rsi(prices: np.ndarray, period: int = 9) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------

    def _prune_old_trades(self) -> None:
        cutoff = time.time() - 3600
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff:
            self._trade_timestamps.popleft()

    def _trades_last_hour(self) -> int:
        self._prune_old_trades()
        return len(self._trade_timestamps)

    def _trade_allowed(self) -> bool:
        return self._trades_last_hour() < self.max_trades_hour

    def _record_close(self, pnl: float) -> None:
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
