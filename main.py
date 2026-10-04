"""XAU/USD Hybrid RL Trading Bot — main trading loop.

Can run standalone or wired to an MCP server via a shared BotContext.
When a BotContext is injected the loop syncs its live state into ctx so
MCP tools always see up-to-date values.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from trade_bot.agents import HybridTradingAgent
from trade_bot.core.config import (
    ACTION_SIZE, BATCH_SIZE, CHECKPOINT_STEPS, DATA_GAP_RESET_SECONDS, DEFAULT_VOLUME, EPIC,
    HEARTBEAT_FILE, LOOP_INTERVAL_SECONDS, MARKET_CLOSED_LONG_POLL_SECONDS,
    MARKET_CLOSED_POLL_SECONDS, MIN_HISTORY_BARS, MODELS_DIR, OUTAGE_MAX_BACKOFF_SECONDS, POINT_VALUE,
    RISK_INITIAL_CAPITAL, SCALP_ENABLED, SIZE, STATE_SIZE, STOP_LOSS_PCT,
    TAKE_PROFIT_PCT, TRANSACTION_COST,
)
from trade_bot.core.logging import get_logger
from trade_bot.ensemble import EnsembleDecisionMaker, EnsembleStrategy
from trade_bot.execution import Trader
from trade_bot.execution.trader import MarketClosedError
from trade_bot.execution.risk import PortfolioRiskManager
from trade_bot.features import FeatureExtractor, TradingState

if TYPE_CHECKING:
    from mcp_server.context import BotContext

logger = get_logger(__name__)

BUY, SELL, HOLD = 0, 1, 2
MAX_HISTORY = 500              # bars kept in memory (the loop may run for weeks)
MAX_CONSECUTIVE_ERRORS = 20    # abort (→ supervisor restart) after this many failing steps in a row
MAX_STALE_PRICE_STEPS = 12     # stop trading after ~1 min without a fresh price


def q_confidence(q_values: np.ndarray) -> float:
    """Confidence of a DQN decision on the same [1/3, 1] scale as PPO's max-probability.

    Q-values live in reward units (tiny, unbounded), so max-min is not comparable to
    PPO's probability; instead take the softmax of Q-values normalised by their spread.
    """
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    z = (q - q.max()) / (q.std() + 1e-9)
    p = np.exp(z)
    return float(p.max() / p.sum())


def step_reward(action: int, price_return: float, cost: float = TRANSACTION_COST) -> float:
    """Reward of taking ``action`` for one step during which price moved ``price_return``.

    BUY earns the move, SELL earns its opposite, HOLD earns nothing; acting costs
    ``cost`` (spread proxy). Expressed in percent so values are O(0.01-1).
    The previous implementation rewarded the raw price change whatever the action,
    which gave the agents nothing to learn from.
    """
    if action == BUY:
        gross = price_return
    elif action == SELL:
        gross = -price_return
    else:
        return 0.0
    return (gross - cost) * 100.0


class XAUUSDHybridTrader:

    def __init__(
        self,
        use_live_data: bool = True,
        dry_run: bool = True,
        use_grok: bool = False,
        ctx: "BotContext | None" = None,
    ):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self._ctx = ctx  # shared MCP context (optional)

        self.agent = HybridTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            dqn_weight=0.5,
            ppo_weight=0.5,
        )

        self.ensemble = EnsembleDecisionMaker(strategy=EnsembleStrategy.WEIGHTED_VOTING)
        self.feature_extractor = FeatureExtractor(lookback_window=MIN_HISTORY_BARS)
        self.trading_state = TradingState()

        self.use_grok = use_grok
        self.grok_agent = None
        if use_grok:
            try:
                from trade_bot.agents.groq import GrokTradeAgent
                self.grok_agent = GrokTradeAgent()
                logger.info("Grok agent initialized — 3-way voting enabled")
            except ImportError as e:
                logger.warning("Grok agent unavailable (ImportError): %s", e)
                self.use_grok = False
            except Exception as e:
                logger.warning("Grok init failed: %s", e)
                self.use_grok = False

        self.price_history: list = []
        self.volume_history: list = []
        self.models_dir = MODELS_DIR

        # Reuse the Trader from BotContext when available (single IG connection)
        if ctx is not None and ctx._trader is not None:
            self.trader = ctx._trader
            logger.info("Reusing Trader from BotContext (single IG connection)")
        else:
            self.trader = Trader() if use_live_data else None

        self.epic = EPIC
        self.position: Optional[dict] = None
        self.last_price: float | None = None
        self.dry_run = dry_run
        self._last_has_position: bool = False

        # Risk accounting (shared with MCP when a context is injected)
        self.risk = (ctx._risk_manager if ctx is not None and ctx._risk_manager is not None
                     else PortfolioRiskManager(initial_capital=RISK_INITIAL_CAPITAL))
        self._last_risk_block: str | None = None

        # Pending RL transition: (state, action, ppo_output, price) awaiting its next-step reward
        self._pending: dict | None = None
        self._price_failures = 0
        self._episode_reward = 0.0
        self._positions_unknown = False   # True while the positions API is failing

        # Market-hours / outage state
        self._market_open = True
        self._market_status = "TRADEABLE"
        self._closed_since: float | None = None
        self._last_price_time: float | None = None

        # Scalping: either via injected BotContext (shared agent) or standalone
        if ctx is not None:
            self._scalper = ctx.scalping
            ctx.dry_run = dry_run
            ctx.save_models_fn = self.save_models
        else:
            from trade_bot.agents.scalping import ScalpingAgent
            self._scalper = ScalpingAgent()
            self._scalper.enabled = SCALP_ENABLED

    # ------------------------------------------------------------------
    # Sync helpers — push live state into the shared MCP context
    # ------------------------------------------------------------------
    def _sync_ctx(self, **kwargs) -> None:
        if self._ctx is None:
            return
        for k, v in kwargs.items():
            setattr(self._ctx, k, v)

    def _training_mode(self) -> bool:
        """Read training flag from ctx (MCP can toggle it live)."""
        if self._ctx is not None:
            return self._ctx.training_mode
        return True  # default when running standalone

    def _should_stop(self) -> bool:
        """Return True when MCP emergency_stop was called."""
        if self._ctx is not None:
            return not self._ctx.bot_running
        return False

    def _sleep(self, seconds: float) -> None:
        """Sleep in 1 s slices so emergency_stop takes effect promptly."""
        end = time.time() + seconds
        while time.time() < end and not self._should_stop():
            time.sleep(min(1.0, max(0.0, end - time.time())))

    # ------------------------------------------------------------------
    # LIVE DATA
    # ------------------------------------------------------------------
    def get_current_price(self) -> float | None:
        """Latest price, or None when none could be obtained.

        Never invents a price: the old fallback of 2000.0 could open real orders at a
        fictitious level when the API was down at start-up. A closed market is not an
        error — it flips ``_market_open`` and the loop idles until trading resumes.
        """
        if self.trader is None:
            return self.last_price
        try:
            price = self.trader.get_price(self.epic)
        except MarketClosedError as e:
            self._on_market_closed(e.status)
            return None
        except Exception as e:
            self._price_failures += 1
            logger.error("Error fetching price (%d in a row): %s", self._price_failures, e)
            return None
        self._on_price_ok(price)
        return price

    def _on_market_closed(self, status: str) -> None:
        self._price_failures = 0                     # closed is not a failure
        self._market_status = status
        self._sync_ctx(market_open=False)
        if self._market_open:
            self._market_open = False
            self._closed_since = time.time()
            logger.warning("Market %s — trading paused, polling every %ds until it reopens",
                           status, MARKET_CLOSED_POLL_SECONDS)
            self._pending = None                     # no reward for a decision across a closure
            if self._training_mode():
                try:
                    self.save_models()               # checkpoint: a closure is the safest moment
                except Exception as e:
                    logger.error("Checkpoint on market close failed: %s", e)

    def _on_price_ok(self, price: float) -> None:
        now = time.time()
        self.last_price = price
        self._price_failures = 0
        gap = now - self._last_price_time if self._last_price_time else 0.0
        if not self._market_open:
            logger.warning("Market reopened after %.1f h — resuming", (now - (self._closed_since or now)) / 3600)
            self._market_open, self._market_status, self._closed_since = True, "TRADEABLE", None
            self._sync_ctx(market_open=True)
        if gap > DATA_GAP_RESET_SECONDS:
            # Weekend / outage gap: old bars would feed a fake 'jump' into every indicator
            logger.info("Data gap of %.0f s — resetting price history (re-warming %d bars)", gap, MIN_HISTORY_BARS)
            self.price_history.clear()
            self.volume_history.clear()
            self._pending = None
        self._last_price_time = now

    def _idle_seconds(self) -> float:
        """How long the loop should sleep when no decision was possible."""
        if not self._market_open:
            closed_for = time.time() - (self._closed_since or time.time())
            return MARKET_CLOSED_LONG_POLL_SECONDS if closed_for > 3600 else MARKET_CLOSED_POLL_SECONDS
        if self._price_failures:
            # Network / API outage: exponential backoff instead of hammering IG every 5 s
            return min(OUTAGE_MAX_BACKOFF_SECONDS, LOOP_INTERVAL_SECONDS * 2 ** min(self._price_failures, 6))
        return 1 if len(self.price_history) < MIN_HISTORY_BARS else LOOP_INTERVAL_SECONDS

    def _write_heartbeat(self, step: int) -> None:
        """Liveness file for the watchdog (healthcheck.py) — written even while the market is closed."""
        try:
            os.makedirs(os.path.dirname(HEARTBEAT_FILE), exist_ok=True)
            tmp = HEARTBEAT_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"ts": time.time(), "step": step, "market_open": self._market_open,
                           "market_status": self._market_status, "last_price": self.last_price,
                           "dry_run": self.dry_run}, f)
            os.replace(tmp, HEARTBEAT_FILE)
        except OSError as e:
            logger.debug("heartbeat write failed: %s", e)

    def _open_positions_count(self) -> int:
        positions = self.trader.get_positions()
        if hasattr(positions, "empty"):
            return 0 if positions.empty else len(positions)
        return len(positions) if hasattr(positions, "__len__") else int(bool(positions))

    def check_positions(self) -> bool:
        """Is the MAIN strategy in a position? (scalp positions are tracked separately)."""
        if self.dry_run or self.trader is None:
            self._positions_unknown = False
            return self.position is not None
        try:
            n = self._open_positions_count()
            if self._scalper.has_position():
                n -= 1
            self._positions_unknown = False
            return n > 0
        except Exception as e:
            # The cached value may be wrong: flag it so no NEW order is sent blindly
            self._positions_unknown = True
            logger.warning("Error checking positions: %s — state unknown, new entries paused", e)
            return self._last_has_position

    # ------------------------------------------------------------------
    # RISK
    # ------------------------------------------------------------------
    def _risk_ok(self, size: float, sl_distance: float, label: str) -> bool:
        allowed, reason = self.risk.check_trade_allowed(size * sl_distance * POINT_VALUE)
        if not allowed:
            if reason != self._last_risk_block:     # log once per reason, not every step
                logger.warning("%s trade blocked by risk manager: %s", label, reason)
            self._last_risk_block = reason
            return False
        self._last_risk_block = None
        return True

    def _record_pnl(self, direction: str, entry: float, exit_price: float, size: float, label: str) -> None:
        sign = 1.0 if direction == "BUY" else -1.0
        pnl = (exit_price - entry) * sign * size * POINT_VALUE
        self.risk.update_pnl(pnl)
        logger.info("%s closed: %s %.2f → %.2f | pnl≈%.2f | today≈%.2f",
                    label, direction, entry, exit_price, pnl, self.risk.today_pnl)

    def _update_virtual_position(self, price: float) -> None:
        """Dry-run: emulate the SL/TP the exchange would apply to the main position."""
        pos = self.position
        if pos is None:
            return
        sign = 1.0 if pos["direction"] == "BUY" else -1.0
        move_pct = sign * (price - pos["entry_price"]) / pos["entry_price"] * 100
        if move_pct <= -STOP_LOSS_PCT or move_pct >= TAKE_PROFIT_PCT:
            self._record_pnl(pos["direction"], pos["entry_price"], price, pos["size"], "[DRY RUN]")
            self.position = None
            self._last_has_position = False
            self._sync_ctx(has_position=False, position=None)

    def _detect_exchange_close(self, has_position: bool, price: float) -> None:
        """Live: the broker closed the main position (SL/TP) → book its estimated PnL."""
        if self._last_has_position and not has_position and self.position:
            pos = self.position
            self._record_pnl(pos["direction"], pos["entry_price"], price, pos["size"], "Position")
            self.position = None
            self._sync_ctx(position=None)

    # ------------------------------------------------------------------
    # EXECUTE TRADE
    # ------------------------------------------------------------------
    def execute_trade(self, action: int, current_price: float):
        direction = "BUY" if action == BUY else "SELL" if action == SELL else None
        if direction is None:
            return None

        sl_dist = current_price * STOP_LOSS_PCT / 100
        tp_dist = current_price * TAKE_PROFIT_PCT / 100
        if not self._risk_ok(SIZE, sl_dist, "Main"):
            return None

        if self.dry_run:
            logger.info("[DRY RUN] %s @ %.2f", direction, current_price)
            self.position = {"direction": direction, "entry_price": current_price,
                             "size": SIZE, "deal_id": "dry_run"}
            self._last_has_position = True
            self._sync_ctx(has_position=True, position=self.position)
            return {"dealReference": "dry_run"}

        try:
            result = self.trader.open_trade(
                epic=self.epic, direction=direction, size=SIZE, sl=sl_dist, tp=tp_dist,
            )
            self.position = {"direction": direction, "entry_price": current_price,
                             "size": SIZE,
                             "deal_id": result.get("dealId") or result.get("dealReference")}
            self._last_has_position = True
            self._sync_ctx(has_position=True, position=self.position)
            logger.info("OPENED %s @ %.2f", direction, current_price)
            return result
        except Exception as e:
            logger.error("Trade execution failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------
    def get_state(self, prices: list, volumes: list | None = None) -> np.ndarray:
        if volumes is None:
            volumes = np.ones(len(prices))
        return self.feature_extractor.extract_features(
            np.array(prices, dtype=np.float64),
            np.array(volumes, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # AGENT OUTPUTS
    # ------------------------------------------------------------------
    def get_dqn_output(self, state: np.ndarray) -> dict:
        q_values = self.agent.dqn_agent.predict(state)
        return {
            "action": int(np.argmax(q_values)),
            "q_values": q_values,
            "confidence": q_confidence(q_values),
        }

    def get_ppo_output(self, state: np.ndarray) -> dict:
        action, logprob, value, probs = self.agent.ppo_agent.act(state)
        return {
            "action": action,
            "probs": probs,
            "value": value,
            "logprob": logprob,
            "confidence": float(np.max(probs)),
        }

    def get_grok_output(self, price: float, prices: list, volumes: list) -> dict:
        if not self.use_grok or not self.grok_agent:
            return {"action": HOLD, "probs": [0.0, 0.0, 1.0], "confidence": 0.0}
        try:
            result = self.grok_agent.analyze_market(price, prices, volumes)
            return {
                "action": result["action"],
                "probs": result["probs"],
                "confidence": result["confidence"],
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning("Grok error, falling back to HOLD: %s", e)
            return {"action": HOLD, "probs": [0.0, 0.0, 1.0], "confidence": 0.0}

    @staticmethod
    def majority_vote(actions: list[int]) -> int:
        """Majority of the voters; any tie (e.g. BUY/SELL/HOLD) resolves to HOLD."""
        counts = {a: actions.count(a) for a in set(actions)}
        best = max(counts.values())
        winners = [a for a, c in counts.items() if c == best]
        return winners[0] if len(winners) == 1 else HOLD

    # ------------------------------------------------------------------
    # SCALPING
    # ------------------------------------------------------------------
    def _close_scalp_on_exchange(self, closed: dict) -> None:
        deal_id = closed.get("deal_id")
        if deal_id and deal_id != "dry_run" and self.trader and not self.dry_run:
            try:
                self.trader.close_position(deal_id)
                logger.info("Scalp position closed on exchange: %s", closed["reason"])
            except Exception as e:
                # The broker-side SL/TP may already have closed it
                logger.error("Scalp exchange close failed (deal %s): %s", deal_id, e)

    def _run_scalping(self, price: float, allow_open: bool = True) -> None:
        """Check scalp SL/TP/timeout, then evaluate signal and open if triggered."""
        sc = self._scalper
        try:
            # 1. Tick — did the open scalp hit SL/TP/timeout?
            if sc.tick(price) and sc.last_closed:
                closed, sc.last_closed = sc.last_closed, None
                self._close_scalp_on_exchange(closed)
                self.risk.update_pnl(closed["pnl_pct"] * closed["entry_price"]
                                     * closed["size"] * POINT_VALUE)

            # 2. Signal — only if no open scalp position
            if not allow_open or sc.has_position() or len(self.price_history) < sc.lookback:
                return

            signal = sc.evaluate(self.price_history[-sc.lookback:],
                                 self.volume_history[-sc.lookback:])
            if signal.action == "HOLD":
                return

            sl_dist = price * sc.sl_pct / 100
            tp_dist = price * sc.tp_pct / 100
            if not self._risk_ok(sc.size, sl_dist, "Scalp"):
                return

            logger.info("SCALP signal %s (conf=%.2f) | %s", signal.action, signal.confidence, signal.reason)

            # 3. Execute scalp trade (parameters come from the agent so MCP overrides apply)
            if self.dry_run:
                sc.open_position(signal.action, price, deal_id="dry_run")
                logger.info("[DRY RUN] SCALP %s @ %.2f", signal.action, price)
            elif self.trader:
                try:
                    result = self.trader.open_trade(
                        epic=self.epic, direction=signal.action, size=sc.size,
                        sl=sl_dist, tp=tp_dist,
                    )
                    deal_id = (result.get("dealId") or result.get("dealReference")) if result else None
                    sc.open_position(signal.action, price, deal_id=deal_id)
                except Exception as e:
                    logger.error("Scalp trade execution failed: %s", e)

        except Exception as e:
            logger.exception("Scalping layer error: %s", e)

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save_models(self):
        self.agent.save_models(self.models_dir)

    def load_models(self) -> bool:
        dqn = os.path.exists(os.path.join(self.models_dir, "dqn_model.npz"))
        ppo = os.path.exists(os.path.join(self.models_dir, "ppo_model.npz"))
        if not (dqn or ppo):
            logger.warning("No saved models found in %s — starting from scratch", self.models_dir)
            return False
        try:
            self.agent.load_models(self.models_dir)
            return True
        except Exception as e:
            # e.g. weights saved with another STATE_SIZE — never crash-loop on a stale file
            logger.error("Could not load saved models (%s) — starting from scratch", e)
            return False

    # ------------------------------------------------------------------
    # LEARNING
    # ------------------------------------------------------------------
    def _learn_from_previous_step(self, price: float, state: np.ndarray) -> None:
        """Close the pending transition now that its outcome (this step's price) is known."""
        prev, self._pending = self._pending, None
        if prev is None or not self._training_mode():
            return
        price_return = (price - prev["price"]) / prev["price"] if prev["price"] > 0 else 0.0
        reward = step_reward(prev["action"], price_return)
        self.agent.remember(prev["state"], prev["action"], reward, state, False, prev["ppo_output"])
        self.agent.train_step(batch_size=BATCH_SIZE, next_state=state)
        self._episode_reward += reward

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def _trim_history(self) -> None:
        if len(self.price_history) > MAX_HISTORY:
            del self.price_history[:-MAX_HISTORY]
            del self.volume_history[:-MAX_HISTORY]

    def _step(self, step: int) -> bool:
        """One decision step. Returns True when a decision was taken (not warm-up / no data)."""
        price = self.get_current_price()
        if price is None:
            return False

        try:
            has_position = self.check_positions()
        except Exception as e:
            logger.warning("Position check error: %s — using cache", e)
            self._positions_unknown = True
            has_position = self._last_has_position

        if self.dry_run:
            self._update_virtual_position(price)
            has_position = self.position is not None
        else:
            self._detect_exchange_close(has_position, price)
        self._last_has_position = has_position

        self.price_history.append(price)
        self.volume_history.append(DEFAULT_VOLUME)
        self._trim_history()

        self._sync_ctx(market_open=self._market_open, last_price=price, step=step,
                       episode_reward=self._episode_reward, has_position=has_position)

        if len(self.price_history) < MIN_HISTORY_BARS:
            logger.debug("Collecting data... %d/%d", len(self.price_history), MIN_HISTORY_BARS)
            return False

        window_p = self.price_history[-MIN_HISTORY_BARS:]
        window_v = self.volume_history[-MIN_HISTORY_BARS:]
        state = self.get_state(window_p, window_v)

        # Reward for the PREVIOUS decision is now observable
        self._learn_from_previous_step(price, state)

        dqn_output = self.get_dqn_output(state)
        ppo_output = self.get_ppo_output(state)

        if self.use_grok and self.grok_agent:
            grok_output = self.get_grok_output(price, window_p, window_v)
            action = self.majority_vote([dqn_output["action"], ppo_output["action"],
                                         grok_output["action"]])
            if grok_output.get("reasoning"):
                logger.info("Grok reasoning: %s", grok_output["reasoning"])
        else:
            action = int(self.ensemble.combine_predictions(dqn_output, ppo_output)["action"])

        self._sync_ctx(last_action=action)

        # Stale data → never open anything on an old price
        stale = self._price_failures >= MAX_STALE_PRICE_STEPS
        can_open = not stale and not self._positions_unknown
        if not has_position and action in (BUY, SELL) and can_open:
            self.execute_trade(action, price)

        if not stale:   # exits (SL/TP/timeout) must keep running even when entries are paused
            self._run_scalping(price, allow_open=can_open)

        self._pending = {"state": state, "action": action, "ppo_output": ppo_output, "price": price}

        if step % 10 == 0:
            logger.info("Step %d | Price %.2f | Action %d | Position %s | today≈%.2f",
                        step, price, action, has_position, self.risk.today_pnl)
        return True

    def live_trading_loop(self, max_steps: int | None = 1000) -> float:

        logger.info("=== LIVE TRADING BOT STARTED (dry_run=%s) ===", self.dry_run)
        self._sync_ctx(bot_running=True)

        step = 0
        errors = 0
        self._episode_reward = 0.0

        try:
            while max_steps is None or step < max_steps:

                if self._should_stop():
                    logger.warning("Bot stopped via MCP emergency_stop")
                    break

                try:
                    decided = self._step(step)
                    errors = 0
                    self._write_heartbeat(step)
                except Exception as e:
                    errors += 1
                    logger.exception("Step %d failed (%d/%d): %s", step, errors, MAX_CONSECUTIVE_ERRORS, e)
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        raise RuntimeError("Too many consecutive step failures — aborting") from e
                    self._sleep(min(60, LOOP_INTERVAL_SECONDS * errors))
                    continue

                if decided:
                    step += 1
                    if step % CHECKPOINT_STEPS == 0 and self._training_mode():
                        logger.info("Saving checkpoint at step %d...", step)
                        self.save_models()
                    self._sleep(LOOP_INTERVAL_SECONDS)
                else:
                    self._sleep(self._idle_seconds())

        except KeyboardInterrupt:
            logger.info("Stopped by user (KeyboardInterrupt)")
        finally:
            self._sync_ctx(bot_running=False)

        logger.info("Trading loop finished | Total reward: %.6f", self._episode_reward)
        return self._episode_reward


# ------------------------------------------------------------------
# Standalone entry point (without MCP server)
# ------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="XAU/USD Hybrid RL Trading Bot")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--forever",  action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--grok",     action="store_true")
    parser.add_argument("--scalping", action="store_true", help="Enable scalping agent at startup")
    args = parser.parse_args()

    bot = XAUUSDHybridTrader(
        use_live_data=True,
        dry_run=args.dry_run,
        use_grok=args.grok,
    )
    if args.scalping:
        bot._scalper.enabled = True
        logger.info("Scalping agent ENABLED via --scalping flag")

    logger.info("Loading models...")
    bot.load_models()

    try:
        max_steps = None if args.forever else args.max_steps
        bot.live_trading_loop(max_steps=max_steps)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
    finally:
        bot.save_models()
        logger.info("Stopped safely")
