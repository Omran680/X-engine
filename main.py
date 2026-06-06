"""XAU/USD Hybrid RL Trading Bot — main trading loop.

Can run standalone or wired to an MCP server via a shared BotContext.
When a BotContext is injected the loop syncs its live state into ctx so
MCP tools always see up-to-date values.
"""

from __future__ import annotations

import numpy as np
from trade_bot.agents import HybridTradingAgent
from trade_bot.features import FeatureExtractor, TradingState
from trade_bot.ensemble import EnsembleDecisionMaker, EnsembleStrategy
from trade_bot.core.config import (
    BATCH_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, EPIC, SIZE,
    CHECKPOINT_INTERVAL, STATE_SIZE, ACTION_SIZE, SCALP_ENABLED,
)
from trade_bot.core.logging import get_logger
from trade_bot.execution import Trader
import time
import os
import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_server.context import BotContext

logger = get_logger(__name__)


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
        self.feature_extractor = FeatureExtractor(lookback_window=20)
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
        self.models_dir = "./models"

        # Reuse the Trader from BotContext when available (single IG connection)
        if ctx is not None and ctx._trader is not None:
            self.trader = ctx._trader
            logger.info("Reusing Trader from BotContext (single IG connection)")
        else:
            self.trader = Trader() if use_live_data else None

        self.epic = EPIC
        self.position = None
        self.last_price: float | None = None
        self.dry_run = dry_run
        self.checkpoint_interval = CHECKPOINT_INTERVAL
        self._last_has_position: bool = False

        # Scalping: either via injected BotContext (shared agent) or standalone
        if ctx is not None:
            self._scalper = ctx.scalping
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

    # ------------------------------------------------------------------
    # LIVE DATA
    # ------------------------------------------------------------------
    def get_current_price(self) -> float:
        if self.trader is None:
            return self.last_price or 2000.0
        try:
            price = self.trader.get_price(self.epic)
            self.last_price = price
            return price
        except Exception as e:
            logger.error("Error fetching price: %s — using last known price %.2f", e, self.last_price or 2000.0)
            return self.last_price or 2000.0

    def check_positions(self) -> bool:
        if self.trader is None:
            return self._last_has_position
        try:
            positions = self.trader.get_positions()
            if hasattr(positions, "empty"):
                return not positions.empty
            if hasattr(positions, "__len__"):
                return len(positions) > 0
            if hasattr(positions, "__bool__"):
                return bool(positions)
            return False
        except Exception as e:
            logger.warning("Error checking positions: %s — using cached value %s", e, self._last_has_position)
            return self._last_has_position

    # ------------------------------------------------------------------
    # EXECUTE TRADE
    # ------------------------------------------------------------------
    def execute_trade(self, action: int, current_price: float):
        direction = "BUY" if action == 0 else "SELL" if action == 1 else None
        if direction is None:
            return None

        if self.dry_run:
            logger.info("[DRY RUN] %s @ %.2f", direction, current_price)
            self.position = {"direction": direction, "entry_price": current_price,
                             "size": SIZE, "deal_id": "dry_run"}
            self._last_has_position = True
            self._sync_ctx(has_position=True, position=self.position)
            return {"dealReference": "dry_run"}

        try:
            sl = (current_price * (1 - STOP_LOSS_PCT / 100)
                  if direction == "BUY"
                  else current_price * (1 + STOP_LOSS_PCT / 100))
            tp = (current_price * (1 + TAKE_PROFIT_PCT / 100)
                  if direction == "BUY"
                  else current_price * (1 - TAKE_PROFIT_PCT / 100))

            result = self.trader.open_trade(
                epic=self.epic,
                direction=direction,
                size=SIZE,
                sl=abs(current_price - sl),
                tp=abs(current_price - tp),
            )
            self.position = {"direction": direction, "entry_price": current_price,
                             "size": SIZE, "deal_id": result.get("dealReference")}
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
            np.array(prices, dtype=np.float32),
            np.array(volumes, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # AGENT OUTPUTS
    # ------------------------------------------------------------------
    def get_dqn_output(self, state: np.ndarray) -> dict:
        q_values = self.agent.dqn_agent.predict(state)
        return {
            "action": int(np.argmax(q_values)),
            "q_values": q_values,
            "confidence": float(np.max(q_values) - np.min(q_values)),
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
            return {"action": 2, "probs": [0.0, 0.0, 1.0], "confidence": 0.0}
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
            return {"action": 2, "probs": [0.0, 0.0, 1.0], "confidence": 0.0}

    # ------------------------------------------------------------------
    # SCALPING
    # ------------------------------------------------------------------
    def _run_scalping(self, price: float) -> None:
        """Check scalp SL/TP/timeout, then evaluate signal and open if triggered."""
        try:
            # 1. Tick — check if open position hit SL/TP/timeout
            exit_reason = self._scalper.tick(price)
            if exit_reason and self._scalper.get_position_info():
                deal_id = (self._scalper.get_position_info() or {}).get("deal_id")
                if deal_id and deal_id != "dry_run" and self.trader:
                    try:
                        self.trader.close_position(deal_id)
                        logger.info("Scalp position closed on exchange: %s", exit_reason)
                    except Exception as e:
                        logger.error("Scalp exchange close failed: %s", e)

            # 2. Signal — only if no open scalp position
            if self._scalper.has_position():
                return

            if len(self.price_history) < self._scalper.lookback:
                return

            signal = self._scalper.evaluate(
                self.price_history[-self._scalper.lookback:],
                self.volume_history[-self._scalper.lookback:],
            )

            if signal.action == "HOLD":
                return

            logger.info(
                "SCALP signal %s (conf=%.2f) | %s",
                signal.action, signal.confidence, signal.reason,
            )

            # 3. Execute scalp trade
            if self.dry_run:
                scalp_pos = self._scalper.open_position(signal.action, price, deal_id="dry_run")
                logger.info("[DRY RUN] SCALP %s @ %.2f", signal.action, price)
            else:
                if self.trader:
                    from trade_bot.core.config import SCALP_SL_PCT, SCALP_TP_PCT, SCALP_SIZE
                    sl_dist = price * SCALP_SL_PCT / 100
                    tp_dist = price * SCALP_TP_PCT / 100
                    try:
                        result = self.trader.open_trade(
                            epic=self.epic,
                            direction=signal.action,
                            size=SCALP_SIZE,
                            sl=sl_dist,
                            tp=tp_dist,
                        )
                        deal_id = result.get("dealReference") if result else None
                        self._scalper.open_position(signal.action, price, deal_id=deal_id)
                    except Exception as e:
                        logger.error("Scalp trade execution failed: %s", e)

        except Exception as e:
            logger.error("Scalping layer error: %s", e)

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save_models(self):
        self.agent.save_models(self.models_dir)

    def load_models(self) -> bool:
        dqn = os.path.exists(os.path.join(self.models_dir, "dqn_model.pkl"))
        ppo = os.path.exists(os.path.join(self.models_dir, "ppo_model.pkl"))
        if dqn or ppo:
            self.agent.load_models(self.models_dir)
            return True
        logger.warning("No saved models found — starting from scratch")
        return False

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def live_trading_loop(self, max_steps: int | None = 1000) -> float:

        logger.info("=== LIVE TRADING BOT STARTED ===")
        self._sync_ctx(bot_running=True)

        step = 0
        episode_reward = 0.0

        try:
            while max_steps is None or step < max_steps:

                # Respect emergency_stop from MCP
                if self._should_stop():
                    logger.warning("Bot stopped via MCP emergency_stop")
                    break

                try:
                    has_position = self.check_positions()
                    self._last_has_position = has_position
                except Exception as e:
                    logger.warning("Position check error: %s — using cache", e)
                    has_position = self._last_has_position

                price = self.get_current_price()
                self.price_history.append(price)
                self.volume_history.append(1000.0)

                # Sync live price into MCP context
                self._sync_ctx(last_price=price, step=step,
                               episode_reward=episode_reward,
                               has_position=has_position)

                if len(self.price_history) < 20:
                    logger.debug("Collecting data... %d/20", len(self.price_history))
                    time.sleep(1)
                    continue

                state = self.get_state(self.price_history[-20:], self.volume_history[-20:])

                dqn_output = self.get_dqn_output(state)
                ppo_output = self.get_ppo_output(state)

                if self.use_grok and self.grok_agent:
                    grok_output = self.get_grok_output(price,
                                                       self.price_history[-20:],
                                                       self.volume_history[-20:])
                    actions = [dqn_output["action"], ppo_output["action"], grok_output["action"]]
                    action = max(set(actions), key=actions.count)
                    reasoning = grok_output.get("reasoning", "")
                    if reasoning:
                        logger.info("Grok reasoning: %s", reasoning)
                else:
                    decision = self.ensemble.combine_predictions(dqn_output, ppo_output)
                    action = decision["action"]

                # Sync action into MCP context
                self._sync_ctx(last_action=action)

                if len(self.price_history) > 1 and self.price_history[-2] > 0:
                    reward = (price - self.price_history[-2]) / self.price_history[-2]
                else:
                    reward = 0.0

                if not has_position and action in [0, 1]:
                    self.execute_trade(action, price)

                # ── Scalping layer ────────────────────────────────────────
                self._run_scalping(price)
                # ─────────────────────────────────────────────────────────

                next_state = state
                done = False

                # Read training_mode live from MCP context (can be toggled)
                if self._training_mode():
                    self.agent.remember(state, action, reward, next_state, done, ppo_output)
                    self.agent.train_step(batch_size=BATCH_SIZE)

                episode_reward += reward
                step += 1

                if step % 10 == 0:
                    logger.info("Step %d | Price %.2f | Action %d | Position %s",
                                step, price, action, has_position)

                if step % (self.checkpoint_interval * 10) == 0:
                    logger.info("Saving checkpoint at step %d...", step)
                    self.save_models()

                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Stopped by user (KeyboardInterrupt)")

        self._sync_ctx(bot_running=False)
        logger.info("Trading loop finished | Total reward: %.6f", episode_reward)
        return episode_reward


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

    if bot.trader:
        try:
            bot.trader.enable_streaming(bot.epic)
        except Exception as e:
            logger.warning("Streaming could not be enabled: %s", e)

    logger.info("Loading models...")
    bot.load_models()

    try:
        max_steps = None if args.forever else args.max_steps
        bot.live_trading_loop(max_steps=max_steps)
        bot.save_models()
    except KeyboardInterrupt:
        bot.save_models()
        logger.info("Stopped safely")
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        bot.save_models()
