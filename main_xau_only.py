import numpy as np
from agent import HybridTradingAgent
from feature_extractor import FeatureExtractor, TradingState
from ensemble_strategies import EnsembleDecisionMaker, EnsembleStrategy
from config import BATCH_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, EPIC, SIZE, CHECKPOINT_INTERVAL
from trader import Trader
import time
import os
import argparse


class XAUUSDHybridTrader:

    def __init__(self, use_live_data=True, dry_run=True):

        self.state_size = 18
        self.action_size = 3  # BUY, SELL, HOLD

        self.agent = HybridTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            dqn_weight=0.5,
            ppo_weight=0.5
        )

        self.ensemble = EnsembleDecisionMaker(strategy=EnsembleStrategy.WEIGHTED_VOTING)

        self.feature_extractor = FeatureExtractor(lookback_window=20)
        self.trading_state = TradingState()

        self.price_history = []
        self.volume_history = []

        self.models_dir = "./models"

        self.trader = Trader() if use_live_data else None
        self.epic = EPIC
        self.position = None
        self.last_price = None
        self.dry_run = dry_run
        self.checkpoint_interval = CHECKPOINT_INTERVAL

        # =========================
        # FIX 2: safe position cache
        # =========================
        self._last_has_position = False

    # =========================
    # LIVE DATA
    # =========================
    def get_current_price(self):
        try:
            price = self.trader.get_price(self.epic)
            self.last_price = price
            return price
        except Exception as e:
            print(f"Error fetching price: {e}")
            return self.last_price or 2000.0

    def check_positions(self):
        try:
            positions = self.trader.get_positions()

            if hasattr(positions, 'empty'):
                return not positions.empty
            elif hasattr(positions, '__len__'):
                return len(positions) > 0
            elif hasattr(positions, '__bool__'):
                return bool(positions)
            return False

        except Exception as e:
            print(f"Error checking positions: {e}")
            return False

    # =========================
    # EXECUTE TRADE
    # =========================
    def execute_trade(self, action, current_price):

        direction = "BUY" if action == 0 else "SELL" if action == 1 else None
        if direction is None:
            return None

        if self.dry_run:
            print(f"[DRY RUN] {direction} @ {current_price}")

            self.position = {
                "direction": direction,
                "entry_price": current_price,
                "size": SIZE,
                "deal_id": "dry_run"
            }

            # FIX 3
            self._last_has_position = True
            return {"dealReference": "dry_run"}

        try:
            sl = current_price * (1 - STOP_LOSS_PCT / 100) if direction == "BUY" else current_price * (1 + STOP_LOSS_PCT / 100)
            tp = current_price * (1 + TAKE_PROFIT_PCT / 100) if direction == "BUY" else current_price * (1 - TAKE_PROFIT_PCT / 100)

            result = self.trader.open_trade(
                epic=self.epic,
                direction=direction,
                size=SIZE,
                sl=abs(current_price - sl),
                tp=abs(current_price - tp)
            )

            self.position = {
                "direction": direction,
                "entry_price": current_price,
                "size": SIZE,
                "deal_id": result.get("dealReference")
            }

            # FIX 3
            self._last_has_position = True

            print(f"OPENED {direction} @ {current_price}")
            return result

        except Exception as e:
            print(f"Trade execution failed: {e}")
            return None

    # =========================
    # STATE
    # =========================
    def get_state(self, prices, volumes=None):

        if volumes is None:
            volumes = np.ones(len(prices))

        return self.feature_extractor.extract_features(
            np.array(prices, dtype=np.float32),
            np.array(volumes, dtype=np.float32)
        )

    # =========================
    # DQN
    # =========================
    def get_dqn_output(self, state):
        q_values = self.agent.dqn_agent.predict(state)
        return {
            "action": int(np.argmax(q_values)),
            "q_values": q_values,
            "confidence": float(np.max(q_values) - np.min(q_values))
        }

    # =========================
    # PPO
    # =========================
    def get_ppo_output(self, state):
        action, logprob, value, probs = self.agent.ppo_agent.act(state)
        return {
            "action": action,
            "probs": probs,
            "value": value,
            "logprob": logprob,
            "confidence": float(np.max(probs))
        }

    # =========================
    # SAVE / LOAD
    # =========================
    def save_models(self):
        self.agent.save_models(self.models_dir)

    def load_models(self):
        dqn = os.path.exists(os.path.join(self.models_dir, "dqn_model.pkl"))
        ppo = os.path.exists(os.path.join(self.models_dir, "ppo_model.pkl"))

        if dqn or ppo:
            self.agent.load_models(self.models_dir)
            return True

        print("No saved models found")
        return False

    # =========================
    # MAIN LOOP
    # =========================
    def live_trading_loop(self, max_steps=1000, training_mode=True):

        print("\n=== LIVE TRADING BOT ===\n")

        step = 0
        episode_reward = 0

        try:
            while max_steps is None or step < max_steps:

                # =========================
                # FIX 2 + SAFE POSITION CHECK
                # =========================
                try:
                    has_position = self.check_positions()
                    self._last_has_position = has_position
                except:
                    has_position = self._last_has_position

                price = self.get_current_price()

                self.price_history.append(price)
                self.volume_history.append(1000.0)

                if len(self.price_history) < 20:
                    print(f"Collecting data... {len(self.price_history)}/20")
                    time.sleep(1)
                    continue

                state = self.get_state(self.price_history[-20:], self.volume_history[-20:])

                dqn_output = self.get_dqn_output(state)
                ppo_output = self.get_ppo_output(state)

                decision = self.ensemble.combine_predictions(dqn_output, ppo_output)
                action = decision["action"]

                # reward
                reward = price - self.price_history[-2] if len(self.price_history) > 1 else 0

                # =========================
                # TRADE LOGIC (UNCHANGED LOGIC)
                # =========================
                if not has_position and action in [0, 1]:
                    self.execute_trade(action, price)

                # =========================
                # FIX 4: next_state simplified
                # =========================
                next_state = state
                done = False

                if training_mode:
                    self.agent.remember(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        ppo_output
                    )
                    self.agent.train_step(batch_size=BATCH_SIZE)

                episode_reward += reward
                step += 1

                if step % 10 == 0:
                    print(f"Step {step} | Price {price:.2f} | Action {action} | Pos {has_position}")

                # checkpoint
                if step % (self.checkpoint_interval * 10) == 0:
                    print("Saving checkpoint...")
                    self.save_models()

                time.sleep(5)

        except KeyboardInterrupt:
            print("\nStopped by user")

        print(f"Done | Reward: {episode_reward}")
        return episode_reward


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bot = XAUUSDHybridTrader(
        use_live_data=True,
        dry_run=args.dry_run
    )

    if bot.trader:
        try:
            bot.trader.enable_streaming(bot.epic)
        except:
            pass

    print("Loading models...")
    bot.load_models()

    try:
        if args.forever:
            bot.live_trading_loop(max_steps=None, training_mode=True)
        else:
            bot.live_trading_loop(max_steps=args.max_steps, training_mode=True)

        bot.save_models()

    except KeyboardInterrupt:
        bot.save_models()
        print("Stopped safely")

    except Exception as e:
        print(f"Error: {e}")
        bot.save_models()