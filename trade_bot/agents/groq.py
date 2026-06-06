"""Groq LLM trading agent — fast inference for real-time market analysis."""

import os
import json
import time
from typing import Dict

from trade_bot.core.config import GROQ_CALL_COOLDOWN, GROQ_MAX_RETRIES
from trade_bot.core.logging import get_logger

logger = get_logger(__name__)


class GrokTradeAgent:
    """Groq-powered trading agent integrated into the DQN/PPO ensemble as a 3rd voter."""

    def __init__(self, api_key: str | None = None):
        from groq import Groq
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.client = Groq(api_key=self.api_key)
        self.model  = "gemma2-9b-it"
        self._last_call:   float = 0
        self._call_cooldown: float = GROQ_CALL_COOLDOWN
        self._max_retries:   int   = GROQ_MAX_RETRIES

    def analyze_market(self, price: float, price_history: list, volume_history: list) -> Dict:
        elapsed = time.time() - self._last_call
        if elapsed < self._call_cooldown:
            time.sleep(self._call_cooldown - elapsed)

        prices = price_history[-20:] if price_history else [price]
        sma_5  = sum(prices[-5:])  / 5  if len(prices) >= 5  else price
        sma_10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else price
        sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else price

        recent_change = ((price - prices[0]) / prices[0] * 100) if prices[0] != 0 else 0.0
        trend         = "UP" if price > sma_20 else "DOWN" if price < sma_20 else "FLAT"
        momentum      = "STRONG" if abs(recent_change) > 0.5 else "WEAK"
        volume        = sum(volume_history[-5:]) / 5 if volume_history else 1000
        volume_trend  = "HIGH" if volume > 1500 else "LOW"
        price_range   = max(prices[-10:]) - min(prices[-10:]) if len(prices) >= 10 else 0.0

        prompt = f"""You are a professional gold (XAU/USD) trader. Analyze this market snapshot and decide: BUY, SELL, or HOLD.

MARKET DATA:
- Current Price: ${price:.2f}
- SMA(5): ${sma_5:.2f} | SMA(10): ${sma_10:.2f} | SMA(20): ${sma_20:.2f}
- Trend: {trend} | Momentum: {momentum} ({recent_change:.2f}% change)
- Volume: {volume_trend} | 10-bar Range: ${price_range:.2f}

Respond with VALID JSON only:
{{"action": "BUY" or "SELL" or "HOLD", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}}"""

        for attempt in range(1, self._max_retries + 1):
            try:
                self._last_call = time.time()
                msg  = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model, temperature=0.3, max_tokens=150, top_p=0.9,
                )
                text  = msg.choices[0].message.content.strip()
                start, end = text.find("{"), text.rfind("}") + 1
                if start < 0 or end <= start:
                    return self._default_hold()
                parsed     = json.loads(text[start:end])
                action_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
                action     = action_map.get(str(parsed.get("action", "HOLD")).upper(), 2)
                confidence = float(min(1.0, max(0.0, parsed.get("confidence", 0.5))))
                reasoning  = str(parsed.get("reasoning", "Groq analysis"))
                if action == 0:
                    probs = [confidence, 0.0, 1.0 - confidence]
                elif action == 1:
                    probs = [0.0, confidence, 1.0 - confidence]
                else:
                    probs = [0.0, 0.0, 1.0]
                return {"action": action, "confidence": confidence,
                        "reasoning": reasoning, "probs": probs}
            except json.JSONDecodeError:
                return self._default_hold()
            except Exception as e:
                wait = self._call_cooldown * (2 ** (attempt - 1))
                logger.warning("Groq attempt %d/%d failed: %s — retry in %.1fs",
                               attempt, self._max_retries, e, wait)
                if attempt < self._max_retries:
                    time.sleep(wait)

        logger.error("Groq exhausted all retries — HOLD")
        return self._default_hold()

    def _default_hold(self) -> Dict:
        return {"action": 2, "confidence": 0.0,
                "reasoning": "Groq unavailable", "probs": [0.0, 0.0, 1.0]}
