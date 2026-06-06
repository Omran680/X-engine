"""IG Markets API client — wraps trading_ig with rate-limiting and retries."""

from trading_ig import IGService
import os
import time
import traceback
from dotenv import load_dotenv
from trade_bot.core.logging import get_logger
from trade_bot.core.config import (
    API_RATE_LIMIT_INTERVAL,
    API_CACHE_TTL,
    API_BACKOFF_RATE_LIMIT,
    API_BACKOFF_BASE,
)

load_dotenv()

USERNAME = os.getenv("IG_USERNAME")
PASSWORD = os.getenv("IG_PASSWORD")
API_KEY  = os.getenv("IG_API_KEY")

logger = get_logger(__name__)


class Trader:
    def __init__(self):
        self.ig = IGService(USERNAME, PASSWORD, API_KEY, acc_type="DEMO")
        self.ig.create_session()

        self._stream = None
        self._last_price: float | None = None
        self._use_stream = False

        self._last_api_call: float = 0
        self._min_api_interval: float = API_RATE_LIMIT_INTERVAL
        self._price_cache: dict = {}
        self._price_cache_ttl: float = API_CACHE_TTL

    def enable_streaming(self, epic: str | None = None) -> None:
        try:
            from trading_ig import StreamingClient

            def _on_price(data):
                try:
                    if data and "snapshot" in data and "bid" in data["snapshot"]:
                        self._last_price = float(data["snapshot"]["bid"])
                except Exception:
                    pass

            self._stream = StreamingClient(USERNAME, PASSWORD, API_KEY, acc_type="DEMO")
            self._stream.connect()

            if epic is not None:
                for method in ("subscribe_epic", "subscribe"):
                    try:
                        getattr(self._stream, method)(epic)
                        break
                    except AttributeError:
                        continue
                    except Exception:
                        break

            try:
                self._stream.on_price = _on_price
            except AttributeError:
                pass

            self._use_stream = True
            logger.info("Streaming enabled for epic: %s", epic)

        except Exception as e:
            logger.warning("Streaming not available: %s", e)
            self._use_stream = False

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_api_interval:
            time.sleep(self._min_api_interval - elapsed)

    def get_price(self, epic: str) -> float:
        now = time.time()
        cached = self._price_cache.get(epic)
        if cached and (now - cached[1]) < self._price_cache_ttl:
            return cached[0]

        if self._use_stream and self._last_price is not None:
            return self._last_price

        self._rate_limit()

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._last_api_call = time.time()
                data = self.ig.fetch_market_by_epic(epic)
                if not data or "snapshot" not in data or "bid" not in data["snapshot"]:
                    raise ValueError(f"Invalid market data structure: {data}")
                price = float(data["snapshot"]["bid"])
                self._last_price = price
                self._price_cache[epic] = (price, time.time())
                return price
            except Exception as e:
                last_exc = e
                logger.warning("Price fetch attempt %d/3 failed: %s", attempt, repr(e))
                if type(e).__name__ == "ApiExceededException":
                    time.sleep(API_BACKOFF_RATE_LIMIT * attempt)
                else:
                    time.sleep(API_BACKOFF_BASE * attempt)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Price fetch failed after all retries")

    def get_account_balance(self):
        self._rate_limit()
        self._last_api_call = time.time()
        return self.ig.fetch_accounts()

    def get_positions(self):
        self._rate_limit()
        try:
            self._last_api_call = time.time()
            return self.ig.fetch_open_positions()
        except Exception as e:
            logger.error("Error fetching positions: %s", repr(e))
            traceback.print_exc()
            return []

    def close_position(self, deal_id: str):
        return self.ig.close_open_position(deal_id=deal_id)

    def open_trade(self, epic: str, direction: str, size: float, sl=None, tp=None):
        try:
            result = self.ig.create_open_position(
                currency_code="USD",
                direction=direction,
                epic=epic,
                expiry="-",
                force_open=True,
                guaranteed_stop=False,
                level=None,
                limit_distance=tp,
                limit_level=None,
                order_type="MARKET",
                quote_id=None,
                size=size,
                stop_distance=sl,
                stop_level=None,
                trailing_stop=False,
                trailing_stop_increment=None,
                time_in_force="FILL_OR_KILL",
            )
            return result
        except Exception as e:
            logger.error("Error opening trade %s %s @ size %.2f: %s", direction, epic, size, e)
            raise
