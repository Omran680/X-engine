"""IG Markets API client — wraps trading_ig with rate-limiting, retries and a lock.

The same Trader instance is shared by the trading loop (main thread) and the MCP
server (background thread), so every exchange call goes through ``self._lock``.
"""

import os
import threading
import time

from dotenv import load_dotenv
from trading_ig import IGService

from trade_bot.core.config import (
    API_BACKOFF_BASE,
    API_BACKOFF_RATE_LIMIT,
    API_CACHE_TTL,
    API_RATE_LIMIT_INTERVAL,
    IG_ACC_TYPE,
    MAX_ORDER_SIZE,
)
from trade_bot.core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


class ConfigError(RuntimeError):
    """Missing/invalid configuration — retrying will not help."""


class MarketClosedError(RuntimeError):
    """Market is not tradeable (weekend, daily break, halt). Expected, not a failure."""

    def __init__(self, status: str):
        super().__init__(f"market status {status}")
        self.status = status


def _credentials() -> tuple[str, str, str]:
    username, password, api_key = (
        os.getenv("IG_USERNAME"), os.getenv("IG_PASSWORD"), os.getenv("IG_API_KEY"),
    )
    missing = [n for n, v in (("IG_USERNAME", username), ("IG_PASSWORD", password),
                              ("IG_API_KEY", api_key)) if not v]
    if missing:
        raise ConfigError(f"Missing environment variables: {', '.join(missing)} (see .env)")
    return username, password, api_key


class Trader:
    def __init__(self):
        if IG_ACC_TYPE not in ("DEMO", "LIVE"):
            raise ConfigError(f"IG_ACC_TYPE must be DEMO or LIVE, got {IG_ACC_TYPE!r}")
        if IG_ACC_TYPE == "LIVE" and os.getenv("IG_ALLOW_LIVE") != "1":
            raise ConfigError("IG_ACC_TYPE=LIVE requires IG_ALLOW_LIVE=1 (explicit opt-in to real money)")

        username, password, api_key = _credentials()
        self.acc_type = IG_ACC_TYPE
        self._lock = threading.RLock()
        self.ig = IGService(username, password, api_key, acc_type=self.acc_type)
        self.ig.create_session()
        logger.info("IG session created (%s account)", self.acc_type)

        self._last_price: float | None = None
        self._last_api_call: float = 0.0
        self._min_api_interval: float = API_RATE_LIMIT_INTERVAL
        self._price_cache: dict = {}
        self._price_cache_ttl: float = API_CACHE_TTL

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def enable_streaming(self, epic: str | None = None) -> bool:
        """Streaming is not supported by trading_ig's public API (no StreamingClient).

        Kept as a no-op so callers need no special-casing; prices come from REST polling.
        """
        logger.info("Streaming not available — using REST polling with cache")
        return False

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_api_interval:
            time.sleep(self._min_api_interval - elapsed)
        self._last_api_call = time.time()

    @staticmethod
    def _is_session_error(exc: Exception) -> bool:
        text = repr(exc).lower()
        return any(t in text for t in ("invalid.session", "client-token", "401", "security.invalid"))

    def _call(self, fn, *args, **kwargs):
        """Rate-limited, serialised exchange call with one transparent re-login."""
        with self._lock:
            self._rate_limit()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not self._is_session_error(e):
                    raise
                logger.warning("IG session expired (%s) — re-authenticating", repr(e))
                self.ig.create_session()
                self._rate_limit()
                return fn(*args, **kwargs)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_price(self, epic: str) -> float:
        now = time.time()
        cached = self._price_cache.get(epic)
        if cached and (now - cached[1]) < self._price_cache_ttl:
            return cached[0]

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                data = self._call(self.ig.fetch_market_by_epic, epic)
                if not data or "snapshot" not in data or "bid" not in data["snapshot"]:
                    raise ValueError(f"Invalid market data structure: {data}")
                status = str(data["snapshot"].get("marketStatus", "TRADEABLE")).upper()
                if status != "TRADEABLE" or data["snapshot"]["bid"] is None:
                    # Closed ≠ error: no retries, no backoff, the caller idles until it reopens
                    self._price_cache.pop(epic, None)
                    raise MarketClosedError(status if status != "TRADEABLE" else "NO_QUOTE")
                price = float(data["snapshot"]["bid"])
                if not price > 0:
                    raise ValueError(f"Non-positive price {price!r} (market closed?)")
                self._last_price = price
                self._price_cache[epic] = (price, time.time())
                return price
            except MarketClosedError:
                raise
            except Exception as e:
                last_exc = e
                logger.warning("Price fetch attempt %d/3 failed: %s", attempt, repr(e))
                if attempt < 3:
                    backoff = API_BACKOFF_RATE_LIMIT if type(e).__name__ == "ApiExceededException" \
                        else API_BACKOFF_BASE
                    time.sleep(backoff * attempt)

        raise last_exc if last_exc else RuntimeError("Price fetch failed after all retries")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_account_balance(self):
        return self._call(self.ig.fetch_accounts)

    def get_positions(self):
        """Open positions. Raises on API failure — an error is NOT 'no positions'.

        (Swallowing the error used to make the bot believe it was flat and open
        duplicate trades whenever the API hiccuped.)
        """
        return self._call(self.ig.fetch_open_positions)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def open_trade(self, epic: str, direction: str, size: float, sl=None, tp=None) -> dict:
        """Open a market position and return the confirmed deal.

        Result always contains ``dealReference``; when IG confirms it also holds
        ``dealId`` (needed to close the position later) and ``level``.
        Raises if the order is rejected.
        """
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {direction!r}")
        if not 0 < size <= MAX_ORDER_SIZE:
            raise ValueError(f"size {size} outside (0, {MAX_ORDER_SIZE}] lots")

        try:
            result = self._call(
                self.ig.create_open_position,
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
        except Exception as e:
            logger.error("Error opening trade %s %s size=%.2f: %s", direction, epic, size, e)
            raise

        result = dict(result) if result else {}
        ref = result.get("dealReference")
        if ref:
            try:
                confirm = self._call(self.ig.fetch_deal_by_deal_reference, ref)
                status = str(confirm.get("dealStatus", "")).upper()
                if status == "REJECTED":
                    raise RuntimeError(f"Order rejected by IG: {confirm.get('reason')}")
                result.update(dealId=confirm.get("dealId"), dealStatus=status,
                              level=confirm.get("level"))
            except RuntimeError:
                raise
            except Exception as e:
                # Order was sent; losing the confirmation must not hide the position.
                logger.warning("Could not confirm deal %s: %s", ref, e)
        return result

    def close_position(self, deal_id: str):
        """Close an open position by its ``dealId`` (not the dealReference)."""
        pos = self._call(self.ig.fetch_open_position_by_deal_id, deal_id)
        market, position = pos["market"], pos["position"]
        opposite = "SELL" if position["direction"] == "BUY" else "BUY"
        return self._call(
            self.ig.close_open_position,
            deal_id=deal_id,
            direction=opposite,
            epic=market["epic"],
            expiry=market.get("expiry", "-"),
            level=None,
            order_type="MARKET",
            quote_id=None,
            size=position["size"],
        )
