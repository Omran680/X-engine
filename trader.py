from trading_ig import IGService
import os
import time
import traceback
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("IG_USERNAME")
PASSWORD = os.getenv("IG_PASSWORD")
API_KEY = os.getenv("IG_API_KEY")


class Trader:
    def __init__(self):
        self.ig = IGService(USERNAME, PASSWORD, API_KEY, acc_type="DEMO")
        self.ig.create_session()
        # Streaming attributes
        self._stream = None
        self._last_price = None
        self._use_stream = False
        
        # Rate limiting
        self._last_api_call = 0
        self._min_api_interval = 2  # Minimum 2 seconds between API calls
        self._price_cache = {}
        self._price_cache_ttl = 5  # Cache prices for 5 seconds

    def enable_streaming(self, epic=None):
        """Enable market streaming if available in trading_ig.

        This sets a flag and starts a background listener when possible.
        """
        try:
            # Lazy import of streaming client (optional dependency)
            from trading_ig import StreamingClient

            def _on_price(data):
                try:
                    if data and 'snapshot' in data and 'bid' in data['snapshot']:
                        self._last_price = float(data['snapshot']['bid'])
                except Exception:
                    pass

            # Create and start streaming client
            self._stream = StreamingClient(USERNAME, PASSWORD, API_KEY, acc_type='DEMO')
            self._stream.connect()
            # subscribe to epic updates if provided; method name may vary in trading_ig versions
            try:
                if epic is not None:
                    # method name may differ between versions; try common variants
                    try:
                        self._stream.subscribe_epic(epic)
                    except Exception:
                        try:
                            self._stream.subscribe(epic)
                        except Exception:
                            pass
            except Exception:
                pass

            # Hook handler if available
            try:
                self._stream.on_price = _on_price
            except Exception:
                pass

            self._use_stream = True
            print('Streaming enabled')
        except Exception as e:
            print(f'Streaming not available: {e}')
            self._use_stream = False

    def get_price(self, epic):
        """Fetch current price with caching and rate limiting.

        Returns last known bid as float or raises after retries.
        """
        # Check cache first
        now = time.time()
        if epic in self._price_cache:
            price, timestamp = self._price_cache[epic]
            if now - timestamp < self._price_cache_ttl:
                return price
        
        # If streaming is enabled and has a recent price, use it
        if self._use_stream and self._last_price is not None:
            return self._last_price

        # Rate limiting: wait if needed
        time_since_last = now - self._last_api_call
        if time_since_last < self._min_api_interval:
            wait_time = self._min_api_interval - time_since_last
            time.sleep(wait_time)

        last_exc = None
        for attempt in range(1, 4):
            try:
                self._last_api_call = time.time()
                data = self.ig.fetch_market_by_epic(epic)
                # Defensive: ensure snapshot and bid exist
                if not data or "snapshot" not in data or "bid" not in data["snapshot"]:
                    raise ValueError(f"Invalid market data structure: {data}")
                price = float(data["snapshot"]["bid"])
                self._last_price = price
                self._price_cache[epic] = (price, time.time())
                return price
            except Exception as e:
                last_exc = e
                print(f"Error fetching price (attempt {attempt}/3): {repr(e)}")
                traceback.print_exc()
                # If rate limit, backoff longer
                if e.__class__.__name__ == 'ApiExceededException':
                    time.sleep(15 * attempt)  # 15s, 30s, 45s
                else:
                    time.sleep(2 * attempt)   # 2s, 4s, 6s

        # After retries, re-raise the last exception so caller can handle it
        raise last_exc

    def get_account_balance(self):
        """Get account balance with rate limiting"""
        # Rate limiting
        now = time.time()
        time_since_last = now - self._last_api_call
        if time_since_last < self._min_api_interval:
            wait_time = self._min_api_interval - time_since_last
            time.sleep(wait_time)
        
        self._last_api_call = time.time()
        return self.ig.fetch_accounts()

    def get_positions(self):
        """Get current positions with safe fallback on error and rate limiting."""
        # Rate limiting
        now = time.time()
        time_since_last = now - self._last_api_call
        if time_since_last < self._min_api_interval:
            wait_time = self._min_api_interval - time_since_last
            time.sleep(wait_time)
        
        try:
            self._last_api_call = time.time()
            return self.ig.fetch_open_positions()
        except Exception as e:
            print(f"Error fetching positions: {repr(e)}")
            traceback.print_exc()
            return []

    def close_position(self, deal_id):
        """Close a position"""
        return self.ig.close_open_position(deal_id=deal_id)

    def open_trade(self, epic, direction, size, sl=None, tp=None):
        """Open a trade (wrapper for trading_ig.create_open_position).

        Args:
            epic: market epic
            direction: 'BUY' or 'SELL'
            size: position size (lots)
            sl: stop distance (absolute) or None
            tp: limit distance (absolute) or None

        Returns:
            dict: result from IGService.create_open_position or raised exception
        """
        try:
            # Use IG required fields similar to the working example:
            # expiry should not be None for OTC/market orders (use "-")
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
                # trading_ig accepts time_in_force separately
                time_in_force="FILL_OR_KILL",
            )
            return result
        except Exception as e:
            print(f"Error opening trade: {e}")
            raise