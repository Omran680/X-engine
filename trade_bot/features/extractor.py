import numpy as np
import pandas as pd
from typing import Tuple, List

class FeatureExtractor:
    """Extracteur de features pour le trading XAU/USD"""
    
    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window

    def extract_features(self, price_history: List[float], volume_history: List[float] = None) -> np.ndarray:
        """
        Extract trading features from price history
        
        Returns:
            np.ndarray of shape (1, feature_size) with normalized features
        """
        if len(price_history) < self.lookback_window:
            return self._get_empty_features()
        
        prices = np.array(price_history[-self.lookback_window:])
        
        features = []
        
        # 1. Price momentum indicators
        features.extend(self._get_momentum(prices))
        
        # 2. Volatility indicators
        features.extend(self._get_volatility(prices))
        
        # 3. Trend indicators
        features.extend(self._get_trend(prices))
        
        # 4. Mean reversion indicators
        features.extend(self._get_mean_reversion(prices))
        
        # 5. Volume indicators (if available)
        if volume_history is not None and len(volume_history) >= self.lookback_window:
            volume = np.array(volume_history[-self.lookback_window:])
            features.extend(self._get_volume_features(volume))
        
        # Normalize features
        features = np.array(features, dtype=np.float32)
        features = self._normalize_features(features)
        
        return features.reshape(1, -1)

    def _get_momentum(self, prices: np.ndarray) -> List[float]:
        """Calculate momentum indicators"""
        features = []
        
        # Rate of change
        roc = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
        features.append(roc)
        
        # Simple moving average (SMA)
        sma_short = np.mean(prices[-5:])
        sma_long = np.mean(prices)
        features.append(prices[-1] - sma_short)
        features.append(sma_short - sma_long)
        
        # Exponential moving average (EMA)
        ema = self._calculate_ema(prices, 5)
        features.append(prices[-1] - ema)
        
        # Momentum (price change over period)
        momentum = prices[-1] - prices[-5] if len(prices) >= 5 else 0
        features.append(momentum)
        
        return features

    def _get_volatility(self, prices: np.ndarray) -> List[float]:
        """Calculate volatility indicators"""
        features = []
        
        # Standard deviation
        volatility = np.std(prices)
        features.append(volatility)
        
        # ATR-like: Average True Range (simplified)
        returns = np.diff(prices)
        atr = np.mean(np.abs(returns))
        features.append(atr)
        
        # Bollinger Bands
        sma = np.mean(prices)
        std = np.std(prices)
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        bb_position = (prices[-1] - lower_band) / (upper_band - lower_band + 1e-8)
        features.append(bb_position)
        
        # Range
        price_range = (np.max(prices) - np.min(prices)) / (np.mean(prices) + 1e-8)
        features.append(price_range)
        
        return features

    def _get_trend(self, prices: np.ndarray) -> List[float]:
        """Calculate trend indicators"""
        features = []
        
        # Linear regression slope
        x = np.arange(len(prices))
        z = np.polyfit(x, prices, 1)
        slope = z[0]
        features.append(slope)
        
        # Higher highs / Lower lows
        recent_high = np.max(prices[-5:]) if len(prices) >= 5 else prices[-1]
        recent_low = np.min(prices[-5:]) if len(prices) >= 5 else prices[-1]
        older_high = np.max(prices[:-5]) if len(prices) > 5 else prices[0]
        older_low = np.min(prices[:-5]) if len(prices) > 5 else prices[0]
        
        higher_highs = 1.0 if recent_high > older_high else -1.0
        lower_lows = 1.0 if recent_low < older_low else -1.0
        features.append(higher_highs)
        features.append(lower_lows)
        
        # Trend strength (ADX-like)
        uptrend = np.sum(np.diff(prices) > 0)
        downtrend = np.sum(np.diff(prices) < 0)
        trend_strength = (uptrend - downtrend) / len(prices)
        features.append(trend_strength)
        
        return features

    def _get_mean_reversion(self, prices: np.ndarray) -> List[float]:
        """Calculate mean reversion indicators"""
        features = []
        
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(prices)
        features.append(rsi)
        
        # Distance to SMA
        sma = np.mean(prices)
        distance_to_mean = (prices[-1] - sma) / (sma + 1e-8)
        features.append(distance_to_mean)
        
        # Stochastic oscillator
        stoch = self._calculate_stochastic(prices)
        features.append(stoch)
        
        return features

    def _get_volume_features(self, volume: np.ndarray) -> List[float]:
        """Calculate volume-based features"""
        features = []
        
        # Volume trend
        vol_sma = np.mean(volume)
        vol_trend = volume[-1] / (vol_sma + 1e-8)
        features.append(vol_trend)
        
        # Volume volatility
        vol_volatility = np.std(volume)
        features.append(vol_volatility)
        
        return features

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        
        return ema

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return 50.0
        
        recent_prices = prices[-period:]
        highest_high = np.max(recent_prices)
        lowest_low = np.min(recent_prices)
        
        if highest_high == lowest_low:
            return 50.0
        
        stoch = 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low)
        return stoch

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range"""
        # Clip extreme values
        features = np.clip(features, -5, 5)
        
        # Normalize
        normalized = features / (np.abs(features).max() + 1e-8)
        
        return normalized

    def _get_empty_features(self) -> np.ndarray:
        """Return empty features when insufficient data"""
        # Total: 5 (momentum) + 4 (volatility) + 4 (trend) + 3 (mean reversion) + 2 (volume) = 18
        return np.zeros((1, 18), dtype=np.float32)


class TradingState:
    """Manage trading state for XAU/USD"""
    
    def __init__(self):
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.entry_time = 0
        self.max_profit = 0
        self.max_loss = 0

    def update(self, current_price: float, action: int):
        """Update state based on action"""
        # action: 0=BUY, 1=SELL, 2=HOLD
        
        if action == 0 and self.position == 0:  # Open long
            self.position = 1
            self.entry_price = current_price
            self.entry_time = 0
            self.max_profit = 0
            self.max_loss = 0
            
        elif action == 1 and self.position == 0:  # Open short
            self.position = -1
            self.entry_price = current_price
            self.entry_time = 0
            self.max_profit = 0
            self.max_loss = 0
            
        elif action == 2:
            # HOLD: do not change position
            pass
        
        # Update max profit/loss
        if self.position != 0:
            pnl = (current_price - self.entry_price) * self.position
            if pnl > self.max_profit:
                self.max_profit = pnl
            if pnl < self.max_loss:
                self.max_loss = pnl
            self.entry_time += 1

    def calculate_reward(self, current_price: float, last_price: float, sl_pct: float = 2.0, tp_pct: float = 3.0) -> float:
        """Calculate reward based on current state"""
        if self.position == 0:
            return 0
        
        # Unrealized PnL
        pnl = (current_price - self.entry_price) * self.position
        pnl_pct = (pnl / self.entry_price) * 100
        
        # Check stop loss / take profit
        if pnl_pct <= -sl_pct:
            return -1.0 + (pnl_pct / sl_pct)  # Penalty
        if pnl_pct >= tp_pct:
            return 1.0 + (pnl_pct / tp_pct) * 0.1  # Bonus
        
        # Small reward for price movement
        price_change = (current_price - last_price) * self.position
        return price_change / self.entry_price

    def get_state_dict(self) -> dict:
        """Return current state as dictionary"""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss
        }
