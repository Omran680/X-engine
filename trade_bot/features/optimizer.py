"""
Advanced Environment & State/Reward Design
Real performance lever: rich features + risk-adjusted rewards
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, List


class EnrichedFeatureExtractor:
    """
    Extract rich features for RL agents:
    - Technical indicators (MACD, RSI, BB, ATR)
    - Multi-timeframe analysis (5min, 15min, 1h)
    - Volatility metrics
    - Trend strength
    - Volume analysis
    
    Output: 48 features instead of raw price
    """
    
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)
        self.volume_history = deque(maxlen=lookback)
        self.returns_history = deque(maxlen=lookback)
    
    def add_data(self, price: float, volume: float):
        """Add new price/volume data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) > 1:
            ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(ret)
    
    def _sma(self, period: int) -> float:
        """Simple Moving Average"""
        if len(self.price_history) < period:
            return self.price_history[-1] if self.price_history else 0
        return np.mean(list(self.price_history)[-period:])
    
    def _ema(self, period: int) -> float:
        """Exponential Moving Average (simplified)"""
        if len(self.price_history) < period:
            return self.price_history[-1] if self.price_history else 0
        prices = list(self.price_history)[-period:]
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema
    
    def _rsi(self, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(self.returns_history) < period:
            return 50.0
        
        returns = list(self.returns_history)[-period:]
        gains = [r for r in returns if r > 0]
        losses = [-r for r in returns if r < 0]
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(np.clip(rsi, 0, 100)) / 50 - 1  # Normalize to [-1, 1]
    
    def _macd(self) -> Tuple[float, float]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(self.price_history) < 26:
            return 0.0, 0.0
        
        ema12 = self._ema(12)
        ema26 = self._ema(26)
        macd = ema12 - ema26
        
        # Signal line (EMA of MACD)
        signal = ema12 * 0.7 + ema26 * 0.3  # Simplified
        
        return float(macd / (self.price_history[-1] + 1e-8)), float((macd - signal) / (self.price_history[-1] + 1e-8))
    
    def _bollinger_bands(self, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands"""
        if len(self.price_history) < period:
            return 0.0, 0.0, 0.0
        
        prices = list(self.price_history)[-period:]
        sma = np.mean(prices)
        std = np.std(prices)
        
        middle = sma
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        
        current = self.price_history[-1]
        
        # Normalize to [-1, 1]
        bb_position = (current - lower) / (upper - lower + 1e-8) if (upper - lower) > 0 else 0.5
        bb_width = (upper - lower) / (middle + 1e-8)
        
        return float(bb_position * 2 - 1), float(bb_width), float((current - middle) / (std + 1e-8))
    
    def _atr(self, period: int = 14) -> float:
        """Average True Range (volatility)"""
        if len(self.price_history) < period:
            return 0.0
        
        prices = list(self.price_history)[-period:]
        tr_list = [prices[0]]
        
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            tr_list.append(high_low)
        
        atr = np.mean(tr_list)
        return float(atr / (self.price_history[-1] + 1e-8))
    
    def _momentum(self, period: int = 10) -> float:
        """Price momentum"""
        if len(self.price_history) < period:
            return 0.0
        
        current = self.price_history[-1]
        past = self.price_history[0] if len(self.price_history) >= period else list(self.price_history)[-(period)]
        
        momentum = (current - past) / (past + 1e-8)
        return float(momentum)
    
    def _volume_signal(self) -> float:
        """Volume trend signal"""
        if len(self.volume_history) < 10:
            return 0.0
        
        recent_vol = np.mean(list(self.volume_history)[-5:])
        past_vol = np.mean(list(self.volume_history)[-10:-5])
        
        vol_ratio = recent_vol / (past_vol + 1e-8)
        return float(np.clip((vol_ratio - 1) / 2, -1, 1))  # Normalize
    
    def _volatility(self) -> float:
        """Historical volatility"""
        if len(self.returns_history) < 20:
            return 0.0
        
        recent_returns = list(self.returns_history)[-20:]
        volatility = np.std(recent_returns)
        return float(min(volatility * 10, 1.0))  # Normalized
    
    def extract_48_features(self) -> np.ndarray:
        """
        Extract 48 rich features:
        - Price levels & trends (8)
        - Technical indicators (16)
        - Volatility metrics (8)
        - Volume metrics (8)
        - Multi-timeframe signals (4)
        
        Returns: Array of shape (48,)
        """
        
        features = []
        
        # 1. Price levels (8 features)
        current_price = self.price_history[-1] if self.price_history else 0
        
        # Normalize prices to [-1, 1]
        min_price = min(self.price_history) if self.price_history else current_price
        max_price = max(self.price_history) if self.price_history else current_price
        price_range = max_price - min_price + 1e-8
        
        features.append((current_price - min_price) / price_range * 2 - 1)  # Normalized price
        features.append(self._sma(5) / (current_price + 1e-8) - 1)  # SMA5 vs price
        features.append(self._sma(20) / (current_price + 1e-8) - 1)  # SMA20
        features.append(self._sma(50) / (current_price + 1e-8) - 1)  # SMA50
        features.append(self._ema(12) / (current_price + 1e-8) - 1)  # EMA12
        features.append(self._ema(26) / (current_price + 1e-8) - 1)  # EMA26
        features.append(self._momentum(10))  # 10-period momentum
        features.append(float(len(self.price_history)) / self.lookback)  # Buffer fill ratio
        
        # 2. Technical indicators (16 features)
        # RSI
        features.append(self._rsi(14))
        features.append(self._rsi(7))
        
        # MACD
        macd, signal = self._macd()
        features.append(macd)
        features.append(signal)
        
        # Bollinger Bands
        bb_pos, bb_width, bb_deviation = self._bollinger_bands()
        features.append(bb_pos)
        features.append(bb_width / 10)  # Normalize
        features.append(bb_deviation)
        
        # Stochastic-like (simplified)
        features.append(bb_pos)  # Reuse as stochastic proxy
        
        # ATR
        features.append(self._atr(14))
        features.append(self._atr(7))
        
        # CCI-like (commodity channel index proxy)
        features.append(self._momentum(5))
        
        # ADX-like trend strength
        if len(self.returns_history) >= 14:
            uptrend = sum(1 for r in list(self.returns_history)[-14:] if r > 0) / 14
            features.append(uptrend * 2 - 1)  # Normalize to [-1, 1]
        else:
            features.append(0.0)
        
        # Additional indicators to reach 16 total
        # Williams %R (oscillator similar to RSI)
        features.append(self._momentum(3))
        
        # Rate of Change (ROC)
        features.append(self._momentum(12))
        
        # Price relative to EMA bands
        if len(self.price_history) >= 20:
            ema = self._ema(12)
            current = self.price_history[-1]
            features.append(float(np.clip((current - ema) / (current + 1e-8), -1, 1)))
        else:
            features.append(0.0)
        
        # Price change acceleration
        if len(self.returns_history) >= 3:
            recent_momentum = self._momentum(3)
            older_momentum = self._momentum(5)
            acceleration = recent_momentum - older_momentum
            features.append(float(np.clip(acceleration / 2, -1, 1)))
        else:
            features.append(0.0)
        
        # 3. Volatility metrics (8 features)
        features.append(self._volatility())
        features.append(self._volatility() * self._momentum(10))  # Vol * momentum interaction
        
        if len(self.returns_history) >= 30:
            vol_30 = np.std(list(self.returns_history)[-30:])
            vol_10 = np.std(list(self.returns_history)[-10:])
            features.append(min((vol_10 - vol_30) / (vol_30 + 1e-8), 1.0))  # Vol expansion/contraction
        else:
            features.append(0.0)
        
        # Return skewness
        if len(self.returns_history) >= 10:
            returns = list(self.returns_history)[-10:]
            skewness = (np.mean([r**3 for r in returns]) / ((np.std(returns)**3) + 1e-8))
            features.append(float(np.clip(skewness, -1, 1)))
        else:
            features.append(0.0)
        
        # Return kurtosis
        if len(self.returns_history) >= 10:
            returns = list(self.returns_history)[-10:]
            kurtosis = (np.mean([r**4 for r in returns]) / ((np.std(returns)**4) + 1e-8))
            features.append(float(np.clip(kurtosis / 5, 0, 1)))
        else:
            features.append(0.0)
        
        # Sharpe-like metric (return/vol)
        if len(self.returns_history) >= 20:
            recent_returns = list(self.returns_history)[-20:]
            sharpe_metric = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
            features.append(float(np.clip(sharpe_metric, -1, 1)))
        else:
            features.append(0.0)
        
        # Max drawdown metric
        if len(self.price_history) >= 20:
            recent_prices = list(self.price_history)[-20:]
            peak = np.max(recent_prices)
            trough = np.min(recent_prices)
            dd = (trough - peak) / peak
            features.append(float(dd))
        else:
            features.append(0.0)
        
        # Additional features to reach 8
        # Volatility ratio (short-term vs long-term)
        if len(self.returns_history) >= 30:
            short_vol = np.std(list(self.returns_history)[-10:])
            long_vol = np.std(list(self.returns_history)[-30:])
            vol_ratio = short_vol / (long_vol + 1e-8)
            features.append(float(np.clip(vol_ratio, 0, 2)))
        else:
            features.append(0.0)
        
        # Extreme moves (large single moves)
        if len(self.returns_history) >= 10:
            extreme_moves = sum(1 for r in list(self.returns_history)[-10:] if abs(r) > 0.02)
            features.append(extreme_moves / 10)
        else:
            features.append(0.0)
        
        # Mean reversion tendency
        if len(self.returns_history) >= 10:
            rets = list(self.returns_history)[-10:]
            reversals = sum(1 for i in range(1, len(rets)) if rets[i] * rets[i-1] < 0)
            features.append(reversals / 9)
        else:
            features.append(0.0)
        
        # 4. Volume metrics (8 features)
        features.append(self._volume_signal())
        
        if len(self.volume_history) >= 20:
            recent_vol = np.mean(list(self.volume_history)[-5:])
            past_vol = np.mean(list(self.volume_history)[-20:-5])
            vol_sma_ratio = recent_vol / (past_vol + 1e-8)
            features.append(float(np.clip((vol_sma_ratio - 1) / 2, -1, 1)))
        else:
            features.append(0.0)
        
        # Volume trend
        if len(self.volume_history) >= 10:
            vol_trend = np.polyfit(range(10), list(self.volume_history)[-10:], 1)[0]
            features.append(float(np.clip(vol_trend / (np.mean(list(self.volume_history)[-10:]) + 1e-8), -1, 1)))
        else:
            features.append(0.0)
        
        # Volume-price correlation
        if len(self.returns_history) >= 10:
            recent_returns = list(self.returns_history)[-10:]
            recent_vols = list(self.volume_history)[-10:]
            correlation = np.corrcoef(recent_returns, recent_vols)[0, 1]
            features.append(float(np.nan_to_num(correlation, 0.0)))
        else:
            features.append(0.0)
        
        # OBV-like (on-balance volume signal)
        features.append(self._volume_signal() * np.mean(list(self.returns_history)[-5:]) if len(self.returns_history) >= 5 else 0.0)
        
        # Volume momentum
        if len(self.volume_history) >= 10:
            vol_mom = (np.mean(list(self.volume_history)[-5:]) - np.mean(list(self.volume_history)[-10:-5])) / (np.mean(list(self.volume_history)[-10:-5]) + 1e-8)
            features.append(float(np.clip(vol_mom / 2, -1, 1)))
        else:
            features.append(0.0)
        
        # Volume volatility (std of volume)
        if len(self.volume_history) >= 10:
            vol_std = np.std(list(self.volume_history)[-10:])
            vol_mean = np.mean(list(self.volume_history)[-10:])
            features.append(float(np.clip(vol_std / (vol_mean + 1e-8), 0, 2)))
        else:
            features.append(0.0)
        
        # Price-volume divergence
        if len(self.returns_history) >= 5 and len(self.volume_history) >= 5:
            price_change = np.mean(list(self.returns_history)[-5:])
            vol_change = (np.mean(list(self.volume_history)[-5:]) - np.mean(list(self.volume_history)[-10:-5])) / (np.mean(list(self.volume_history)[-10:-5]) + 1e-8)
            divergence = price_change - vol_change if vol_change != 0 else 0
            features.append(float(np.clip(divergence / 2, -1, 1)))
        else:
            features.append(0.0)
        
        # Market strength (price change / volume change ratio)
        if len(self.returns_history) >= 5:
            price_vol = np.mean(np.abs(list(self.returns_history)[-5:]))
            if len(self.volume_history) >= 5:
                vol_norm = np.mean(list(self.volume_history)[-5:]) / (np.max(list(self.volume_history)) + 1e-8)
                strength = price_vol / (vol_norm + 1e-8)
                features.append(float(np.clip(strength / 10, -1, 1)))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 5. Multi-timeframe signals (5 features)
        # Trend alignment (are all timeframes aligned?)
        sma_5_10_align = 1 if self._sma(5) > self._sma(10) else -1
        sma_20_50_align = 1 if self._sma(20) > self._sma(50) else -1
        ema_12_26_align = 1 if self._ema(12) > self._ema(26) else -1
        all_aligned = (sma_5_10_align + sma_20_50_align + ema_12_26_align) / 3
        
        features.append(float(all_aligned))
        features.append(float(sma_5_10_align))
        features.append(float(sma_20_50_align))
        features.append(float(ema_12_26_align))
        
        # Multi-timeframe volatility consensus
        if len(self.returns_history) >= 20:
            vol_ratio = self._volatility() / (1.0 + 1e-8)
            features.append(float(np.clip(vol_ratio, 0, 1)))
        else:
            features.append(0.0)
        
        # Return as numpy array
        assert len(features) == 48, f"Expected 48 features, got {len(features)}"
        return np.array(features, dtype=np.float32)
    
    def extract_18_features(self) -> np.ndarray:
        """Legacy 18-feature extraction (subset of 48)"""
        features_48 = self.extract_48_features()
        # Select 18 most important features
        indices = [0, 1, 2, 3, 8, 9, 10, 11, 12, 14, 18, 22, 24, 26, 28, 40, 41, 42]
        return features_48[indices]


class RiskAdjustedRewardCalculator:
    """
    Calculate risk-adjusted rewards instead of simple profit
    
    Reward = α*Profit - β*Drawdown - γ*Volatility + δ*WashedoutPenalty
    """
    
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.1):
        self.alpha = alpha  # Profit weight
        self.beta = beta    # Drawdown penalty
        self.gamma = gamma  # Volatility penalty
        
        self.equity_history = deque(maxlen=100)
        self.position_returns = []
        self.max_equity = 1.0
    
    def add_equity_update(self, equity: float):
        """Update equity tracking"""
        self.equity_history.append(equity)
        if len(self.equity_history) > 1:
            ret = (equity - self.equity_history[-2]) / self.equity_history[-2]
            self.position_returns.append(ret)
    
    def calculate_reward(self, 
                        trade_pnl: float,
                        entry_volatility: float,
                        current_drawdown: float,
                        position_holding_time: int,
                        is_winning_trade: bool) -> float:
        """
        Calculate risk-adjusted reward
        
        Args:
            trade_pnl: Profit/loss from trade
            entry_volatility: Market volatility at entry
            current_drawdown: Current drawdown (negative)
            position_holding_time: Number of steps held
            is_winning_trade: Whether trade is profitable
        
        Returns:
            Risk-adjusted reward
        """
        
        # Base profit reward
        profit_reward = self.alpha * trade_pnl
        
        # Drawdown penalty (penalize being in drawdown)
        drawdown_penalty = self.beta * abs(current_drawdown)
        
        # Volatility penalty (penalize high volatility trades)
        volatility_penalty = self.gamma * entry_volatility
        
        # Holding time bonus (encourage quick profitable exits)
        holding_time_factor = 1.0 / (1.0 + position_holding_time * 0.01)
        
        # Washout penalty (penalize frequent losing trades)
        recent_losses = sum(1 for r in self.position_returns[-10:] if r < -0.01)
        washout_penalty = 0.1 * recent_losses if recent_losses > 3 else 0.0
        
        # Combine
        reward = profit_reward - drawdown_penalty - volatility_penalty - washout_penalty
        reward *= holding_time_factor
        
        # Clip to reasonable range
        reward = float(np.clip(reward, -0.1, 0.1))
        
        return reward
    
    def calculate_batch_reward(self, returns_array: np.ndarray) -> float:
        """Calculate reward for batch of returns (Sharpe-like)"""
        if len(returns_array) == 0:
            return 0.0
        
        mean_ret = np.mean(returns_array)
        std_ret = np.std(returns_array)
        
        if std_ret < 1e-8:
            return float(mean_ret)
        
        # Sharpe-like ratio
        sharpe_like = mean_ret / std_ret
        return float(np.clip(sharpe_like, -1.0, 1.0))
