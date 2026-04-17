"""
Market Regime Detection & Curriculum Learning
- Detect bull/bear/sideways regimes
- Enable phase-specific training
- Progressive difficulty curriculum
"""

import numpy as np
from collections import deque
from enum import Enum
from typing import Dict, List


class MarketRegime(Enum):
    """Market regime types"""
    BULL = 1          # Uptrend, low volatility
    BEAR = -1         # Downtrend, low volatility
    SIDEWAYS = 0      # Range-bound, any vola
    VOLATILE_UP = 2   # Uptrend, high volatility
    VOLATILE_DOWN = -2  # Downtrend, high volatility


class MarketRegimeDetector:
    """
    Detect current market regime:
    - Trend direction (ADX, slope)
    - Volatility (ATR, std)
    - Support/resistance levels
    
    Output: Regime type + confidence
    """
    
    def __init__(self, window=30):
        self.window = window
        self.price_history = deque(maxlen=window)
        self.returns_history = deque(maxlen=window)
        self.regime_history = deque(maxlen=100)
    
    def add_price(self, price: float):
        """Add price data"""
        self.price_history.append(price)
        
        if len(self.price_history) > 1:
            ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(ret)
    
    def _detect_trend(self) -> tuple:
        """Detect trend direction and strength (ADX-like)"""
        if len(self.price_history) < 10:
            return 0, 0.5
        
        prices = list(self.price_history)
        
        # Use polyfit to get trend slope
        x = np.arange(len(prices))
        z = np.polyfit(x, prices, 1)
        trend_slope = z[0] / (prices[-1] + 1e-8)
        
        # Calculate ADX-like strength
        ups = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        downs = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        
        trend_strength = abs(ups - downs) / len(prices)
        
        trend_direction = 1 if trend_slope > 0 else -1 if trend_slope < 0 else 0
        
        return trend_direction, trend_strength
    
    def _detect_volatility(self) -> float:
        """Calculate volatility level"""
        if len(self.returns_history) < 10:
            return 0.2
        
        returns = list(self.returns_history)
        volatility = np.std(returns)
        
        # Classify as low/high
        # Normalized to [0, 1]
        return min(volatility * 20, 1.0)
    
    def _detect_support_resistance(self) -> tuple:
        """Find support and resistance levels"""
        if len(self.price_history) < 10:
            return min(self.price_history), max(self.price_history)
        
        prices = list(self.price_history)
        support = np.percentile(prices, 25)
        resistance = np.percentile(prices, 75)
        
        return float(support), float(resistance)
    
    def _calculate_range_ratio(self) -> float:
        """Current price position in recent range"""
        if len(self.price_history) < 5:
            return 0.5
        
        recent = list(self.price_history)[-5:]
        low = min(recent)
        high = max(recent)
        current = recent[-1]
        
        if high - low < 1e-8:
            return 0.5
        
        position = (current - low) / (high - low)
        return float(position)
    
    def detect_regime(self) -> Dict:
        """Detect current regime"""
        trend_dir, trend_strength = self._detect_trend()
        volatility = self._detect_volatility()
        sup, res = self._detect_support_resistance()
        range_pos = self._calculate_range_ratio()
        
        # Decide regime based on trend and volatility
        if trend_strength < 0.3:
            # No clear trend
            regime = MarketRegime.SIDEWAYS
        elif trend_dir > 0:
            if volatility > 0.6:
                regime = MarketRegime.VOLATILE_UP
            else:
                regime = MarketRegime.BULL
        else:
            if volatility > 0.6:
                regime = MarketRegime.VOLATILE_DOWN
            else:
                regime = MarketRegime.BEAR
        
        self.regime_history.append(regime.value)
        
        return {
            'regime': regime,
            'regime_name': regime.name,
            'trend_direction': trend_dir,
            'trend_strength': float(trend_strength),
            'volatility': float(volatility),
            'support': float(sup),
            'resistance': float(res),
            'range_position': float(range_pos),
            'confidence': 0.5 + 0.5 * trend_strength
        }
    
    def get_regime_history(self, last_n=20) -> List[int]:
        """Get recent regime history"""
        return list(self.regime_history)[-last_n:]


class DifficultyScheduler:
    """
    Curriculum learning: progressively increase market difficulty
    
    Phases:
    1. EASY: Bull market, low volatility (teach profitable trades)
    2. MEDIUM: Mixed trends, moderate volatility
    3. HARD: Bear market, high volatility (teach risk management)
    4. EXPERT: All conditions, market stress (teach robustness)
    """
    
    class Difficulty(Enum):
        EASY = 1
        MEDIUM = 2
        HARD = 3
        EXPERT = 4
    
    def __init__(self, total_episodes=1000):
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.current_difficulty = self.Difficulty.EASY
        
        # Thresholds for difficulty progression
        self.easy_threshold = int(total_episodes * 0.15)      # ep 0-150
        self.medium_threshold = int(total_episodes * 0.40)    # ep 150-400
        self.hard_threshold = int(total_episodes * 0.75)      # ep 400-750
        # expert: 750+
    
    def update(self, episode: int):
        """Update difficulty based on episode number"""
        self.current_episode = episode
        
        if episode < self.easy_threshold:
            self.current_difficulty = self.Difficulty.EASY
        elif episode < self.medium_threshold:
            self.current_difficulty = self.Difficulty.MEDIUM
        elif episode < self.hard_threshold:
            self.current_difficulty = self.Difficulty.HARD
        else:
            self.current_difficulty = self.Difficulty.EXPERT
    
    def get_difficulty_config(self) -> Dict:
        """Get config for current difficulty"""
        configs = {
            self.Difficulty.EASY: {
                'name': 'EASY',
                'target_regime': MarketRegime.BULL,
                'allowed_regimes': [MarketRegime.BULL, MarketRegime.SIDEWAYS],
                'max_volatility': 0.3,
                'reward_scale': 1.0,
                'training_episodes': self.easy_threshold,
                'description': 'Bull market, low volatility - learn to profit'
            },
            self.Difficulty.MEDIUM: {
                'name': 'MEDIUM',
                'target_regime': None,
                'allowed_regimes': [
                    MarketRegime.BULL,
                    MarketRegime.BEAR,
                    MarketRegime.SIDEWAYS
                ],
                'max_volatility': 0.6,
                'reward_scale': 0.8,
                'training_episodes': self.medium_threshold - self.easy_threshold,
                'description': 'Mixed trends - adapt to changes'
            },
            self.Difficulty.HARD: {
                'name': 'HARD',
                'target_regime': None,
                'allowed_regimes': [
                    MarketRegime.BEAR,
                    MarketRegime.VOLATILE_DOWN,
                    MarketRegime.VOLATILE_UP
                ],
                'max_volatility': 0.8,
                'reward_scale': 0.6,
                'training_episodes': self.hard_threshold - self.medium_threshold,
                'description': 'High volatility - learn risk management'
            },
            self.Difficulty.EXPERT: {
                'name': 'EXPERT',
                'target_regime': None,
                'allowed_regimes': [
                    MarketRegime.BULL,
                    MarketRegime.BEAR,
                    MarketRegime.SIDEWAYS,
                    MarketRegime.VOLATILE_UP,
                    MarketRegime.VOLATILE_DOWN
                ],
                'max_volatility': 1.0,
                'reward_scale': 0.5,
                'training_episodes': self.total_episodes - self.hard_threshold,
                'description': 'All conditions - maximum robustness'
            }
        }
        
        return configs[self.current_difficulty]
    
    def get_curriculum_progress(self) -> Dict:
        """Get progress in curriculum"""
        config = self.get_difficulty_config()
        progress = (self.current_episode - self._get_difficulty_start()) / config['training_episodes']
        
        return {
            'current_difficulty': self.current_difficulty.name,
            'episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'progress_in_difficulty': min(progress, 1.0),
            'config': config
        }
    
    def _get_difficulty_start(self) -> int:
        """Get episode where current difficulty starts"""
        if self.current_difficulty == self.Difficulty.EASY:
            return 0
        elif self.current_difficulty == self.Difficulty.MEDIUM:
            return self.easy_threshold
        elif self.current_difficulty == self.Difficulty.HARD:
            return self.medium_threshold
        else:
            return self.hard_threshold


class RegimeAwareTrainer:
    """
    Train agents differently on different regimes
    - DQN specializes in BULL (quick opportunities)
    - PPO specializes in VOLATILE (risk management)
    - Combined on mixed regimes
    """
    
    def __init__(self):
        self.dqn_regime_scores = {}
        self.ppo_regime_scores = {}
        
        for regime in MarketRegime:
            self.dqn_regime_scores[regime] = deque(maxlen=50)
            self.ppo_regime_scores[regime] = deque(maxlen=50)
    
    def log_performance(self, regime: MarketRegime, agent_type: str, score: float):
        """Log agent performance in specific regime"""
        if agent_type == "DQN":
            self.dqn_regime_scores[regime].append(score)
        elif agent_type == "PPO":
            self.ppo_regime_scores[regime].append(score)
    
    def get_regime_specialist(self, regime: MarketRegime) -> str:
        """Recommend which agent is better for this regime"""
        dqn_avg = np.mean(self.dqn_regime_scores[regime]) if self.dqn_regime_scores[regime] else 0
        ppo_avg = np.mean(self.ppo_regime_scores[regime]) if self.ppo_regime_scores[regime] else 0
        
        if dqn_avg > ppo_avg:
            return "DQN"
        else:
            return "PPO"
    
    def get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get ensemble weights optimized for regime"""
        dqn_avg = np.mean(self.dqn_regime_scores[regime]) if self.dqn_regime_scores[regime] else 0.5
        ppo_avg = np.mean(self.ppo_regime_scores[regime]) if self.ppo_regime_scores[regime] else 0.5
        
        # Normalize to weights
        total = abs(dqn_avg) + abs(ppo_avg) + 0.1
        
        dqn_weight = abs(dqn_avg) / total
        ppo_weight = abs(ppo_avg) / total
        
        return {
            'dqn': dqn_weight,
            'ppo': ppo_weight,
            'dqn_score': float(dqn_avg),
            'ppo_score': float(ppo_avg),
            'specialist': self.get_regime_specialist(regime)
        }


class TrainingProgression:
    """
    Overall training progression strategy:
    1. Start easy (bull market)
    2. Add difficulty progressively
    3. Evaluate meta-score
    4. Adapt curriculum based on performance
    """
    
    def __init__(self, total_episodes=1000):
        self.total_episodes = total_episodes
        self.difficulty_scheduler = DifficultyScheduler(total_episodes)
        self.regime_trainer = RegimeAwareTrainer()
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_sharpe = deque(maxlen=100)
        self.episode_win_rate = deque(maxlen=100)
        
        self.training_history = []
    
    def update_episode(self, episode: int, 
                      total_reward: float,
                      sharpe_ratio: float,
                      win_rate: float,
                      regime: MarketRegime,
                      agent_type: str = None):
        """Update training progress for an episode"""
        
        self.difficulty_scheduler.update(episode)
        
        self.episode_rewards.append(total_reward)
        self.episode_sharpe.append(sharpe_ratio)
        self.episode_win_rate.append(win_rate)
        
        if agent_type:
            self.regime_trainer.log_performance(regime, agent_type, sharpe_ratio)
        
        # Log to history
        self.training_history.append({
            'episode': episode,
            'reward': total_reward,
            'sharpe': sharpe_ratio,
            'win_rate': win_rate,
            'regime': regime.name if regime else 'UNKNOWN',
            'difficulty': self.difficulty_scheduler.current_difficulty.name
        })
    
    def should_increase_difficulty(self) -> bool:
        """Decide if difficulty should be increased"""
        if len(self.episode_sharpe) < 20:
            return False
        
        recent_sharpe = np.mean(list(self.episode_sharpe)[-20:])
        
        # If Sharpe > 0.5, can increase difficulty
        return recent_sharpe > 0.5
    
    def get_meta_score(self) -> Dict:
        """Calculate overall training meta-score"""
        if len(self.episode_rewards) == 0:
            return {
                'meta_score': 0.0,
                'avg_reward': 0.0,
                'avg_sharpe': 0.0,
                'avg_win_rate': 0.0,
                'stability': 0.0,
                'progress': 0.0
            }
        
        avg_reward = np.mean(self.episode_rewards)
        avg_sharpe = np.mean(self.episode_sharpe)
        avg_win_rate = np.mean(self.episode_win_rate)
        
        # Stability = 1 - std(returns) / mean(returns)
        reward_std = np.std(self.episode_rewards) if np.mean(self.episode_rewards) != 0 else 1
        stability = 1.0 - (reward_std / (abs(avg_reward) + 1e-8))
        stability = float(np.clip(stability, 0, 1))
        
        # Progress = episode / total
        progress = len(self.training_history) / self.total_episodes
        
        # Meta score = weighted average
        meta_score = 0.4 * avg_sharpe + 0.3 * avg_win_rate + 0.2 * stability + 0.1 * progress
        
        return {
            'meta_score': float(meta_score),
            'avg_reward': float(avg_reward),
            'avg_sharpe': float(avg_sharpe),
            'avg_win_rate': float(avg_win_rate),
            'stability': float(stability),
            'progress': float(progress),
            'difficulty': self.difficulty_scheduler.current_difficulty.name
        }
