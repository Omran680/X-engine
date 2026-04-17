"""
Meta Evaluation & Transfer Learning
- Continuous performance evaluation
- Dynamic weight adjustment in ensemble
- Transfer learning between agents
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple
from enum import Enum


class PerformanceMetric(Enum):
    """Performance metrics tracked"""
    SHARPE_RATIO = "sharpe"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_dd"
    PROFIT_FACTOR = "profit_factor"
    CONSISTENCY = "consistency"
    RECOVERY_FACTOR = "recovery"


class MetaEvaluator:
    """
    Continuous evaluation of agent performance
    - Track metrics across sliding windows
    - Detect performance degradation
    - Recommend weight adjustments
    - Identify when to switch agents
    """
    
    def __init__(self, window_size=100, lookback_periods=5):
        self.window_size = window_size  # Trades to evaluate
        self.lookback_periods = lookback_periods  # Historical periods
        
        # DQN metrics
        self.dqn_trades = deque(maxlen=window_size)
        self.dqn_metrics_history = deque(maxlen=lookback_periods)
        
        # PPO metrics
        self.ppo_trades = deque(maxlen=window_size)
        self.ppo_metrics_history = deque(maxlen=lookback_periods)
        
        # Ensemble metrics
        self.ensemble_trades = deque(maxlen=window_size)
        self.ensemble_metrics_history = deque(maxlen=lookback_periods)
    
    def log_trade(self, agent: str, entry_price: float, exit_price: float, 
                  size: float, holding_time: int, confidence: float):
        """Log a completed trade"""
        
        pnl = (exit_price - entry_price) * size
        pnl_pct = (exit_price - entry_price) / entry_price
        
        trade_record = {
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'size': size,
            'holding_time': holding_time,
            'confidence': confidence,
            'is_win': pnl > 0
        }
        
        if agent == "DQN":
            self.dqn_trades.append(trade_record)
        elif agent == "PPO":
            self.ppo_trades.append(trade_record)
        elif agent == "ENSEMBLE":
            self.ensemble_trades.append(trade_record)
    
    def _calculate_metrics(self, trades: deque) -> Dict[str, float]:
        """Calculate all metrics for a trade list"""
        
        if len(trades) == 0:
            return {
                'sharpe': 0.0,
                'win_rate': 0.0,
                'max_dd': 0.0,
                'profit_factor': 0.0,
                'consistency': 0.0,
                'recovery': 0.0,
                'total_pnl': 0.0,
                'avg_trade': 0.0,
                'num_trades': 0
            }
        
        trades_list = list(trades)
        pnls = [t['pnl_pct'] for t in trades_list]
        
        # 1. Sharpe Ratio
        mean_ret = np.mean(pnls)
        std_ret = np.std(pnls)
        sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(252)  # Annualized
        
        # 2. Win Rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0
        
        # 3. Max Drawdown
        cum_pnl = np.cumprod(1 + np.array(pnls))
        peak = np.maximum.accumulate(cum_pnl)
        dd = (cum_pnl - peak) / peak
        max_dd = np.min(dd) if len(dd) > 0 else 0
        
        # 4. Profit Factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        # 5. Consistency (1 - coeff of variation)
        consistency = 1.0 / (1.0 + (std_ret / (abs(mean_ret) + 1e-8)))
        consistency = float(np.clip(consistency, 0, 1))
        
        # 6. Recovery Factor (total return / max drawdown)
        total_ret = np.prod(1 + np.array(pnls)) - 1
        recovery = abs(total_ret / max_dd) if max_dd < 0 else 0
        
        return {
            'sharpe': float(np.clip(sharpe, -5, 5)),
            'win_rate': float(win_rate),
            'max_dd': float(max_dd),
            'profit_factor': float(np.clip(profit_factor, 0, 10)),
            'consistency': float(consistency),
            'recovery': float(np.clip(recovery, 0, 100)),
            'total_pnl': float(total_ret),
            'avg_trade': float(mean_ret),
            'num_trades': len(pnls)
        }
    
    def evaluate_agents(self) -> Dict:
        """Evaluate both agents and ensemble"""
        
        dqn_metrics = self._calculate_metrics(self.dqn_trades)
        ppo_metrics = self._calculate_metrics(self.ppo_trades)
        ensemble_metrics = self._calculate_metrics(self.ensemble_trades)
        
        # Store in history
        self.dqn_metrics_history.append(dqn_metrics)
        self.ppo_metrics_history.append(ppo_metrics)
        self.ensemble_metrics_history.append(ensemble_metrics)
        
        return {
            'dqn': dqn_metrics,
            'ppo': ppo_metrics,
            'ensemble': ensemble_metrics,
            'timestamp': len(self.dqn_metrics_history)
        }
    
    def get_recommended_weights(self) -> Dict[str, float]:
        """
        Recommend ensemble weights based on recent performance
        
        Returns:
            {'dqn_weight': 0.4, 'ppo_weight': 0.6}
        """
        
        if len(self.dqn_metrics_history) == 0:
            return {'dqn_weight': 0.5, 'ppo_weight': 0.5}
        
        recent_dqn_sharpe = np.mean([m['sharpe'] for m in list(self.dqn_metrics_history)[-3:]])
        recent_ppo_sharpe = np.mean([m['sharpe'] for m in list(self.ppo_metrics_history)[-3:]])
        
        # Normalize Sharpe to weights
        total = abs(recent_dqn_sharpe) + abs(recent_ppo_sharpe) + 0.1
        dqn_weight = abs(recent_dqn_sharpe) / total
        ppo_weight = abs(recent_ppo_sharpe) / total
        
        # Ensure neither is below 30% (need both perspectives)
        dqn_weight = np.clip(dqn_weight, 0.3, 0.7)
        ppo_weight = 1.0 - dqn_weight
        
        return {
            'dqn_weight': float(dqn_weight),
            'ppo_weight': float(ppo_weight),
            'dqn_sharpe': float(recent_dqn_sharpe),
            'ppo_sharpe': float(recent_ppo_sharpe)
        }
    
    def detect_performance_degradation(self, agent: str, threshold_sharpe=0.0) -> bool:
        """Detect if agent performance degraded"""
        
        if agent == "DQN":
            history = self.dqn_metrics_history
        else:
            history = self.ppo_metrics_history
        
        if len(history) < 2:
            return False
        
        recent = list(history)[-1]['sharpe']
        previous = np.mean([h['sharpe'] for h in list(history)[:-1]])
        
        # Degradation if recent < previous and below threshold
        return (recent < previous) and (recent < threshold_sharpe)
    
    def should_switch_agents(self) -> Tuple[bool, str]:
        """
        Recommend switching agents if one significantly underperforms
        
        Returns:
            (should_switch, recommended_agent)
        """
        
        if len(self.dqn_metrics_history) < 3 or len(self.ppo_metrics_history) < 3:
            return False, None
        
        recent_dqn = np.mean([m['sharpe'] for m in list(self.dqn_metrics_history)[-3:]])
        recent_ppo = np.mean([m['sharpe'] for m in list(self.ppo_metrics_history)[-3:]])
        
        # If one is significantly better, switch
        if abs(recent_dqn - recent_ppo) > 1.0:
            if recent_dqn > recent_ppo:
                return True, "DQN"
            else:
                return True, "PPO"
        
        return False, None
    
    def get_meta_score(self) -> float:
        """Get overall meta-score (combination of all metrics)"""
        
        if len(self.ensemble_metrics_history) == 0:
            return 0.0
        
        metrics = list(self.ensemble_metrics_history)[-1]
        
        # Weighted combination
        sharpe_score = np.clip(metrics['sharpe'] / 2, -1, 1)
        win_rate_score = metrics['win_rate']
        recovery_score = np.clip(metrics['recovery'] / 100, 0, 1)
        
        meta_score = 0.4 * sharpe_score + 0.3 * win_rate_score + 0.3 * recovery_score
        
        return float(meta_score)
    
    def get_evaluation_report(self) -> Dict:
        """Get comprehensive evaluation report"""
        
        eval_result = self.evaluate_agents()
        weights = self.get_recommended_weights()
        should_switch, switch_to = self.should_switch_agents()
        
        return {
            'dqn_metrics': eval_result['dqn'],
            'ppo_metrics': eval_result['ppo'],
            'ensemble_metrics': eval_result['ensemble'],
            'recommended_weights': weights,
            'should_switch': should_switch,
            'switch_to': switch_to,
            'meta_score': self.get_meta_score(),
            'timestamp': eval_result['timestamp']
        }


class TransferLearner:
    """
    Transfer learning between agents
    - Initialize PPO from DQN features
    - Or vice-versa
    - Accelerates convergence
    """
    
    @staticmethod
    def transfer_dqn_to_ppo(dqn_agent, ppo_agent):
        """Initialize PPO actor with DQN feature extractor"""
        
        try:
            # Copy first layer (shared feature extraction)
            dqn_params = dqn_agent.model.get_params()
            ppo_params = ppo_agent.model.get_params()
            
            # Transfer shared features
            ppo_params['w1'] = dqn_params['w1'].copy()
            ppo_params['b1'] = dqn_params['b1'].copy()
            
            ppo_agent.model.set_params(ppo_params)
            
            print("✅ Transfer learning: DQN → PPO (shared features)")
            return True
        except Exception as e:
            print(f"⚠️ Transfer learning failed: {e}")
            return False
    
    @staticmethod
    def transfer_ppo_to_dqn(ppo_agent, dqn_agent):
        """Initialize DQN feature extractor from PPO"""
        
        try:
            ppo_params = ppo_agent.model.get_params()
            dqn_params = dqn_agent.model.get_params()
            
            # Transfer shared features
            dqn_params['w1'] = ppo_params['w1'].copy()
            dqn_params['b1'] = ppo_params['b1'].copy()
            
            dqn_agent.model.set_params(dqn_params)
            
            print("✅ Transfer learning: PPO → DQN (shared features)")
            return True
        except Exception as e:
            print(f"⚠️ Transfer learning failed: {e}")
            return False
    
    @staticmethod
    def warm_start_from_pretrained(agent, pretrained_path):
        """Initialize agent from pretrained weights"""
        try:
            agent.load(pretrained_path)
            print(f"✅ Warm start: loaded from {pretrained_path}")
            return True
        except Exception as e:
            print(f"⚠️ Warm start failed: {e}")
            return False


class PaperTradingEngine:
    """
    Simulate trading before live deployment
    - Replays historical data
    - Logs all decisions and outcomes
    - Validates ensemble behavior
    """
    
    def __init__(self, agent, feature_extractor, 
                 historical_prices: List[float],
                 historical_volumes: List[float],
                 transaction_cost=0.001):
        """
        Initialize paper trading
        
        Args:
            agent: AdvancedHybridTradingAgent
            feature_extractor: EnrichedFeatureExtractor
            historical_prices: Price history
            historical_volumes: Volume history
            transaction_cost: Transaction cost ratio (0.1% = 0.001)
        """
        
        self.agent = agent
        self.feature_extractor = feature_extractor
        self.prices = historical_prices
        self.volumes = historical_volumes
        self.transaction_cost = transaction_cost
        
        # Tracking
        self.trades = []
        self.equity_curve = [10000.0]
        self.current_equity = 10000.0
        self.position = None  # None, 'LONG', or 'SHORT'
        self.entry_price = None
        self.entry_confidence = None
        self.entry_step = None
    
    def run_paper_trading(self, days=20) -> Dict:
        """
        Run paper trading simulation
        
        Args:
            days: Number of days to simulate (assume ~250 day trading days)
        
        Returns:
            Performance report
        """
        
        steps_per_day = len(self.prices) // 250  # Rough estimate
        max_steps = min(len(self.prices), days * steps_per_day)
        
        print(f"\n🟢 Paper Trading Started")
        print(f"   Period: {days} days (~{max_steps} steps)")
        print(f"   Initial Capital: ${self.current_equity:.2f}")
        print(f"   Transaction Cost: {self.transaction_cost*100:.2f}%")
        
        for step in range(max_steps):
            price = self.prices[step]
            volume = self.volumes[step] if step < len(self.volumes) else 1000
            
            # Update feature extractor
            self.feature_extractor.add_data(price, volume)
            
            # Get agent decision
            state = self.feature_extractor.extract_48_features()
            decision = self.agent.act(state, market_metrics={
                'returns': [],
                'drawdown': 0.0
            })
            
            action = decision['action']
            confidence = decision['confidence']
            
            # Execute trade
            if action == 1 and self.position is None:  # BUY
                self.position = 'LONG'
                self.entry_price = price
                self.entry_confidence = confidence
                self.entry_step = step
            
            elif action == 2 and self.position == 'LONG':  # SELL
                holding_steps = step - self.entry_step
                pnl_pct = (price - self.entry_price) / self.entry_price - self.transaction_cost *2
                pnl = self.current_equity * pnl_pct
                
                self.current_equity += pnl
                
                self.trades.append({
                    'entry_step': self.entry_step,
                    'entry_price': self.entry_price,
                    'exit_step': step,
                    'exit_price': price,
                    'holding_steps': holding_steps,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': self.entry_confidence
                })
                
                self.position = None
                self.entry_price = None
            
            # Log equity
            self.equity_curve.append(self.current_equity)
        
        # Close any open position
        if self.position == 'LONG':
            final_price = self.prices[-1]
            pnl_pct = (final_price - self.entry_price) / self.entry_price - self.transaction_cost * 2
            pnl = self.current_equity * pnl_pct
            self.current_equity += pnl
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate paper trading report"""
        
        if len(self.trades) == 0:
            return {
                'status': 'No trades executed',
                'total_return': 0.0,
                'final_equity': self.current_equity
            }
        
        pnls = [t['pnl_pct'] for t in self.trades]
        total_return = (self.current_equity - 10000.0) / 10000.0
        
        equity_array = np.array(self.equity_curve)
        max_equity = np.max(equity_array)
        max_dd = (np.min(equity_array) - max_equity) / max_equity
        
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        
        report = {
            'trades_executed': len(self.trades),
            'total_return': float(total_return),
            'final_equity': float(self.current_equity),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_dd),
            'avg_trade_duration': float(np.mean([t['holding_steps'] for t in self.trades])),
            'best_trade': float(max(pnls)),
            'worst_trade': float(min(pnls)),
            'equity_curve': self.equity_curve
        }
        
        print(f"\n📊 Paper Trading Report")
        print(f"   Trades: {report['trades_executed']}")
        print(f"   Return: {report['total_return']:.2%}")
        print(f"   Final Equity: ${report['final_equity']:.2f}")
        print(f"   Sharpe: {report['sharpe_ratio']:.3f}")
        print(f"   Win Rate: {report['win_rate']:.1%}")
        print(f"   Max DD: {report['max_drawdown']:.2%}")
        
        return report
