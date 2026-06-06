"""
Advanced Ensemble Manager for DQN + PPO Hybrid Trading
Implements hierarchical and voting-based arbitration with performance history.
"""

import numpy as np
from collections import deque
from datetime import datetime


class PerformanceTracker:
    """Track performance metrics for ensemble voting"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.dqn_returns = deque(maxlen=window_size)
        self.ppo_returns = deque(maxlen=window_size)
        self.dqn_trades = []  # [(return, action, state), ...]
        self.ppo_trades = []
        
    def log_trade(self, agent_type, action, return_val, drawdown):
        """Log trade outcome for an agent"""
        if agent_type == "DQN":
            self.dqn_returns.append(return_val)
            self.dqn_trades.append({
                'return': return_val,
                'action': action,
                'drawdown': drawdown,
                'timestamp': datetime.now()
            })
        elif agent_type == "PPO":
            self.ppo_returns.append(return_val)
            self.ppo_trades.append({
                'return': return_val,
                'action': action,
                'drawdown': drawdown,
                'timestamp': datetime.now()
            })
    
    def get_sharpe_ratio(self, agent_type):
        """Calculate Sharpe ratio for agent"""
        if agent_type == "DQN":
            returns = list(self.dqn_returns)
        else:
            returns = list(self.ppo_returns)
        
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_ret = np.mean(returns_array)
        std_ret = np.std(returns_array)
        
        if std_ret < 1e-8:
            return 0.0
        
        return mean_ret / (std_ret + 1e-8)
    
    def get_win_rate(self, agent_type):
        """Get win rate (trades with positive returns)"""
        if agent_type == "DQN":
            trades = self.dqn_trades
        else:
            trades = self.ppo_trades
        
        if len(trades) == 0:
            return 0.5
        
        wins = sum(1 for t in trades if t['return'] > 0)
        return wins / len(trades)
    
    def get_max_drawdown(self, agent_type):
        """Get maximum drawdown for agent"""
        if agent_type == "DQN":
            returns = list(self.dqn_returns)
        else:
            returns = list(self.ppo_returns)
        
        if len(returns) == 0:
            return 0.0
        
        cum_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        
        return np.min(drawdown) if len(drawdown) > 0 else 0.0


class HierarchicalEnsembleManager:
    """
    Hierarchical ensemble: DQN proposes mode, PPO executes policy.
    DQN = Market Mode Detector (aggressive/defensive/neutral)
    PPO = Policy Executor (fine-grained risk management)
    """
    
    MODES = {
        0: "NEUTRAL",    # No action
        1: "AGGRESSIVE",  # Full position size
        2: "DEFENSIVE"    # Reduced position, tight stops
    }
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.mode = "NEUTRAL"
        self.performance = PerformanceTracker()
        
    def decide_mode(self, dqn_action, dqn_confidence):
        """DQN decides market mode"""
        if dqn_action == 0:  # HOLD
            self.mode = "NEUTRAL"
        elif dqn_confidence > 0.7:  # High confidence
            self.mode = "AGGRESSIVE"
        else:
            self.mode = "DEFENSIVE"
        
        return self.mode
    
    def execute_action(self, mode, ppo_action, ppo_params):
        """PPO executes within the mode decided by DQN"""
        position_size = ppo_params.get('position_size', 1.0)
        stop_loss = ppo_params.get('stop_loss', 0.02)
        take_profit = ppo_params.get('take_profit', 0.03)
        
        # Modify execution based on mode
        if mode == "DEFENSIVE":
            position_size *= 0.5  # Half position
            stop_loss *= 0.5  # Tighter stops
            take_profit *= 0.5  # Lock profits earlier
        elif mode == "AGGRESSIVE":
            position_size = min(position_size * 1.2, 1.0)  # Slightly larger
        
        return {
            'action': ppo_action,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }


class VotingEnsembleManager:
    """
    Voting-based ensemble: Both agents propose, arbitrer decides.
    Uses Sharpe ratio, win rate, and agreement metrics.
    """
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.performance = PerformanceTracker()
        self.agreement_count = 0
        self.disagreement_count = 0
        
    def vote(self, dqn_action, dqn_q_values, ppo_action, ppo_probs):
        """
        Arbitrate between DQN and PPO using voting and performance metrics.
        
        Strategies:
        1. If both agree → execute immediately
        2. If disagreement:
           - Use Sharpe ratio weights (60% PPO, 40% DQN)
           - Or vote only if confidence high enough
        3. Fallback: PPO (more stable)
        """
        
        dqn_sharpe = self.performance.get_sharpe_ratio("DQN")
        ppo_sharpe = self.performance.get_sharpe_ratio("PPO")
        
        # Check agreement
        if dqn_action == ppo_action:
            self.agreement_count += 1
            return {
                'action': dqn_action,
                'confidence': 0.9,
                'source': 'agreement',
                'dqn_q': dqn_q_values[dqn_action],
                'ppo_prob': ppo_probs[ppo_action]
            }
        
        self.disagreement_count += 1
        
        # Weighted voting based on recent Sharpe ratios
        total_sharpe = abs(dqn_sharpe) + abs(ppo_sharpe) + 0.1
        dqn_weight = 0.4 * (abs(dqn_sharpe) / total_sharpe)
        ppo_weight = 0.6 * (abs(ppo_sharpe) / total_sharpe)
        
        # DQN confidence = Q-value / max possible Q-value
        dqn_conf = dqn_q_values[dqn_action] / (np.max(np.abs(dqn_q_values)) + 1e-8)
        
        # PPO confidence = probability of action
        ppo_conf = ppo_probs[ppo_action]
        
        dqn_score = dqn_weight * abs(dqn_conf)
        ppo_score = ppo_weight * ppo_conf
        
        if dqn_score > ppo_score:
            chosen_action = dqn_action
            chosen_source = "DQN_weighted"
            confidence = dqn_conf
        else:
            chosen_action = ppo_action
            chosen_source = "PPO_weighted"
            confidence = ppo_conf
        
        return {
            'action': chosen_action,
            'confidence': min(abs(confidence), 1.0),
            'source': chosen_source,
            'dqn_score': dqn_score,
            'ppo_score': ppo_score,
            'agreement_ratio': self.agreement_count / (self.agreement_count + self.disagreement_count + 1)
        }
    
    def get_statistics(self):
        """Return ensemble statistics"""
        total = self.agreement_count + self.disagreement_count
        agreement_rate = self.agreement_count / (total + 1e-8)
        
        return {
            'agreement_rate': agreement_rate,
            'dqn_sharpe': self.performance.get_sharpe_ratio("DQN"),
            'ppo_sharpe': self.performance.get_sharpe_ratio("PPO"),
            'dqn_win_rate': self.performance.get_win_rate("DQN"),
            'ppo_win_rate': self.performance.get_win_rate("PPO"),
            'total_votes': total
        }


class RiskAwareModeManager:
    """
    Adaptive mode selection based on market risk metrics.
    Adjusts DQN/PPO weights based on volatility and drawdown.
    """
    
    def __init__(self):
        self.volatility_window = deque(maxlen=30)
        self.drawdown_window = deque(maxlen=30)
        self.current_volatility = 0.0
        self.current_drawdown = 0.0
        
    def update_market_metrics(self, returns, equity_drawdown):
        """Update market volatility and drawdown estimates"""
        if len(returns) > 0:
            self.volatility_window.append(np.std(returns))
            self.current_volatility = np.mean(list(self.volatility_window))
        
        if equity_drawdown is not None:
            self.drawdown_window.append(equity_drawdown)
            self.current_drawdown = np.mean(list(self.drawdown_window))
    
    def get_adaptive_weights(self):
        """
        Adjust DQN (opportunistic) / PPO (defensive) weights based on risk.
        
        High volatility/drawdown → favor PPO (60% PPO, 40% DQN)
        Low risk → slightly favor DQN (50/50)
        """
        risk_score = self.current_volatility + abs(self.current_drawdown)
        
        # Normalize risk score (0 = low risk, 1 = high risk)
        risk_normalized = min(risk_score / 0.1, 1.0)
        
        # Interpolate weights
        ppo_weight = 0.5 + 0.1 * risk_normalized
        dqn_weight = 1.0 - ppo_weight
        
        return {
            'dqn_weight': dqn_weight,
            'ppo_weight': ppo_weight,
            'risk_score': risk_score,
            'volatility': self.current_volatility,
            'drawdown': self.current_drawdown
        }
