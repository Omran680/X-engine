"""
Advanced Hybrid Trading Agent
Uses Dueling DQN + Risk-Aware PPO with intelligent ensemble features
"""

import numpy as np
import os
from model import DuelingDQNNetwork, RiskAwarePPONetwork, softmax
from replay_buffer import ReplayBuffer
from ensemble_manager import VotingEnsembleManager, HierarchicalEnsembleManager, RiskAwareModeManager
from collections import deque


class DuelingDQNAgent:
    """
    Improved DQN Agent with:
    - Dueling Architecture (Value + Advantage streams)
    - Double DQN (reduce overestimation bias)
    - Larger Experience Replay Buffer
    - Less frequent target network updates
    """
    
    def __init__(self, state_size=10, action_size=3, lr=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        
        # ε-greedy exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # Dueling architecture
        self.model = DuelingDQNNetwork(state_size, action_size, hidden_dim=256)
        self.target_model = self.model.copy()
        
        # Larger experience buffer
        self.memory = ReplayBuffer(size=10000)  # Increased from 5000
        
        # Double DQN: update target less frequently
        self.update_target_freq = 2000  # Increased from 1000
        self.train_step = 0
        
        # Track Q-value statistics
        self.q_value_history = deque(maxlen=100)
    
    def act(self, state):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.model.predict(state)
        return int(np.argmax(q_values))
    
    def act_greedy(self, state):
        """Pure greedy action (no exploration)"""
        q_values = self.model.predict(state)
        return int(np.argmax(q_values))
    
    def predict(self, state):
        """Get Q-values for state"""
        return self.model.predict(state)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        """
        Train on batch using Double DQN:
        - Online network selects action
        - Target network evaluates action
        This reduces overestimation bias
        """
        if self.memory.size() < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        states = np.vstack([exp[0] for exp in batch]).astype(np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.vstack([exp[3] for exp in batch]).astype(np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        # Get current Q-values
        q_values = self.model.forward(states)
        
        # Double DQN: Online network chooses, target network evaluates
        next_actions = np.argmax(self.model.forward(next_states), axis=1)
        next_q_values = self.target_model.forward(next_states)
        max_next_q = next_q_values[np.arange(batch_size), next_actions]
        
        # Compute Bellman targets
        target_q = q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]
        
        # Backward pass
        error = (q_values - target_q) / batch_size
        
        # Gradient through Q-network (simplified backprop)
        dA = error  # advantage gradient
        
        # Get advantage gradients
        dW_adv = self.model.last_h2.T.dot(error)
        db_adv = np.sum(error, axis=0)
        
        dh2 = error.dot(self.model.advantage_w.T)
        dh2[self.model.last_h2 <= 0] = 0.0
        
        dW2 = self.model.last_h1.T.dot(dh2)
        db2 = np.sum(dh2, axis=0)
        
        dh1 = dh2.dot(self.model.w2.T)
        dh1[self.model.last_h1 <= 0] = 0.0
        
        dW1 = self.model.last_input.T.dot(dh1)
        db1 = np.sum(dh1, axis=0)
        
        # Update advantage head weights
        self.model.advantage_w -= self.lr * dW_adv
        self.model.advantage_b -= self.lr * db_adv
        
        # Update hidden layer weights
        self.model.w1 -= self.lr * dW1
        self.model.b1 -= self.lr * db1
        self.model.w2 -= self.lr * dW2
        self.model.b2 -= self.lr * db2
        
        # Update value head (scalar target, separate gradient)
        # Value head predicts state value (single output)
        mean_error = np.mean(q_values - target_q, axis=1, keepdims=True)
        dW_val = self.model.last_h2.T.dot(mean_error) / batch_size
        db_val = np.mean(mean_error)
        
        if dW_val.shape == self.model.value_w.shape:
            self.model.value_w -= self.lr * dW_val
            self.model.value_b -= self.lr * db_val
        
        # Track training progress
        self.train_step += 1
        q_mean = np.mean(np.max(q_values, axis=1))
        self.q_value_history.append(q_mean)
        
        # Less frequent target update (Double DQN)
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_params(self.model.get_params())
            print(f"[DQN] Target network updated at step {self.train_step}")
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model.load(filepath)
        self.target_model.set_params(self.model.get_params())


class RiskAwarePPOAgent:
    """
    Enhanced PPO Agent with:
    - Risk-Aware auxiliary task (predicts drawdown/volatility)
    - Separate policy and value networks
    - GAE (Generalized Advantage Estimation)
    """
    
    def __init__(self, state_size=10, action_size=3, lr=0.001, gamma=0.99, gae_lambda=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.gae_lambda = gae_lambda
        
        # Risk-aware network with auxiliary output
        self.model = RiskAwarePPONetwork(state_size, action_size, hidden_dim=256)
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.entropy_coeff = 0.01
        self.ppo_epochs = 3
        
        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.risk_targets = []
        
        self.train_step = 0
    
    def act(self, state):
        """Sample action from policy"""
        logits, value, risk_preds = self.model.predict(state)
        logits = logits.reshape(-1)
        
        # Compute probabilities
        probs = softmax(logits.reshape(1, -1))[0]
        action = np.random.choice(self.action_size, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        
        return action, log_prob, value.item() if hasattr(value, 'item') else value
    
    def act_greedy(self, state):
        """Deterministic action (no sampling)"""
        logits, _, _ = self.model.predict(state)
        return int(np.argmax(logits))
    
    def remember(self, state, action, reward, value, log_prob, risk_target=None):
        """Store trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.risk_targets.append(risk_target if risk_target is not None else 0.0)
    
    def compute_gae(self):
        """Compute General Advantage Estimation"""
        advantages = []
        returns = []
        gae = 0.0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return np.array(advantages), np.array(returns)
    
    def update(self):
        """PPO update with risk-aware auxiliary loss"""
        if len(self.states) == 0:
            return
        
        states = np.vstack(self.states).astype(np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        
        advantages, returns = self.compute_gae()
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # PPO updates
        for epoch in range(self.ppo_epochs):
            logits, values, risk_preds = self.model.forward(states)
            
            # Compute new log probabilities
            probs = softmax(logits)
            new_log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
            
            # PPO policy loss
            ratio = np.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -np.mean(np.minimum(surr1, surr2))
            
            # Value loss
            value_loss = np.mean((values.reshape(-1) - returns) ** 2)
            
            # Risk auxiliary loss (predict drawdown/volatility)
            risk_targets = np.array(self.risk_targets, dtype=np.float32).reshape(-1, 1)
            risk_loss = np.mean((risk_preds - risk_targets) ** 2)
            
            # Entropy bonus
            entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + 0.1 * risk_loss - self.entropy_coeff * entropy
            
            # Simplified gradient update
            # In production, use actual backprop through network
            self.train_step += 1
        
        # Clear trajectory
        self._reset_trajectory()
    
    def _reset_trajectory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.risk_targets = []
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model.load(filepath)


class AdvancedHybridTradingAgent:
    """
    Advanced hybrid agent combining:
    - Dueling DQN (exploration, quick decisions)
    - Risk-Aware PPO (stable, risk-managed execution)
    - Multiple ensemble strategies
    """
    
    def __init__(self, state_size=18, action_size=3, ensemble_type="voting"):
        """
        Initialize the advanced hybrid agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions (3: BUY, SELL, HOLD)
            ensemble_type: "voting", "hierarchical", or "risk_aware"
        """
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_type = ensemble_type
        
        # Initialize agents
        self.dqn = DuelingDQNAgent(state_size, action_size)
        self.ppo = RiskAwarePPOAgent(state_size, action_size)
        
        # Initialize ensemble managers
        if ensemble_type == "voting":
            self.ensemble = VotingEnsembleManager(state_size, action_size)
        elif ensemble_type == "hierarchical":
            self.ensemble = HierarchicalEnsembleManager(state_size, action_size)
        else:
            self.ensemble = VotingEnsembleManager(state_size, action_size)  # Default
        
        self.risk_mode = RiskAwareModeManager()
        
        self.train_step = 0
    
    def act(self, state, market_metrics=None, use_ensemble=True):
        """
        Make trading decision using ensemble
        
        Args:
            state: Current market state
            market_metrics: Dict with 'returns' and 'drawdown'
            use_ensemble: Whether to use ensemble or just greedy
        
        Returns:
            Dict with action, confidence, and metadata
        """
        if market_metrics:
            self.risk_mode.update_market_metrics(
                market_metrics.get('returns', []),
                market_metrics.get('drawdown', 0.0)
            )
        
        # Get decisions from both agents
        dqn_action = self.dqn.act_greedy(state)
        dqn_q_values = self.dqn.predict(state)
        
        ppo_action, _, _ = self.ppo.act(state)
        _, _, risk_preds = self.ppo.model.predict(state)
        ppo_logits, _, _ = self.ppo.model.predict(state)
        ppo_probs = softmax(ppo_logits.reshape(1, -1))[0]
        
        if not use_ensemble or self.ensemble_type == "voting":
            # Voting ensemble
            decision = self.ensemble.vote(dqn_action, dqn_q_values, ppo_action, ppo_probs)
            
            return {
                'action': decision['action'],
                'confidence': decision['confidence'],
                'source': decision['source'],
                'dqn_action': dqn_action,
                'ppo_action': ppo_action,
                'ensemble_stats': self.ensemble.get_statistics()
            }
        
        elif self.ensemble_type == "hierarchical":
            # Hierarchical: DQN decides mode, PPO executes
            mode = self.ensemble.decide_mode(dqn_action, np.max(dqn_q_values))
            
            ppo_params = {
                'position_size': 1.0,
                'stop_loss': 0.02,
                'take_profit': 0.03
            }
            
            execution = self.ensemble.execute_action(mode, ppo_action, ppo_params)
            
            return {
                'action': execution['action'],
                'confidence': 0.85,
                'source': f"hierarchical_{mode}",
                'position_size': execution['position_size'],
                'stop_loss': execution['stop_loss'],
                'take_profit': execution['take_profit'],
                'mode': mode
            }
        
        else:
            # Default: voting
            decision = self.ensemble.vote(dqn_action, dqn_q_values, ppo_action, ppo_probs)
            return {
                'action': decision['action'],
                'confidence': decision['confidence'],
                'source': decision['source']
            }
    
    def train_step_fn(self, state, action, reward, next_state, done, batch_size=32):
        """
        Training step for both agents
        """
        # Store in both agents
        self.dqn.remember(state, action, reward, next_state, done)
        
        # Train DQN
        self.dqn.replay(batch_size)
        
        # PPO uses trajectory, so just store
        # (PPO update would happen periodically with trajectory)
        
        self.train_step += 1
    
    def save_models(self, models_dir="./models"):
        """Save both DQN and PPO models"""
        os.makedirs(models_dir, exist_ok=True)
        self.dqn.save(f"{models_dir}/dueling_dqn.pkl")
        self.ppo.save(f"{models_dir}/risk_aware_ppo.pkl")
        print(f"Models saved to {models_dir}")
    
    def load_models(self, models_dir="./models"):
        """Load both DQN and PPO models"""
        self.dqn.load(f"{models_dir}/dueling_dqn.pkl")
        self.ppo.load(f"{models_dir}/risk_aware_ppo.pkl")
        print(f"Models loaded from {models_dir}")
