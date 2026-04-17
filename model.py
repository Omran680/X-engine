import numpy as np
import pickle
import os


def relu(x):
    return np.maximum(0.0, x)


def softmax(x):
    x = np.atleast_2d(x).astype(np.float32)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)


class DQNNetwork:
    """Simple NumPy-based DQN network"""
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.w2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.w3 = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(action_size, dtype=np.float32)

        self.last_input = None
        self.last_h1 = None
        self.last_h2 = None

    def forward(self, x):
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        self.last_h1 = relu(x.dot(self.w1) + self.b1)
        self.last_h2 = relu(self.last_h1.dot(self.w2) + self.b2)
        return self.last_h2.dot(self.w3) + self.b3

    def predict(self, state):
        q_values = self.forward(state)
        return q_values.reshape(-1) if q_values.ndim == 2 and q_values.shape[0] == 1 else q_values

    def get_params(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'w2': self.w2.copy(),
            'b2': self.b2.copy(),
            'w3': self.w3.copy(),
            'b3': self.b3.copy()
        }

    def set_params(self, params):
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.w2 = params['w2'].copy()
        self.b2 = params['b2'].copy()
        self.w3 = params['w3'].copy()
        self.b3 = params['b3'].copy()

    def copy(self):
        clone = DQNNetwork(self.state_size, self.action_size, self.hidden_dim)
        clone.set_params(self.get_params())
        return clone

    def save(self, filepath):
        """Save network weights to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"DQN Network saved to {filepath}")

    def load(self, filepath):
        """Load network weights from file"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"DQN Network loaded from {filepath}")


class PPONetwork:
    """Simple NumPy-based actor-critic network"""
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.actor_w = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.actor_b = np.zeros(action_size, dtype=np.float32)

        self.critic_w = rng.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.critic_b = np.zeros(1, dtype=np.float32)

        self.last_input = None
        self.last_h1 = None

    def forward(self, x):
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        self.last_h1 = relu(x.dot(self.w1) + self.b1)

        logits = self.last_h1.dot(self.actor_w) + self.actor_b
        values = self.last_h1.dot(self.critic_w) + self.critic_b
        return logits, values

    def predict(self, state):
        return self.forward(state)

    def get_params(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'actor_w': self.actor_w.copy(),
            'actor_b': self.actor_b.copy(),
            'critic_w': self.critic_w.copy(),
            'critic_b': self.critic_b.copy()
        }

    def set_params(self, params):
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.actor_w = params['actor_w'].copy()
        self.actor_b = params['actor_b'].copy()
        self.critic_w = params['critic_w'].copy()
        self.critic_b = params['critic_b'].copy()

    def save(self, filepath):
        """Save network weights to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"PPO Network saved to {filepath}")

    def load(self, filepath):
        """Load network weights from file"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"PPO Network loaded from {filepath}")


class DuelingDQNNetwork:
    """
    Improved DQN with Dueling Architecture
    Separates Value(state) from Advantage(state, action)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    
    Benefits:
    - Better convergence for state-only decisions
    - Reduces overestimation of actions
    - Improved stability
    """
    
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction
        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.w2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Value stream (predicts state value)
        self.value_w = rng.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.value_b = np.zeros(1, dtype=np.float32)
        
        # Advantage stream (predicts advantage of each action)
        self.advantage_w = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.advantage_b = np.zeros(action_size, dtype=np.float32)
        
        self.last_input = None
        self.last_h1 = None
        self.last_h2 = None
    
    def forward(self, x):
        """Forward pass returns Q-values"""
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        
        # Feature extraction
        self.last_h1 = relu(x.dot(self.w1) + self.b1)
        self.last_h2 = relu(self.last_h1.dot(self.w2) + self.b2)
        
        # Value stream
        values = self.last_h2.dot(self.value_w) + self.value_b  # Shape: (batch, 1)
        
        # Advantage stream
        advantages = self.last_h2.dot(self.advantage_w) + self.advantage_b  # Shape: (batch, actions)
        
        # Normalize advantages (subtract mean)
        advantages_normalized = advantages - np.mean(advantages, axis=1, keepdims=True)
        
        # Combine: Q(s,a) = V(s) + A(s,a)
        q_values = values + advantages_normalized
        
        return q_values
    
    def predict(self, state):
        """Predict Q-values for a state"""
        q_values = self.forward(state)
        return q_values.reshape(-1) if q_values.ndim == 2 and q_values.shape[0] == 1 else q_values
    
    def get_params(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'w2': self.w2.copy(),
            'b2': self.b2.copy(),
            'value_w': self.value_w.copy(),
            'value_b': self.value_b.copy(),
            'advantage_w': self.advantage_w.copy(),
            'advantage_b': self.advantage_b.copy()
        }
    
    def set_params(self, params):
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.w2 = params['w2'].copy()
        self.b2 = params['b2'].copy()
        self.value_w = params['value_w'].copy()
        self.value_b = params['value_b'].copy()
        self.advantage_w = params['advantage_w'].copy()
        self.advantage_b = params['advantage_b'].copy()
    
    def copy(self):
        """Create a copy of this network"""
        clone = DuelingDQNNetwork(self.state_size, self.action_size, self.hidden_dim)
        clone.set_params(self.get_params())
        return clone
    
    def save(self, filepath):
        """Save network weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Dueling DQN Network saved to {filepath}")
    
    def load(self, filepath):
        """Load network weights"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"Dueling DQN Network loaded from {filepath}")


class RiskAwarePPONetwork:
    """
    Enhanced PPO with Risk-Aware auxiliary task
    Predicts both policy logits AND risk metrics (drawdown, volatility)
    This helps the policy learn to manage risk explicitly
    """
    
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        
        # Shared feature layer
        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Actor head
        self.actor_w = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.actor_b = np.zeros(action_size, dtype=np.float32)
        
        # Critic head (value)
        self.critic_w = rng.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.critic_b = np.zeros(1, dtype=np.float32)
        
        # Risk auxiliary head (predict risk metrics)
        self.risk_w = rng.randn(hidden_dim, 2).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.risk_b = np.zeros(2, dtype=np.float32)
        
        self.last_input = None
        self.last_h1 = None
    
    def forward(self, x):
        """Forward pass returns (logits, values, risk_predictions)"""
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        
        # Shared feature extraction
        self.last_h1 = relu(x.dot(self.w1) + self.b1)
        
        # Actor logits (policy)
        logits = self.last_h1.dot(self.actor_w) + self.actor_b
        
        # Critic value
        values = self.last_h1.dot(self.critic_w) + self.critic_b
        
        # Risk predictions (drawdown, volatility)
        risk_preds = self.last_h1.dot(self.risk_w) + self.risk_b
        
        return logits, values, risk_preds
    
    def predict(self, state):
        """Predict policy, value, and risk metrics"""
        return self.forward(state)
    
    def get_params(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'actor_w': self.actor_w.copy(),
            'actor_b': self.actor_b.copy(),
            'critic_w': self.critic_w.copy(),
            'critic_b': self.critic_b.copy(),
            'risk_w': self.risk_w.copy(),
            'risk_b': self.risk_b.copy()
        }
    
    def set_params(self, params):
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.actor_w = params['actor_w'].copy()
        self.actor_b = params['actor_b'].copy()
        self.critic_w = params['critic_w'].copy()
        self.critic_b = params['critic_b'].copy()
        self.risk_w = params['risk_w'].copy()
        self.risk_b = params['risk_b'].copy()
    
    def save(self, filepath):
        """Save network weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Risk-Aware PPO Network saved to {filepath}")
    
    def load(self, filepath):
        """Load network weights"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"Risk-Aware PPO Network loaded from {filepath}")

