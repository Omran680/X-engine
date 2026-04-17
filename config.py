# ============= TRADING CONFIG =============
# Default asset: Gold (XAU/USD)
EPIC = "CS.D.IN_GOLD.MFI.IP"
SIZE = 0.1
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 3.0

# ============= AGENT CONFIG =============
STATE_SIZE = 18  # Feature vector size (from FeatureExtractor)
ACTIONS = ["BUY", "SELL", "HOLD"]  # Trading actions
ACTION_SIZE = 3

# ============= HYPERPARAMETERS =============
GAMMA = 0.99  # Discount factor
LR = 0.0003  # Learning rate

# ============= DQN CONFIG =============
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000  # Update target network every N steps

# ============= PPO CONFIG =============
PPO_EPOCHS = 3  # Number of training epochs per batch
PPO_CLIP_RATIO = 0.2  # PPO clipping parameter
PPO_ENTROPY_COEFF = 0.01  # Entropy bonus coefficient
GAE_LAMBDA = 0.95  # Generalized Advantage Estimation lambda

# ============= EXPERIENCE REPLAY =============
BATCH_SIZE = 32
MEMORY_SIZE = 5000
REPLAY_START_SIZE = 100  # Start training after N experiences

# ============= ENSEMBLE CONFIG =============
DQN_WEIGHT = 0.5  # Initial DQN weight in ensemble
PPO_WEIGHT = 0.5  # Initial PPO weight in ensemble
ENSEMBLE_STRATEGY = "weighted_voting"  # Options: voting, weighted_voting, averaging, stacking

# ============= FEATURE EXTRACTION =============
LOOKBACK_WINDOW = 20  # Historical window for feature extraction
FEATURE_NORMALIZATION = "minmax"  # Options: minmax, zscore

# ============= TRAINING CONFIG =============
EPISODES = 100
STEPS_PER_EPISODE = 5000
VERBOSE = True

# ============= EVALUATION CONFIG =============
EVAL_INTERVAL = 10  # Evaluate every N episodes
EVAL_EPISODES = 5

# ============= LOGGING =============
LOG_DIR = "./logs"
MODELS_DIR = "./models"
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N episodes
