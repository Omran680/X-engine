import os as _os

# Project root (…/X-engine) — all runtime paths are absolute so the bot behaves
# the same whatever the current working directory (launchd, MCP client, cron…).
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# ============= TRADING CONFIG =============
# Default asset: Gold (XAU/USD)
EPIC = "CS.D.IN_GOLD.MFI.IP"
SIZE = 0.1
STOP_LOSS_PCT = 0.2
TAKE_PROFIT_PCT = 0.3

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
LOG_DIR = _os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = _os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_STEPS = 500  # Save a checkpoint every N live steps
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR

# ============= LOOP / DATA =============
LOOP_INTERVAL_SECONDS = 5      # Sleep between decision steps
MIN_HISTORY_BARS = 20          # Price bars needed before the first decision
TRANSACTION_COST = 0.00005     # Per-action cost (0.005 %, spread proxy) charged in the RL reward
DEFAULT_VOLUME = 1000.0        # IG gives no tick volume → neutral constant (volume features are inert)

# ============= MARKET HOURS / RESILIENCE =============
MARKET_CLOSED_POLL_SECONDS = 60      # poll cadence while the market is closed (weekend, daily break)
MARKET_CLOSED_LONG_POLL_SECONDS = 300  # …after being closed for more than 1 h
DATA_GAP_RESET_SECONDS = 300         # no price for this long ⇒ drop history (a gap would poison the features)
OUTAGE_MAX_BACKOFF_SECONDS = 120     # cap of the exponential backoff on network/API outages
HEARTBEAT_FILE = _os.path.join(PROJECT_ROOT, "logs", "heartbeat.json")

# ============= RISK / SAFETY =============
RISK_INITIAL_CAPITAL = 10_000.0
MAX_DAILY_LOSS_PCT = 0.05      # Stop opening trades after -5 % of capital in a day
MAX_CONSECUTIVE_LOSSES = 3     # Pause after N losses in a row (resets next day)
MAX_ORDER_SIZE = 1.0           # Hard cap on lots for any order (incl. MCP open_trade)
MAX_OPEN_POSITIONS = 2         # Main + scalp
POINT_VALUE = 1.0              # PnL per point per lot (estimate for risk accounting)

# ============= IG ACCOUNT =============
IG_ACC_TYPE = _os.getenv("IG_ACC_TYPE", "DEMO").upper()   # DEMO | LIVE

# ============= MCP SERVER =============
MCP_DEFAULT_HOST = "127.0.0.1"  # Never expose order-sending tools on 0.0.0.0 by default

# ============= API RATE LIMITING =============
API_RATE_LIMIT_INTERVAL = 2    # Min seconds between IG Markets API calls
API_CACHE_TTL = 5              # Price cache TTL in seconds
API_BACKOFF_RATE_LIMIT = 15    # Base backoff when rate-limited (seconds)
API_BACKOFF_BASE = 2           # Base backoff for generic API errors (seconds)

# ============= GROQ LLM =============
GROQ_CALL_COOLDOWN = 2         # Min seconds between Groq calls
GROQ_MAX_RETRIES = 3           # Max retries on Groq API error
GROQ_MODEL = _os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # gemma2-9b-it was decommissioned

# ============= SCALPING =============
SCALP_ENABLED = False          # Off by default — enable via MCP or --scalping flag
SCALP_LOOKBACK = 10            # Bars used for scalp signal (shorter than main 20)
SCALP_SL_PCT = 0.15            # Stop-loss  0.15 % (tight)
SCALP_TP_PCT = 0.25            # Take-profit 0.25 % (tight)
SCALP_SIZE = 0.1               # Lot size for scalp orders
SCALP_MAX_TRADES_PER_HOUR = 10 # Circuit-breaker: max trades per 60 min window
SCALP_MAX_HOLD_SECONDS = 300   # Auto-exit after 5 min if TP/SL not hit
SCALP_BB_SQUEEZE_FACTOR = 0.003 # BB width / price threshold for "squeeze"
SCALP_MOMENTUM_THRESHOLD = 0.0005  # Min 3-bar ROC to confirm breakout
SCALP_RSI_LOW = 35             # RSI below this → oversold (avoid SELL)
SCALP_RSI_HIGH = 65            # RSI above this → overbought (avoid BUY)
SCALP_VOLUME_SPIKE = 1.2       # Volume must be N× average to confirm
