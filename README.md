# trade-bot · XAU/USD Hybrid RL Trading Bot

Système de trading algorithmique pour l'or (XAU/USD) via **IG Markets**.  
Combine DQN + PPO (ensemble RL), un agent scalping BB-squeeze, et un serveur **MCP** pour le contrôle temps-réel.

---

## Architecture

```
trade_bot/                  ← package Python principal
│
├── core/                   Configuration & logging
│   ├── config.py           Toutes les constantes (hyperparamètres, API, scalping…)
│   └── logging.py          Logger rotatif (console + fichier, 10 MB × 5 backups)
│
├── models/                 Réseaux de neurones (NumPy pur, sans framework ML)
│   ├── networks.py         DQNNetwork, PPONetwork, relu, softmax
│   └── replay_buffer.py    Experience replay (deque-based)
│
├── agents/                 Agents de trading
│   ├── base.py             DQNAgent · PPOAgent · HybridTradingAgent
│   ├── scalping.py         ScalpingAgent (BB-squeeze · RSI · volume · auto-exit)
│   └── groq.py             GrokTradeAgent (LLM Groq, 3ème votant optionnel)
│
├── features/               Extraction de features techniques
│   ├── extractor.py        FeatureExtractor (18 indicateurs) · TradingState
│   └── optimizer.py        Extraction enrichie 48 features (multi-timeframe)
│
├── ensemble/               Décision d'ensemble
│   ├── strategies.py       EnsembleDecisionMaker · EnsembleStrategy
│   └── manager.py          HierarchicalEnsembleManager · VotingEnsembleManager
│
├── execution/              Exchange & gestion du risque
│   ├── trader.py           Client IG Markets (rate-limit · cache · retry · streaming)
│   └── risk.py             PortfolioRiskManager · Kelly criterion · trailing stop
│
└── training/               Utilitaires d'entraînement offline
    ├── curriculum.py       Curriculum learning progressif
    ├── trainer.py          Orchestration multi-agent
    ├── validator.py        Pipeline de validation
    └── evaluator.py        Métriques Sharpe · drawdown · win-rate

mcp_server/                 Serveur MCP (Model Context Protocol)
├── context.py              BotContext — état partagé thread-safe (Lock)
├── read_tools.py           7 outils READ-ONLY  → jamais de mutation
├── exec_tools.py           8 outils EXECUTION  → exchange + mutations
└── server.py               Routeur MCP · transport stdio ou HTTP/SSE

main.py                     Boucle de trading (XAUUSDHybridTrader)
launch.py                   Launcher unifié : bot + MCP server dans 1 process
run_forever.sh              Relance automatique sur crash
```

---

## Prérequis

- Python 3.10+
- Compte IG Markets (DEMO ou LIVE)
- Clé API Groq *(optionnel — agent LLM 3ème votant)*

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Créer un fichier `.env` à la racine :

```env
IG_USERNAME=votre_username
IG_PASSWORD=votre_password
IG_API_KEY=votre_api_key
GROQ_API_KEY=votre_groq_key   # optionnel
```

Tous les paramètres sont dans `trade_bot/core/config.py` :

| Constante | Défaut | Description |
|-----------|--------|-------------|
| `EPIC` | `CS.D.IN_GOLD.MFI.IP` | Instrument IG Markets |
| `SIZE` | `0.1` | Taille position principale (lots) |
| `STOP_LOSS_PCT` | `0.2` | Stop-loss en % |
| `TAKE_PROFIT_PCT` | `0.3` | Take-profit en % |
| `STATE_SIZE` | `18` | Taille vecteur de features |
| `SCALP_SL_PCT` | `0.15` | Stop-loss scalping (%) |
| `SCALP_TP_PCT` | `0.25` | Take-profit scalping (%) |
| `SCALP_MAX_TRADES_PER_HOUR` | `10` | Circuit-breaker scalping |
| `SCALP_MAX_HOLD_SECONDS` | `300` | Auto-exit scalping (5 min) |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## Lancement

### Production (recommandé)

```bash
# Lance bot + MCP, redémarre automatiquement sur crash
nohup ./run_forever.sh > logs/run.log 2>&1 &
echo $! > bot.pid

# Logs en temps réel
tail -f logs/trade_bot.log
```

### Options

```
python3 launch.py [OPTIONS]

  --forever          Boucle infinie (sinon --max-steps N, défaut 50)
  --dry-run          Simule les trades, ne touche jamais l'exchange
  --grok             Active l'agent Groq (3ème votant LLM)
  --scalping         Active l'agent scalping au démarrage
  --http             MCP en HTTP/SSE  (défaut : stdio pour Claude Desktop)
  --port 8765        Port SSE (défaut 8765)
```

### Exemples

```bash
# Test sans risque
python3 launch.py --dry-run --forever --http

# Production : scalping + Groq activés
python3 launch.py --forever --scalping --grok --http

# Passer de DEMO à LIVE
# → modifier acc_type="LIVE" dans trade_bot/execution/trader.py
```

### Arrêter proprement

```bash
kill $(cat bot.pid)   # sauvegarde les modèles avant de quitter
```

---

## Serveur MCP — 15 outils

Le serveur MCP tourne en **background thread** dans le même process que le bot.  
Il partage un `BotContext` unique : les outils voient l'état du bot en temps réel.

### READ-ONLY (aucun effet de bord)

| Outil | Description |
|-------|-------------|
| `get_price` | Prix bid actuel pour un epic IG Markets |
| `get_positions` | Positions ouvertes sur le compte |
| `get_account_balance` | Solde et équité |
| `get_risk_metrics` | Sharpe ratio, win rate, drawdown, PnL |
| `get_bot_status` | Step, dernier prix, action, uptime, scalping |
| `get_scalp_signal` | Signal scalping calculé sur une liste de prix |
| `get_scalp_status` | Métriques scalping : trades, PnL, paramètres |

### EXECUTION ⚠️ (appels exchange / mutations d'état)

| Outil | Description |
|-------|-------------|
| `open_trade` | Ouvre un ordre marché BUY/SELL |
| `close_position` | Ferme une position par deal ID |
| `set_training_mode` | Active/désactive l'entraînement en ligne |
| `save_models` | Sauvegarde DQN + PPO immédiatement |
| `emergency_stop` | Arrêt immédiat du bot (positions non fermées) |
| `enable_scalping` | Active/désactive l'agent scalping |
| `set_scalp_params` | Modifie SL/TP/size/lookback à chaud |
| `scalp_force_exit` | Ferme la position scalping en urgence |

### Connexion Claude Desktop

Ajouter dans `~/.claude/claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "trade-bot": {
      "command": "/chemin/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/chemin/vers/trade-bot"
    }
  }
}
```

Ou lancer avec `--http` et pointer sur `http://localhost:8765/sse`.

---

## Stratégie de trading

### Agent principal — Ensemble DQN + PPO

```
18 features extraites des 20 dernières barres
  Momentum   : ROC, SMA-diff, EMA-diff, Momentum brut
  Volatilité : StdDev, ATR, Bollinger position, Range
  Tendance   : Pente LR, Higher-Highs/Lower-Lows, ADX-like
  Mean-rev.  : RSI(14), Distance SMA, Stochastique
  Volume     : Vol/SMA-vol, Vol StdDev

     ↓                    ↓
 DQN Agent           PPO Agent
(off-policy)        (on-policy)
Double DQN          Actor-Critic + GAE
Target network      Clipped surrogate

     ↓                    ↓
  Ensemble — weighted voting (poids adaptatifs)
              ↓
      BUY / SELL / HOLD

  [+ Groq LLM optionnel → vote majoritaire à 3]
```

### Agent scalping — BB Squeeze

Signal **BUY** si toutes les conditions sont vraies :

1. `bb_width < 0.003` — bandes de Bollinger en compression
2. `3-bar ROC > 0.05 %` — micro-momentum positif
3. `RSI < 65` — pas suracheté
4. `prix > SMA` — cassure haussière

Signal **SELL** : conditions symétriques.  
Signal **HOLD** : dès qu'une condition échoue.

**Contrôles de risque intégrés :**
- SL 0.15 % / TP 0.25 % (serré — scalp)
- Auto-exit après 5 minutes si ni SL ni TP
- Circuit-breaker : max 10 trades/heure glissante

---

## Logs et modèles

```
logs/
└── trade_bot.log     Rotation auto (10 MB × 5 fichiers)

models/
├── dqn_model.pkl     Poids DQN sauvegardés
└── ppo_model.pkl     Poids PPO sauvegardés
```

Checkpoint automatique toutes les **500 steps** + sauvegarde sur `Ctrl+C` ou MCP `save_models`.

---

## Dépendances

| Package | Version | Rôle |
|---------|---------|------|
| `numpy` | ≥1.21, <3 | Calcul vectoriel, backprop NumPy |
| `pandas` | ≥1.3, <3 | Traitement données |
| `scipy` | ≥1.7, <2 | Calcul scientifique |
| `scikit-learn` | ≥1.0, <2 | Utilitaires ML |
| `trading-ig` | ≥0.0.7 | Client IG Markets REST API |
| `python-dotenv` | ≥0.19, <2 | Chargement `.env` |
| `mcp` | ≥1.27 | Serveur MCP (Anthropic) |
| `groq` | ≥0.4 | Client API Groq (LLM) |
| `matplotlib` | ≥3.4, <4 | Visualisation *(optionnel)* |
