"""trade_bot — modular XAU/USD hybrid RL trading system.

Package layout
--------------
trade_bot.core        config constants · logging factory
trade_bot.models      neural networks · replay buffer
trade_bot.agents      DQN · PPO · Hybrid · Scalping · Groq
trade_bot.features    feature extraction · trading state
trade_bot.ensemble    voting strategies · dynamic managers
trade_bot.execution   IG Markets API client · risk manager
trade_bot.training    curriculum · trainer · validator · meta-evaluator
"""
