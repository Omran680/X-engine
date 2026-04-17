"""
Example: Integration of Advanced Hybrid Trading System

This file shows how to use the new AdvancedHybridTradingAgent
with Dueling DQN, Risk-Aware PPO, and Voting Ensemble.
"""

import numpy as np
from advanced_agent import AdvancedHybridTradingAgent
from feature_extractor import FeatureExtractor, TradingState
from collections import deque


class AdvancedHybridTraderExample:
    """Example usage of the new advanced trading system"""
    
    def __init__(self, ensemble_type="voting", use_live_data=True):
        """
        Initialize advanced trader
        
        Args:
            ensemble_type: "voting" or "hierarchical"
            use_live_data: Whether to connect to live market data
        """
        
        # Initialize the advanced agent
        self.agent = AdvancedHybridTradingAgent(
            state_size=18,
            action_size=3,
            ensemble_type=ensemble_type
        )
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(lookback_window=20)
        self.trading_state = TradingState()
        
        # Market data tracking
        self.price_history = deque(maxlen=100)
        self.returns_history = deque(maxlen=60)  # For risk calculation
        self.equity_history = []
        self.current_equity = 10000.0
        self.max_equity = 10000.0
        
        # Trading metrics
        self.trades = []
        self.position = None
        self.entry_price = None
        
        # Non-live mode: simulate prices
        self.use_live_data = use_live_data
        if not use_live_data:
            self.simulated_price = 1800.0
            self.price_step = np.random.normal(0, 2)
    
    def get_market_state(self, price, volume=None):
        """Extract features from current price"""
        # Add to history
        self.price_history.append(price)
        
        # Calculate returns for risk metrics
        if len(self.price_history) > 1:
            ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(ret)
        
        # Extract features
        features = self.feature_extractor.extract_features(
            prices=list(self.price_history),
            volumes=[volume] if volume else None,
            trading_state=self.trading_state
        )
        
        return features
    
    def calculate_risk_metrics(self):
        """Calculate current drawdown and volatility"""
        if len(self.equity_history) < 2:
            return 0.0
        
        equity_array = np.array(self.equity_history)
        drawdown = (equity_array[-1] - np.max(equity_array)) / (np.max(equity_array) + 1e-8)
        drawdown = abs(drawdown)
        
        return drawdown
    
    def execute_trade(self, decision, price, position_size=0.1):
        """Execute trading decision"""
        action = decision['action']
        confidence = decision['confidence']
        source = decision['source']
        
        print(f"\n{'='*60}")
        print(f"Trading Decision (Step {len(self.trades)})")
        print(f"{'='*60}")
        print(f"Price: ${price:.2f}")
        print(f"Action: {['HOLD', 'BUY', 'SELL'][action]}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Source: {source}")
        
        if 'ensemble_stats' in decision:
            stats = decision['ensemble_stats']
            print(f"\nEnsemble Statistics:")
            print(f"  DQN Sharpe: {stats['dqn_sharpe']:.3f}")
            print(f"  PPO Sharpe: {stats['ppo_sharpe']:.3f}")
            print(f"  Agreement Rate: {stats['agreement_rate']:.1%}")
            print(f"  DQN Win Rate: {stats['dqn_win_rate']:.1%}")
            print(f"  PPO Win Rate: {stats['ppo_win_rate']:.1%}")
        
        # Execute action
        if action == 1:  # BUY
            if self.position is None:
                self.position = position_size
                self.entry_price = price
                print(f"\n✓ BUY SIGNAL: Entering position (size: {position_size})")
        
        elif action == 2:  # SELL
            if self.position is not None:
                pnl = (price - self.entry_price) * self.position
                pnl_pct = (price - self.entry_price) / self.entry_price
                self.current_equity += pnl
                
                trade_record = {
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': confidence,
                    'source': source
                }
                self.trades.append(trade_record)
                
                print(f"\n✓ SELL SIGNAL: Closing position")
                print(f"  Entry: ${self.entry_price:.2f}")
                print(f"  Exit: ${price:.2f}")
                print(f"  PnL: ${pnl:.2f} ({pnl_pct:.2%})")
                
                self.position = None
                self.entry_price = None
        
        print(f"Equity: ${self.current_equity:.2f}")
        print(f"{'='*60}")
    
    def train_step(self, price, volume=None):
        """Single training step"""
        
        # Get market state
        state = self.get_market_state(price, volume)
        
        # Calculate risk metrics
        drawdown = self.calculate_risk_metrics()
        
        # Get trading decision from ensemble
        decision = self.agent.act(
            state=state,
            market_metrics={
                'returns': list(self.returns_history),
                'drawdown': drawdown
            },
            use_ensemble=True
        )
        
        # Execute trade
        self.execute_trade(decision, price)
        
        # Store equity
        self.equity_history.append(self.current_equity)
        self.max_equity = max(self.max_equity, self.current_equity)
        
        # Simulate reward and train
        if len(self.trades) > 0:
            recent_pnl = self.trades[-1]['pnl']
            reward = recent_pnl / abs(self.entry_price) if self.entry_price else 0
        else:
            reward = 0.0
        
        # Train both agents
        next_state = self.get_market_state(price, volume) if len(self.price_history) > 1 else state
        done = False
        
        self.agent.train_step_fn(
            state=state,
            action=decision['action'],
            reward=reward,
            next_state=next_state,
            done=done,
            batch_size=32
        )
    
    def simulate_prices(self, num_steps=100):
        """Simulate price data and run trading"""
        print("\n" + "="*60)
        print("SIMULATED TRADING WITH ADVANCED HYBRID AGENT")
        print("="*60)
        print(f"Ensemble Type: voting")
        print(f"Starting Equity: ${self.current_equity:.2f}")
        print("="*60 + "\n")
        
        for step in range(num_steps):
            # Generate simulated price (random walk with drift)
            self.price_step += np.random.normal(0, 1) * 0.1
            price_change = 1 + self.price_step / 1000
            self.simulated_price *= price_change
            volume = np.random.randint(1000, 5000)
            
            # Execute trading step
            self.train_step(self.simulated_price, volume)
            
            # Occasional printout
            if (step + 1) % 10 == 0:
                print(f"\n📊 Progress: {step + 1}/{num_steps} steps")
                print(f"   Trades closed: {len(self.trades)}")
                if self.trades:
                    recent_returns = [t['pnl_pct'] for t in self.trades[-5:]]
                    avg_recent = np.mean(recent_returns) if recent_returns else 0
                    print(f"   Recent avg PnL: {avg_recent:.2%}")
    
    def get_performance_summary(self):
        """Generate performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if len(self.trades) == 0:
            print("No completed trades")
            return
        
        pnl_list = [t['pnl'] for t in self.trades]
        pnl_pct_list = [t['pnl_pct'] for t in self.trades]
        
        total_pnl = sum(pnl_list)
        wins = sum(1 for pnl in pnl_list if pnl > 0)
        losses = sum(1 for pnl in pnl_list if pnl < 0)
        
        print(f"Total Trades: {len(self.trades)}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {wins/len(self.trades):.1%}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Avg PnL per Trade: ${np.mean(pnl_list):.2f}")
        print(f"Total Return: {(self.current_equity - 10000) / 10000:.2%}")
        print(f"Max Drawdown: {(1 - self.current_equity / self.max_equity):.2%}")
        
        if len(pnl_pct_list) > 1:
            sharpe = np.mean(pnl_pct_list) / (np.std(pnl_pct_list) + 1e-8)
            print(f"Sharpe Ratio: {sharpe:.3f}")
        
        print("="*60)


if __name__ == "__main__":
    """Example usage"""
    
    # Create trader with advanced agent
    trader = AdvancedHybridTraderExample(
        ensemble_type="voting",  # or "hierarchical"
        use_live_data=False
    )
    
    # Simulate trading
    trader.simulate_prices(num_steps=50)
    
    # Print summary
    trader.get_performance_summary()
    
    # Save models
    print("\nSaving models...")
    trader.agent.save_models("./models")
