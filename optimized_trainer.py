"""
OPTIMIZED TRAINING ORCHESTRATOR
Integrates all optimization components:
- Rich state features (48 features)
- Risk-adjusted rewards
- Curriculum learning
- Meta evaluation
- Transfer learning
- Paper trading validation
"""

import numpy as np
from environment_optimizer import EnrichedFeatureExtractor, RiskAdjustedRewardCalculator
from training_curriculum import (
    MarketRegimeDetector, DifficultyScheduler, RegimeAwareTrainer, TrainingProgression
)
from meta_evaluator import MetaEvaluator, TransferLearner, PaperTradingEngine
from advanced_agent import AdvancedHybridTradingAgent
from typing import Dict, List, Tuple


class OptimizedTrainer:
    """
    Complete training orchestrator with all optimizations:
    
    1. Enhanced Environment
       - 48 rich features (technical indicators, volatility, volume)
       - Risk-adjusted rewards (Sharpe-optimized)
    
    2. Curriculum Learning
       - Phase 1: Bull market (learn profitability)
       - Phase 2: Mixed markets (learn adaptation)
       - Phase 3: Bear/volatility (learn risk management)
       - Phase 4: All conditions (robustness)
    
    3. Continuous Evaluation
       - Track Sharpe, Win Rate, Drawdown per agent
       - Adjust ensemble weights dynamically
       - Detect degradation, recommend switching
    
    4. Transfer Learning
       - Initialize PPO from DQN features
       - Warm-start from pretrained
    
    5. Paper Trading
       - Validate before live deployment
       - 20+ days simulation
    """
    
    def __init__(self, agent: AdvancedHybridTradingAgent, total_episodes=1000):
        """
        Initialize optimized trainer
        
        Args:
            agent: AdvancedHybridTradingAgent
            total_episodes: Total training episodes
        """
        
        # Core agent
        self.agent = agent
        self.total_episodes = total_episodes
        
        # 1. Rich environment
        self.feature_extractor = EnrichedFeatureExtractor(lookback=60)
        self.reward_calculator = RiskAdjustedRewardCalculator(
            alpha=1.0,    # Profit weight
            beta=0.3,     # Drawdown penalty
            gamma=0.1     # Volatility penalty
        )
        
        # 2. Curriculum learning
        self.curriculum = TrainingProgression(total_episodes)
        self.regime_detector = MarketRegimeDetector()
        self.regime_trainer = RegimeAwareTrainer()
        
        # 3. Meta evaluation
        self.meta_evaluator = MetaEvaluator(window_size=100)
        
        # 4. Transfer learning
        self.transfer_learner = TransferLearner()
        
        # Tracking
        self.episode_count = 0
        self.step_count = 0
        self.current_regime = None
    
    def prepare_state(self, price: float, volume: float = None) -> np.ndarray:
        """
        Prepare enriched state from market data
        
        Returns: 48-feature vector
        """
        self.feature_extractor.add_data(price, volume or 1000)
        return self.feature_extractor.extract_48_features()
    
    def calculate_risk_adjusted_reward(self,
                                      trade_pnl: float,
                                      entry_volatility: float,
                                      current_drawdown: float,
                                      holding_steps: int,
                                      is_win: bool) -> float:
        """
        Calculate risk-adjusted reward
        
        Reward = α*Profit - β*Drawdown - γ*Volatility + adjustments
        
        Returns: Reward in range [-0.1, 0.1]
        """
        return self.reward_calculator.calculate_reward(
            trade_pnl,
            entry_volatility,
            current_drawdown,
            holding_steps,
            is_win
        )
    
    def get_current_difficulty(self) -> Dict:
        """Get current training difficulty level"""
        return self.curriculum.difficulty_scheduler.get_difficulty_config()
    
    def update_after_episode(self, 
                            episode: int,
                            total_reward: float,
                            sharpe_ratio: float,
                            win_rate: float,
                            regime = None):
        """
        Update training state after episode
        
        Args:
            episode: Episode number
            total_reward: Total episode reward
            sharpe_ratio: Episode Sharpe ratio
            win_rate: Winning trades %
            regime: Market regime
        """
        
        self.episode_count = episode
        self.curriculum.update_episode(
            episode, total_reward, sharpe_ratio, win_rate, regime
        )
        
        # Update regime tracking
        if regime:
            self.current_regime = regime
            self.regime_trainer.log_performance(regime, "HYBRID", sharpe_ratio)
    
    def evaluate_and_adjust(self) -> Dict:
        """
        Evaluate agents and adjust ensemble weights
        
        Returns:
            Adjustment report with recommendations
        """
        eval_report = self.meta_evaluator.get_evaluation_report()
        
        new_weights = eval_report['recommended_weights']
        
        print(f"\n📊 Meta-Evaluation Report")
        print(f"   DQN Sharpe: {eval_report['dqn_metrics']['sharpe']:.3f}")
        print(f"   PPO Sharpe: {eval_report['ppo_metrics']['sharpe']:.3f}")
        print(f"   Recommended Weights: DQN {new_weights['dqn_weight']:.1%}, PPO {new_weights['ppo_weight']:.1%}")
        print(f"   Meta-Score: {eval_report['meta_score']:.3f}")
        
        if eval_report['should_switch']:
            print(f"   ⚠️ Performance gap detected - consider switching to {eval_report['switch_to']}")
        
        return eval_report
    
    def apply_transfer_learning(self):
        """Apply transfer learning from DQN to PPO"""
        print("\n🔄 Applying Transfer Learning...")
        
        success = self.transfer_learner.transfer_dqn_to_ppo(
            self.agent.dqn,
            self.agent.ppo
        )
        
        return success
    
    def run_paper_trading(self, prices: List[float], volumes: List[float],
                         days: int = 20) -> Dict:
        """
        Validate agent with paper trading
        
        Args:
            prices: Historical prices
            volumes: Historical volumes
            days: Days to simulate
        
        Returns:
            Paper trading report
        """
        
        print(f"\n🟢 Running Paper Trading Validation")
        
        engine = PaperTradingEngine(
            self.agent,
            self.feature_extractor,
            prices,
            volumes,
            transaction_cost=0.001
        )
        
        report = engine.run_paper_trading(days)
        
        print(f"\n✅ Paper Trading Complete")
        print(f"   Final Return: {report.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {report.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {report.get('max_drawdown', 0):.2%}")
        
        return report
    
    def get_training_summary(self) -> Dict:
        """Get overall training progress summary"""
        
        meta_score = self.curriculum.get_meta_score()
        difficulty = self.curriculum.difficulty_scheduler.get_difficulty_config()
        
        return {
            'episode': self.episode_count,
            'total_episodes': self.total_episodes,
            'progress': self.episode_count / self.total_episodes,
            'meta_score': meta_score['meta_score'],
            'avg_sharpe': meta_score['avg_sharpe'],
            'avg_win_rate': meta_score['avg_win_rate'],
            'stability': meta_score['stability'],
            'difficulty': difficulty['name'],
            'current_regime': self.current_regime.name if self.current_regime else 'UNKNOWN'
        }
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint"""
        print(f"\n💾 Saving checkpoint to {checkpoint_path}")
        self.agent.save_models(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        print(f"\n📂 Loading checkpoint from {checkpoint_path}")
        self.agent.load_models(checkpoint_path)


class TrainingStrategy:
    """
    High-level training strategy wrapper
    Shows best practices for training with optimization
    """
    
    @staticmethod
    def recommended_training_pipeline(agent: AdvancedHybridTradingAgent,
                                     total_episodes: int = 1000,
                                     checkpoint_interval: int = 100) -> 'OptimizedTrainer':
        """
        Recommended training pipeline:
        1. Start with curriculum learning (easy → hard)
        2. Apply transfer learning early
        3. Continuous meta-evaluation
        4. Paper trading at checkpoints
        5. Save best models
        
        Returns:
            OptimizedTrainer configured with best practices
        """
        
        trainer = OptimizedTrainer(agent, total_episodes)
        
        print("""
╔════════════════════════════════════════════════════════════╗
║       OPTIMIZED TRAINING PIPELINE INITIALIZED              ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  ✅ Phase 1: Environment Optimization                     ║
║     - 48 rich features (tech + sentiment + vol)          ║
║     - Risk-adjusted rewards (Sharpe formula)             ║
║                                                            ║
║  ✅ Phase 2: Curriculum Learning                          ║
║     - Episode 0-150:   EASY (Bull market)               ║
║     - Episode 150-400: MEDIUM (Mixed trends)            ║
║     - Episode 400-750: HARD (High volatility)            ║
║     - Episode 750+:    EXPERT (All conditions)            ║
║                                                            ║
║  ✅ Phase 3: Meta Evaluation                              ║
║     - Continuous performance tracking                     ║
║     - Dynamic weight adjustment                           ║
║     - Degradation detection                               ║
║                                                            ║
║  ✅ Phase 4: Transfer Learning                            ║
║     - DQN → PPO feature sharing                           ║
║     - Warm-start acceleration                             ║
║                                                            ║
║  ✅ Phase 5: Paper Trading                                ║
║     - Validation on historical data                       ║
║     - 20+ day simulation before live                      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        return trainer
    
    @staticmethod
    def get_training_loop_template():
        """Get template for main training loop"""
        
        return """
# Main training loop with optimized components
trainer = OptimizedTrainer(agent, total_episodes=1000)
agent = trainer.agent

for episode in range(1000):
    # 1. Get current difficulty
    difficulty = trainer.get_current_difficulty()
    print(f"Difficulty: {difficulty['name']}")
    
    # RL Training Loop
    state = trainer.prepare_state(price=1800.0, volume=5000)
    done = False
    
    while not done:
        # 2. Agent makes decision
        decision = agent.act(state)
        
        # 3. Simulate trade outcome
        # ... trading logic ...
        
        # 4. Calculate RICH reward (not just profit!)
        reward = trainer.calculate_risk_adjusted_reward(
            trade_pnl=0.01,
            entry_volatility=0.02,
            current_drawdown=-0.05,
            holding_steps=10,
            is_win=True
        )
        
        # 5. Train both agents
        agent.train_step_fn(state, action, reward, next_state, done)
        
        state = next_state
    
    # 6. Update training curriculum
    trainer.update_after_episode(
        episode=episode,
        total_reward=episode_reward,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        regime=detected_regime
    )
    
    # 7. Every 100 episodes: evaluate and adjust
    if episode % 100 == 0:
        eval_report = trainer.evaluate_and_adjust()
        trainer.save_checkpoint(f"./models/checkpoint_{episode}")
    
    # 8. At end: paper trading validation
    if episode == 999:
        paper_report = trainer.run_paper_trading(
            prices=historical_prices,
            volumes=historical_volumes,
            days=20
        )
        
        if paper_report['final_equity'] > 10500:
            agent.save_models('./models/final')
        """


# Example usage
if __name__ == "__main__":
    
    # Create agent
    agent = AdvancedHybridTradingAgent(
        state_size=48,  # Now using 48 features!
        action_size=3,
        ensemble_type="voting"
    )
    
    # Create optimized trainer
    trainer = TrainingStrategy.recommended_training_pipeline(
        agent,
        total_episodes=1000,
        checkpoint_interval=100
    )
    
    # Show training loop template
    print("\n📚 Training Loop Template:")
    print(TrainingStrategy.get_training_loop_template())
    
    print("\n✅ Trainer initialized. Ready to train with optimization!")
