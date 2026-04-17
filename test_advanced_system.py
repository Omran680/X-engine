#!/usr/bin/env python3
"""
Test d'intégrité du système avancé
Vérifie que tous les composants fonctionnent ensemble
"""

import numpy as np
import sys


def test_imports():
    """Test que tous les imports fonctionnent"""
    print("=" * 60)
    print("TEST 1: Vérification des imports")
    print("=" * 60)
    
    try:
        from model import DuelingDQNNetwork, RiskAwarePPONetwork
        print("✅ model.py - Architectures améliorées")
    except ImportError as e:
        print(f"❌ model.py - ERREUR: {e}")
        return False
    
    try:
        from ensemble_manager import (
            PerformanceTracker,
            VotingEnsembleManager,
            HierarchicalEnsembleManager,
            RiskAwareModeManager
        )
        print("✅ ensemble_manager.py - 4 managers")
    except ImportError as e:
        print(f"❌ ensemble_manager.py - ERREUR: {e}")
        return False
    
    try:
        from advanced_agent import (
            DuelingDQNAgent,
            RiskAwarePPOAgent,
            AdvancedHybridTradingAgent
        )
        print("✅ advanced_agent.py - 3 agents")
    except ImportError as e:
        print(f"❌ advanced_agent.py - ERREUR: {e}")
        return False
    
    print()
    return True


def test_dqn_network():
    """Test l'architecture Dueling DQN"""
    print("=" * 60)
    print("TEST 2: Architecture Dueling DQN")
    print("=" * 60)
    
    try:
        from model import DuelingDQNNetwork
        
        net = DuelingDQNNetwork(state_size=18, action_size=3)
        state = np.random.randn(1, 18).astype(np.float32)
        
        q_values = net.forward(state)
        print(f"✅ Forward pass: {q_values.shape}")
        
        q_pred = net.predict(state)
        print(f"✅ Predict: {q_pred.shape}")
        
        params = net.get_params()
        print(f"✅ Get params: {len(params)} tensors")
        
        net.set_params(params)
        print(f"✅ Set params: OK")
        
        net_copy = net.copy()
        print(f"✅ Copy network: OK")
        
        print()
        return True
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_network():
    """Test l'architecture Risk-Aware PPO"""
    print("=" * 60)
    print("TEST 3: Architecture Risk-Aware PPO")
    print("=" * 60)
    
    try:
        from model import RiskAwarePPONetwork
        
        net = RiskAwarePPONetwork(state_size=18, action_size=3)
        state = np.random.randn(1, 18).astype(np.float32)
        
        logits, values, risk = net.forward(state)
        print(f"✅ Forward pass:")
        print(f"   - Logits: {logits.shape}")
        print(f"   - Values: {values.shape}")
        print(f"   - Risk: {risk.shape}")
        
        params = net.get_params()
        print(f"✅ Get params: {len(params)} tensors")
        
        print()
        return True
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agents():
    """Test les agents"""
    print("=" * 60)
    print("TEST 4: Agents (DQN + PPO)")
    print("=" * 60)
    
    try:
        from advanced_agent import DuelingDQNAgent, RiskAwarePPOAgent
        
        state_size = 18
        action_size = 3
        
        # Test DQN
        dqn = DuelingDQNAgent(state_size, action_size)
        s = np.random.randn(1, state_size).astype(np.float32)
        a = dqn.act(s)
        print(f"✅ DuelingDQNAgent.act(): action={a}")
        
        dqn.remember(s, a, 1.0, s, False)
        dqn.replay(batch_size=1)
        print(f"✅ DuelingDQNAgent.replay(): OK")
        
        # Test PPO
        ppo = RiskAwarePPOAgent(state_size, action_size)
        a, log_prob, value = ppo.act(s)
        print(f"✅ RiskAwarePPOAgent.act(): action={a}, log_prob={log_prob:.3f}, value={value:.3f}")
        
        ppo.remember(s, a, 1.0, value, log_prob, risk_target=0.05)
        ppo.update()
        print(f"✅ RiskAwarePPOAgent.update(): OK")
        
        print()
        return True
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_managers():
    """Test les managers d'ensemble"""
    print("=" * 60)
    print("TEST 5: Ensemble Managers")
    print("=" * 60)
    
    try:
        from ensemble_manager import (
            PerformanceTracker,
            VotingEnsembleManager,
            HierarchicalEnsembleManager,
            RiskAwareModeManager
        )
        
        # Test PerformanceTracker
        tracker = PerformanceTracker(window_size=30)
        tracker.log_trade("DQN", 1, 0.02, 0.05)
        tracker.log_trade("PPO", 2, 0.01, 0.03)
        dqn_sharpe = tracker.get_sharpe_ratio("DQN")
        print(f"✅ PerformanceTracker: DQN Sharpe={dqn_sharpe:.3f}")
        
        # Test VotingEnsembleManager
        voting = VotingEnsembleManager(18, 3)
        dqn_q = np.array([1.0, 2.0, 0.5])
        ppo_probs = np.array([0.2, 0.6, 0.2])
        decision = voting.vote(1, dqn_q, 1, ppo_probs)
        print(f"✅ VotingEnsembleManager: decision={decision['action']}, source={decision['source']}")
        
        # Test HierarchicalEnsembleManager
        hier = HierarchicalEnsembleManager(18, 3)
        mode = hier.decide_mode(dqn_action=1, dqn_confidence=0.8)
        print(f"✅ HierarchicalEnsembleManager: mode={mode}")
        
        # Test RiskAwareModeManager
        risk_mgr = RiskAwareModeManager()
        risk_mgr.update_market_metrics([0.01, 0.02, -0.01], 0.1)
        weights = risk_mgr.get_adaptive_weights()
        print(f"✅ RiskAwareModeManager: DQN={weights['dqn_weight']:.2%}, PPO={weights['ppo_weight']:.2%}")
        
        print()
        return True
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_agent():
    """Test l'agent hybride complet"""
    print("=" * 60)
    print("TEST 6: Advanced Hybrid Trading Agent")
    print("=" * 60)
    
    try:
        from advanced_agent import AdvancedHybridTradingAgent
        
        # Test mode voting
        agent = AdvancedHybridTradingAgent(
            state_size=18,
            action_size=3,
            ensemble_type="voting"
        )
        print(f"✅ AdvancedHybridTradingAgent (voting): created")
        
        # Test décision
        state = np.random.randn(1, 18).astype(np.float32)
        decision = agent.act(
            state,
            market_metrics={'returns': [0.01, 0.02, -0.01], 'drawdown': 0.05}
        )
        print(f"✅ Agent.act(): action={decision['action']}, source={decision['source']}")
        
        # Test entraînement
        agent.train_step_fn(state, 1, 0.01, state, False, batch_size=8)
        print(f"✅ Agent.train_step_fn(): OK")
        
        # Test mode hierarchical
        agent_h = AdvancedHybridTradingAgent(
            state_size=18,
            ensemble_type="hierarchical"
        )
        decision_h = agent_h.act(state)
        print(f"✅ AdvancedHybridTradingAgent (hierarchical): action={decision_h['action']}")
        
        print()
        return True
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test l'intégration complète"""
    print("=" * 60)
    print("TEST 7: Intégration Complète (10 steps)")
    print("=" * 60)
    
    try:
        from advanced_agent import AdvancedHybridTradingAgent
        
        agent = AdvancedHybridTradingAgent(state_size=18, ensemble_type="voting")
        
        for step in range(10):
            state = np.random.randn(1, 18).astype(np.float32)
            next_state = np.random.randn(1, 18).astype(np.float32)
            
            decision = agent.act(state, market_metrics={'returns': [0.01], 'drawdown': 0.01})
            agent.train_step_fn(state, decision['action'], 0.001, next_state, False)
            
            if (step + 1) % 3 == 0:
                stats = decision.get('ensemble_stats', {})
                print(f"  Step {step+1}: action={decision['action']}, "
                      f"agreement={stats.get('agreement_rate', 0):.1%}")
        
        print("✅ Integration test: 10 steps completed")
        print()
        return True
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "TEST D'INTÉGRITÉ: SYSTÈME DE TRADING AVANCÉ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        test_imports,
        test_dqn_network,
        test_ppo_network,
        test_agents,
        test_ensemble_managers,
        test_hybrid_agent,
        test_integration,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur fatale dans {test_func.__name__}: {e}")
            results.append(False)
    
    # Résumé
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅" if result else "❌"
        print(f"{status} {test_func.__name__}")
    
    print()
    print(f"RÉSULTAT: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        print("\n🚀 Le système est prêt à l'emploi!")
        print("\nCommande suivante:")
        print("  python3 advanced_example.py")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) échoué(s)")
        print("\nVérifiez les erreurs ci-dessus")
        return 1


if __name__ == "__main__":
    sys.exit(main())
