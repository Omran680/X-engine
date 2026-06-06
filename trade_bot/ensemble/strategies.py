import numpy as np
from enum import Enum

class EnsembleStrategy(Enum):
    """Stratégies d'agrégation pour l'ensemble learning"""
    VOTING = "voting"                      # Vote majoritaire
    WEIGHTED_VOTING = "weighted_voting"    # Vote pondéré par confiance
    AVERAGING = "averaging"                # Moyenne des scores
    STACKING = "stacking"                  # Meta-learning
    MAJORITY_VOTING = "majority_voting"    # Strictement majoritaire


class EnsembleDecisionMaker:
    """Système de décision d'ensemble pour PPO+DQN"""
    
    def __init__(self, strategy=EnsembleStrategy.WEIGHTED_VOTING):
        self.strategy = strategy
        self.dqn_weight = 0.5
        self.ppo_weight = 0.5
        self.history = []

    def combine_predictions(self, dqn_output, ppo_output):
        """
        Combine DQN and PPO predictions
        
        Args:
            dqn_output: dict with 'action', 'q_values', 'confidence'
            ppo_output: dict with 'action', 'probs', 'value', 'confidence'
        
        Returns:
            dict with 'action', 'confidence', 'rationale'
        """
        if self.strategy == EnsembleStrategy.VOTING:
            return self._majority_voting(dqn_output, ppo_output)
        
        elif self.strategy == EnsembleStrategy.WEIGHTED_VOTING:
            return self._weighted_voting(dqn_output, ppo_output)
        
        elif self.strategy == EnsembleStrategy.AVERAGING:
            return self._averaging(dqn_output, ppo_output)
        
        elif self.strategy == EnsembleStrategy.STACKING:
            return self._stacking(dqn_output, ppo_output)
        
        elif self.strategy == EnsembleStrategy.MAJORITY_VOTING:
            return self._majority_voting_strict(dqn_output, ppo_output)

    def _majority_voting(self, dqn_output, ppo_output):
        """Simple majority voting"""
        dqn_action = dqn_output['action']
        ppo_action = ppo_output['action']
        
        if dqn_action == ppo_action:
            # Agreement
            confidence = (dqn_output['confidence'] + ppo_output['confidence']) / 2
            return {
                'action': dqn_action,
                'confidence': confidence,
                'rationale': 'AGREEMENT_BOTH_AGENTS',
                'breakdown': {'dqn': dqn_action, 'ppo': ppo_action}
            }
        else:
            # Disagreement - use higher confidence
            if dqn_output['confidence'] > ppo_output['confidence']:
                return {
                    'action': dqn_action,
                    'confidence': dqn_output['confidence'],
                    'rationale': 'DQN_HIGHER_CONFIDENCE',
                    'breakdown': {'dqn': dqn_action, 'ppo': ppo_action}
                }
            else:
                return {
                    'action': ppo_action,
                    'confidence': ppo_output['confidence'],
                    'rationale': 'PPO_HIGHER_CONFIDENCE',
                    'breakdown': {'dqn': dqn_action, 'ppo': ppo_action}
                }

    def _weighted_voting(self, dqn_output, ppo_output):
        """Weighted voting based on confidence and algorithm strength"""
        dqn_action = dqn_output['action']
        ppo_action = ppo_output['action']
        
        # Ajuster les poids basés sur la confiance
        dqn_score = dqn_output['confidence'] * self.dqn_weight
        ppo_score = ppo_output['confidence'] * self.ppo_weight
        
        # Créer un vecteur de votes pondérés
        action_size = 3  # BUY, SELL, HOLD
        action_votes = np.zeros(action_size)
        action_votes[dqn_action] += dqn_score
        action_votes[ppo_action] += ppo_score
        
        final_action = np.argmax(action_votes)
        final_confidence = np.max(action_votes) / (dqn_score + ppo_score + 1e-8)
        
        return {
            'action': final_action,
            'confidence': float(final_confidence),
            'rationale': 'WEIGHTED_VOTING',
            'breakdown': {
                'dqn': {'action': dqn_action, 'score': float(dqn_score)},
                'ppo': {'action': ppo_action, 'score': float(ppo_score)},
                'final': final_action
            }
        }

    def _averaging(self, dqn_output, ppo_output):
        """Average Q-values and policy probabilities"""
        dqn_q_values = dqn_output['q_values']  # Array [0-3]
        ppo_probs = ppo_output['probs']        # Array [0-3]
        
        # Normalize both to [0, 1]
        dqn_normalized = (dqn_q_values - np.min(dqn_q_values)) / (np.max(dqn_q_values) - np.min(dqn_q_values) + 1e-8)
        
        # Weighted average
        combined_scores = (dqn_normalized * self.dqn_weight) + (ppo_probs * self.ppo_weight)
        final_action = np.argmax(combined_scores)
        confidence = combined_scores[final_action]
        
        return {
            'action': final_action,
            'confidence': float(confidence),
            'rationale': 'AVERAGING',
            'breakdown': {
                'dqn_normalized': dqn_normalized.tolist(),
                'ppo_probs': ppo_probs.tolist(),
                'combined': combined_scores.tolist()
            }
        }

    def _stacking(self, dqn_output, ppo_output):
        """Meta-learner stacking approach"""
        # Combiner les features des deux algorithmes
        dqn_action = dqn_output['action']
        ppo_action = ppo_output['action']
        
        # Feature vector: [dqn_action_onehot, ppo_action_onehot, dqn_conf, ppo_conf, dqn_value, ppo_value]
        action_size = 3
        features = np.concatenate([
            np.eye(action_size)[dqn_action],
            np.eye(action_size)[ppo_action],
            [dqn_output['confidence']],
            [ppo_output['confidence']],
            [dqn_output.get('value', 0)],
            [ppo_output.get('value', 0)]
        ])
        
        # Simple meta-learner (can be replaced with trained neural network)
        meta_decision = self._meta_learner(features)
        
        return {
            'action': meta_decision,
            'confidence': 0.7,  # Meta-learner confidence
            'rationale': 'STACKING_META_LEARNER',
            'breakdown': {
                'dqn': dqn_action,
                'ppo': ppo_action,
                'meta': meta_decision
            }
        }

    def _majority_voting_strict(self, dqn_output, ppo_output):
        """Strict majority voting - only if confident agreement"""
        dqn_action = dqn_output['action']
        ppo_action = ppo_output['action']
        
        if dqn_action == ppo_action and dqn_output['confidence'] > 0.7 and ppo_output['confidence'] > 0.7:
            return {
                'action': dqn_action,
                'confidence': min(dqn_output['confidence'], ppo_output['confidence']),
                'rationale': 'CONFIDENT_AGREEMENT',
                'breakdown': {'dqn': dqn_action, 'ppo': ppo_action}
            }
        else:
            # Default to HOLD if no confident agreement
            return {
                'action': 2,  # HOLD
                'confidence': 0.5,
                'rationale': 'NO_CONFIDENT_AGREEMENT_HOLD',
                'breakdown': {'dqn': dqn_action, 'ppo': ppo_action}
            }

    def _meta_learner(self, features):
        """Simple meta-learner (can be enhanced)"""
        # Weighted sum of features
        weights = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        score = np.sum(features[:6] * weights)
        
        if score > 0.6:
            return int(features[0])  # Favor DQN decision
        elif score > 0.3:
            return int(features[3])  # Favor PPO decision
        else:
            return 2  # HOLD

    def update_weights(self, dqn_reward, ppo_reward):
        """Ajuster les poids en fonction des rewards"""
        total_reward = dqn_reward + ppo_reward + 1e-8
        self.dqn_weight = dqn_reward / total_reward
        self.ppo_weight = ppo_reward / total_reward
        
        self.history.append({
            'dqn_weight': self.dqn_weight,
            'ppo_weight': self.ppo_weight,
            'dqn_reward': dqn_reward,
            'ppo_reward': ppo_reward
        })

    def get_performance_metrics(self):
        """Retourner les métriques de performance"""
        if not self.history:
            return {}
        
        dqn_rewards = [h['dqn_reward'] for h in self.history]
        ppo_rewards = [h['ppo_reward'] for h in self.history]
        
        return {
            'avg_dqn_reward': np.mean(dqn_rewards),
            'avg_ppo_reward': np.mean(ppo_rewards),
            'dqn_variance': np.var(dqn_rewards),
            'ppo_variance': np.var(ppo_rewards),
            'current_dqn_weight': self.dqn_weight,
            'current_ppo_weight': self.ppo_weight
        }
