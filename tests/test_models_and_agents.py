import numpy as np
import pytest

from trade_bot.agents.base import DQNAgent, HybridTradingAgent, PPOAgent
from trade_bot.ensemble import EnsembleDecisionMaker, EnsembleStrategy
from trade_bot.models.networks import DQNNetwork, PPONetwork


def test_weights_roundtrip_without_pickle(tmp_path):
    net = DQNNetwork(18, 3, seed=1)
    path = str(tmp_path / "dqn.npz")
    net.save(path)
    other = DQNNetwork(18, 3, seed=2)
    other.load(path)
    assert np.allclose(net.w1, other.w1) and np.allclose(net.b3, other.b3)
    # allow_pickle=False in the loader: an object array must be rejected, not executed
    np.savez(str(tmp_path / "evil.npz"), w1=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        other.load(str(tmp_path / "evil.npz"))


def test_load_rejects_wrong_shape(tmp_path):
    path = str(tmp_path / "ppo.npz")
    PPONetwork(10, 3).save(path)
    with pytest.raises(ValueError, match="shape"):
        PPONetwork(18, 3).load(path)


def test_load_missing_file_keeps_weights(tmp_path):
    net = DQNNetwork(18, 3, seed=3)
    before = net.w1.copy()
    net.load(str(tmp_path / "nope.npz"))
    assert np.array_equal(before, net.w1)


def test_save_is_atomic_no_tmp_left(tmp_path):
    DQNNetwork(18, 3).save(str(tmp_path / "a.npz"))
    assert [p.name for p in tmp_path.iterdir()] == ["a.npz"]


def test_dqn_replay_learns_a_simple_reward():
    rng = np.random.RandomState(0)
    agent = DQNAgent(state_size=4, action_size=3, lr=0.01)
    s = rng.randn(1, 4).astype(np.float32)
    for _ in range(200):
        agent.remember(s, 0, 1.0, s, True)     # action 0 always pays 1
        agent.remember(s, 1, -1.0, s, True)
    for _ in range(300):
        agent.replay(32)
    q = agent.predict(s)
    assert np.all(np.isfinite(q)) and q[0] > q[1]


def test_ppo_clipped_update_is_finite_and_moves_weights():
    rng = np.random.RandomState(0)
    agent = PPOAgent(state_size=6, action_size=3)
    before = agent.model.actor_w.copy()
    for _ in range(64):
        s = rng.randn(1, 6).astype(np.float32)
        a, lp, v, _ = agent.act(s)
        agent.store_transition(s, a, 1.0 if a == 0 else -1.0, v, lp)
    adv, ret, states = agent.compute_gae(0.0)
    agent.train(adv, ret, states)
    assert np.all(np.isfinite(agent.model.actor_w))
    assert not np.array_equal(before, agent.model.actor_w)
    assert len(agent.reward_buffer) == 0


def test_hybrid_save_load_roundtrip(tmp_path):
    a = HybridTradingAgent(state_size=18, action_size=3)
    a.save_models(str(tmp_path))
    b = HybridTradingAgent(state_size=18, action_size=3)
    b.load_models(str(tmp_path))
    assert np.allclose(a.dqn_agent.model.w2, b.dqn_agent.model.w2)
    assert np.allclose(a.ppo_agent.model.critic_w, b.ppo_agent.model.critic_w)


# ── ensemble ──────────────────────────────────────────────────────────────
DQN = {"action": 0, "q_values": np.array([0.3, 0.1, 0.0]), "confidence": 0.8}
PPO = {"action": 1, "probs": np.array([0.2, 0.6, 0.2]), "confidence": 0.6, "value": 0.1}


@pytest.mark.parametrize("strategy", list(EnsembleStrategy))
def test_every_strategy_returns_a_valid_python_action(strategy):
    out = EnsembleDecisionMaker(strategy).combine_predictions(DQN, PPO)
    assert out["action"] in (0, 1, 2) and isinstance(out["action"], int)


def test_stacking_follows_the_more_confident_voter():
    d = EnsembleDecisionMaker(EnsembleStrategy.STACKING).combine_predictions(DQN, PPO)
    assert d["action"] == 0    # DQN more confident; was an index of a one-hot vector before


def test_update_weights_stay_a_distribution_with_negative_rewards():
    e = EnsembleDecisionMaker()
    e.update_weights(-2.0, -0.5)
    assert 0 <= e.dqn_weight <= 1 and abs(e.dqn_weight + e.ppo_weight - 1) < 1e-9
