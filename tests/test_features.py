import numpy as np

from main import BUY, HOLD, SELL, XAUUSDHybridTrader, q_confidence, step_reward
from trade_bot.core.config import STATE_SIZE
from trade_bot.features import FeatureExtractor


def _prices(base=2000.0, n=20, seed=0):
    rng = np.random.RandomState(seed)
    return base + np.cumsum(rng.randn(n))


def test_state_shape_and_bounds_with_and_without_volume():
    fx = FeatureExtractor(20)
    for vol in (np.full(20, 1000.0), None):
        f = fx.extract_features(_prices(), vol)
        assert f.shape == (1, STATE_SIZE)
        assert np.all(np.abs(f) <= 1.0) and np.all(np.isfinite(f))


def test_features_are_scale_free():
    fx = FeatureExtractor(20)
    p = _prices(2000.0)
    a = fx.extract_features(p, np.full(20, 1000.0))
    b = fx.extract_features(p * 1.5, np.full(20, 1000.0))     # same shape of move, other price level
    assert np.allclose(a, b, atol=1e-3)


def test_no_single_feature_dominates():
    f = FeatureExtractor(20).extract_features(_prices(), np.full(20, 1000.0))[0]
    # old behaviour: RSI/stochastic saturated at ±1 while everything else was ~0
    assert (np.abs(f) > 0.01).sum() >= 8


def test_flat_prices_do_not_produce_nan():
    f = FeatureExtractor(20).extract_features(np.full(20, 2000.0), np.full(20, 1000.0))
    assert np.all(np.isfinite(f))


# ── reward / confidence helpers ──────────────────────────────────────────
def test_reward_depends_on_the_action_taken():
    up = 0.001
    assert step_reward(BUY, up) > 0 > step_reward(SELL, up)
    assert step_reward(HOLD, up) == 0 == step_reward(HOLD, -up)
    assert step_reward(SELL, -up) > 0 > step_reward(BUY, -up)


def test_q_confidence_is_comparable_to_a_probability():
    assert 1 / 3 - 1e-9 <= q_confidence(np.array([5.0, -3.0, 0.1])) <= 1.0
    assert abs(q_confidence(np.array([1.0, 1.0, 1.0])) - 1 / 3) < 1e-6   # indifferent agent
    assert q_confidence(np.array([5.0, 0.0, 0.0])) > q_confidence(np.array([0.5, 0.4, 0.0]))


def test_majority_vote_ties_resolve_to_hold():
    mv = XAUUSDHybridTrader.majority_vote
    assert mv([BUY, BUY, SELL]) == BUY
    assert mv([BUY, SELL, HOLD]) == HOLD
    assert mv([SELL, SELL, SELL]) == SELL
