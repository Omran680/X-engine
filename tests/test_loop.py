import numpy as np
import pytest

import main as main_mod
from main import BUY, HOLD, SELL, XAUUSDHybridTrader


class FakeTrader:
    def __init__(self, prices):
        self.prices = list(prices)
        self.last = prices[0]
        self.positions = []
        self.opened, self.closed = [], []
        self.positions_fail = False
        self.price_fail = False

    def get_price(self, epic):
        if self.price_fail:
            raise RuntimeError("no price")
        if self.prices:
            self.last = self.prices.pop(0)
        return self.last

    def get_positions(self):
        if self.positions_fail:
            raise RuntimeError("positions api down")
        return self.positions

    def open_trade(self, **kw):
        self.opened.append(kw)
        self.positions.append(kw)
        return {"dealReference": "REF", "dealId": f"DEAL{len(self.opened)}"}

    def close_position(self, deal_id):
        self.closed.append(deal_id)


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "LOOP_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(XAUUSDHybridTrader, "_sleep", lambda self, seconds: None)
    monkeypatch.setattr(main_mod, "MODELS_DIR", str(tmp_path / "models"))


def make_bot(prices, dry_run=True, action=HOLD):
    bot = XAUUSDHybridTrader(use_live_data=False, dry_run=dry_run)
    bot.trader = FakeTrader(prices)
    bot.models_dir = None
    bot.ensemble.combine_predictions = lambda d, p: {"action": action}
    return bot


def flat(n, base=2000.0):
    return [base + 0.01 * (i % 3) for i in range(n)]


def test_no_price_means_no_decision_and_no_invented_price():
    bot = make_bot(flat(30))
    bot.trader.price_fail = True
    assert bot.get_current_price() is None            # old code answered 2000.0
    assert bot._step(0) is False and bot.price_history == []


def test_dry_run_position_is_virtual_and_closed_on_take_profit():
    bot = make_bot(flat(21) + [2000.0 * 1.0035], dry_run=True, action=BUY)
    for i in range(21):
        bot._step(i)
    assert bot.position is not None and bot.position["deal_id"] == "dry_run"
    assert len(bot.trader.opened) == 0                       # never touches the exchange
    bot.ensemble.combine_predictions = lambda d, p: {"action": HOLD}   # don't re-enter on the same bar
    bot._step(21)                                            # +0.35 % ≥ TP 0.3 %
    assert bot.position is None and bot.risk.today_pnl > 0
    assert bot.risk.get_risk_metrics()["num_trades"] == 1


def test_dry_run_does_not_reopen_every_step_while_in_position():
    bot = make_bot(flat(40), dry_run=True, action=SELL)
    for i in range(40):
        bot._step(i)
    # old code: live positions API said "flat" → a new dry-run order EVERY step
    assert bot.position is not None and bot.risk.get_risk_metrics()["num_trades"] == 0


def test_rl_transition_uses_realised_reward_and_next_state():
    prices = [2000.0] * 19 + [2000.0, 2002.0, 2004.0]
    bot = make_bot(prices, action=BUY)
    for i in range(len(prices)):
        bot._step(i)
    mem = list(bot.agent.dqn_agent.memory.buffer)
    assert len(mem) == 2
    s, a, r, s2, done = mem[-1]
    assert a == BUY and r > 0 and not done
    assert not np.array_equal(s, s2)                         # old code: next_state == state
    assert bot._episode_reward > 0


def test_training_mode_off_learns_nothing():
    bot = make_bot(flat(30))
    bot.agent.remember = lambda *a, **k: pytest.fail("must not learn")
    bot._training_mode = lambda: False
    for i in range(30):
        bot._step(i)


def test_history_is_bounded(monkeypatch):
    monkeypatch.setattr(main_mod, "MAX_HISTORY", 30)
    bot = make_bot(flat(100))
    for i in range(100):
        bot._step(i)
    assert len(bot.price_history) <= 30 and len(bot.volume_history) == len(bot.price_history)


def test_positions_api_outage_pauses_new_entries_instead_of_duplicating():
    bot = make_bot(flat(40), dry_run=False, action=BUY)
    bot.trader.positions_fail = True
    for i in range(30):
        bot._step(i)
    assert bot.trader.opened == []                           # state unknown → no blind orders
    bot.trader.positions_fail = False
    bot._step(30)
    assert len(bot.trader.opened) == 1                       # recovers


def test_live_open_uses_deal_id_and_sl_tp_distances():
    bot = make_bot(flat(25), dry_run=False, action=BUY)
    for i in range(25):
        bot._step(i)
    assert bot.position["deal_id"].startswith("DEAL")
    kw = bot.trader.opened[0]
    assert kw["sl"] == pytest.approx(2000 * 0.002, rel=0.01) and kw["tp"] == pytest.approx(2000 * 0.003, rel=0.01)


def test_broker_side_close_books_pnl_and_frees_the_slot():
    bot = make_bot(flat(25) + [2010.0], dry_run=False, action=BUY)
    for i in range(25):
        bot._step(i)
    bot.trader.positions.clear()                             # broker hit TP
    bot.ensemble.combine_predictions = lambda d, p: {"action": HOLD}
    bot._step(25)
    assert bot.position is None and bot.risk.get_risk_metrics()["num_trades"] == 1


def test_scalp_exit_is_sent_to_the_exchange_with_its_deal_id():
    bot = make_bot(flat(25) + [1990.0], dry_run=False)
    bot._scalper.enabled = True
    for i in range(25):
        bot._step(i)
    pos = bot._scalper.open_position("BUY", 2000.0, deal_id="SCALP9")
    bot.trader.positions.append({"scalp": True})
    bot._step(25)                                            # price 1990 → stop-loss
    assert bot.trader.closed == ["SCALP9"]                   # old code never closed it
    assert bot.risk.get_risk_metrics()["num_trades"] == 1


def test_scalp_uses_runtime_overridden_params_for_the_order():
    bot = make_bot(flat(25), dry_run=False)
    sc = bot._scalper
    sc.enabled = True
    sc.set_params(size=0.3, sl_pct=0.5, tp_pct=1.0)
    sc.evaluate = lambda p, v: type("S", (), dict(action="BUY", confidence=1, reason="t"))()
    for i in range(25):
        bot._step(i)
    kw = [o for o in bot.trader.opened if o["size"] == 0.3][0]
    assert kw["sl"] == pytest.approx(2000 * 0.005, rel=0.01)


def test_risk_manager_blocks_trading_after_daily_loss():
    bot = make_bot(flat(30), dry_run=True, action=BUY)
    bot.risk.update_pnl(-600.0)
    bot.risk.consecutive_losses = 0
    for i in range(30):
        bot._step(i)
    assert bot.position is None


def test_loop_survives_a_failing_step_and_stops_on_emergency_stop(monkeypatch):
    bot = make_bot(flat(60))
    calls = {"n": 0}
    real = bot._step

    def flaky(step):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(step)

    bot._step = flaky
    bot.live_trading_loop(max_steps=25)
    assert calls["n"] > 25                                   # kept going after the exception


def test_loop_aborts_after_too_many_consecutive_errors():
    bot = make_bot(flat(10))
    bot._step = lambda s: (_ for _ in ()).throw(RuntimeError("x"))
    with pytest.raises(RuntimeError, match="consecutive"):
        bot.live_trading_loop(max_steps=5)


def test_load_models_survives_incompatible_weights(tmp_path):
    from trade_bot.models.networks import DQNNetwork
    d = tmp_path / "m"
    DQNNetwork(7, 3).save(str(d / "dqn_model.npz"))
    bot = make_bot(flat(5))
    bot.models_dir = str(d)
    assert bot.load_models() is False                        # logged, not a crash loop


# ── market hours / outages ────────────────────────────────────────────────
from trade_bot.execution.trader import MarketClosedError


def closed_trader(trader, closed):
    real = trader.get_price

    def get_price(epic):
        if closed["on"]:
            raise MarketClosedError("CLOSED")
        return real(epic)
    trader.get_price = get_price


def test_closed_market_is_not_an_error_and_loop_keeps_idling(caplog, tmp_path):
    bot = make_bot(flat(200), dry_run=False, action=BUY)
    bot.models_dir = str(tmp_path)
    closed = {"on": False}
    closed_trader(bot.trader, closed)
    for i in range(25):
        bot._step(i)
    closed["on"] = True
    for i in range(500):                       # a whole weekend of polls
        assert bot._step(i) is False
    assert bot._market_open is False and bot._price_failures == 0
    assert bot._idle_seconds() == main_mod.MARKET_CLOSED_POLL_SECONDS
    assert len(bot.trader.opened) <= 1         # nothing new sent while closed
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_long_closure_polls_slower(monkeypatch):
    bot = make_bot(flat(5))
    bot._on_market_closed("CLOSED")
    bot._closed_since -= 7200
    assert bot._idle_seconds() == main_mod.MARKET_CLOSED_LONG_POLL_SECONDS


def test_resume_after_weekend_resets_history_and_pending_then_trades_again(monkeypatch):
    bot = make_bot(flat(300) , dry_run=True, action=HOLD)
    closed = {"on": False}
    closed_trader(bot.trader, closed)
    for i in range(25):
        bot._step(i)
    assert len(bot.price_history) >= 20 and bot._pending is not None
    closed["on"] = True
    bot._step(25)
    assert bot._pending is None                # no reward across a closure
    # simulate the clock jumping 2 days
    bot._last_price_time -= 2 * 86400
    bot._closed_since -= 2 * 86400
    closed["on"] = False
    bot._step(26)
    assert bot._market_open is True
    assert len(bot.price_history) == 1         # gap → history dropped, re-warming
    bot.ensemble.combine_predictions = lambda d, p: {"action": BUY}
    for i in range(27, 60):
        bot._step(i)
    assert bot.position is not None            # trades again once warmed up


def test_checkpoint_is_saved_when_market_closes(tmp_path):
    bot = make_bot(flat(30))
    bot.models_dir = str(tmp_path)
    bot._on_market_closed("CLOSED")
    assert (tmp_path / "dqn_model.npz").exists()


def test_outage_backoff_grows_and_is_capped(monkeypatch):
    monkeypatch.setattr(main_mod, "LOOP_INTERVAL_SECONDS", 5)
    bot = make_bot(flat(5))
    delays = []
    for _ in range(10):
        bot._price_failures += 1
        delays.append(bot._idle_seconds())
    assert delays == sorted(delays) and delays[-1] == main_mod.OUTAGE_MAX_BACKOFF_SECONDS


def test_heartbeat_written_even_when_market_closed(tmp_path, monkeypatch):
    hb = tmp_path / "hb.json"
    monkeypatch.setattr(main_mod, "HEARTBEAT_FILE", str(hb))
    bot = make_bot(flat(5))
    closed = {"on": True}
    closed_trader(bot.trader, closed)
    bot._on_market_closed("CLOSED")
    bot._write_heartbeat(3)
    import json
    data = json.loads(hb.read_text())
    assert data["market_open"] is False and data["market_status"] == "CLOSED"


def test_trader_raises_market_closed_without_retrying(monkeypatch):
    from tests.test_trader import FakeIG
    import trade_bot.execution.trader as tm
    monkeypatch.setenv("IG_USERNAME", "u"); monkeypatch.setenv("IG_PASSWORD", "p"); monkeypatch.setenv("IG_API_KEY", "k")
    calls = []

    class ClosedIG(FakeIG):
        def fetch_market_by_epic(self, epic):
            calls.append(1)
            return {"snapshot": {"bid": 2000.0, "marketStatus": "CLOSED"}}

    monkeypatch.setattr(tm, "IGService", ClosedIG)
    t = tm.Trader(); t._min_api_interval = 0
    with pytest.raises(MarketClosedError):
        t.get_price("E")
    assert len(calls) == 1                     # no 3× retry with sleeps
