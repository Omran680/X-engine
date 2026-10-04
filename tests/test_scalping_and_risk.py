import datetime as dt
import time

import numpy as np
import pytest

from trade_bot.agents.scalping import ScalpingAgent
from trade_bot.execution import risk as risk_mod
from trade_bot.execution.risk import PortfolioRiskManager


def _agent(**kw):
    a = ScalpingAgent()
    a.enabled = True
    a.set_params(**kw)
    return a


def test_tick_exposes_the_closed_deal_for_exchange_close():
    a = _agent()
    a.open_position("BUY", 2000.0, deal_id="DEAL123")
    assert a.tick(a._position.sl_price - 0.5) == "stop_loss"
    assert a.last_closed["deal_id"] == "DEAL123"      # was lost: _position cleared before the caller read it
    assert a.last_closed["reason"] == "stop_loss" and a.last_closed["pnl_pct"] < 0
    assert not a.has_position()


def test_max_hold_seconds_param_is_honoured():
    a = _agent(max_hold_seconds=10)
    pos = a.open_position("SELL", 2000.0, deal_id="d")
    pos.entry_time = time.time() - 11
    assert a.tick(2000.0) == "timeout"


def test_preview_is_pure_and_ignores_enabled_flag():
    a = ScalpingAgent()                      # disabled
    p = list(2000 + np.sin(np.arange(10)) * 0.01)
    before = (a._last_signal, a.total_trades, len(a._trade_timestamps))
    sig = a.preview(p)
    assert sig.reason != "scalping disabled"
    assert before == (a._last_signal, a.total_trades, len(a._trade_timestamps))


def test_set_params_validation_lives_in_exec_tools():
    from mcp_server.exec_tools import _validate_scalp_params
    assert _validate_scalp_params({"lookback": "12", "sl_pct": 0.2}) == {"lookback": 12, "sl_pct": 0.2}
    for bad in ({"sl_pct": -1}, {"lookback": 0}, {"nope": 1}, {"size": 50}, {"rsi_low": "abc"}):
        with pytest.raises(ValueError):
            _validate_scalp_params(bad)


# ── risk manager ─────────────────────────────────────────────────────────
def test_daily_loss_limit_uses_the_whole_days_pnl_not_the_last_trade():
    rm = PortfolioRiskManager(10_000)
    for _ in range(2):
        rm.update_pnl(-200.0)                # -400 today, limit is -500
    rm.consecutive_losses = 0
    ok, why = rm.check_trade_allowed(150.0)  # -400 - 150 < -500
    assert not ok and "Daily" in why
    assert rm.check_trade_allowed(50.0)[0]


def test_consecutive_loss_lockout_clears_next_day(monkeypatch):
    rm = PortfolioRiskManager(10_000)
    for _ in range(3):
        rm.update_pnl(-10.0)
    assert not rm.check_trade_allowed(1.0)[0]

    class Tomorrow(dt.date):
        @classmethod
        def today(cls):
            return dt.date.today() + dt.timedelta(days=1)

    monkeypatch.setattr(risk_mod, "date", Tomorrow)
    assert rm.check_trade_allowed(1.0)[0]    # was a permanent lock-out
    assert rm.today_pnl == 0.0


def test_metrics_win_rate_and_today_pnl():
    rm = PortfolioRiskManager(10_000)
    rm.update_pnl(30.0)
    rm.update_pnl(-10.0)
    m = rm.get_risk_metrics()
    assert m["num_trades"] == 2 and abs(m["win_rate"] - 0.5) < 1e-9 and m["today_pnl"] == 20.0


def test_trailing_stop_buy_and_sell():
    rm = PortfolioRiskManager()
    assert rm.calculate_trailing_stop(100, 90) == pytest.approx(99.0)
    assert rm.calculate_trailing_stop(100, 110) == pytest.approx(108.9)
    assert rm.calculate_trailing_stop(100, 110, direction="SELL") == pytest.approx(101.0)
    assert rm.calculate_trailing_stop(100, 90, direction="SELL") == pytest.approx(90.9)
