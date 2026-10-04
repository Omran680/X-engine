import pytest

from trade_bot.execution import trader as trader_mod


class FakeIG:
    def __init__(self, *a, **k):
        self.calls = []
        self.fail_positions = False
        self.expired_once = False
        self.sessions = 0

    def create_session(self):
        self.sessions += 1

    def fetch_open_positions(self):
        if self.fail_positions:
            raise RuntimeError("api down")
        return []

    def fetch_market_by_epic(self, epic):
        if self.expired_once:
            self.expired_once = False
            raise Exception("error.security.client-token-invalid")
        return {"snapshot": {"bid": 2001.5}}

    def create_open_position(self, **kw):
        self.calls.append(("open", kw))
        return {"dealReference": "REF1"}

    def fetch_deal_by_deal_reference(self, ref):
        return {"dealStatus": self.status, "dealId": "DIAAAA", "level": 2001.5, "reason": "X"}

    status = "ACCEPTED"

    def fetch_open_position_by_deal_id(self, deal_id):
        return {"market": {"epic": "CS.D.IN_GOLD.MFI.IP", "expiry": "-"},
                "position": {"direction": "BUY", "size": 0.1}}

    def close_open_position(self, **kw):
        self.calls.append(("close", kw))
        return {"dealReference": "CLOSE1"}


@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setenv("IG_USERNAME", "u")
    monkeypatch.setenv("IG_PASSWORD", "p")
    monkeypatch.setenv("IG_API_KEY", "k")
    monkeypatch.setattr(trader_mod, "IGService", FakeIG)
    monkeypatch.setattr(trader_mod, "API_RATE_LIMIT_INTERVAL", 0)
    t = trader_mod.Trader()
    t._min_api_interval = 0
    return t


def test_missing_credentials_is_a_config_error(monkeypatch):
    monkeypatch.delenv("IG_USERNAME", raising=False)
    monkeypatch.setattr(trader_mod, "IGService", FakeIG)
    with pytest.raises(trader_mod.ConfigError, match="IG_USERNAME"):
        trader_mod.Trader()


def test_live_account_needs_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("IG_USERNAME", "u"); monkeypatch.setenv("IG_PASSWORD", "p")
    monkeypatch.setenv("IG_API_KEY", "k"); monkeypatch.delenv("IG_ALLOW_LIVE", raising=False)
    monkeypatch.setattr(trader_mod, "IGService", FakeIG)
    monkeypatch.setattr(trader_mod, "IG_ACC_TYPE", "LIVE")
    with pytest.raises(trader_mod.ConfigError, match="IG_ALLOW_LIVE"):
        trader_mod.Trader()
    monkeypatch.setenv("IG_ALLOW_LIVE", "1")
    assert trader_mod.Trader().acc_type == "LIVE"


def test_get_positions_propagates_errors_instead_of_pretending_flat(trader):
    trader.ig.fail_positions = True
    with pytest.raises(RuntimeError):
        trader.get_positions()


def test_open_trade_returns_deal_id_and_enforces_size_cap(trader):
    res = trader.open_trade("EPIC", "BUY", 0.1, sl=4.0, tp=6.0)
    assert res["dealReference"] == "REF1" and res["dealId"] == "DIAAAA"
    with pytest.raises(ValueError):
        trader.open_trade("EPIC", "BUY", 50)
    with pytest.raises(ValueError):
        trader.open_trade("EPIC", "LONG", 0.1)


def test_rejected_order_raises(trader):
    trader.ig.status = "REJECTED"
    with pytest.raises(RuntimeError, match="rejected"):
        trader.open_trade("EPIC", "SELL", 0.1)


def test_close_position_sends_the_opposite_side_with_full_params(trader):
    trader.close_position("DIAAAA")          # used to raise TypeError (deal_id only)
    kind, kw = trader.ig.calls[-1]
    assert kind == "close" and kw["direction"] == "SELL" and kw["size"] == 0.1
    assert kw["deal_id"] == "DIAAAA" and kw["epic"] == "CS.D.IN_GOLD.MFI.IP"


def test_session_expiry_triggers_one_relogin(trader):
    trader.ig.expired_once = True
    assert trader.get_price("EPIC") == 2001.5
    assert trader.ig.sessions == 2
