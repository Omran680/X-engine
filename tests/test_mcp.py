import asyncio
import json

import pytest

import mcp_server.context as context_mod
from mcp_server import exec_tools, read_tools
from mcp_server.server import _BearerAuth, _is_loopback, run_sse


class FakeTrader:
    def __init__(self):
        self.opened, self.closed = [], []

    def open_trade(self, epic, direction, size, sl=None, tp=None):
        self.opened.append((epic, direction, size, sl, tp))
        return {"dealReference": "R", "dealId": "D1"}

    def close_position(self, deal_id):
        self.closed.append(deal_id)
        return {"ok": True}

    def get_price(self, epic):
        return 2000.0


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setattr(context_mod, "Trader", FakeTrader)
    c = context_mod.BotContext()
    c.last_price = 2000.0
    return c


def call(mod, name, args, ctx):
    out = asyncio.run(mod.handle(name, args, ctx))
    return json.loads(out[0].text)


def test_open_trade_validates_size_and_records_deal_id(ctx):
    with pytest.raises(ValueError):
        call(exec_tools, "open_trade", {"direction": "BUY", "size": 25}, ctx)       # > MAX_ORDER_SIZE
    with pytest.raises(ValueError):
        call(exec_tools, "open_trade", {"direction": "UP", "size": 0.1}, ctx)
    res = call(exec_tools, "open_trade", {"direction": "BUY", "size": 0.1}, ctx)
    assert res["status"] == "opened" and ctx.position["deal_id"] == "D1"
    assert ctx._trader.opened[0][0] == "CS.D.IN_GOLD.MFI.IP"          # default epic from config


def test_mcp_orders_go_through_the_risk_manager(ctx):
    ctx._risk_manager.update_pnl(-900.0)
    ctx._risk_manager.consecutive_losses = 0
    res = call(exec_tools, "open_trade", {"direction": "SELL", "size": 0.1}, ctx)
    assert res["status"] == "error" and "Risk manager" in res["error"]
    assert ctx._trader.opened == []


def test_save_models_persists_the_live_agent_or_reports_it_cannot(ctx):
    res = call(exec_tools, "save_models", {}, ctx)
    assert res["status"] == "error" and "No live agent" in res["error"]
    saved = []
    ctx.save_models_fn = lambda: saved.append(1)
    assert call(exec_tools, "save_models", {}, ctx)["status"] == "saved" and saved == [1]


def test_scalp_signal_read_tool_is_side_effect_free(ctx):
    prices = [2000 + i * 0.001 for i in range(12)]
    before = ctx.scalping.get_status()
    call(read_tools, "get_scalp_signal", {"prices": prices}, ctx)
    assert ctx.scalping.get_status() == before


def test_set_scalp_params_rejects_out_of_range(ctx):
    with pytest.raises(ValueError):
        call(exec_tools, "set_scalp_params", {"sl_pct": 99}, ctx)
    assert call(exec_tools, "set_scalp_params", {"sl_pct": 0.3}, ctx)["status"] == "ok"
    assert ctx.scalping.sl_pct == 0.3


def test_scalp_force_exit_closes_on_exchange_and_books_pnl(ctx):
    ctx.scalping.open_position("BUY", 1990.0, deal_id="SC1")
    res = call(exec_tools, "scalp_force_exit", {}, ctx)
    assert res["status"] == "exited" and ctx._trader.closed == ["SC1"]
    assert ctx._risk_manager.get_risk_metrics()["num_trades"] == 1


def test_scalp_force_exit_in_dry_run_never_touches_exchange(ctx):
    ctx.dry_run = True
    ctx.scalping.open_position("BUY", 1990.0, deal_id="SC1")
    call(exec_tools, "scalp_force_exit", {}, ctx)
    assert ctx._trader.closed == []


# ── transport security ───────────────────────────────────────────────────
def test_loopback_detection():
    assert _is_loopback("127.0.0.1") and _is_loopback("::1") and _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0") and not _is_loopback("192.168.1.5")


def test_sse_refuses_public_bind_without_token(ctx, monkeypatch):
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="without authentication"):
        asyncio.run(run_sse(ctx, port=0, host="0.0.0.0"))


def test_bearer_middleware_rejects_missing_and_wrong_tokens():
    sent = []

    async def inner(scope, receive, send):
        sent.append("inner")

    app = _BearerAuth(inner, "s3cret")

    def run(headers):
        out = []

        async def send(msg):
            out.append(msg)

        asyncio.run(app({"type": "http", "headers": headers}, None, send))
        return out

    assert run([])[0]["status"] == 401
    assert run([(b"authorization", b"Bearer nope")])[0]["status"] == 401
    run([(b"authorization", b"Bearer s3cret")])
    assert sent == ["inner"]
