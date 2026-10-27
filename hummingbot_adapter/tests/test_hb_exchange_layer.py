"""T5: the exchange layer (QuoteEngine.to_orders) on synthetic cases."""

from __future__ import annotations

import re
from decimal import Decimal

import numpy as np
import pytest

from hummingbot_adapter.quotes import QuoteEngine

TICK = 1e-5
ENGINE = QuoteEngine(sigma=2.7568397609849377e-05, kappa=12066.410319238006)
PRICE = re.compile(r"^\d+\.\d{5}$")


def _book(rng: np.random.Generator, spread_ticks: int) -> float:
    bid_ticks = int(rng.integers(5_000, 40_000))
    bid = round(bid_ticks * TICK, 5)
    ask = round((bid_ticks + spread_ticks) * TICK, 5)
    return (ask + bid) / 2


def _check_orders(orders, quotes, q):
    sides = [order["side"] for order in orders]
    assert sides == [side for side, ok in (("buy", q < 10), ("sell", q > -10)) if ok]
    for order in orders:
        assert set(order) == {"side", "price", "amount"}
        assert PRICE.fullmatch(order["price"]), order["price"]
        price = Decimal(order["price"])
        assert price % Decimal("0.00001") == 0
        quote = quotes["bid"] if order["side"] == "buy" else quotes["ask"]
        assert abs(float(price) - quote) <= 1e-12
        assert order["amount"] == "100"
        assert price * Decimal(order["amount"]) >= Decimal("1")


def test_orders_on_the_tick_grid_with_five_decimals_and_100_doge():
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(3_000):
        mid = _book(rng, 1)  # a 1-tick spread puts the mid clamp on the grid too
        q = int(rng.integers(-10, 11))
        tau = float(rng.uniform(0.0, 86_400.0))
        if rng.random() < 0.5:
            quotes = ENGINE.b3(rng.uniform(-1.0, 1.0, 2), mid, q, tau)
        else:
            quotes = ENGINE.b0(mid, q, tau, 0.8908584807525282)
        orders = ENGINE.to_orders(quotes, q)
        _check_orders(orders, quotes, q)
        checked += len(orders)
    assert checked > 5_000


def test_no_bid_at_q_max_and_no_ask_at_minus_q_max():
    quotes = ENGINE.b0(0.080005, 0, 3_600.0, 0.8908584807525282)
    assert [order["side"] for order in ENGINE.to_orders(quotes, 10)] == ["sell"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, -10)] == ["buy"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, 9)] == ["buy", "sell"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, -9)] == ["buy", "sell"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, 0)] == ["buy", "sell"]
    assert ENGINE.to_orders(quotes, 0, q_max=3, unit=100)[0]["amount"] == "100"
    assert [order["side"] for order in ENGINE.to_orders(quotes, 3, q_max=3)] == ["sell"]
    # beyond the cap (a PROTOCOL §7 halt state) only the side that reduces |q| is posted
    assert [order["side"] for order in ENGINE.to_orders(quotes, 11)] == ["sell"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, -11)] == ["buy"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, 4, q_max=3)] == ["sell"]
    assert [order["side"] for order in ENGINE.to_orders(quotes, -4, q_max=3)] == ["buy"]


def test_notional_below_one_usdt_is_refused():
    assert ENGINE.to_orders({"bid": 0.01, "ask": 0.01001}, 0)[0] == {"side": "buy", "price": "0.01000", "amount": "100"}
    with pytest.raises(ValueError, match="notional"):
        ENGINE.to_orders({"bid": 0.00999, "ask": 0.0101}, 0)
    with pytest.raises(ValueError, match="notional"):
        ENGINE.to_orders({"bid": 0.009, "ask": 0.00999}, 10)  # only the ask is posted, at 0.999 USDT
    assert ENGINE.to_orders({"bid": 0.009, "ask": 0.0101}, 10) == [{"side": "sell", "price": "0.01010", "amount": "100"}]


@pytest.mark.parametrize(
    "quotes",
    [
        {"bid": 0.081205, "ask": 0.08123},  # a half-tick price
        {"bid": 0.08121, "ask": 0.0812305},
        {"bid": 0.0812100001, "ask": 0.08123},
    ],
)
def test_off_grid_prices_are_refused(quotes):
    with pytest.raises(ValueError, match="tick grid"):
        ENGINE.to_orders(quotes, 0)


def test_only_the_posted_side_must_be_on_the_grid():
    assert ENGINE.to_orders({"bid": 0.081205, "ask": 0.08123}, 10) == [{"side": "sell", "price": "0.08123", "amount": "100"}]
    assert ENGINE.to_orders({"bid": 0.08121, "ask": 0.0812305}, -10) == [{"side": "buy", "price": "0.08121", "amount": "100"}]


def test_float_noise_of_a_grid_price_is_accepted_and_printed_exactly():
    bid = 8_123 * TICK  # 0.08123000000000001: noise in the price, none in bid / TICK
    orders = ENGINE.to_orders({"bid": bid, "ask": 8_124 * TICK}, 0)
    assert [order["price"] for order in orders] == ["0.08123", "0.08124"]
    bid = 8_000 * TICK  # 0.08, whose tick count 0.08 / 1e-5 is 7999.999999999999
    assert bid / TICK != round(bid / TICK)
    orders = ENGINE.to_orders({"bid": bid, "ask": 0.08001}, 0)
    assert [order["price"] for order in orders] == ["0.08000", "0.08001"]
    noisy = np.nextafter(0.08001, 1.0)  # one ulp above a grid price
    assert ENGINE.to_orders({"bid": bid, "ask": noisy}, 0)[1]["price"] == "0.08001"


def test_clamp_at_a_two_tick_spread_can_leave_the_grid():
    """With an even spread the mid is on the grid, so the mid clamp (bid <= S - tick/2) gives a half-tick
    price; the exchange layer refuses it rather than move the frozen quote."""

    mid = (0.08122 + 0.0812) / 2
    quotes = ENGINE.b3(np.array([1.0, 1.0]), mid, -10, 86_400.0)
    assert quotes["bid"] == pytest.approx(mid - TICK / 2, abs=1e-15)
    with pytest.raises(ValueError, match="tick grid"):
        ENGINE.to_orders(quotes, -10)


@pytest.mark.parametrize("q", [0.5, float("nan"), float("inf")])
def test_inventory_must_be_a_whole_number_of_units(q):
    with pytest.raises(ValueError):
        ENGINE.to_orders({"bid": 0.08121, "ask": 0.08122}, q)


def test_crossed_quotes_and_invalid_settings_are_refused():
    with pytest.raises(ValueError, match="crossed"):
        ENGINE.to_orders({"bid": 0.08122, "ask": 0.08122}, 0)
    with pytest.raises(ValueError):
        ENGINE.to_orders({"bid": 0.08121, "ask": 0.08122}, 0, q_max=0)
    with pytest.raises(ValueError):
        ENGINE.to_orders({"bid": 0.08121, "ask": 0.08122}, 0, unit=100.5)


def test_quote_engine_refuses_invalid_inputs():
    with pytest.raises(ValueError):
        ENGINE.b3(np.array([0.0, np.nan]), 0.08, 0, 100.0)
    with pytest.raises(ValueError):
        ENGINE.b3(np.array([0.0, 0.0, 0.0]), 0.08, 0, 100.0)
    with pytest.raises(ValueError):
        ENGINE.b0(0.08, 0, 100.0, 0.0)
    with pytest.raises(ValueError):
        ENGINE.b0(-0.08, 0, 100.0, 0.5)
    with pytest.raises(ValueError):
        ENGINE.b0(0.08, 0, float("nan"), 0.5)
    with pytest.raises(ValueError):
        QuoteEngine(sigma=0.0, kappa=1.0)
