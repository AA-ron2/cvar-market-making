"""execution.py (WO-HB2 deliverable 1), every rule with a fake clock in int µs:
- round_away_from_mid (pre-run clarification a): an off-grid bid is rounded down and an ask up, on-grid prices
  (float noise included) stay put, and the rounded pair of every frozen quote is accepted by QuoteEngine.to_orders;
- Requoter (PROTOCOL §5.3): post, re-quote only after >= 1 tick and >= 1 s, no bid at q >= 10, no ask at q <= -10;
- RateLimiter (§5.3): 100 new orders per rolling (t - 10 s, t] and 200,000 per UTC day, skips counted, cancels apart;
- Ledger (§5.1, §7, §9) against a Decimal model of binance_paper_trade's fee arithmetic: a 100-DOGE buy credits
  99.9 DOGE and the reconciliation residual stays exactly 0;
- Safety: each §7 trigger at and just past its threshold, its action, and the halt across a restart;
- DayClock (§5.5): the close at 23:59:50, the wait and the reset at 00:00:00, and the missed close;
- StepClock and dt_seconds (clarification c; CLOCK_MONOTONIC_RAW, clarification i), and the atomic state file (§5.4).
"""

from __future__ import annotations

import ast
import bisect
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hummingbot_adapter.execution import (
    CANCEL,
    CLOCK_RAW_RATE_PPM,
    CLOSE,
    DAY_US,
    FEE_ASSETS,
    KEEP,
    NONE,
    POST,
    REQUOTE,
    RESET,
    SIDES,
    TRADE,
    US_PER_SECOND,
    WAIT,
    DayClock,
    Ledger,
    RateLimiter,
    Requoter,
    RestingOrder,
    RoundedQuotes,
    Safety,
    SafetyEvent,
    StepClock,
    dt_seconds,
    load_state,
    monotonic_raw_ns,
    round_away_from_mid,
    save_state,
)
from hummingbot_adapter.features import FeatureEngine
from hummingbot_adapter.quotes import QuoteEngine
from cvar_mm.gym.data_loader import timestamps_us_to_datetime_and_deltas

REPO_ROOT = Path(__file__).resolve().parents[2]
US = 1_000_000
TICK = 1e-5
TICK_DEC = Decimal("0.00001")
DAY0 = int(datetime(2026, 10, 20, tzinfo=timezone.utc).timestamp()) * US  # Tue 20 Oct 2026 00:00:00 UTC
MIDNIGHT = DAY0 + DAY_US  # Wed 21 Oct 2026 00:00:00 UTC
B0_GAMMA = 0.8908584807525282  # PROTOCOL §2
HB_ENGINE = QuoteEngine(sigma=7.556448594347906e-06, kappa=41813.40928239842)  # hb_calibration.json (2026-10-01)
FEE_RATE = Decimal("0.001")  # the paper default, maker and taker (PROTOCOL §3)


def _actions(events: list[SafetyEvent]) -> list[tuple[str, str]]:
    return [(event.action, event.trigger) for event in events]


def _grid(ticks: int) -> float:
    """The float of the grid price `ticks` x 0.00001 (the nearest double to the decimal)."""

    return float(Decimal(ticks) * TICK_DEC)


class PaperAccount:
    """The balance arithmetic of Hummingbot 2.17.0's binance_paper_trade under its default
    DeductedFromReturnsTradeFee of 0.1 %, in Decimal:
    - a LIMIT buy of q DOGE at p pays q*p USDT and is credited q - q*0.001 DOGE (the fee comes out of the DOGE);
      a sell gives q DOGE and is credited q*p - q*p*0.001 USDT;
    - a MARKET order walks the book: one fill event per level at Decimal(level price), while the balances move by
      the order amount at the average price.
    Each fill event carries the fee as hb_forward_bot._fee books it: amount * 0.001 DOGE on a buy, amount * price *
    0.001 USDT on a sell. Balances start at the paper defaults (DOGE 1,000,000; USDT 100,000)."""

    def __init__(self, doge: Decimal = Decimal(1_000_000), usdt: Decimal = Decimal(100_000)):
        self.balances = {"DOGE": doge, "USDT": usdt}

    def limit_fill(self, side: str, price: Decimal, amount: Decimal) -> tuple[str, Decimal, Decimal, Decimal, str]:
        if side == "buy":
            self.balances["USDT"] -= amount * price
            self.balances["DOGE"] += amount - amount * FEE_RATE
            return side, price, amount, amount * FEE_RATE, "DOGE"
        notional = amount * price
        self.balances["DOGE"] -= amount
        self.balances["USDT"] += notional - notional * FEE_RATE
        return side, price, amount, notional * FEE_RATE, "USDT"

    def market_fill(self, side: str,
                    levels: list[tuple[float, float]]) -> list[tuple[str, Decimal, Decimal, Decimal, str]]:
        amount = sum((Decimal(size) for _, size in levels), Decimal(0))
        average = sum((Decimal(price) * Decimal(size) for price, size in levels), Decimal(0)) / amount
        if side == "buy":
            self.balances["USDT"] -= amount * average
            self.balances["DOGE"] += amount - amount * FEE_RATE
        else:
            self.balances["DOGE"] -= amount
            self.balances["USDT"] += amount * average - amount * average * FEE_RATE
        events = []
        for level_price, level_size in levels:
            price, size = Decimal(level_price), Decimal(level_size)
            fee, asset = (size * FEE_RATE, "DOGE") if side == "buy" else (size * price * FEE_RATE, "USDT")
            events.append((side, price, size, fee, asset))
        return events


class FakeClocks:
    """A wall clock and a monotonic clock in ns, moved by hand."""

    def __init__(self, wall_ns: int, mono_ns: int = 7_000_000_000_123):
        self.wall = wall_ns
        self.mono = mono_ns

    def wall_ns(self) -> int:
        return self.wall

    def monotonic_ns(self) -> int:
        return self.mono


def _book_row(bid_ticks: int, spread_ticks: int, amounts: list[float]) -> dict[str, float]:
    row = {"asks[0].price": round((bid_ticks + spread_ticks) * TICK, 5), "bids[0].price": round(bid_ticks * TICK, 5)}
    for level in range(5):
        row[f"bids[{level}].amount"] = amounts[level]
        row[f"asks[{level}].amount"] = amounts[5 + level]
    return row


# --- clarification a: round_away_from_mid -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bid, ask, expected",
    [
        (8_123 * TICK, 8_124 * TICK, (0.08123, 0.08124)),  # 0.08123000000000001: noise in the price
        (0.08, 0.08001, (0.08, 0.08001)),  # 0.08 / TICK is 7999.999999999999: a plain floor would drop a tick
        (0.07002, 0.07003, (0.07002, 0.07003)),  # 0.07002 / TICK is 7001.999999999999
        (0.07005, 7_006 * TICK, (0.07005, 0.07006)),  # 0.07006000000000001 / TICK is 7006.000000000001
        (0.08, float(np.nextafter(0.08001, 1.0)), (0.08, 0.08001)),  # one ulp above the grid: a plain ceil adds a tick
    ],
)
def test_on_grid_prices_with_float_noise_stay_put_unflagged(bid, ask, expected):
    mid = (bid + ask) / 2
    rounded = round_away_from_mid({"bid": bid, "ask": ask}, mid)
    assert rounded == RoundedQuotes(bid=expected[0], ask=expected[1], bid_rounded=False, ask_rounded=False)
    assert abs(rounded.bid - bid) <= 1e-9 * TICK and abs(rounded.ask - ask) <= 1e-9 * TICK


def test_exact_half_tick_bid_is_floored_and_ask_is_ceiled():
    """A 2-tick spread puts the mid on the grid, so the frozen clamp S -/+ tick/2 lands half a tick off it."""

    mid = (0.08122 + 0.0812) / 2
    bid, ask = mid - 0.5 * TICK, mid + 0.5 * TICK
    assert abs(bid / TICK - 8_120.5) < 1e-6 and abs(ask / TICK - 8_121.5) < 1e-6
    assert round_away_from_mid({"bid": bid, "ask": ask}, mid) == RoundedQuotes(0.0812, 0.08122, True, True)
    # one side at a time
    assert round_away_from_mid({"bid": bid, "ask": 0.08124}, mid) == RoundedQuotes(0.0812, 0.08124, True, False)
    assert round_away_from_mid({"bid": 0.08118, "ask": ask}, mid) == RoundedQuotes(0.08118, 0.08122, False, True)


def test_any_off_grid_price_is_rounded_away_from_the_mid():
    mid = 0.08122
    rounded = round_away_from_mid({"bid": 0.081203, "ask": 0.081237}, mid)
    assert rounded == RoundedQuotes(0.0812, 0.08124, True, True)
    # a millionth of a tick off the grid is off it (float noise is ~1e-12 tick here)
    rounded = round_away_from_mid({"bid": 0.08121 - 1e-11, "ask": 0.08123 + 1e-11}, mid)
    assert rounded == RoundedQuotes(0.0812, 0.08124, True, True)


def test_a_clamped_frozen_quote_refused_by_the_exchange_layer_is_accepted_after_rounding():
    quotes = HB_ENGINE.b3(np.array([1.0, 1.0]), (0.08122 + 0.0812) / 2, -10, 86_400.0)
    assert set(quotes) == {"bid", "ask", "gamma", "skew", "delta_bid", "delta_ask"}
    rounded = round_away_from_mid(quotes, (0.08122 + 0.0812) / 2)
    assert rounded.bid_rounded and rounded.bid == 0.0812
    with pytest.raises(ValueError, match="tick grid"):
        HB_ENGINE.to_orders(quotes, -10)  # the raw clamp is refused by the exchange layer
    orders = HB_ENGINE.to_orders({"bid": rounded.bid, "ask": rounded.ask}, -10)
    assert orders[0] == {"side": "buy", "price": "0.08120", "amount": "100"}


@pytest.mark.parametrize("spread_ticks", [1, 2, 3])
def test_rounding_property_over_frozen_b3_and_b0_quotes(spread_ticks):
    """rounded bid <= raw bid < mid < raw ask <= rounded ask, |rounded - raw| < 1 tick, a flag iff the raw price was
    off the grid (exactly, with float noise of 1e-9 tick allowed), and the rounded pair accepted by to_orders.
    An unflagged price comes back as its grid float, which may differ from the raw float by that noise."""

    rng = np.random.default_rng(1_000 + spread_ticks)
    one_tick = Fraction(1, 100_000)
    noise = Fraction(1, 10**9)
    flagged = {"bid": 0, "ask": 0}
    for draw in range(3_000):
        bid_ticks = int(rng.integers(5_000, 40_000))
        best_bid, best_ask = round(bid_ticks * TICK, 5), round((bid_ticks + spread_ticks) * TICK, 5)
        mid = (best_ask + best_bid) / 2
        q = int(rng.integers(-10, 11))
        tau = float(rng.uniform(0.0, 86_400.0))
        if draw % 2:
            action = rng.uniform(-1.0, 1.0, 2) if draw % 7 else rng.choice([-1.0, 1.0], 2)
            raw = HB_ENGINE.b3(action, mid, q, tau)
        else:
            raw = HB_ENGINE.b0(mid, q, tau, B0_GAMMA)
        assert raw["bid"] < mid < raw["ask"]
        rounded = round_away_from_mid(raw, mid)
        for side, away, value, flag in (
            ("bid", math.floor, rounded.bid, rounded.bid_rounded),
            ("ask", math.ceil, rounded.ask, rounded.ask_rounded),
        ):
            exact = Fraction(raw[side]) * 100_000  # the raw price in ticks, exactly
            nearest = round(exact)
            on_grid = abs(exact - nearest) <= noise
            assert flag is (not on_grid), (side, raw[side])
            assert value == _grid(nearest if on_grid else away(exact)), (side, raw[side])
            gap = Fraction(value) - Fraction(raw[side])  # rounded - raw
            if on_grid:
                assert abs(gap) <= 2 * noise * one_tick
            else:
                flagged[side] += 1
                assert (gap < 0 if side == "bid" else gap > 0) and abs(gap) < one_tick
        assert rounded.bid < mid < rounded.ask
        for q_orders in (q, 0):
            orders = HB_ENGINE.to_orders({"bid": rounded.bid, "ask": rounded.ask}, q_orders)
            prices = {order["side"]: float(order["price"]) for order in orders}
            posted = (("buy", rounded.bid, q_orders < 10), ("sell", rounded.ask, q_orders > -10))
            assert prices == {side: price for side, price, ok in posted if ok}
    if spread_ticks % 2:
        assert flagged == {"bid": 0, "ask": 0}  # an odd spread puts S -/+ tick/2 on the grid
    else:
        assert flagged["bid"] > 100 and flagged["ask"] > 100, flagged


@pytest.mark.parametrize(
    "bid, ask, mid",
    [
        (0.08121, 0.08122, 0.08121),  # bid at the mid
        (0.08122, 0.08123, 0.08121),  # bid above the mid
        (0.0812, 0.08121, 0.08121),  # ask at the mid
        (0.0812, 0.0812, 0.08121),  # ask below the mid
        (0.08123, 0.0812, 0.08121),  # crossed
        (float("nan"), 0.08122, 0.08121),
        (0.0812, float("inf"), 0.08121),
        (0.0812, 0.08122, float("nan")),
    ],
)
def test_quotes_that_do_not_straddle_the_mid_are_refused(bid, ask, mid):
    with pytest.raises(ValueError):
        round_away_from_mid({"bid": bid, "ask": ask}, mid)


@pytest.mark.parametrize("tick", [0.0, -1e-5, float("nan")])
def test_rounding_refuses_an_invalid_tick(tick):
    with pytest.raises(ValueError):
        round_away_from_mid({"bid": 0.0812, "ask": 0.08122}, 0.08121, tick)


# --- §5.3: RestingOrder and the Requoter ---------------------------------------------------------------------------


def _resting(side: str = "buy", price: str = "0.08120", post_ts_us: int = DAY0, mid: float = 0.081215,
             order_id: str | None = None) -> RestingOrder:
    return RestingOrder(order_id or f"{side}://DOGE-USDT/0001", side, Decimal(price), post_ts_us, mid)


def test_resting_order_json_round_trip_and_refusals():
    order = _resting("sell", "0.08124", DAY0 + 123_456, 0.081215)
    data = json.loads(json.dumps(order.to_json()))
    assert data == {"order_id": "sell://DOGE-USDT/0001", "side": "sell", "price": "0.08124",
                    "post_ts_us": DAY0 + 123_456, "mid_at_post": 0.081215}
    assert RestingOrder.from_json(data) == order
    for kwargs in (
        {"side": "bid"},
        {"price": 0.0812},  # a float price
        {"price": Decimal(0)},
        {"price": Decimal("-0.0812")},
        {"post_ts_us": 1.5},
        {"mid_at_post": float("nan")},
    ):
        values = {"order_id": "x", "side": "buy", "price": Decimal("0.0812"), "post_ts_us": DAY0, "mid_at_post": 0.08}
        with pytest.raises(ValueError):
            RestingOrder(**{**values, **kwargs})


def test_requoter_posts_whenever_nothing_rests():
    requoter = Requoter()
    assert requoter.decide("buy", Decimal("0.08120"), None, 0, DAY0) == POST
    assert requoter.decide("sell", Decimal("0.08123"), None, 0, DAY0) == POST
    # after a fill the side has no resting order: the replacement goes out at the next step, with no age rule
    assert requoter.decide("buy", Decimal("0.08120"), None, 9, DAY0 + 1) == POST
    assert requoter.decide("sell", Decimal("0.08123"), None, -9, DAY0 + 1) == POST


@pytest.mark.parametrize(
    "move, age_us, expected",
    [
        (Decimal("0.00001"), US, REQUOTE),  # exactly 1 tick and exactly 1 s
        (Decimal("-0.00001"), US, REQUOTE),
        (Decimal("0.00005"), 3_600 * US, REQUOTE),
        (Decimal("0.00001"), US - 1, KEEP),  # 1 µs short of 1 s
        (Decimal("0.00005"), 999_999, KEEP),
        (Decimal("0.00005"), 0, KEEP),
        (Decimal("0.000009"), 10 * US, KEEP),  # less than a tick
        (Decimal("-0.0000099999"), 10 * US, KEEP),
        (Decimal(0), 3_600 * US, KEEP),  # an unchanged target never re-quotes
    ],
)
def test_requote_needs_one_tick_and_one_second(move, age_us, expected):
    requoter = Requoter()
    post_ts = DAY0 + 5 * US
    for side, price in (("buy", "0.08120"), ("sell", "0.08124")):
        resting = _resting(side, price, post_ts)
        assert requoter.decide(side, resting.price + move, resting, 0, post_ts + age_us) == expected


def test_requote_sequence_on_a_fake_clock():
    requoter = Requoter()
    t = DAY0 + 60 * US
    resting = _resting("buy", "0.08120", t)
    decisions = []
    for offset_us, target in ((300_000, "0.08122"), (999_999, "0.08122"), (US, "0.08122")):
        decisions.append(requoter.decide("buy", Decimal(target), resting, 0, t + offset_us))
    assert decisions == [KEEP, KEEP, REQUOTE]
    resting = _resting("buy", "0.08122", t + US)  # the re-quote is the new resting order
    assert requoter.decide("buy", Decimal("0.08120"), resting, 0, t + 1_500_000) == KEEP
    assert requoter.decide("buy", Decimal("0.08120"), resting, 0, t + 2 * US) == REQUOTE


def test_sides_are_cancelled_at_the_inventory_cap():
    requoter = Requoter()
    now = DAY0 + 10 * US
    bid, ask = _resting("buy", "0.08120", now), _resting("sell", "0.08124", now)  # age 0: the cap ignores the age
    for q in (10, 11):
        assert requoter.decide("buy", bid.price, bid, q, now) == CANCEL
        assert requoter.decide("buy", bid.price, None, q, now) == NONE
        assert requoter.decide("sell", ask.price, ask, q, now) == KEEP
        assert requoter.decide("sell", ask.price, None, q, now) == POST
    for q in (-10, -11):
        assert requoter.decide("sell", ask.price, ask, q, now) == CANCEL
        assert requoter.decide("sell", ask.price, None, q, now) == NONE
        assert requoter.decide("buy", bid.price, bid, q, now) == KEEP
        assert requoter.decide("buy", bid.price, None, q, now) == POST
    assert requoter.decide("buy", bid.price, None, 9, now) == POST
    assert requoter.decide("sell", ask.price, None, -9, now) == POST
    assert [requoter.allowed("buy", q) for q in (8, 9, 10, 11)] == [True, True, False, False]
    assert [requoter.allowed("sell", q) for q in (-8, -9, -10, -11)] == [True, True, False, False]


def test_no_target_cancels_the_resting_order():
    requoter = Requoter()
    for side in SIDES:
        resting = _resting(side, "0.08120" if side == "buy" else "0.08124", DAY0)
        assert requoter.decide(side, None, resting, 0, DAY0) == CANCEL
        assert requoter.decide(side, None, None, 0, DAY0) == NONE


def test_requoter_refusals():
    requoter = Requoter()
    with pytest.raises(ValueError):
        requoter.decide("buy", Decimal("0.0812"), _resting("sell", "0.08124"), 0, DAY0)  # a sell on the bid side
    with pytest.raises(ValueError):
        requoter.decide("buy", Decimal("0.0812"), None, 9.5, DAY0)
    with pytest.raises(ValueError):
        requoter.decide("bid", Decimal("0.0812"), None, 0, DAY0)
    for kwargs in ({"q_max": 0}, {"min_move_ticks": 0}, {"min_age_us": -1}, {"q_max": 10.5}):
        with pytest.raises(ValueError):
            Requoter(**kwargs)


# --- §5.3: the RateLimiter -----------------------------------------------------------------------------------------


def test_the_101st_new_order_in_a_rolling_10_s_window_is_skipped():
    rate = RateLimiter()
    t0 = DAY0 + 3_600 * US
    for k in range(100):
        assert rate.acquire(t0 + k * 10_000) is None  # 100 new orders within 0.99 s
    assert rate.acquire(t0 + 990_000) == "rate_10s"
    assert (rate.day_orders, rate.skipped_window, rate.skipped_day, rate.max_in_window) == (100, 1, 0, 100)
    # the window is (t - 10 s, t]: the first order counts until t0 + 10 s - 1 µs and has left it at t0 + 10 s
    assert rate.acquire(t0 + 10 * US - 1) == "rate_10s"
    assert rate.acquire(t0 + 10 * US) is None
    assert rate.acquire(t0 + 10 * US) == "rate_10s"  # the second order (t0 + 10 ms) still counts
    assert rate.acquire(t0 + 10 * US + 10_000) is None
    assert (rate.day_orders, rate.skipped_window, rate.skipped_day, rate.max_in_window) == (102, 3, 0, 100)


def test_the_daily_cap_skips_and_counts():
    rate = RateLimiter(max_per_day=5)
    reasons = [rate.acquire(DAY0 + k * 20 * US) for k in range(8)]  # 20 s apart: the window never binds
    assert reasons == [None] * 5 + ["rate_day"] * 3
    assert (rate.day_orders, rate.skipped_day, rate.skipped_window) == (5, 3, 0)
    rate.reset_day()
    assert (rate.day_orders, rate.skipped_day, rate.skipped_window, rate.cancels, rate.max_in_window) == (0, 0, 0, 0, 0)
    assert rate.acquire(MIDNIGHT) is None
    assert rate.day_orders == 1


def test_the_production_daily_cap_is_200000_new_orders():
    rate = RateLimiter()
    for k in range(200_000):  # one every 0.4 s: 80,000 s, never more than 25 in a 10 s window
        if rate.acquire(DAY0 + k * 400_000) is not None:
            pytest.fail(f"new order {k + 1} was skipped")
    assert rate.day_orders == 200_000 and rate.max_in_window == 25
    assert rate.acquire(DAY0 + 80_000 * US) == "rate_day"
    assert rate.acquire(DAY0 + 86_000 * US) == "rate_day"
    assert (rate.day_orders, rate.skipped_day, rate.skipped_window) == (200_000, 2, 0)


def test_cancels_are_counted_apart_and_take_no_capacity():
    rate = RateLimiter(max_per_window=2)
    assert rate.acquire(DAY0) is None
    for _ in range(50):
        rate.count_cancel()
    assert (rate.cancels, rate.day_orders) == (50, 1)
    assert rate.acquire(DAY0 + 1) is None
    assert rate.acquire(DAY0 + 2) == "rate_10s"
    assert (rate.cancels, rate.day_orders, rate.skipped_window) == (50, 2, 1)


def test_reset_day_keeps_the_rolling_window_across_midnight():
    rate = RateLimiter()
    before = MIDNIGHT - 500_000  # 23:59:59.5
    for k in range(100):
        assert rate.acquire(before + k) is None
    rate.reset_day()
    assert rate.day_orders == 0
    assert rate.acquire(MIDNIGHT + 100_000) == "rate_10s"  # 00:00:00.1: the 100 orders are within 10 s
    assert rate.skipped_window == 1 and rate.day_orders == 0
    assert rate.acquire(before + 10 * US) is None  # the first has left the window
    assert rate.day_orders == 1


def test_rate_limiter_matches_a_brute_force_count_of_both_caps():
    rng = np.random.default_rng(11)
    rate = RateLimiter(max_per_day=1_500)
    accepted: list[int] = []
    reasons = {"rate_10s": 0, "rate_day": 0}
    times, t = [], DAY0
    while len(times) < 4_000:  # bursts of 50-250 requests, then a quiet gap of 1-15 s
        for _ in range(int(rng.integers(50, 250))):
            t += int(rng.choice([0, 1, 5_000, 20_000, 50_000]))
            times.append(t)
        t += int(rng.integers(US, 15 * US))
    for t in times:
        in_window = bisect.bisect_right(accepted, t) - bisect.bisect_right(accepted, t - 10 * US)
        full = {name for name, hit in (("rate_10s", in_window >= 100), ("rate_day", len(accepted) >= 1_500)) if hit}
        reason = rate.acquire(t)
        if full:
            assert reason in full
            reasons[reason] += 1
        else:
            assert reason is None
            accepted.append(t)
    assert len(accepted) == 1_500 and reasons["rate_10s"] > 100 and reasons["rate_day"] > 100
    assert (rate.day_orders, rate.skipped_window, rate.skipped_day) == (1_500, reasons["rate_10s"], reasons["rate_day"])
    largest = max(bisect.bisect_right(accepted, a) - bisect.bisect_right(accepted, a - 10 * US) for a in accepted)
    assert rate.max_in_window == largest == 100


def test_rate_limiter_state_round_trip_through_json():
    rng = np.random.default_rng(3)
    original = RateLimiter(max_per_day=300)
    t = DAY0
    for _ in range(300):
        t += int(rng.integers(0, 100_000))
        original.acquire(t)
        if rng.random() < 0.3:
            original.count_cancel()
    restored = RateLimiter(max_per_day=300)
    restored.set_state(json.loads(json.dumps(original.get_state())))
    assert restored.get_state() == original.get_state()
    for _ in range(600):
        t += int(rng.integers(0, 100_000))
        assert restored.acquire(t) == original.acquire(t)
    assert restored.get_state() == original.get_state()
    assert original.skipped_window > 0 and original.skipped_day > 0


def test_rate_limiter_set_state_refusals():
    state = {"window": [DAY0, DAY0 + 1], "day": DAY0 // DAY_US, "day_orders": 2, "max_in_window": 2,
             "skipped_window": 0, "skipped_day": 0, "cancels": 0}
    RateLimiter().set_state(state)
    with pytest.raises(ValueError):
        RateLimiter().set_state({**state, "window": [DAY0 + 1, DAY0]})
    with pytest.raises(ValueError):
        RateLimiter(max_per_window=1).set_state(state)
    with pytest.raises(ValueError):
        RateLimiter().set_state({**state, "cancels": 1.5})
    with pytest.raises(KeyError):
        RateLimiter().set_state({key: value for key, value in state.items() if key != "skipped_day"})


def test_rate_limiter_refuses_time_going_backwards():
    rate = RateLimiter()
    assert rate.acquire(DAY0 + 5) is None
    before = rate.get_state()
    with pytest.raises(ValueError, match="backwards"):
        rate.acquire(DAY0 + 4)
    assert rate.get_state() == before
    assert rate.acquire(DAY0 + 5) is None  # equal times are fine
    with pytest.raises(ValueError):
        rate.acquire(DAY0 + 5.5)
    for kwargs in ({"max_per_window": 0}, {"window_us": 0}, {"max_per_day": 0}, {"window_us": 1.5}):
        with pytest.raises(ValueError):
            RateLimiter(**kwargs)


# --- the fill ledger (§5.1, §7, §9) ---------------------------------------------------------------------------------


def test_a_100_doge_buy_credits_99_9_doge_and_the_residual_is_zero():
    account = PaperAccount()
    ledger = Ledger()
    ledger.start_day(account.balances["DOGE"])
    fill = account.limit_fill("buy", Decimal("0.08120"), Decimal(100))
    assert account.balances == {"DOGE": Decimal("1000099.9"), "USDT": Decimal("99991.88")}
    assert ledger.book_fill(*fill) == Decimal(100)
    assert (ledger.q, ledger.position, ledger.cash) == (1, Decimal(100), Decimal("-8.12"))
    assert ledger.fees == {"DOGE": Decimal("0.1"), "USDT": Decimal(0)}
    assert ledger.residual(account.balances["DOGE"]) == 0
    # a sell pays its fee in USDT; the DOGE side still reconciles exactly
    fill = account.limit_fill("sell", Decimal("0.08123"), Decimal(100))
    assert ledger.book_fill(*fill) == Decimal(-100)
    assert account.balances == {"DOGE": Decimal("999999.9"), "USDT": Decimal("99999.994877")}
    assert (ledger.q, ledger.cash) == (0, Decimal("0.003"))
    assert ledger.fees == {"DOGE": Decimal("0.1"), "USDT": Decimal("0.008123")}
    assert ledger.residual(account.balances["DOGE"]) == 0
    assert ledger.mark(0.0813) == 0.003  # gross: W ignores both fees
    assert ledger.fills == 2 and ledger.fill_notional == Decimal("16.243")


def test_a_long_random_fill_sequence_keeps_the_residual_exactly_zero():
    rng = np.random.default_rng(2026)
    account = PaperAccount()
    ledger = Ledger()
    ledger.start_day(account.balances["DOGE"])
    usdt0 = account.balances["USDT"]
    q, cash = 0, Decimal(0)
    for _ in range(5_000):
        sides = [side for side, ok in (("buy", q < 10), ("sell", q > -10)) if ok]
        side = sides[int(rng.integers(len(sides)))]
        price = Decimal(int(rng.integers(7_000, 9_000))) * TICK_DEC
        ledger.book_fill(*account.limit_fill(side, price, Decimal(100)))
        q += 1 if side == "buy" else -1
        cash += (-100 if side == "buy" else 100) * price
        assert ledger.q == q
        assert ledger.residual(account.balances["DOGE"]) == 0
        assert account.balances["USDT"] == usdt0 + ledger.cash - ledger.fees["USDT"]  # cash is before fees
    assert ledger.cash == cash and ledger.fills == 5_000
    assert ledger.fees["DOGE"] > 0 and ledger.fees["USDT"] > 0


def test_fees_are_booked_in_the_asset_the_fill_reports():
    ledger = Ledger()
    ledger.start_day(Decimal(1_000_000))
    ledger.book_fill("buy", Decimal("0.08"), Decimal(100), Decimal("0.008"), "USDT")  # a fee charged in USDT
    assert ledger.fees == {"DOGE": Decimal(0), "USDT": Decimal("0.008")}
    assert ledger.residual(Decimal(1_000_100)) == 0  # no DOGE fee: the whole 100 DOGE arrives
    ledger.book_fill("buy", Decimal("0.08"), Decimal(100), Decimal("0.1"), "DOGE")
    assert ledger.fees == {"DOGE": Decimal("0.1"), "USDT": Decimal("0.008")}
    assert ledger.residual(Decimal("1000199.9")) == 0


def test_a_fill_missing_from_the_ledger_breaks_the_reconciliation_and_halts():
    account = PaperAccount()
    ledger = Ledger()
    ledger.start_day(account.balances["DOGE"])
    safety = Safety()
    ledger.book_fill(*account.limit_fill("buy", Decimal("0.08120"), Decimal(100)))
    account.limit_fill("sell", Decimal("0.08125"), Decimal(100))  # filled on the account, never booked
    residual = ledger.residual(account.balances["DOGE"])
    assert residual == Decimal(-100)
    assert _actions(safety.check_residual(residual, DAY0)) == [("halt", "residual")]


@pytest.mark.parametrize("side, levels", [("sell", [(0.08116, 120.0), (0.08115, 380.0)]),
                                          ("buy", [(0.08125, 120.0), (0.08126, 380.0)])])
def test_a_market_close_filled_level_by_level(side, levels):
    """The close of q = +5 (sell) or -5 (buy) fills in two level pieces of 120 and 380 DOGE: q is refused in
    between (not a whole unit), whole again after, the residual exactly 0, and the close's notional kept apart."""

    account = PaperAccount()
    ledger = Ledger()
    ledger.start_day(account.balances["DOGE"])
    opening = "buy" if side == "sell" else "sell"
    for price in ("0.08120", "0.08118", "0.08119", "0.08121", "0.08117"):
        ledger.book_fill(*account.limit_fill(opening, Decimal(price), Decimal(100)))
    assert ledger.q == (5 if side == "sell" else -5)
    fill_notional = ledger.fill_notional
    pieces = account.market_fill(side, levels)
    ledger.book_fill(*pieces[0], close=True)
    with pytest.raises(ValueError, match="whole number"):
        _ = ledger.q
    assert ledger.units == (Decimal("3.8") if side == "sell" else Decimal("-3.8"))
    assert Safety().check_inventory(ledger.units, DAY0) == []
    ledger.book_fill(*pieces[1], close=True)
    assert ledger.q == 0 and ledger.position == 0
    assert ledger.residual(account.balances["DOGE"]) == 0
    assert ledger.fill_notional == fill_notional
    assert ledger.close_notional == pieces[0][1] * pieces[0][2] + pieces[1][1] * pieces[1][2]
    assert ledger.fills == 7


def test_wealth_is_cash_plus_100_q_mid_with_a_running_peak_and_drawdown():
    rng = np.random.default_rng(5)
    ledger = Ledger()
    ledger.start_day(Decimal(1_000_000))
    ledger.book_fill("buy", Decimal("0.08"), Decimal(100), Decimal("0.1"), "DOGE")
    assert ledger.mark(0.0799) == float(Decimal("-8")) + 100 * 0.0799
    assert ledger.peak == 0.0 and ledger.drawdown == -ledger.wealth > 0  # the peak starts at W = 0
    q, cash, peak = 1, Decimal("-8"), 0.0
    for _ in range(3_000):
        if rng.random() < 0.3:
            side = "buy" if (q < 10 and rng.random() < 0.5) or q <= -10 else "sell"
            price = Decimal(int(rng.integers(7_900, 8_100))) * TICK_DEC
            fee = (Decimal("0.1"), "DOGE") if side == "buy" else (100 * price * FEE_RATE, "USDT")
            ledger.book_fill(side, price, Decimal(100), *fee)
            q += 1 if side == "buy" else -1
            cash += (-100 if side == "buy" else 100) * price
        mid = (int(rng.integers(7_900, 8_100)) + 0.5) * TICK
        expected = float(cash) + 100 * q * mid
        assert ledger.mark(mid) == expected and ledger.wealth == expected
        peak = max(peak, expected)
        assert ledger.peak == peak and ledger.drawdown == peak - expected
    assert ledger.q == q and ledger.cash == cash


def test_start_day_resets_the_day_and_refuses_a_non_flat_book():
    account = PaperAccount()
    ledger = Ledger()
    with pytest.raises(RuntimeError):
        ledger.residual(account.balances["DOGE"])  # no reference before the first day
    ledger.start_day(account.balances["DOGE"])
    ledger.book_fill(*account.limit_fill("buy", Decimal("0.08120"), Decimal(100)))
    ledger.mark(0.0813)
    with pytest.raises(ValueError, match="flat"):
        ledger.start_day(account.balances["DOGE"])
    ledger.book_fill(*account.limit_fill("sell", Decimal("0.08110"), Decimal(100)))
    ledger.mark(0.0811)
    assert ledger.drawdown > 0
    ledger.start_day(account.balances["DOGE"])  # 00:00 UTC, after the close: the balance after the close
    assert ledger.doge_reference == account.balances["DOGE"] == Decimal("999999.9")
    assert (ledger.position, ledger.cash, ledger.fills) == (0, 0, 0)
    assert ledger.fees == {"DOGE": 0, "USDT": 0}
    assert (ledger.wealth, ledger.peak, ledger.drawdown) == (0.0, 0.0, 0.0)
    assert (ledger.fill_notional, ledger.close_notional) == (0, 0)
    assert ledger.residual(account.balances["DOGE"]) == 0
    for balance in (999_999.9, Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValueError):
            ledger.start_day(balance)


@pytest.mark.parametrize(
    "fill",
    [
        ("bid", Decimal("0.08"), Decimal(100), Decimal("0.1"), "DOGE"),
        ("buy", 0.08, Decimal(100), Decimal("0.1"), "DOGE"),
        ("buy", Decimal("0.08"), 100, Decimal("0.1"), "DOGE"),
        ("buy", Decimal("NaN"), Decimal(100), Decimal("0.1"), "DOGE"),
        ("buy", Decimal(0), Decimal(100), Decimal("0.1"), "DOGE"),
        ("buy", Decimal("0.08"), Decimal(0), Decimal("0.1"), "DOGE"),
        ("buy", Decimal("0.08"), Decimal(-100), Decimal("0.1"), "DOGE"),
        ("buy", Decimal("0.08"), Decimal(100), Decimal("-0.1"), "DOGE"),
        ("buy", Decimal("0.08"), Decimal(100), Decimal("0.1"), "BNB"),
    ],
)
def test_invalid_fills_are_refused_and_change_nothing(fill):
    ledger = Ledger()
    ledger.start_day(Decimal(1_000_000))
    ledger.book_fill("buy", Decimal("0.08"), Decimal(100), Decimal("0.1"), "DOGE")
    before = ledger.get_state()
    with pytest.raises(ValueError):
        ledger.book_fill(*fill)
    assert ledger.get_state() == before


def test_ledger_state_round_trip_through_json():
    rng = np.random.default_rng(8)
    account = PaperAccount()
    ledger = Ledger()
    ledger.start_day(account.balances["DOGE"])
    for k in range(200):
        side = "buy" if rng.random() < 0.5 else "sell"
        ledger.book_fill(*account.limit_fill(side, Decimal(int(rng.integers(7_900, 8_100))) * TICK_DEC, Decimal(100)))
        ledger.mark((int(rng.integers(7_900, 8_100)) + 0.5) * TICK)
    ledger.book_fill(*account.market_fill("sell" if ledger.position > 0 else "buy", [(0.08, 37.0)])[0], close=True)
    restored = Ledger()
    restored.set_state(json.loads(json.dumps(ledger.get_state())))
    assert restored.get_state() == ledger.get_state()
    assert restored.doge_reference == ledger.doge_reference and restored.fees == ledger.fees
    for _ in range(50):
        price = Decimal(int(rng.integers(7_900, 8_100))) * TICK_DEC
        fill = ("buy", price, Decimal(100), Decimal("0.1"), "DOGE")
        assert restored.book_fill(*fill) == ledger.book_fill(*fill)
        mid = (int(rng.integers(7_900, 8_100)) + 0.5) * TICK
        assert restored.mark(mid) == ledger.mark(mid)
    assert restored.get_state() == ledger.get_state()
    with pytest.raises(ValueError):
        Ledger(unit=10).set_state(ledger.get_state())
    with pytest.raises(ValueError):
        Ledger().set_state({**ledger.get_state(), "fees": {"DOGE": "0"}})


# --- §7: the safety triggers -----------------------------------------------------------------------------------------


def test_inventory_trigger_halts_only_past_q_max():
    safety = Safety()
    for q in (10, -10, Decimal(10), Decimal("-10"), Decimal("9.99")):
        assert safety.check_inventory(q, DAY0) == []
    assert safety.quoting and not safety.halted
    assert _actions(safety.check_inventory(11, DAY0 + 7)) == [("halt", "q_ledger")]
    assert safety.halted and not safety.quoting and safety.halt_ts_us == DAY0 + 7
    for q in (-11, Decimal("10.01"), Decimal("-10.01")):  # between a market order's level fills q may be fractional
        fresh = Safety()
        assert _actions(fresh.check_inventory(q, DAY0)) == [("halt", "q_ledger")]


def test_residual_trigger_halts_only_past_one_doge():
    safety = Safety()
    for residual in (Decimal(0), Decimal(1), Decimal(-1), Decimal("0.999999")):
        assert safety.check_residual(residual, DAY0) == []
    assert safety.quoting
    assert _actions(safety.check_residual(Decimal("1.000001"), DAY0)) == [("halt", "residual")]
    assert safety.halted and not safety.quoting
    assert _actions(Safety().check_residual(Decimal("-1.000001"), DAY0)) == [("halt", "residual")]


def test_stale_book_pauses_at_30_s_resumes_on_change_and_halts_at_5_min():
    safety = Safety()
    t0 = DAY0 + 1_000 * US
    assert safety.book(t0, changed=True) == []
    assert safety.book(t0 + 29_999_999, changed=False) == []
    assert safety.quoting
    assert _actions(safety.book(t0 + 30 * US, changed=False)) == [("pause", "stale_book")]
    assert safety.stale_paused and not safety.quoting and not safety.halted
    assert safety.book(t0 + 30 * US + 100_000, changed=False) == []  # one pause event
    assert _actions(safety.book(t0 + 31 * US, changed=True)) == [("resume", "stale_book")]
    assert safety.quoting and not safety.stale_paused
    # no change for 5 min on 0.1 s ticks: the pause at 30 s, the halt at 300 s
    t1 = t0 + 31 * US
    events = []
    for k in range(1, 3_011):
        events += [(event.action, event.trigger, k * 100_000) for event in safety.book(t1 + k * 100_000, changed=False)]
    assert events == [("pause", "stale_book", 30 * US), ("halt", "stale_book", 300 * US)]
    assert safety.halted and not safety.quoting
    # the updates return: the stale pause lifts, the halt holds
    assert _actions(safety.book(t1 + 302 * US, changed=True)) == [("resume", "stale_book")]
    assert safety.halted and not safety.stale_paused and not safety.quoting


def test_stale_book_halt_boundary():
    safety = Safety()
    safety.book(DAY0, changed=True)
    assert _actions(safety.book(DAY0 + 30 * US, changed=False)) == [("pause", "stale_book")]
    assert safety.book(DAY0 + 300 * US - 1, changed=False) == []
    assert _actions(safety.book(DAY0 + 300 * US, changed=False)) == [("halt", "stale_book")]


def test_more_than_ten_rejects_within_60_s_halt():
    safety = Safety()
    for k in range(10):
        assert safety.reject(DAY0 + k * US, "failure") == []
    assert safety.quoting
    assert _actions(safety.reject(DAY0 + 59_999_999, "failure")) == [("halt", "rejects")]
    assert safety.halted and not safety.quoting


def test_rejects_older_than_60_s_expire():
    safety = Safety()
    for k in range(10):
        safety.reject(DAY0 + k * US, "failure")
    assert safety.reject(DAY0 + 60_500_000, "failure") == []  # the one at 0 s has expired: 10 in the window
    assert safety.reject(DAY0 + 61_500_000, "failure") == []  # the one at 1 s too: still 10
    assert not safety.halted
    assert _actions(safety.reject(DAY0 + 61_500_000, "failure")) == [("halt", "rejects")]  # 11 within 60 s


def test_drawdown_over_40_usdt_closes_and_pauses_once_until_the_new_day():
    safety = Safety()
    assert safety.check_drawdown(0.0) == [] and safety.check_drawdown(40.0) == []
    assert safety.quoting
    assert _actions(safety.check_drawdown(40.000001)) == [("close_and_pause", "drawdown")]
    assert safety.drawdown_paused and not safety.quoting and not safety.halted
    assert safety.check_drawdown(55.0) == []  # once per day
    assert safety.drawdown_paused
    safety.new_day()
    assert not safety.drawdown_paused and safety.quoting
    assert _actions(safety.check_drawdown(40.000001)) == [("close_and_pause", "drawdown")]  # a new day can trigger
    with pytest.raises(ValueError):
        Safety().check_drawdown(float("nan"))


def test_the_drawdown_trigger_reads_the_gross_ledger_drawdown():
    """q = 10 bought at 0.08 (80 USDT): W = -40 at a mid of 0.04 is not past the limit, below it is; the fees
    (1 DOGE booked) play no part."""

    ledger = Ledger()
    ledger.start_day(Decimal(1_000_000))
    for _ in range(10):
        ledger.book_fill("buy", Decimal("0.08"), Decimal(100), Decimal("0.1"), "DOGE")
    safety = Safety()
    ledger.mark(0.04)
    assert ledger.drawdown == 40.0 and safety.check_drawdown(ledger.drawdown) == []
    ledger.mark(0.03999)
    assert ledger.drawdown > 40.0
    assert _actions(safety.check_drawdown(ledger.drawdown)) == [("close_and_pause", "drawdown")]


def test_clock_offset_warns_past_half_a_second_and_pauses_past_two_until_below_two():
    safety = Safety()
    assert safety.clock_offset(0.5) == [] and safety.clock_offset(-0.5) == [] and safety.clock_offset(0.0) == []
    warn = ("warn", "clock_offset")
    assert _actions(safety.clock_offset(0.5000001)) == [warn]
    assert _actions(safety.clock_offset(-0.5000001)) == [warn]
    assert _actions(safety.clock_offset(2.0)) == [warn]
    assert safety.quoting and not safety.clock_paused
    assert _actions(safety.clock_offset(2.0000001)) == [warn, ("pause", "clock_offset")]
    assert safety.clock_paused and not safety.quoting
    assert _actions(safety.clock_offset(2.0)) == [warn]  # not below 2 s: still paused
    assert _actions(safety.clock_offset(3.0)) == [warn]  # no second pause event
    assert safety.clock_paused
    assert _actions(safety.clock_offset(1.9999999)) == [warn, ("resume", "clock_offset")]
    assert safety.quoting
    assert _actions(safety.clock_offset(-2.5)) == [warn, ("pause", "clock_offset")]
    assert _actions(safety.clock_offset(-0.2)) == [("resume", "clock_offset")]
    assert safety.quoting and not safety.halted
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            safety.clock_offset(bad)


def test_quoting_is_off_while_any_halt_or_pause_holds():
    for mask in range(16):
        flags = [bool(mask >> bit & 1) for bit in range(4)]
        safety = Safety()
        safety.halted, safety.drawdown_paused, safety.stale_paused, safety.clock_paused = flags
        assert safety.quoting is (not any(flags)), flags


def test_halt_is_idempotent_and_keeps_the_first_reason():
    safety = Safety()
    assert _actions(safety.halt("q_ledger", "|q| = 11 > 10", DAY0 + 1)) == [("halt", "q_ledger")]
    assert safety.halt("rejects", "11 rejects", DAY0 + 2) == []
    assert safety.check_residual(Decimal(5), DAY0 + 3) == []
    assert safety.check_inventory(12, DAY0 + 4) == []
    assert (safety.halt_reason, safety.halt_ts_us) == ("q_ledger: |q| = 11 > 10", DAY0 + 1)


def test_halt_survives_a_restart_and_the_new_day_until_the_author_clears_it(tmp_path):
    safety = Safety()
    safety.check_inventory(11, DAY0 + 5)
    path = tmp_path / "state.json"
    save_state(path, {"safety": safety.get_state()})
    restored = Safety()  # a container restart: new objects from the state file
    restored.set_state(load_state(path)["safety"])
    assert restored.halted and not restored.quoting
    assert (restored.halt_reason, restored.halt_ts_us) == (safety.halt_reason, DAY0 + 5)
    restored.new_day()
    restored.book(MIDNIGHT, changed=True)
    restored.clock_offset(0.0)
    assert restored.check_inventory(0, MIDNIGHT) == [] and restored.check_residual(Decimal(0), MIDNIGHT) == []
    assert restored.halted and not restored.quoting
    # the author clears the flag in the state file (clear_halt.py): the next start quotes again
    state = load_state(path)
    state["safety"].update({"halted": False, "halt_reason": None, "halt_ts_us": None})
    save_state(path, state)
    cleared = Safety()
    cleared.set_state(load_state(path)["safety"])
    assert cleared.quoting and not cleared.halted


def test_drawdown_and_clock_pauses_survive_a_restart_within_the_day():
    safety = Safety()
    safety.check_drawdown(41.0)
    safety.clock_offset(2.5)
    for k in range(4):
        safety.reject(DAY0 + k * US, "failure")
    restored = Safety()
    restored.set_state(json.loads(json.dumps(safety.get_state())))
    assert restored.get_state() == safety.get_state()
    assert restored.drawdown_paused and restored.clock_paused and not restored.quoting
    assert restored.check_drawdown(50.0) == []  # the day's close-and-pause happened already
    assert _actions(restored.clock_offset(0.1)) == [("resume", "clock_offset")]
    assert not restored.quoting  # the drawdown pause holds until 00:00 UTC
    restored.new_day()
    assert restored.quoting
    for k in range(6):  # the restored rejects still count: 4 + 7 = 11 within 60 s
        assert restored.reject(DAY0 + (10 + k) * US, "failure") == []
    assert _actions(restored.reject(DAY0 + 20 * US, "failure")) == [("halt", "rejects")]


def test_smoke_thresholds_are_parameters():
    """§7: the smoke test trips the kill switch once with a tiny threshold; BotCore passes the settings in."""

    safety = Safety(drawdown_max_usdt=0.01, q_halt=2)
    assert safety.check_drawdown(0.01) == []
    assert _actions(safety.check_drawdown(0.02)) == [("close_and_pause", "drawdown")]
    assert safety.check_inventory(2, DAY0) == []
    assert _actions(safety.check_inventory(3, DAY0)) == [("halt", "q_ledger")]
    stale = Safety(stale_pause_us=US, stale_halt_us=2 * US)
    stale.book(DAY0, changed=True)
    assert _actions(stale.book(DAY0 + US, changed=False)) == [("pause", "stale_book")]
    assert _actions(stale.book(DAY0 + 2 * US, changed=False)) == [("halt", "stale_book")]
    rejects = Safety(reject_max=1, reject_window_us=US)
    assert rejects.reject(DAY0, "a") == [] and rejects.reject(DAY0 + US, "b") == []  # the first has expired
    assert _actions(rejects.reject(DAY0 + US, "c")) == [("halt", "rejects")]
    requoter = Requoter(q_max=3)
    assert requoter.decide("buy", Decimal("0.0812"), None, 3, DAY0) == NONE
    assert requoter.decide("buy", Decimal("0.0812"), None, 2, DAY0) == POST


def test_safety_refuses_invalid_settings():
    for kwargs in (
        {"stale_pause_us": 300 * US},
        {"stale_pause_us": 301 * US},
        {"stale_pause_us": 0},
        {"q_halt": 10.5},
        {"drawdown_max_usdt": float("nan")},
        {"clock_pause_s": float("inf")},
        {"reject_window_us": 1.5},
    ):
        with pytest.raises(ValueError):
            Safety(**kwargs)


# --- §5.5: the UTC day ---------------------------------------------------------------------------------------------


def test_day_clock_close_at_23_59_50_wait_and_reset_at_midnight():
    clock = DayClock()
    close = MIDNIGHT - 10 * US
    assert clock.close_start_us(DAY0) == close
    assert clock.phase(DAY0, DAY0, closed=False) == TRADE  # 00:00:00.000000, the first step of the day
    assert clock.phase(close - 1, DAY0, closed=False) == TRADE  # 23:59:49.999999
    assert clock.phase(close, DAY0, closed=False) == CLOSE  # 23:59:50.000000
    assert clock.phase(close, DAY0, closed=True) == WAIT
    assert clock.phase(MIDNIGHT - 1, DAY0, closed=False) == CLOSE  # until the close is done
    assert clock.phase(MIDNIGHT - 1, DAY0, closed=True) == WAIT  # 23:59:59.999999
    assert clock.phase(MIDNIGHT, DAY0, closed=True) == RESET  # 00:00:00.000000 the next day
    assert clock.phase(MIDNIGHT, DAY0, closed=False) == RESET
    assert clock.phase(MIDNIGHT + 5 * DAY_US + 17, DAY0, closed=True) == RESET
    assert clock.phase(MIDNIGHT, MIDNIGHT, closed=False) == TRADE


def test_day_id_day_start_and_t_elapsed():
    assert DayClock.day_id(DAY0) == "2026-10-20"
    assert DayClock.day_id(MIDNIGHT - 1) == "2026-10-20"
    assert DayClock.day_id(MIDNIGHT) == "2026-10-21"
    assert DayClock.day_start_us(DAY0 + 12_345_678_901) == DAY0
    assert DayClock.day_start_us(MIDNIGHT - 1) == DAY0 and DayClock.day_start_us(MIDNIGHT) == MIDNIGHT
    assert DayClock.t_elapsed(DAY0, DAY0) == 0.0
    assert DayClock.t_elapsed(DAY0 + 3_600_000_001, DAY0) == 3_600.000001
    t_close = DayClock.t_elapsed(MIDNIGHT - 10 * US, DAY0)
    assert t_close == 86_390.0 and 86_400 - t_close == 10.0  # tau = T - t_elapsed at the close
    rng = np.random.default_rng(4)
    for offset in rng.integers(0, DAY_US, 1_000):
        assert DayClock.t_elapsed(DAY0 + int(offset), DAY0) == int(offset) / 1e6


def test_day_clock_refusals():
    clock = DayClock()
    with pytest.raises(ValueError):
        clock.phase(DAY0 + 10, DAY0 + 1, closed=False)  # the day must start at 00:00 UTC
    with pytest.raises(ValueError):
        clock.phase(DAY0 - 1, DAY0, closed=False)  # before the day's start
    with pytest.raises(ValueError):
        clock.phase(DAY0 + 0.5, DAY0, closed=False)
    with pytest.raises(ValueError):
        DayClock.day_id(1.5)
    for close_before_us in (0, -1, DAY_US, DAY_US + 1, 10.5):
        with pytest.raises(ValueError):
            DayClock(close_before_us)


def test_close_wait_and_reset_over_midnight_on_a_fake_clock():
    """Steps every 0.5 s from 23:59:48 to 00:00:01: the close at 23:59:50 (a market order counted by the rate
    limiter), the wait, and at 00:00:00 the reset of the ledger (DOGE reference = the balance after the close),
    of the drawdown pause and of the daily order counts."""

    clock, account, ledger, safety, rate = DayClock(), PaperAccount(), Ledger(), Safety(), RateLimiter()
    day_start, closed = DAY0, False
    ledger.start_day(account.balances["DOGE"])
    for price in ("0.08120", "0.08121", "0.08119"):
        assert rate.acquire(DAY0 + 600 * US) is None
        ledger.book_fill(*account.limit_fill("buy", Decimal(price), Decimal(100)))
    safety.check_drawdown(41.0)
    phases = []
    t = MIDNIGHT - 12 * US
    while t <= MIDNIGHT + US:
        phase = clock.phase(t, day_start, closed)
        phases.append(((t - MIDNIGHT) / US, phase))
        if phase == CLOSE:
            assert rate.acquire(t) is None  # the close is a new order too
            for fill in account.market_fill("sell", [(0.08118, 300.0)]):
                ledger.book_fill(*fill, close=True)
            closed = ledger.position == 0
        elif phase == RESET:
            day_start = DayClock.day_start_us(t)
            ledger.start_day(account.balances["DOGE"])
            safety.new_day()
            rate.reset_day()
            closed = False
        t += 500_000
    expected = [(-12.0 + k / 2, TRADE) for k in range(4)] + [(-10.0, CLOSE)]
    expected += [(-9.5 + k / 2, WAIT) for k in range(19)] + [(0.0, RESET), (0.5, TRADE), (1.0, TRADE)]
    assert phases == expected
    assert day_start == MIDNIGHT and DayClock.day_id(day_start) == "2026-10-21"
    assert ledger.doge_reference == account.balances["DOGE"] and ledger.residual(account.balances["DOGE"]) == 0
    assert ledger.q == 0 and safety.quoting and rate.day_orders == 0


def test_missed_close_recovery_closes_the_carry_before_the_new_day(tmp_path):
    """The bot saved its state at 23:59:40 with q = -3 and was down across 23:59:50: back before midnight its first
    step closes; back after midnight the restored day is over (RESET), and the new day cannot start until the
    carried position is closed (start_day refuses a non-flat book)."""

    account, ledger = PaperAccount(), Ledger()
    ledger.start_day(account.balances["DOGE"])
    for price in ("0.08124", "0.08125", "0.08126"):
        ledger.book_fill(*account.limit_fill("sell", Decimal(price), Decimal(100)))
    path = tmp_path / "state.json"
    save_state(path, {"day": {"start_us": DAY0, "closed": False}, "ledger": ledger.get_state(),
                      "clock_last_us": MIDNIGHT - 20 * US})
    state = load_state(path)
    clock = DayClock()
    assert clock.phase(MIDNIGHT - 5 * US, state["day"]["start_us"], state["day"]["closed"]) == CLOSE
    restart = MIDNIGHT + 7 * 60 * US
    assert clock.phase(restart, state["day"]["start_us"], state["day"]["closed"]) == RESET
    restored = Ledger()
    restored.set_state(state["ledger"])
    assert restored.q == -3
    with pytest.raises(ValueError, match="flat"):
        restored.start_day(account.balances["DOGE"])
    for fill in account.market_fill("buy", [(0.08125, 200.0), (0.08126, 100.0)]):
        restored.book_fill(*fill, close=True)
    assert restored.q == 0 and restored.residual(account.balances["DOGE"]) == 0
    restored.start_day(account.balances["DOGE"])
    assert DayClock.day_start_us(restart) == MIDNIGHT and restored.doge_reference == account.balances["DOGE"]


# --- clarification c: StepClock and dt_seconds -----------------------------------------------------------------------


def test_step_clock_is_the_anchor_wall_time_plus_monotonic_time():
    clocks = FakeClocks(wall_ns=(DAY0 + 1_234_567) * 1_000 + 891)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns)
    clock.anchor()
    assert clock.now_us() == DAY0 + 1_234_567
    clocks.mono += 2_500_000_000
    clocks.wall -= 10 * 10**9  # the wall clock steps back 10 s: ignored until the next anchor
    assert clock.now_us() == DAY0 + 1_234_567 + 2_500_000
    clocks.wall += 20 * 10**9
    clocks.mono += 1_000
    assert clock.now_us() == DAY0 + 1_234_567 + 2_500_001
    assert clock.wall_us() == clocks.wall // 1_000


def test_step_clock_never_decreases_when_re_anchored_on_a_wall_clock_that_went_back():
    clocks = FakeClocks(wall_ns=(MIDNIGHT - 2 * US) * 1_000)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns)
    clock.anchor()
    clocks.mono += 2 * 10**9
    last = clock.now_us()
    assert last == MIDNIGHT
    clocks.wall = (last - 300_000) * 1_000  # at the 00:00 reset the wall clock is 0.3 s behind ts
    clock.anchor(floor_us=last)
    seen = [clock.now_us()]
    for _ in range(10):
        clocks.mono += 50_000_000
        seen.append(clock.now_us())
    assert seen == [last] * 7 + [last + 50_000 * k for k in range(1, 5)]  # flat until the new anchor catches up
    clocks.wall = (seen[-1] - 5 * US) * 1_000
    clock.anchor()  # without a floor the last value returned still bounds the next
    assert clock.now_us() == seen[-1]


def test_step_clock_peek_does_not_advance():
    clocks = FakeClocks(wall_ns=DAY0 * 1_000)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns)
    clock.anchor()
    now = clock.now_us()
    clocks.mono += 5_000_000
    assert clock.peek_us() == now + 5_000 and clock.peek_us() == now + 5_000
    assert clock.last_us == now
    clocks.wall = (now - US) * 1_000
    clock.anchor()
    assert clock.now_us() == now  # the floor is the last value now_us returned, not the peek


def test_step_clock_anchor_floor():
    clocks = FakeClocks(wall_ns=DAY0 * 1_000)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns)
    with pytest.raises(RuntimeError):
        clock.now_us()
    with pytest.raises(RuntimeError):
        clock.peek_us()
    clock.anchor(floor_us=DAY0 + 2 * US)  # start-up: the state's clock_last_us is 2 s ahead of the wall clock
    assert clock.peek_us() == DAY0 + 2 * US and clock.now_us() == DAY0 + 2 * US
    clocks.mono += 3 * 10**9
    assert clock.now_us() == DAY0 + 3 * US
    clock.anchor(floor_us=DAY0)  # a lower floor keeps the last value
    assert clock.now_us() == DAY0 + 3 * US
    with pytest.raises(ValueError):
        clock.anchor(floor_us=DAY0 + 0.5)


def test_step_clock_is_non_decreasing_under_random_wall_jumps_and_anchors():
    rng = np.random.default_rng(12)
    clocks = FakeClocks(wall_ns=DAY0 * 1_000)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns)
    clock.anchor()
    previous = clock.now_us()
    anchor_wall_us, anchor_mono_ns = DAY0, clocks.mono
    for _ in range(5_000):
        step_ns = int(rng.integers(0, 200_000)) * 1_000  # whole µs, so "monotonic µs since the anchor" is unambiguous
        clocks.mono += step_ns
        clocks.wall += step_ns + int(rng.integers(-50_000_000, 50_000_000))  # the wall clock drifts and jumps
        if rng.random() < 0.02:
            clock.anchor(floor_us=previous if rng.random() < 0.5 else None)
            anchor_wall_us, anchor_mono_ns = clocks.wall // 1_000, clocks.mono
        ts = clock.now_us()
        assert ts == max(previous, anchor_wall_us + (clocks.mono - anchor_mono_ns) // 1_000)
        previous = ts


def test_step_clock_runs_on_clock_monotonic_raw_unless_a_clock_is_injected(monkeypatch):
    """Clarification i: the step clock's monotonic source is CLOCK_MONOTONIC_RAW, not the NTP-slewed
    CLOCK_MONOTONIC (about 0.03 % fast in the Docker VM). It exists on Linux only, so it is faked here."""

    calls = []

    def clock_gettime_ns(clock_id: int) -> int:
        calls.append(clock_id)
        return 5_000_000_000 + 1_000 * len(calls)

    monkeypatch.setattr(time, "CLOCK_MONOTONIC_RAW", 4, raising=False)
    monkeypatch.setattr(time, "clock_gettime_ns", clock_gettime_ns, raising=False)
    assert monotonic_raw_ns() == 5_000_001_000 and calls == [4]
    clock = StepClock(wall_ns=lambda: DAY0 * 1_000)
    assert clock.monotonic_ns is monotonic_raw_ns
    clock.anchor()  # the second reading
    assert clock.now_us() == DAY0 + 1  # the third, 1,000 ns later
    assert calls == [4, 4, 4]
    clocks = FakeClocks(wall_ns=DAY0 * 1_000)  # what the unit tests inject
    assert StepClock(clocks.wall_ns, clocks.monotonic_ns).monotonic_ns == clocks.monotonic_ns


def test_step_clock_stays_below_its_ceiling_and_a_new_days_anchor_drops_the_floor():
    """Clarification k: ts stays below the end of the day whose files are open, non-decreasing; the new day's
    anchor is the wall clock with no floor, so a lead of the old day's step clock is not carried."""

    clocks = FakeClocks(wall_ns=(MIDNIGHT - 3 * US) * 1_000)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns)
    clock.anchor()
    clock.ceiling_us = MIDNIGHT
    clocks.wall -= 10**9  # the step clock now leads the wall clock by 1 s
    clocks.mono += 2 * 10**9
    assert clock.now_us() == MIDNIGHT - US
    seen = []
    for _ in range(4):
        clocks.mono += 600_000_000
        clocks.wall += 600_000_000
        seen.append(clock.now_us())
    assert seen == [MIDNIGHT - 400_000, MIDNIGHT - 1, MIDNIGHT - 1, MIDNIGHT - 1]  # held inside the day
    assert clock.peek_us() == MIDNIGHT - 1 and clock.anchored_us() == MIDNIGHT + 1_400_000
    assert clock.unheld_us() == MIDNIGHT + 1_400_000  # the step clock itself, for the §7 check
    clocks.wall = (MIDNIGHT + 250_000) * 1_000  # the wall clock in the new day: the switch
    clock.anchor_new_day(MIDNIGHT)
    assert clock.now_us() == MIDNIGHT + 250_000 and clock.ceiling_us == MIDNIGHT + DAY_US  # the lead is dropped
    clocks.mono += 500_000_000
    assert clock.now_us() == MIDNIGHT + 750_000
    for bad in (MIDNIGHT + 1, MIDNIGHT + 0.5):  # not a 00:00 UTC, not an integer: refused, nothing changed
        with pytest.raises(ValueError):
            clock.anchor_new_day(bad)
    assert clock.now_us() == MIDNIGHT + 750_000 and clock.ceiling_us == MIDNIGHT + DAY_US
    clocks.wall = (MIDNIGHT + DAY_US - 6_000) * 1_000  # the wall clock stepped back 6 ms across the next 00:00
    clock.anchor_new_day(MIDNIGHT + DAY_US)
    assert clock.now_us() == MIDNIGHT + DAY_US  # the anchor is kept inside the new day


# --- clarification j: the rate correction ---------------------------------------------------------------------------


def test_step_clock_with_c_0_is_the_clock_before_j_bit_for_bit():
    """Clarification j: with rate_ppm "0" (the default) every value is k's, the wall clock at the anchor + the RAW ns
    since it // 1000, for any elapsed time (so peek, now, unheld and the ceiling are k's too); the uncorrected clock
    is that same value whatever c is."""

    rng = np.random.default_rng(7)
    elapsed = [0, 1, 999, 1_000, 1_001, 86_400 * 10**9 + 999, 10**18 + 1] + [int(e) for e in rng.integers(0, 10**16, 300)]
    for rate in (None, "0", "12.8"):
        clocks = FakeClocks(wall_ns=(DAY0 + 1_234_567) * 1_000 + 891)
        clock = StepClock(clocks.wall_ns, clocks.monotonic_ns) if rate is None else StepClock(
            clocks.wall_ns, clocks.monotonic_ns, rate)
        assert clock.rate_ppm == (rate or "0")
        clock.anchor()
        mono0_ns = clocks.mono
        for e in elapsed:
            clocks.mono = mono0_ns + e
            k_value = DAY0 + 1_234_567 + e // 1000  # StepClock.anchored_us at 2f5f6e2 (clarification k)
            assert clock.uncorrected_us() == k_value
            if rate != "12.8":
                assert clock.anchored_us() == clock.unheld_us() == k_value


def test_step_clock_with_c_12_8_maps_86_400_x_1_0000128_s_of_raw_time_to_86_400_s():
    """Clarification j, the author's test: RAW 12.8 ppm fast; 86,400 × (1 + 12.8e-6) s of RAW time since the anchor
    is 86,400 s of step-clock time, ± 1 µs (exactly, here); the uncorrected clock keeps RAW's extra 1.10592 s."""

    clocks = FakeClocks(wall_ns=(DAY0 + 1_234_567) * 1_000 + 891)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns, rate_ppm="12.8")
    clock.anchor()
    start = clock.now_us()
    clocks.mono += 86_400 * 10**9 + 86_400 * 12_800  # 86,400 s × (1 + 12.8e-6) = 86,401.10592 s
    assert abs(clock.now_us() - start - 86_400 * US) <= 1
    assert clock.uncorrected_us() - start == 86_401_105_920


@pytest.mark.parametrize("rate_ppm", ["12.8", "0.1", "-3.7", "13.92", "1e-3", "0.000001"])
def test_step_clock_correction_is_the_exact_rational_floor(rate_ppm):
    """Clarification j: c is built exactly from its decimal string (never a binary float), and the corrected time is
    floor(RAW ns since the anchor / 1000 / (1 + c·1e-6)) in exact arithmetic, non-decreasing in the RAW time."""

    factor = 1 + Fraction(rate_ppm) / 10**6
    rng = np.random.default_rng(11)
    clocks = FakeClocks(wall_ns=DAY0 * 1_000)
    clock = StepClock(clocks.wall_ns, clocks.monotonic_ns, rate_ppm)
    clock.anchor()
    mono0_ns, previous = clocks.mono, None
    for e in sorted([0, 1, 999, 1_000, 10**12, 3 * 86_400 * 10**9] + [int(x) for x in rng.integers(0, 10**15, 300)]):
        clocks.mono = mono0_ns + e
        value = clock.anchored_us() - DAY0
        assert value == math.floor(Fraction(e, 1000) / factor)
        assert previous is None or value >= previous
        previous = value


def test_step_clock_refuses_a_rate_that_is_not_an_exact_decimal_string():
    clocks = FakeClocks(wall_ns=DAY0 * 1_000)
    for bad in (12.8, 0, Decimal("12.8"), None):
        with pytest.raises(TypeError):
            StepClock(clocks.wall_ns, clocks.monotonic_ns, bad)
    for bad in ("", "abc", "nan", "inf", "-1000000", "-2000000"):
        with pytest.raises(ValueError):
            StepClock(clocks.wall_ns, clocks.monotonic_ns, bad)


def test_the_rate_is_one_decimal_string_constant_in_the_adapter():
    """Clarification j: c is a single code constant, CLOCK_RAW_RATE_PPM in execution.py, a decimal string (the shell
    hands it to its StepClock: test_hb_bot_config)."""

    assert isinstance(CLOCK_RAW_RATE_PPM, str) and 0 < Fraction(CLOCK_RAW_RATE_PPM) < 100
    defined = []
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AnnAssign) else []
            defined += [path.name for target in targets if isinstance(target, ast.Name)
                        and target.id == "CLOCK_RAW_RATE_PPM"]
    assert defined == ["execution.py"]


def test_rate_limiter_counts_in_the_utc_day_it_is_given():
    """Clarification k: the bot gives the wall clock's UTC day; a close carried past 00:00 has its ts held in the
    old day but counts under the new day's cap (PROTOCOL §5.3), and the rolling window still runs on ts."""

    limiter = RateLimiter(max_per_window=100, window_us=10 * US, max_per_day=2)
    assert limiter.acquire(MIDNIGHT - 2 * US) is None and limiter.acquire(MIDNIGHT - US) is None
    assert limiter.acquire(MIDNIGHT - 1) == "rate_day"  # the old day's cap
    assert limiter.acquire(MIDNIGHT - 1, day_us=MIDNIGHT + 500_000) is None  # the same ts, the new day's cap
    assert limiter.day_orders == 1 and limiter.max_in_window == 3
    limiter.count_cancel(MIDNIGHT - 1, day_us=MIDNIGHT + 600_000)
    assert limiter.cancels == 1 and limiter.day_orders == 1
    with pytest.raises(ValueError):
        limiter.acquire(MIDNIGHT, day_us=MIDNIGHT - US)  # the day never goes back


def test_dt_seconds_equals_the_frozen_loaders_dt_bitwise():
    rng = np.random.default_rng(42)
    gaps = rng.integers(0, 3 * US, 50_000)
    gaps[rng.random(50_000) < 0.1] = 0
    gaps[rng.integers(0, 50_000, 20)] = rng.integers(60 * US, 6 * 3_600 * US, 20)  # restart gaps
    ts = DAY0 + np.cumsum(gaps).astype(np.int64)
    ours = np.array([dt_seconds(int(a), int(b)) for a, b in zip(ts[:-1], ts[1:])])
    reference = np.diff(ts).astype(np.float64) / 1_000_000.0
    assert np.array_equal(ours.view(np.int64), reference.view(np.int64))
    _, loader_dt = timestamps_us_to_datetime_and_deltas(pd.Index(ts))
    assert np.array_equal(ours.view(np.int64), loader_dt[1:].view(np.int64))
    assert dt_seconds(DAY0, DAY0) == 0.0


def test_dt_seconds_refuses_negative_or_non_integer_input():
    with pytest.raises(ValueError, match="backwards"):
        dt_seconds(DAY0 + 1, DAY0)
    for previous, ts in ((DAY0 + 0.5, DAY0 + 2), (DAY0, DAY0 + 1.5), (True, DAY0)):
        with pytest.raises(ValueError):
            dt_seconds(previous, ts)


# --- §5.4: the state file ---------------------------------------------------------------------------------------------


def test_save_state_writes_a_temporary_file_then_replaces_atomically(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    old = {"version": 1, "a": [1, 2]}
    save_state(path, old)
    new = {"version": 1, "a": [3], "nested": {"x": 0.1 + 0.2, "none": None, "flag": True, "ts": DAY0 + 1}}
    calls = []
    real_replace = os.replace

    def replace(source, target):
        calls.append((Path(source).name, Path(target).name, json.loads(Path(target).read_text(encoding="utf-8")),
                      json.loads(Path(source).read_text(encoding="utf-8"))))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    save_state(path, new)
    assert calls == [("state.json.tmp", "state.json", old, new)]  # the old state intact until the replace
    assert load_state(path) == new
    assert load_state(path)["nested"]["x"] == 0.1 + 0.2
    assert sorted(item.name for item in tmp_path.iterdir()) == ["state.json"]


def test_save_state_refuses_nan_and_keeps_the_previous_state(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, {"wealth": 1.0})
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            save_state(path, {"wealth": bad})
    with pytest.raises(TypeError):
        save_state(path, {"price": Decimal("0.08")})  # Decimals go into the state as strings
    assert load_state(path) == {"wealth": 1.0}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["state.json"]


def test_load_state_is_none_when_no_state_was_saved(tmp_path):
    assert load_state(tmp_path / "state.json") is None
    assert load_state(str(tmp_path / "missing" / "state.json")) is None


def test_state_round_trip_resumes_every_rule_identically(tmp_path):
    """A restart within the UTC day (§5.4): the features (S0, window), the ledger with its booked fees and DOGE
    reference, the rate window and counts, the safety flags and rejects, and the resting orders go through
    save_state / load_state into new objects, which then decide exactly as the originals do."""

    rng = np.random.default_rng(21)
    account, ledger, rate, safety = PaperAccount(), Ledger(), RateLimiter(max_per_day=400), Safety()
    features = FeatureEngine()
    ledger.start_day(account.balances["DOGE"])
    t = DAY0 + 3_600 * US
    features.reset(_book_row(8_100, 1, list(rng.uniform(1.0, 1e5, 10))), DayClock.t_elapsed(t, DAY0), 86_400.0, 0)
    safety.book(t, changed=True)
    for _ in range(300):
        previous, t = t, t + int(rng.integers(0, 300_000))
        features.update(_book_row(int(rng.integers(8_090, 8_110)), int(rng.integers(1, 4)),
                                  list(rng.uniform(1.0, 1e5, 10))),
                        dt_seconds(previous, t), DayClock.t_elapsed(t, DAY0), 86_400.0, ledger.q)
        if rate.acquire(t) is None and rng.random() < 0.5:
            side = "buy" if ledger.q < 10 and (ledger.q <= -10 or rng.random() < 0.5) else "sell"
            ledger.book_fill(*account.limit_fill(side, Decimal(int(rng.integers(8_090, 8_110))) * TICK_DEC,
                                                 Decimal(100)))
        ledger.mark(features.get_state()["last_mid"])
        if rng.random() < 0.02:
            safety.reject(t, "failure")
    resting = [_resting("buy", "0.08095", t - 400_000, 0.080975, "buy://DOGE-USDT/7"),
               _resting("sell", "0.08101", t - 900_000, 0.080985, "sell://DOGE-USDT/8")]
    path = tmp_path / "state.json"
    save_state(path, {"day": {"id": DayClock.day_id(DAY0), "start_us": DAY0}, "features": features.get_state(),
                      "ledger": ledger.get_state(), "rate": rate.get_state(), "safety": safety.get_state(),
                      "resting": [order.to_json() for order in resting], "clock_last_us": t})
    state = load_state(path)
    features2, ledger2, rate2, safety2 = FeatureEngine(), Ledger(), RateLimiter(max_per_day=400), Safety()
    features2.set_state(state["features"])
    ledger2.set_state(state["ledger"])
    rate2.set_state(state["rate"])
    safety2.set_state(state["safety"])
    resting2 = [RestingOrder.from_json(order) for order in state["resting"]]
    assert state["day"] == {"id": "2026-10-20", "start_us": DAY0}
    assert features2.get_state() == features.get_state()
    assert ledger2.get_state() == ledger.get_state() and ledger2.residual(account.balances["DOGE"]) == 0
    assert rate2.get_state() == rate.get_state() and safety2.get_state() == safety.get_state()
    assert resting2 == resting
    requoter = Requoter()
    t += 30 * US  # the first step after the gap carries the whole gap as its dt
    row = _book_row(8_100, 2, list(rng.uniform(1.0, 1e5, 10)))
    args = (row, dt_seconds(state["clock_last_us"], t), DayClock.t_elapsed(t, DAY0), 86_400.0, ledger.q)
    assert np.array_equal(features2.update(*args), features.update(*args))
    for order in resting:
        target = order.price + TICK_DEC
        assert requoter.decide(order.side, target, resting2[resting.index(order)], ledger2.q, t) == \
            requoter.decide(order.side, target, order, ledger.q, t)
    for _ in range(200):
        t += int(rng.integers(0, 100_000))
        assert rate2.acquire(t) == rate.acquire(t)
        assert _actions(safety2.reject(t, "failure")) == _actions(safety.reject(t, "failure"))
        mid = (int(rng.integers(8_090, 8_110)) + 0.5) * TICK
        assert ledger2.mark(mid) == ledger.mark(mid)
    assert rate2.get_state() == rate.get_state() and safety2.get_state() == safety.get_state()
    assert ledger2.get_state() == ledger.get_state()


# --- module rules ---------------------------------------------------------------------------------------------------


def test_execution_imports_no_hummingbot_module():
    """WO-HB2 deliverable 1: pure Python with no Hummingbot imports (it is unit-tested outside the container)."""

    tree = ast.parse((REPO_ROOT / "hummingbot_adapter" / "execution.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            found.append(node.module)
    assert found
    assert not [module for module in found if module.split(".")[0] == "hummingbot"]
    stdlib = sys.stdlib_module_names
    others = {module for module in found if module != "__future__" and module.split(".")[0] not in stdlib}
    assert others <= {"hummingbot_adapter.quotes"}


def test_defaults_are_the_protocol_constants():
    requoter = Requoter()
    assert (requoter.q_max, requoter.min_move, requoter.min_age_us) == (10, Decimal("0.00001"), US)
    rate = RateLimiter()
    assert (rate.max_per_window, rate.window_us, rate.max_per_day) == (100, 10 * US, 200_000)
    safety = Safety()
    assert (safety.q_halt, safety.residual_max) == (10, Decimal(1))
    assert (safety.stale_pause_us, safety.stale_halt_us) == (30 * US, 300 * US)
    assert (safety.reject_max, safety.reject_window_us, safety.drawdown_max) == (10, 60 * US, 40.0)
    assert (safety.clock_warn_s, safety.clock_pause_s) == (0.5, 2.0)
    assert not (safety.halted or safety.drawdown_paused or safety.stale_paused or safety.clock_paused)
    assert DayClock().close_before_us == 10 * US and Ledger().unit == 100
    assert (US_PER_SECOND, DAY_US) == (US, 86_400 * US)
    assert SIDES == ("buy", "sell") and FEE_ASSETS == ("DOGE", "USDT")
    for unit in (0, -100, 100.5):
        with pytest.raises(ValueError):
            Ledger(unit=unit)


def test_a_restored_stale_pause_lifts_only_on_a_real_book_change():
    safety = Safety()
    safety.book(DAY0, True)
    assert [event.action for event in safety.book(DAY0 + 30 * US, False)] == ["pause"]
    restored = Safety()
    restored.set_state(json.loads(json.dumps(safety.get_state())))  # a restart while paused
    assert restored.book(DAY0 + 40 * US, False) == [] and restored.stale_paused and not restored.quoting
    assert [event.action for event in restored.book(DAY0 + 41 * US, True)] == ["resume"] and restored.quoting
