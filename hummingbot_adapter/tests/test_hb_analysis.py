"""WO-HB3: the forward-test analysis (hummingbot_adapter.analysis) on hand-computed examples and synthetic day folders.

The day folders are written in the bots' own layout and finalized with telemetry.finalize_day (gzip, SHA256SUMS in the
day folder and the bot folder), so the verification is the one a real day passes."""

from __future__ import annotations

import csv
import gzip
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hummingbot_adapter import analysis
from hummingbot_adapter.provenance import file_sha256
from hummingbot_adapter.telemetry import SNAPSHOT_HEADER, finalize_day, headers
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES
from cvar_mm.stochastic_processes.fill_probability_models import DepthDependentPoissonFillFunction

US = 1_000_000
TICK = 1e-5
DEV_DAY = "2026-10-21"  # the warm-up: only with --dev
RUN_DAY = "2026-10-22"


def day_start(date: str) -> int:
    return analysis.day_bounds(date)[0]


# --- synthetic day folders ---------------------------------------------------------------------------------------


def step(ts: int, mid: float, q: int, wealth: float, *, previous: int | None, bid: float | None = None,
         ask: float | None = None, drawdown: float = 0.0, phase: str = "trade", residual: float = 0.0,
         rounded: tuple[int, int] = (0, 0)) -> dict:
    row = {"ts_us": ts, "utc": analysis.utc_iso(ts), "phase": phase,
           "dt": "" if previous is None else (ts - previous) / 1e6,
           "t_elapsed": 0.0, "tau": 0.0, "mid": mid, "gamma": 0.5, "skew": 0.0, "bid_target": mid - 2 * TICK,
           "ask_target": mid + 2 * TICK, "bid_quote": mid - 2 * TICK, "ask_quote": mid + 2 * TICK,
           "bid_rounded": rounded[0], "ask_rounded": rounded[1], "resting_bid": "" if bid is None else bid,
           "resting_ask": "" if ask is None else ask, "q": q, "cash": wealth - 100 * q * mid, "wealth": wealth,
           "peak": 0.0, "drawdown": drawdown, "fees_doge": 0, "fees_usdt": 0, "doge_balance": 1_000_000,
           "residual_doge": residual, "quoting": 1, "stale_paused": 0, "clock_paused": 0, "drawdown_paused": 0,
           "halted": 0}
    for index, name in enumerate(FORMULA_FEATURE_NAMES):
        row[name] = 0.0
        row[f"z_{name}"] = [0.5, -3.5, 10.0, -10.0, 1.0, 2.0, 0.0][index]
    row["action_gamma"] = row["action_skew"] = 0.0
    return row


def snapshot(ts: int, mid: float, spread_ticks: int = 1) -> dict:
    half = spread_ticks * TICK / 2
    return {"exchange": "binance", "symbol": "DOGEUSDT", "timestamp": ts, "local_timestamp": ts,
            "asks[0].price": round(mid + half, 6), "asks[0].amount": 100, "bids[0].price": round(mid - half, 6),
            "bids[0].amount": 100}


def write_csv(path: Path, header: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if row.get(column) is None else row.get(column, "") for column in header])


def write_day(root: Path, bot: str, date: str, steps: list[dict], *, fills: list[dict] = (), orders: list[dict] = (),
              trades: list[dict] = (), health: list[dict] = (), finalize: bool = True) -> Path:
    model = analysis.MODEL_OF[bot]
    folder = root / bot / date
    folder.mkdir(parents=True)
    tables = headers(model)
    write_csv(folder / f"steps_{date}.csv", tables["steps"], steps)
    write_csv(folder / f"snapshots_{date}.csv", SNAPSHOT_HEADER, [snapshot(row["ts_us"], row["mid"]) for row in steps])
    write_csv(folder / f"fills_{date}.csv", tables["fills"], list(fills))
    write_csv(folder / f"orders_{date}.csv", tables["orders"], list(orders))
    write_csv(folder / f"trades_{date}.csv", tables["trades"], list(trades))
    write_csv(folder / f"health_{date}.csv", tables["health"], list(health))
    if finalize:
        finalize_day(folder, date)
    return folder


def health_row(ts: int, event: str, **detail) -> dict:
    return {"ts_us": ts, "utc": analysis.utc_iso(ts), "event": event, "detail": json.dumps(detail, sort_keys=True)}


def fill_row(ts: int, side: str, price: str, *, post_ts: int, mid_at_post: float, mid_at_fill: float,
             close: int = 0, amount: str = "100", fee: str = "0.1", fee_asset: str = "DOGE",
             order_id: str | None = None) -> dict:
    sign = 1 if side == "buy" else -1
    return {"ts_us": ts, "utc": analysis.utc_iso(ts), "order_id": order_id or f"{side}://{ts}", "side": side,
            "order_type": "market" if close else "limit", "price": price, "amount": amount, "mid_at_fill": mid_at_fill,
            "delta_fill_ticks": sign * (mid_at_fill - float(price)) / TICK, "post_ts_us": post_ts,
            "mid_at_post": mid_at_post, "delta_post_ticks": sign * (mid_at_post - float(price)) / TICK, "fee": fee,
            "fee_asset": fee_asset, "close": close, "exchange_trade_id": str(ts)}


def market_submit(ts: int, order_id: str, side: str, amount: str, reason: str) -> dict:
    return {"ts_us": ts, "utc": "", "event": "submit", "order_id": order_id, "side": side, "order_type": "market",
            "price": "", "amount": amount, "mid_at_post": 0.080005, "reason": reason}


def scenario(root: Path, date: str, *, late_fill: bool = False) -> dict[str, Path]:
    """Two bots on one synthetic day: steps every second for a minute, a 40 s gap, the 23:59:50 close and a wait step;
    one buy and one sell limit fill, and the 23:59:50 close at market (the bots' q path is synthetic, not the fills'
    sum). late_fill adds a limit fill at 23:59:49.5, 9.5 s before the file's last step."""

    start = day_start(date)
    out = {}
    for bot, sign in (("hb_b0", 1), ("hb_b3", -1)):
        times = [start + 100_000 + k * US for k in range(60)] + [start + 100_000 + 100 * US]
        times += [start + analysis.CLOSE_US - 2 * US, start + analysis.CLOSE_US + 3 * US, start + analysis.DAY_US - US]
        steps, previous = [], None
        for index, ts in enumerate(times):
            q = 0 if index < 10 or index >= len(times) - 1 else sign * (1 if index < 30 else 9)
            wealth = 0.0 if index < 10 else -0.001 * (index % 7)
            steps.append(step(ts, 0.08 + (index % 3) * TICK, q, wealth, previous=previous,
                              bid=0.08 - 2 * TICK, ask=0.08 + 2 * TICK, drawdown=max(0.0, 0.001 * (index % 7))))
            previous = ts
        fills = [fill_row(start + 9_500_000, "buy" if sign > 0 else "sell", "0.07998" if sign > 0 else "0.08002",
                          post_ts=start + 5_100_000, mid_at_post=0.08, mid_at_fill=0.07999 if sign > 0 else 0.08001),
                 fill_row(start + 29_500_000, "sell" if sign > 0 else "buy", "0.08002" if sign > 0 else "0.07998",
                          post_ts=start + 25_100_000, mid_at_post=0.08, mid_at_fill=0.08001 if sign > 0 else 0.07999,
                          fee="0.008", fee_asset="USDT")]
        if late_fill:
            fills.append(fill_row(start + analysis.CLOSE_US - 500_000, "buy", "0.07998",
                                  post_ts=start + analysis.CLOSE_US - 3 * US, mid_at_post=0.08, mid_at_fill=0.07999))
        close_side = "sell" if sign > 0 else "buy"
        fills.append(fill_row(start + analysis.CLOSE_US + 2 * US, close_side,
                              "0.07999000000000000117" if sign > 0 else "0.08001", post_ts=start + analysis.CLOSE_US,
                              mid_at_post=0.080005, mid_at_fill=0.080005, close=1, amount="900",
                              fee="0.07199" if sign > 0 else "0.9", fee_asset="USDT" if sign > 0 else "DOGE",
                              order_id=f"{close_side}://close"))
        orders = [{"ts_us": start + 100_000 + k * US, "utc": "", "event": "submit", "order_id": f"o{k}", "side": "buy",
                   "order_type": "limit", "price": "0.07998", "amount": "100", "mid_at_post": 0.08, "reason": "quote"}
                  for k in range(3)]
        orders += [{"ts_us": start + 100_000 + 2 * US, "utc": "", "event": "cancel", "order_id": "o1", "side": "buy",
                    "order_type": "limit", "price": "0.07998", "amount": "100", "mid_at_post": "", "reason": "requote"},
                   market_submit(start + analysis.CLOSE_US, f"{close_side}://close", close_side, "900", "day_close")]
        trades = [{"ts_us": start + 9_400_000, "utc": "", "exchange_ts": "1", "trade_id": "t1",
                   "price": 0.07998 if sign > 0 else 0.08002, "amount": 50.0, "side": "sell" if sign > 0 else "buy"}]
        health = [health_row(start + 50_000, "reset", doge_reference="1000000"),
                  health_row(start + 61 * US, "pause", trigger="stale_book", detail="x", kind="incident"),
                  health_row(start + 100 * US, "resume", trigger="stale_book", detail="x", kind="incident"),
                  health_row(start + 3600 * US, "clock_check", error=None, offset_s=-0.02, wall_offset_s=0.01),
                  health_row(start + 7200 * US, "feed", kind="ws_closed", level="WARNING", message="closed")]
        out[bot] = write_day(root, bot, date, steps, fills=fills, orders=orders, trades=trades, health=health)
    return out


@pytest.fixture
def inputs_dir(tmp_path: Path) -> dict[str, Path]:
    params = tmp_path / "hb_calibration_test.json"
    params.write_text(json.dumps({"sigma": 7.5e-06, "A": 0.095, "kappa": 41_000.0}), encoding="utf-8")
    tardis = tmp_path / "tardis_2026-10-01.csv"
    start = day_start("2026-10-01")
    write_csv(tardis, SNAPSHOT_HEADER, [snapshot(start + k * US, 0.09 + (k % 4) * TICK, 1 + k % 2) for k in range(50)])
    run_log = tmp_path / "RUN_LOG.md"
    run_log.write_text("# Run log (UTC)\n\n2026-10-21T18:15:40Z: before the run, about 2026-10-21.\n\n"
                       "2026-10-22T21:57:08Z: planned network change for 2026-10-22 and 2026-10-23.\n- both bots\n\n"
                       "2026-10-23T05:49:24Z: day 2026-10-21, incident at 22:47Z on hb_b0 only.\n\n"
                       "2026-10-24T00:30:00Z: after the run: 2026-10-21 was quiet on both bots.\n", encoding="utf-8")
    return {"params": params, "tardis": tardis, "run_log": run_log, "root": tmp_path / "hb_data",
            "out": tmp_path / "hb_analysis" / "label"}


def cli(inputs_dir: dict[str, Path], days: list[str], *extra: str) -> list[str]:
    return ["--data-root", str(inputs_dir["root"]), "--days", *days, "--params", str(inputs_dir["params"]),
            "--calibration-day-file", str(inputs_dir["tardis"]), "--run-log", str(inputs_dir["run_log"]),
            "--out-dir", str(inputs_dir["out"]), *extra]


# --- markouts, deltas and retention ------------------------------------------------------------------------------


def test_markouts_deltas_and_retention_on_a_known_mid_path():
    step_ts = np.array([0, 1, 2, 5, 11, 61], dtype=np.int64) * US + 1_000
    step_mid = np.array([0.10000, 0.10002, 0.10001, 0.09997, 0.10005, 0.09990])
    fill_ts = np.array([1 * US + 1_000, 1 * US + 500_000, 51 * US + 1_000], dtype=np.int64)
    sign = np.array([1.0, -1.0, 1.0])  # a buy, a sell, a buy whose 60 s markout lies past the last step
    price = np.array([0.09998, 0.10004, 0.09999])
    mid_at_post = np.array([0.10000, 0.10001, 0.10000])
    delta_post = sign * (mid_at_post - price) / TICK
    assert delta_post == pytest.approx([2.0, 3.0, 1.0], abs=1e-9)
    # the as-of lookup at an exact step time takes that step's mid
    assert analysis.asof_mid(step_ts, step_mid, np.array([2 * US + 1_000]))[0] == 0.10001
    assert analysis.asof_mid(step_ts, step_mid, np.array([2 * US + 999]))[0] == 0.10002
    marks = analysis.markouts(fill_ts, sign, price, step_ts, step_mid)
    # h = 1 s: S(2.001 s) = 0.10001 (exact boundary); S(2.5 s) = 0.10001; S(52.001 s) = 0.10005
    assert marks[1] == pytest.approx([(0.10001 - 0.09998) / TICK, -(0.10001 - 0.10004) / TICK,
                                      (0.10005 - 0.09999) / TICK], abs=1e-9)
    # h = 10 s: S(11.001 s) = 0.10005 (exact boundary), S(11.5 s) = 0.10005, S(61.001 s) = 0.09990 (last step)
    assert marks[10] == pytest.approx([7.0, -1.0, -9.0], abs=1e-9)
    # h = 60 s: S(61.001 s) = 0.09990 (the last step, kept); 61.5 s and 111.001 s lie past the last step: dropped
    assert marks[60][0] == pytest.approx(-8.0, abs=1e-9)
    assert np.isnan(marks[60][1]) and np.isnan(marks[60][2])
    assert analysis.retention(marks[10], delta_post) == pytest.approx((7.0 - 1.0 - 9.0) / 6.0, abs=1e-9)
    # R(60) over the kept fill only: -8 / 2
    assert analysis.retention(marks[60], delta_post) == pytest.approx(-4.0, abs=1e-9)


def test_markouts_and_deltas_through_a_day(tmp_path):
    date = DEV_DAY
    folders = scenario(tmp_path, date)
    data = analysis.read_day(analysis.verify_day(tmp_path, "hb_b0", date))
    result = analysis.analyse_day(data, A=0.095, kappa=41_000.0)
    qb = result["qb"]
    assert (qb["limit_fills"], qb["buys"], qb["sells"], qb["unknown_order_fills"]) == (2, 1, 1, 0)
    assert qb["mean_delta_post_ticks"] == pytest.approx(2.0, abs=1e-9)
    assert qb["mean_delta_fill_ticks"] == pytest.approx(1.0, abs=1e-9)
    assert qb["delta_post_mismatches"] == qb["delta_fill_mismatches"] == 0
    assert qb["dropped_1s"] == 0 and qb["dropped_60s"] == 0
    assert folders["hb_b0"].name == date


# --- Q-C -----------------------------------------------------------------------------------------------------------


def test_fee_curve_and_break_even_fees():
    curve = analysis.fee_curve(P_pre=-0.5, P_post=-0.6, F=1000.0, C=50.0)
    assert [point["fee_pct"] for point in curve] == [0.0, 0.01, 0.02, 0.075, 0.10]
    assert curve[0] == {"fee_pct": 0.0, "P_pre_net_usdt": -0.5, "P_post_net_usdt": -0.6}
    assert curve[1]["P_pre_net_usdt"] == pytest.approx(-0.5 - 0.0001 * 1000.0)
    assert curve[3]["P_post_net_usdt"] == pytest.approx(-0.6 - 0.00075 * 1050.0)
    assert curve[4]["P_pre_net_usdt"] == pytest.approx(-1.5)
    assert analysis.break_even_pct(0.8, 1000.0) == pytest.approx(0.08)
    assert analysis.break_even_pct(-0.6, 1050.0) == pytest.approx(-0.6 / 1050.0 * 100)
    assert analysis.break_even_pct(0.3, 0.0) is None  # n/a: no fill


def test_q_c_notional_close_and_fees_through_a_day(tmp_path):
    scenario(tmp_path, DEV_DAY)
    b0 = analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path, "hb_b0", DEV_DAY)), 0.095, 41_000.0)
    b3 = analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path, "hb_b3", DEV_DAY)), 0.095, 41_000.0)
    # F = 0.07998 * 100 + 0.08002 * 100; C = the market fill's binary-expansion price rounded to the tick, x 900
    assert b0["qc"]["F_exact"] == "16.00000" and b0["qc"]["C_exact"] == "71.99100"
    assert b3["qc"]["C_exact"] == "72.00900"
    # fees: 0.1 DOGE x 0.07998 + 0.008 USDT on the limit fills; the close 0.07199 USDT (B0), 0.9 DOGE x 0.08001 (B3)
    assert b0["qc"]["paper_fees_limit_usdt"] == pytest.approx(0.1 * 0.07998 + 0.008, abs=1e-15)
    assert b3["qc"]["paper_fees_close_usdt"] == pytest.approx(0.9 * 0.08001, abs=1e-15)
    # the close cost: half-spread against the mid at the submission, plus the fee
    assert b0["qc"]["close_cost_usdt"] == pytest.approx((0.080005 - 0.07999) * 900 + 0.07199, abs=1e-12)
    assert b3["qc"]["close_cost_usdt"] == pytest.approx((0.08001 - 0.080005) * 900 + 0.9 * 0.08001, abs=1e-12)
    P_pre, F = b0["qc"]["P_pre_usdt"], b0["qc"]["F_usdt"]
    assert b0["qc"]["f_star_pre_pct"] == pytest.approx(P_pre / F * 100)
    assert b0["qc"]["limit_fills_after_last_window_step"] == 0


# --- PIT -------------------------------------------------------------------------------------------------------------


def test_pit_with_ties_and_at_the_extremes():
    shadow = np.array([1.0, 2.0, 2.0, 3.0, 4.0] * 8)  # 40 paths: 8 below 2.0, 16 equal
    assert analysis.pit(shadow, 2.0) == (8 + 0.5 * 16) / 40
    assert analysis.pit(np.arange(40.0), 100.0) == 1.0
    assert analysis.pit(np.arange(40.0), -1.0) == 0.0


# --- Q-A time weights ----------------------------------------------------------------------------------------------


def test_time_weights_follow_fast_simulate_on_a_five_step_example():
    """fast_simulate: after the fills of the interval to step i, |q_i| is weighted by dt_i (step 0 carries nothing)."""

    start = day_start(RUN_DAY)
    _, close, _ = analysis.day_bounds(RUN_DAY)
    ts = start + np.array([0, 1, 3, 3, 6], dtype=np.int64) * US  # a dt = 0 step at 3 s
    q = np.array([0, 2, -8, 5, 9])
    assert analysis.interval_weights(ts, close).tolist() == [0.0, 1.0, 2.0, 0.0, 3.0]
    stats = analysis.inventory_stats(ts, q, close)
    assert stats["mean_abs_q"] == pytest.approx((2 * 1 + 8 * 2 + 5 * 0 + 9 * 3) / 6)
    assert stats["near_cap_share"] == pytest.approx((2 + 3) / 6)
    assert stats["weighted_s"] == 6.0


def test_rows_from_23_59_50_carry_no_weight():
    start = day_start(RUN_DAY)
    _, close, _ = analysis.day_bounds(RUN_DAY)
    ts = np.array([close - 4 * US, close - 2 * US, close, close + US, close + 5 * US], dtype=np.int64)
    q = np.array([10, 10, -10, -10, 0])
    weights = analysis.interval_weights(ts, close)
    assert weights.tolist() == [0.0, 2.0, 0.0, 0.0, 0.0]
    assert analysis.inventory_stats(ts, q, close)["mean_abs_q"] == 10.0
    assert start < ts[0]


def test_drawdown_runs_from_the_day_start_at_zero():
    assert analysis.drawdown_path(np.array([-1.0, 0.5, 0.2, 0.7, -0.3])).tolist() == pytest.approx(
        [1.0, 0.0, 0.3, 0.0, 1.0])


# --- E (Q-B) -------------------------------------------------------------------------------------------------------


def test_expected_fills_on_a_hand_computed_three_step_example():
    A, kappa = 0.5, 50_000.0
    start = day_start(RUN_DAY)
    _, close, _ = analysis.day_bounds(RUN_DAY)
    ts = start + np.array([0, 1, 3], dtype=np.int64) * US
    mid = np.array([0.1, 0.10001, 0.1])
    bid = np.array([0.1 - 1e-5, np.nan, 0.09])  # resting after each step; step 2's orders meet no interval
    ask = np.array([0.1 + 2e-5, 0.10001 + 3e-5, 0.2])
    result = analysis.expected_fills(ts, mid, bid, ask, close, A, kappa)
    # interval 1 (dt 1 s): step 0's bid at 1 tick and ask at 2 ticks; interval 2 (dt 2 s): step 1's ask at 3 ticks
    by_hand = [1 - math.exp(-A * math.exp(-kappa * 1e-5) * 1.0), 1 - math.exp(-A * math.exp(-kappa * 2e-5) * 1.0),
               1 - math.exp(-A * math.exp(-kappa * 3e-5) * 2.0)]
    assert result["E_bid"] == pytest.approx(by_hand[0], rel=1e-9)
    assert result["E_ask"] == pytest.approx(by_hand[1] + by_hand[2], rel=1e-9)
    assert result["E"] == pytest.approx(sum(by_hand), rel=1e-9)
    assert result["intervals"] == 2
    frozen = DepthDependentPoissonFillFunction(A, kappa)  # the simulator's own execution probability
    assert frozen.execution_probabilities(np.array([1e-5, 2e-5]), 1.0) == pytest.approx(by_hand[:2], rel=1e-9)
    assert frozen.execution_probabilities(np.array([3e-5]), 2.0)[0] == pytest.approx(by_hand[2], rel=1e-9)


# --- Q-D helpers -------------------------------------------------------------------------------------------------


def test_rolling_ten_second_window_and_gaps():
    ts = np.array([0, 1, 1, 9, 10, 10.000001, 25], dtype=np.float64)
    ts_us = np.round(ts * US).astype(np.int64)
    assert analysis.max_in_window(ts_us) == 5  # (0.000001 s, 10.000001 s]: 1, 1, 9, 10, 10.000001
    gaps = analysis.step_gaps(np.array([40, 50, 100], dtype=np.int64) * US, 0, 200 * US)
    assert [(a // US, b // US) for a, b in gaps] == [(0, 40), (50, 100), (100, 200)]
    assert analysis.union_seconds([(0, 10 * US), (5 * US, 12 * US), (20 * US, 30 * US)], 0, 25 * US) == 17.0


def test_trade_fill_classification():
    # the paper engine's test is Decimal against float, exactly: float(0.08198) lies below 0.08198 and float(0.08013)
    # above 0.08013, so a taker sell printed "at" a bid of 0.08198 is through it for the engine, at 0.08013 it is not
    assert Decimal(0.08198) < Decimal("0.08198") and Decimal(0.08013) > Decimal("0.08013")
    trade_ts = np.array([100, 200, 300, 400, 600], dtype=np.int64) * 1_000
    trade_prices = np.array([0.08198, 0.08197, 0.08013, 0.08198, 0.08013])
    trade_side = np.array(["sell", "sell", "buy", "sell", "sell"], dtype=object)
    labels = analysis.classify_trade_fills(
        fill_ts=np.array([250_000, 120_000, 350_000, 500_000_000, 650_000], dtype=np.int64),
        post_ts=np.array([0, 50_000, 0, 0, 450_000], dtype=object),
        sign=np.array([1.0, 1.0, -1.0, 1.0, 1.0]),
        fill_prices=["0.08198", "0.08198", "0.08013", "0.08198", "0.08013"],
        trade_ts=trade_ts, trade_prices=trade_prices, trade_side=trade_side)
    # a bid with a taker sell through it in ticks (0.08197); a bid with one at its price, through for the engine; an
    # ask with a taker buy at its price, through for the engine (float above); no trade in the window; a bid of
    # 0.08013 with a taker sell at its price that the engine does not count as through
    assert labels == ["through", "at_price_engine_through", "at_price_engine_through", "no_trade",
                      "at_price_not_engine_through"]


# --- refusals ----------------------------------------------------------------------------------------------------


def test_a_date_outside_the_run_is_refused_without_dev(tmp_path, inputs_dir, capsys):
    scenario(inputs_dir["root"], DEV_DAY)
    assert analysis.main(cli(inputs_dir, [DEV_DAY])) == 2
    assert "outside the main run" in capsys.readouterr().err
    assert not inputs_dir["out"].exists()


def test_a_day_without_sha256sums_is_refused(tmp_path, inputs_dir, capsys):
    start = day_start(RUN_DAY)
    for bot in ("hb_b0", "hb_b3"):
        write_day(inputs_dir["root"], bot, RUN_DAY, [step(start + US, 0.08, 0, 0.0, previous=None)], finalize=False)
    assert analysis.main(cli(inputs_dir, [RUN_DAY])) == 2
    assert "not finalized" in capsys.readouterr().err


def test_a_day_whose_hash_fails_is_refused(tmp_path, inputs_dir, capsys):
    scenario(inputs_dir["root"], RUN_DAY)
    path = inputs_dir["root"] / "hb_b3" / RUN_DAY / f"fills_{RUN_DAY}.csv.gz"
    path.write_bytes(gzip.compress(b"ts_us\n1\n"))
    assert analysis.main(cli(inputs_dir, [RUN_DAY])) == 2
    assert "SHA-256" in capsys.readouterr().err
    assert not inputs_dir["out"].exists()


def test_an_output_folder_with_files_is_never_overwritten(tmp_path, inputs_dir, capsys):
    scenario(inputs_dir["root"], RUN_DAY)
    inputs_dir["out"].mkdir(parents=True)
    (inputs_dir["out"] / "results.json").write_text("{}", encoding="utf-8")
    assert analysis.main(cli(inputs_dir, [RUN_DAY])) == 2
    assert "never overwritten" in capsys.readouterr().err


# --- the outputs ---------------------------------------------------------------------------------------------------


def paper_values(root: Path, date: str) -> dict[str, dict[str, float]]:
    """The paper's Q-A values of both bots on a synthetic day, through the analysis itself."""

    values = {}
    for bot in ("hb_b0", "hb_b3"):
        qa = analysis.analyse_day(analysis.read_day(analysis.verify_day(root, bot, date)), 0.095, 41_000.0)["qa"]
        values[bot] = {"P_pre": qa["P_pre_usdt"], "maxdd": qa["maxdd_pre_usdt"], "mean_abs_q": qa["mean_abs_q"],
                       "near_cap": qa["near_cap_share"]}
    return values


SHADOW_PITS = {"P_pre/100": (10 + 0.5 * 5) / 40, "MaxDD/100": 30 / 40, "mean |q|": 0.0, "near-cap share": 39.5 / 40}


def shadow_files(folder: Path, root: Path, params: Path, date: str, paper: dict[str, dict[str, float]]) -> None:
    """Synthetic shadow replays of both bots, each recorded against that bot's own snapshots file. Each column
    straddles the paper value with its own counts below and tied (SHADOW_PITS), so a column or scale mix-up shows."""

    folder.mkdir()
    for bot, name in (("hb_b0", "b0"), ("hb_b3", "b3")):
        p = paper[bot]
        columns = {
            "Final PnL": [p["P_pre"] / 100 - 1e-6] * 10 + [p["P_pre"] / 100] * 5 + [p["P_pre"] / 100 + 1e-6] * 25,
            "Max DD": [p["maxdd"] / 100 - 1e-7] * 30 + [p["maxdd"] / 100 + 1e-6] * 10,
            "Mean |q|": [p["mean_abs_q"] + 0.5] * 40,
            "Near Cap Fraction": [p["near_cap"] - 0.01] * 39 + [p["near_cap"]],
            "Terminal q": [-10.0] * 20 + [0.0] * 20,
        }
        keys = [(seed, rollout) for seed in (42, 43) for rollout in range(20)]
        rows = [{"Day": date, "Fill Seed": seed, "Rollout": rollout, **{c: v[i] for c, v in columns.items()}}
                for i, (seed, rollout) in enumerate(keys)]
        episodes = folder / f"shadow_{name}_{date}_episodes.csv"
        pd.DataFrame(rows).to_csv(episodes, index=False, lineterminator="\n")
        snapshots = root / bot / date / f"snapshots_{date}.csv.gz"
        record = {"model": name.upper(), "date": date, "pipeline_settings": True,
                  "inputs": {"day_file": {"sha256": file_sha256(snapshots)}, "params": {"sha256": file_sha256(params)}},
                  "episodes": {"sha256": file_sha256(episodes)}}
        (folder / f"shadow_{name}_{date}.json").write_text(json.dumps(record), encoding="utf-8")


def test_every_output_table_is_keyed_by_bot_and_day(tmp_path, inputs_dir):
    scenario(inputs_dir["root"], DEV_DAY)
    shadow_files(tmp_path / "shadow", inputs_dir["root"], inputs_dir["params"], DEV_DAY,
                 paper_values(inputs_dir["root"], DEV_DAY))
    assert analysis.main(cli(inputs_dir, [DEV_DAY], "--dev", "--shadow-dir", str(tmp_path / "shadow"))) == 0
    target = inputs_dir["out"] / "dev"
    assert sorted(path.name for path in inputs_dir["out"].iterdir()) == ["dev"]  # --dev writes only to <out-dir>/dev
    results = json.loads((target / "results.json").read_text(encoding="utf-8"))
    assert results["dev"] is True and results["label"] == analysis.LABEL
    assert set(results["tables"]) >= {"qa_inventory", "qb_markouts", "qb_expected_fills", "qc_fee_curve",
                                      "qc_break_even", "s7_afterwards", "qd_operations", "qd_gaps", "qd_exploratory",
                                      "qd_network_window", "runlog_entries", "data_description", "feature_shift",
                                      "shadow_vs_paper", "qb_fill_classes", "pnl_split"}
    for name, table in results["tables"].items():
        frame = pd.read_csv(target / f"{name}.csv", dtype={"day": str})
        assert list(frame.columns[:2]) == ["bot", "day"], name
        assert table["title"].startswith(analysis.DEV_PREFIX) and analysis.LABEL in table["title"], name
        for bot, day in zip(frame["bot"], frame["day"]):
            if name == "data_description" and bot == "tardis_calibration":
                assert day == "2026-10-01"
            else:
                assert (bot, day) in {("hb_b0", DEV_DAY), ("hb_b3", DEV_DAY)}, name
    # RUN_LOG: entries written from the run's start on, keyed by the analysed days they name; full text kept
    entries = pd.read_csv(target / "runlog_entries.csv")
    assert entries[["bot", "written_utc"]].values.tolist() == [
        ["hb_b0", "2026-10-23T05:49:24Z"], ["hb_b0", "2026-10-24T00:30:00Z"], ["hb_b3", "2026-10-24T00:30:00Z"]]
    assert entries["entry"].iloc[0] == "day 2026-10-21, incident at 22:47Z on hb_b0 only."
    markdown = (target / "forward_test_tables.md").read_text(encoding="utf-8")
    headings = [line for line in markdown.splitlines() if line.startswith("#")]
    assert headings and all(line.lstrip("# ").startswith(analysis.DEV_PREFIX) for line in headings)
    assert analysis.DEV_PREFIX in (target / "appendix_table.tex").read_text(encoding="utf-8")
    for name in ("fig_forward_inventory.pdf", "fig_forward_inventory.png", "fig_forward_inventory_data.csv"):
        assert (target / name).stat().st_size > 0
    sums = (target / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in sums}
    assert set(listed) == {path.name for path in target.iterdir()} - {"SHA256SUMS"}
    assert all(file_sha256(target / name) == digest for name, digest in listed.items())
    # shadow against paper: each metric's own column and scale, its PIT, and the q row without a PIT
    shadow = pd.read_csv(target / "shadow_vs_paper.csv")
    assert len(shadow) == 10 and set(shadow["paths"]) == {40}
    qa = pd.read_csv(target / "qa_inventory.csv").set_index("bot")
    for bot in ("hb_b0", "hb_b3"):
        rows = shadow[shadow["bot"] == bot].set_index("metric")
        for metric, expected in SHADOW_PITS.items():
            assert rows.loc[metric, "pit"] == expected, (bot, metric)
        assert rows.loc["P_pre/100", "paper"] == qa.loc[bot, "P_pre_usdt"] / 100
        assert rows.loc["MaxDD/100", "paper"] == qa.loc[bot, "maxdd_pre_usdt"] / 100
        assert (rows.loc["P_pre/100", "below"], rows.loc["P_pre/100", "ties"]) == (10, 5)
        terminal = rows.loc["q (paper: before the close; shadow: end of file)"]
        assert np.isnan(terminal["pit"]) and terminal["shadow_share_q_le_minus_8"] == 0.5
        assert terminal["paper"] == qa.loc[bot, "q_pre"]
        assert rows.loc["near-cap share", "role"] == analysis.EXPLORATORY_ROLE
    tex = (target / "appendix_table.tex").read_text(encoding="utf-8")
    assert tex.count(" & 0.312 & 0.312 \\\\") == 1  # the PIT P_pre row: B0 and B3 (the columns of one day)


def test_a_shadow_replay_of_another_file_is_refused(tmp_path, inputs_dir, capsys):
    scenario(inputs_dir["root"], DEV_DAY)
    zeros = {"P_pre": 0.0, "maxdd": 0.0, "mean_abs_q": 0.0, "near_cap": 0.0}
    shadow_files(tmp_path / "shadow", inputs_dir["root"], inputs_dir["params"], DEV_DAY,
                 {"hb_b0": zeros, "hb_b3": zeros})
    record_path = tmp_path / "shadow" / f"shadow_b3_{DEV_DAY}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["inputs"]["day_file"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert analysis.main(cli(inputs_dir, [DEV_DAY], "--dev", "--shadow-dir", str(tmp_path / "shadow"))) == 2
    assert "not this bot's snapshots file" in capsys.readouterr().err
    assert not inputs_dir["out"].exists()  # refused before any output


def test_q_a_and_q_d_through_a_day(tmp_path):
    scenario(tmp_path, DEV_DAY)
    result = analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path, "hb_b3", DEV_DAY)), 0.095, 41_000.0)
    qa, qd = result["qa"], result["qd"]
    assert qa["q_pre"] == -9 and qa["window_steps"] == 62 and qa["dt_mismatches"] == 0
    assert qa["maxdd_pre_usdt"] == pytest.approx(qa["maxdd_logged_usdt"])
    assert qd["stale_book_pauses"] == 1 and qd["stale_book_pause_s"] == pytest.approx(39.0)
    assert qd["ws_reconnects"] == 1 and qd["clock_checks"] == 1 and qd["max_abs_step_offset_s"] == pytest.approx(0.02)
    assert qd["new_orders"] == 4 and qd["max_new_orders_10s"] == 3 and qd["cancels"] == 1  # 3 quotes and the close
    # gaps over 30 s: 59.1 s -> 100.1 s and 100.1 s -> 23:59:48; 23:59:59 -> 24:00 is only 1 s
    assert qd["gaps_over_30s"] == 2
    exploratory = result["exploratory"]
    # the taker buy printed at the sell's price 0.08002: through for the engine iff float(0.08002) lies above it
    engine = Decimal("0.08002") < Decimal(0.08002)
    assert exploratory["at_price_trade_fills"] == 1
    assert (exploratory["at_price_engine_through"], exploratory["at_price_not_engine_through"]) == (engine, not engine)
    assert exploratory["crossed_or_locked_source"].startswith("health rows")  # no day_end row, no heartbeat
    assert [item["feature"] for item in result["feature_shift"]] == list(analysis.FORMULA_FEATURE_NAMES)
    assert result["feature_shift"][2]["share_abs_z_clipped"] == 1.0
    assert result["feature_shift"][1]["share_abs_z_gt_3"] == 1.0


def test_a_markout_past_the_file_end_is_dropped_and_counted(tmp_path):
    """A limit fill at 23:59:49.5, 9.5 s before the file's last step: its 1 s markout is kept, its 10 s and 60 s
    markouts are dropped and counted, and the means and R(h) of those horizons leave it out."""

    scenario(tmp_path / "base", DEV_DAY)
    scenario(tmp_path / "late", DEV_DAY, late_fill=True)
    base, late = (analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path / name, "hb_b0", DEV_DAY)),
                                       0.095, 41_000.0)["qb"] for name in ("base", "late"))
    assert (late["limit_fills"], late["dropped_1s"], late["dropped_10s"], late["dropped_60s"]) == (3, 0, 1, 1)
    assert (base["dropped_1s"], base["dropped_10s"], base["dropped_60s"]) == (0, 0, 0)
    for key in ("mean_m_10s", "R_10s", "mean_m_60s", "R_60s"):
        assert late[key] == pytest.approx(base[key], abs=1e-12), key
    assert late["mean_m_1s"] != pytest.approx(base["mean_m_1s"])  # the late fill's 1 s markout is in


def consistent_day(root: Path, date: str, close_purpose: str, close_ts_offset_us: int) -> None:
    """A minimal B0 day whose fills and steps agree: a 100-DOGE buy, then a 100-DOGE market sell with the given
    purpose; no step after the sell (it fills close_ts_offset_us after 23:59:50)."""

    start = day_start(date)
    close_ts = start + analysis.CLOSE_US + close_ts_offset_us
    times = [start + 100_000, start + 1_100_000, start + analysis.CLOSE_US - 2 * US]
    if close_ts_offset_us < 0:  # a §7 close inside the day: the last step follows it, flat
        times = times[:2] + [close_ts + 400_000, start + analysis.CLOSE_US - 2 * US]
    cash_after_buy = -8.0
    steps, previous = [], None
    for ts in times:
        q = 1 if times[0] < ts and (close_ts_offset_us >= 0 or ts < close_ts) else 0
        cash = cash_after_buy if ts > times[0] else 0.0
        if close_ts_offset_us < 0 and ts > close_ts:
            cash = cash_after_buy + 7.999
        mid = 0.08001 if ts == times[-1] else 0.08
        steps.append(step(ts, mid, q, cash + 100 * q * mid, previous=previous))
        previous = ts
    fills = [fill_row(start + 500_000, "buy", "0.08", post_ts=start + 100_000, mid_at_post=0.08, mid_at_fill=0.08,
                      fee="0.1", fee_asset="DOGE", order_id="buy://a"),
             fill_row(close_ts, "sell", "0.07999000000000000117", post_ts=close_ts - 5 * US, mid_at_post=0.08,
                      mid_at_fill=0.08, close=1, fee="0.007999", fee_asset="USDT", order_id="sell://close")]
    orders = [{"ts_us": start + 100_000, "utc": "", "event": "submit", "order_id": "buy://a", "side": "buy",
               "order_type": "limit", "price": "0.08", "amount": "100", "mid_at_post": 0.08, "reason": "quote"},
              market_submit(close_ts - 5 * US, "sell://close", "sell", "100", close_purpose)]
    write_day(root, "hb_b0", date, steps, fills=fills, orders=orders)


def test_p_post_is_w_after_the_last_fill_when_no_step_follows_the_close(tmp_path):
    consistent_day(tmp_path, DEV_DAY, "day_close", 5 * US)  # the close fills at 23:59:55; the last step is 23:59:48
    qc = analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path, "hb_b0", DEV_DAY)), 0.095, 41_000.0)["qc"]
    assert qc["last_step_after_close_fills"] is False and qc["flat_at_day_end"] is True
    assert qc["P_post_last_step_usdt"] == pytest.approx(0.001, abs=1e-12)  # W at 23:59:48, before the close
    expected = float(-(Decimal("0.08") * 100) + Decimal("0.07999000000000000117") * 100)
    assert qc["P_post_usdt"] == expected  # the cash after the close
    assert qc["C_exact"] == "7.99900" and qc["close_fills"] == 1 and qc["s7_close_fills"] == 0
    assert qc["f_star_post_pct"] == pytest.approx(expected / (8.0 + 7.999) * 100)
    assert qc["s7_close_in_C"] is False and qc["C_day_close_usdt"] == pytest.approx(7.999)


def test_a_mid_day_s7_close_is_part_of_c_and_flags_the_day(tmp_path):
    """The author's reading (WO-HB3b): a §7 close inside the day joins C in P_post - f (F + C) and f*_post; F stays
    the limit fills, so P_pre and f*_pre leave it out; s7_close_in_C flags the bot-day."""

    consistent_day(tmp_path, DEV_DAY, "halt", -3600 * US)  # a halt close at 22:59:50, flat afterwards
    qc = analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path, "hb_b0", DEV_DAY)), 0.095, 41_000.0)["qc"]
    assert qc["C_exact"] == "7.99900" and qc["C_day_close_usdt"] == 0.0 and qc["s7_close_in_C"] is True
    assert qc["close_fills"] == 0 and qc["close_side"] == "none"  # the day's close: none (flat after the halt)
    assert (qc["s7_close_fills"], qc["s7_close_purposes"], qc["close_fills_unknown_purpose"]) == (1, "halt", 0)
    assert qc["s7_close_notional_usdt"] == pytest.approx(7.999)
    assert qc["F_exact"] == "8.00" and qc["f_star_pre_pct"] == pytest.approx(qc["P_pre_usdt"] / 8.0 * 100)
    assert qc["f_star_post_pct"] == pytest.approx(qc["P_post_usdt"] / (8.0 + 7.999) * 100)
    assert qc["fee_curve"][4]["P_post_net_usdt"] == pytest.approx(qc["P_post_usdt"] - 0.001 * (8.0 + 7.999))
    assert qc["fee_curve"][4]["P_pre_net_usdt"] == pytest.approx(qc["P_pre_usdt"] - 0.001 * 8.0)
    assert qc["paper_fees_s7_close_usdt"] == pytest.approx(0.007999) and qc["paper_fees_close_usdt"] == 0.0
    # its cost: the half-spread against the mid at its submission (0.08 - 0.07999) x 100, plus the 0.007999 fee
    assert qc["s7_close_cost_usdt"] == pytest.approx(0.001 + 0.007999, abs=1e-12)


def test_the_fill_classes_on_hand_computed_fills(tmp_path):
    """One limit fill per class of the at-price method, against a constant mid of 0.08200: a bid with a taker sell
    through it (through); a bid of 0.08198 with a taker sell at its price, which float(0.08198) puts below it, so
    through for the engine (at_price_engine_through); a bid of 0.08013 with a taker sell at its price, which
    float(0.08013) puts above it (at_price_not_engine_through); an ask with no trade (no_trade)."""

    assert Decimal(0.08198) < Decimal("0.08198") and Decimal(0.08013) > Decimal("0.08013")
    start = day_start(DEV_DAY)
    times = [start + k * US for k in range(201)]
    steps, previous = [], None
    for ts in times:
        steps.append(step(ts, 0.082, 0, 0.0, previous=previous, bid=0.08198, ask=0.08202))
        previous = ts
    # (fill second, side, price, mid at post, mid at fill): delta_post 2, 2, 3, 2 and delta_fill -1/2, +1/2, +1/2, -1/2
    specs = [(10, "buy", "0.08198", 0.082, 0.081975), (20, "buy", "0.08198", 0.082, 0.081985),
             (30, "buy", "0.08013", 0.08016, 0.080135), (40, "sell", "0.08202", 0.082, 0.082025)]
    fills = [fill_row(start + second * US + 500_000, side, price, post_ts=start + second * US, mid_at_post=post,
                      mid_at_fill=at_fill, order_id=f"{side}://{second}") for second, side, price, post, at_fill in specs]
    trades = [{"ts_us": start + second * US + 200_000, "utc": "", "exchange_ts": "1", "trade_id": f"t{second}",
               "price": price, "amount": 10.0, "side": "sell"}
              for second, price in ((10, 0.08197), (20, 0.08198), (30, 0.08013))]
    write_day(tmp_path, "hb_b0", DEV_DAY, steps, fills=fills, trades=trades)
    result = analysis.analyse_day(analysis.read_day(analysis.verify_day(tmp_path, "hb_b0", DEV_DAY)), 0.095, 41_000.0)
    rows = {row["class"]: row for row in result["fill_classes"]}
    assert list(rows) == ["through", "at_price_engine_through", "at_price_not_engine_through", "no_trade", "all",
                          "through + no_trade"]
    expected = {  # class: (delta_post, delta_fill, m(h) for every h, against the constant mid 0.082)
        "through": (2.0, -0.5, 2.0), "at_price_engine_through": (2.0, 0.5, 2.0),
        "at_price_not_engine_through": (3.0, 0.5, 187.0), "no_trade": (2.0, -0.5, 2.0),
    }
    for name, (post, at_fill, markout) in expected.items():
        row = rows[name]
        assert (row["fills"], row["share"], row["fills_over_E"]) == (1, 0.25, None), name
        assert row["mean_delta_post_ticks"] == pytest.approx(post, abs=1e-9), name
        assert row["mean_delta_fill_ticks"] == pytest.approx(at_fill, abs=1e-9), name
        for horizon in (1, 10, 60):
            assert row[f"mean_m_{horizon}s"] == pytest.approx(markout, abs=1e-9), (name, horizon)
            assert row[f"R_{horizon}s"] == pytest.approx(markout / post, abs=1e-9), (name, horizon)
            assert row[f"dropped_{horizon}s"] == 0
    everything, sensitivity = rows["all"], rows["through + no_trade"]
    assert everything["fills"] == 4 and everything["mean_delta_post_ticks"] == pytest.approx(2.25, abs=1e-9)
    assert everything["mean_delta_fill_ticks"] == pytest.approx(0.0, abs=1e-9)
    assert everything["R_10s"] == pytest.approx((2 + 2 + 187 + 2) / 9, abs=1e-9)
    assert everything["mean_m_60s"] == pytest.approx(result["qb"]["mean_m_60s"])  # all = the Q-B numbers
    assert everything["fills_over_E"] == pytest.approx(result["qb"]["fills_over_E"])
    assert sensitivity["fills"] == 2 and sensitivity["mean_delta_fill_ticks"] == pytest.approx(-0.5, abs=1e-9)
    assert sensitivity["fills_over_E"] == pytest.approx(2 / result["qb"]["E"])
    assert result["exploratory"]["at_price_trade_fills"] == 2


def test_the_pnl_split_on_a_hand_computed_day(tmp_path):
    """WO-HB3d: P_pre = fill edge + inventory on a day logged the ledger's way (W = cash + 100 q S at each step's mid).
    A buy before the first step enters step 0, at step 0's mid (its mid_at_fill differs); a sell at step 2's own ts
    enters step 2, the first step with ts >= its ts, whose wealth includes it (with ">" it would take step 3's mid and
    the fill edge would be 0.013); a §7 market sell inside the window is the third part; a buy enters the last window
    step; a limit buy after the last window step and the 23:59:50 close are left out, and the path after the window
    moves (W 0.019 and 0.015 there, inventory terms 0.004 and 0.004). The checks then fail on a perturbed W or q."""

    start = day_start(DEV_DAY)
    close = start + analysis.CLOSE_US
    path = [  # (ts, mid, q and cash after the fills the step includes)
        (start + 1 * US, 0.08, 1, "-7.999"),  # buy 100 @ 0.07999 at 0.5 s: edge 100 x 0.00001 = 0.001
        (start + 2 * US, 0.08002, 1, "-7.999"),  # inventory 100 x 1 x 0.00002 = 0.002
        (start + 3 * US, 0.08003, 0, "0.005"),  # inventory 0.001; sell 100 @ 0.08004 at 3 s: edge 0.001
        (start + 4 * US, 0.07999, 2, "-15.991"),  # buy 200 @ 0.07998 at 3.5 s: edge 200 x 0.00001 = 0.002
        (start + 5 * US, 0.07997, 0, "0.005"),  # inventory 100 x 2 x -0.00002 = -0.004; §7 sell 200 @ 0.07998: 0.002
        (close - US, 0.08001, 1, "-7.991"),  # buy 100 @ 0.07996 at 23:59:48.5: edge 0.005; P_pre = W = 0.010
        (close + US, 0.08005, 2, "-15.991"),  # a limit buy 100 @ 0.08 at 23:59:49.5, after the window
        (close + 6 * US, 0.08007, 0, "0.015"),  # the 23:59:50 close: sell 200 @ 0.08003 at 23:59:55
    ]
    steps, previous = [], None
    for ts, mid, q, cash in path:
        wealth = float(Decimal(cash)) + float(100 * q) * mid  # Ledger.mark
        steps.append(step(ts, mid, q, wealth, previous=previous, phase="trade" if ts < close else "close"))
        previous = ts
    fills = [fill_row(start + 500_000, "buy", "0.07999", post_ts=start + 100_000, mid_at_post=0.08,
                      mid_at_fill=0.08001),
             fill_row(start + 3 * US, "sell", "0.08004", post_ts=start + 2 * US, mid_at_post=0.08002,
                      mid_at_fill=0.08003),
             fill_row(start + 3_500_000, "buy", "0.07998", post_ts=start + 3 * US, mid_at_post=0.08003,
                      mid_at_fill=0.07999, amount="200"),
             fill_row(start + 4_500_000, "sell", "0.07998", post_ts=start + 4_400_000, mid_at_post=0.07999,
                      mid_at_fill=0.07998, close=1, amount="200", order_id="sell://halt"),
             fill_row(close - 1_500_000, "buy", "0.07996", post_ts=close - 3 * US, mid_at_post=0.07998,
                      mid_at_fill=0.07998),
             fill_row(close - 500_000, "buy", "0.08", post_ts=close - 2 * US, mid_at_post=0.08001, mid_at_fill=0.08001),
             fill_row(close + 5 * US, "sell", "0.08003", post_ts=close + 100_000, mid_at_post=0.08005,
                      mid_at_fill=0.08005, close=1, amount="200", order_id="sell://close")]
    orders = [market_submit(start + 4_400_000, "sell://halt", "sell", "200", "halt"),
              market_submit(close + 100_000, "sell://close", "sell", "200", "day_close")]
    write_day(tmp_path, "hb_b0", DEV_DAY, steps, fills=fills, orders=orders)
    day = analysis.read_day(analysis.verify_day(tmp_path, "hb_b0", DEV_DAY))
    result = analysis.analyse_day(day, 0.095, 41_000.0)
    split = result["pnl_split"]
    assert split["P_pre_usdt"] == result["qa"]["P_pre_usdt"] == pytest.approx(0.010, abs=1e-12)
    expected = {"fill_edge_usdt": 0.009, "inventory_usdt": -0.001, "market_fill_edge_usdt": 0.002,
                "fill_edge_buys_usdt": 0.008, "fill_edge_sells_usdt": 0.001,
                "mean_fill_edge_ticks": 0.009 / (500 * TICK)}  # 500 DOGE of limit fills in the window: 1.8 ticks
    for key, value in expected.items():
        assert split[key] == pytest.approx(value, abs=1e-12), key
    assert (split["limit_fills"], split["market_fills"], split["fills_at_step_ts"]) == (4, 1, 1)
    assert split["position_mismatch_steps"] == 0 and split["within_tolerance"] is True
    assert abs(split["parts_minus_P_pre_usdt"]) <= 1e-12 and split["max_abs_step_mismatch_usdt"] <= 1e-12
    # the checks fail when W or q disagrees with the fills: W at step 2 off by 0.001; q at step 1 of 2 units
    ts, mid = day.steps["ts_us"].to_numpy(), day.steps["mid"].to_numpy()
    q, wealth, last = day.steps["q"].to_numpy(), day.steps["wealth"].to_numpy(), result["qa"]["window_steps"] - 1
    bad_wealth = wealth.copy()
    bad_wealth[2] += 0.001
    off = analysis.pnl_split(ts, mid, q, bad_wealth, last, day.fills)
    assert off["max_abs_step_mismatch_usdt"] == pytest.approx(0.001, abs=1e-12) and off["within_tolerance"] is False
    bad_q = q.copy()
    bad_q[1] = 2
    off = analysis.pnl_split(ts, mid, bad_q, wealth, last, day.fills)
    assert off["position_mismatch_steps"] == 1 and off["within_tolerance"] is False
