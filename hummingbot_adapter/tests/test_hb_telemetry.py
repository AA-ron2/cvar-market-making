"""telemetry.py (WO-HB2 deliverable 2): the Tardis book_snapshot_25 layout and number formats; the frozen-loader
round trip of a bot's snapshot file (S, dt and both market features bitwise, and the frozen FormulaASActionWrapper
on that file against FeatureEngine, as smoke check d), also after a restart that truncates the files; the steps
file's exact floats; DayLog's header, buffers, sizes and restart truncation; the day's gzip files and SHA256SUMS;
and the background Finalizer. No data needed, except the Jan 7 header check (HB_ARTIFACTS)."""

from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import warnings
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hummingbot_adapter import telemetry
from hummingbot_adapter.execution import DayClock, dt_seconds
from hummingbot_adapter.features import FeatureEngine, row_columns
from hummingbot_adapter.telemetry import (
    FILL_HEADER,
    HEALTH_HEADER,
    KINDS,
    ORDER_HEADER,
    SNAPSHOT_HEADER,
    SNAPSHOT_LEVELS,
    STEP_TAIL,
    SUMS_NAME,
    TRADE_HEADER,
    DayLog,
    Finalizer,
    finalize_day,
    fmt_book,
    fmt_float,
    fmt_value,
    headers,
    snapshot_fields,
    step_header,
    utc_iso,
)
from hummingbot_adapter.tests.support import REPO_ROOT, RecordingAgent
from cvar_mm.gym.data_loader import load_single_day_with_features, read_market_data_csv
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES
from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
from cvar_mm.gym.notebook_support import build_formula_replay_env
from cvar_mm.rewards import PnLReward


def book_row(book, depth: int) -> dict[str, float]:
    """The FeatureEngine row of a connector book (bids, asks), as bot_core.book_row builds it (that function is
    checked against the Jan 7 replay rows in test_hb_loop_parity.py)."""

    bids, asks = book
    row = {"asks[0].price": asks[0][0], "bids[0].price": bids[0][0]}
    for level in range(depth):
        row[f"bids[{level}].amount"] = bids[level][1]
        row[f"asks[{level}].amount"] = asks[level][1]
    return row


TICK = 1e-5
T_DAY = 86_400.0  # clarification c: T = 86,400 s
IMBALANCE_DEPTH = 5
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DATE = "2026-10-21"
DAY_START_US = (datetime(2026, 10, 21, tzinfo=timezone.utc) - EPOCH) // timedelta(microseconds=1)
JAN7_FILE = "binance_book_snapshot_25_2025-01-07_DOGEUSDT.csv"
CALIBRATION_FILE = REPO_ROOT / "hummingbot_adapter" / "config" / "hb_calibration.json"
CALIBRATION = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
MARKET = slice(2, 7)  # log_mid_rel, rolling_vol_ticks, momentum_ticks, spread_ticks, lob_imbalance
REFUSALS = (ValueError, RuntimeError, FileExistsError)


# --- helpers -----------------------------------------------------------------------------------------------------


def exchange_price(ticks: int) -> float:
    """The double nearest ticks * 0.00001, as float() of Binance's price text gives it."""

    return float(np.round(int(ticks) * TICK, 5))


def _bits(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values)
    if array.dtype == np.float64:
        return array.view(np.uint64)
    if array.dtype == np.float32:
        return array.view(np.uint32)
    raise AssertionError(f"expected float32 or float64, got {array.dtype}")


def assert_bitwise(actual, expected) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape)
    differ = np.flatnonzero(_bits(actual).ravel() != _bits(expected).ravel())
    assert differ.size == 0, f"{differ.size} values differ; first flat indices {differ[:5].tolist()}"


@dataclasses.dataclass(frozen=True)
class SyntheticDay:
    ts: list[int]  # each step's ts_us (clarification c): non-decreasing, with repeats (dt = 0)
    local: list[int]  # the wall-clock receipt time in µs (local_timestamp)
    books: list[tuple]  # (bids, asks), best level first, as BotCore.on_tick receives the connector's book


def synthetic_day(seed: int, rows: int, *, min_levels: int = SNAPSHOT_LEVELS) -> SyntheticDay:
    """Exchange-like books: prices are the doubles nearest k * 0.00001, amounts whole DOGE (as floats, as the
    connector gives them); the best bid starts at 0.08 USDT (the 2026 level) and walks by ticks; spreads of 1-4
    ticks (an even spread puts the mid on the grid); deeper levels 1-2 ticks apart; steps about 0.1 s apart
    with 8 % repeated ts_us. With min_levels < 25 each side has a random depth in [min_levels, 25]."""

    rng = np.random.default_rng(seed)
    gaps = rng.exponential(100_000.0, rows).astype(np.int64)
    gaps[rng.random(rows) < 0.08] = 0
    gaps[0] = 1_234_567  # the day's first step, 1.23 s after 00:00 UTC
    ts = (DAY_START_US + np.cumsum(gaps)).tolist()
    local = [stamp + int(lag) for stamp, lag in zip(ts, rng.integers(50, 3_000, rows))]
    best_bid = 8_000 + np.cumsum(rng.choice([-1, 0, 1], size=rows, p=[0.3, 0.4, 0.3]))
    spread = rng.choice([1, 2, 3, 4], size=rows, p=[0.7, 0.15, 0.1, 0.05])
    books = []
    for k in range(rows):
        depths = [
            SNAPSHOT_LEVELS if min_levels == SNAPSHOT_LEVELS else int(rng.integers(min_levels, SNAPSHOT_LEVELS + 1))
            for _ in range(2)
        ]
        ask_ticks = best_bid[k] + spread[k] + np.concatenate([[0], np.cumsum(rng.integers(1, 3, depths[0] - 1))])
        bid_ticks = best_bid[k] - np.concatenate([[0], np.cumsum(rng.integers(1, 3, depths[1] - 1))])
        asks = tuple((exchange_price(tick), float(rng.integers(1, 3_000_000))) for tick in ask_ticks)
        bids = tuple((exchange_price(tick), float(rng.integers(1, 3_000_000))) for tick in bid_ticks)
        books.append((bids, asks))
    return SyntheticDay(ts, local, books)


def add_snapshot(log: DayLog, day: SyntheticDay, k: int) -> None:
    """One snapshot row, as BotCore._write_step writes it."""

    bids, asks = day.books[k]
    log.add("snapshots", snapshot_fields(day.ts[k], day.local[k], asks, bids))


def write_snapshots(log: DayLog, day: SyntheticDay, rows: Sequence[int], flush_every: int = 10) -> dict[str, int]:
    """One row per step, with the buffers flushed about once per second (every 10 steps of about 0.1 s)."""

    for count, k in enumerate(rows, 1):
        add_snapshot(log, day, k)
        if count % flush_every == 0:
            log.flush()
    return log.flush()


def bot_step(engine: FeatureEngine, day: SyntheticDay, k: int, previous_us: int | None) -> np.ndarray:
    """The bot's observation at step k (BotCore._step): the connector book as a FeatureEngine row (book_row),
    dt from the steps' ts_us (execution.dt_seconds), t_elapsed since 00:00 UTC and T = 86,400 s."""

    row = book_row(day.books[k], IMBALANCE_DEPTH)
    t_elapsed = DayClock.t_elapsed(day.ts[k], DAY_START_US)
    if previous_us is None:
        return engine.reset(row, t_elapsed, T_DAY, 0)
    return engine.update(row, dt_seconds(previous_us, day.ts[k]), t_elapsed, T_DAY, 0)


def bot_observations(day: SyntheticDay) -> tuple[np.ndarray, np.ndarray]:
    """The bot's observations and float64 mids (FeatureEngine's last_mid) over the whole day."""

    engine, observations, mids, previous = FeatureEngine(), [], [], None
    for k in range(len(day.books)):
        observations.append(bot_step(engine, day, k, previous))
        mids.append(engine.get_state()["last_mid"])
        previous = day.ts[k]
    return np.array(observations), np.array(mids)


def in_memory(day: SyntheticDay, rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """S = (ask0 + bid0) / 2 and dt via execution.dt_seconds for the steps in `rows` (the first dt is 0, as the
    frozen loader sets it)."""

    mids = np.array([(day.books[k][1][0][0] + day.books[k][0][0][0]) / 2 for k in rows], dtype=np.float64)
    stamps = [day.ts[k] for k in rows]
    dts = np.array([0.0] + [dt_seconds(a, b) for a, b in zip(stamps, stamps[1:])], dtype=np.float64)
    return mids, dts


def load_bot_file(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # the loader warns (and re-sorts) if a timestamp goes back
        return load_single_day_with_features(str(path), tick_size=TICK, imbalance_depth=IMBALANCE_DEPTH)


def replay_raw_observations(S: np.ndarray, dt: np.ndarray, features: dict, seed: int = 11) -> np.ndarray:
    """The frozen wrapper's raw observation at every row (RecordingAgent), with the committed calibration
    (hb_calibration.json) and random actions."""

    config = dataclasses.replace(
        ReplayExperimentConfig(), A=float(CALIBRATION["A"]), kappa=float(CALIBRATION["kappa"])
    )
    env = build_formula_replay_env(
        S, dt, config, sigma=float(CALIBRATION["sigma"]), market_features=features, reward_fn=PnLReward(),
        num_trajectories=1,
    )
    action_rng = np.random.default_rng(seed)
    agent = RecordingAgent(env, len(S), lambda state: action_rng.uniform(-1.0, 1.0, size=(1, 2)))
    generate_trajectory_stats(env, agent, seed=42, fixed_clock_seconds=config.fixed_clock_seconds)
    agent.finish()
    return agent.raw_obs


def add_one_step(log: DayLog, k: int) -> None:
    """One step's rows in all six files of a B0 DayLog, with the field types BotCore writes."""

    ts = DAY_START_US + 3_600_000_000 + 100_000 * k
    bid, ask = exchange_price(8_000 - k % 3), exchange_price(8_001 + k % 2)
    mid = (bid + ask) / 2
    t_elapsed = DayClock.t_elapsed(ts, DAY_START_US)
    log.add("snapshots", snapshot_fields(ts, ts + 900, [(ask, 1_000.0 + k)], [(bid, 2_000.0)]))
    step = [ts, utc_iso(ts), "trade", 0.1, t_elapsed, T_DAY - t_elapsed, mid] + [None] * len(STEP_TAIL)
    assert len(step) == len(step_header("B0"))
    log.add("steps", step)
    order_id = f"buy-DOGE-USDT-{k}"
    log.add("orders", [ts, utc_iso(ts), "submit", order_id, "buy", "limit", Decimal("0.08000"), Decimal(100), mid,
                       "quote µ"])  # a non-ASCII field: sizes are bytes
    log.add("fills", [ts, utc_iso(ts), order_id, "buy", "limit", Decimal("0.08000"), Decimal(100), mid, 0.5,
                      ts - 2_000_000, mid, 0.5, Decimal("0.100"), "DOGE", False, str(1_000 + k)])
    log.add("trades", [ts, utc_iso(ts), 1_792_544_400.5 + k, 4_000_000 + k, bid, 1_500.0, "sell"])
    log.add_health(ts, "heartbeat", rss_mb=512.25, q="0", quoting=True)


def header_bytes(kind: str, model: str = "B0") -> bytes:
    return (",".join(headers(model)[kind]) + "\n").encode("utf-8")


def write_day(bot_dir: Path, day_text: str, steps: int = 6) -> dict[str, bytes]:
    """A closed day's six plain files; returns {file name: bytes}."""

    log = DayLog(bot_dir / day_text, day_text, headers("B0"))
    for k in range(steps):
        add_one_step(log, k)
    log.close()
    return {path.name: path.read_bytes() for path in log.paths.values()}


def verify_sums(folder: Path, lines: list[str]) -> list[str]:
    """Check coreutils lines '<sha256>  <path>' against the files under folder; returns the paths."""

    names = []
    for line in lines:
        assert re.fullmatch(r"[0-9a-f]{64}  [0-9A-Za-z_./-]+", line), line
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
        names.append(name)
    return names


# --- the snapshot layout and number formats --------------------------------------------------------------------


def test_snapshot_header_is_the_tardis_book_snapshot_25_layout():
    expected = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for level in range(25):
        expected += [f"asks[{level}].price", f"asks[{level}].amount", f"bids[{level}].price", f"bids[{level}].amount"]
    assert list(SNAPSHOT_HEADER) == expected
    assert len(SNAPSHOT_HEADER) == len(set(SNAPSHOT_HEADER)) == 104
    assert headers("B0")["snapshots"] == headers("B3")["snapshots"] == SNAPSHOT_HEADER
    assert set(row_columns(IMBALANCE_DEPTH)) <= set(SNAPSHOT_HEADER)  # what FeatureEngine and the loader read


def test_snapshot_header_equals_the_header_of_the_jan7_tardis_file(hb_artifacts):
    with open(hb_artifacts / JAN7_FILE, "rb") as handle:
        first_line = handle.readline()  # the header line only: the file is 576 MB
    assert first_line == (",".join(SNAPSHOT_HEADER) + "\n").encode("ascii")


def test_fmt_book_prints_whole_amounts_without_a_point_and_prices_as_the_shortest_decimal():
    for value, text in (
        (51051.0, "51051"), (55438, "55438"), (np.float64(10854.0), "10854"), (0.0, "0"), (1.0, "1"),
        (float(10**15), "1000000000000000"), (0.38782, "0.38782"), (0.3878, "0.3878"), (0.39, "0.39"),
        (0.09456, "0.09456"), (0.0946, "0.0946"), (0.0001, "0.0001"), (12.5, "12.5"),
    ):
        assert fmt_book(value) == text
    rng = np.random.default_rng(17)
    for ticks in rng.integers(10, 10_000_000, 20_000).tolist():  # 0.0001 to 100 USDT
        price = exchange_price(ticks)
        text = format(Decimal(ticks).scaleb(-5).normalize(), "f")  # the decimal text without trailing zeros
        assert fmt_book(price) == text, ticks
        assert float(text) == price
    for amount in rng.integers(1, 10**12, 5_000).tolist():
        assert fmt_book(float(amount)) == str(amount)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), np.float32("nan"), -1.0, -0.00001, -55438.0, 1e-05, 9e-05, 1e16,
     1.5e300],
)
def test_fmt_book_refuses_non_finite_negative_and_exponent_values(value):
    with pytest.raises(ValueError):
        fmt_book(value)


def test_snapshot_fields_reproduce_the_first_jan7_row_from_its_parsed_values(inputs):
    with open(inputs["jan7"], encoding="ascii", newline="") as handle:  # checked against SHA256SUMS_inputs.txt
        handle.readline()  # the header
        text = handle.readline().split(",")[:14]  # the first data row, fields 1-14
    asks = [(float(text[4]), float(text[5])), (float(text[8]), float(text[9])), (float(text[12]), float(text[13]))]
    bids = [(float(text[6]), float(text[7])), (float(text[10]), float(text[11]))]
    assert snapshot_fields(int(text[2]), int(text[3]), asks, bids)[:14] == text


def test_snapshot_fields_put_the_best_level_first_and_leave_missing_levels_empty():
    asks = [(0.08003, 1_500.0), (0.08004, 20_000.0), (0.08006, 7.0)]
    bids = [(0.08001, 900.0)]
    fields = snapshot_fields(DAY_START_US + 5, DAY_START_US + 1_005, asks, bids)
    assert len(fields) == len(SNAPSHOT_HEADER)
    record = dict(zip(SNAPSHOT_HEADER, fields))
    assert (record["exchange"], record["symbol"]) == ("binance", "DOGEUSDT")
    assert (record["timestamp"], record["local_timestamp"]) == (str(DAY_START_US + 5), str(DAY_START_US + 1_005))
    assert [(record[f"asks[{i}].price"], record[f"asks[{i}].amount"]) for i in range(3)] == [
        ("0.08003", "1500"), ("0.08004", "20000"), ("0.08006", "7")
    ]
    assert (record["bids[0].price"], record["bids[0].amount"]) == ("0.08001", "900")
    empty = {name for name, value in record.items() if value == ""}
    assert empty == (
        {f"asks[{i}].{field}" for i in range(3, 25) for field in ("price", "amount")}
        | {f"bids[{i}].{field}" for i in range(1, 25) for field in ("price", "amount")}
    )
    assert next(csv.reader([",".join(fields)])) == fields  # one CSV line that needs no quoting
    assert snapshot_fields(1, 2, [], [])[4:] == [""] * 100
    for bad_asks, bad_bids in (
        ([(0.08, 1.0)] * 26, []), ([], [(0.08, 1.0)] * 26), ([(float("nan"), 1.0)], []), ([(0.08, -1.0)], []),
    ):
        with pytest.raises(ValueError):
            snapshot_fields(1, 2, bad_asks, bad_bids)


def test_utc_iso_is_exact_to_the_microsecond():
    assert utc_iso(0) == "1970-01-01T00:00:00.000000Z"
    assert utc_iso(1_736_208_002_624_124) == "2025-01-07T00:00:02.624124Z"
    assert utc_iso(np.int64(1_736_208_002_624_124)) == "2025-01-07T00:00:02.624124Z"
    assert utc_iso(DAY_START_US) == "2026-10-21T00:00:00.000000Z"
    assert utc_iso(DAY_START_US - 1) == "2026-10-20T23:59:59.999999Z"
    assert utc_iso(DAY_START_US + 86_390_000_000 - 1) == "2026-10-21T23:59:49.999999Z"
    rng = np.random.default_rng(23)
    for ts in rng.integers(0, 4_102_444_800_000_000, 20_000).tolist():  # to 2100, integer arithmetic only
        days, rest = divmod(ts, 86_400_000_000)
        seconds, micro = divmod(rest, 1_000_000)
        hours, rest = divmod(seconds, 3_600)
        minutes, seconds = divmod(rest, 60)
        day_text = (date(1970, 1, 1) + timedelta(days=days)).isoformat()
        assert utc_iso(ts) == f"{day_text}T{hours:02d}:{minutes:02d}:{seconds:02d}.{micro:06d}Z", ts


def test_step_header_b3_adds_the_observations_their_z_scores_and_the_raw_action():
    b0, b3 = step_header("B0"), step_header("B3")
    observations = list(FORMULA_FEATURE_NAMES)
    z_scores = [f"z_{name}" for name in observations]
    action = ["action_gamma", "action_skew"]
    assert len(observations) == 7
    assert [name for name in b3 if name not in b0] == observations + z_scores + action
    assert [name for name in b3 if name in b0] == list(b0)
    assert not set(observations + z_scores + action) & set(b0)
    assert len(set(b3)) == len(b3) == len(b0) + 16
    required = {
        "ts_us", "utc", "mid", "gamma", "skew", "bid_target", "ask_target", "bid_quote", "ask_quote", "bid_rounded",
        "ask_rounded", "resting_bid", "resting_ask", "q", "cash", "wealth", "peak", "drawdown", "fees_doge",
        "fees_usdt", "stale_paused", "clock_paused", "drawdown_paused", "halted",
    }
    assert required <= set(b0)
    assert headers("B3")["steps"] == b3 and headers("B0")["steps"] == b0
    for model in ("B1", "B2", "b3", ""):
        with pytest.raises(ValueError):
            step_header(model)


def test_order_fill_trade_and_health_headers_hold_the_work_order_columns():
    assert set(headers("B3")) == set(KINDS) == {"snapshots", "steps", "orders", "fills", "trades", "health"}
    assert {"ts_us", "event", "order_id", "side", "price", "amount", "mid_at_post"} <= set(ORDER_HEADER)
    assert {"ts_us", "side", "price", "amount", "mid_at_fill", "delta_fill_ticks", "post_ts_us", "mid_at_post",
            "delta_post_ticks", "fee", "fee_asset"} <= set(FILL_HEADER)
    assert {"ts_us", "price", "amount", "side"} <= set(TRADE_HEADER)
    assert HEALTH_HEADER == ("ts_us", "utc", "event", "detail")
    for kind, header in headers("B3").items():
        assert len(set(header)) == len(header), kind
        assert not any(character in name for name in header for character in ',"\n\r'), kind


def test_fmt_float_widens_float32_exactly_and_refuses_non_finite_values():
    assert fmt_float(None) == ""
    assert fmt_float(np.float32(0.1)) == "0.10000000149011612" == repr(float(np.float32(0.1)))
    rng = np.random.default_rng(29)
    values32 = (rng.standard_normal(10_000) * 10.0 ** rng.integers(-8, 8, 10_000)).astype(np.float32)
    for value in values32:
        text = fmt_float(value)
        assert text == repr(float(value))
        assert float(text) == float(value) and np.float32(float(text)).view(np.uint32) == value.view(np.uint32)
    for value in rng.standard_normal(5_000) * 10.0 ** rng.integers(-12, 12, 5_000):
        assert float(fmt_float(float(value))) == value
    for bad in (float("nan"), float("inf"), float("-inf"), np.float64("inf"), np.float32("nan"), np.float32("-inf")):
        with pytest.raises(ValueError):
            fmt_float(bad)


@pytest.mark.parametrize("text", ["a,b", 'say "x"', "line\nbreak", "carriage\rreturn", "trailing,", ("a", "b")])
def test_fmt_value_refuses_text_that_would_need_csv_quoting(text):
    with pytest.raises(ValueError):
        fmt_value(text)


def test_fmt_value_writes_none_bools_ints_decimals_and_floats():
    assert fmt_value(None) == ""
    assert (fmt_value(True), fmt_value(False)) == ("1", "0")
    assert (fmt_value(42), fmt_value(np.int64(-7)), fmt_value(Decimal("0.100"))) == ("42", "-7", "0.1")
    # Hummingbot's paper events quantize prices to 1e-10 and amounts to 1e-7: Decimals print normalized
    assert [fmt_value(Decimal(text)) for text in ("0.0797600000", "100.0000000", "0E-7", "1E+2", "-2.0000000")] \
        == ["0.07976", "100", "0", "100", "-2"]
    assert fmt_value("quote_rounded") == "quote_rounded"
    assert (fmt_value(0.1), fmt_value(1 / 3), fmt_value(np.float64(0.080005))) == ("0.1", repr(1 / 3), "0.080005")
    for bad in (float("nan"), float("inf"), np.float64("-inf")):
        with pytest.raises(ValueError):
            fmt_value(bad)


# --- the frozen-loader round trip (smoke check d, offline) --------------------------------------------------------


@pytest.fixture(scope="module")
def bot_day(tmp_path_factory):
    """A 2,500-step synthetic day written through a B3 DayLog as the bot writes it, and the bot's in-memory
    observations and mids for it."""

    day = synthetic_day(seed=2026, rows=2_500)
    folder = tmp_path_factory.mktemp("hb_data") / "hb_b3" / DATE
    log = DayLog(folder, DATE, headers("B3"))
    sizes = write_snapshots(log, day, range(len(day.books)))
    log.close()
    observations, mids = bot_observations(day)
    return {"day": day, "path": log.paths["snapshots"], "sizes": sizes, "obs": observations, "mids": mids}


def test_bot_snapshot_file_reloads_bitwise_through_the_frozen_loader(bot_day):
    day, path, observations = bot_day["day"], bot_day["path"], bot_day["obs"]
    rows = range(len(day.books))
    mids, expected_dt = in_memory(day, rows)
    spreads = np.array([round((asks[0][0] - bids[0][0]) / TICK) for bids, asks in day.books])
    assert np.count_nonzero(expected_dt[1:] == 0.0) > 100, "the day must repeat ts_us (dt = 0)"
    assert np.any(spreads % 2 == 0) and np.any(spreads % 2 == 1)
    assert path.read_bytes().split(b"\n", 1)[0] + b"\n" == (",".join(SNAPSHOT_HEADER) + "\n").encode("ascii")
    assert path.stat().st_size == bot_day["sizes"]["snapshots"]

    S, dt, index, features = load_bot_file(path)
    assert len(S) == len(day.books) == 2_500
    assert_bitwise(S, mids)
    assert_bitwise(bot_day["mids"], mids)  # FeatureEngine's S, the one the quotes use
    assert_bitwise(dt, expected_dt)
    assert_bitwise(features["spread_ticks"], observations[:, 5])
    assert_bitwise(features["lob_imbalance"], observations[:, 6])
    assert index.as_unit("us").asi8.tolist() == day.ts

    frame = read_market_data_csv(str(path))
    assert list(frame.columns) == [name for name in SNAPSHOT_HEADER if name != "timestamp"]
    assert set(frame["exchange"]) == {"binance"} and set(frame["symbol"]) == {"DOGEUSDT"}
    assert frame["local_timestamp"].tolist() == day.local
    for level in range(SNAPSHOT_LEVELS):
        for side, position in (("asks", 1), ("bids", 0)):
            prices = np.array([book[position][level][0] for book in day.books])
            amounts = np.array([book[position][level][1] for book in day.books])
            assert_bitwise(frame[f"{side}[{level}].price"].to_numpy(np.float64), prices)
            assert frame[f"{side}[{level}].amount"].dtype == np.int64  # whole numbers, printed without ".0"
            assert_bitwise(frame[f"{side}[{level}].amount"].to_numpy(np.float64), amounts)


def test_frozen_wrapper_on_the_bot_file_sees_the_bot_market_features(bot_day):
    assert FORMULA_FEATURE_NAMES[MARKET] == (
        "log_mid_rel", "rolling_vol_ticks", "momentum_ticks", "spread_ticks", "lob_imbalance"
    )
    S, dt, _, features = load_bot_file(bot_day["path"])
    raw_obs = replay_raw_observations(S, dt, features)
    observations = bot_day["obs"]
    assert raw_obs.dtype == observations.dtype == np.float32 and raw_obs.shape == observations.shape
    mismatches = [
        k for k in range(len(S)) if not np.array_equal(_bits(raw_obs[k, MARKET]), _bits(observations[k, MARKET]))
    ]
    assert mismatches == []
    assert np.count_nonzero(observations[:, 3]) > 2_000 and np.count_nonzero(observations[:, 4]) > 1_000


def test_steps_file_keeps_float32_observations_and_float64_values_exactly(tmp_path, bot_day):
    day, observations, mids = bot_day["day"], bot_day["obs"], bot_day["mids"]
    z_scores = ((observations - np.float32(0.25)) / np.float32(3.0)).astype(np.float32)  # stand-in float32 values
    actions = np.random.default_rng(3).uniform(-1.0, 1.0, (len(observations), 2))
    header = step_header("B3")
    log = DayLog(tmp_path / "hb_b3" / DATE, DATE, headers("B3"))
    previous = None
    for k in range(len(observations)):
        values = dict.fromkeys(header)
        t_elapsed = DayClock.t_elapsed(day.ts[k], DAY_START_US)
        values.update(ts_us=day.ts[k], utc=utc_iso(day.ts[k]), phase="trade", t_elapsed=t_elapsed,
                      tau=T_DAY - t_elapsed, mid=float(mids[k]),
                      dt=None if previous is None else dt_seconds(previous, day.ts[k]))
        values.update({name: float(value) for name, value in zip(FORMULA_FEATURE_NAMES, observations[k])})
        values.update({f"z_{name}": float(value) for name, value in zip(FORMULA_FEATURE_NAMES, z_scores[k])})
        values.update(action_gamma=float(actions[k, 0]), action_skew=float(actions[k, 1]))
        log.add("steps", [values[name] for name in header])  # as BotCore._write_step: float() of each float32
        previous = day.ts[k]
    log.close()

    with open(log.paths["steps"], encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(header) and len(rows) == len(observations) + 1
    assert all(len(row) == len(header) for row in rows)
    records = [dict(zip(header, row)) for row in rows[1:]]
    back = np.array([[float(record[name]) for name in FORMULA_FEATURE_NAMES] for record in records])
    assert_bitwise(back, observations.astype(np.float64))  # the exact widening of each float32
    assert_bitwise(back.astype(np.float32), observations)
    back_z = np.array([[float(record[f"z_{name}"]) for name in FORMULA_FEATURE_NAMES] for record in records])
    assert_bitwise(back_z.astype(np.float32), z_scores)
    assert_bitwise(np.array([[float(r["action_gamma"]), float(r["action_skew"])] for r in records]), actions)
    assert_bitwise(np.array([float(record["mid"]) for record in records]), mids)
    assert [int(record["ts_us"]) for record in records] == day.ts
    parsed = [datetime.strptime(record["utc"], "%Y-%m-%dT%H:%M:%S.%fZ") for record in records]
    assert [(moment.replace(tzinfo=timezone.utc) - EPOCH) // timedelta(microseconds=1) for moment in parsed] == day.ts
    frame = pd.read_csv(log.paths["steps"], float_precision="round_trip")
    assert_bitwise(frame[list(FORMULA_FEATURE_NAMES)].to_numpy(np.float64), observations.astype(np.float64))


def test_restart_truncates_to_the_saved_sizes_and_the_replay_matches_the_restored_bot(tmp_path):
    """PROTOCOL §5.4: the state is saved with the files' sizes; steps written after that save are lost with the
    process. After the restart the files hold exactly the steps the restored state has seen, the first step
    after the gap carries the gap as its dt, and the frozen loader and wrapper on the file see what the restored
    bot computes."""

    day = synthetic_day(seed=99, rows=900)
    saved_at, lost = 400, 37
    folder = tmp_path / "hb_b3" / DATE
    log = DayLog(folder, DATE, headers("B3"))
    engine, before, previous = FeatureEngine(), [], None
    for k in range(saved_at):
        add_snapshot(log, day, k)
        before.append(bot_step(engine, day, k, previous))
        previous = day.ts[k]
    log.add_health(previous, "heartbeat", steps=saved_at)
    saved_sizes = log.flush()
    saved_state = json.dumps(engine.get_state())  # saved with saved_sizes (BotCore._save)
    for k in range(saved_at, saved_at + lost):  # flushed after the last save; then the process is killed
        add_snapshot(log, day, k)
        bot_step(engine, day, k, previous)
        previous = day.ts[k]
    log.add_health(previous, "heartbeat", steps=saved_at + lost)
    grown = log.flush()
    log.close()

    restored = FeatureEngine()
    restored.set_state(json.loads(saved_state))
    reopened = DayLog(folder, DATE, headers("B3"), sizes=saved_sizes)
    assert reopened.truncations == [
        ("snapshots", grown["snapshots"], saved_sizes["snapshots"]), ("health", grown["health"], saved_sizes["health"])
    ]
    assert reopened.short_files == []
    after, previous = [], day.ts[saved_at - 1]  # the restored last_step_us
    for k in range(saved_at + lost, len(day.books)):
        add_snapshot(reopened, day, k)
        after.append(bot_step(restored, day, k, previous))
        previous = day.ts[k]
    reopened.close()

    kept = list(range(saved_at)) + list(range(saved_at + lost, len(day.books)))
    mids, expected_dt = in_memory(day, kept)
    observations = np.array(before + after)
    S, dt, _, features = load_bot_file(reopened.paths["snapshots"])
    assert_bitwise(S, mids)
    assert_bitwise(dt, expected_dt)
    assert dt[saved_at] == dt_seconds(day.ts[saved_at - 1], day.ts[saved_at + lost]) > 0
    assert_bitwise(features["spread_ticks"], observations[:, 5])
    assert_bitwise(features["lob_imbalance"], observations[:, 6])
    assert_bitwise(replay_raw_observations(S, dt, features)[:, MARKET], observations[:, MARKET])


def test_a_file_with_missing_deep_levels_still_loads(tmp_path):
    day = synthetic_day(seed=5, rows=400, min_levels=IMBALANCE_DEPTH)
    log = DayLog(tmp_path / "hb_b3" / DATE, DATE, headers("B3"))
    write_snapshots(log, day, range(len(day.books)))
    log.close()
    path = log.paths["snapshots"]
    depths = [(len(asks), len(bids)) for bids, asks in day.books]
    assert min(min(pair) for pair in depths) == IMBALANCE_DEPTH and any(max(pair) < 25 for pair in depths)

    lines = path.read_bytes().decode("ascii").split("\n")
    assert lines[-1] == "" and len(lines) == len(day.books) + 2
    for line, (n_asks, n_bids) in zip(lines[1:-1], depths):
        fields = line.split(",")
        assert len(fields) == 104
        for level in range(SNAPSHOT_LEVELS):
            for pair, present in ((fields[4 + 4 * level: 6 + 4 * level], level < n_asks),
                                  (fields[6 + 4 * level: 8 + 4 * level], level < n_bids)):
                if present:
                    assert "" not in pair
                else:
                    assert pair == ["", ""]  # a missing level is two empty fields

    S, dt, _, features = load_bot_file(path)
    mids, expected_dt = in_memory(day, range(len(day.books)))
    observations, _ = bot_observations(day)
    assert_bitwise(S, mids)
    assert_bitwise(dt, expected_dt)
    assert_bitwise(features["spread_ticks"], observations[:, 5])
    assert_bitwise(features["lob_imbalance"], observations[:, 6])
    frame = read_market_data_csv(str(path))
    for level in range(SNAPSHOT_LEVELS):
        for side, position in (("asks", 1), ("bids", 0)):
            present = np.array([level < len(book[position]) for book in day.books])
            for field, item in (("price", 0), ("amount", 1)):
                column = frame[f"{side}[{level}].{field}"].to_numpy(np.float64)
                assert np.all(np.isnan(column[~present]))
                expected = np.array([book[position][level][item] for book in day.books if level < len(book[position])])
                assert_bitwise(column[present], expected)


# --- DayLog ------------------------------------------------------------------------------------------------------


def test_day_log_needs_the_six_kinds(tmp_path):
    full = headers("B0")
    for wrong in ({kind: full[kind] for kind in KINDS if kind != "trades"}, {**full, "extra": ("ts_us",)}):
        with pytest.raises(ValueError):
            DayLog(tmp_path / DATE, DATE, wrong)


def test_header_is_written_once_and_a_reopened_log_appends(tmp_path):
    folder = tmp_path / "hb_b0" / DATE
    log = DayLog(folder, DATE, headers("B0"))
    assert log.paths == {kind: folder / f"{kind}_{DATE}.csv" for kind in KINDS}
    for k in range(3):
        add_one_step(log, k)
    log.close()
    log = DayLog(folder, DATE, headers("B0"))  # a restart without recorded sizes: append
    for k in range(3, 5):
        add_one_step(log, k)
    log.close()
    for kind, header in headers("B0").items():
        data = log.paths[kind].read_bytes()
        assert data.startswith(header_bytes(kind)) and data.count(header_bytes(kind)) == 1, kind
        with open(log.paths[kind], encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == list(header) and len(rows) == 6, kind
        assert all(len(row) == len(header) for row in rows), kind


def test_an_existing_file_with_another_header_is_refused_and_left_unchanged(tmp_path):
    folder = tmp_path / "hb_b0" / DATE
    log = DayLog(folder, DATE, headers("B0"))
    add_one_step(log, 0)
    log.close()
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(ValueError, match="another header"):
        DayLog(folder, DATE, headers("B3"))  # the steps file was written with B0's header
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    for path in folder.iterdir():
        path.unlink()  # no handle was left open (Windows refuses to delete an open file)


def test_rows_reach_the_disk_only_at_flush_and_flush_returns_the_byte_sizes(tmp_path):
    log = DayLog(tmp_path / "hb_b0" / DATE, DATE, headers("B0"))
    first = log.flush()
    assert first == {kind: len(header_bytes(kind)) for kind in KINDS}
    on_disk = {kind: path.read_bytes() for kind, path in log.paths.items()}
    for k in range(30):
        add_one_step(log, k)
    assert {kind: path.read_bytes() for kind, path in log.paths.items()} == on_disk  # add() only buffers
    second = log.flush()
    for kind, path in log.paths.items():
        data = path.read_bytes()
        assert second[kind] == path.stat().st_size == len(data) > first[kind], kind
        assert data.startswith(on_disk[kind]) and data.endswith(b"\n"), kind
    assert log.paths["orders"].read_bytes().count("µ".encode("utf-8")) == 30  # sizes count bytes, not characters
    add_one_step(log, 30)
    log.close()  # close() writes what is still buffered
    assert all(path.stat().st_size > second[kind] for kind, path in log.paths.items())


def test_reopening_with_the_saved_sizes_drops_rows_written_after_the_save(tmp_path):
    folder = tmp_path / "hb_b0" / DATE
    log = DayLog(folder, DATE, headers("B0"))
    for k in range(5):
        add_one_step(log, k)
    saved = log.flush()
    saved_bytes = {kind: path.read_bytes() for kind, path in log.paths.items()}
    ts = DAY_START_US + 7_200_000_000
    log.add("snapshots", snapshot_fields(ts, ts + 1, [(0.08001, 5.0)], [(0.08, 6.0)]))  # after the save
    log.add("steps", [ts, utc_iso(ts), "trade", 0.1, 7_200.0, 79_200.0, 0.080005] + [None] * len(STEP_TAIL))
    log.add_health(ts, "heartbeat", note="after the last save")
    grown = log.flush()
    log.close()

    reopened = DayLog(folder, DATE, headers("B0"), sizes=saved)
    assert reopened.truncations == [(kind, grown[kind], saved[kind]) for kind in ("snapshots", "steps", "health")]
    assert reopened.short_files == []
    assert {kind: path.read_bytes() for kind, path in reopened.paths.items()} == saved_bytes
    add_one_step(reopened, 100)
    sizes = reopened.flush()
    reopened.close()
    for kind, path in reopened.paths.items():
        data = path.read_bytes()
        assert sizes[kind] == len(data) and data.startswith(saved_bytes[kind]), kind
        assert data.count(header_bytes(kind)) == 1 and data.count(b"\n") == saved_bytes[kind].count(b"\n") + 1, kind


def test_a_file_shorter_than_its_recorded_size_is_reported_and_left_as_it_is(tmp_path):
    folder = tmp_path / "hb_b0" / DATE
    log = DayLog(folder, DATE, headers("B0"))
    for k in range(4):
        add_one_step(log, k)
    saved = log.flush()
    log.close()
    fills = log.paths["fills"]
    data = fills.read_bytes()
    cut = data.rstrip(b"\n").rfind(b"\n") + 1  # the last row was lost
    fills.write_bytes(data[:cut])
    log.paths["trades"].unlink()

    reopened = DayLog(folder, DATE, headers("B0"), sizes=saved)
    assert reopened.short_files == [("fills", cut, saved["fills"]), ("trades", 0, saved["trades"])]
    assert reopened.truncations == []
    sizes = reopened.flush()
    reopened.close()
    assert fills.read_bytes() == data[:cut] and sizes["fills"] == cut
    assert log.paths["trades"].read_bytes() == header_bytes("trades") and sizes["trades"] == len(header_bytes("trades"))
    assert sizes["orders"] == saved["orders"]


def test_health_detail_round_trips_through_csv_and_json(tmp_path):
    log = DayLog(tmp_path / "hb_b3" / DATE, DATE, headers("B3"))
    ts = DAY_START_US + 123_456_789
    detail = {
        "error": 'ValueError("bad, worse")',
        "note": 'he said "stop", then left',
        "files": [("fills", 120, 98)],
        "text": "line one\r\nline two",
        "fee": Decimal("0.100"),
        "nested": {"a,b": [1, 2.5, None, True], "path": r"C:\hb_data\hb_b3"},
        "unit": "µs",
    }
    log.add_health(ts, "deviation", **detail)
    log.add_health(ts + 1, "heartbeat")
    log.close()
    path = log.paths["health"]
    assert b"\r" not in path.read_bytes()
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(HEALTH_HEADER) and len(rows) == 3
    assert rows[1][:3] == [str(ts), utc_iso(ts), "deviation"] and len(rows[1]) == 4
    assert json.loads(rows[1][3]) == {
        "error": 'ValueError("bad, worse")',
        "note": 'he said "stop", then left',
        "files": [["fills", 120, 98]],
        "text": "line one\r\nline two",
        "fee": "0.100",
        "nested": {"a,b": [1, 2.5, None, True], "path": r"C:\hb_data\hb_b3"},
        "unit": "µs",
    }
    assert rows[2] == [str(ts + 1), utc_iso(ts + 1), "heartbeat", "{}"]


def test_files_hold_no_carriage_returns_and_a_refused_row_leaves_nothing_behind(tmp_path):
    log = DayLog(tmp_path / "hb_b0" / DATE, DATE, headers("B0"))
    for k in range(3):
        add_one_step(log, k)
    ts = DAY_START_US + 1
    log.add_health(ts, "deviation", text="first\r\nsecond\rthird")
    before = log.flush()
    for bad in ("a\rb", "a\nb", "a,b", 'a"b'):
        with pytest.raises(ValueError):
            log.add("orders", [ts, utc_iso(ts), "fail", "buy-1", "buy", "limit", None, None, None, bad])
    assert log.flush() == before
    log.close()
    for kind, path in log.paths.items():
        data = path.read_bytes()
        assert b"\r" not in data and data.endswith(b"\n"), kind


# --- the day's end: gzip and SHA256SUMS ------------------------------------------------------------------------------


def test_finalize_day_gzips_each_file_and_writes_verifiable_sums(tmp_path):
    bot_dir = tmp_path / "hb_b0"
    folder = bot_dir / DATE
    original = write_day(bot_dir, DATE)
    lines = finalize_day(folder, DATE)
    assert sorted(path.name for path in folder.iterdir()) == sorted([name + ".gz" for name in original] + [SUMS_NAME])
    for name, data in original.items():
        packed = (folder / (name + ".gz")).read_bytes()
        assert gzip.decompress(packed) == data, name
        with gzip.open(folder / (name + ".gz"), "rb") as handle:
            assert handle.read() == data, name
        assert packed[:3] == b"\x1f\x8b\x08" and packed[3] == 0, name  # deflate, no FNAME (no file name)
        assert packed[4:8] == b"\x00\x00\x00\x00", name  # MTIME 0
    sums = (folder / SUMS_NAME).read_bytes()
    assert b"\r" not in sums and sums.endswith(b"\n")
    assert sums.decode("ascii").splitlines() == lines
    assert verify_sums(folder, lines) == sorted(name + ".gz" for name in original)


def test_gzip_output_is_deterministic(tmp_path):
    first, second = tmp_path / "a" / "hb_b0", tmp_path / "b" / "hb_b0"
    original = write_day(first, DATE)
    (second / DATE).mkdir(parents=True)
    for name, data in original.items():  # the same bytes elsewhere, with other modification times
        (second / DATE / name).write_bytes(data)
        os.utime(second / DATE / name, (1_000_000_000 + len(name), 1_000_000_000 + len(name)))
    lines_first = finalize_day(first / DATE, DATE)
    lines_second = finalize_day(second / DATE, DATE)
    assert lines_first == lines_second
    for name in original:
        assert (first / DATE / (name + ".gz")).read_bytes() == (second / DATE / (name + ".gz")).read_bytes(), name
    for name, data in original.items():  # and again, from the same bytes
        (second / DATE / name).write_bytes(data)
    assert finalize_day(second / DATE, DATE) == lines_first


def test_finalize_day_completes_an_interrupted_run(tmp_path):
    names = [f"{kind}_{DATE}.csv" for kind in KINDS]
    reference_dir = tmp_path / "reference" / "hb_b0"
    original = write_day(reference_dir, DATE)
    reference_lines = finalize_day(reference_dir / DATE, DATE)
    packed = {name: (reference_dir / DATE / (name + ".gz")).read_bytes() for name in names}

    folder = tmp_path / "hb_b0" / DATE
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(original[name])
    # interrupted while compressing the fourth file: snapshots and steps compressed and removed, orders compressed
    # but not yet removed, a partial fills .gz.tmp beside its plain file, trades and health not yet started, and a
    # stale SHA256SUMS.tmp from an earlier attempt
    for name in names[:3]:
        (folder / (name + ".gz")).write_bytes(packed[name])
    for name in names[:2]:
        (folder / name).unlink()
    (folder / (names[3] + ".gz.tmp")).write_bytes(packed[names[3]][:7])
    (folder / (SUMS_NAME + ".tmp")).write_text("garbage\n", encoding="utf-8")

    lines = finalize_day(folder, DATE)
    assert sorted(path.name for path in folder.iterdir()) == sorted([name + ".gz" for name in names] + [SUMS_NAME])
    for name in names:
        assert (folder / (name + ".gz")).read_bytes() == packed[name], name
        assert gzip.decompress(packed[name]) == original[name], name
    assert lines == reference_lines
    verify_sums(folder, (folder / SUMS_NAME).read_text(encoding="utf-8").splitlines())


def test_the_bot_folder_sums_list_every_finalized_day(tmp_path):
    bot_dir = tmp_path / "hb_b3"
    days = ["2026-10-21", "2026-10-22"]
    for day_text in days + ["2026-10-23"]:  # the 23rd is still being written: no SHA256SUMS yet
        write_day(bot_dir, day_text)
    finalize_day(bot_dir / days[0], days[0])
    first = (bot_dir / SUMS_NAME).read_text(encoding="utf-8").splitlines()
    assert [name.split("/")[0] for name in verify_sums(bot_dir, first)] == [days[0]] * len(KINDS)
    finalize_day(bot_dir / days[1], days[1])
    data = (bot_dir / SUMS_NAME).read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    lines = data.decode("ascii").splitlines()
    names = verify_sums(bot_dir, lines)
    assert lines[: len(KINDS)] == first
    assert names == [f"{day_text}/{kind}_{day_text}.csv.gz" for day_text in days for kind in sorted(KINDS)]
    for day_text in days:  # each line is the day's own SHA256SUMS line with the day folder in the path
        day_lines = (bot_dir / day_text / SUMS_NAME).read_text(encoding="utf-8").splitlines()
        assert [line.replace("  ", f"  {day_text}/", 1) for line in day_lines] == [
            line for line in lines if f"  {day_text}/" in line
        ]


def test_coreutils_sha256sum_verifies_the_day_and_bot_folder_sums(tmp_path):
    tool = shutil.which("sha256sum")
    if tool is None:
        pytest.skip("coreutils sha256sum is not on PATH")
    bot_dir = tmp_path / "hb_b3"
    days = ["2026-10-21", "2026-10-22"]
    for day_text in days:
        write_day(bot_dir, day_text)
        finalize_day(bot_dir / day_text, day_text)
    checks = ((bot_dir / days[0], len(KINDS)), (bot_dir / days[1], len(KINDS)), (bot_dir, 2 * len(KINDS)))
    for folder, files in checks:
        result = subprocess.run([tool, "--check", "--strict", SUMS_NAME], cwd=folder, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count(": OK") == files


def test_a_finalized_day_is_never_overwritten_by_a_restart_from_an_older_state(tmp_path):
    """BotCore._end_day closes the day's DayLog and queues finalize_day; only _begin_day's save then moves the
    state to the next day. If the state on disk still names the finalized day (a kill in between, or state saves
    failing until a restart), _restore reopens that day's DayLog with the recorded sizes and the next step ends
    the day again and finalizes it. The run cannot be repeated (PROTOCOL §8), so the finalized files must survive:
    either DayLog refuses to reopen a finalized day or finalize_day refuses to replace a .csv.gz with other bytes."""

    folder = tmp_path / "hb_b0" / DATE
    log = DayLog(folder, DATE, headers("B0"))
    for k in range(6):
        add_one_step(log, k)
    recorded = log.flush()  # the sizes in the last state saved before the day ended
    log.close()
    finalize_day(folder, DATE)  # the Finalizer, after _end_day
    finalized = {path.name: path.read_bytes() for path in folder.glob("*.csv.gz")}
    sums = (folder / SUMS_NAME).read_bytes()
    assert len(finalized) == len(KINDS)

    try:
        reopened = DayLog(folder, DATE, headers("B0"), sizes=recorded)  # _restore from the older state
    except REFUSALS:
        pass
    else:
        reopened.close()
        try:
            finalize_day(folder, DATE)  # _end_day at the next step
        except REFUSALS:
            pass
    changed = sorted(name for name, data in finalized.items() if (folder / name).read_bytes() != data)
    assert changed == [], f"finalized files replaced: {changed}"
    assert (folder / SUMS_NAME).read_bytes() == sums


# --- the Finalizer ---------------------------------------------------------------------------------------------------


def test_finalizer_submit_returns_while_the_day_is_still_being_finalized(tmp_path, monkeypatch):
    entered, release, released = threading.Event(), threading.Event(), []

    def slow_finalize(folder, day_text):
        entered.set()
        released.append(release.wait(timeout=10.0))  # False if submit() itself waited for the day
        return [f"done {day_text}"]

    monkeypatch.setattr(telemetry, "finalize_day", slow_finalize)
    finalizer = Finalizer()
    finalizer.submit(tmp_path, DATE)
    assert entered.wait(timeout=10.0)
    assert finalizer.busy and finalizer.drain() == ([], [])
    release.set()
    finalizer.wait(timeout=10.0)
    assert released == [True]
    assert not finalizer.busy and finalizer.drain() == ([(DATE, [f"done {DATE}"])], [])
    assert finalizer.drain() == ([], [])  # each result is handed out once


def test_finalizer_finalizes_queued_days_in_order_and_collects_errors(tmp_path):
    bot_dir = tmp_path / "hb_b0"
    days = ["2026-10-20", "2026-10-21", "2026-10-22"]
    originals = {day_text: write_day(bot_dir, day_text) for day_text in days}
    finalizer = Finalizer()
    finalizer.submit(bot_dir / days[0], days[0])
    finalizer.submit(bot_dir / "2026-10-19", "2026-10-19")  # no such folder: an error, collected, not raised
    finalizer.submit(bot_dir / days[1], days[1])
    finalizer.submit(bot_dir / days[2], days[2])
    finalizer.wait(timeout=60.0)
    assert not finalizer.busy
    done, errors = finalizer.drain()
    assert [day_text for day_text, _ in done] == days
    assert len(errors) == 1 and errors[0][0] == "2026-10-19"
    assert "FileNotFoundError" in errors[0][1]
    for day_text, lines in done:
        assert (bot_dir / day_text / SUMS_NAME).read_text(encoding="utf-8").splitlines() == lines
        for name, data in originals[day_text].items():
            assert not (bot_dir / day_text / name).exists()
            assert gzip.decompress((bot_dir / day_text / (name + ".gz")).read_bytes()) == data
    names = verify_sums(bot_dir, (bot_dir / SUMS_NAME).read_text(encoding="utf-8").splitlines())
    assert [name.split("/")[0] for name in names] == [day_text for day_text in days for _ in KINDS]


def test_finalizer_never_leaves_a_queued_day_behind(tmp_path, monkeypatch):
    """Bursts of submits from two threads while the worker drains the queue (it stops when the queue is empty,
    so a submit that arrives while it stops must start a new one). After each burst nothing more is submitted:
    every queued day must be finalized without a later submit to rescue it."""

    def quick_finalize(folder, day_text):
        return [day_text]

    monkeypatch.setattr(telemetry, "finalize_day", quick_finalize)
    finalizer = Finalizer()
    rng = np.random.default_rng(31)
    queued: dict[int, list[str]] = {0: [], 1: []}
    finalized: list[str] = []
    errors: list[tuple[str, str]] = []
    for burst in range(150):
        plan = {
            producer: [(f"p{producer}-{burst:03d}-{i}", float(pause))
                       for i, pause in enumerate(rng.uniform(0.0, 0.0003, int(rng.integers(1, 5))))]
            for producer in (0, 1)
        }

        def produce(items: list[tuple[str, float]]) -> None:
            for day_text, pause in items:
                finalizer.submit(tmp_path, day_text)
                time.sleep(pause)  # lets the worker drain the queue and stop between submits

        threads = [threading.Thread(target=produce, args=(plan[producer],)) for producer in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
        for producer in (0, 1):
            queued[producer] += [day_text for day_text, _ in plan[producer]]
        finalizer.wait(timeout=30.0)  # no submit follows: a day left in the queue now stays there
        done_now, errors_now = finalizer.drain()
        finalized += [day_text for day_text, _ in done_now]
        errors += errors_now
        assert sorted(finalized) == sorted(queued[0] + queued[1]), f"burst {burst}"
    for thread in threading.enumerate():
        if thread.name == "finalize":
            thread.join(timeout=30.0)
    assert not any(thread.name == "finalize" and thread.is_alive() for thread in threading.enumerate())
    done = finalized + [day_text for day_text, _ in finalizer.drain()[0]]
    assert len(done) == len(set(done))
    for producer in (0, 1):  # each producer's days in the order it queued them
        assert [day_text for day_text in done if day_text.startswith(f"p{producer}-")] == queued[producer]
    assert errors == [] and not finalizer.busy
