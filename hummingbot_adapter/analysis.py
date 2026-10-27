"""The forward-test analysis, per bot and prospective day (PROTOCOL §9: Q-A to Q-D, data description, feature shift,
shadow replay against paper; §7 afterwards). Every table is descriptive.

    python -m hummingbot_adapter.analysis --data-root DIR --days 2026-10-22 [2026-10-23] --params hb_calibration.json
        --calibration-day-file <2026-10-01 CSV> --run-log hummingbot_adapter/RUN_LOG.md --out-dir DIR
        [--shadow-dir DIR] [--dev]

Inputs, read only: <data-root>/hb_b0 (B0) and <data-root>/hb_b3 (B3), one folder per UTC day with the six telemetry
files {snapshots,steps,orders,fills,trades,health}_<date>.csv.gz. A day is read only if its SHA256SUMS lists exactly
those files, every hash verifies, and the bot folder's SHA256SUMS holds the same lines; each file must start with the
telemetry header. --params is hb_calibration.json (sigma, A, kappa: A and kappa enter E, Q-B). Days outside the main
run (2026-10-22 and -23) are refused unless --dev. With --shadow-dir, shadow_<b0|b3>_<date>_episodes.csv and its JSON
(hummingbot_adapter.shadow_replay) must exist for every bot and day, made from that bot's own snapshots file.

Conventions (PROTOCOL §9 and the readings WO-HB3 fixes where §9 is silent; ANALYSIS.md):
- trading window: the steps with ts < 23:59:50 UTC of the day; "before the close" is its last step;
- time weights as fast_simulate: step i >= 1 of the window carries dt_i = (ts_i - ts_{i-1}) / 1e6 (int64 µs) with its
  own q (the inventory after the fills before it); step 0, dt = 0 and steps from 23:59:50 carry no weight;
- E (Q-B) on the same intervals: the orders resting after step i-1, at their distance from step i-1's mid, meet dt_i;
- Q-B uses limit fills only (close == 0); a markout whose t_i + h lies beyond the file's last step is dropped and
  counted, and R(h) sums over the fills whose markout at h is kept; S(t) is the mid of the last step with ts <= t;
- §7 afterwards uses the realized path of every step of the day, as the bot's own §7 check.

Outputs in --out-dir (with --dev in --out-dir/dev, every title starting "DEVELOPMENT – warm-up, excluded"):
forward_test_tables.md, one CSV per table, results.json, appendix_table.tex, fig_forward_inventory.pdf/.png with its
data CSV, and SHA256SUMS. The folder must be new or empty and may not lie inside the repository or the data root.
Exit codes: 0 done; 2 a refusal.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np  # numpy and pandas before cvar_mm (and anything that brings torch): one OpenMP runtime on Windows
import pandas as pd

from hummingbot_adapter.bot_config import BOT_NAMES, CALIBRATION_SHA256, CALIBRATION_SHA256_RUN, protocol_default
from hummingbot_adapter.provenance import REPO_ROOT, file_sha256, git_output, is_inside
from hummingbot_adapter.telemetry import (
    FILL_HEADER,
    HEALTH_HEADER,
    KINDS,
    ORDER_HEADER,
    SNAPSHOT_HEADER,
    SUMS_NAME,
    TRADE_HEADER,
    headers,
    utc_iso,
)
from cvar_mm.gym.calibration import _sigma_from_day
from cvar_mm.gym.data_loader import load_single_day_calibration_arrays
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES

LABEL = "descriptive; forward test; not pre-specified for inference"
DEV_PREFIX = "DEVELOPMENT – warm-up, excluded"
RUN_DAYS = ("2026-10-22", "2026-10-23")  # PROTOCOL §6, RUN_LOG pre-run freeze (2026-10-21T18:15:40Z)
RUN_WINDOW = ("2026-10-22T00:00:00Z", "2026-10-24T00:00:00Z")
NETWORK_WINDOW = ("2026-10-22T22:47:00Z", "2026-10-23T05:39:00Z")  # the iPhone hotspot (RUN_LOG 2026-10-23T05:49:24Z)
MODEL_OF = {bot: model for model, bot in BOT_NAMES.items()}  # hb_b0 -> B0, hb_b3 -> B3
SHADOW_NAME = {"B0": "b0", "B3": "b3"}
US = 1_000_000
DAY_US = 86_400 * US
CLOSE_US = DAY_US - protocol_default("close_seconds_before_midnight") * US  # 23:59:50 (PROTOCOL §5.5)
TICK = protocol_default("tick_size")  # USDT
TICK_DECIMAL = Decimal("0.00001")
UNIT_DOGE = protocol_default("order_amount_doge")  # one model unit of inventory = 100 DOGE (PROTOCOL §5.1)
Q_MAX = protocol_default("q_max")
NEAR_CAP = 0.8 * Q_MAX  # |q| >= 8 (PROTOCOL §9 Q-A, as in the replay)
CAP_DOGE = Q_MAX * UNIT_DOGE  # 1,000 DOGE at the cap
NET_LIMIT_USDT = protocol_default("drawdown_limit_usdt")  # 40 USDT (§7)
LOSS_SHARES = (0.10, 0.20, 0.30)  # §7 afterwards: of the notional at the cap
FEE_GRID_PCT = (0.0, 0.01, 0.02, 0.075, 0.10)  # §9 Q-C, in %
MARKOUT_HORIZONS_S = (1, 10, 60)
DAY_CLOSE_PURPOSES = ("day_close", "carry")  # bot_core._flatten's reasons for the day's close (§5.4, §5.5)
S7_CLOSE_PURPOSES = ("halt", "drawdown")  # its reasons for a §7 close inside the day
DOWNTIME_GAP_S = 30.0  # a step gap longer than this is downtime (WO-HB3)
FLAG_DOWNTIME_MIN = 60.0  # PROTOCOL §8: a day with more than 1 h of downtime is flagged in every table
VOL_WINDOW_S = 600.0  # CALIBRATION_VOL_WINDOW_SECONDS, the frozen sigma estimator's blocks
Z_CLIP = 10.0  # the frozen VecNormalize clip_obs (PROTOCOL §5.2)
Z_LARGE = 3.0
SHADOW_PATHS = 40  # fill seeds 42 and 43 x 20 rollouts (PROTOCOL §9)
PIT_FLAGS = (0.05, 0.95)
AT_PRICE_WINDOW_US = 500_000  # exploratory: the public trades logged in the 0.5 s before a fill
STEP_COLUMNS = ("ts_us", "phase", "dt", "mid", "resting_bid", "resting_ask", "q", "wealth", "drawdown",
                "residual_doge", "bid_rounded", "ask_rounded", "quoting", "drawdown_paused", "halted")
FLAG_COLUMNS = ("bid_rounded", "ask_rounded", "quoting", "drawdown_paused", "halted")
TRIGGER_EVENTS = ("pause", "resume", "warn", "halt", "close_and_pause")
RUN_LOG_ENTRY = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z): (.*)$")
DATE_IN_NAME = re.compile(r"\d{4}-\d{2}-\d{2}")
EPOCH_DATE = Date(1970, 1, 1)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AnalysisRefused(RuntimeError):
    """An input fails a rule of WO-HB3 (a day not finalized or not verified, a date outside the run, an out-dir)."""


# --- time --------------------------------------------------------------------------------------------------------


def day_bounds(date: str) -> tuple[int, int, int]:
    """00:00, 23:59:50 and 24:00 UTC of the day in int µs (exact integer arithmetic)."""

    start = (Date.fromisoformat(date) - EPOCH_DATE).days * DAY_US
    return start, start + CLOSE_US, start + DAY_US


def iso_us(text: str) -> int:
    """An ISO time 'YYYY-MM-DDTHH:MM:SSZ' in int µs."""

    moment = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (moment - EPOCH) // timedelta(microseconds=1)


def utc_text(ts_us: int | None) -> str:
    return "never" if ts_us is None else utc_iso(int(ts_us))


def in_network_window(ts_us: int) -> bool:
    """Inside [22:47Z on 2026-10-22, 05:39Z on 2026-10-23), the mobile-hotspot window (exploratory)."""

    return iso_us(NETWORK_WINDOW[0]) <= ts_us < iso_us(NETWORK_WINDOW[1])


# --- inputs ------------------------------------------------------------------------------------------------------


@dataclass
class DayFiles:
    bot: str
    model: str
    date: str
    folder: Path
    paths: dict[str, Path]
    sha256: dict[str, str]
    sums_sha256: str


@dataclass
class DayData:
    files: DayFiles
    steps: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    health: pd.DataFrame


def verify_day(data_root: Path, bot: str, date: str) -> DayFiles:
    """A finalized day folder: SHA256SUMS present, listing exactly the six telemetry files of the day (and nothing
    else is in the folder), every hash verified, and the same lines in the bot folder's SHA256SUMS."""

    folder = data_root / bot / date
    sums = folder / SUMS_NAME
    if not sums.is_file():
        raise AnalysisRefused(f"{folder}: no {SUMS_NAME}; the day is not finalized and is not read")
    expected = {f"{kind}_{date}.csv.gz": kind for kind in KINDS}
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest) or name in listed:
            raise AnalysisRefused(f"{sums}: malformed or repeated line {line!r}")
        listed[name] = digest
    if set(listed) != set(expected):
        raise AnalysisRefused(f"{sums} lists {sorted(listed)}, expected {sorted(expected)}")
    present = {path.name for path in folder.iterdir() if path.is_file() and path.name != SUMS_NAME}
    if present != set(expected):
        raise AnalysisRefused(f"{folder} holds {sorted(present)}, expected exactly {sorted(expected)}")
    for name, digest in listed.items():
        actual = file_sha256(folder / name)
        if actual != digest:
            raise AnalysisRefused(f"{folder / name}: SHA-256 {actual}, {SUMS_NAME} records {digest}")
    bot_sums = data_root / bot / SUMS_NAME
    if not bot_sums.is_file():
        raise AnalysisRefused(f"{bot_sums} is missing")
    bot_lines = {}
    for line in bot_sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if name.startswith(f"{date}/"):
            bot_lines[name[len(date) + 1:]] = digest
    if bot_lines != listed:
        raise AnalysisRefused(f"{bot_sums} disagrees with {sums} for {date}")
    return DayFiles(
        bot=bot, model=MODEL_OF[bot], date=date, folder=folder,
        paths={kind: folder / name for name, kind in expected.items()},
        sha256={kind: listed[name] for name, kind in expected.items()},
        sums_sha256=file_sha256(sums),
    )


def _check_header(path: Path, header: tuple[str, ...]) -> None:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        first = handle.readline().rstrip("\n")
    if first != ",".join(header):
        raise AnalysisRefused(f"{path.name} does not start with the telemetry header")


def read_day(files: DayFiles) -> DayData:
    """The day's steps, fills, orders, trades and health rows; timestamps read as exact int64 µs."""

    model_headers = headers(files.model)
    for kind in KINDS:
        _check_header(files.paths[kind], tuple(model_headers[kind]))
    z_columns = [f"z_{name}" for name in FORMULA_FEATURE_NAMES] if files.model == "B3" else []
    step_dtypes = {name: "float64" for name in STEP_COLUMNS} | {name: "float64" for name in z_columns}
    step_dtypes |= {"ts_us": "int64", "phase": str} | {name: "int64" for name in FLAG_COLUMNS}
    # the files print floats as shortest round-trip reprs: read them back exactly (round_trip)
    steps = pd.read_csv(files.paths["steps"], usecols=list(STEP_COLUMNS) + z_columns, dtype=step_dtypes,
                        float_precision="round_trip")
    fills = pd.read_csv(files.paths["fills"], float_precision="round_trip", dtype={
        "ts_us": "int64", "utc": str, "order_id": str, "side": str, "order_type": str, "price": str, "amount": str,
        "mid_at_fill": "float64", "delta_fill_ticks": "float64", "post_ts_us": "Int64", "mid_at_post": "float64",
        "delta_post_ticks": "float64", "fee": str, "fee_asset": str, "close": "int64", "exchange_trade_id": str,
    })
    orders = pd.read_csv(files.paths["orders"], dtype={name: str for name in ORDER_HEADER} | {"ts_us": "int64"})
    trades = pd.read_csv(files.paths["trades"], float_precision="round_trip", dtype={
        "ts_us": "int64", "utc": str, "exchange_ts": str, "trade_id": str, "price": "float64", "amount": "float64",
        "side": str,
    })
    health = pd.read_csv(files.paths["health"], dtype={name: str for name in HEALTH_HEADER} | {"ts_us": "int64"})
    start, _, end = day_bounds(files.date)
    for kind, frame in (("steps", steps), ("fills", fills), ("orders", orders), ("trades", trades), ("health", health)):
        ts = frame["ts_us"].to_numpy()
        if ts.size and (np.any(np.diff(ts) < 0) or ts[0] < start or ts[-1] >= end):
            raise AnalysisRefused(f"{files.bot} {files.date} {kind}: ts not non-decreasing inside the UTC day")
    if steps.empty:
        raise AnalysisRefused(f"{files.bot} {files.date}: no steps")
    q = steps["q"].to_numpy()
    if not np.array_equal(q, np.round(q)):
        raise AnalysisRefused(f"{files.bot} {files.date}: a step's q is not a whole number of units")
    return DayData(files, steps, fills, orders, trades, health)


def read_params(path: Path) -> dict[str, Any]:
    """sigma, A and kappa of hb_calibration.json, its SHA-256 and whether that is the pre-registered calibration."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(key in data for key in ("sigma", "A", "kappa")):
        raise AnalysisRefused(f"{path}: expected hb_calibration.json with the keys sigma, A and kappa")
    values = {}
    for key in ("sigma", "A", "kappa"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AnalysisRefused(f"{path}: {key} must be a finite positive number, got {value!r}")
        values[key] = float(value)
    digest = file_sha256(path)
    files = [item for item in data.get("files", []) if isinstance(item, dict)]
    recorded = {item.get(name) for item in files for name in ("csv_sha256", "input_sha256")} - {None}
    return {"path": str(path), "sha256": digest, "pre_registered": digest in (CALIBRATION_SHA256, CALIBRATION_SHA256_RUN),
            "recorded_input_sha256": sorted(recorded), "recorded_files": files, **values}


# --- Q-A: inventory ------------------------------------------------------------------------------------------------


def interval_weights(ts_us: np.ndarray, close_us: int) -> np.ndarray:
    """fast_simulate's time weights on the step rows: row i >= 1 carries dt_i = (ts_i - ts_{i-1}) / 1e6 seconds (int64
    µs differences, as the frozen loader) when ts_i < 23:59:50; row 0, rows with dt_i = 0 and rows from 23:59:50
    carry 0 (PROTOCOL §9; WO-HB3 step 1c)."""

    ts_us = np.asarray(ts_us, dtype=np.int64)
    weights = np.zeros(len(ts_us), dtype=np.float64)
    if len(ts_us) > 1:
        differences = np.diff(ts_us)
        if np.any(differences < 0):
            raise ValueError("ts_us decreases")
        weights[1:] = differences.astype(np.float64) / 1e6
    weights[ts_us >= close_us] = 0.0
    return weights


def inventory_stats(ts_us: np.ndarray, q: np.ndarray, close_us: int) -> dict[str, float]:
    """Time-weighted mean |q| and near-cap share (|q| >= 8) over the trading window."""

    weights = interval_weights(ts_us, close_us)
    total = float(weights.sum())
    if total <= 0:
        return {"mean_abs_q": math.nan, "near_cap_share": math.nan, "weighted_s": 0.0}
    absolute = np.abs(np.asarray(q, dtype=np.float64))
    return {
        "mean_abs_q": float(np.sum(absolute * weights) / total),
        "near_cap_share": float(np.sum((absolute >= NEAR_CAP) * weights) / total),
        "weighted_s": total,
    }


def drawdown_path(wealth: np.ndarray) -> np.ndarray:
    """The running peak of W (from W = 0 at the day's start, as the ledger and fast_simulate) minus W."""

    wealth = np.asarray(wealth, dtype=np.float64)
    peak = np.maximum.accumulate(np.concatenate(([0.0], wealth)))[1:]
    return peak - wealth


# --- Q-B: markouts and expected fills --------------------------------------------------------------------------------


def asof_mid(step_ts: np.ndarray, step_mid: np.ndarray, at_us: np.ndarray) -> np.ndarray:
    """S(t): the mid of the last step with ts <= t (t at or after the first step)."""

    index = np.searchsorted(step_ts, at_us, side="right") - 1
    if np.any(index < 0):
        raise ValueError("a time before the day's first step")
    return np.asarray(step_mid, dtype=np.float64)[index]


def markouts(fill_ts: np.ndarray, sign: np.ndarray, price: np.ndarray, step_ts: np.ndarray, step_mid: np.ndarray,
             horizons_s: tuple[int, ...] = MARKOUT_HORIZONS_S) -> dict[int, np.ndarray]:
    """m_i(h) = s_i (S(t_i + h) - p_i) / tick; NaN (dropped) where t_i + h lies beyond the file's last step."""

    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    result = {}
    for horizon in horizons_s:
        target = fill_ts + horizon * US
        kept = target <= step_ts[-1]
        values = np.full(len(fill_ts), np.nan)
        if kept.any():
            mids = asof_mid(step_ts, step_mid, target[kept])
            values[kept] = sign[kept] * (mids - price[kept]) / TICK
        result[horizon] = values
    return result


def retention(markout: np.ndarray, delta_post: np.ndarray) -> float:
    """R(h) = sum of m_i(h) / sum of delta_post,i over the fills whose markout at h is kept."""

    kept = ~np.isnan(markout)
    denominator = float(np.sum(delta_post[kept]))
    return math.nan if not kept.any() or denominator == 0 else float(np.sum(markout[kept]) / denominator)


def expected_fills(ts_us: np.ndarray, mid: np.ndarray, resting_bid: np.ndarray, resting_ask: np.ndarray, close_us: int,
                   A: float, kappa: float) -> dict[str, float]:
    """E = sum over the window's intervals and resting sides of 1 - exp(-A exp(-kappa delta) dt): interval i (row i of
    the window, i >= 1) has dt_i of interval_weights; the orders resting after step i-1 meet it at their distance delta
    from step i-1's mid in USDT, as fast_simulate's quote of step t-1 meets dt[t] (PROTOCOL §9 Q-B)."""

    weights = interval_weights(ts_us, close_us)[1:]
    previous_mid = np.asarray(mid, dtype=np.float64)[:-1]
    out = {}
    for side, prices, sign in (("bid", resting_bid, 1.0), ("ask", resting_ask, -1.0)):
        resting = np.asarray(prices, dtype=np.float64)[:-1]
        depth = sign * (previous_mid - resting)
        present = ~np.isnan(resting)
        probability = -np.expm1(-A * np.exp(-kappa * depth[present]) * weights[present])
        out[f"E_{side}"] = float(probability.sum())
    out["E"] = out["E_bid"] + out["E_ask"]
    out["intervals"] = int(np.count_nonzero(weights))
    return out


# --- Q-C and §7 afterwards ---------------------------------------------------------------------------------------


def tick_round(value: Decimal) -> Decimal:
    return value.quantize(TICK_DECIMAL, rounding=ROUND_HALF_EVEN)


def fee_curve(P_pre: float, P_post: float, F: float, C: float) -> list[dict[str, float]]:
    return [{"fee_pct": fee, "P_pre_net_usdt": P_pre - fee / 100 * F, "P_post_net_usdt": P_post - fee / 100 * (F + C)}
            for fee in FEE_GRID_PCT]


def break_even_pct(pnl: float, notional: float) -> float | None:
    """The fee in % at which the P&L is 0: P / notional * 100; None ("n/a") for a zero notional."""

    return None if notional == 0 else pnl / notional * 100


def first_above(ts_us: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    above = np.flatnonzero(np.asarray(values) > threshold)
    return None if above.size == 0 else int(ts_us[above[0]])


def net_drawdown(ts_us: np.ndarray, wealth: np.ndarray, fill_ts: np.ndarray, fill_notional: np.ndarray,
                 fee_fraction: float) -> np.ndarray:
    """The drawdown of the net equity W - f x (notional of the limit fills with ts <= the step's ts)."""

    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(fill_notional, dtype=np.float64))))
    booked = cumulative[np.searchsorted(fill_ts, ts_us, side="right")]
    return drawdown_path(np.asarray(wealth, dtype=np.float64) - fee_fraction * booked)


# --- Q-D -----------------------------------------------------------------------------------------------------------


def step_gaps(ts_us: np.ndarray, start_us: int, end_us: int,
              minimum_s: float = DOWNTIME_GAP_S) -> list[tuple[int, int]]:
    """The gaps longer than minimum_s between consecutive steps, from 00:00 to the first step and from the last step to
    24:00 included."""

    points = np.concatenate(([start_us], np.asarray(ts_us, dtype=np.int64), [end_us]))
    lengths = np.diff(points)
    return [(int(points[i]), int(points[i + 1])) for i in np.flatnonzero(lengths > minimum_s * US)]


def max_in_window(ts_us: np.ndarray, window_us: int = 10 * US) -> int:
    """The largest number of events in any window (t - window, t], the RateLimiter's rolling window."""

    ts_us = np.asarray(ts_us, dtype=np.int64)
    if ts_us.size == 0:
        return 0
    left = np.searchsorted(ts_us, ts_us - window_us, side="right")
    return int(np.max(np.arange(ts_us.size) - left + 1))


def union_seconds(intervals: list[tuple[int, int]], low: int, high: int) -> float:
    clipped = sorted((max(a, low), min(b, high)) for a, b in intervals if min(b, high) > max(a, low))
    total, current_start, current_end = 0, None, None
    for a, b in clipped:
        if current_end is None or a > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = a, b
        else:
            current_end = max(current_end, b)
    if current_end is not None:
        total += current_end - current_start
    return total / US


AT_PRICE_LABELS = ("at_price_engine_through", "at_price_not_engine_through")


def classify_trade_fills(fill_ts: np.ndarray, post_ts: np.ndarray, sign: np.ndarray, fill_prices: list[str],
                         trade_ts: np.ndarray, trade_prices: np.ndarray, trade_side: np.ndarray) -> list[str]:
    """Exploratory: for each limit fill, the opposite-side taker trades (taker sells for our bid, taker buys for our
    ask) logged after its order's post and in the 0.5 s up to the fill, compared in whole ticks:
    - "through": one printed strictly through the order price;
    - an at-price fill, none through but one exactly at the price, split by the paper engine's own test (it fills a
      bid on a taker sell strictly below the bid, an ask on a taker buy strictly above it, comparing the Decimal order
      price with the float trade price exactly): "at_price_engine_through" if such an at-price print passes that test
      (so it can have triggered the fill), "at_price_not_engine_through" if none does;
    - "no_trade": none reached the price (a book cross at a clock tick, or a trigger outside the window)."""

    trade_ticks = np.rint(np.asarray(trade_prices, dtype=np.float64) / TICK).astype(np.int64)
    labels = []
    for ts, post, s, text in zip(fill_ts, post_ts, sign, fill_prices):
        order_price = Decimal(text)
        ticks = int((order_price / TICK_DECIMAL).to_integral_value())
        low = int(ts) - AT_PRICE_WINDOW_US
        if post is not None and not pd.isna(post):
            low = max(int(post), low)
        first, last = np.searchsorted(trade_ts, low, side="right"), np.searchsorted(trade_ts, int(ts), side="right")
        index = first + np.flatnonzero(trade_side[first:last] == ("sell" if s > 0 else "buy"))
        if np.any(trade_ticks[index] < ticks) if s > 0 else np.any(trade_ticks[index] > ticks):
            labels.append("through")
            continue
        at_price = [Decimal(float(trade_prices[i])) for i in index[trade_ticks[index] == ticks]]
        if not at_price:
            labels.append("no_trade")
            continue
        engine = any((order_price > value) if s > 0 else (order_price < value) for value in at_price)
        labels.append(AT_PRICE_LABELS[0] if engine else AT_PRICE_LABELS[1])
    return labels


FILL_CLASSES = ("through", *AT_PRICE_LABELS, "no_trade")
THROUGH_OR_NO_TRADE = "through + no_trade"


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def fill_class_rows(labels: list[str], delta_post: np.ndarray, delta_fill: np.ndarray, marks: dict[int, np.ndarray],
                    known: np.ndarray, in_window: np.ndarray, E: float) -> list[dict[str, Any]]:
    """Exploratory (WO-HB3b): Q-B per fill class of classify_trade_fills, then over all limit fills and over the
    through and no_trade fills together. Per row: the fills, the mean delta_post (fills of known orders) and
    delta_fill, and per horizon the mean m(h), R(h) and the dropped markouts, as in Q-B; for the two aggregates also
    the window's fills over E, the second a sensitivity to the paper engine's Decimal-against-float comparison."""

    labels = np.asarray(labels, dtype=object)
    groups = [(name, labels == name) for name in FILL_CLASSES]
    groups += [("all", np.ones(labels.size, dtype=bool)),
               (THROUGH_OR_NO_TRADE, (labels == "through") | (labels == "no_trade"))]
    rows = []
    for name, member in groups:
        row: dict[str, Any] = {"class": name, "fills": int(member.sum()),
                               "share": float(member.mean()) if labels.size else math.nan,
                               "mean_delta_post_ticks": _mean(delta_post[member & known]),
                               "mean_delta_fill_ticks": _mean(delta_fill[member])}
        for horizon, values in marks.items():
            row[f"mean_m_{horizon}s"] = _mean(values[member & ~np.isnan(values)])
            row[f"R_{horizon}s"] = retention(values[member & known], delta_post[member & known])
            row[f"dropped_{horizon}s"] = int(np.sum(member & np.isnan(values)))
        aggregate = name in ("all", THROUGH_OR_NO_TRADE)
        row["fills_over_E"] = float(np.sum(member & in_window)) / E if aggregate and E > 0 else None
        rows.append(row)
    return rows


SPLIT_TOLERANCE_USDT = 1e-9  # WO-HB3d: the split's parts against the logged wealth


def pnl_split(ts_us: np.ndarray, mid: np.ndarray, q: np.ndarray, wealth: np.ndarray, last: int,
              fills: pd.DataFrame) -> dict[str, Any]:
    """Exploratory (WO-HB3d): P_pre = fill edge + inventory over the trading window (steps 0..last).

    inventory = sum over the steps i = 1..last of 100 q_{i-1} (S_i - S_{i-1}); a fill's edge is s a (S - p), S the mid
    of the first step whose wealth includes the fill. A step marks W = cash + 100 q S over every fill booked before it
    (Ledger.mark, before its quotes), its ts is read when its tick starts, the paper connector books fills outside
    BotCore.on_tick at the step clock's reading, and the step clock never goes back: so a fill's ts is never later than
    that step's, and the convention takes the first step with ts >= the fill's ts. A fill at a step's own ts is taken
    to have been booked before that step. That holds while the clock advances between a step and a later fill, not
    while ts is held (at a restart floor ahead of the wall clock): such a fill would be put on a step whose wealth
    excludes it, and the checks below flag the day. fills_at_step_ts counts the fills the tie rule places.

    A day starts flat with zero cash (Ledger.start_day): a fill before the day's first step enters step 0 and no term
    precedes it. Then W_i - W_{i-1} = 100 q_{i-1} (S_i - S_{i-1}) + the edges of the fills step i adds, up to the bot's
    float rounding of W, where q_{i-1} is the logged q, the position of its wealth (position_mismatch_steps checks it
    against the fills). The fill edge is over the limit fills (close == 0); a market fill inside the window (a §7
    close) is the third part, so the parts sum to P_pre on any day. Fills that only later steps include (the 23:59:50
    close, a limit fill after the last window step) are left out. Each edge is exact (fractions of the float mid and
    the decimal price and amount) and rounded once; the totals are math.fsum. The checks: the parts' sum minus P_pre,
    and the largest |W_i - W_{i-1} - (step i's parts)| over the window (W_{-1} = 0)."""

    window_ts = np.asarray(ts_us, dtype=np.int64)[: last + 1]
    S = np.asarray(mid, dtype=np.float64)[: last + 1]
    units = np.asarray(q, dtype=np.float64)[: last + 1]
    W = np.asarray(wealth, dtype=np.float64)[: last + 1]
    fill_ts = fills["ts_us"].to_numpy(dtype=np.int64)
    first_step = np.searchsorted(window_ts, fill_ts, side="left")
    included = first_step <= last  # the fill's first including step is a window step
    at_step_ts = np.zeros(len(fills), dtype=bool)
    at_step_ts[included] = window_ts[first_step[included]] == fill_ts[included]
    sign = np.where(fills["side"].to_numpy(dtype=object) == "buy", 1, -1)
    amounts = [Fraction(a) for a in fills["amount"]]
    prices = [Fraction(p) for p in fills["price"]]
    edge = np.zeros(len(fills))
    signed_doge = np.zeros(len(fills))
    for k in np.flatnonzero(included):
        signed = amounts[k] if sign[k] > 0 else -amounts[k]
        edge[k] = float(signed * (Fraction(float(S[first_step[k]])) - prices[k]))
        signed_doge[k] = float(signed)
    limit = fills["close"].to_numpy() == 0
    inventory_terms = np.zeros(last + 1)
    inventory_terms[1:] = UNIT_DOGE * units[:-1] * np.diff(S)
    step_edges = np.bincount(first_step[included], weights=edge[included], minlength=last + 1)
    previous = np.concatenate(([0.0], W[:-1]))
    position = np.cumsum(np.bincount(first_step[included], weights=signed_doge[included], minlength=last + 1))
    P_pre = float(W[last])
    in_limit, in_market = included & limit, included & ~limit
    limit_doge = float(sum((amounts[k] for k in np.flatnonzero(in_limit)), Fraction(0)))
    fill_edge = math.fsum(edge[in_limit])
    sum_minus = math.fsum([*inventory_terms, *edge[included], -P_pre])
    max_step = float(np.max(np.abs((W - previous) - (inventory_terms + step_edges))))
    return {
        "P_pre_usdt": P_pre,
        "fill_edge_usdt": fill_edge,
        "inventory_usdt": math.fsum(inventory_terms),
        "fill_edge_buys_usdt": math.fsum(edge[in_limit & (sign > 0)]),
        "fill_edge_sells_usdt": math.fsum(edge[in_limit & (sign < 0)]),
        "market_fill_edge_usdt": math.fsum(edge[in_market]),
        "limit_fills": int(in_limit.sum()),
        "market_fills": int(in_market.sum()),
        "fills_at_step_ts": int(at_step_ts.sum()),
        "mean_fill_edge_ticks": fill_edge / (limit_doge * TICK) if limit_doge else math.nan,
        "parts_minus_P_pre_usdt": sum_minus,
        "max_abs_step_mismatch_usdt": max_step,
        "position_mismatch_steps": int(np.sum(~np.isclose(position, UNIT_DOGE * units, rtol=0.0, atol=1e-9))),
        "within_tolerance": bool(abs(sum_minus) <= SPLIT_TOLERANCE_USDT and max_step <= SPLIT_TOLERANCE_USDT),
    }


# --- the other sections ---------------------------------------------------------------------------------------------


def describe_snapshots(path: Path) -> dict[str, Any]:
    """Rows, mid, sigma by the frozen estimator and the share of 1-tick spreads, through the frozen loader."""

    mid, dt, _, bid, ask = load_single_day_calibration_arrays(str(path))
    return {
        "rows": int(len(mid)),
        "mid_mean": float(np.mean(mid)),
        "mid_min": float(np.min(mid)),
        "mid_max": float(np.max(mid)),
        "sigma": _sigma_from_day(mid, dt, VOL_WINDOW_S),
        "one_tick_spread_share": float(np.mean(np.rint((ask - bid) / TICK) == 1)),
    }


def pit(shadow: np.ndarray, paper: float) -> float:
    """(#{m < m_paper} + 1/2 #{m = m_paper}) / n (Diebold, Gunther & Tay 1998)."""

    shadow = np.asarray(shadow, dtype=np.float64)
    return float((np.sum(shadow < paper) + 0.5 * np.sum(shadow == paper)) / shadow.size)


def run_log_entries(path: Path, start: str = RUN_WINDOW[0]) -> list[tuple[str, str]]:
    """The RUN_LOG.md entries written from the run's start on (an entry about a day is written after it, so there is
    no upper bound): (the entry's UTC stamp, its full text)."""

    entries: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RUN_LOG_ENTRY.match(line)
        if match:
            entries.append([match.group(1), match.group(2)])
        elif entries and line.strip():
            entries[-1][1] += " " + line.strip()
    low = iso_us(start)
    return [(stamp, text) for stamp, text in entries if iso_us(stamp) >= low]


def entry_days(stamp: str, text: str, days: list[str]) -> list[str]:
    """The analysed days an entry concerns: those whose date appears in its stamp or its text."""

    return [day for day in days if day in stamp or day in text]


def _detail(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        return {}
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


# --- one bot and day -------------------------------------------------------------------------------------------------


def analyse_day(day: DayData, A: float, kappa: float) -> dict[str, Any]:
    """Every per-bot, per-day quantity of PROTOCOL §9 and §7 afterwards (the tables are built from these)."""

    files = day.files
    start, close, end = day_bounds(files.date)
    steps, fills, orders, trades, health = day.steps, day.fills, day.orders, day.trades, day.health
    ts = steps["ts_us"].to_numpy(dtype=np.int64)
    mid = steps["mid"].to_numpy(dtype=np.float64)
    q = steps["q"].to_numpy(dtype=np.float64)
    wealth = steps["wealth"].to_numpy(dtype=np.float64)
    window = ts < close
    if not window.any():
        raise AnalysisRefused(f"{files.bot} {files.date}: no step before 23:59:50")
    last = int(np.flatnonzero(window)[-1])
    out: dict[str, Any] = {"bot": files.bot, "model": files.model, "day": files.date}

    # data checks: the logged dt against the int64 ts differences, the logged drawdown against W's
    logged_dt = steps["dt"].to_numpy(dtype=np.float64)
    recomputed_dt = np.concatenate(([np.nan], np.diff(ts).astype(np.float64) / 1e6))
    dt_mismatch = int(np.sum(~((logged_dt == recomputed_dt) | (np.isnan(logged_dt) & np.isnan(recomputed_dt)))))

    # Q-A
    inventory = inventory_stats(ts, q, close)
    drawdown_window = drawdown_path(wealth[: last + 1])
    out["qa"] = {
        **inventory,
        "window_steps": last + 1,
        "q_pre": int(q[last]),
        "P_pre_usdt": float(wealth[last]),
        "maxdd_pre_usdt": float(drawdown_window.max()),
        "maxdd_logged_usdt": float(steps["drawdown"].to_numpy()[: last + 1].max()),
        "last_window_step_utc": utc_iso(int(ts[last])),
        "dt_mismatches": dt_mismatch,
    }

    # fills: sides, prices, the logged deltas recomputed
    limit = fills[fills["close"] == 0].reset_index(drop=True)
    closes = fills[fills["close"] == 1].reset_index(drop=True)
    if not set(fills["close"].unique()) <= {0, 1}:
        raise AnalysisRefused(f"{files.bot} {files.date}: fills.close holds {sorted(fills['close'].unique())}")
    sign = np.where(limit["side"] == "buy", 1.0, -1.0)
    price = np.array([float(value) for value in limit["price"]], dtype=np.float64)
    fill_ts = limit["ts_us"].to_numpy(dtype=np.int64)
    delta_post = sign * (limit["mid_at_post"].to_numpy(dtype=np.float64) - price) / TICK
    delta_fill = sign * (limit["mid_at_fill"].to_numpy(dtype=np.float64) - price) / TICK
    known = ~np.isnan(limit["mid_at_post"].to_numpy(dtype=np.float64))
    post_difference = np.abs(delta_post - limit["delta_post_ticks"].to_numpy(dtype=np.float64))[known]
    fill_difference = np.abs(delta_fill - limit["delta_fill_ticks"].to_numpy(dtype=np.float64))
    marks = markouts(fill_ts, sign, price, ts, mid)  # every limit fill; R(h) also needs delta_post (known orders)
    qb: dict[str, Any] = {
        "limit_fills": int(len(limit)),
        "buys": int(np.sum(sign > 0)),
        "sells": int(np.sum(sign < 0)),
        "unknown_order_fills": int(np.sum(~known)),
        "mean_delta_post_ticks": float(np.mean(delta_post[known])) if known.any() else math.nan,
        "mean_delta_fill_ticks": float(np.mean(delta_fill)) if len(limit) else math.nan,
        "delta_post_mismatches": int(np.sum(~(post_difference <= 1e-9))),
        "delta_fill_mismatches": int(np.sum(~(fill_difference <= 1e-9))),
        "max_abs_delta_difference": float(np.nanmax(np.concatenate((post_difference, fill_difference, [0.0])))),
    }
    for horizon, values in marks.items():
        kept = ~np.isnan(values)
        qb[f"mean_m_{horizon}s"] = float(np.mean(values[kept])) if kept.any() else math.nan
        qb[f"R_{horizon}s"] = retention(values[known], delta_post[known])
        qb[f"dropped_{horizon}s"] = int(np.sum(~kept))
    window_fills = int(np.sum(fill_ts < close))
    expected = expected_fills(ts, mid, steps["resting_bid"].to_numpy(), steps["resting_ask"].to_numpy(), close,
                              A, kappa)
    qb |= {f"{key}": value for key, value in expected.items()}
    qb["limit_fills_in_window"] = window_fills
    qb["fills_over_E"] = window_fills / expected["E"] if expected["E"] > 0 else math.nan
    out["qb"] = qb

    # Q-C. C is the notional of every market close of the day (fills with close == 1, prices rounded to the tick): the
    # 23:59:50 close, the close of a position carried past 00:00, and a §7 close (halt or drawdown) inside the day,
    # which joins the day's close in P_post - f (F + C) and f*_post and flags the bot-day, s7_close_in_C (the author's
    # decision of 2026-10-24, WO-HB3b). F stays the limit fills, so P_pre and f*_pre leave a §7 close out. A close's
    # purpose is its order's submit row's reason.
    submits = orders[orders["event"] == "submit"]
    purpose = dict(zip(submits["order_id"], submits["reason"]))
    close_purpose = closes["order_id"].map(purpose).fillna("unknown")
    day_close = closes[close_purpose.isin(DAY_CLOSE_PURPOSES)].reset_index(drop=True)
    s7_close = closes[close_purpose.isin(S7_CLOSE_PURPOSES)].reset_index(drop=True)

    def tick_notional(frame: pd.DataFrame) -> Decimal:
        return sum((tick_round(Decimal(p)) * Decimal(a) for p, a in zip(frame["price"], frame["amount"])), Decimal(0))

    F = sum((Decimal(p) * Decimal(a) for p, a in zip(limit["price"], limit["amount"])), Decimal(0))
    C = tick_notional(closes)
    # P_post: W after the day's last fill. A day that ends flat ends with W = its cash, -sum(s p a) over every fill
    # (the ledger's own Decimal arithmetic), also when no step follows the close (a close still in flight at 00:00,
    # or a carry); a day that does not end flat keeps W at its last step, flagged.
    signed = [(1 if s == "buy" else -1, Decimal(p), Decimal(a)) for s, p, a in zip(fills["side"], fills["price"],
                                                                                 fills["amount"])]
    flat_at_end = sum((s * a for s, _, a in signed), Decimal(0)) == 0
    P_pre = float(wealth[last])
    P_post = float(-sum((s * p * a for s, p, a in signed), Decimal(0))) if flat_at_end else float(wealth[-1])

    def fees_usdt(frame: pd.DataFrame) -> Decimal:
        return sum((Decimal(fee) * (tick_round(Decimal(p)) if asset == "DOGE" else Decimal(1))
                    for fee, asset, p in zip(frame["fee"], frame["fee_asset"], frame["price"])), Decimal(0))

    def close_cost(frame: pd.DataFrame) -> Decimal:
        """The half-spread against the mid at the order's submission, plus the fee."""

        return sum((Decimal(1 if s == "buy" else -1) * (tick_round(Decimal(p)) - Decimal(repr(m))) * Decimal(a)
                    for s, p, m, a in zip(frame["side"], frame["price"], frame["mid_at_post"], frame["amount"])),
                   Decimal(0)) + fees_usdt(frame)

    close_done = [_detail(text) for event, text in zip(health["event"], health["detail"]) if event == "close_done"]
    out["qc"] = {
        "F_usdt": float(F),
        "F_exact": str(F),
        "C_usdt": float(C),
        "C_exact": str(C),
        "C_day_close_usdt": float(tick_notional(day_close)),
        "P_pre_usdt": P_pre,
        "P_post_usdt": P_post,
        "P_post_last_step_usdt": float(wealth[-1]),
        "f_star_pre_pct": break_even_pct(P_pre, float(F)),
        "f_star_post_pct": break_even_pct(P_post, float(F + C)),
        "paper_fees_limit_usdt": float(fees_usdt(limit)),
        "paper_fees_close_usdt": float(fees_usdt(day_close)),
        "paper_fees_s7_close_usdt": float(fees_usdt(s7_close)),
        "paper_fees_total_usdt": float(fees_usdt(fills)),
        "close_cost_usdt": float(close_cost(day_close)),
        "close_cost_logged_usdt": [float(item["cost_usdt"]) for item in close_done
                                   if item.get("reason") in DAY_CLOSE_PURPOSES and "cost_usdt" in item],
        "close_fills": int(len(day_close)),
        "close_side": ",".join(sorted(set(day_close["side"]))) or "none",
        "close_amount_doge": float(sum((Decimal(a) for a in day_close["amount"]), Decimal(0))),
        "close_prices": [str(tick_round(Decimal(p))) for p in day_close["price"]],
        "close_utc": [utc_iso(int(value)) for value in day_close["ts_us"]],
        "s7_close_fills": int(len(s7_close)),
        "s7_close_purposes": ",".join(sorted(set(close_purpose[close_purpose.isin(S7_CLOSE_PURPOSES)]))) or "none",
        "s7_close_notional_usdt": float(tick_notional(s7_close)),
        "s7_close_cost_usdt": float(close_cost(s7_close)),
        "close_fills_unknown_purpose": int(np.sum(~close_purpose.isin(DAY_CLOSE_PURPOSES + S7_CLOSE_PURPOSES))),
        "flat_at_day_end": bool(flat_at_end),
        "limit_fills_after_last_window_step": int(np.sum(fill_ts > ts[last])),
        "last_step_after_close_fills": bool(len(closes) == 0 or ts[-1] > closes["ts_us"].max()),
        "s7_close_in_C": bool(len(s7_close) > 0),
        "fee_curve": fee_curve(P_pre, P_post, float(F), float(C)),
    }

    # §7 afterwards: the realized path of every step of the day
    first_mid = float(mid[0])
    cap_notional = CAP_DOGE * first_mid
    gross = drawdown_path(wealth)
    notional = np.array([float(Decimal(p) * Decimal(a)) for p, a in zip(limit["price"], limit["amount"])])
    limits = []
    for share in LOSS_SHARES:
        threshold = share * cap_notional
        limits.append({"limit": f"gross drawdown > {share:.0%} of the notional at the cap", "fee_pct": None,
                       "threshold_usdt": threshold, "first_utc": utc_text(first_above(ts, gross, threshold))})
    for fee in FEE_GRID_PCT:
        path = net_drawdown(ts, wealth, fill_ts, notional, fee / 100)
        limits.append({"limit": f"net-equity drawdown > {NET_LIMIT_USDT:g} USDT", "fee_pct": fee,
                       "threshold_usdt": NET_LIMIT_USDT, "first_utc": utc_text(first_above(ts, path, NET_LIMIT_USDT))})
    out["s7"] = {"first_mid": first_mid, "cap_notional_usdt": cap_notional,
                 "max_gross_drawdown_usdt": float(gross.max()), "limits": limits}

    # Q-D
    event = health["event"].to_numpy(dtype=object)
    details = [_detail(text) if name in {"feed", "clock_check", "book_snapshot", "deviation", "incident", "day_end",
                                         *TRIGGER_EVENTS} else {} for name, text in zip(event, health["detail"])]
    health_ts = health["ts_us"].to_numpy(dtype=np.int64)
    feed = [(t, d.get("kind")) for t, name, d in zip(health_ts, event, details) if name == "feed"]
    gaps = step_gaps(ts, start, end)
    downtime_s = sum(b - a for a, b in gaps) / US
    triggers: dict[str, int] = {}
    pause_open: dict[str, int] = {}
    pauses: list[tuple[str, int, int]] = []
    for t, name, d in zip(health_ts, event, details):
        if name not in TRIGGER_EVENTS:
            continue
        key = f"{d.get('trigger')}:{name}"
        triggers[key] = triggers.get(key, 0) + 1
        if name == "pause":
            pause_open.setdefault(d.get("trigger"), int(t))
        elif name == "resume" and d.get("trigger") in pause_open:
            pauses.append((d.get("trigger"), pause_open.pop(d.get("trigger")), int(t)))
        elif name in ("halt", "close_and_pause"):
            flag = steps["halted" if name == "halt" else "drawdown_paused"].to_numpy()
            later = np.flatnonzero((ts > t) & (flag == 0))
            pauses.append((f"{d.get('trigger')}:{name}", int(t), int(ts[later[0]]) if later.size else end))
    pauses += [(trigger, begin, end) for trigger, begin in pause_open.items()]
    for flag_name in ("halted", "drawdown_paused"):  # a halt or drawdown pause carried in from the day before
        flag = steps[flag_name].to_numpy()
        if flag[0] == 1:
            cleared = np.flatnonzero(flag == 0)
            pauses.append((f"{flag_name}:carried", start, int(ts[cleared[0]]) if cleared.size else end))
    stale = [(a, b) for trigger, a, b in pauses if trigger == "stale_book"]
    submits = orders.loc[orders["event"] == "submit", "ts_us"].to_numpy(dtype=np.int64)
    clock_rows = [(int(t), d) for t, name, d in zip(health_ts, event, details) if name == "clock_check"]
    good_clock = [(t, d) for t, d in clock_rows if d.get("error") is None]
    day_end = [d for name, d in zip(event, details) if name == "day_end"]
    residual = np.abs(steps["residual_doge"].to_numpy(dtype=np.float64))
    out["qd"] = {
        "uptime_pct": 100.0 * (1.0 - downtime_s / (DAY_US / US)),
        "downtime_min": downtime_s / 60.0,
        "gaps_over_30s": len(gaps),
        "longest_gap_s": max([(b - a) / US for a, b in gaps], default=0.0),
        "longest_step_gap_s": float(np.max(np.diff(ts)) / US) if len(ts) > 1 else 0.0,
        "container_restarts": int(np.sum(event == "restart")),
        "fresh_starts": int(np.sum(event == "start")),
        "stops": int(np.sum(event == "stop")),
        "ws_reconnects": sum(1 for _, kind in feed if kind == "ws_closed"),
        "ws_errors": sum(1 for _, kind in feed if kind == "ws_error"),
        "resubscriptions": sum(1 for _, kind in feed if kind == "subscribed"),
        "stale_book_pauses": len(stale),
        "stale_book_pause_s": sum((b - a) / US for a, b in stale),
        "quoting_paused_min": union_seconds([(a, b) for _, a, b in pauses], start, close) / 60.0,
        "rejects": int(np.sum(orders["event"] == "fail")),
        "order_failed_rows": int(np.sum(event == "order_failed")),
        "new_orders": int(submits.size),
        "max_new_orders_10s": max_in_window(submits),
        "rate_skips": int(np.sum((orders["event"] == "skip") & orders["reason"].fillna("").str.startswith("rate"))),
        "not_quoting_skips": int(np.sum((orders["event"] == "skip") & (orders["reason"] == "not_quoting"))),
        "cancels": int(np.sum(orders["event"] == "cancel")),
        "cancelled": int(np.sum(orders["event"] == "cancelled")),
        "s7_triggers": "; ".join(f"{key}={value}" for key, value in sorted(triggers.items())) or "none",
        "clock_checks": len(clock_rows),
        "clock_check_errors": len(clock_rows) - len(good_clock),
        "max_abs_step_offset_s": max((abs(float(d["offset_s"])) for _, d in good_clock), default=math.nan),
        "max_abs_wall_offset_s": max((abs(float(d["wall_offset_s"])) for _, d in good_clock), default=math.nan),
        "max_abs_residual_doge": float(residual.max()),
        "pauses": [{"trigger": trigger, "start_utc": utc_iso(a), "end_utc": utc_iso(min(b, end - 1)),
                    "seconds": (b - a) / US} for trigger, a, b in pauses],
        "gaps": [{"start_utc": utc_iso(a), "end_utc": utc_iso(min(b, end - 1)), "seconds": (b - a) / US,
                  "in_network_window": in_network_window(a)} for a, b in gaps],
        "bot_counters": day_end[-1].get("counters", {}) if day_end else None,
        "bot_rate": day_end[-1].get("rate", {}) if day_end else None,
    }

    # exploratory
    labels = classify_trade_fills(fill_ts, limit["post_ts_us"].to_numpy(dtype=object), sign, list(limit["price"]),
                                  trades["ts_us"].to_numpy(dtype=np.int64), trades["price"].to_numpy(dtype=np.float64),
                                  trades["side"].to_numpy(dtype=object))
    deviations = [d.get("event") for name, d in zip(event, details) if name in ("deviation", "incident")]
    submits_quote = orders[(orders["event"] == "submit") & (orders["order_type"] == "limit")]
    rounded_quotes = submits_quote["reason"] == "quote_rounded"
    # the bot writes crossed_book rows for the first 100 crossed or locked books of a day only (bot_core
    # CROSSED_ROWS_PER_DAY); its counters hold them all: the day_end row's, else the day's last heartbeat's
    counters = out["qd"]["bot_counters"]
    heartbeats = np.flatnonzero(event == "heartbeat")
    last_heartbeat = _detail(health["detail"].iloc[heartbeats[-1]]) if heartbeats.size else {}
    crossed_rows = int(np.sum(event == "crossed_book"))
    if counters is not None:
        crossed, crossed_source = counters.get("crossed_books"), "day_end counters"
    elif "crossed_books" in last_heartbeat:
        crossed, crossed_source = last_heartbeat["crossed_books"], "last heartbeat (no day_end row)"
    else:
        crossed, crossed_source = crossed_rows, "health rows (a lower bound: capped at 100 per day)"
    out["fill_classes"] = fill_class_rows(labels, delta_post, delta_fill, marks, known, fill_ts < close,
                                          expected["E"])
    out["pnl_split"] = pnl_split(ts, mid, q, wealth, last, fills)
    out["exploratory"] = {
        "at_price_trade_fills": sum(labels.count(label) for label in AT_PRICE_LABELS),
        "at_price_engine_through": labels.count(AT_PRICE_LABELS[0]),
        "at_price_not_engine_through": labels.count(AT_PRICE_LABELS[1]),
        "through_trade_fills": labels.count("through"),
        "no_trade_fills": labels.count("no_trade"),
        "book_resyncs": int(np.sum(event == "book_snapshot")),
        "snapshots_ahead_of_diffs": sum(1 for name, d in zip(event, details)
                                        if name == "book_snapshot" and d.get("ahead_of_diffs") is True),
        "snapshot_errors": sum(1 for _, kind in feed if kind == "snapshot_error"),
        "crossed_or_locked_books": crossed,
        "crossed_or_locked_source": crossed_source,
        "crossed_book_rows": crossed_rows,
        "skipped_steps": None if counters is None else counters.get("skipped_steps"),
        "re_anchors": deviations.count("clock_jump"),
        "clock_backward": deviations.count("clock_backward"),
        "round_bid_steps": int(steps["bid_rounded"].sum()),
        "round_ask_steps": int(steps["ask_rounded"].sum()),
        "round_bid_orders": int(np.sum(rounded_quotes & (submits_quote["side"] == "buy"))),
        "round_ask_orders": int(np.sum(rounded_quotes & (submits_quote["side"] == "sell"))),
    }

    # the network window (exploratory): clock rows and gaps by inside or outside
    split = []
    for part, inside in (("inside", True), ("outside", False)):
        rows = [d for t, d in good_clock if in_network_window(t) == inside]
        part_gaps = [(a, b) for a, b in gaps if in_network_window(a) == inside]
        split.append({
            "part": part,
            "clock_checks": len(rows),
            "max_abs_step_offset_s": max((abs(float(d["offset_s"])) for d in rows), default=math.nan),
            "max_abs_wall_offset_s": max((abs(float(d["wall_offset_s"])) for d in rows), default=math.nan),
            "mean_wall_offset_s": float(np.mean([float(d["wall_offset_s"]) for d in rows])) if rows else math.nan,
            "gaps_over_30s": len(part_gaps),
            "downtime_s": sum(b - a for a, b in part_gaps) / US,
        })
    out["network"] = split

    # B3's features over the window
    if files.model == "B3":
        shift = []
        for name in FORMULA_FEATURE_NAMES:
            z = steps[f"z_{name}"].to_numpy(dtype=np.float64)[: last + 1]
            shift.append({"feature": name, "mean_z": float(np.mean(z)),
                          "share_abs_z_gt_3": float(np.mean(np.abs(z) > Z_LARGE)),
                          "share_abs_z_clipped": float(np.mean(np.abs(z) == Z_CLIP))})
        out["feature_shift"] = shift

    # the inventory path for the figure: the day's first step, every change of q, the last step
    change = np.ones(len(q), dtype=bool)
    change[1:] = np.diff(q) != 0
    change[-1] = True
    out["series"] = {"ts_us": ts[change].tolist(), "q": q[change].astype(int).tolist()}
    out["downtime_flag"] = out["qd"]["downtime_min"] > FLAG_DOWNTIME_MIN
    return out


# --- tables ---------------------------------------------------------------------------------------------------------


@dataclass
class Table:
    name: str
    title: str
    frame: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _keyed(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["bot", "day", *columns])
    return frame


def build_tables(results: list[dict[str, Any]], data_rows: list[dict[str, Any]], incidents: list[tuple[str, str]],
                 shadow: list[dict[str, Any]] | None, prefix: str) -> list[Table]:
    """Every table is keyed by (bot, day): one bot and one day per row, never pooled."""

    def title(text: str) -> str:
        return f"{prefix}{text} ({LABEL})"

    flag = {(r["bot"], r["day"]): r["downtime_flag"] for r in results}
    tables = []

    by_day: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        by_day.setdefault(r["day"], {})[r["model"]] = r
    rows = []
    for r in results:
        qa = r["qa"]
        b0 = by_day[r["day"]].get("B0")
        above = None if r["model"] != "B3" or b0 is None else qa["mean_abs_q"] > b0["qa"]["mean_abs_q"]
        rows.append({"bot": r["bot"], "day": r["day"], "model": r["model"], "mean_abs_q": qa["mean_abs_q"],
                     "near_cap_share": qa["near_cap_share"], "q_pre": qa["q_pre"], "P_pre_usdt": qa["P_pre_usdt"],
                     "maxdd_pre_usdt": qa["maxdd_pre_usdt"], "maxdd_logged_usdt": qa["maxdd_logged_usdt"],
                     "window_steps": qa["window_steps"], "weighted_s": qa["weighted_s"],
                     "last_window_step_utc": qa["last_window_step_utc"], "dt_mismatches": qa["dt_mismatches"],
                     "flag_b3_mean_abs_q_above_b0": above,
                     "flag_b3_q_pre_le_minus_8": (qa["q_pre"] <= -NEAR_CAP) if r["model"] == "B3" else None,
                     "downtime_flag": flag[(r["bot"], r["day"])]})
    tables.append(Table("qa_inventory", title("Q-A: inventory, P&L and drawdown before the close"),
                        pd.DataFrame(rows), [
        "Time weights as fast_simulate (each step's q after the fills before it, times the Δt since the previous step); "
        "the trading window ends at 23:59:50 UTC; P_pre and MaxDD in USDT for 100-DOGE orders (divide by 100 to "
        "compare with the replay, which traded 1 DOGE per unit). maxdd_logged_usdt is the bot's own drawdown column "
        "(a check). Flags: B3's mean |q| above B0's on that day; B3's q before the close <= -8 (PROTOCOL §9 Q-A).",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **{k: v for k, v in r["qb"].items()
                                                  if k not in ("E", "E_bid", "E_ask", "intervals", "fills_over_E",
                                                               "limit_fills_in_window")},
             "downtime_flag": flag[(r["bot"], r["day"])]} for r in results]
    tables.append(Table("qb_markouts", title("Q-B: edge at posting and at the fill, markouts and retention"),
                        pd.DataFrame(rows), [
        "Limit fills only (close == 0); ticks of 0.00001 USDT; m(h) = s(S(t+h) - p)/tick with S(t) the mid of the "
        "last step with ts <= t; a markout beyond the file's last step is dropped (dropped_h) and R(h) sums over the "
        "fills whose markout at h is kept. delta_*_mismatches count |recomputed - logged| > 1e-9 tick (expected 0).",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], "limit_fills_in_window": r["qb"]["limit_fills_in_window"],
             "E": r["qb"]["E"], "E_bid": r["qb"]["E_bid"], "E_ask": r["qb"]["E_ask"],
             "fills_over_E": r["qb"]["fills_over_E"], "intervals": r["qb"]["intervals"],
             "downtime_flag": flag[(r["bot"], r["day"])]} for r in results]
    tables.append(Table("qb_expected_fills", title("Q-B: paper fills against the expected fills E"),
                        pd.DataFrame(rows), [
        "E = Σ 1 - exp(-A e^(-κ δ) Δt) over the window's intervals and resting sides, A and κ from hb_calibration.json; "
        "the orders resting after step i-1, at their distance from step i-1's mid, meet the Δt to step i (fast_simulate's "
        "timing). An order cancelled inside an interval (a §7 pause) counts for the whole interval, as in the simulator.",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **point, "s7_close_in_C": r["qc"]["s7_close_in_C"],
             "downtime_flag": flag[(r["bot"], r["day"])]} for r in results for point in r["qc"]["fee_curve"]]
    tables.append(Table("qc_fee_curve", title("Q-C: P&L net of a maker fee f on the fee grid"), pd.DataFrame(rows), [
        "P_pre - f F and P_post - f (F + C), USDT; f in %. s7_close_in_C: a §7 close inside the day is part of C "
        "(see the next table).",
    ]))

    rows = []
    for r in results:
        qc = r["qc"]
        rows.append({"bot": r["bot"], "day": r["day"], "F_usdt": qc["F_usdt"], "C_usdt": qc["C_usdt"],
                     "C_day_close_usdt": qc["C_day_close_usdt"], "P_pre_usdt": qc["P_pre_usdt"],
                     "P_post_usdt": qc["P_post_usdt"],
                     "f_star_pre_pct": qc["f_star_pre_pct"], "f_star_post_pct": qc["f_star_post_pct"],
                     "paper_fees_limit_usdt": qc["paper_fees_limit_usdt"],
                     "paper_fees_close_usdt": qc["paper_fees_close_usdt"],
                     "paper_fees_s7_close_usdt": qc["paper_fees_s7_close_usdt"],
                     "paper_fees_total_usdt": qc["paper_fees_total_usdt"], "close_cost_usdt": qc["close_cost_usdt"],
                     "close_side": qc["close_side"], "close_amount_doge": qc["close_amount_doge"],
                     "close_price": ";".join(qc["close_prices"]), "close_fill_utc": ";".join(qc["close_utc"]),
                     "s7_close_fills": qc["s7_close_fills"], "s7_close_purposes": qc["s7_close_purposes"],
                     "s7_close_notional_usdt": qc["s7_close_notional_usdt"],
                     "s7_close_cost_usdt": qc["s7_close_cost_usdt"],
                     "close_fills_unknown_purpose": qc["close_fills_unknown_purpose"],
                     "flat_at_day_end": qc["flat_at_day_end"],
                     "last_step_after_close_fills": qc["last_step_after_close_fills"],
                     "P_post_last_step_usdt": qc["P_post_last_step_usdt"],
                     "limit_fills_after_last_window_step": qc["limit_fills_after_last_window_step"],
                     "s7_close_in_C": qc["s7_close_in_C"], "downtime_flag": flag[(r["bot"], r["day"])]})
    tables.append(Table("qc_break_even", title("Q-C: traded notional, break-even fees, paper fees and the close"),
                        pd.DataFrame(rows), [
        "F: limit fills; C: every market close of the day at its fill price rounded to the tick: the day's close (the "
        "23:59:50 close, or the close of a position carried past 00:00; C_day_close_usdt) and any §7 close (halt or "
        "drawdown) inside the day, which s7_close_in_C flags (the author's decision of 2026-10-24); f* = P / notional "
        "in % (n/a for a zero notional); P_post: W after the day's last fill (its cash when the day ends flat, also if "
        "no step follows the close; flat_at_day_end); the paper fees are the ledger's 0.1 % (USDT fees plus DOGE fees "
        "times the fill price); a close's cost is its half-spread against the mid at its submission plus its fee.",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **item, "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results for item in r["s7"]["limits"]]
    tables.append(Table("s7_afterwards", title("§7 afterwards: when each loss limit would have triggered"),
                        pd.DataFrame(rows), [
        "The realized path of every step of the day; the notional at the cap is 1,000 DOGE times the day's first mid; "
        "net equity is W minus f times the notional of the limit fills up to the step.",
    ]))

    columns = ["uptime_pct", "downtime_min", "gaps_over_30s", "longest_gap_s", "longest_step_gap_s",
               "container_restarts", "fresh_starts", "stops", "ws_reconnects", "ws_errors", "resubscriptions",
               "stale_book_pauses", "stale_book_pause_s", "quoting_paused_min", "rejects", "order_failed_rows",
               "new_orders", "max_new_orders_10s", "rate_skips", "not_quoting_skips", "cancels", "cancelled",
               "s7_triggers", "clock_checks", "clock_check_errors", "max_abs_step_offset_s", "max_abs_wall_offset_s",
               "max_abs_residual_doge"]
    rows = [{"bot": r["bot"], "day": r["day"], **{c: r["qd"][c] for c in columns},
             "downtime_flag": flag[(r["bot"], r["day"])]} for r in results]
    tables.append(Table("qd_operations", title("Q-D: operation within the exchange rules"), pd.DataFrame(rows), [
        "Downtime: step gaps over 30 s (from 00:00 and to 24:00 included); uptime = 1 - downtime / 24 h. New orders: "
        "every submit (quotes and closes) against 100 per rolling 10 s and 200,000 per day; cancels: the bot's cancel "
        "requests. Clock offsets from the hourly checks against Binance (step clock and wall clock).",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **gap, "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results for gap in r["qd"]["gaps"]]
    tables.append(Table("qd_gaps", title("Q-D: step gaps over 30 s"),
                        _keyed(rows, ["start_utc", "end_utc", "seconds", "in_network_window", "downtime_flag"])))

    rows = []
    for r in results:
        counters, rate = r["qd"]["bot_counters"] or {}, r["qd"]["bot_rate"] or {}
        rows.append({"bot": r["bot"], "day": r["day"], "new_orders": r["qd"]["new_orders"],
                     "bot_day_orders": rate.get("day_orders"), "max_new_orders_10s": r["qd"]["max_new_orders_10s"],
                     "bot_max_in_window": rate.get("max_in_window"), "cancels": r["qd"]["cancels"],
                     "bot_cancels": rate.get("cancels"), "bot_rate_skips": None if not rate else
                     rate.get("skipped_window", 0) + rate.get("skipped_day", 0),
                     "max_abs_residual_doge": r["qd"]["max_abs_residual_doge"],
                     "bot_max_abs_residual_doge": counters.get("max_abs_residual_doge"),
                     "bot_reconnects": counters.get("reconnects"), "bot_triggers": json.dumps(counters.get("triggers")),
                     "downtime_flag": flag[(r["bot"], r["day"])]})
    tables.append(Table("qd_bot_counters", title("Q-D: the recomputed counts beside the bot's own day_end counters"),
                        pd.DataFrame(rows), [
        "A check: the bot's counters from its day_end health row (the rolling 10 s window and the daily counts of its "
        "RateLimiter; bot_reconnects counts websocket closes and errors). n/a when the day has no day_end row.",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **pause, "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results for pause in r["qd"]["pauses"]]
    tables.append(Table("qd_pauses", title("Q-D: §7 quoting pauses and halts"),
                        _keyed(rows, ["trigger", "start_utc", "end_utc", "seconds", "downtime_flag"])))

    rows = [{"bot": r["bot"], "day": r["day"], **r["exploratory"], "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results]
    tables.append(Table("qd_exploratory", title("Q-D, exploratory: trade fills at the price, book and clock events"),
                        pd.DataFrame(rows), [
        "Exploratory (not in PROTOCOL §9). Among the opposite-side taker trades logged after a limit fill's order was "
        "posted and in the 0.5 s up to the fill, in whole ticks: through, one printed strictly through the order price; "
        "at_price (= engine_through + not_engine_through), none through but one exactly at it; no_trade, none reached "
        "it (a book cross at a clock tick, or a trigger outside the window). The paper engine fills on a print strictly "
        "through the order, comparing the Decimal price with the float print exactly, so an at-tick print can trigger "
        "a fill (engine_through) or cannot (not_engine_through). Book re-syncs: applied REST snapshots "
        "(clarification h); crossed or locked books from the bot's counters (its health rows stop at 100 a day); "
        "round-away quotes: clarification a.",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **item, "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results for item in r["fill_classes"]]
    tables.append(Table("qb_fill_classes", title("Q-B by fill class, exploratory: edge, markouts and fills over E"),
                        pd.DataFrame(rows), [
        "Exploratory (WO-HB3b; not in PROTOCOL §9). The limit fills by the at-price method's class (the table above), "
        "then all of them and the through + no_trade fills together; Q-B's definitions per class (ticks; R(h) over "
        "the fills whose markout at h is kept). fills_over_E: the window's fills of the row over E; for through + "
        "no_trade a sensitivity to the paper engine's Decimal-against-float comparison. Hypothesis, reported and not "
        "tuned on: delta_fill near +1/2 tick for at-price fills and <= -1/2 tick for no_trade (book-cross) fills; a "
        "positive delta_fill elsewhere would reflect the depth stream (100 ms batches) lagging the trade stream.",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], "model": r["model"], **r["pnl_split"],
             "downtime_flag": flag[(r["bot"], r["day"])]} for r in results]
    tables.append(Table("pnl_split", title("P&L split, exploratory: P_pre into fill edge and inventory"),
                        pd.DataFrame(rows), [
        "Exploratory (WO-HB3d; not in PROTOCOL §9). USDT for 100-DOGE orders, over the trading window: P_pre = fill "
        "edge + inventory (+ the edge of a market fill inside the window, a §7 close; none is expected). inventory = "
        "Σ over the window's steps i >= 1 of 100 q_{i-1} (S_i - S_{i-1}); fill edge = Σ over the limit fills of "
        "s a (S - p), with S the mid of the first step whose wealth includes the fill: the first step with ts >= the "
        "fill's ts (each step marks W over every fill booked before it, and a fill's ts never exceeds that step's; a "
        "fill at a step's own ts is taken to be in that step, which fails only while the step clock is held, as at a "
        "restart floor; fills_at_step_ts counts them). S is that step's mid, not mid_at_fill (Q-B's delta_fill). A day "
        "starts flat, so a fill before its "
        "first step enters step 0; the 23:59:50 close and a limit fill after the last window step are left out. "
        "mean_fill_edge_ticks: the fill edge per DOGE in ticks. Checks (within 1e-9 USDT): the parts' sum minus "
        "P_pre, and the largest |W_i - W_{i-1} - step i's parts| over the window; position_mismatch_steps: window "
        "steps whose 100 q differs from the cumulative fills (expected 0).",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **part, "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results for part in r["network"]]
    tables.append(Table("qd_network_window", title("Q-D, exploratory: clock checks and gaps inside and outside the "
                                                   "mobile-hotspot window"), pd.DataFrame(rows), [
        f"Exploratory: the window {NETWORK_WINDOW[0]} to {NETWORK_WINDOW[1]} (iPhone hotspot, RUN_LOG "
        "2026-10-23T05:49:24Z); a gap belongs to the part where it starts.",
    ]))

    rows = []
    bots = sorted({r["bot"] for r in results})
    days = sorted({r["day"] for r in results})
    for stamp, text in incidents:
        named = [bot for bot in bots if bot in text]
        for day in entry_days(stamp, text, days):
            for bot in (named if len(named) == 1 else bots):
                rows.append({"bot": bot, "day": day, "written_utc": stamp, "entry": text,
                             "downtime_flag": flag.get((bot, day))})
    tables.append(Table("runlog_entries", title("RUN_LOG.md entries about the run's days"),
                        _keyed(rows, ["written_utc", "entry", "downtime_flag"]), [
        f"Every RUN_LOG.md entry written from {RUN_WINDOW[0]} on, listed under each analysed day whose date appears "
        "in its stamp or text (an entry about a day is written after it); an entry that names one bot only is listed "
        "for it, any other for both. The CSV and results.json hold the full text; this table cuts it at 500 characters.",
    ]))

    tables.append(Table("data_description", title("Data description: the snapshot files"), pd.DataFrame(data_rows), [
        "Through the frozen loader (load_single_day_calibration_arrays); sigma by the frozen estimator "
        "(_sigma_from_day, 600 s blocks), USDT per sqrt(s); the share of rows with a 1-tick spread.",
    ]))

    rows = [{"bot": r["bot"], "day": r["day"], **item, "downtime_flag": flag[(r["bot"], r["day"])]}
            for r in results for item in r.get("feature_shift", [])]
    tables.append(Table("feature_shift", title("Feature shift of B3's normalized observation"),
                        _keyed(rows, ["feature", "mean_z", "share_abs_z_gt_3", "share_abs_z_clipped",
                                      "downtime_flag"]), [
        "z = the frozen VecNormalize normalization (clip 10) of each feature at each step of the trading window "
        "(the steps file's z columns); shares of steps.",
    ]))

    if shadow is not None:
        tables.append(Table("shadow_vs_paper", title("Shadow replay against paper: the Q-A quantities and their PIT"),
                            _keyed(shadow, ["model", "metric", "role", "paper", "shadow_mean", "shadow_p05",
                                            "shadow_p95", "below", "ties", "paths", "pit", "pit_flag",
                                            "shadow_share_q_le_minus_8", "downtime_flag"]), [
            "PIT = (#{m < m_paper} + ½ #{m = m_paper}) / 40 over the shadow paths (fill seeds 42 and 43 x 20 rollouts) "
            "of the frozen simulator on that bot's own snapshot file; flagged below 0.05 or above 0.95 (descriptive). "
            "PROTOCOL §9's PIT set is P_pre/100, MaxDD/100 and mean |q|; the near-cap share's PIT is exploratory; the "
            "shadow's q is at the end of the file, so it gets no PIT. P_pre and MaxDD divided by 100 (USDT per DOGE "
            "unit, as the replay). The paper window ends at 23:59:50, the replay runs to the end of the file. 5-95 %: "
            "numpy percentiles (linear interpolation).",
        ]))
    return tables


# --- outputs --------------------------------------------------------------------------------------------------------


MARKDOWN_TEXT_LIMIT = 500  # characters of a text cell in forward_test_tables.md; the CSVs keep the full text


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    text = str(value)
    if len(text) > MARKDOWN_TEXT_LIMIT:
        text = text[:MARKDOWN_TEXT_LIMIT] + "…"
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows) columns: " + ", ".join(frame.columns) + "\n"
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (Path, Decimal)):
        return str(value)
    if value is pd.NA:
        return None
    return value


def _tex(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def appendix_table(results: list[dict[str, Any]], shadow: list[dict[str, Any]] | None, prefix: str) -> str:
    """The booktabs table of the thesis appendix: one column per bot and day (B0 then B3 per day)."""

    order = sorted(results, key=lambda r: (r["day"], r["model"]))
    pits = {(row["bot"], row["day"], row["metric"]): row["pit"] for row in shadow or []}

    def pit_cell(r: dict[str, Any], metric: str) -> str:
        return _tex(pits.get((r["bot"], r["day"], metric)), ".3f")

    def date_label(day: str) -> str:
        return Date.fromisoformat(day).strftime("%d %b").lstrip("0")

    rows = [
        ("Limit fills (paper / $E$)", lambda r: f"{r['qb']['limit_fills_in_window']} / {_tex(r['qb']['E'], '.1f')} "
                                                f"({_tex(r['qb']['fills_over_E'], '.3f')})"),
        ("Mean $|q|$ (units)", lambda r: _tex(r["qa"]["mean_abs_q"], ".3f")),
        ("Near-cap share", lambda r: _tex(r["qa"]["near_cap_share"], ".3f")),
        ("$q$ before close (units)", lambda r: f"{r['qa']['q_pre']:+d}"),
        ("$P_{\\mathrm{pre}}$ (USDT)", lambda r: _tex(r["qa"]["P_pre_usdt"], ".4f")),
        ("MaxDD before close (USDT)", lambda r: _tex(r["qa"]["maxdd_pre_usdt"], ".4f")),
        ("$f^{*}_{\\mathrm{pre}}$ (\\%)", lambda r: _tex(r["qc"]["f_star_pre_pct"], ".4f")),
        ("$R(10\\,\\mathrm{s})$", lambda r: _tex(r["qb"]["R_10s"], ".3f")),
        ("PIT $P_{\\mathrm{pre}}$", lambda r: pit_cell(r, "P_pre/100")),
        ("PIT MaxDD", lambda r: pit_cell(r, "MaxDD/100")),
        ("PIT mean $|q|$", lambda r: pit_cell(r, "mean |q|")),
        ("Downtime (min)", lambda r: _tex(r["qd"]["downtime_min"], ".2f")),
    ]
    header = " & ".join([""] + [f"{r['model']} {date_label(r['day'])}" for r in order]) + r" \\"
    body = [" & ".join([label] + [cell(r) for r in order]) + r" \\" for label, cell in rows]
    caption = (f"{prefix}Hummingbot forward validation (paper trading, binance\\_paper\\_trade, DOGE-USDT), per bot and "
               f"prospective day ({LABEL}). USDT figures are for 100-DOGE orders (divide by 100 to compare with the "
               "replay); the trading window ends at 23:59:50 UTC; $E$: expected fills under the calibrated execution "
               "model; $f^{*}_{\\mathrm{pre}}$: the break-even maker fee before the close; $R(10\\,\\mathrm{s})$: "
               "retention of the posted edge at 10 s; PIT: the paper value's rank among the 40 shadow-replay paths "
               "(n/a without the shadow replay).")
    flagged = [f"{r['model']} {date_label(r['day'])}" for r in order if r["downtime_flag"]]
    if flagged:
        caption += " More than 1 h of downtime: " + ", ".join(flagged) + "."
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        r"\label{tab:hb-forward-validation}",
        r"\begin{tabular}{l" + "r" * len(order) + "}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def inventory_figure(results: list[dict[str, Any]], days: list[str], out_dir: Path, prefix: str) -> list[Path]:
    """q against UTC time, one panel per day (docs/DESIGN_SYSTEM.md): B0 #333333 solid, B3 #009E73 solid with sparse
    markers, dotted lines at |q| = 8, the cap at |q| = 10, the 23:59:50 close dashed, the hotspot window shaded."""

    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    style = {
        "B0": dict(color="#333333", linestyle="-", marker=None, label="B0 — A–S"),
        "B3": dict(color="#009E73", linestyle="-", marker="o", label="B3 — CVaR-Constrained PPO"),
    }
    neutral, shade = "#6B7280", "#E5E7EB"
    hotspot = iso_us(NETWORK_WINDOW[0]), iso_us(NETWORK_WINDOW[1])
    rc = {"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
          "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42}
    data_rows = []
    with matplotlib.rc_context(rc):
        figure = Figure(figsize=(6.5, 2.2 + 1.9 * len(days)), layout="constrained")
        axes = figure.subplots(len(days), 1, squeeze=False)[:, 0]
        for panel, (axis, day) in enumerate(zip(axes, days)):
            start, close, end = day_bounds(day)
            low, high = max(hotspot[0], start), min(hotspot[1], end)
            if high > low:
                band = (low - start) / 3.6e9, (high - start) / 3.6e9
                axis.axvspan(*band, color=shade, alpha=0.9, lw=0, zorder=0)  # named in the legend only (WO-HB3d)
            for level in (-NEAR_CAP, NEAR_CAP):
                axis.axhline(level, color=neutral, linestyle=":", linewidth=0.9, zorder=1)
            for level in (-Q_MAX, Q_MAX):
                axis.axhline(level, color=neutral, linestyle="-.", linewidth=0.6, zorder=1)
            axis.axhline(0, color=neutral, linewidth=0.6, zorder=1)
            axis.axvline((close - start) / 3.6e9, color=neutral, linestyle="--", linewidth=0.9, zorder=1)
            for r in sorted((r for r in results if r["day"] == day), key=lambda r: r["model"]):
                hours = (np.asarray(r["series"]["ts_us"], dtype=np.int64) - start) / 3.6e9
                q = np.asarray(r["series"]["q"])
                spec = style[r["model"]]
                axis.plot(hours, q, drawstyle="steps-post", color=spec["color"], linestyle=spec["linestyle"],
                          linewidth=1.1, zorder=3)
                if spec["marker"]:  # sparse markers, evenly spaced in time, on the step function's value
                    marks = np.arange(0.75, 24.0, 1.5)
                    marks = marks[(marks >= hours[0]) & (marks <= hours[-1])]
                    values = q[np.searchsorted(hours, marks, side="right") - 1]
                    axis.plot(marks, values, linestyle="none", marker=spec["marker"], markersize=3.5,
                              color=spec["color"], zorder=4)
                data_rows += [{"bot": r["bot"], "day": day, "ts_us": int(t), "utc": utc_iso(int(t)),
                               "hours": float(h), "q": int(v)} for t, h, v in zip(r["series"]["ts_us"], hours, q)]
            axis.set_xlim(-0.2, 24.3)  # the frame clears 00:00 and the 23:59:50 line
            axis.set_xticks(range(0, 25, 3))
            axis.set_ylim(-Q_MAX - 0.8, Q_MAX + 0.8)
            axis.set_yticks([-10, -8, -4, 0, 4, 8, 10])
            axis.set_ylabel("Inventory q\n(units of 100 DOGE)")
            axis.set_title(f"({chr(ord('a') + panel)}) {day}", loc="left")
            axis.grid(axis="y", color=shade, linewidth=0.5)
        axes[-1].set_xlabel("Time (UTC, h)")
        handles = [Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], markersize=3.5,
                          label=s["label"]) for s in style.values()]
        handles += [Line2D([], [], color=neutral, linestyle=":", label="|q| = 8 (near cap)"),
                    Line2D([], [], color=neutral, linestyle="-.", label="|q| = 10 (cap)"),
                    Line2D([], [], color=neutral, linestyle="--", label="23:59:50 close"),
                    Patch(color=shade, label="mobile hotspot (exploratory)")]
        figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
        figure.suptitle(f"{prefix}Inventory per bot and prospective day\n({LABEL})", fontsize=9)
        paths = [out_dir / "fig_forward_inventory.pdf", out_dir / "fig_forward_inventory.png"]
        figure.savefig(paths[0], metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(paths[1], dpi=300, metadata={"Software": None})
    data_path = out_dir / "fig_forward_inventory_data.csv"
    pd.DataFrame(data_rows, columns=["bot", "day", "ts_us", "utc", "hours", "q"]).to_csv(
        data_path, index=False, lineterminator="\n")
    return paths + [data_path]


def figure_caption(prefix: str) -> str:
    return (f"{prefix}Inventory q (units of 100 DOGE; cap 10) of B0 (A–S, charcoal) and B3 (CVaR-constrained PPO, green "
            "with markers) against UTC time on each prospective day of the Hummingbot forward validation (paper "
            "trading on binance_paper_trade, DOGE-USDT): the fill ledger's q at every step, as a step function. Dotted: "
            "the near-cap level |q| = 8; dash-dot: the cap; dashed: the 23:59:50 close; grey band: the mobile-hotspot "
            f"network window (exploratory). One bot per line, one day per panel; {LABEL}.")


# --- the command ---------------------------------------------------------------------------------------------------


def load_shadow(shadow_dir: Path, days: dict[tuple[str, str], DayFiles], params: dict[str, Any]
                ) -> dict[tuple[str, str], dict[str, Any]]:
    """The shadow replay of every bot and day (hummingbot_adapter.shadow_replay), refused unless each episode file and
    its JSON exist and record that bot's own snapshots file, these params, the pipeline's seeds and rollouts, and the
    episode file's own SHA-256, with 40 distinct paths for the day."""

    shadow = {}
    for (bot, day), files in days.items():
        name = SHADOW_NAME[files.model]
        episodes_path = shadow_dir / f"shadow_{name}_{day}_episodes.csv"
        record_path = shadow_dir / f"shadow_{name}_{day}.json"
        for path in (episodes_path, record_path):
            if not path.is_file():
                raise AnalysisRefused(f"no shadow replay file {path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        problems = []
        if record.get("model") != files.model or record.get("date") != day:
            problems.append(f"model/date {record.get('model')}/{record.get('date')}")
        if record.get("inputs", {}).get("day_file", {}).get("sha256") != files.sha256["snapshots"]:
            problems.append("not this bot's snapshots file")
        if record.get("inputs", {}).get("params", {}).get("sha256") != params["sha256"]:
            problems.append("other calibration parameters than --params")
        if record.get("pipeline_settings") is not True:
            problems.append("not the pipeline's fill seeds and rollouts")
        episodes_sha256 = file_sha256(episodes_path)
        if record.get("episodes", {}).get("sha256") != episodes_sha256:
            problems.append("the episode file's SHA-256 differs from its record")
        if problems:
            raise AnalysisRefused(f"{record_path}: {'; '.join(problems)}")
        episodes = pd.read_csv(episodes_path, dtype={"Day": str}, float_precision="round_trip")
        episodes = episodes[episodes["Day"] == day]
        if len(episodes) != SHADOW_PATHS or episodes.duplicated(["Fill Seed", "Rollout"]).any():
            raise AnalysisRefused(f"{episodes_path}: {len(episodes)} distinct paths for {day}, expected {SHADOW_PATHS}")
        shadow[(bot, day)] = {"episodes": episodes, "files": {episodes_path.name: episodes_sha256,
                                                              record_path.name: file_sha256(record_path)}}
    return shadow


PIT_ROLE, EXPLORATORY_ROLE, DESCRIPTIVE_ROLE = "PIT (PROTOCOL §9)", "PIT, exploratory", "descriptive, no PIT"


def shadow_rows(results: list[dict[str, Any]], shadow: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """The Q-A quantities of the paper run beside those of the 40 shadow paths of that bot and day (PROTOCOL §9 Q-A:
    "in the paper run and in the shadow replay"): the PIT of P_pre/100, MaxDD/100 and mean |q| (§9's set); the
    near-cap share with an exploratory PIT; the shadow's q at the end of the file (it does not close at 23:59:50)
    beside the paper's q before the close, with the share of paths at q <= -8 and no PIT."""

    rows = []
    for r in results:
        episodes = shadow[(r["bot"], r["day"])]["episodes"]
        for metric, column, paper, role in (
            ("P_pre/100", "Final PnL", r["qa"]["P_pre_usdt"] / UNIT_DOGE, PIT_ROLE),
            ("MaxDD/100", "Max DD", r["qa"]["maxdd_pre_usdt"] / UNIT_DOGE, PIT_ROLE),
            ("mean |q|", "Mean |q|", r["qa"]["mean_abs_q"], PIT_ROLE),
            ("near-cap share", "Near Cap Fraction", r["qa"]["near_cap_share"], EXPLORATORY_ROLE),
            ("q (paper: before the close; shadow: end of file)", "Terminal q", float(r["qa"]["q_pre"]),
             DESCRIPTIVE_ROLE),
        ):
            values = episodes[column].to_numpy(dtype=np.float64)
            value = None if role == DESCRIPTIVE_ROLE else pit(values, paper)
            rows.append({"bot": r["bot"], "day": r["day"], "model": r["model"], "metric": metric, "role": role,
                         "paper": paper, "shadow_mean": float(values.mean()),
                         "shadow_p05": float(np.percentile(values, 5)), "shadow_p95": float(np.percentile(values, 95)),
                         "below": int(np.sum(values < paper)), "ties": int(np.sum(values == paper)),
                         "paths": int(values.size), "pit": value,
                         "pit_flag": None if value is None else value < PIT_FLAGS[0] or value > PIT_FLAGS[1],
                         "shadow_share_q_le_minus_8": float(np.mean(values <= -NEAR_CAP))
                         if column == "Terminal q" else None,
                         "downtime_flag": r["downtime_flag"]})
    return rows


def write_outputs(target: Path, tables: list[Table], results: list[dict[str, Any]], shadow: list[dict[str, Any]] | None,
                  days: list[str], record: dict[str, Any], prefix: str) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    lines = [f"# {prefix}Hummingbot forward validation: analysis tables ({LABEL})", ""]
    lines += [f"- {key}: {value}" for key, value in record["summary"].items()] + [""]
    for table in tables:
        lines += [f"## {table.title}", ""] + [f"_{note}_" for note in table.notes] + [""]
        lines += [markdown_table(table.frame), ""]
        with open(target / f"{table.name}.csv", "x", encoding="utf-8", newline="") as handle:
            table.frame.to_csv(handle, index=False, lineterminator="\n")
    figure_files = inventory_figure(results, days, target, prefix)
    lines += [f"## {prefix}Figure: fig_forward_inventory ({LABEL})", "", figure_caption(prefix), ""]
    with open(target / "forward_test_tables.md", "x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    with open(target / "appendix_table.tex", "x", encoding="utf-8", newline="\n") as handle:
        handle.write(appendix_table(results, shadow, prefix))
    record["tables"] = {table.name: {"title": table.title, "notes": table.notes,
                                     "rows": json_safe(table.frame.to_dict(orient="records"))} for table in tables}
    record["figure"] = {"files": [path.name for path in figure_files], "caption": figure_caption(prefix)}
    record["per_bot_day"] = json_safe([{k: v for k, v in r.items() if k != "series"} for r in results])
    with open(target / "results.json", "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(json_safe(record), indent=2) + "\n")
    names = sorted(path.name for path in target.iterdir() if path.is_file() and path.name != SUMS_NAME)
    sums = [f"{file_sha256(target / name)}  {name}" for name in names]
    with open(target / SUMS_NAME, "x", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(line + "\n" for line in sums))
    return sums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--days", required=True, nargs="+")
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--calibration-day-file", required=True, type=Path)
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shadow-dir", type=Path)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args(argv)
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        days = []
        for text in args.days:
            try:
                day = Date.fromisoformat(text).isoformat()
            except ValueError as exc:
                raise AnalysisRefused(f"--days: {text!r} is not YYYY-MM-DD") from exc
            if day in days:
                raise AnalysisRefused(f"--days repeats {day}")
            if day not in RUN_DAYS and not args.dev:
                raise AnalysisRefused(f"{day} is outside the main run {RUN_DAYS} (warm-up or later); use --dev")
            days.append(day)
        prefix = f"{DEV_PREFIX}: " if args.dev else ""
        target = args.out_dir / "dev" if args.dev else args.out_dir
        for root, what in ((REPO_ROOT, "the repository"), (args.data_root, "the data root")):
            if is_inside(target, root):
                raise AnalysisRefused(f"--out-dir {args.out_dir} lies inside {what}")
        if target.exists() and any(target.iterdir()):
            raise AnalysisRefused(f"{target} is not empty; outputs are never overwritten")
        for path, what in ((args.calibration_day_file, "--calibration-day-file"), (args.run_log, "--run-log")):
            if not path.is_file():
                raise AnalysisRefused(f"{what}: no such file {path}")
        if args.shadow_dir is not None and not args.shadow_dir.is_dir():
            raise AnalysisRefused(f"--shadow-dir: no such folder {args.shadow_dir}")
        params = read_params(args.params)
        verified = {(bot, day): verify_day(args.data_root, bot, day) for day in days for bot in MODEL_OF}
        print(f"verified {len(verified)} bot-days: " + ", ".join(f"{b} {d} ({v.sums_sha256[:12]})"
                                                                for (b, d), v in verified.items()))
        shadow_inputs = None if args.shadow_dir is None else load_shadow(args.shadow_dir, verified, params)
        results, data_rows = [], []
        for (bot, day), files in verified.items():
            data = read_day(files)
            results.append(analyse_day(data, params["A"], params["kappa"]))
            del data
            data_rows.append({"bot": bot, "day": day, "file": files.paths["snapshots"].name,
                              **describe_snapshots(files.paths["snapshots"]),
                              "downtime_flag": results[-1]["downtime_flag"]})
            print(f"analysed {bot} {day}")
        calibration_name = args.calibration_day_file.name
        calibration_day = DATE_IN_NAME.findall(calibration_name)
        calibration_sha = file_sha256(args.calibration_day_file)
        calibration_recorded = calibration_sha in params["recorded_input_sha256"]
        data_rows.append({"bot": "tardis_calibration", "day": calibration_day[0] if calibration_day else "n/a",
                          "file": calibration_name, **describe_snapshots(args.calibration_day_file),
                          "downtime_flag": None})
        shadow = None if shadow_inputs is None else shadow_rows(results, shadow_inputs)
        incidents = run_log_entries(args.run_log)
        tables = build_tables(results, data_rows, incidents, shadow, prefix)
        git_head = git_output(REPO_ROOT, "rev-parse", "HEAD")
        git_status = git_output(REPO_ROOT, "status", "--porcelain", "--", "hummingbot_adapter", "cvar_mm")
        record = {
            "label": LABEL,
            "dev": args.dev,
            "created_utc": created_utc,
            "command": sys.argv if argv is None else ["hummingbot_adapter.analysis", *argv],
            "summary": {
                "days": ", ".join(days),
                "params": f"{params['path']} (SHA-256 {params['sha256']}; pre-registered hb_calibration.json: "
                          f"{'yes' if params['pre_registered'] else 'NO'})",
                "calibration day file": f"{args.calibration_day_file} (SHA-256 {calibration_sha}; recorded in "
                                        f"--params: {'yes' if calibration_recorded else 'NO'})",
                "run log": f"{args.run_log} (SHA-256 {file_sha256(args.run_log)})",
                "day folders": "; ".join(f"{b} {d}: SHA256SUMS {v.sums_sha256}" for (b, d), v in verified.items()),
                "shadow replay": "none (PIT n/a)" if args.shadow_dir is None else str(args.shadow_dir),
                "code": f"{git_head.stdout.strip()} (uncommitted in hummingbot_adapter or cvar_mm: "
                        f"{'; '.join(git_status.stdout.split(chr(10))).strip('; ') or 'none'})",
            },
            "inputs": {
                "params": params,
                "calibration_day_file": {"path": str(args.calibration_day_file), "sha256": calibration_sha},
                "run_log": {"path": str(args.run_log), "sha256": file_sha256(args.run_log)},
                "days": {f"{b} {d}": {"folder": str(v.folder), "sums_sha256": v.sums_sha256, "files": v.sha256}
                         for (b, d), v in verified.items()},
                "shadow_dir": None if args.shadow_dir is None else str(args.shadow_dir),
                "shadow_files": None if shadow_inputs is None else {
                    f"{b} {d}": item["files"] for (b, d), item in shadow_inputs.items()},
            },
            "readings": {
                "trading_window": "steps with ts < 23:59:50 UTC; before the close = its last step",
                "time_weights": "fast_simulate's: step i >= 1 carries (ts_i - ts_{i-1}) / 1e6 with its own q; "
                                "step 0, dt = 0 and steps from 23:59:50 carry 0",
                "q_b_fills": "limit fills only (close == 0)",
                "markouts": "S(t) = mid of the last step with ts <= t; a markout beyond the file's last step is "
                            "dropped and counted; R(h) over the fills whose markout at h is kept",
                "expected_fills": "the orders resting after step i-1 at their distance from step i-1's mid meet dt_i",
                "q_c": "F = the limit fills; C = every market close, the day's close and any §7 close inside the day "
                       "(the author's decision of 2026-10-24, WO-HB3b; flag s7_close_in_C); P_post = W after the "
                       "day's last fill",
                "fill_classes": "exploratory: Q-B per class of the at-price method, all fills, and through + "
                                "no_trade with its fills over E (WO-HB3b)",
                "pnl_split": "exploratory: P_pre = fill edge + inventory over the window; a fill's S = the mid of "
                             "the first step with ts >= its ts (the first step whose wealth includes it, unless the "
                             "step clock is held; fills_at_step_ts counts the ties); a market fill inside the window "
                             "is a third part (WO-HB3d)",
                "s7_afterwards": "every step of the day (the realized path)",
                "shadow_replay": "the whole snapshot file; the paper window ends at 23:59:50",
                "exploratory": "at-price trade fills (0.5 s window), the hotspot split, book and clock events",
            },
            "git_commit": git_head.stdout.strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        }
        sums = write_outputs(target, tables, results, shadow, days, record, prefix)
    except AnalysisRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(sums)} files to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
