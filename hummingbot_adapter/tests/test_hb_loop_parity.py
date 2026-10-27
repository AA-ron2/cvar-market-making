"""B4: the offline loop parity of WO-HB2. The bot's step function (bot_core.ModelStepper) runs over the first
20,000 Jan 7 rows with the Jan 1-6 calibration and the replay's q path injected, as T2/T3 feed the engines:
- B3: observations, z-scores, NumpyPolicy actions and pre-rounding quotes against the frozen replay (T2, T3);
- B0: quotes against a direct avellaneda_stoikov_quotes call at fast_simulate's tau (T4);
- the bot's clock: execution.dt_seconds on the file's integer timestamps against the frozen loader's dt;
- the live book path: book_row, book_mid and snapshot_fields on the top-25 books of the same Jan 7 lines.
At every step the rounded quotes (clarification a) must also become valid 100-DOGE orders (QuoteEngine.to_orders).
HB_PARITY_EXPORT=<file.npz> also exports the B3 loop's inputs and expected values, for the same loop inside the
Hummingbot container (numpy and pandas versions differ there).
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from hummingbot_adapter.bot_config import file_sha256, protocol_default
from hummingbot_adapter.bot_core import ModelStepper, book_mid, book_row
from hummingbot_adapter.execution import dt_seconds
from hummingbot_adapter.features import FeatureEngine
from hummingbot_adapter.quotes import QuoteEngine
from hummingbot_adapter.telemetry import SNAPSHOT_HEADER, SNAPSHOT_LEVELS, snapshot_fields
from hummingbot_adapter.tests.support import B0_GAMMA
from cvar_mm.gym.formula_as import avellaneda_stoikov_quotes

ROWS = 20_000  # the WO's offline loop: the first 20,000 rows of the Jan 7 file
EXPORT_ENV = "HB_PARITY_EXPORT"
TICK = 1e-5
GRID = Decimal("0.00001")
GRID_NOISE_TICKS = 1e-9  # float noise of a frozen k * tick product; a half-tick off the grid is far above it
SB3_TOLERANCE = 1e-6  # PROTOCOL §5.2


def _engine(settings) -> QuoteEngine:
    gamma_min, gamma_max = settings["gamma_range"]
    return QuoteEngine(
        sigma=settings["sigma"], kappa=settings["kappa"], tick_size=settings["tick_size"],
        gamma_min=gamma_min, gamma_max=gamma_max, skew_ticks_max=settings["skew_ticks_max"],
    )


def _features() -> FeatureEngine:
    return FeatureEngine(tick_size=1e-5, q_max=10, window=100, imbalance_depth=5)


def _row(replay, k: int) -> dict:
    return dict(zip(replay.row_columns, replay.row_values[k]))


def _bits(value) -> int:
    return int(np.float64(value).view(np.uint64))


def _off_grid_ticks(price: float) -> float:
    ticks = price / TICK
    return abs(ticks - round(ticks))


def _rounding_problems(quotes: dict, rounded, mid: float) -> list[str]:
    """Clarification a, checked from its statement rather than its code: each rounded price is on the 0.00001
    grid, the bid at or below its target and the ask at or above it (away from the mid), each by less than one
    tick, and a side's flag is set exactly when its price moved by more than float noise."""

    problems = []
    for side, target, price, flag, sign in (
        ("bid", quotes["bid"], rounded.bid, rounded.bid_rounded, 1.0),
        ("ask", quotes["ask"], rounded.ask, rounded.ask_rounded, -1.0),
    ):
        move_ticks = sign * (target - price) / TICK  # >= 0 when the price moved away from the mid
        if _off_grid_ticks(price) > GRID_NOISE_TICKS:
            problems.append(f"{side} {price!r} is off the grid")
        if not -GRID_NOISE_TICKS <= move_ticks < 1.0:
            problems.append(f"{side} moved {move_ticks!r} ticks from {target!r} to {price!r}")
        if bool(flag) != (move_ticks > GRID_NOISE_TICKS):
            problems.append(f"{side} flag {flag!r} for a move of {move_ticks!r} ticks")
    if not rounded.bid < mid < rounded.ask:
        problems.append(f"rounded bid {rounded.bid!r} and ask {rounded.ask!r} do not straddle the mid {mid!r}")
    return problems


def _order_problems(engine: QuoteEngine, rounded, q) -> list[str]:
    """The step's next stage (WO-HB2 deliverable 3): the rounded quotes become a bid unless q >= 10 and an ask
    unless q <= -10, 100 DOGE each, at the rounded price as an exact 5-decimal grid price."""

    try:
        orders = engine.to_orders({"bid": rounded.bid, "ask": rounded.ask}, q, 10, 100)
    except ValueError as error:
        return [f"to_orders refused the rounded quotes: {error}"]
    sides = [side for side, allowed in (("buy", q < 10), ("sell", q > -10)) if allowed]
    problems = [] if [order["side"] for order in orders] == sides else [f"sides {orders} at q = {q!r}"]
    for order in orders:
        price = Decimal(order["price"])
        target = rounded.bid if order["side"] == "buy" else rounded.ask
        if price % GRID != 0 or abs(float(price) - target) > GRID_NOISE_TICKS * TICK or order["amount"] != "100":
            problems.append(f"order {order} for the rounded price {target!r}")
    return problems


def _assert_the_bot_builds_these_engines(settings) -> None:
    """BotCore builds its QuoteEngine and FeatureEngine from the PROTOCOL.md defaults in bot_config; they must be
    the replay's, so only sigma and kappa differ live (hb_calibration.json instead of the Jan 1-6 calibration)."""

    assert protocol_default("tick_size") == settings["tick_size"] == TICK
    assert protocol_default("q_max") == settings["q_max"] == 10
    assert protocol_default("feature_window") == settings["feature_window"] == 100
    assert protocol_default("imbalance_depth") == 5
    assert (protocol_default("gamma_min"), protocol_default("gamma_max")) == settings["gamma_range"]
    assert protocol_default("skew_ticks_max") == settings["skew_ticks_max"]


def _export(path: Path, replay, numpy_actions: np.ndarray, policy_path: Path) -> None:
    """The B3 loop's inputs and expected values for the container run (allow_pickle=False readable)."""

    if path.suffix != ".npz" or not path.parent.is_dir():
        pytest.fail(f"{EXPORT_ENV} must name a .npz file in an existing folder, got {str(path)!r}")
    settings = replay.settings
    gamma_min, gamma_max = settings["gamma_range"]
    arrays = {
        "rows": replay.row_values[:ROWS],
        "columns": np.array(replay.row_columns),
        "dt": replay.dt[:ROWS],
        "t": replay.t[:ROWS],
        "T": np.float64(replay.terminal_time),
        "q": replay.q[:ROWS],
        "raw_obs": replay.raw_obs[:ROWS],
        "z": replay.norm_obs[:ROWS],
        "mid": replay.mid[:ROWS],
        "bid": replay.bid[:ROWS],
        "ask": replay.ask[:ROWS],
        "actions": numpy_actions[:ROWS],
        "sb3_actions": replay.actions[:ROWS],
        "sigma": np.float64(settings["sigma"]),
        "kappa": np.float64(settings["kappa"]),
        "tick_size": np.float64(settings["tick_size"]),
        "gamma_min": np.float64(gamma_min),
        "gamma_max": np.float64(gamma_max),
        "skew_ticks_max": np.float64(settings["skew_ticks_max"]),
        "policy_sha256": np.array(file_sha256(policy_path)),
    }
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)
    with np.load(path, allow_pickle=False) as data:
        assert sorted(data.files) == sorted(arrays)
        for key, value in arrays.items():
            assert data[key].dtype == np.asarray(value).dtype and np.array_equal(data[key], value), key
    print(f"\nHB-PARITY loop export: {path}, {ROWS} rows, b3_policy.npz SHA-256 {arrays['policy_sha256']}")


@pytest.fixture(scope="module")
def jan7_head(inputs) -> tuple[tuple[str, ...], list[str]]:
    """The header and the first ROWS + 1 data lines of the Jan 7 file as text (LF line ends, nothing parsed)."""

    lines = []
    with open(inputs["jan7"], encoding="utf-8", newline="") as handle:
        header = handle.readline()
        for _ in range(ROWS + 1):
            line = handle.readline()
            if not line.endswith("\n") or line.endswith("\r\n"):
                raise ValueError(f"{inputs['jan7']}: line {len(lines) + 2} is not a complete LF-terminated line")
            lines.append(line[:-1])
    if not header.endswith("\n") or header.endswith("\r\n"):
        raise ValueError(f"{inputs['jan7']}: the header is not an LF-terminated line")
    return tuple(header[:-1].split(",")), lines


def test_b4_b3_step_function_reproduces_the_replay_over_the_first_20000_jan7_rows(
    jan7_replay, b3_numpy_policy, numpy_actions, hb_artifacts
):
    replay = jan7_replay
    assert len(replay.actions) > ROWS and numpy_actions.shape == replay.actions.shape
    _assert_the_bot_builds_these_engines(replay.settings)
    export = os.environ.get(EXPORT_ENV)
    if export:
        _export(Path(export), replay, numpy_actions, hb_artifacts / "b3_policy.npz")
    engine = _engine(replay.settings)
    stepper = ModelStepper("B3", engine, _features(), policy=b3_numpy_policy)
    T = replay.terminal_time
    bad: dict[str, list[int]] = {
        name: [] for name in ("obs", "z", "mid", "bid/ask", "action", "action vs SB3", "rounding", "orders")
    }
    moved_bid = moved_ask = moved_rows = 0
    worst_sb3 = 0.0
    for k in range(ROWS):
        row = _row(replay, k)
        if k == 0:
            decision = stepper.start(row, replay.t[0], T, replay.q[0])
        else:
            decision = stepper.step(row, replay.dt[k], replay.t[k], T, replay.q[k])
        if decision.obs.dtype != np.float32 or not np.array_equal(decision.obs, replay.raw_obs[k]):
            bad["obs"].append(k)
        if decision.z.dtype != np.float32 or not np.array_equal(decision.z, replay.norm_obs[k]):
            bad["z"].append(k)
        if decision.mid != replay.mid[k]:
            bad["mid"].append(k)
        if not (decision.quotes["bid"] == replay.bid[k] and decision.quotes["ask"] == replay.ask[k]):
            bad["bid/ask"].append(k)
        action = decision.action
        if action.shape != (2,) or not np.array_equal(action, numpy_actions[k]):
            bad["action"].append(k)
        gap = float(np.max(np.abs(action - replay.actions[k])))
        worst_sb3 = max(worst_sb3, gap)
        if not gap <= SB3_TOLERANCE:
            bad["action vs SB3"].append(k)
        rounded = decision.rounded
        if _rounding_problems(decision.quotes, rounded, decision.mid):
            bad["rounding"].append(k)
        if _order_problems(engine, rounded, replay.q[k]):
            bad["orders"].append(k)
        moved_bid += rounded.bid_rounded
        moved_ask += rounded.ask_rounded
        moved_rows += rounded.bid_rounded or rounded.ask_rounded
    counts = ", ".join(f"{name} {len(rows)}" for name, rows in bad.items())
    print(f"\nHB-PARITY loop B3: {ROWS} rows (start + {ROWS - 1} steps), T = {T!r} s; mismatches: {counts}; "
          f"max |action - SB3 predict| = {worst_sb3!r}; round_away_from_mid moved a price on {moved_rows} rows "
          f"(bid {moved_bid}, ask {moved_ask})")
    failures = {name: rows[:10] for name, rows in bad.items() if rows}
    assert not failures, f"first mismatching rows: {failures}"


def test_b4_bot_clock_dt_equals_the_frozen_loader_dt_over_the_first_20000_steps(jan7_replay, jan7_head):
    replay = jan7_replay
    header, lines = jan7_head
    index = {name: position for position, name in enumerate(header)}
    stamps, misaligned = [], []
    for k, line in enumerate(lines):
        fields = line.split(",")
        text = fields[index["timestamp"]]
        if not (text.isascii() and text.isdigit()):
            raise AssertionError(f"line {k}: timestamp {text!r} is not an integer number of µs")
        stamps.append(int(text))  # exact: the integer text, never through float64
        values = np.array([float(fields[index[name]]) for name in replay.row_columns])
        if not np.array_equal(values, replay.row_values[k]):
            misaligned.append(k)  # the line is not the replay's row k
    bot_dt = np.array([dt_seconds(stamps[k - 1], stamps[k]) for k in range(1, ROWS + 1)], dtype=np.float64)
    loader_dt = np.ascontiguousarray(replay.dt[1:ROWS + 1], dtype=np.float64)
    mismatches = np.flatnonzero(bot_dt.view(np.uint64) != loader_dt.view(np.uint64))
    print(f"\nHB-PARITY loop clock: {ROWS} steps, dt_seconds vs the frozen loader's dt: {mismatches.size} bitwise "
          f"mismatches, {len(misaligned)} lines misaligned with the replay's rows; ts_us from {stamps[0]} to "
          f"{stamps[-1]}; dt in [{float(bot_dt.min())!r}, {float(bot_dt.max())!r}] s, zero-dt steps "
          f"{int(np.sum(bot_dt == 0))}")
    assert misaligned == []
    assert mismatches.tolist() == []


def test_b4_b0_step_function_quotes_equal_a_direct_call_over_the_first_20000_jan7_rows(jan7_replay, inputs):
    assert float(inputs["b0_gamma"].read_text(encoding="utf-8").strip()) == B0_GAMMA == protocol_default("gamma_b0")
    replay = jan7_replay
    settings = replay.settings
    engine = _engine(settings)
    stepper = ModelStepper("B0", engine, _features(), gamma_b0=B0_GAMMA)
    dt = replay.dt
    t = np.cumsum(dt)  # fast_simulate's t_cumulative
    T = float(dt.sum())  # run_b0_fair.py's terminal_time
    tau = np.maximum(T - t, 0.0)  # fast_simulate's tau_arr
    q = np.random.default_rng(42).integers(-10, 11, size=ROWS)
    keys = ("bid", "ask", "delta_bid", "delta_ask", "gamma", "skew")
    bad: dict[str, list[int]] = {name: [] for name in ("quotes", "mid", "no policy", "rounding", "orders")}
    moved_bid = moved_ask = moved_rows = 0
    for k in range(ROWS):
        row = _row(replay, k)
        if k == 0:
            decision = stepper.start(row, t[0], T, q[0])
        else:
            decision = stepper.step(row, dt[k], t[k], T, q[k])
        _, direct = avellaneda_stoikov_quotes(
            midprice=np.full(1, replay.S[k]),
            inventory=np.full(1, float(q[k])),
            tau=np.full(1, tau[k]),
            gamma=np.full(1, B0_GAMMA),
            skew=np.zeros(1),
            sigma=settings["sigma"],
            kappa=settings["kappa"],
            tick_size=settings["tick_size"],
        )
        if any(_bits(decision.quotes[key]) != _bits(direct[key][0]) for key in keys):
            bad["quotes"].append(k)
        if decision.mid != replay.S[k]:
            bad["mid"].append(k)
        if decision.z is not None or decision.action is not None:
            bad["no policy"].append(k)
        rounded = decision.rounded
        if _rounding_problems(decision.quotes, rounded, decision.mid):
            bad["rounding"].append(k)
        if _order_problems(engine, rounded, q[k]):
            bad["orders"].append(k)
        moved_bid += rounded.bid_rounded
        moved_ask += rounded.ask_rounded
        moved_rows += rounded.bid_rounded or rounded.ask_rounded
    counts = ", ".join(f"{name} {len(rows)}" for name, rows in bad.items())
    print(f"\nHB-PARITY loop B0: {ROWS} rows (start + {ROWS - 1} steps), gamma_B0 = {B0_GAMMA!r}, T = {T!r} s, "
          f"q in [{q.min()}, {q.max()}]; mismatches: {counts}; round_away_from_mid moved a price on {moved_rows} "
          f"rows (bid {moved_bid}, ask {moved_ask})")
    assert np.unique(q).size == 21
    failures = {name: rows[:10] for name, rows in bad.items() if rows}
    assert not failures, f"first mismatching rows: {failures}"


def test_b4_live_book_path_gives_the_replay_rows_and_writes_the_tardis_lines_back(jan7_replay, jan7_head):
    """What BotCore does with the connector's top 25 at a step: book_row and book_mid give the replay's row and
    mid bitwise, and snapshot_fields writes the line back byte for byte (the shadow replay reads that file)."""

    replay = jan7_replay
    header, lines = jan7_head
    assert header == SNAPSHOT_HEADER
    index = {name: position for position, name in enumerate(header)}
    bad: dict[str, list[int]] = {"book_row": [], "book_mid": [], "snapshot line": []}
    repeats, previous = 0, None
    for k, line in enumerate(lines):
        fields = line.split(",")
        bids, asks = (
            tuple(
                (float(fields[index[f"{side}[{level}].price"]]), float(fields[index[f"{side}[{level}].amount"]]))
                for level in range(SNAPSHOT_LEVELS)
            )
            for side in ("bids", "asks")
        )
        row = book_row((bids, asks), 5)
        expected = replay.row_values[k]
        if set(row) != set(replay.row_columns) or any(
            _bits(row[name]) != _bits(value) for name, value in zip(replay.row_columns, expected)
        ):
            bad["book_row"].append(k)
        if book_mid((bids, asks)) != replay.mid[k]:
            bad["book_mid"].append(k)
        stamp, local = int(fields[index["timestamp"]]), int(fields[index["local_timestamp"]])
        if ",".join(snapshot_fields(stamp, local, asks, bids)) != line:
            bad["snapshot line"].append(k)
        repeats += fields[4:] == previous
        previous = fields[4:]
    counts = ", ".join(f"{name} {len(rows)}" for name, rows in bad.items())
    print(f"\nHB-PARITY loop book: {len(lines)} Jan 7 lines, header equal; mismatches: {counts}; for information, "
          f"{repeats} lines repeat the previous line's top 25 (the bot steps only when the top 25 changed)")
    failures = {name: rows[:10] for name, rows in bad.items() if rows}
    assert not failures, f"first mismatching lines: {failures}"
