"""BotCore and ModelStepper (bot_core.py) and the clear_halt CLI (clear_halt.py), end to end with a fake paper
exchange and a fake clock (WO-HB2 deliverable 3; PROTOCOL §5 and §7).

FakePaperExchange implements bot_core.ExchangePort as Hummingbot 2.17.0's binance_paper_trade behaves: Decimal
balances seeded DOGE 1,000,000 / USDT 100,000; a LIMIT order rests and fills in full at its own price when the
opposite best price reaches it (tick path) or a public trade prints strictly through it (trade path); the fee is
DeductedFromReturnsTradeFee(0.001), so a 100-DOGE buy credits 99.9 DOGE and debits 100 p USDT, and a sell pays
amount * price * 0.001 USDT; cancel is synchronous and fires the cancel event inside cancel(); a MARKET order
executes at the first tick at least 5 s after its submission, walking the book with one fill per level consumed;
nothing persists across a restart. Rig.tick() runs one clock tick in Hummingbot's order: the created events of
earlier orders (paper fires them 10 ms after submission), the connector (queued market orders, then crossed limit
orders), then the strategy's on_tick. FakeClock drives execution.StepClock; its wall clock can be stepped apart
from its monotonic clock.

The expectations come from the WO-HB2 specification and PROTOCOL.md, not from the code: B0's quotes are
recomputed with QuoteEngine at tau = T - t_elapsed, the snapshot rows are spelled out in the Tardis layout, and
the steps rows are recomputed from the logged inputs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from hummingbot_adapter import clear_halt
from hummingbot_adapter.bot_config import CALIBRATION_SHA256, MODES, BotSettings, VerifiedInputs, check_settings
from hummingbot_adapter.bot_core import BotCore, ModelStepper, book_row
from hummingbot_adapter.execution import RateLimiter, StepClock, round_away_from_mid
from hummingbot_adapter.features import FeatureEngine, row_columns
from hummingbot_adapter.monitor import ClockChecker
from hummingbot_adapter.policy import ACTION_DIM, HIDDEN, OBS_DIM, NumpyPolicy
from hummingbot_adapter.quotes import QuoteEngine
from cvar_mm.gym.data_loader import load_single_day_with_features
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES

TICK = 1e-5
TICK_D = Decimal("0.00001")
US = 1_000_000
DAY_US = 86_400 * US
T_DAY = 86_400.0
SIGMA, KAPPA, A = 7.556448594347906e-06, 41813.40928239842, 0.09500929388592372  # config/hb_calibration.json
GAMMA_B0 = 0.8908584807525282  # PROTOCOL §2
FEE_RATE = Decimal("0.001")  # PROTOCOL §3: the paper default, maker and taker
SEED = {"DOGE": Decimal(1_000_000), "USDT": Decimal(100_000)}  # PROTOCOL §3: the paper balances
SIDES = ("buy", "sell")
KINDS = ("snapshots", "steps", "orders", "fills", "trades", "health")
BOTS = {"B0": "hb_b0", "B3": "hb_b3"}
# Day D of the tests. It lies before the real clock, which clear_halt reads to judge the age of a state file.
D0 = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()) * US
DATE0, DATE1 = "2026-10-01", "2026-10-02"
INPUTS = VerifiedInputs(sigma=SIGMA, kappa=KAPPA, A=A, calibration_sha256=CALIBRATION_SHA256, policy_path=None,
                        policy_sha256=None, gamma_file_sha256=None)
ENGINE = QuoteEngine(SIGMA, KAPPA)
TARDIS_HEADER = ["exchange", "symbol", "timestamp", "local_timestamp"] + [
    f"{side}[{level}].{field}" for level in range(25) for side in ("asks", "bids") for field in ("price", "amount")
]  # the header line of binance_book_snapshot_25_2025-01-07_DOGEUSDT.csv
STATUS_PS1 = ("q", "wealth", "drawdown", "halted", "halt_reason", "stale_paused", "clock_paused", "drawdown_paused",
              "new_orders", "max_new_orders_10s", "skipped_rate_10s", "skipped_rate_day")  # deploy/status.ps1


def at(hours: int = 0, minutes: int = 0, seconds: float = 0.0, day: int = 0) -> int:
    """UTC µs of hh:mm:ss on day D + day."""

    return D0 + day * DAY_US + (hours * 3_600 + minutes * 60) * US + round(seconds * US)


# --- policies -------------------------------------------------------------------------------------------------


def policy_arrays(seed: int) -> dict[str, np.ndarray]:
    """Random arrays with the shapes and dtypes of b3_policy.npz (policy.py)."""

    rng = np.random.default_rng(seed)

    def weights(*shape: int, scale: float) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        "W1": weights(HIDDEN, OBS_DIM, scale=0.5), "b1": weights(HIDDEN, scale=0.1),
        "W2": weights(HIDDEN, HIDDEN, scale=0.2), "b2": weights(HIDDEN, scale=0.1),
        "W3": weights(ACTION_DIM, HIDDEN, scale=0.3), "b3": weights(ACTION_DIM, scale=0.1),
        "obs_mean": rng.normal(0.0, 0.5, OBS_DIM), "obs_var": rng.uniform(0.5, 4.0, OBS_DIM),
        "epsilon": np.array(1e-8), "clip_obs": np.array(10.0),
    }


def constant_policy(action: tuple[float, float]) -> NumpyPolicy:
    """A NumpyPolicy whose action is `action` for every observation (zero output weights)."""

    arrays = policy_arrays(9)
    arrays["W3"] = np.zeros((ACTION_DIM, HIDDEN), dtype=np.float32)
    arrays["b3"] = np.asarray(action, dtype=np.float32)
    return NumpyPolicy(arrays)


@pytest.fixture(scope="module")
def b3_policy() -> NumpyPolicy:
    """The exported actor when HB_ARTIFACTS names the pre-registered inputs, else random arrays of its shapes."""

    raw = os.environ.get("HB_ARTIFACTS")
    if raw:
        return NumpyPolicy.load(Path(raw) / "b3_policy.npz")
    return NumpyPolicy(policy_arrays(3))


# --- the fake clock and books -------------------------------------------------------------------------------


class FakeClock:
    """A wall clock and a monotonic clock in ns: advance_us() moves both, shift_wall_us() only the wall clock (an
    NTP step), so the two can differ."""

    def __init__(self, wall_us: int, monotonic_ns: int = 7_000_000_000):
        self._wall_ns = wall_us * 1_000
        self._monotonic_ns = monotonic_ns

    def wall_ns(self) -> int:
        return self._wall_ns

    def monotonic_ns(self) -> int:
        return self._monotonic_ns

    @property
    def wall_us(self) -> int:
        return self._wall_ns // 1_000

    def advance_us(self, us: int) -> None:
        if us < 0:
            raise ValueError(f"time runs forward, got {us} µs")
        self._wall_ns += us * 1_000
        self._monotonic_ns += us * 1_000

    def shift_wall_us(self, us: int) -> None:
        self._wall_ns += us * 1_000

    def step_clock(self, rate_ppm: str = "0") -> StepClock:
        return StepClock(self.wall_ns, self.monotonic_ns, rate_ppm)


def make_book(bid_ticks: int, spread: int = 1, salt: int = 0, levels: int = 25):
    """A Binance-like top 25: the best bid at bid_ticks ticks, the best ask `spread` ticks above, levels one tick
    apart, whole DOGE amounts; `salt` changes the amounts only."""

    bids = tuple((round((bid_ticks - i) * TICK, 5), float(10_000 + 137 * i + salt)) for i in range(levels))
    asks = tuple((round((bid_ticks + spread + i) * TICK, 5), float(12_000 + 151 * i + salt)) for i in range(levels))
    return bids, asks


def mid_of(book) -> float:
    bids, asks = book
    return (asks[0][0] + bids[0][0]) / 2


def thin_top(book, side: str, amount: float):
    """The book with `amount` DOGE at the best level of one side (a market order then walks two levels)."""

    bids, asks = book
    if side == "bids":
        return ((bids[0][0], amount),) + bids[1:], asks
    return bids, ((asks[0][0], amount),) + asks[1:]


def crossing_book(price: Decimal, side: str, salt: int = 0):
    """A 1-tick book that reaches a resting order: the best ask at a bid's price, or the best bid at an ask's."""

    ticks = int(price / TICK_D)
    return make_book(ticks - 1, 1, salt) if side == "buy" else make_book(ticks, 1, salt)


def tardis_text(value: float) -> str:
    """A price or amount as the Tardis files print it: whole numbers without '.0', others in their shortest form
    (the Jan 7 file has 0.3878 and 55438)."""

    return str(int(value)) if float(value).is_integer() else repr(float(value))


def tardis_row(ts_us: int, local_us: int, book) -> list[str]:
    bids, asks = book
    fields = ["binance", "DOGEUSDT", str(ts_us), str(local_us)]
    for level in range(25):
        for side in (asks, bids):
            fields += [tardis_text(side[level][0]), tardis_text(side[level][1])] if level < len(side) else ["", ""]
    return fields


def row_of(book) -> dict[str, float]:
    """The FeatureEngine row of a book (Tardis column names)."""

    bids, asks = book
    row = {"asks[0].price": asks[0][0], "bids[0].price": bids[0][0]}
    for level in range(5):
        row[f"bids[{level}].amount"] = bids[level][1]
        row[f"asks[{level}].amount"] = asks[level][1]
    return row


def expected_b0(book, q: int, ts_us: int, day_start_us: int = D0) -> tuple[Decimal, Decimal]:
    """B0's posted prices: the frozen quotes at tau = T - t_elapsed, rounded away from the mid (clarification a)."""

    mid = mid_of(book)
    quotes = ENGINE.b0(mid, q, T_DAY - (ts_us - day_start_us) / 1e6, GAMMA_B0)
    rounded = round_away_from_mid(quotes, mid)
    return Decimal(f"{rounded.bid:.5f}"), Decimal(f"{rounded.ask:.5f}")


# --- the fake paper exchange ----------------------------------------------------------------------------------


class FakePaperExchange:
    """bot_core.ExchangePort with binance_paper_trade's semantics (see the module docstring)."""

    MARKET_DELAY_US = 5 * US  # paper_trade_exchange.pyx TRADE_EXECUTION_DELAY

    def __init__(self, clock: FakeClock, balances: dict[str, Decimal] | None = None):
        self.clock = clock
        self.core: BotCore | None = None
        self.balances = dict(SEED if balances is None else balances)
        self.limits: dict[str, tuple[str, Decimal, Decimal]] = {}  # resting LIMIT orders: id -> (side, price, amount)
        self.queued: list[tuple[str, str, Decimal, int]] = []  # MARKET orders: (id, side, amount, submitted µs)
        self.calls: list[tuple] = []
        self.fail_cancel = False
        self._acks: list[str] = []
        self._count = 0
        self._filling: str | None = None

    def attach(self, core: BotCore) -> None:
        self.core = core

    def _new_id(self, side: str) -> str:
        self._count += 1
        return f"{side}://DOGE-USDT/{self._count:026x}"

    # ExchangePort

    def submit_limit(self, side: str, price: Decimal, amount: Decimal) -> str:
        if not (isinstance(price, Decimal) and isinstance(amount, Decimal)):
            raise TypeError("price and amount must be Decimal objects.")  # strategy_base.pyx
        order_id = self._new_id(side)
        self.limits[order_id] = (side, price, amount)
        self.calls.append(("limit", order_id, side, price, amount))
        self._acks.append(order_id)
        return order_id

    def submit_market(self, side: str, amount: Decimal) -> str:
        if not isinstance(amount, Decimal):
            raise TypeError("price and amount must be Decimal objects.")
        order_id = self._new_id(side)
        self.queued.append((order_id, side, amount, self.clock.wall_us))
        self.calls.append(("market", order_id, side, None, amount))
        self._acks.append(order_id)
        return order_id

    def cancel(self, order_id: str) -> None:
        if self.fail_cancel:
            raise RuntimeError("cancel failed")
        if order_id == self._filling:  # paper would erase an order it is still filling
            raise AssertionError(f"cancel of {order_id} inside its own fill event")
        self.calls.append(("cancel", order_id))
        if self.limits.pop(order_id, None) is not None:
            self.core.on_cancelled(order_id)  # did_cancel_order runs inside cancel()

    def balance(self, asset: str) -> Decimal:
        return self.balances[asset]

    def set_balance(self, asset: str, amount: Decimal) -> None:
        self.balances[asset] = Decimal(amount)

    def open_order_ids(self) -> set[str]:
        return set(self.limits)

    # the engine

    def _settle(self, side: str, price: Decimal, amount: Decimal) -> tuple[Decimal, str]:
        """The balances as paper writes them; returns the fee and its asset (DeductedFromReturnsTradeFee)."""

        if side == "buy":
            fee = amount * FEE_RATE
            self.balances["USDT"] -= amount * price
            self.balances["DOGE"] += amount - fee
            return fee, "DOGE"
        fee = amount * price * FEE_RATE
        self.balances["DOGE"] -= amount
        self.balances["USDT"] += amount * price - fee
        return fee, "USDT"

    def _fill_limit(self, order_id: str, mid: float) -> None:
        side, price, amount = self.limits[order_id]
        fee, asset = self._settle(side, price, amount)
        self._filling = order_id
        try:  # OrderFilledEvent, then the completed event; the order is erased only after both
            self.core.on_fill(order_id, side, price, amount, fee, asset, FEE_RATE, str(self.clock.wall_us), mid)
            base, quote = (amount - fee, amount * price) if side == "buy" else (amount, amount * price - fee)
            self.core.on_completed(order_id, base, quote)
        finally:
            self._filling = None
        del self.limits[order_id]

    def deliver_acks(self) -> None:
        acks, self._acks = self._acks, []
        for order_id in acks:
            self.core.on_created(order_id)

    def match(self, book) -> None:
        """The tick path: a bid fills when the best ask <= its price, an ask when the best bid >= its price."""

        bids, asks = book
        if not bids or not asks:
            return
        best_bid, best_ask = Decimal(repr(bids[0][0])), Decimal(repr(asks[0][0]))
        for order_id, (side, price, _) in list(self.limits.items()):
            if order_id in self.limits and (best_ask <= price if side == "buy" else best_bid >= price):
                self._fill_limit(order_id, mid_of(book))

    def public_trade(self, price: float, amount: float, taker_side: str) -> None:
        """A public trade: the strategy's listener, then the paper matcher (a taker sell fills bids strictly
        above its price, a taker buy fills asks strictly below; Decimal against float, exactly)."""

        self._count += 1
        self.core.on_public_trade(self.clock.wall_us / 1e6, self._count, price, amount, taker_side)
        for order_id, (side, order_price, _) in list(self.limits.items()):
            through = (side == "buy" and taker_side == "sell" and order_price > price) or (
                side == "sell" and taker_side == "buy" and order_price < price)
            if order_id in self.limits and through:
                self._fill_limit(order_id, self.core.last_mid)

    def run_market(self, book) -> None:
        """MARKET orders at least 5 s old walk the book: one fill per level consumed (taker), then completed. The
        fill events carry Decimal(row.price) and Decimal(row.amount) of the book's floats (binary expansions)."""

        now = self.clock.wall_us
        due = [order for order in self.queued if order[3] <= now - self.MARKET_DELAY_US]
        self.queued = [order for order in self.queued if order[3] > now - self.MARKET_DELAY_US]
        for order_id, side, amount, _ in due:
            remaining, fills = amount, []
            for price, size in (book[1] if side == "buy" else book[0]):
                if remaining <= 0:
                    break
                take = min(remaining, Decimal(size))
                fills.append((Decimal(price), take))
                remaining -= take
            assert remaining == 0, "the fake book is too thin for the market order"
            paid = Decimal(0)
            for level, (price, take) in enumerate(fills):
                fee, asset = self._settle(side, price, take)
                paid += take * price
                self.core.on_fill(order_id, side, price, take, fee, asset, FEE_RATE, f"{order_id}_{level}",
                                  mid_of(book))
            self.core.on_completed(order_id, amount, paid)

    def foreign_fill(self, side: str, price: Decimal, amount: Decimal, mid: float) -> str:
        """A fill of an order the bot does not know (one placed by hand, say)."""

        order_id = self._new_id(side)
        fee, asset = self._settle(side, price, amount)
        self.core.on_fill(order_id, side, price, amount, fee, asset, FEE_RATE, "manual", mid)
        return order_id


# --- one bot process ---------------------------------------------------------------------------------------------


class Rig:
    """One bot process: its settings, clock, paper exchange and BotCore."""

    def __init__(self, settings: BotSettings, clock: FakeClock, policy: NumpyPolicy | None = None,
                 balances: dict[str, Decimal] | None = None, rate_ppm: str = "0"):
        self.settings, self.clock = settings, clock
        self.exchange = FakePaperExchange(clock, balances)
        self.core = BotCore(settings, INPUTS, self.exchange, clock.step_clock(rate_ppm), policy,
                            probe=lambda: {"connector_ready": True, "ready_to_trade": True}, rss_mb=lambda: 321.5)
        self.exchange.attach(self.core)
        self.book = None

    def tick(self, book=None, dt: float = 0.1) -> None:
        """One clock tick dt s after the last; without a book the last one is read again."""

        self.clock.advance_us(round(dt * US))
        self._tick(book)

    def tick_at(self, ts_us: int, book=None) -> None:
        self.clock.advance_us(ts_us - self.clock.wall_us)
        self._tick(book)

    def _tick(self, book) -> None:
        if book is not None:
            self.book = book
        self.exchange.deliver_acks()
        if self.book is not None:
            self.exchange.run_market(self.book)
            self.exchange.match(self.book)
        self.core.on_tick(self.book, self.clock.wall_us)


def make_rig(tmp_path: Path, model: str = "B0", start_us: int = at(10), policy: NumpyPolicy | None = None,
             balances: dict[str, Decimal] | None = None, mode: str = "smoke", rate_ppm: str = "0", **overrides) -> Rig:
    settings = BotSettings(model=model, mode=mode, data_root=str(tmp_path), **overrides)
    return Rig(settings, FakeClock(start_us), policy, balances, rate_ppm)


def restart(old: Rig, gap_s: float, policy: NumpyPolicy | None = None, settings: BotSettings | None = None) -> Rig:
    """A new process gap_s s after the old one's last tick: a new monotonic origin, a paper exchange re-seeded with
    the PROTOCOL balances and no orders, and a new BotCore."""

    clock = FakeClock(old.clock.wall_us + round(gap_s * US), monotonic_ns=3_000_000_000)
    return Rig(settings or old.settings, clock, policy, rate_ppm=old.core.clock.rate_ppm)


def kill(rig: Rig) -> None:
    """Kill the process: the rows buffered since the last save are lost, the OS closes the files, and nothing else
    runs (no stop())."""

    log = rig.core.log
    if log is not None:
        for rows in log._buffers.values():  # what a kill loses
            rows.clear()
        log.close()
    rig.core.finalizer.wait()


def resting(rig: Rig, side: str) -> tuple[str, Decimal] | None:
    """The bot's live LIMIT order on a side, as the paper exchange holds it (at most one per side, §5.3)."""

    found = [(order_id, price) for order_id, (order_side, price, _) in rig.exchange.limits.items()
             if order_side == side]
    assert len(found) <= 1, f"{len(found)} live {side} orders"
    return found[0] if found else None


def live_sides(rig: Rig) -> list[str]:
    return sorted(side for side, _, _ in rig.exchange.limits.values())


def cross(rig: Rig, side: str, dt: float = 1.1, salt: int = 0) -> tuple[str, Decimal]:
    """A tick whose book fills the resting order on `side`; returns that order."""

    order = resting(rig, side)
    assert order is not None, f"no {side} order to cross"
    rig.tick(crossing_book(order[1], side, salt), dt)
    assert order[0] not in rig.exchange.limits
    return order


def nudge(rig: Rig, dt: float = 0.2) -> None:
    """A tick whose top 25 differs from the last one in an amount only (a step without a price move)."""

    bids, asks = rig.book
    rig.tick((((bids[0][0], bids[0][1] + 1.0),) + bids[1:], asks), dt)


def limit_calls(rig: Rig, since: int = 0) -> list[tuple]:
    return [call for call in rig.exchange.calls[since:] if call[0] == "limit"]


# --- reading the outputs --------------------------------------------------------------------------------------


def day_file(rig: Rig, kind: str, date: str = DATE0) -> Path:
    folder = rig.settings.bot_dir / date
    plain = folder / f"{kind}_{date}.csv"
    return plain if plain.exists() else folder / f"{kind}_{date}.csv.gz"


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def table(rig: Rig, kind: str, date: str = DATE0) -> list[dict[str, str]]:
    """A telemetry file's rows so far (the open day's buffers are flushed first)."""

    if rig.core.log is not None and rig.core.date == date:
        rig.core.log.flush()
    header, rows = read_csv(day_file(rig, kind, date))
    return [dict(zip(header, row, strict=True)) for row in rows]


def health(rig: Rig, date: str = DATE0) -> list[tuple[str, dict]]:
    return [(row["event"], json.loads(row["detail"])) for row in table(rig, "health", date)]


def events(rig: Rig, name: str, date: str = DATE0, **match) -> list[dict]:
    """The details of the health rows `name` whose detail holds `match`."""

    return [detail for event, detail in health(rig, date)
            if event == name and all(detail.get(key) == value for key, value in match.items())]


def state_path(rig: Rig) -> Path:
    return rig.settings.bot_dir / "state.json"


def state_of(rig: Rig) -> dict:
    return json.loads(state_path(rig).read_text(encoding="utf-8"))


def assert_finalized(rig: Rig, date: str) -> None:
    """The day's six files gzipped, no plain file left, and SHA256SUMS in the day and bot folders (§11)."""

    folder = rig.settings.bot_dir / date
    names = sorted(f"{kind}_{date}.csv.gz" for kind in KINDS)
    assert sorted(path.name for path in folder.iterdir()) == sorted(names + ["SHA256SUMS"])
    expected = [f"{hashlib.sha256((folder / name).read_bytes()).hexdigest()}  {name}" for name in names]
    assert (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines() == expected
    bot_sums = (rig.settings.bot_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert all(f"{line.split('  ')[0]}  {date}/{line.split('  ')[1]}" in bot_sums for line in expected)


# --- start-up and stepping --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["B0", "B3"])
def test_fresh_start_mid_day_quotes_both_sides_and_writes_every_record(tmp_path, b3_policy, model):
    rig = make_rig(tmp_path, model, policy=b3_policy if model == "B3" else None)
    book = make_book(8_000)
    rig.tick(book, dt=0)
    ts, mid = at(10), mid_of(book)
    assert rig.settings.bot_dir == tmp_path / "smoke" / BOTS[model]
    assert state_path(rig).exists()  # saved at start-up

    live = {side: (price, amount) for side, price, amount in rig.exchange.limits.values()}
    assert sorted(live) == list(SIDES) and len(rig.exchange.limits) == 2
    for price, amount in live.values():
        assert amount == Decimal(100) and price == price.quantize(TICK_D)
    bid, ask = live["buy"][0], live["sell"][0]
    assert float(bid) < mid < float(ask)
    if model == "B0":
        assert (bid, ask) == expected_b0(book, 0, ts)

    rig.tick(book, dt=1.0)  # the same top 25: no step; the periodic save flushes the files
    assert rig.core.counters["steps"] == 1

    assert health(rig)[0][0] == "start"
    [start] = events(rig, "start")
    assert (start["fresh"], start["model"], start["mode"]) == (True, model, "smoke")
    assert (start["sigma"], start["kappa"], start["calibration_sha256"]) == (SIGMA, KAPPA, CALIBRATION_SHA256)
    assert start["changed_thresholds"] == {} and start["balances"] == {"DOGE": "1000000", "USDT": "100000"}
    heartbeat = events(rig, "heartbeat")[0]
    assert heartbeat["rss_mb"] == 321.5 and heartbeat["probe"] == {"connector_ready": True, "ready_to_trade": True}

    header, rows = read_csv(day_file(rig, "snapshots"))
    assert header == TARDIS_HEADER and rows == [tardis_row(ts, ts, book)]

    [step] = table(rig, "steps")
    assert (step["ts_us"], step["utc"], step["phase"], step["dt"]) == (str(ts), "2026-10-01T10:00:00.000000Z",
                                                                        "trade", "")
    assert (float(step["t_elapsed"]), float(step["tau"]), float(step["mid"])) == (36_000.0, 50_400.0, mid)
    assert (Decimal(step["resting_bid"]), Decimal(step["resting_ask"])) == (bid, ask)
    assert (step["q"], step["quoting"], step["halted"], Decimal(step["residual_doge"])) == ("0", "1", "0", 0)
    b3_columns = list(FORMULA_FEATURE_NAMES) + [f"z_{name}" for name in FORMULA_FEATURE_NAMES]
    b3_columns += ["action_gamma", "action_skew"]
    assert all((column in step) == (model == "B3") for column in b3_columns)
    if model == "B0":
        assert (float(step["gamma"]), float(step["skew"])) == (GAMMA_B0, 0.0)  # gamma_B0 with skew 0

    orders = table(rig, "orders")
    submits = {(row["order_id"], row["side"], row["order_type"], Decimal(row["price"]), row["amount"],
                float(row["mid_at_post"]), row["ts_us"]) for row in orders if row["event"] == "submit"}
    assert submits == {(order_id, side, "limit", price, "100", mid, str(ts))
                       for order_id, (side, price, _) in rig.exchange.limits.items()}
    assert {row["order_id"] for row in orders if row["event"] == "ack"} == set(rig.exchange.limits)

    state = state_of(rig)
    assert (state["day"]["date"], state["last_step_us"], state["status"]["halted"]) == (DATE0, ts, False)
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files and all(tmp_path / "smoke" in path.parents for path in files)  # a smoke run writes under smoke/
    for kind in KINDS:
        assert b"\r" not in day_file(rig, kind).read_bytes()  # newline="\n"


@pytest.mark.parametrize("balances", [
    {"DOGE": Decimal(999_999), "USDT": Decimal(100_000)},
    {"DOGE": Decimal(1_000_000), "USDT": Decimal("100000.5")},
])
def test_fresh_start_refuses_paper_balances_other_than_the_protocol_ones(tmp_path, balances):
    rig = make_rig(tmp_path, balances=balances)
    with pytest.raises(ValueError, match="PROTOCOL paper balances"):
        rig.tick(make_book(8_000), dt=0)
    assert rig.exchange.calls == [] and not state_path(rig).exists()


def test_no_step_unless_the_top_25_changed(tmp_path):
    rig = make_rig(tmp_path)
    book = make_book(8_000)
    rig.tick(book, dt=0)
    for _ in range(5):
        rig.tick((tuple(book[0]), tuple(book[1])), dt=0.1)  # an equal copy
    assert rig.core.counters["steps"] == 1
    bids, asks = book
    deeper = (bids[:24] + ((bids[24][0], bids[24][1] + 1.0),), asks)  # the 25th bid level changed
    rig.tick(deeper, dt=0.1)
    rig.tick(deeper, dt=0.1)
    assert rig.core.counters["steps"] == 2 and rig.core.counters["ticks"] == 8
    assert len(table(rig, "steps")) == 2 and len(table(rig, "snapshots")) == 2


def test_stale_book_pauses_at_30_s_resumes_on_change_and_halts_at_5_min_closing_at_market(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    assert rig.core.ledger.q == 1 and live_sides(rig) == ["buy", "sell"]
    rig.tick(dt=29.9)
    assert live_sides(rig) == ["buy", "sell"] and not rig.core.safety.stale_paused
    live = set(rig.exchange.limits)
    rig.tick(dt=0.1)  # no change of the top 25 for 30 s: cancel the quotes and pause
    assert rig.exchange.limits == {} and rig.core.safety.stale_paused and not rig.core.safety.halted
    [pause] = events(rig, "pause", trigger="stale_book")
    assert pause["kind"] == "incident"
    orders = table(rig, "orders")
    assert {row["order_id"] for row in orders if row["event"] == "cancel" and row["reason"] == "stale_book"} == live
    rig.tick(dt=60.0)
    assert rig.exchange.limits == {} and not rig.core.safety.halted
    rig.tick(make_book(8_003), dt=0.1)  # updates return: resume, and quote at this step
    assert events(rig, "resume", trigger="stale_book") and not rig.core.safety.stale_paused
    assert live_sides(rig) == ["buy", "sell"]

    calls = len(rig.exchange.calls)
    rig.tick(dt=299.9)
    assert rig.core.safety.stale_paused and not rig.core.safety.halted and rig.exchange.limits == {}
    rig.tick(dt=0.1)  # 5 min: halt, close at market
    assert rig.core.safety.halted and rig.core.safety.halt_reason.startswith("stale_book")
    [halt] = events(rig, "halt", trigger="stale_book")
    assert halt["kind"] == "incident"
    [(_, side, amount, _)] = rig.exchange.queued
    assert (side, amount) == ("sell", Decimal(100))
    rig.tick(dt=5.0)  # paper executes the close 5 s later
    assert rig.core.ledger.q == 0 and rig.exchange.queued == [] and events(rig, "close_done", reason="halt")
    rig.tick(make_book(8_004), dt=1.0)  # the book moves again: still halted, nothing posted
    assert rig.core.safety.halted and limit_calls(rig, calls) == []
    assert state_of(rig)["status"]["halted"] is True


def test_a_crossing_fill_books_q_1_and_99_9_doge_with_residual_0_and_the_replacement_goes_out_at_the_next_step(
        tmp_path):
    rig = make_rig(tmp_path)
    book0 = make_book(8_000)
    rig.tick(book0, dt=0)
    bid_id, bid_price = resting(rig, "buy")
    ask_id, ask_price = resting(rig, "sell")
    rig.tick(crossing_book(bid_price, "buy"), dt=0.5)
    fill_ts, fill_mid = rig.clock.wall_us, mid_of(rig.book)

    assert rig.exchange.balances == {"DOGE": Decimal("1000099.9"), "USDT": Decimal(100_000) - 100 * bid_price}
    ledger = rig.core.ledger
    assert (ledger.q, ledger.position, ledger.cash) == (1, Decimal(100), -100 * bid_price)
    assert ledger.fees == {"DOGE": Decimal("0.1"), "USDT": Decimal(0)}
    assert ledger.residual(rig.exchange.balance("DOGE")) == 0

    replacement = resting(rig, "buy")  # posted at the step of the same tick, the first step after the fill
    assert replacement[0] != bid_id and replacement[1] == expected_b0(rig.book, 1, fill_ts)[0]
    assert resting(rig, "sell") == (ask_id, ask_price)  # 0.5 s old: kept
    step = table(rig, "steps")[-1]
    assert (step["ts_us"], step["q"], Decimal(step["residual_doge"])) == (str(fill_ts), "1", 0)
    assert Decimal(step["doge_balance"]) == Decimal("1000099.9") and Decimal(step["fees_doge"]) == Decimal("0.1")
    assert float(step["wealth"]) == float(-100 * bid_price) + 100 * fill_mid  # W = cash + 100 q S
    assert Decimal(step["resting_bid"]) == replacement[1]

    [fill] = table(rig, "fills")
    assert (fill["order_id"], fill["side"], fill["order_type"], fill["close"]) == (bid_id, "buy", "limit", "0")
    assert (Decimal(fill["price"]), Decimal(fill["amount"])) == (bid_price, 100)
    assert (Decimal(fill["fee"]), fill["fee_asset"]) == (Decimal("0.1"), "DOGE")
    assert (int(fill["ts_us"]), int(fill["post_ts_us"])) == (fill_ts, at(10))
    assert (float(fill["mid_at_fill"]), float(fill["mid_at_post"])) == (fill_mid, mid_of(book0))
    assert float(fill["delta_fill_ticks"]) == pytest.approx((fill_mid - float(bid_price)) / TICK, abs=1e-6)
    assert float(fill["delta_post_ticks"]) == pytest.approx((mid_of(book0) - float(bid_price)) / TICK, abs=1e-6)
    assert rig.core.counters["fills_buy"] == 1

    # an ask filled through the trade path, between two ticks: its replacement waits for the next step
    rig.clock.advance_us(30_000)
    rig.exchange.public_trade(round(float(ask_price) + TICK, 5), 500.0, "buy")
    assert ask_id not in rig.exchange.limits and rig.core.ledger.q == 0 and resting(rig, "sell") is None
    assert rig.core.ledger.fees["USDT"] == 100 * ask_price * FEE_RATE
    rig.tick(dt=0.07)  # the same top 25: no step, no replacement
    assert resting(rig, "sell") is None
    nudge(rig)
    assert resting(rig, "sell") is not None
    step = table(rig, "steps")[-1]
    assert (step["q"], Decimal(step["residual_doge"])) == ("0", 0)
    [trade] = table(rig, "trades")
    assert (float(trade["price"]), float(trade["amount"]), trade["side"]) == (round(float(ask_price) + TICK, 5),
                                                                              500.0, "buy")


def test_requote_only_after_the_target_moved_a_tick_and_the_order_is_a_second_old(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    first = {side: resting(rig, side) for side in SIDES}
    rig.tick(make_book(8_002), dt=0.5)  # the targets moved 2 ticks; the orders are 0.5 s old
    targets = expected_b0(rig.book, 0, rig.clock.wall_us)
    assert (targets[0] - first["buy"][1], targets[1] - first["sell"][1]) == (2 * TICK_D, 2 * TICK_D)
    assert {side: resting(rig, side) for side in SIDES} == first
    rig.tick(make_book(8_002, salt=1), dt=0.4999)  # 0.9999 s old
    assert {side: resting(rig, side) for side in SIDES} == first
    rig.tick(make_book(8_002, salt=2), dt=0.0001)  # exactly 1 s old: re-quote both sides
    second = {side: resting(rig, side) for side in SIDES}
    assert all(second[side][0] != first[side][0] for side in SIDES)
    assert (second["buy"][1], second["sell"][1]) == expected_b0(rig.book, 0, rig.clock.wall_us)
    orders = table(rig, "orders")
    assert {row["order_id"] for row in orders if row["event"] == "cancel" and row["reason"] == "requote"} == {
        first[side][0] for side in SIDES}
    rig.tick(make_book(8_002, salt=3), dt=1.5)  # 1.5 s old, the targets did not move
    assert expected_b0(rig.book, 0, rig.clock.wall_us) == (second["buy"][1], second["sell"][1])
    assert {side: resting(rig, side) for side in SIDES} == second
    rig.tick(make_book(8_003), dt=0.1)  # moved exactly 1 tick, 1.6 s old: re-quote
    third = {side: resting(rig, side) for side in SIDES}
    assert (third["buy"][1], third["sell"][1]) == (second["buy"][1] + TICK_D, second["sell"][1] + TICK_D)
    assert rig.core.rate.cancels == 4 and rig.core.rate.day_orders == 6


@pytest.mark.parametrize("side", SIDES)
def test_no_bid_is_posted_while_q_is_10_and_no_ask_while_q_is_minus_10(tmp_path, side):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    sign, other = (1, "sell") if side == "buy" else (-1, "buy")
    for k in range(10):
        cross(rig, side, salt=k)  # each fill's replacement goes out at the same tick's step, until the cap
    assert rig.core.ledger.q == 10 * sign and resting(rig, side) is None and resting(rig, other) is not None
    for _ in range(3):
        nudge(rig, dt=1.1)
        assert resting(rig, side) is None and resting(rig, other) is not None
    steps = table(rig, "steps")
    assert steps[-1]["resting_bid" if side == "buy" else "resting_ask"] == "" and steps[-1]["quoting"] == "1"
    cross(rig, other)  # q = 9: the side is quoted again at that step
    assert rig.core.ledger.q == 9 * sign and resting(rig, side) is not None
    cross(rig, side)  # back at the cap
    assert rig.core.ledger.q == 10 * sign and resting(rig, side) is None
    assert not rig.core.safety.halted  # |q| = 10 is the cap, not beyond it


@pytest.mark.parametrize("side", SIDES)
def test_a_resting_order_on_a_side_that_reached_the_cap_is_cancelled_at_the_next_step(tmp_path, side):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    sign, other = (1, "sell") if side == "buy" else (-1, "buy")
    for k in range(9):
        cross(rig, side, salt=k)
    order_id, price = resting(rig, side)
    # a fill of an order the bot does not know takes q to the cap while the side's own order still rests
    rig.exchange.foreign_fill(side, price - TICK_D if side == "buy" else price + TICK_D, Decimal(100),
                              mid_of(rig.book))
    assert rig.core.ledger.q == 10 * sign and resting(rig, side) == (order_id, price)
    nudge(rig)
    assert resting(rig, side) is None and resting(rig, other) is not None
    assert [row["reason"] for row in table(rig, "orders")
            if row["event"] == "cancel" and row["order_id"] == order_id] == ["inventory_cap"]


def test_rate_caps_skip_and_count_new_orders_and_keep_the_old_order_on_a_skipped_requote(tmp_path):
    rig = make_rig(tmp_path)
    rig.core.rate = RateLimiter(max_per_window=2, window_us=10 * US, max_per_day=5)
    rig.tick(make_book(8_000), dt=0)  # two new orders fill the 10 s window
    first = {side: resting(rig, side) for side in SIDES}
    rig.tick(make_book(8_003), dt=1.5)  # both sides are due for a re-quote: skipped, the old orders kept
    assert {side: resting(rig, side) for side in SIDES} == first
    assert [call for call in rig.exchange.calls if call[0] == "cancel"] == []
    targets = expected_b0(rig.book, 0, rig.clock.wall_us)
    skips = [row for row in table(rig, "orders") if row["event"] == "skip"]
    assert [(row["side"], Decimal(row["price"]), row["reason"]) for row in skips] == [
        ("buy", targets[0], "rate_10s:requote"), ("sell", targets[1], "rate_10s:requote")]
    step = table(rig, "steps")[-1]
    assert (Decimal(step["resting_bid"]), Decimal(step["resting_ask"])) == (first["buy"][1], first["sell"][1])
    status = state_of(rig)["status"]
    assert (status["skipped_rate_10s"], status["skipped_rate_day"], status["new_orders"],
            status["max_new_orders_10s"]) == (2, 0, 2, 2)

    rig.tick(make_book(8_003, salt=1), dt=8.5)  # 10 s after the first two: the window has room again
    second = {side: resting(rig, side) for side in SIDES}
    assert (second["buy"][1], second["sell"][1]) == expected_b0(rig.book, 0, rig.clock.wall_us)
    assert rig.core.rate.day_orders == 4 and rig.core.rate.cancels == 2

    rig.tick(make_book(8_006), dt=10.0)  # the day's 5th order goes out, the 6th would exceed the day cap
    third = {side: resting(rig, side) for side in SIDES}
    assert third["buy"][0] != second["buy"][0] and third["sell"] == second["sell"]
    assert [row["reason"] for row in table(rig, "orders") if row["event"] == "skip"][-1] == "rate_day:requote"
    cross(rig, "buy")  # the fill's replacement is a new order too: skipped, nothing rests on the bid
    assert resting(rig, "buy") is None
    skips = [row for row in table(rig, "orders") if row["event"] == "skip"]
    assert [row["reason"] for row in skips if row["side"] == "buy"][-1] == "rate_day"
    status = state_of(rig)["status"]
    assert (status["new_orders"], status["skipped_rate_10s"]) == (5, 2) and status["skipped_rate_day"] >= 2
    assert status["cancels"] == rig.core.rate.cancels == 3


def test_daily_close_at_23_59_50_then_wait_rows_and_the_reset_at_the_first_step_after_midnight(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23, 58))
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    rig.tick_at(at(23, 59, 49.9), make_book(7_999, salt=5))
    assert live_sides(rig) == ["buy", "sell"]  # still quoting before 23:59:50
    live = set(rig.exchange.limits)

    close_book = thin_top(make_book(7_998), "bids", 60.0)
    rig.tick_at(at(23, 59, 50), close_book)  # cancel both quotes, close at market
    assert rig.exchange.limits == {}
    [(close_id, side, amount, submitted)] = rig.exchange.queued
    assert (side, amount, submitted) == ("sell", Decimal(100), at(23, 59, 50))
    orders = table(rig, "orders")
    assert {row["order_id"] for row in orders if row["event"] == "cancel" and row["reason"] == "day_close"} == live
    [submit] = [row for row in orders if row["order_id"] == close_id and row["event"] == "submit"]
    assert (submit["order_type"], submit["reason"], float(submit["mid_at_post"])) == ("market", "day_close",
                                                                                      mid_of(close_book))
    assert events(rig, "close_submitted", reason="day_close")
    rig.tick_at(at(23, 59, 54.9), thin_top(make_book(7_998, salt=1), "bids", 60.0))
    assert len(rig.exchange.queued) == 1 and rig.core.ledger.q == 1 and rig.exchange.limits == {}

    exec_book = thin_top(make_book(7_998, salt=2), "bids", 60.0)
    rig.tick_at(at(23, 59, 55), exec_book)  # 5 s later: the close walks two bid levels, 60 and 40 DOGE
    bids = exec_book[0]
    walked = [(Decimal(bids[0][0]), Decimal(60)), (Decimal(bids[1][0]), Decimal(40))]
    assert rig.core.ledger.position == 0 and rig.exchange.balances["DOGE"] == Decimal("999999.9")
    fills = [row for row in table(rig, "fills") if row["order_id"] == close_id]
    assert [(Decimal(row["price"]), Decimal(row["amount"]), row["close"], row["order_type"]) for row in fills] == [
        (price, take, "1", "market") for price, take in walked]
    assert [row["exchange_trade_id"] for row in fills] == [f"{close_id}_0", f"{close_id}_1"]
    mid_post = Decimal(repr(mid_of(close_book)))
    cost = sum((mid_post - price) * take + take * price * FEE_RATE for price, take in walked)  # half-spread + fee
    [done] = events(rig, "close_done", reason="day_close")
    assert Decimal(done["cost_usdt"]) == cost

    for k in range(3):  # 23:59:56-58: WAIT rows are still written, with no quotes
        rig.tick(make_book(7_997, salt=10 + k), dt=1.0)
    rig.tick_at(at(23, 59, 59.95), make_book(7_996))
    old_steps = table(rig, "steps")
    assert [row["phase"] for row in old_steps[-5:]] == ["wait"] * 5
    assert all(row["resting_bid"] == row["resting_ask"] == "" and row["quoting"] == "0" for row in old_steps[-5:])
    assert {row["phase"] for row in old_steps} == {"trade", "close", "wait"}
    assert rig.exchange.limits == {} and len(table(rig, "snapshots")) == len(old_steps)

    new_book = make_book(7_995, salt=20)
    rig.tick_at(at(0, 0, 0, day=1), new_book)  # the first step of the new UTC day resets it
    rig.core.finalizer.wait()
    assert_finalized(rig, DATE0)
    assert any(event == "day_end" for event, _ in health(rig, DATE0))
    assert rig.core.date == DATE1 and (rig.settings.bot_dir / DATE1).is_dir()
    ledger = rig.core.ledger
    assert (ledger.position, ledger.cash, ledger.fees) == (0, 0, {"DOGE": 0, "USDT": 0})
    assert (ledger.wealth, ledger.peak, ledger.drawdown) == (0.0, 0.0, 0.0)
    assert ledger.doge_reference == rig.exchange.balance("DOGE") == Decimal("999999.9")  # the balance after the close
    assert rig.core.features.get_state()["initial_mid"] == mid_of(new_book)  # S0 = the new day's first mid
    counters = rig.core.counters
    assert (counters["steps"], counters["fills_buy"], counters["fills_sell"], counters["triggers"]) == (1, 0, 0, {})
    [reset] = events(rig, "reset", DATE1)
    assert Decimal(reset["doge_reference"]) == Decimal("999999.9")
    assert (resting(rig, "buy")[1], resting(rig, "sell")[1]) == expected_b0(new_book, 0, at(0, 0, 0, day=1),
                                                                           D0 + DAY_US)
    [step] = table(rig, "steps", DATE1)
    assert (step["phase"], step["dt"], float(step["t_elapsed"]), step["q"]) == ("trade", "", 0.0, "0")
    rig.tick(dt=1.0)  # the periodic save
    state = state_of(rig)
    assert (state["day"]["date"], state["status"]["new_orders"], state["status"]["wealth"]) == (DATE1, 2, 0.0)


def test_a_flat_bot_closes_at_23_59_50_by_cancelling_and_a_start_inside_the_close_window_waits(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23, 59, 40))
    rig.tick(make_book(8_000), dt=0)
    assert live_sides(rig) == ["buy", "sell"]
    rig.tick_at(at(23, 59, 50), make_book(8_000, salt=1))
    assert rig.exchange.limits == {} and rig.exchange.queued == []
    [done] = events(rig, "close_done", reason="day_close")
    assert done["flat_already"] is True
    nudge(rig, dt=2.0)
    assert [row["phase"] for row in table(rig, "steps")] == ["trade", "wait", "wait"]
    assert len(limit_calls(rig)) == 2

    late = make_rig(tmp_path / "late", start_us=at(23, 59, 55))  # a fresh start after 23:59:50
    late.tick(make_book(8_000), dt=0)
    nudge(late, dt=1.0)
    assert late.exchange.calls == [] and [row["phase"] for row in table(late, "steps")] == ["wait", "wait"]
    late.tick_at(at(0, 0, 0.2, day=1), make_book(8_001))
    assert live_sides(late) == ["buy", "sell"] and late.core.date == DATE1


# --- restarts ------------------------------------------------------------------------------------------------------


def test_restart_within_the_day_restores_balances_s0_window_and_ledger_truncates_the_files_and_spans_the_gap(
        tmp_path, b3_policy):
    rig = make_rig(tmp_path, "B3", policy=b3_policy)
    rng = np.random.default_rng(11)
    bid = 8_000
    rig.tick(make_book(bid), dt=0)
    for k in range(40):
        bid += int(rng.choice([-1, 0, 1]))
        rig.tick(make_book(bid, int(rng.choice([1, 1, 2])), salt=k), dt=0.3)
    cross(rig, "buy", salt=99)  # the paper balances now differ from the seed
    rig.tick(dt=1.0)  # no step; the periodic save records the last one
    saved = state_of(rig)
    sizes = saved["day"]["sizes"]
    paths = {kind: day_file(rig, kind) for kind in KINDS}
    saved_bytes = {kind: paths[kind].read_bytes() for kind in KINDS}
    assert all(len(saved_bytes[kind]) == sizes[kind] for kind in KINDS)
    last_row = saved_bytes["steps"].decode("utf-8").splitlines()[-1]
    assert saved["last_step_us"] == int(last_row.split(",")[0])  # the save came after the last step
    assert Decimal(saved["balances"]["DOGE"]) != SEED["DOGE"] and len(saved["orders"]) == 2

    nudge(rig)  # steps after the save: buffered, lost in the kill
    nudge(rig)
    for path in paths.values():  # rows the state never saw (a write after the last save)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("junk,row\n")
    kill(rig)

    new = restart(rig, gap_s=37.3, policy=b3_policy)
    assert new.exchange.balances == SEED
    new.tick(None, dt=0)  # start-up without a book: the restore, no step
    assert new.exchange.balances == {asset: Decimal(saved["balances"][asset]) for asset in SEED}
    assert new.core.features.get_state() == saved["features"]  # S0 and the window
    assert new.core.ledger.get_state() == saved["ledger"]
    assert new.core.counters["steps"] == saved["counters"]["steps"]
    [restart_row] = events(new, "restart")
    assert sorted(restart_row["lost_orders"]) == sorted(order["order_id"] for order in saved["orders"])
    assert restart_row["seeded_balances"] == {"DOGE": "1000000", "USDT": "100000"}
    assert restart_row["restored_balances"] == saved["balances"] and new.core.counters["lost_orders"] == 2
    for kind in KINDS:
        data = paths[kind].read_bytes()
        assert data[:sizes[kind]] == saved_bytes[kind] and b"junk" not in data
        header = saved_bytes[kind].split(b"\n")[0] + b"\n"
        assert data.count(header) == 1  # no second header
    assert len(table(new, "steps")) == saved["counters"]["steps"]

    book = make_book(bid + 1, salt=300)
    new.tick(book, dt=0.1)
    ts = new.clock.wall_us
    step = table(new, "steps")[-1]
    assert int(step["ts_us"]) == ts and ts - saved["last_step_us"] > 38 * US
    assert float(step["dt"]) == (ts - saved["last_step_us"]) / 1e6  # the gap is the first step's dt
    features = new.core.features.get_state()
    assert features["initial_mid"] == saved["features"]["initial_mid"]
    assert features["dt"][-1] == (ts - saved["last_step_us"]) / 1e6
    assert features["dS"][-1] == mid_of(book) - saved["features"]["last_mid"]
    assert (step["q"], Decimal(step["residual_doge"])) == (saved["status"]["q"], 0)
    assert live_sides(new) == ["buy", "sell"]


def test_a_telemetry_file_shorter_than_the_state_record_is_a_deviation_and_the_restart_proceeds(tmp_path):
    """The state is fsynced, the CSV files only flushed: after a host crash a file can be shorter than the size the
    state recorded (PROTOCOL §5.4 restore; WO-HB2: errors are never silent)."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    for k in range(5):
        nudge(rig, dt=0.3)
    rig.tick(dt=1.0)
    recorded = state_of(rig)["day"]["sizes"]["steps"]
    kill(rig)
    path = day_file(rig, "steps")
    data = path.read_bytes()
    path.write_bytes(data[:data.rstrip(b"\n").rfind(b"\n") + 1])  # the last row did not reach the disk
    shorter = path.stat().st_size

    new = restart(rig, gap_s=20)
    new.tick(make_book(8_001), dt=0)
    [deviation] = events(new, "deviation", event="telemetry_shorter_than_state")
    assert deviation["files"] == [["steps", shorter, recorded]]
    assert live_sides(new) == ["buy", "sell"] and len(table(new, "steps")) == 6


def test_a_failed_day_finalization_is_logged_as_a_deviation_the_bot_runs_on_and_the_next_start_finishes_it(
        tmp_path):
    rig = make_rig(tmp_path, start_us=at(23, 59, 40))
    rig.tick(make_book(8_000), dt=0)
    blocker = rig.settings.bot_dir / DATE0 / f"snapshots_{DATE0}.csv.gz.tmp"
    blocker.mkdir()  # gzip cannot write its temporary file: finalize_day fails
    for k in range(25):  # through 23:59:50 and the 00:00 reset
        rig.tick(make_book(8_000, salt=k + 1), dt=1.0)
    assert rig.core.date == DATE1
    rig.core.finalizer.wait()
    nudge(rig)
    nudge(rig)
    [deviation] = events(rig, "deviation", DATE1, event="finalize_failed")
    assert deviation["date"] == DATE0
    assert live_sides(rig) == ["buy", "sell"] and (rig.settings.bot_dir / DATE0 / f"steps_{DATE0}.csv").exists()
    rig.core.stop()
    blocker.rmdir()
    new = restart(rig, gap_s=10)
    new.tick(make_book(8_001), dt=0)  # the start-up finalizes the leftover day
    new.core.finalizer.wait()
    assert_finalized(new, DATE0)


def test_restart_after_a_missed_close_closes_the_carried_position_at_market_first_then_starts_the_new_day(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23))
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    rig.tick(dt=1.0)
    assert state_of(rig)["ledger"]["position"] == "100"
    kill(rig)

    new = restart(rig, gap_s=(at(1, day=1) - rig.clock.wall_us) / US)
    book = make_book(8_010)
    new.tick(book, dt=0)  # its first step closes the carried position at market
    assert events(new, "deviation", event="missed_close")
    assert events(new, "deviation", event="close_after_midnight")
    [(close_id, side, amount, _)] = new.exchange.queued
    assert (side, amount) == ("sell", Decimal(100)) and new.exchange.limits == {}
    assert new.core.date == DATE0  # the old day stays open until the position is flat
    for k in range(4):
        new.tick(make_book(8_010, salt=k + 1), dt=1.0)
    assert len(new.exchange.queued) == 1 and new.exchange.limits == {}
    first_book = make_book(8_011)
    new.tick(first_book, dt=1.0)  # 5 s: the close executes; the step then starts the new day
    assert new.core.date == DATE1 and new.core.ledger.position == 0
    new.core.finalizer.wait()
    assert_finalized(new, DATE0)
    old_steps = table(new, "steps", DATE0)
    assert old_steps and all(int(row["ts_us"]) < at(0, day=1) for row in old_steps)  # no step until flat
    old_events = [event for event, _ in health(new, DATE0)]
    assert {"restart", "close_submitted", "close_done", "day_end"} <= set(old_events)
    assert events(new, "close_done", DATE0, reason="carry")
    [step] = table(new, "steps", DATE1)
    assert int(step["ts_us"]) == new.clock.wall_us and step["phase"] == "trade"
    assert new.core.ledger.doge_reference == new.exchange.balance("DOGE") == Decimal("999999.9")
    assert new.core.features.get_state()["initial_mid"] == mid_of(first_book)
    assert live_sides(new) == ["buy", "sell"]


def test_halt_cancels_closes_and_persists_across_restarts_until_clear_halt_clears_it(tmp_path):
    rig = make_rig(tmp_path, halt_q_max=1)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1: at the (smoke) limit, not beyond it
    assert not rig.core.safety.halted
    ask_id, _ = resting(rig, "sell")
    cross(rig, "buy")  # q = 2 > 1: halt inside the fill event
    safety = rig.core.safety
    assert safety.halted and safety.halt_reason.startswith("q_ledger")
    assert state_of(rig)["safety"]["halted"] is True  # saved at once, before any later tick
    [halt] = events(rig, "halt", trigger="q_ledger")
    assert halt["kind"] == "deviation"
    assert rig.exchange.limits == {} and ask_id not in rig.exchange.limits
    [(_, side, amount, _)] = rig.exchange.queued
    assert (side, amount) == ("sell", Decimal(200))
    calls = len(rig.exchange.calls)
    for k in range(6):
        rig.tick(make_book(7_990 + k), dt=1.0)
    assert rig.core.ledger.q == 0 and limit_calls(rig, calls) == []
    rig.core.stop()

    second = restart(rig, gap_s=60)  # the flag survives the restart
    for k in range(5):
        second.tick(make_book(8_000 + k), dt=0.5)
    assert second.core.safety.halted and second.exchange.calls == []
    second.core.stop()

    path = state_path(second)
    before = state_of(second)
    assert clear_halt.main([str(path), "--by", "test", "--note", "unit test", "--min-age-s", "0"]) == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    assert (after["safety"]["halted"], after["safety"]["halt_reason"], after["safety"]["halt_ts_us"]) == (
        False, None, None)
    assert (after["status"]["halted"], after["status"]["halt_reason"]) == (False, None)  # what status.ps1 prints
    cleared = ("halted", "halt_reason", "halt_ts_us")
    assert {k: v for k, v in after["safety"].items() if k not in cleared} == {
        k: v for k, v in before["safety"].items() if k not in cleared}
    kept = ("safety", "status")  # the status block only mirrors the state for status.ps1
    assert {k: v for k, v in after.items() if k not in kept} == {k: v for k, v in before.items() if k not in kept}
    header, rows = read_csv(path.with_name("halt_clearances.csv"))
    assert header == ["cleared_utc", "bot", "halt_reason", "halt_ts_us", "by", "note"]
    [row] = rows
    assert row[1:] == ["hb_b0", before["safety"]["halt_reason"], str(before["safety"]["halt_ts_us"]), "test",
                       "unit test"]

    third = restart(second, gap_s=30)  # cleared: the restart quotes again
    third.tick(make_book(8_005), dt=0)
    assert live_sides(third) == ["buy", "sell"] and not third.core.safety.halted
    third.core.stop()  # a graceful stop cancels the quotes before the last save
    assert third.exchange.limits == {} and state_of(third)["orders"] == []

    assert clear_halt.main([str(path), "--by", "test", "--min-age-s", "0"]) == 1  # not halted
    assert clear_halt.main([str(tmp_path / "missing" / "state.json"), "--by", "test"]) == 2  # no state
    young = tmp_path / "young" / "state.json"
    young.parent.mkdir()
    young.write_text(json.dumps({**before, "saved_wall_us": time.time_ns() // 1_000}), encoding="utf-8")
    assert clear_halt.main([str(young), "--by", "test"]) == 3  # saved just now: the bot looks alive
    assert json.loads(young.read_text(encoding="utf-8"))["safety"]["halted"] is True
    assert not (young.parent / "halt_clearances.csv").exists()


def test_a_halt_holds_across_the_daily_reset(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23, 59, 30), halt_q_max=1)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")
    cross(rig, "buy")  # q = 2: halt, close at market
    calls = len(rig.exchange.calls)
    for k in range(40):  # through 23:59:50 and the 00:00 reset
        rig.tick(make_book(7_990 + k % 3, salt=k), dt=1.0)
    assert rig.core.date == DATE1 and rig.core.safety.halted and rig.core.ledger.position == 0
    assert limit_calls(rig, calls) == [] and rig.exchange.limits == {}
    [reset] = events(rig, "reset", DATE1)
    assert reset["halted"] is True


def test_drawdown_kill_switch_closes_and_pauses_until_midnight_across_a_same_day_restart(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23, 50), drawdown_limit_usdt=0.0001)
    rig.tick(make_book(8_000), dt=0)
    assert events(rig, "start")[0]["changed_thresholds"] == {"drawdown_limit_usdt": 0.0001}  # a smoke threshold
    calls = len(rig.exchange.calls)
    cross(rig, "buy")  # q = 1 and the mid below the fill price: W falls more than 0.0001 USDT below its peak
    safety = rig.core.safety
    assert safety.drawdown_paused and not safety.halted and rig.exchange.limits == {}
    assert state_of(rig)["safety"]["drawdown_paused"] is True  # saved at once
    [trigger] = events(rig, "close_and_pause", trigger="drawdown")
    assert trigger["kind"] == "finding" and trigger["drawdown"] > 0.0001
    [(_, side, amount, _)] = rig.exchange.queued
    assert (side, amount) == ("sell", Decimal(100))
    for k in range(6):
        rig.tick(make_book(7_995 - k), dt=1.0)
    assert rig.core.ledger.q == 0 and limit_calls(rig, calls) == []
    assert events(rig, "close_done", reason="drawdown")
    steps = table(rig, "steps")
    assert steps[-1]["drawdown_paused"] == "1" and steps[-1]["quoting"] == "0"  # the day is kept
    rig.tick(dt=1.0)
    kill(rig)

    new = restart(rig, gap_s=120)
    for k in range(5):
        new.tick(make_book(8_000 + k), dt=1.0)
    assert new.core.safety.drawdown_paused and new.exchange.calls == []
    new.tick_at(at(23, 59, 50), make_book(8_010))
    new.tick_at(at(0, 0, 0.5, day=1), make_book(8_011))  # the pause ends at the reset
    assert new.core.date == DATE1 and not new.core.safety.drawdown_paused
    assert live_sides(new) == ["buy", "sell"]


def test_more_than_10_failures_within_60_s_halt(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    for k in range(10):  # failures of market orders paper reports without a message (ids the bot does not hold)
        rig.clock.advance_us(US)
        rig.core.on_failed(f"sell://DOGE-USDT/failed{k:020x}", None)
    assert not rig.core.safety.halted
    rig.clock.advance_us(60 * US)  # the first ten leave the 60 s window
    for k in range(10):
        rig.core.on_failed(f"sell://DOGE-USDT/failed{100 + k:020x}", None)
        rig.clock.advance_us(US)
    assert not rig.core.safety.halted and live_sides(rig) == ["buy", "sell"]
    rig.core.on_failed("sell://DOGE-USDT/failed" + "f" * 20, None)  # the 11th within 60 s
    assert rig.core.safety.halted and rig.core.safety.halt_reason.startswith("rejects")
    [halt] = events(rig, "halt", trigger="rejects")
    assert halt["kind"] == "deviation"
    assert rig.exchange.limits == {} and [(side, amount) for _, side, amount, _ in rig.exchange.queued] == [
        ("sell", Decimal(100))]
    assert rig.core.counters["fails"] == 21
    assert len([row for row in table(rig, "orders") if row["event"] == "fail"]) == 21


def test_a_failure_message_with_a_comma_is_logged_and_counted_like_any_other(tmp_path):
    """A failure is counted towards the >10-in-60-s halt and recorded in orders.csv (WO-HB2 deliverable 2),
    whatever its message says; one failure must not halt the bot (in the shell, a raising hook halts through
    BotCore.fail)."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    rig.core.on_failed("buy://DOGE-USDT/" + "a" * 26, 'Insufficient USDT balance, order "x" failed')
    assert not rig.core.safety.halted and rig.core.counters["fails"] == 1
    assert [row["event"] for row in table(rig, "orders")].count("fail") == 1


def test_clock_offset_warns_above_half_a_second_pauses_above_2_s_resumes_below_and_a_failed_check_is_only_logged(
        tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    offset = {"us": 0}

    def server_ms() -> int:
        if offset["us"] is None:
            raise OSError("api.binance.com unreachable")
        return (rig.core.clock.peek_us() - offset["us"]) // 1_000

    checker = ClockChecker(rig.core.clock.peek_us, rig.core.clock.wall_us, fetch_ms=server_ms)

    def check(us: int | None) -> None:
        offset["us"] = us
        rig.core.on_clock_check(checker.check_once())

    check(400_000)
    check(-600_000)
    assert [detail["detail"] for detail in events(rig, "warn", trigger="clock_offset")] == ["offset -0.6 s"]
    assert live_sides(rig) == ["buy", "sell"] and not rig.core.safety.clock_paused
    live = set(rig.exchange.limits)
    check(3_000_000)  # > 2 s: pause quoting and cancel
    assert rig.core.safety.clock_paused and rig.exchange.limits == {}
    assert events(rig, "pause", trigger="clock_offset")[0]["kind"] == "incident"
    orders = table(rig, "orders")
    assert {row["order_id"] for row in orders if row["event"] == "cancel" and row["reason"] == "clock_offset"} == live
    nudge(rig)
    check(2_000_000)  # not below 2 s: still paused
    nudge(rig)
    check(None)  # a failed check is only logged
    nudge(rig)
    assert rig.core.safety.clock_paused and rig.exchange.limits == {}
    assert events(rig, "clock_check")[-1]["error"] == "OSError('api.binance.com unreachable')"
    check(1_000_000)  # below 2 s: resume; still above 0.5 s: warn
    assert not rig.core.safety.clock_paused and events(rig, "resume", trigger="clock_offset")
    nudge(rig)
    assert live_sides(rig) == ["buy", "sell"]
    assert len(events(rig, "warn", trigger="clock_offset")) == 4
    assert rig.core.counters["clock_checks"] == 6 and rig.core.counters["max_abs_clock_offset_s"] == 3.0
    assert len(events(rig, "clock_check")) == 6


def test_production_refuses_changed_constants_and_restoring_a_state_saved_under_other_settings(tmp_path):
    with pytest.raises(ValueError, match="production thresholds"):
        check_settings(BotSettings(model="B0", mode="production", drawdown_limit_usdt=10.0))
    with pytest.raises(ValueError, match="production thresholds"):
        make_rig(tmp_path / "refused", mode="production", halt_q_max=2)  # BotCore runs the check
    assert check_settings(BotSettings(model="B3", mode="smoke", halt_q_max=2, stale_pause_s=1.0)) == [
        "halt_q_max", "stale_pause_s"]
    assert check_settings(BotSettings(model="B3", mode="production")) == []
    changed_constants = {"q_max": 5, "tick_size": 1e-4, "fee_rate": 0.002, "gamma_b0": 0.9, "order_amount_doge": 50,
                         "requote_min_age_s": 0.5, "requote_min_ticks": 2, "rate_max_per_window": 50,
                         "rate_max_per_day": 1_000, "close_seconds_before_midnight": 20, "paper_doge_balance": 999_999,
                         "feature_window": 50, "sampling_interval_s": 1.0, "skew_ticks_max": 4.0}
    for name, value in changed_constants.items():
        for mode in MODES:
            with pytest.raises(ValueError, match="differ from PROTOCOL.md"):
                check_settings(BotSettings(model="B0", mode=mode, **{name: value}))

    prod = make_rig(tmp_path, mode="production")
    assert prod.settings.bot_dir == tmp_path / "hb_b0"
    prod.tick(make_book(8_000), dt=0)
    prod.core.stop()
    changed = BotSettings(model="B0", mode="production", data_root=str(tmp_path), heartbeat_interval_s=30.0)
    assert check_settings(changed) == []
    other = restart(prod, gap_s=10, settings=changed)
    with pytest.raises(ValueError, match="production settings changed"):
        other.tick(make_book(8_001), dt=0)
    assert other.exchange.calls == []

    smoke = make_rig(tmp_path)
    smoke.tick(make_book(8_000), dt=0)
    smoke.core.stop()
    resumed = restart(smoke, gap_s=10, settings=BotSettings(model="B0", mode="smoke", data_root=str(tmp_path),
                                                            heartbeat_interval_s=30.0))
    resumed.tick(make_book(8_001), dt=0)
    assert events(resumed, "restart")[0]["config_changed"] is True


def test_fail_halts_cancels_closes_and_logs_an_exception_deviation_without_raising(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        assert rig.core.fail("on_tick", error) is None
    safety = rig.core.safety
    assert safety.halted and safety.halt_reason.startswith("exception")
    [deviation] = events(rig, "deviation", event="exception")
    assert (deviation["where"], deviation["error"]) == ("on_tick", "RuntimeError('boom')")
    assert "RuntimeError: boom" in deviation["traceback"]
    assert events(rig, "halt", trigger="exception")[0]["kind"] == "deviation"
    assert rig.exchange.limits == {} and [(side, amount) for _, side, amount, _ in rig.exchange.queued] == [
        ("sell", Decimal(100))]
    assert rig.core.counters["exceptions"] == 1
    calls = len(rig.exchange.calls)
    nudge(rig, dt=6.0)
    assert limit_calls(rig, calls) == [] and rig.core.ledger.q == 0

    broken = make_rig(tmp_path / "broken")  # the halt's own actions fail: fail() still neither raises nor clears
    broken.tick(make_book(8_000), dt=0)
    broken.exchange.fail_cancel = True
    broken.core.fail("did_fill_order", ValueError("bad fill"))
    assert broken.core.safety.halted and broken.core.safety.halt_reason


def test_a_book_thinner_than_five_levels_or_crossed_is_no_step(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    invalid = [
        make_book(8_001, levels=4),
        (make_book(8_002)[0], make_book(8_000)[1]),  # best bid 0.08002 above the best ask 0.08001
        (make_book(8_001)[0], make_book(8_000)[1]),  # locked: best bid = best ask
        (make_book(8_001)[0], ()),
    ]
    for book in invalid:
        rig.tick(book, dt=0.2)
    counters = rig.core.counters
    assert (counters["steps"], counters["skipped_steps"]) == (1, 4)
    assert len(table(rig, "steps")) == len(table(rig, "snapshots")) == 1
    assert len(events(rig, "incident", event="thin_book")) == 1 and counters["thin_books"] == 2
    crossed = events(rig, "crossed_book")  # clarification h: each crossed or locked top of book is counted
    assert [(row["locked"], row["count"]) for row in crossed] == [(False, 1), (True, 2)]
    assert counters["crossed_books"] == 2 and rig.core.state(None)["status"]["crossed_books"] == 2
    five = make_book(8_001, levels=5)  # imbalance_depth levels are enough; the missing ones print empty
    rig.tick(five, dt=0.2)
    assert counters["steps"] == 2
    _, rows = read_csv(day_file(rig, "snapshots"))
    assert rows[-1] == tardis_row(rig.clock.wall_us, rig.clock.wall_us, five) and rows[-1][-1] == ""


def test_a_fill_of_an_unknown_order_is_booked_and_logged_as_a_deviation(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    order_id = rig.exchange.foreign_fill("buy", Decimal("0.07990"), Decimal(100), mid_of(rig.book))
    assert rig.core.ledger.q == 1 and rig.core.counters["unknown_fills"] == 1 and rig.core.counters["fills_buy"] == 1
    [deviation] = events(rig, "deviation", event="fill_of_unknown_order")
    assert (deviation["order_id"], deviation["side"], Decimal(deviation["price"])) == (order_id, "buy", Decimal("0.07990"))
    [fill] = table(rig, "fills")
    assert (fill["order_id"], fill["order_type"], fill["post_ts_us"], fill["mid_at_post"], fill["fee_asset"]) == (
        order_id, "", "", "", "DOGE")
    nudge(rig)
    step = table(rig, "steps")[-1]
    assert (step["q"], Decimal(step["residual_doge"])) == ("1", 0)


def test_connector_events_go_to_the_orders_trades_and_health_files(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    rig.core.on_created("buy://DOGE-USDT/" + "0" * 26)  # an ack for an order the bot does not hold
    rig.core.on_cancelled("sell://DOGE-USDT/" + "0" * 26)
    rig.core.on_public_trade(1_790_000_000.25, 777, 0.08001, 1_234.0, "sell")
    rig.core.on_feed_event("ws_closed", "WARNING", "The websocket connection was closed (code 1006)")
    rig.core.on_book_snapshot(10, 11)
    orders = table(rig, "orders")
    assert [(row["event"], row["reason"]) for row in orders[-2:]] == [("ack", "not_live"), ("cancelled", "not_live")]
    [trade] = table(rig, "trades")
    assert (trade["exchange_ts"], trade["trade_id"], trade["price"], trade["amount"], trade["side"]) == (
        "1790000000.25", "777", "0.08001", "1234.0", "sell")
    assert events(rig, "feed", kind="ws_closed") and events(rig, "book_snapshot", uid=11)
    assert rig.core.counters["reconnects"] == 1 and rig.core.counters["book_snapshots"] == 1


def test_state_file_holds_what_a_restart_and_status_ps1_read_and_is_saved_every_second(tmp_path, b3_policy):
    rig = make_rig(tmp_path, "B3", policy=b3_policy)
    rig.tick(make_book(8_000), dt=0)
    steps_file, stamps = day_file(rig, "steps"), [rig.clock.wall_us]
    for k in range(30):
        rig.tick(make_book(8_000, salt=k + 1), dt=0.1)  # new amounts, the same prices: a step, no fill
        stamps.append(rig.clock.wall_us)
        assert rig.clock.wall_us - state_of(rig)["saved_wall_us"] < US  # replaced at least once per second
        on_disk = steps_file.read_bytes().count(b"\n") - 1  # rows on disk, without flushing here
        assert on_disk >= sum(stamp <= rig.clock.wall_us - US for stamp in stamps)  # flushed at least every second
    assert not state_path(rig).with_name("state.json.tmp").exists()
    assert rig.core.status_text().startswith(f"hb_b3 B3 smoke day {DATE0} steps 31 q 0 ")
    cross(rig, "buy")
    rig.tick(dt=1.0)
    state = state_of(rig)
    assert set(STATUS_PS1) <= set(state["status"])
    assert (state["status"]["q"], state["status"]["halted"], state["status"]["halt_reason"]) == ("1", False, None)
    counters = state["counters"]
    assert {"fills_buy", "fills_sell", "triggers", "round_bid_steps", "round_ask_steps", "round_bid_orders",
            "round_ask_orders"} <= set(counters)
    assert counters["fills_buy"] == 1 and isinstance(counters["triggers"], dict)
    assert state["day"]["date"] == DATE0 and state["last_step_us"] == rig.core.last_step_us
    assert state["features"] == rig.core.features.get_state()
    ledger = state["ledger"]
    assert (ledger["position"], ledger["fees"]["DOGE"], ledger["doge_reference"]) == ("100", "0.100", "1000000")
    assert state["safety"]["halted"] is False
    orders = {order["order_id"]: order for order in state["orders"]}
    assert set(orders) == set(rig.exchange.limits) == set(state["resting"].values())
    for order_id, (side, price, _) in rig.exchange.limits.items():
        order = orders[order_id]
        assert (order["side"], Decimal(order["price"])) == (side, price) and state["resting"][side] == order_id
        assert isinstance(order["post_ts_us"], int) and isinstance(order["mid_at_post"], float)


# --- the step function ----------------------------------------------------------------------------------------


def _book_path(seed: int, rows: int):
    rng = np.random.default_rng(seed)
    bid, books = 8_000, []
    for k in range(rows):
        bid += int(rng.choice([-1, 0, 1]))
        books.append(make_book(bid, int(rng.choice([1, 1, 2, 3])), salt=k))
    return books


def test_model_stepper_b0_quotes_the_frozen_formula_at_tau_T_minus_t_elapsed():
    stepper = ModelStepper("B0", ENGINE, FeatureEngine(), gamma_b0=GAMMA_B0)
    reference = FeatureEngine()
    for k, book in enumerate(_book_path(1, 60)):
        row, t, q = row_of(book), 30_000.0 + 0.37 * k, (k % 21) - 10
        assert book_row(book, 5) == row
        if k == 0:
            decision, obs = stepper.start(row, t, T_DAY, q), reference.reset(row, t, T_DAY, q)
        else:
            decision, obs = stepper.step(row, 0.37, t, T_DAY, q), reference.update(row, 0.37, t, T_DAY, q)
        mid = mid_of(book)
        assert decision.mid == mid and np.array_equal(decision.obs, obs)
        assert decision.quotes == ENGINE.b0(mid, q, T_DAY - t, GAMMA_B0)
        assert decision.rounded == round_away_from_mid(decision.quotes, mid)
        assert decision.z is None and decision.action is None


def test_model_stepper_b3_normalizes_acts_and_quotes_with_the_numpy_policy(b3_policy):
    stepper = ModelStepper("B3", ENGINE, FeatureEngine(), policy=b3_policy)
    reference = FeatureEngine()
    for k, book in enumerate(_book_path(2, 60)):
        row, t, q = row_of(book), 50_000.0 + 0.21 * k, (k % 7) - 3
        if k == 0:
            decision, obs = stepper.start(row, t, T_DAY, q), reference.reset(row, t, T_DAY, q)
        else:
            decision, obs = stepper.step(row, 0.21, t, T_DAY, q), reference.update(row, 0.21, t, T_DAY, q)
        mid = mid_of(book)
        assert np.array_equal(decision.obs, obs)
        assert np.array_equal(decision.z, b3_policy.normalize(obs))
        assert np.array_equal(decision.action, b3_policy.act(obs))
        assert decision.quotes == ENGINE.b3(decision.action, mid, q, T_DAY - t)
        assert decision.rounded == round_away_from_mid(decision.quotes, mid)


def test_model_stepper_rounds_the_mid_clamp_away_from_the_mid_and_the_core_counts_it(tmp_path):
    """Skew +5 ticks pushes B3's bid onto the mid clamp S - tick/2, off the grid at an even spread
    (clarification a): the bid is posted one tick lower and the event is counted per step and per order."""

    policy = constant_policy((0.0, 1.0))
    rig = make_rig(tmp_path, "B3", policy=policy)
    book = make_book(8_000, spread=2)  # the mid 0.08001 is on the grid
    rig.tick(book, dt=0)
    assert resting(rig, "buy")[1] == Decimal("0.08000")  # S - tick/2 = 0.080005, rounded down
    [step] = table(rig, "steps")
    clamp = mid_of(book) - 0.5 * TICK  # the frozen mid clamp, S - tick/2
    assert (float(step["bid_target"]), step["bid_rounded"], step["ask_rounded"]) == (clamp, "1", "0")
    assert (rig.core.counters["round_bid_steps"], rig.core.counters["round_bid_orders"]) == (1, 1)
    assert [row["reason"] for row in table(rig, "orders") if row["event"] == "submit"] == ["quote_rounded", "quote"]
    rig.tick(make_book(8_000, spread=1), dt=0.3)  # an odd spread: the clamp is on the grid
    assert rig.core.counters["round_bid_steps"] == 1


def test_model_stepper_refuses_a_model_without_its_inputs(b3_policy):
    with pytest.raises(ValueError, match="NumpyPolicy"):
        ModelStepper("B3", ENGINE, FeatureEngine())
    with pytest.raises(ValueError, match="gamma_b0"):
        ModelStepper("B0", ENGINE, FeatureEngine())
    with pytest.raises(ValueError, match="B0 or B3"):
        ModelStepper("B1", ENGINE, FeatureEngine(), policy=b3_policy, gamma_b0=GAMMA_B0)


@pytest.mark.parametrize("model", ["B0", "B3"])
def test_steps_rows_equal_the_step_function_on_the_logged_inputs_and_the_snapshot_file_loads_with_the_frozen_loader(
        tmp_path, b3_policy, model):
    policy = b3_policy if model == "B3" else None
    rig = make_rig(tmp_path, model, start_us=at(9, 30), policy=policy)
    rng = np.random.default_rng(5)
    bid = 8_000
    rig.tick(make_book(bid), dt=0)
    for k in range(250):
        draw = rng.random()
        side = "buy" if draw < 0.04 else "sell" if draw < 0.08 else None
        if side is not None and resting(rig, side) is not None:
            cross(rig, side, dt=float(rng.choice([0.1, 1.2])), salt=k)
            bid = int(Decimal(repr(rig.book[0][0][0])) / TICK_D)
        else:
            bid += int(rng.choice([-1, 0, 1]))
            rig.tick(make_book(bid, int(rng.choice([1, 1, 1, 2])), salt=k), dt=float(rng.choice([0.1, 0.2, 0.7])))
    rig.tick(dt=1.0)
    steps = table(rig, "steps")
    assert len(steps) > 200 and len({row["q"] for row in steps}) > 2

    S, dt, _, _ = load_single_day_with_features(str(day_file(rig, "snapshots")), tick_size=TICK)
    assert len(S) == len(steps)
    ts = [int(row["ts_us"]) for row in steps]
    assert ts == sorted(ts)
    assert [float(row["mid"]) for row in steps] == S.tolist()
    assert [float(row["dt"]) for row in steps[1:]] == dt[1:].tolist()  # dt as the frozen loader derives it

    _, rows = read_csv(day_file(rig, "snapshots"))
    columns = row_columns(5)
    index = {name: TARDIS_HEADER.index(name) for name in columns}
    engine = FeatureEngine()
    for k, (step, snapshot) in enumerate(zip(steps, rows, strict=True)):
        row = {name: float(snapshot[index[name]]) for name in columns}
        t, q = (ts[k] - D0) / 1e6, int(Decimal(step["q"]))
        assert float(step["t_elapsed"]) == t and float(step["tau"]) == T_DAY - t
        obs = engine.reset(row, t, T_DAY, q) if k == 0 else engine.update(row, (ts[k] - ts[k - 1]) / 1e6, t, T_DAY, q)
        mid = engine.get_state()["last_mid"]
        if model == "B3":
            z, action = policy.normalize(obs), policy.act(obs)
            assert [float(step[name]) for name in FORMULA_FEATURE_NAMES] == [float(value) for value in obs]
            assert [float(step[f"z_{name}"]) for name in FORMULA_FEATURE_NAMES] == [float(value) for value in z]
            assert (float(step["action_gamma"]), float(step["action_skew"])) == tuple(float(v) for v in action)
            quotes = ENGINE.b3(action, mid, q, T_DAY - t)
        else:
            quotes = ENGINE.b0(mid, q, T_DAY - t, GAMMA_B0)
        rounded = round_away_from_mid(quotes, mid)
        logged = tuple(float(step[name]) for name in ("gamma", "skew", "bid_target", "ask_target", "bid_quote",
                                                         "ask_quote"))
        assert logged == (quotes["gamma"], quotes["skew"], quotes["bid"], quotes["ask"], rounded.bid, rounded.ask)
        assert (step["bid_rounded"], step["ask_rounded"]) == (str(int(rounded.bid_rounded)),
                                                              str(int(rounded.ask_rounded)))


# --- the clock ------------------------------------------------------------------------------------------------------


def test_ts_follows_the_monotonic_clock_between_anchors_and_the_day_switches_on_the_wall_clock(tmp_path):
    """Clarification c, with k: ts_us = the wall clock at the anchor + the monotonic time since it, non-decreasing
    and inside the UTC day of the files it is written to; local_timestamp is the wall clock at receipt. The 23:59:50
    close and the day switch follow the wall clock, and the new day is anchored on the wall clock with no floor:
    the step clock's lead is dropped at the switch, not held. A wall-clock step below the 2 s limits leaves ts on
    the monotonic clock until then (a larger forward one re-anchors at once:
    test_a_host_sleep_reanchors_the_step_clock_as_downtime_without_a_stale_halt)."""

    rig = make_rig(tmp_path, start_us=at(23, 59, 30))
    rig.tick(make_book(8_000), dt=0)  # k = 0 at 23:59:30; one tick per second, each with a new top 25
    for k in range(1, 35):
        if k == 11:
            rig.clock.shift_wall_us(-1_500_000)  # an NTP step of the wall clock: the step clock now leads by 1.5 s
        rig.tick(make_book(8_000 + k % 2, salt=k), dt=1.0)
    assert rig.core.date == DATE1 and rig.core.counters["clock_jumps"] == 0
    rig.core.finalizer.wait()
    _, old = read_csv(day_file(rig, "snapshots", DATE0))
    new = [[row[name] for name in TARDIS_HEADER] for row in table(rig, "snapshots", DATE1)]
    stamps = [int(row[2]) for row in old + new]
    receipts = [int(row[3]) for row in old + new]
    assert len(old) == 32 and len(new) == 3  # the switch at k = 32, the first tick with the wall clock past 00:00
    assert stamps == sorted(stamps)  # non-decreasing across the switch
    # the old day: ts = the anchor + the monotonic time, kept inside the day once the step clock passes 00:00 first
    assert stamps[:30] == [at(23, 59, 30 + k) for k in range(30)] and stamps[30:32] == [at(0, day=1) - 1] * 2
    assert receipts == [at(23, 59, 30 + k) for k in range(11)] + [at(23, 59, 28.5 + k) for k in range(11, 35)]
    # the 23:59:50 close on the wall clock (k = 22, ts 23:59:52), then the wait until the wall clock's 00:00
    assert [row["phase"] for row in table(rig, "steps", DATE0)] == ["trade"] * 22 + ["wait"] * 10
    # the new day starts on the wall clock: the 1.5 s lead is dropped, not held (no Δt = 0 run)
    assert stamps[32:] == [at(0, 0, 0.5, day=1), at(0, 0, 1.5, day=1), at(0, 0, 2.5, day=1)]
    assert [row["dt"] for row in table(rig, "steps", DATE1)] == ["", "1.0", "1.0"]
    [anchor] = events(rig, "clock_anchor", DATE1)
    assert anchor["lead_dropped_s"] == 1.5 and anchor["ts_us"] == at(0, 0, 0.5, day=1)
    assert anchor["old_day_last_us"] == at(0, day=1) - 1


# --- clarification k: the day boundary ------------------------------------------------------------------------------


class CreepingClock(FakeClock):
    """Time moves on while the bot works: every read of the monotonic clock advances both clocks by 1 µs (FakeClock
    stands still within a tick)."""

    def monotonic_ns(self) -> int:
        self._monotonic_ns += 1_000
        self._wall_ns += 1_000
        return self._monotonic_ns


def assert_day_files_inside(rig: Rig, date: str, day: int) -> None:
    """Clarification k: in each of the day's six files ts_us is non-decreasing in file order and inside that UTC day."""

    start, end = at(0, day=day), at(0, day=day + 1)
    for kind in KINDS:
        stamps = [int(row["timestamp" if kind == "snapshots" else "ts_us"]) for row in table(rig, kind, date)]
        assert stamps == sorted(stamps), (date, kind)
        assert all(start <= stamp < end for stamp in stamps), (date, kind)


# Clarification j: the step clock's rate correction c (ppm). k's day tests run with c = 0 (the clock before j, bit for
# bit, so their exact values hold) and with c = 12.8, where the step clock runs 12.8 ppm slower than the fake
# monotonic clock: a time the test computes from the wall clock is then within C_TOLERANCE_US (a test spans < 30 s).
RATES = ["0", "12.8"]
C_TOLERANCE_US = {"0": 0, "12.8": 400}


@pytest.mark.parametrize("rate_ppm", RATES, ids=lambda rate: f"c={rate}")
@pytest.mark.parametrize("lead_s", [1.5, -1.5], ids=["step_clock_leading", "step_clock_lagging"])
def test_the_day_switch_follows_the_wall_clock_with_the_step_clock_1_5_s_off_at_23_59_59(tmp_path, b3_policy, lead_s,
                                                                                        rate_ppm):
    """Clarification k, the author's tests: with the step clock 1.5 s ahead of or behind the wall clock at 23:59:59,
    the 23:59:50 close and the day switch follow the wall clock; each day's files keep ts non-decreasing and inside
    that UTC day; the new day is anchored on the wall clock, so no lead is carried: B3's first step of the day is
    the frozen loader's row 0 (S0 its own mid, an empty window) and no later step has Δt = 0. Clarification j: the
    same with the rate correction c = 12.8 ppm (Δt then 0.5 s / (1 + 12.8e-6), to the µs)."""

    tolerance = C_TOLERANCE_US[rate_ppm]
    rig = make_rig(tmp_path, "B3", start_us=at(23, 59, 40), policy=b3_policy, rate_ppm=rate_ppm)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1 at 23:59:41.1: the 23:59:50 close has a position to close
    rig.clock.shift_wall_us(round(-lead_s * US))  # the step clock now leads (lags) the wall clock by 1.5 s
    k = 0
    while rig.clock.wall_us < at(0, 0, 5, day=1):
        k += 1
        rig.tick(make_book(8_000 + k % 2, salt=k), dt=0.5)
    rig.core.finalizer.wait()
    assert rig.core.date == DATE1 and rig.core.counters["clock_jumps"] == 0
    assert_day_files_inside(rig, DATE0, 0)
    assert_day_files_inside(rig, DATE1, 1)
    # the close at the first tick with the wall clock at 23:59:50 (wall 23:59:50.1), whatever the step clock read
    [close] = [row for row in table(rig, "orders", DATE0) if row["event"] == "submit" and row["order_type"] == "market"]
    assert abs(int(close["ts_us"]) - (at(23, 59, 50.1) + round(lead_s * US))) <= tolerance
    # the switch at the first tick with the wall clock in the new day; the old day's snapshots hold no later receipt
    [anchor] = events(rig, "clock_anchor", DATE1)
    assert anchor["ts_us"] == anchor["wall_us"] == at(0, 0, 0.1, day=1)
    assert abs(anchor["lead_dropped_s"] - lead_s) <= tolerance / US
    assert max(int(row["local_timestamp"]) for row in table(rig, "snapshots", DATE0)) < at(0, day=1)
    held = sum(int(row["ts_us"]) == at(0, day=1) - 1 for row in table(rig, "steps", DATE0))
    assert held == (3 if lead_s > 0 else 0)  # leading: the old day's last steps stay inside it
    steps = table(rig, "steps", DATE1)
    assert int(steps[0]["ts_us"]) == at(0, 0, 0.1, day=1) and steps[0]["dt"] == ""
    assert len(steps) == 11  # no Δt = 0 run at the day's start
    if rate_ppm == "0":
        assert {row["dt"] for row in steps[1:]} == {"0.5"}
    else:
        assert all(abs(float(row["dt"]) - 0.5 / (1 + float(rate_ppm) * 1e-6)) <= 1e-6 for row in steps[1:])
    first = {name: float(steps[0][name]) for name in ("log_mid_rel", "rolling_vol_ticks", "momentum_ticks")}
    assert first == {"log_mid_rel": 0.0, "rolling_vol_ticks": 0.0, "momentum_ticks": 0.0}  # row 0: a reset
    [end] = events(rig, "day_end", DATE0)  # the old day was closed, and flat, before the switch
    assert end["closed"] is True and [e["position"] for e in events(rig, "close_done", DATE0, reason="day_close")] == ["0"]


@pytest.mark.parametrize("rate_ppm", RATES, ids=lambda rate: f"c={rate}")
def test_every_days_files_stay_non_decreasing_when_time_moves_on_within_a_tick(tmp_path, rate_ppm):
    """Clarification k: the paper connector fires a cancel's event inside the bot's own call; the event shares the
    tick's ts, so a re-quote's cancel, cancelled and submit rows stay in order while time moves on within the tick
    (fills, the 23:59:50 close and the day switch included). Clarification j: also with c = 12.8 ppm."""

    rig = Rig(BotSettings(model="B0", mode="smoke", data_root=str(tmp_path)), CreepingClock(at(23, 59, 25)),
              rate_ppm=rate_ppm)
    rig.tick(make_book(8_000), dt=0)
    for k in range(1, 60):  # the mid moves 3 ticks a tick: re-quotes of orders older than 1 s
        if k in (4, 12):
            cross(rig, "buy" if k == 4 else "sell", dt=0.7, salt=k)
        else:
            rig.tick(make_book(8_000 + 3 * (k % 3), salt=k), dt=0.7)
    rig.core.finalizer.wait()
    assert rig.core.date == DATE1
    orders = table(rig, "orders", DATE0)
    assert [row["event"] for row in orders].count("cancelled") >= 5 and len(table(rig, "fills", DATE0)) >= 2
    assert_day_files_inside(rig, DATE0, 0)
    assert_day_files_inside(rig, DATE1, 1)


@pytest.mark.parametrize("rate_ppm", RATES, ids=lambda rate: f"c={rate}")
def test_a_close_carried_past_midnight_stays_in_the_old_days_files_and_the_new_day_starts_on_the_wall_clock(tmp_path,
                                                                                                          rate_ppm):
    """No tick across 23:59:50 (the host was busy): the position is closed at market first (§5.4); its rows stay
    inside the old day, its order counts in the new UTC day's cap (§5.3), and the new day starts on the wall clock
    (clarification j: also with c = 12.8 ppm)."""

    rig = make_rig(tmp_path, start_us=at(23, 59), rate_ppm=rate_ppm)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    bid_ticks = int(Decimal(repr(rig.book[0][0][0])) / TICK_D)
    rig.tick_at(at(0, 0, 0.5, day=1), make_book(bid_ticks, salt=1))  # the carry close is submitted
    assert rig.core.carry and rig.core.date == DATE0 and rig.core.rate.day_orders == 1
    rig.tick(make_book(bid_ticks, salt=2), dt=5.1)  # it fills (paper: 5 s); the day switches at this tick
    rig.core.finalizer.wait()
    assert rig.core.date == DATE1 and rig.core.ledger.position == 0
    assert_day_files_inside(rig, DATE0, 0)
    assert_day_files_inside(rig, DATE1, 1)
    carry = [row for row in table(rig, "orders", DATE0) if row["reason"] == "carry"]
    assert carry and {int(row["ts_us"]) for row in carry} == {at(0, day=1) - 1}
    [anchor] = events(rig, "clock_anchor", DATE1)
    assert anchor["ts_us"] == at(0, 0, 5.6, day=1)


@pytest.mark.parametrize("rate_ppm", RATES, ids=lambda rate: f"c={rate}")
def test_a_restart_after_midnight_writes_the_old_days_last_rows_inside_it(tmp_path, rate_ppm):
    """A stop before 00:00 and a start after it: the restored day's rows (the restart) stay inside that day, and
    the switch at the first tick starts the new day on the wall clock (clarification j: also with c = 12.8 ppm)."""

    rig = make_rig(tmp_path, start_us=at(23, 59, 40), rate_ppm=rate_ppm)
    rig.tick(make_book(8_000), dt=0)
    nudge(rig, dt=1.1)
    rig.core.stop()
    new = restart(rig, gap_s=30)  # back at 00:00:11.1
    new.tick(make_book(8_001), dt=0)
    new.core.finalizer.wait()
    assert new.core.date == DATE1
    [restarted] = events(new, "restart", DATE0)
    # the real downtime, from the unheld step clock (with c, the old process's last ts lagged its wall clock by c)
    assert restarted["gap_s"] == pytest.approx(30.0, abs=1e-6 + C_TOLERANCE_US[rate_ppm] / US)
    assert_day_files_inside(new, DATE0, 0)
    assert_day_files_inside(new, DATE1, 1)
    [anchor] = events(new, "clock_anchor", DATE1)
    assert anchor["ts_us"] == at(0, 0, 11.1, day=1)


@pytest.mark.parametrize("action", ["fail", "clock_pause", "fill_halt", "stop"])
def test_an_entry_point_outside_a_tick_keeps_the_orders_file_in_order(tmp_path, action):
    """Clarification k: fail(), a §7 clock pause, a q-ledger halt from a fill and stop() cancel both resting quotes
    outside a tick; the paper connector's synchronous cancel events share the entry point's ts, so the orders file
    stays non-decreasing while time moves on (CreepingClock)."""

    overrides = {"halt_q_max": 1} if action == "fill_halt" else {}
    rig = Rig(BotSettings(model="B0", mode="smoke", data_root=str(tmp_path), **overrides), CreepingClock(at(10)))
    rig.tick(make_book(8_000), dt=0)
    nudge(rig, dt=1.1)  # both quotes resting and acknowledged
    assert live_sides(rig) == ["buy", "sell"]
    if action == "fail":
        rig.core.fail("on_tick", RuntimeError("probe"))
    elif action == "clock_pause":
        rig.core.on_clock_check({"error": None, "server_ms": 1, "offset_s": 3.0, "wall_offset_s": 3.0, "rtt_s": 0.3,
                                 "wall_us": 1, "rate_step": None, "rate_monotonic": None, "rate_over_s": None})
    elif action == "fill_halt":
        cross(rig, "buy")  # q = 1
        nudge(rig, dt=1.1)  # the bid is posted again
        cross(rig, "buy")  # q = 2 > halt_q_max: the halt fires inside the fill event
        assert rig.core.safety.halted
    else:
        rig.core.stop()
    if not rig.core.stopped:  # stop() has flushed and closed the files itself
        rig.core.log.flush()
    _, rows = read_csv(day_file(rig, "orders"))
    stamps = [int(row[0]) for row in rows]
    assert stamps == sorted(stamps) and [row[2] for row in rows].count("cancelled") >= 2
    assert live_sides(rig) == []


def test_a_wall_clock_stepped_back_across_midnight_during_a_carry_neither_raises_nor_halts(tmp_path):
    """The order cap's day and the switch follow the day's highest wall reading (clarification k): a wall clock
    stepped back across 00:00 while a carried close runs cannot move the cap's day back (RateLimiter would raise and
    halt the bot). The carry cancels the old day's quotes first (§5.5), the state is still saved every second while
    ts is held, and the §7 check reads the unheld step clock."""

    rig = make_rig(tmp_path, start_us=at(23, 59))
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1 with both quotes resting (the bid posted again at the next step)
    nudge(rig, dt=1.1)
    assert live_sides(rig) == ["buy", "sell"]
    bid_ticks = int(Decimal(repr(rig.book[0][0][0])) / TICK_D)
    rig.tick_at(at(0, 0, 0.5, day=1), make_book(bid_ticks, salt=1))  # no tick across 23:59:50: a carry
    assert rig.core.carry and live_sides(rig) == [] and len(rig.exchange.queued) == 1  # quotes cancelled, one close
    saved = state_of(rig)["saved_wall_us"]
    rig.clock.shift_wall_us(-1_000_000)  # the wall clock steps back across 00:00, to 23:59:59.6 at the next tick
    rig.tick(make_book(bid_ticks, salt=2), dt=0.1)
    assert rig.clock.wall_us < at(0, day=1) and not rig.core.safety.halted and rig.core.carry
    assert rig.core._cap_day_us() // DAY_US == rig.core.rate.day == at(0, day=1) // DAY_US  # the cap's day stays
    rig.tick(make_book(bid_ticks, salt=2), dt=1.1)
    assert not rig.core.safety.halted and rig.core.carry and rig.core.date == DATE0
    assert state_of(rig)["saved_wall_us"] > saved  # saves go on while ts is held at the old day's end
    assert rig.core.clock.peek_us() == at(0, day=1) - 1 and rig.core.clock.unheld_us() > at(0, day=1)
    for _ in range(5):
        rig.tick(make_book(bid_ticks, salt=3), dt=1.1)
    assert rig.core.date == DATE1 and rig.core.ledger.position == 0 and not rig.core.safety.halted
    assert [call[2] for call in rig.exchange.calls if call[0] == "market"] == ["sell"]
    rig.core.finalizer.wait()
    assert_day_files_inside(rig, DATE0, 0)
    assert_day_files_inside(rig, DATE1, 1)


def test_a_wall_clock_stepped_back_after_the_close_posts_no_quote_before_midnight(tmp_path):
    """§5.5: after the 23:59:50 close nothing is posted until 00:00. The phase follows the day's highest wall reading
    (clarification k), so a wall clock stepped back below 23:59:50 does not reopen trading."""

    rig = make_rig(tmp_path, start_us=at(23, 59, 45))
    rig.tick(make_book(8_000), dt=0)
    for k in range(1, 8):  # past 23:59:50: the close (flat: done at once)
        rig.tick(make_book(8_000 + k % 2, salt=k), dt=1.0)
    assert rig.core.closed and live_sides(rig) == []
    since = len(rig.exchange.calls)
    rig.clock.shift_wall_us(-4_000_000)  # the wall clock back to 23:59:48
    for k in range(8, 14):
        rig.tick(make_book(8_000 + k % 2, salt=k), dt=0.5)
    assert limit_calls(rig, since) == [] and rig.core.date == DATE0
    assert {row["phase"] for row in table(rig, "steps")[8:]} == {"wait"}


def test_a_flat_bot_asleep_across_the_close_cancels_its_quotes_in_the_old_day_and_logs_the_missed_close(tmp_path):
    """§5.5 step 1 when the 23:59:50 close never ran and the bot is flat (a host sleep): the old day's quotes are
    cancelled at the switch, in the old day's files, and the missed close is a deviation; no old-day order lives
    into the new day (clarification k)."""

    rig = make_rig(tmp_path, start_us=at(23, 59, 30))
    rig.tick(make_book(8_000), dt=0)
    nudge(rig, dt=1.1)  # both quotes resting and acknowledged
    old = {order_id for order_id in rig.exchange.limits}
    rig.clock.shift_wall_us(40 * US)  # asleep from 23:59:31.1 to 00:00:11.2
    rig.tick(dt=0.1)  # the wake tick: no book change, so no step of the new day yet
    assert rig.core.date == DATE1 and live_sides(rig) == [] and rig.core.ledger.position == 0
    rig.core.finalizer.wait()
    cancelled = [row for row in table(rig, "orders", DATE0) if row["event"] == "cancelled"]
    assert {row["order_id"] for row in cancelled} == old
    [missed] = events(rig, "deviation", DATE0, event="missed_close")
    assert missed["position"] == "0"
    rig.tick(crossing_book(Decimal(repr(rig.book[0][0][0])), "buy"), dt=1.1)  # would have filled the old bid
    assert rig.core.ledger.position == 0 and not table(rig, "fills", DATE1)
    assert_day_files_inside(rig, DATE0, 0)
    assert_day_files_inside(rig, DATE1, 1)


def test_a_closed_day_restored_with_the_wall_clock_behind_the_close_posts_no_quote(tmp_path):
    """§5.5 step 4 across a restart: a day already closed stays in the wait phase until 00:00 whatever the wall clock
    reads (here 6 s before the stop, below 23:59:50), so nothing is posted after the close."""

    rig = make_rig(tmp_path, start_us=at(23, 59, 48))
    rig.tick(make_book(8_000), dt=0)
    for k in range(1, 5):  # past 23:59:50: the close (flat: done at once)
        rig.tick(make_book(8_000 + k % 2, salt=k), dt=1.0)
    assert rig.core.closed
    rig.core.stop()
    new = restart(rig, gap_s=-6)  # the wall clock reads 23:59:46
    for k in range(5, 12):
        new.tick(make_book(8_000 + k % 2, salt=k), dt=0.5)
    assert new.clock.wall_us < at(0, day=1) and new.core.closed and limit_calls(new) == []
    assert {row["phase"] for row in table(new, "steps")[5:]} == {"wait"}


def test_a_restart_just_before_the_close_with_a_leading_step_clock_logs_no_missed_close(tmp_path):
    """The 23:59:50 close follows the wall clock (clarification k), so a restart at wall 23:59:49.4 is no missed close
    even if the saved step clock had already passed 23:59:50; the close comes at the wall clock's 23:59:50."""

    rig = make_rig(tmp_path, start_us=at(23, 59, 40))
    rig.tick(make_book(8_000), dt=0)
    rig.clock.shift_wall_us(-1_500_000)  # the step clock leads by 1.5 s
    rig.tick_at(at(23, 59, 48.6), make_book(8_001))  # the step clock reads 23:59:50.1
    rig.core.stop()
    new = restart(rig, gap_s=0.8)  # wall 23:59:49.4
    new.tick(make_book(8_002), dt=0)
    assert not events(new, "deviation", event="missed_close") and not new.core.closed
    new.tick(make_book(8_001), dt=0.7)  # wall 23:59:50.1: the close
    assert new.core.closed


def test_a_host_sleep_reanchors_the_step_clock_as_downtime_without_a_stale_halt(tmp_path):
    """Observed in the WO-HB2 trial run: a sleeping host stops the Docker VM's monotonic clock while the wall
    clock moves on, so the step clock fell 4.7 h behind. More than 2 s apart, the bot re-anchors on the wall
    clock at once, logs the gap as downtime and restarts the stale-book timer (no stale halt from the gap)."""

    rig = make_rig(tmp_path, start_us=at(10))
    rig.tick(make_book(8_000), dt=0)
    nudge(rig, dt=1.1)
    last_step = rig.core.last_step_us
    rig.clock.shift_wall_us(4 * 3_600 * US)  # asleep 4 h: the wall clock jumps, the monotonic clock does not
    rig.tick(make_book(8_002), dt=0.1)
    [jump] = events(rig, "deviation", event="clock_jump")
    assert jump["jump_s"] == 4 * 3_600.0 and rig.core.counters["clock_jumps"] == 1
    assert not rig.core.safety.halted and not rig.core.safety.stale_paused and rig.core.safety.quoting
    last = table(rig, "steps")[-1]
    assert abs(int(last["ts_us"]) - rig.clock.wall_us) < US  # the step clock is back on the wall clock
    assert float(last["dt"]) == (int(last["ts_us"]) - last_step) / 1e6 > 4 * 3_600  # the gap is the step's dt
    assert len(live_sides(rig)) == 2


def test_a_sleep_across_midnight_closes_the_carried_position_first_then_starts_the_new_day(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23))
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    nudge(rig, dt=1.1)
    rig.clock.shift_wall_us(2 * 3_600 * US)  # asleep from 23:00 until about 01:00: the 23:59:50 close is missed
    bids, asks = rig.book  # an amount changes, the prices do not: nothing else fills
    rig.tick((((bids[0][0], bids[0][1] + 2.0),) + bids[1:], asks), dt=0.1)
    assert events(rig, "deviation", event="clock_jump") and events(rig, "deviation", event="close_after_midnight")
    assert rig.core.carry and [call[2] for call in rig.exchange.calls if call[0] == "market"] == ["sell"]
    assert live_sides(rig) == []  # §5.5: the old day's quotes are cancelled before the carried close (k)
    woke = rig.clock.wall_us
    rig.tick((((bids[0][0], bids[0][1] + 3.0),) + bids[1:], asks), dt=5.1)  # the close executes; the day switches
    assert rig.core.date == DATE1 and rig.core.ledger.position == 0 and not rig.core.carry
    # the RUN_LOG line carries the wake time (the wall clock), not the old day's held ts (clarification k)
    from hummingbot_adapter.telemetry import utc_iso
    [line] = (rig.settings.bot_dir / "run_log_incidents.md").read_text(encoding="utf-8").splitlines()
    assert line.startswith(f"{utc_iso(woke)[:19]}Z: hb_b0 (smoke): incident: clock re-anchor (clarification g)")
    rig.core.finalizer.wait()
    assert_day_files_inside(rig, DATE0, 0)
    assert_day_files_inside(rig, DATE1, 1)


def test_a_step_clock_ahead_of_the_wall_clock_keeps_ts_and_is_one_clock_backward_incident_per_episode(tmp_path):
    """Clarification i: g re-anchors forward only. When the step clock leads the wall clock by more than 2 s
    (here the wall clock is stepped back), ts stays on the step clock: no re-anchor and no hold. One
    clock_backward incident per episode, one more when the lead grows by a further 2 s; a lead back at 1 s or
    less ends the episode. Each incident is a RUN_LOG line and a count in the status block."""

    from hummingbot_adapter import run_log

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    rig.clock.shift_wall_us(-1_500_000)  # 1.5 s apart: within the limit
    nudge(rig)
    rig.clock.shift_wall_us(-500_000)  # exactly 2 s: still within it (the rule is more than 2 s)
    nudge(rig)
    assert rig.core.counters["clock_jumps"] == rig.core.counters["clock_backward"] == 0
    rig.clock.shift_wall_us(-500_000)  # the step clock 2.5 s ahead
    before = rig.core.clock.last_us
    for k in range(1, 4):
        nudge(rig)
        assert rig.core.clock.last_us == before + k * 200_000  # ts keeps the step clock: no hold, no step back
    assert rig.core.counters["clock_jumps"] == 0 and rig.core.counters["clock_backward"] == 1
    [incident] = events(rig, "incident", event="clock_backward")
    assert incident["lead_s"] == 2.5 and incident["step_clock_us"] == incident["wall_us"] + 2_500_000
    rig.clock.shift_wall_us(-1_900_000)  # 4.4 s: grown by less than a further 2 s, the same episode
    nudge(rig)
    assert rig.core.counters["clock_backward"] == 1
    rig.clock.shift_wall_us(-200_000)  # 4.6 s: grown by more than 2 s since the incident
    nudge(rig)
    assert rig.core.counters["clock_backward"] == 2
    rig.clock.shift_wall_us(3_700_000)  # back to 0.9 s ahead: the episode ends
    nudge(rig)
    rig.clock.shift_wall_us(-1_200_000)  # 2.1 s ahead: a new episode
    nudge(rig)
    assert rig.core.counters["clock_backward"] == 3 and rig.core.counters["clock_jumps"] == 0
    assert [row["dt"] for row in table(rig, "steps")] == [""] + ["0.2"] * 9  # every step kept its tick's Δt
    assert rig.core.state(None)["status"]["clock_backward"] == 3 and not rig.core.safety.clock_paused
    lines = (rig.settings.bot_dir / "run_log_incidents.md").read_text(encoding="utf-8").splitlines()
    assert [line.split(": hb_b0 (smoke): incident: ")[1] for line in lines] == [
        f"clock backward (clarification i): the step clock led the wall clock by +{lead} s; ts kept, never moved "
        "back; left to the §7 offset check; minutes affected 0.00." for lead in ("2.500", "4.600", "2.100")]
    log = tmp_path / "RUN_LOG.md"
    log.write_text("# Run log (UTC)\n", encoding="utf-8", newline="\n")
    assert run_log.main(["--data-root", str(tmp_path), "--smoke", "--run-log", str(log)]) == 0
    appended = "".join(f"\n{line}\n" for line in sorted(lines))  # run_log.py: in time order (to the second)
    assert log.read_text(encoding="utf-8") == "# Run log (UTC)\n" + appended


def test_ts_us_never_decreases_in_a_step_snapshot_or_fill_row_when_the_wall_clock_steps_back_3_s(tmp_path):
    """Clarification i, the author's acceptance test: the fake wall clock steps back by 3 s between fills on both
    sides. ts_us is non-decreasing in every steps, snapshots and fills row, every step keeps its tick's Δt, and the
    receipt times (the wall clock) show the step back."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)  # both sides posted
    cross(rig, "buy")
    nudge(rig, dt=1.1)  # the filled side is posted again at the next step
    cross(rig, "sell")
    nudge(rig, dt=1.1)
    fills_before = len(table(rig, "fills"))
    rig.clock.shift_wall_us(-3 * US)
    nudge(rig, dt=1.1)
    cross(rig, "buy")
    nudge(rig, dt=1.1)
    cross(rig, "sell")
    nudge(rig, dt=1.1)
    assert fills_before == 2 and len(table(rig, "fills")) == 4  # two fills on each side of the step back
    for kind, column in (("steps", "ts_us"), ("snapshots", "timestamp"), ("fills", "ts_us")):
        stamps = [int(row[column]) for row in table(rig, kind)]
        assert stamps and stamps == sorted(stamps), kind
    steps = table(rig, "steps")
    assert len(steps) == 10 and {row["dt"] for row in steps[1:]} == {"1.1"}  # neither moved back nor held
    receipts = [int(row["local_timestamp"]) for row in table(rig, "snapshots")]
    assert sum(later < earlier for earlier, later in zip(receipts, receipts[1:])) == 1  # the wall clock stepped back
    assert rig.core.counters["clock_backward"] == 1 and rig.core.counters["clock_jumps"] == 0
    assert not rig.core.safety.halted and not rig.core.safety.clock_paused


def test_a_wall_clock_more_than_2_s_ahead_reanchors_and_the_step_spans_the_gap_but_2_s_does_not(tmp_path):
    """Clarification g, forward only (i): the bot re-anchors when the wall clock is MORE than 2 s ahead of the step
    clock (a sleep or a freeze), and the next step's Δt spans the gap; at 2 s ts stays on the step clock."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    rig.clock.shift_wall_us(2_000_000)  # the wall clock exactly 2 s ahead: within the limit
    nudge(rig)
    assert rig.core.counters["clock_jumps"] == 0 and table(rig, "steps")[-1]["dt"] == "0.2"
    assert not (rig.settings.bot_dir / "run_log_incidents.md").exists()
    rig.clock.shift_wall_us(100_000)  # 2.1 s ahead: re-anchored on the wall clock
    nudge(rig)
    [jump] = events(rig, "deviation", event="clock_jump")
    assert rig.core.counters["clock_jumps"] == 1 and jump["jump_s"] == 2.1
    assert table(rig, "steps")[-1]["dt"] == "2.3"  # the 0.2 s tick plus the 2.1 s gap
    [line] = (rig.settings.bot_dir / "run_log_incidents.md").read_text(encoding="utf-8").splitlines()
    assert "clock re-anchor (clarification g): the step clock and the wall clock differed by +2.100 s" in line
    assert rig.core.counters["clock_backward"] == 0


def test_a_forward_reanchor_ends_an_open_clock_backward_episode(tmp_path):
    """A re-anchor ends a clock_backward episode at once: a step back right after it is a new episode, logged
    even though it is smaller than the earlier lead plus 2 s."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    rig.clock.shift_wall_us(-2_500_000)  # the step clock 2.5 s ahead: the first episode
    nudge(rig)
    rig.clock.shift_wall_us(5_000_000)  # then the wall clock 2.5 s ahead: a forward re-anchor
    nudge(rig)
    rig.clock.shift_wall_us(-3_000_000)  # at once 3 s behind again: a new episode
    nudge(rig)
    assert rig.core.counters["clock_jumps"] == 1 and rig.core.counters["clock_backward"] == 2
    assert [event["lead_s"] for event in events(rig, "incident", event="clock_backward")] == [2.5, 3.0]


def test_a_state_saved_before_clarification_i_restores_with_the_clock_backward_count_at_0(tmp_path):
    """The running bots' states, saved before clarification i, have no clock_backward counter (state version 1):
    the restart that brings in clarification i restores them with that count at 0, and the guard works after it."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    nudge(rig, dt=1.1)
    rig.core.stop()
    state = state_of(rig)
    del state["counters"]["clock_backward"], state["status"]["clock_backward"]
    state_path(rig).write_text(json.dumps(state), encoding="utf-8")
    new = restart(rig, gap_s=10)
    new.tick(make_book(8_001), dt=0)
    assert new.core.refused is None and new.core.counters["clock_backward"] == 0
    assert events(new, "restart") and new.core.counters["steps"] == 3
    new.clock.shift_wall_us(-3 * US)
    nudge(new)
    assert new.core.counters["clock_backward"] == 1 and new.core.counters["clock_jumps"] == 0


# --- the order caps across midnight ---------------------------------------------------------------------------


def test_the_carry_close_after_midnight_counts_against_the_new_utc_days_order_cap(tmp_path):
    """PROTOCOL §5.3: new orders count against 200,000 per UTC day, the daily close included. A position carried
    past midnight is closed at the first step of the new UTC day (§5.4); that order is the new day's first, so the
    previous day's count must not skip it."""

    rig = make_rig(tmp_path, start_us=at(23, 59))
    rig.core.rate = RateLimiter(max_per_window=100, window_us=10 * US, max_per_day=3)
    rig.tick(make_book(8_000), dt=0)  # orders 1 and 2
    cross(rig, "buy")  # q = 1; the replacement bid is order 3, the day's last
    assert rig.core.rate.day_orders == 3 and rig.core.ledger.q == 1
    bid_ticks = int(Decimal(repr(rig.book[0][0][0])) / TICK_D)
    rig.tick_at(at(23, 59, 50), make_book(bid_ticks, salt=1))  # the close would be the day's 4th order: skipped
    assert rig.exchange.queued == [] and rig.core.ledger.q == 1
    assert [row["reason"] for row in table(rig, "orders") if row["event"] == "skip"][-1] == "rate_day:day_close"
    rig.tick_at(at(0, 0, 0.5, day=1), make_book(bid_ticks, salt=2))  # a new UTC day: the carry close is its 1st order
    assert [(side, amount) for _, side, amount, _ in rig.exchange.queued] == [("sell", Decimal(100))]
    rig.tick(make_book(bid_ticks, salt=3), dt=5.0)
    assert rig.core.date == DATE1 and rig.core.ledger.position == 0 and live_sides(rig) == ["buy", "sell"]


# --- review round: refused starts, write errors, the day switch and deferred finalization ---------------------


def test_a_refused_start_leaves_the_state_file_byte_identical_and_the_bot_inert(tmp_path):
    rig = make_rig(tmp_path, mode="production")
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")
    nudge(rig, dt=1.1)  # a step and a save with q = 1 (prices unchanged: nothing else fills)
    kill(rig)
    saved = state_path(rig).read_bytes()
    other = BotSettings(model="B0", mode="production", data_root=str(tmp_path), heartbeat_interval_s=30.0)
    refused = restart(rig, gap_s=5.0, settings=other)
    with pytest.raises(ValueError, match="production settings changed"):
        refused.tick(make_book(8_002), dt=0)
    for k in range(5):  # inert: no exception, no order, nothing saved
        refused.tick(make_book(8_003 + k), dt=0.5)
    refused.core.fail("on_tick", RuntimeError("after the refusal"))
    assert state_path(rig).read_bytes() == saved
    assert refused.exchange.calls == [] and refused.core.refused is not None and not refused.core.started
    record = rig.settings.bot_dir / "start_refused.txt"
    assert "production settings changed" in record.read_text(encoding="utf-8")
    assert (rig.settings.bot_dir / "state.prev.json").read_bytes() == saved

    good = restart(refused, gap_s=1.0, settings=rig.settings)  # the settings the state was saved under
    good.tick(make_book(8_002), dt=0)
    assert good.core.started and good.core.ledger.q == 1 and not record.exists()
    [row] = events(good, "previous_start_refused")
    assert (rig.settings.bot_dir / row["record"]).exists() and "start refused" in row["first_lines"][0]


def test_a_telemetry_write_error_is_counted_and_retried_and_never_halts(tmp_path, monkeypatch):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    saved_at = state_of(rig)["saved_wall_us"]
    log, failures = rig.core.log, {"left": 3}
    real_flush = log.flush

    def flaky_flush():
        if failures["left"]:
            failures["left"] -= 1
            raise OSError(13, "Permission denied")  # e.g. a CSV opened with a share lock on the Windows host
        return real_flush()

    monkeypatch.setattr(log, "flush", flaky_flush)
    for _ in range(3):
        nudge(rig, dt=1.1)  # a step, then a due save whose flush fails
        assert state_of(rig)["saved_wall_us"] == saved_at  # the state still describes the last good flush
    assert rig.core.counters["flush_failures"] == 3
    assert not rig.core.safety.halted and rig.core.safety.quoting and len(live_sides(rig)) == 2
    nudge(rig, dt=1.1)
    assert state_of(rig)["saved_wall_us"] > saved_at
    assert len(events(rig, "flush_failed")) == 3  # the buffered rows reach the files once writing works again
    assert len(table(rig, "steps")) == rig.core.counters["steps"] == len(table(rig, "snapshots"))


def test_the_day_switches_at_the_first_tick_after_midnight_even_without_a_book_change(tmp_path):
    rig = make_rig(tmp_path, start_us=at(23, 59, 0))
    rig.tick(make_book(8_000), dt=0)
    assert len(live_sides(rig)) == 2
    rig.tick_at(at(23, 59, 50))  # the close is a time event: flat, so closed at once
    assert rig.core.closed and live_sides(rig) == []
    rig.tick_at(at(0, 0, 0.05, day=1))  # the same book: no step, but the UTC day has ended
    assert rig.core.date == DATE1 and rig.core.day_steps == 0 and events(rig, "reset", DATE1)
    rig.exchange.public_trade(0.08, 500.0, "buy")  # after 00:00: the new day's trades file
    rig.core.finalizer.wait()
    rig.tick(dt=0.1)
    assert_finalized(rig, DATE0)
    assert len(table(rig, "trades", DATE1)) == 1
    first = make_book(8_003)
    rig.tick(first, dt=0.1)  # the new day's first step: S0 is its mid
    assert rig.core.features.get_state()["initial_mid"] == mid_of(first) and len(live_sides(rig)) == 2


def test_a_close_still_in_flight_at_midnight_is_an_incident_and_the_day_switches_once_it_fills(tmp_path):
    rig = make_rig(tmp_path, start_us=at(12))
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1
    nudge(rig, dt=1.1)  # saved with q = 1
    kill(rig)
    new = restart(rig, gap_s=(at(23, 59, 57) - rig.clock.wall_us) / US)
    new.tick(make_book(8_001), dt=0)  # down across 23:59:50: the missed close goes out now
    assert events(new, "deviation", event="missed_close")
    assert [call[0] for call in new.exchange.calls] == ["market"]
    new.tick_at(at(0, 0, 0.5, day=1))  # still queued: paper executes it 5 s after submission
    assert new.core.date == DATE0 and new.core.carry
    assert events(new, "incident", event="close_in_flight_at_midnight")
    assert not events(new, "deviation", event="close_after_midnight")
    new.tick_at(at(0, 0, 2.5, day=1))  # executed in the connector's tick: flat, and the day switches
    assert new.core.date == DATE1 and new.core.ledger.position == 0 and not new.core.carry


def test_a_day_is_finalized_only_after_the_next_days_state_is_saved(tmp_path, monkeypatch):
    from hummingbot_adapter import bot_core as core_module

    rig = make_rig(tmp_path, start_us=at(23, 59, 40))
    rig.tick(make_book(8_000), dt=0)
    rig.tick_at(at(23, 59, 50))
    rig.tick_at(at(23, 59, 52))  # saved: day D, closed and flat
    real_save = core_module.save_state

    def save_except_the_new_day(path, state):
        if state["day"]["date"] == DATE1:
            raise OSError(28, "No space left on device")
        real_save(path, state)

    monkeypatch.setattr(core_module, "save_state", save_except_the_new_day)
    rig.tick_at(at(0, 0, 0.2, day=1))  # the switch: the new day's state cannot be saved
    nudge(rig, dt=1.1)  # a step of the new day; its save fails too
    rig.core.finalizer.wait()
    folder = rig.settings.bot_dir / DATE0
    assert not (folder / "SHA256SUMS").exists() and (folder / f"steps_{DATE0}.csv").exists()
    assert state_of(rig)["day"]["date"] == DATE0 and rig.core.counters["save_failures"] >= 2
    kill(rig)  # the process dies before any save of the new day
    monkeypatch.setattr(core_module, "save_state", real_save)

    new = restart(rig, gap_s=2.0)
    new.tick(make_book(8_002), dt=0)  # restores day D, switches, sets the dead process's D+1 files aside
    new.core.finalizer.wait()
    new.tick(dt=0.1)
    assert new.core.date == DATE1
    assert_finalized(new, DATE0)
    [orphans] = events(new, "deviation", DATE1, event="orphaned_files")
    assert sorted(orphans["files"]) == sorted(f"{kind}_{DATE1}.csv.orphan-1" for kind in KINDS)
    assert all((new.settings.bot_dir / DATE1 / name).exists() for name in orphans["files"])


def test_completed_rows_carry_the_quote_notional_in_the_reason_not_in_the_price_column(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    order_id, price = cross(rig, "buy")
    [row] = [row for row in table(rig, "orders") if row["event"] == "complete"]
    assert (row["order_id"], row["price"], Decimal(row["amount"])) == (order_id, "", Decimal("99.9"))
    assert Decimal(row["reason"].removeprefix("quote_amount=")) == 100 * price


# --- round 2: clarification g's incident lines, clarification h's book re-sync, the clock-rate diagnostic --------


def test_each_applied_book_snapshot_is_logged_and_counted_for_status_ps1(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    # the shell reports each change of snapshot_uid with the book's last applied diff id at that moment
    rig.core.on_book_snapshot(25_715_645_287, 25_715_652_463, 25_715_652_470)  # a newer diff already applied
    rig.core.on_book_snapshot(25_715_652_463, 25_715_660_001, 25_715_659_990)  # only older diffs replayed onto it
    rows = events(rig, "book_snapshot")
    assert [(row["previous_uid"], row["uid"], row["last_diff_uid"], row["ahead_of_diffs"]) for row in rows] == [
        (25_715_645_287, 25_715_652_463, 25_715_652_470, False), (25_715_652_463, 25_715_660_001, 25_715_659_990, True)]
    status = rig.core.state(None)["status"]
    assert rig.core.counters["book_snapshots"] == status["book_snapshots"] == 2
    assert status["snapshots_ahead_of_diffs"] == 1 and status["snapshot_errors"] == 0
    rig.core.on_feed_event("snapshot_error", "ERROR", "Unexpected error fetching order book snapshot for DOGE-USDT.")
    assert rig.core.state(None)["status"]["snapshot_errors"] == 1


def test_a_reanchor_is_a_run_log_incident_line_and_a_count_for_status_ps1(tmp_path):
    from hummingbot_adapter import run_log
    from hummingbot_adapter.telemetry import utc_iso

    rig = make_rig(tmp_path, start_us=at(10))
    rig.tick(make_book(8_000), dt=0)
    rig.clock.shift_wall_us(150 * US)  # 2.5 minutes asleep
    rig.tick(make_book(8_001), dt=0.1)
    [line] = (rig.settings.bot_dir / "run_log_incidents.md").read_text(encoding="utf-8").splitlines()
    assert line == (f"{utc_iso(rig.clock.wall_us)[:19]}Z: hb_b0 (smoke): incident: clock re-anchor (clarification g): "
                    "the step clock and the wall clock differed by +150.000 s; re-anchored on the wall clock; "
                    "minutes affected 2.50.")
    assert rig.core.state(None)["status"]["clock_jumps"] == 1
    log = tmp_path / "RUN_LOG.md"
    log.write_text("# Run log (UTC)\n", encoding="utf-8", newline="\n")
    for _ in range(2):  # appended once, as its own entry; a second run finds nothing new
        assert run_log.main(["--data-root", str(tmp_path), "--smoke", "--run-log", str(log)]) == 0
        assert log.read_text(encoding="utf-8") == f"# Run log (UTC)\n\n{line}\n"


def test_the_latest_clock_check_and_its_rates_are_in_the_status_block(tmp_path):
    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    result = {"error": None, "server_ms": 1, "offset_s": -0.25, "wall_offset_s": -0.2, "raw_offset_s": -0.24,
              "rtt_s": 0.3, "wall_us": 1, "rate_step": -1e-05, "rate_monotonic": -0.02, "rate_over_s": 3_600.0}
    rig.core.on_clock_check({**result, "rate_step": 1.0, "rate_monotonic": -0.5})  # absurd rates: no trading rule
    assert not events(rig, "warn") and not events(rig, "pause") and rig.core.safety.quoting
    rig.core.on_clock_check(result)
    clock = rig.core.state(None)["status"]["clock"]
    assert (clock["offset_s"], clock["rate_step"], clock["rate_monotonic"], clock["rate_over_s"]) == (
        -0.25, -1e-05, -0.02, 3_600.0)
    assert (clock["raw_offset_s"], clock["wall_offset_s"]) == (-0.24, -0.2)  # clarification j, printed by status.ps1
    status_ps1 = (Path(__file__).resolve().parents[1] / "deploy" / "status.ps1").read_text(encoding="utf-8")
    assert "$t.clock.raw_offset_s" in status_ps1 and "$t.clock.wall_offset_s" in status_ps1
    row = events(rig, "clock_check")[-1]
    assert (row["rate_step"], row["rate_monotonic"], row["rate_over_s"]) == (-1e-05, -0.02, 3_600.0)
    assert row["raw_offset_s"] == -0.24
    assert clock["checked_utc"].startswith("2026-10-01T10:00:00")
    assert not rig.core.safety.clock_paused  # a diagnostic: the rates change no trading rule


def test_a_failed_incident_write_never_halts_and_the_line_is_written_at_a_later_save(tmp_path):
    rig = make_rig(tmp_path, start_us=at(10))
    rig.tick(make_book(8_000), dt=0)
    blocker = rig.settings.bot_dir / "run_log_incidents.md"
    blocker.mkdir()  # opening it to append fails with an OSError, as a share lock on the Windows host would
    rig.clock.shift_wall_us(150 * US)
    rig.tick(make_book(8_001), dt=0.1)  # the re-anchor: its line cannot be written
    assert rig.core.counters["clock_jumps"] == 1 and rig.core.counters["incident_write_failures"] >= 1
    assert not rig.core.safety.halted and rig.core.safety.quoting
    [failed] = events(rig, "incident_write_failed")
    assert len(failed["lines"]) == 1 and "clock re-anchor (clarification g)" in failed["lines"][0]
    blocker.rmdir()
    rig.tick(make_book(8_002), dt=1.1)  # the next save writes the kept line
    assert blocker.read_text(encoding="utf-8") == failed["lines"][0]


# --- clarification j: the rate correction in the bot ------------------------------------------------------------------


class FastRawClock(FakeClock):
    """A monotonic clock that runs ppm faster than the wall clock (as CLOCK_MONOTONIC_RAW runs ahead of Binance)."""

    def __init__(self, wall_us: int, ppm: int):
        super().__init__(wall_us)
        self.ppm = ppm

    def advance_us(self, us: int) -> None:
        if us < 0:
            raise ValueError(f"time runs forward, got {us} µs")
        self._wall_ns += us * 1_000
        self._monotonic_ns += us * 1_000 * (10**6 + self.ppm) // 10**6


@pytest.mark.parametrize("rate_ppm", ["0", "50000"], ids=["uncorrected", "corrected"])
def test_the_rate_correction_applies_to_ts_the_guard_and_the_clock_check_and_the_raw_offset_is_logged(tmp_path,
                                                                                                     rate_ppm):
    """Clarification j: c corrects every timestamp, the step-vs-wall guard and the §7 clock check, and the check also
    logs the uncorrected offset. A monotonic clock 5 % fast (c = 50,000 ppm, so that 60 s of ticks show it): with
    c the step clock stays on the wall clock (here the server's clock); without it, it leads by 3 s, a clock_backward
    incident and a §7 clock pause."""

    rig = Rig(BotSettings(model="B0", mode="smoke", data_root=str(tmp_path)), FastRawClock(at(10), ppm=50_000),
              rate_ppm=rate_ppm)
    checker = ClockChecker(rig.core.clock.unheld_us, rig.core.clock.wall_us, raw_us=rig.core.clock.uncorrected_us,
                           fetch_ms=lambda: rig.clock.wall_us // 1000, monotonic_ns=rig.clock.monotonic_ns)
    rig.tick(make_book(8_000), dt=0)
    for k in range(1, 61):
        rig.tick(make_book(8_000 + k % 2, salt=k), dt=1.0)
    result = checker.check_once()
    rig.core.on_clock_check(result)
    steps = table(rig, "steps")
    assert result["raw_offset_s"] == pytest.approx(3.0, abs=1e-5)  # the RAW clock's own lead, logged either way
    assert events(rig, "clock_check")[-1]["raw_offset_s"] == result["raw_offset_s"]
    if rate_ppm == "0":
        assert int(steps[-1]["ts_us"]) == at(10, 1, 3)
        assert result["offset_s"] == pytest.approx(3.0, abs=1e-5) and rig.core.counters["clock_backward"] == 1
        assert rig.core.safety.clock_paused
    else:
        assert abs(int(steps[-1]["ts_us"]) - at(10, 1)) <= 1  # the corrected clock: the wall clock's 10:01:00
        assert {round(float(row["dt"]), 5) for row in steps[1:]} == {1.0}
        assert abs(result["offset_s"]) <= 2e-6 and rig.core.counters["clock_backward"] == 0
        assert not rig.core.safety.clock_paused and rig.core.counters["clock_jumps"] == 0


# --- clarification m: no quote once halted or paused, also inside a re-quote's cancel ---------------------------------


def raise_inside_the_cancel(core: BotCore, trigger: str) -> None:
    """What the paper connector's synchronous cancel can bring about inside the bot's own cancel() call: a halt
    (the shell's did_cancel_order handler raises, and hb_forward_bot._fail halts through BotCore.fail), or a §7
    pause applied as the bot applies it."""

    if trigger == "halt":
        core.fail("did_cancel_order", RuntimeError("handler raised"))
        return
    safety = core.safety
    events = {"clock": lambda: safety.clock_offset(2.5),
              "stale": lambda: safety.book(safety.last_book_change_us + safety.stale_pause_us, False),
              "drawdown": lambda: safety.check_drawdown(safety.drawdown_max + 1.0)}[trigger]()
    assert any(event.action in ("pause", "close_and_pause") for event in events)
    for event in events:
        core._apply(event, core._event_ts())


@pytest.mark.parametrize("trigger", ["halt", "clock", "stale", "drawdown"])
def test_a_halt_or_pause_raised_inside_a_requotes_cancel_posts_no_quote_and_its_cancel_and_close_still_run(
        tmp_path, trigger):
    """Clarification m: a halt (or a §7 pause) raised inside the re-quote's cancel of the bid: neither the new bid nor
    the ask (posted next in the same loop) goes out, each logged as a skip (not_quoting); the halt's (pause's)
    cancel of the resting ask and, for the halt and the drawdown pause, the market close of q = 1 still run."""

    rig = make_rig(tmp_path)
    rig.tick(make_book(8_000), dt=0)
    cross(rig, "buy")  # q = 1: a halt or a drawdown pause has a position to close
    nudge(rig, dt=1.1)  # both quotes over a second old, neither filled
    assert live_sides(rig) == ["buy", "sell"] and rig.core.ledger.q == 1
    old = {side: resting(rig, side)[0] for side in SIDES}
    exchange, cancel = rig.exchange, rig.exchange.cancel
    trap = {"armed": True, "mark": None}

    def cancel_then_raise(order_id: str) -> None:
        cancel(order_id)
        if trap["armed"]:
            trap["armed"], trap["mark"] = False, len(exchange.calls)
            raise_inside_the_cancel(rig.core, trigger)

    exchange.cancel = cancel_then_raise
    rig.tick(make_book(7_993, salt=2), dt=0.1)  # both targets 3 ticks lower, neither quote crossed: re-quote both
    assert trap["mark"] is not None and not rig.core.safety.quoting
    assert limit_calls(rig, trap["mark"]) == []  # no quote after the halt or pause, on either side
    assert rig.exchange.limits == {}  # the bid by its re-quote's cancel, the ask by the halt's (pause's) cancel
    orders = table(rig, "orders")
    cancels = {row["order_id"]: row["reason"] for row in orders if row["event"] == "cancel"}
    reason = {"halt": "exception", "clock": "clock_offset", "stale": "stale_book", "drawdown": "drawdown"}[trigger]
    assert (cancels[old["buy"]], cancels[old["sell"]]) == ("requote", reason)
    skips = [(row["side"], row["reason"]) for row in orders if row["event"] == "skip"]
    assert skips == [("buy", "not_quoting"), ("sell", "not_quoting")]
    closing = trigger in ("halt", "drawdown")
    assert [(side, amount) for _, side, amount, _ in rig.exchange.queued] == ([("sell", Decimal(100))] if closing else [])
    calls = len(exchange.calls)
    nudge(rig, dt=6.0)  # the market close fills (paper: 5 s); a halt or the drawdown pause still posts nothing
    if closing:
        assert rig.core.ledger.q == 0 and limit_calls(rig, calls) == []
