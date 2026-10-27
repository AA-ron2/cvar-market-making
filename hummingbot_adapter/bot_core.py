"""The forward-test bot without Hummingbot (WO-HB2 deliverable 3).

hb_forward_bot.py is a thin StrategyV2Base shell: at every 0.1 s tick it passes the top-25 book to
BotCore.on_tick, forwards the connector's events, and executes orders through an ExchangePort. BotCore:
- takes one step when the top 25 changed (PROTOCOL §5.1): ModelStepper, the bot's step function
  (FeatureEngine; NumpyPolicy or gamma_B0; QuoteEngine at tau = T - t_elapsed; round_away_from_mid), then
  QuoteEngine.to_orders, the Requoter and the RateLimiter (§5.3);
- books every fill in the Ledger and runs the §7 Safety triggers;
- runs the UTC day (§5.5): the close from 23:59:50 (a time event: it does not wait for a book change), the
  wait, and the switch at the first tick of a new UTC day; S0 and the window restart at that day's first step.
  A position still open at 00:00 (the bot was down across 23:59:50, or its close is still in flight) is
  closed at market first; no step is taken until then;
- writes the telemetry (DayLog) and saves its state at least once per second (§5.4). At start-up it restores
  that state; the paper engine re-seeds its balances at every start, so the saved account balances are set
  back too, which keeps the ledger, the account and the reconciliation reference continuous. A day's files
  are finalized only after the next day's state is saved. A start that fails (an unreadable or mismatching
  state, a finalized day, missing inputs) is refused: the bot stays inert and never writes the state.
Pure Python: the unit tests run it with a fake exchange and a fake clock.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol

import numpy as np

from hummingbot_adapter.bot_config import BotSettings, VerifiedInputs, check_settings, config_hash
from hummingbot_adapter.execution import (
    CANCEL,
    CLOSE,
    DAY_US,
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
    round_away_from_mid,
    save_state,
)
from hummingbot_adapter.features import FeatureEngine
from hummingbot_adapter.monitor import memory_rss_mb
from hummingbot_adapter.policy import NumpyPolicy
from hummingbot_adapter.quotes import QuoteEngine
from hummingbot_adapter.run_log import INCIDENTS_FILE
from hummingbot_adapter.telemetry import DayLog, Finalizer, dec_text, headers, snapshot_fields, utc_iso

STATE_VERSION = 1
ASSETS = ("DOGE", "USDT")
CLOCK_JUMP_US = 2 * US_PER_SECOND  # step clock vs wall clock; the §7 clock-pause threshold
CLOCK_BACKWARD_CLEAR_US = US_PER_SECOND  # a clock_backward episode ends once the step clock's lead is back to 1 s
Level = tuple[float, float]
Book = tuple[tuple[Level, ...], tuple[Level, ...]]  # (bids, asks), best level first

COUNTERS = (
    "ticks", "steps", "skipped_steps", "fills_buy", "fills_sell", "close_fills", "submits", "cancelled",
    "exchange_cancels", "fails", "round_bid_steps", "round_ask_steps", "round_bid_orders", "round_ask_orders",
    "lost_orders", "unknown_fills", "reconnects", "book_snapshots", "save_failures", "flush_failures",
    "exceptions", "clock_checks", "clock_jumps", "clock_backward", "crossed_books", "thin_books",
    "incident_write_failures", "snapshots_ahead_of_diffs", "snapshot_errors",
)
CROSSED_ROWS_PER_DAY = 100  # health rows for crossed or locked books; beyond that they are only counted


def new_counters() -> dict[str, Any]:
    counters: dict[str, Any] = {key: 0 for key in COUNTERS}
    counters["triggers"] = {}
    counters["max_abs_residual_doge"] = "0"
    counters["max_abs_clock_offset_s"] = 0.0
    return counters


def dec(value: Decimal) -> str:
    """A Decimal as exact fixed-point text without trailing zeros (telemetry.dec_text)."""

    return dec_text(value)


def book_row(book: Book, imbalance_depth: int) -> dict[str, float]:
    """The FeatureEngine row (Tardis column names) of a book; refuses a book thinner than imbalance_depth."""

    bids, asks = book
    if len(bids) < imbalance_depth or len(asks) < imbalance_depth:
        raise ValueError(f"the book has {len(bids)} bid and {len(asks)} ask levels; {imbalance_depth} are needed")
    row = {"asks[0].price": asks[0][0], "bids[0].price": bids[0][0]}
    for level in range(imbalance_depth):
        row[f"bids[{level}].amount"] = bids[level][1]
        row[f"asks[{level}].amount"] = asks[level][1]
    return row


def book_mid(book: Book) -> float:
    bids, asks = book
    return (asks[0][0] + bids[0][0]) / 2


# --- the step function --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    mid: float
    obs: np.ndarray
    z: np.ndarray | None
    action: np.ndarray | None
    quotes: dict[str, float]
    rounded: RoundedQuotes


class ModelStepper:
    """The bot's step function: the row's observation (FeatureEngine, kept for both models: S0 and the window
    persist across restarts), then B3's NumpyPolicy action or B0's gamma with skew 0, the frozen A-S quotes
    at tau = T - t_elapsed (QuoteEngine), and the rounding away from the mid (clarification a)."""

    def __init__(self, model: str, quotes: QuoteEngine, features: FeatureEngine,
                 policy: NumpyPolicy | None = None, gamma_b0: float | None = None):
        if model not in ("B0", "B3"):
            raise ValueError(f"model must be B0 or B3, got {model!r}")
        if model == "B3" and policy is None:
            raise ValueError("B3 needs its NumpyPolicy")
        if model == "B0" and gamma_b0 is None:
            raise ValueError("B0 needs gamma_b0")
        self.model, self.quotes, self.features, self.policy, self.gamma_b0 = model, quotes, features, policy, gamma_b0

    def start(self, row: Mapping[str, float], t_elapsed: float, T: float, q: float) -> Decision:
        """The first step of a day: S0 is this row's mid and the window is empty."""

        return self._decide(self.features.reset(row, t_elapsed, T, q), t_elapsed, T, q)

    def step(self, row: Mapping[str, float], dt: float, t_elapsed: float, T: float, q: float) -> Decision:
        return self._decide(self.features.update(row, dt, t_elapsed, T, q), t_elapsed, T, q)

    def _decide(self, obs: np.ndarray, t_elapsed: float, T: float, q: float) -> Decision:
        mid = self.features.get_state()["last_mid"]
        tau = T - t_elapsed
        if self.model == "B3":
            z = self.policy.normalize(obs)
            action = self.policy.act(obs)
            quotes = self.quotes.b3(action, mid, q, tau)
        else:
            z = action = None
            quotes = self.quotes.b0(mid, q, tau, self.gamma_b0)
        rounded = round_away_from_mid(quotes, mid, self.quotes.config.tick_size)
        return Decision(mid=mid, obs=obs, z=z, action=action, quotes=quotes, rounded=rounded)


# --- orders and the exchange --------------------------------------------------------------------------------


@dataclass
class OrderInfo:
    order_id: str
    side: str
    kind: str  # "limit" (a quote) or "market" (a close)
    price: Decimal | None
    amount: Decimal
    post_ts_us: int
    mid_at_post: float
    purpose: str  # "quote", or the close's reason: day_close, halt, drawdown, carry
    rounded: bool = False
    filled: Decimal = Decimal(0)
    cancelling: bool = False
    cost_usdt: Decimal = Decimal(0)

    def as_resting(self) -> RestingOrder:
        return RestingOrder(self.order_id, self.side, self.price, self.post_ts_us, self.mid_at_post)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        for key in ("price", "amount", "filled", "cost_usdt"):
            data[key] = None if data[key] is None else str(data[key])
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> OrderInfo:
        values = dict(data)
        for key in ("amount", "filled", "cost_usdt"):
            values[key] = Decimal(values[key])
        values["price"] = None if values["price"] is None else Decimal(values["price"])
        return cls(**values)


class ExchangePort(Protocol):
    """What BotCore needs from the connector; hb_forward_bot implements it on binance_paper_trade."""

    def submit_limit(self, side: str, price: Decimal, amount: Decimal) -> str: ...

    def submit_market(self, side: str, amount: Decimal) -> str: ...

    def cancel(self, order_id: str) -> None: ...

    def balance(self, asset: str) -> Decimal: ...

    def set_balance(self, asset: str, amount: Decimal) -> None: ...

    def open_order_ids(self) -> set[str]: ...


# --- the bot ----------------------------------------------------------------------------------------------------


def _pins_ts(method: Callable[..., Any]) -> Callable[..., Any]:
    """A BotCore entry point outside the tick (a connector event, a safety action, a stop): one ts, the step clock
    now, for everything the call writes, including the events the paper connector fires inside it, e.g. a
    cancel's, so each day's files stay non-decreasing (clarification k). Inside a tick the tick's ts is kept;
    before the start nothing is read."""

    @functools.wraps(method)
    def pinned(self: BotCore, *args: Any, **kwargs: Any) -> Any:
        if self._tick_ts is not None or not self.started:
            return method(self, *args, **kwargs)
        self._tick_ts = self.clock.now_us()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._tick_ts = None

    return pinned


class BotCore:
    def __init__(
        self,
        settings: BotSettings,
        inputs: VerifiedInputs,
        exchange: ExchangePort,
        clock: StepClock,
        policy: NumpyPolicy | None = None,
        probe: Callable[[], dict[str, Any]] | None = None,
        rss_mb: Callable[[], float | None] = memory_rss_mb,
    ):
        self.changed_thresholds = check_settings(settings)
        self.settings, self.inputs, self.exchange, self.clock = settings, inputs, exchange, clock
        self.config_hash = config_hash(settings)
        s = settings
        self.tick = Decimal(repr(s.tick_size))
        self.quote_engine = QuoteEngine(inputs.sigma, inputs.kappa, s.tick_size, s.gamma_min, s.gamma_max,
                                        s.skew_ticks_max)
        self.features = FeatureEngine(s.tick_size, s.q_max, s.feature_window, s.imbalance_depth)
        self.model = ModelStepper(s.model, self.quote_engine, self.features, policy,
                                  s.gamma_b0 if s.model == "B0" else None)
        self.requoter = Requoter(s.q_max, s.requote_min_ticks, round(s.requote_min_age_s * US_PER_SECOND), self.tick)
        self.rate = RateLimiter(s.rate_max_per_window, s.rate_window_s * US_PER_SECOND, s.rate_max_per_day)
        self.ledger = Ledger(s.order_amount_doge)
        self.safety = Safety(
            q_halt=s.halt_q_max,
            residual_max_doge=Decimal(repr(s.residual_max_doge)),
            stale_pause_us=round(s.stale_pause_s * US_PER_SECOND),
            stale_halt_us=round(s.stale_halt_s * US_PER_SECOND),
            reject_max=s.reject_max,
            reject_window_us=round(s.reject_window_s * US_PER_SECOND),
            drawdown_max_usdt=s.drawdown_limit_usdt,
            clock_warn_s=s.clock_warn_s,
            clock_pause_s=s.clock_pause_s,
        )
        self.dayclock = DayClock(s.close_seconds_before_midnight * US_PER_SECOND)
        self.finalizer = Finalizer()
        self.headers = headers(s.model)
        self.bot_dir = s.bot_dir
        self.state_path = self.bot_dir / "state.json"
        self.probe = probe or (lambda: {})
        self.rss_mb = rss_mb
        self.started = False
        self.stopped = False
        self.refused: str | None = None
        self.day_start_us: int | None = None
        self.date: str | None = None
        self.log: DayLog | None = None
        self.closed = False
        self.carry = False
        self.day_steps = 0
        self.orders: dict[str, OrderInfo] = {}
        self.resting: dict[str, str | None] = {"buy": None, "sell": None}
        self.close_order: str | None = None
        self.last_step_us: int | None = None
        self.last_mid: float | None = None
        self.last_book: Book | None = None
        self.counters = new_counters()
        self.last_save_us: int | None = None
        self.last_heartbeat_us: int | None = None
        self._pending_finalize: list[tuple[Any, str]] = []
        self._logged_once: set[str] = set()
        self._failing = False
        self._last_crossed: Book | None = None
        self._pending_incidents: list[str] = []
        self._backward_lead_us = 0  # the step clock's lead at the last clock_backward incident; 0: no episode open
        self._tick_ts: int | None = None  # the ts of the tick or entry point being processed, shared by its events (k)
        self._day_wall_us: int | None = None  # the highest wall-clock reading in the open UTC day: never goes back (k)
        self.clock_status: dict[str, Any] = {}  # the latest successful clock check (status.ps1)

    # --- the tick --------------------------------------------------------------------------------------------

    def on_tick(self, book: Book | None, local_us: int) -> None:
        """Every sampling tick: start or restore on the first one; then the stale-book check, the day's end
        (the switch at the first tick of a new UTC day, or the close from 23:59:50, both on the wall clock:
        clarification k), one step if the top 25 changed, and the periodic save and heartbeat. A book thinner
        than imbalance_depth levels or crossed is no step. A refused start makes every later tick a no-op."""

        if self.stopped or self.refused is not None:
            return
        if not self.started:
            try:
                self._start()
            except Exception as error:
                self._refuse(error)
                raise
        ts = self.clock.now_us()
        wall_us = self.clock.wall_us()
        lead_us = self.clock.anchored_us() - wall_us  # > 0: the step clock is ahead of the wall clock
        if -lead_us > CLOCK_JUMP_US:  # the wall clock ahead: a sleep or a freeze (clarification g, forward only)
            ts = self._clock_jump(ts, wall_us)
        elif lead_us > self._backward_lead_us + CLOCK_JUMP_US:  # the step clock ahead: never moved back (i)
            self._clock_backward(ts, wall_us, lead_us)
        elif lead_us <= CLOCK_BACKWARD_CLEAR_US:
            self._backward_lead_us = 0
        # the day's events follow the wall clock (clarification k), through its highest reading in the day, so a
        # wall clock stepped back can neither undo the 23:59:50 close nor move the order cap's day back
        day_wall = self._day_wall_us = wall_us if self._day_wall_us is None else max(self._day_wall_us, wall_us)
        self._tick_ts = ts
        try:
            self.counters["ticks"] += 1
            valid = book is not None and self._valid(book, ts)
            changed = valid and book != self.last_book
            for event in self.safety.book(ts, changed):
                self._apply(event, ts)
            mid = book_mid(book) if valid else self.last_mid
            if day_wall >= self.day_start_us + DAY_US:
                ts = self._tick_ts = self._roll_day(ts, day_wall, mid)
            elif not self.closed and day_wall >= self.dayclock.close_start_us(self.day_start_us) and mid is not None:
                self._day_close(ts, mid)
            if changed:
                self.last_book = book
                if self.carry:
                    self.counters["skipped_steps"] += 1  # no step until the carried position is closed
                else:
                    self._step(ts, self._day_wall_us, int(local_us), book)
            self._periodic(ts)
        finally:
            self._tick_ts = None

    def _cap_day_us(self) -> int:
        """The UTC time whose day the daily order cap counts in (PROTOCOL §5.3): the wall clock (clarification k;
        while a position carried past 00:00 is closed, ts stays in the old day), never before the open day or the
        cap's own day, so a wall clock stepped back across 00:00 cannot move the cap's day back."""

        wall_us = self.clock.wall_us() if self._day_wall_us is None else max(self._day_wall_us, self.clock.wall_us())
        return max(wall_us, self.day_start_us, (self.rate.day or 0) * DAY_US)

    def _event_ts(self) -> int:
        """The ts of a connector event: inside a tick that tick's ts (the paper connector fires some events within
        the bot's own calls, e.g. a cancel), so each day's files stay non-decreasing (clarification k); otherwise
        the step clock now."""

        return self.clock.now_us() if self._tick_ts is None else self._tick_ts

    def _clock_jump(self, ts: int, wall_us: int) -> int:
        """Clarification g, forward only since clarification i: the wall clock is more than 2 s ahead of the step
        clock, so the host was suspended (a sleeping VM's monotonic clocks stop; observed in the WO-HB2 trial
        run) or the wall clock was stepped forward. Re-anchor on the wall clock, restart the stale-book timer (a
        suspension is downtime, like a restart's gap, not a stale feed), and log a deviation with the gap; the
        first step after it carries the gap as its Δt."""

        self.clock.anchor(floor_us=ts)
        self._backward_lead_us = 0  # a re-anchor ends any clock_backward episode
        now = self.clock.now_us()
        self.safety.last_book_change_us = None
        self.counters["clock_jumps"] += 1
        jump_s = (wall_us - ts) / 1e6
        self._health(now, "deviation", event="clock_jump", step_clock_us=ts, wall_us=wall_us, jump_s=jump_s,
                     detail="step clock and wall clock differed by more than 2 s (host sleep or clock step); "
                            "re-anchored on the wall clock; the gap is downtime (PROTOCOL §8)")
        self._incident(now, f"clock re-anchor (clarification g): the step clock and the wall clock differed by "
                            f"{jump_s:+.3f} s; re-anchored on the wall clock", abs(jump_s) / 60, wall_us)
        return now

    def _clock_backward(self, ts: int, wall_us: int, lead_us: int) -> None:
        """Clarification i: the step clock leads the wall clock by more than 2 s (the wall clock was stepped back,
        or the step clock's source ran fast). ts is never moved back and nothing is re-anchored: the §7 offset
        check against Binance decides (a warning above 0.5 s, a quoting pause above 2 s). One incident per
        episode, and one more each time the lead grows by a further 2 s; the episode ends once the lead is back
        to 1 s or less."""

        self._backward_lead_us = lead_us
        self.counters["clock_backward"] += 1
        lead_s = lead_us / 1e6
        self._health(ts, "incident", event="clock_backward", step_clock_us=wall_us + lead_us, wall_us=wall_us,
                     lead_s=lead_s, detail="the step clock leads the wall clock by more than 2 s; ts is not moved "
                                           "back (no re-anchor); the §7 offset check against Binance decides")
        self._incident(ts, f"clock backward (clarification i): the step clock led the wall clock by {lead_s:+.3f} s; "
                           "ts kept, never moved back; left to the §7 offset check", 0.0, wall_us)

    def _incident(self, ts: int, what: str, minutes: float, wall_us: int) -> None:
        """One RUN_LOG.md line (§8: UTC time, bot, what happened and the action, minutes affected) in the bot's
        incidents file; the repository is read-only in the container, so run_log.py appends these lines to
        RUN_LOG.md. The line is stamped with the wall clock, the real UTC time (the incidents file is no day file;
        ts may be held at a day's end, clarification k). Written at once (incidents are rare); a failed write is
        counted, kept in the health log and retried at every save, never raised into the tick."""

        stamp = utc_iso(wall_us)[:19] + "Z"
        line = f"{stamp}: {self.settings.bot} ({self.settings.mode}): incident: {what}; minutes affected {minutes:.2f}.\n"
        self._pending_incidents.append(line)
        self._write_incidents(ts)

    def _write_incidents(self, ts: int) -> None:
        if not self._pending_incidents:
            return
        try:
            with open(self.bot_dir / INCIDENTS_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write("".join(self._pending_incidents))
        except OSError as error:  # e.g. the file opened with a share lock on the Windows host: retried at the next save
            self.counters["incident_write_failures"] += 1
            if self.counters["incident_write_failures"] % 60 == 1:  # the first failure, then one row a minute
                self._health(ts, "incident_write_failed", error=repr(error), lines=list(self._pending_incidents),
                             failures=self.counters["incident_write_failures"])
            return
        self._pending_incidents.clear()

    def _valid(self, book: Book, ts: int) -> bool:
        bids, asks = book
        depth = self.settings.imbalance_depth
        if len(bids) < depth or len(asks) < depth:
            self.counters["skipped_steps"] += 1
            self.counters["thin_books"] += 1
            self._once(ts, "thin_book", "incident", event="thin_book",
                       detail=f"{len(bids)} bid and {len(asks)} ask levels; {depth} are needed")
            return False
        if bids[0][0] < asks[0][0]:
            return True
        self.counters["skipped_steps"] += 1  # a crossed or locked top of book is no step
        if book != self._last_crossed:  # each distinct crossed or locked book is counted once
            self._last_crossed = book
            self.counters["crossed_books"] += 1
            if self.counters["crossed_books"] <= CROSSED_ROWS_PER_DAY:
                self._health(ts, "crossed_book", best_bid=bids[0][0], best_ask=asks[0][0],
                             locked=bids[0][0] == asks[0][0], count=self.counters["crossed_books"])
        return False

    def _phase(self, day_us: int) -> str:
        """The day's phase on its wall clock (clarification k). A closed day only waits until 00:00 (§5.5 step 4),
        whatever the wall clock reads (a step back across 23:59:50, or after a restart)."""

        phase = self.dayclock.phase(day_us, self.day_start_us, self.closed)
        return WAIT if phase == TRADE and self.closed else phase

    def _step(self, ts: int, day_wall_us: int, local_us: int, book: Book) -> None:
        s = self.settings
        day_us = max(day_wall_us, self.day_start_us)  # the phase follows the day's wall clock (clarification k)
        phase = self._phase(day_us)
        if phase == RESET:
            raise RuntimeError("a step of a later UTC day before the day switched")
        row = book_row(book, s.imbalance_depth)
        t_elapsed = DayClock.t_elapsed(ts, self.day_start_us)
        T = float(s.day_seconds)
        q = self.ledger.q
        if self.day_steps == 0:
            dt = None
            decision = self.model.start(row, t_elapsed, T, q)
        else:
            dt = dt_seconds(self.last_step_us, ts)
            decision = self.model.step(row, dt, t_elapsed, T, q)
        self.day_steps += 1
        self.counters["steps"] += 1
        self.last_step_us, self.last_mid = ts, decision.mid
        self.counters["round_bid_steps"] += decision.rounded.bid_rounded
        self.counters["round_ask_steps"] += decision.rounded.ask_rounded
        self.ledger.mark(decision.mid)
        doge = self.exchange.balance("DOGE")
        residual = self.ledger.residual(doge)
        if abs(residual) > abs(Decimal(self.counters["max_abs_residual_doge"])):
            self.counters["max_abs_residual_doge"] = dec(abs(residual))
        events = (self.safety.check_inventory(self.ledger.units, ts) + self.safety.check_residual(residual, ts)
                  + self.safety.check_drawdown(self.ledger.drawdown))
        for event in events:
            self._apply(event, ts)
        phase = self._phase(day_us)
        if phase == CLOSE:
            self._day_close(ts, decision.mid)
        elif self.safety.halted or self.safety.drawdown_paused:
            self._flatten(ts, decision.mid, "halt" if self.safety.halted else "drawdown")  # retries a failed close
        quoting = phase == TRADE and self.safety.quoting
        self._quote(ts, decision, q, quoting)
        self._write_step(ts, local_us, book, phase, dt, t_elapsed, T, decision, doge, residual, quoting)

    # --- quoting -------------------------------------------------------------------------------------------------

    def _quote(self, ts: int, decision: Decision, q: int, quoting: bool) -> None:
        targets: dict[str, Decimal | None] = {"buy": None, "sell": None}
        if quoting:
            rounded = decision.rounded
            for order in self.quote_engine.to_orders({"bid": rounded.bid, "ask": rounded.ask}, q,
                                                     self.settings.q_max, self.settings.order_amount_doge):
                targets[order["side"]] = Decimal(order["price"])
        for side in SIDES:
            order_id = self.resting[side]
            resting = None if order_id is None else self.orders[order_id].as_resting()
            action = self.requoter.decide(side, targets[side], resting, q, ts)
            was_rounded = decision.rounded.bid_rounded if side == "buy" else decision.rounded.ask_rounded
            if action == CANCEL:
                self._cancel(order_id, ts, "not_quoting" if not quoting else "inventory_cap")
            elif action == POST:
                self._post(side, targets[side], ts, decision.mid, was_rounded)
            elif action == REQUOTE:
                why = self.rate.acquire(ts, self._cap_day_us())
                if why is not None:
                    self._log_order(ts, "skip", None, side, "limit", targets[side], self.settings.order_amount_doge,
                                    decision.mid, why + ":requote")
                    continue
                self._cancel(order_id, ts, "requote")
                self._post(side, targets[side], ts, decision.mid, was_rounded, acquired=True)
            elif action not in (KEEP, NONE):
                raise ValueError(f"unknown requote action {action!r}")

    def _post(self, side: str, price: Decimal, ts: int, mid: float, rounded: bool, acquired: bool = False) -> None:
        """A quote (the closes go out through _flatten). Clarification m: none goes out unless §7 allows quoting
        now; a halt or pause raised inside a re-quote's cancel (the paper connector runs its events inside the
        call) must not let the re-quote, or the other side's, go out after the halt's cancel-and-close."""

        amount = Decimal(self.settings.order_amount_doge)
        if not self.safety.quoting:
            self._log_order(ts, "skip", None, side, "limit", price, amount, mid, "not_quoting")
            return
        if not acquired:
            why = self.rate.acquire(ts, self._cap_day_us())
            if why is not None:
                self._log_order(ts, "skip", None, side, "limit", price, amount, mid, why)
                return
        order_id = self.exchange.submit_limit(side, price, amount)
        self.orders[order_id] = OrderInfo(order_id, side, "limit", price, amount, ts, mid, "quote", rounded)
        self.resting[side] = order_id
        self.counters["submits"] += 1
        if rounded:
            self.counters["round_bid_orders" if side == "buy" else "round_ask_orders"] += 1
        self._log_order(ts, "submit", order_id, side, "limit", price, amount, mid, "quote_rounded" if rounded else "quote")

    def _cancel(self, order_id: str | None, ts: int, reason: str) -> None:
        info = None if order_id is None else self.orders.get(order_id)
        if info is None:
            return
        if not info.cancelling:  # counted before any state changes: a failure cannot leave a live order orphaned
            self.rate.count_cancel(ts, self._cap_day_us())
        if self.resting.get(info.side) == order_id:
            self.resting[info.side] = None
        if info.cancelling:
            return
        info.cancelling = True
        self._log_order(ts, "cancel", order_id, info.side, info.kind, info.price, info.amount, None, reason)
        self.exchange.cancel(order_id)  # paper: did_cancel_order (on_cancelled) runs inside this call

    def _cancel_all(self, ts: int, reason: str) -> None:
        for side in SIDES:
            self._cancel(self.resting[side], ts, reason)

    def _flatten(self, ts: int, mid: float, reason: str) -> None:
        """A market order that closes the position, unless one is in flight (PROTOCOL §5.5 and §7)."""

        position = self.ledger.position
        if position == 0 or self.close_order is not None:
            return
        side, amount = ("sell", position) if position > 0 else ("buy", -position)
        why = self.rate.acquire(ts, self._cap_day_us())
        if why is not None:
            self._log_order(ts, "skip", None, side, "market", None, amount, mid, why + ":" + reason)
            return
        order_id = self.exchange.submit_market(side, amount)
        self.orders[order_id] = OrderInfo(order_id, side, "market", None, amount, ts, mid, reason)
        self.close_order = order_id
        self.counters["submits"] += 1
        self._log_order(ts, "submit", order_id, side, "market", None, amount, mid, reason)
        self._health(ts, "close_submitted", reason=reason, side=side, amount=dec(amount), mid=mid,
                     wealth_before=self.ledger.wealth, q=dec(self.ledger.units))

    def _day_close(self, ts: int, mid: float) -> None:
        """23:59:50 UTC (§5.5): cancel both orders and close at market; the day is closed once flat."""

        if self.resting["buy"] or self.resting["sell"]:
            self._cancel_all(ts, "day_close")
        if self.ledger.position == 0 and self.close_order is None:
            self.closed = True
            self._health(ts, "close_done", reason="day_close", flat_already=True, wealth=self.ledger.wealth)
            return
        self._flatten(ts, mid, "day_close")

    # --- the UTC day ---------------------------------------------------------------------------------------

    def _open_day(self, day_start_us: int, sizes: Mapping[str, int] | None, fresh: bool) -> None:
        self.day_start_us = day_start_us
        self.clock.ceiling_us = day_start_us + DAY_US  # the day's rows stay inside the day (clarification k)
        self.date = DayClock.day_id(day_start_us)
        self.log = DayLog(self.bot_dir / self.date, self.date, self.headers, sizes, fresh=fresh)
        if self.log.orphaned:
            self._health(self.clock.last_us, "deviation", event="orphaned_files",
                         detail="files of this day left by a process that died before saving its state were set aside",
                         files=self.log.orphaned)

    def _roll_day(self, ts: int, day_wall_us: int, mid: float | None) -> int:
        """The first tick with the wall clock in a new UTC day (clarification k): switch days if flat and return the
        new day's first ts; otherwise close the position at market first (§5.5, a deviation if the bot missed the
        23:59:50 close, an incident if that close is still in flight) and return ts, which stays inside the old day
        until the switch. Quotes still resting (the 23:59:50 close never ran: the host slept or stalled) are
        cancelled first in either case (§5.5 step 1), so no old-day order lives into the new day."""

        missed = not self.closed and not self.carry and self.close_order is None
        if self.resting["buy"] or self.resting["sell"]:
            self._cancel_all(ts, "missed_close")
        if self.ledger.position != 0 or self.close_order is not None:
            if not self.carry:
                self.carry = True
                in_flight = self.close_order is not None
                self._health(ts, "incident" if in_flight else "deviation",
                             event="close_in_flight_at_midnight" if in_flight else "close_after_midnight",
                             detail="the day's position is closed at market before the new day starts (PROTOCOL §5.4)",
                             position=dec(self.ledger.position))
            if mid is not None:
                self._flatten(ts, mid, "carry")
            return ts
        if missed:  # flat, but the 23:59:50 close never ran
            self._health(ts, "deviation", event="missed_close", position="0",
                         detail="the 23:59:50 close never ran (the host slept or stalled); flat, quotes cancelled at "
                                "the day switch (PROTOCOL §5.5)")
        if self.carry:
            self._health(ts, "carry_end", skipped_steps=self.counters["skipped_steps"])
        self._end_day(ts)
        return self._begin_day(ts, day_wall_us)

    def _end_day(self, ts: int) -> None:
        self._health(ts, "day_end", closed=self.closed, wealth=self.ledger.wealth, peak=self.ledger.peak,
                     fees={asset: dec(value) for asset, value in self.ledger.fees.items()},
                     fill_notional=dec(self.ledger.fill_notional), close_notional=dec(self.ledger.close_notional),
                     rate=self.rate.get_state() | {"window": None}, counters=self.counters)
        self.log.close()
        self._pending_finalize.append((self.bot_dir / self.date, self.date))  # submitted after the next save
        self.log = None

    def _begin_day(self, old_ts: int, day_wall_us: int) -> int:
        """00:00 UTC (§5.5): the ledger, the peak and drawdown and the DOGE reference restart now; S0 and the
        window at the day's first step (the frozen loader's row 0). The step clock is re-anchored on the wall clock
        with no floor (clarifications c and k): a lead or lag of the old day's step clock is dropped here, not
        carried into the new day, and ts starts at the wall clock inside the new day. Returns the new day's ts."""

        wall_us = self.clock.wall_us()
        lead_s = (self.clock.anchored_us() - wall_us) / 1e6  # the step clock's lead on the wall clock, dropped
        day_start_us = DayClock.day_start_us(day_wall_us)
        self.clock.anchor_new_day(day_start_us)
        ts = self._day_wall_us = self.clock.now_us()
        self._backward_lead_us = 0  # a re-anchor ends any clock_backward episode
        self._open_day(day_start_us, None, fresh=True)
        doge = self.exchange.balance("DOGE")
        self.ledger.start_day(doge)
        self.safety.new_day()
        self.counters = new_counters()
        self.closed = self.carry = False
        self.day_steps = 0
        self._health(ts, "reset", doge_reference=dec(doge), usdt=dec(self.exchange.balance("USDT")),
                     halted=self.safety.halted)
        self._health(ts, "clock_anchor", wall_us=wall_us, ts_us=ts, old_day_last_us=old_ts, lead_dropped_s=lead_s)
        self._save(ts)
        return ts

    # --- start-up and restore (§5.4) ------------------------------------------------------------------------

    def _start(self) -> None:
        self.bot_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(self.state_path)
        if state is not None:  # keep the last good state beside it before this process ever saves
            self._copy_atomic(self.state_path, self.bot_dir / "state.prev.json")
        wall_us = self.clock.wall_us()
        self.clock.anchor(floor_us=None if state is None else state["clock_last_us"])
        if state is not None:  # the restored day's rows stay inside it, also after a stop across midnight (k)
            self.clock.ceiling_us = state["day"]["start_us"] + DAY_US
        ts = self.clock.now_us()
        if state is None:
            self._fresh_start(ts)
        else:
            self._restore(state, ts)
        self._health(ts, "clock_anchor", wall_us=wall_us, floor_us=None if state is None else state["clock_last_us"],
                     ts_us=ts)
        refused = self.bot_dir / "start_refused.txt"
        if refused.exists():  # an earlier start was refused: keep its record beside the state, log it
            first = refused.read_text(encoding="utf-8").splitlines()[:2]
            kept = self.bot_dir / f"start_refused_{DayClock.day_id(ts)}_{ts}.txt"
            os.replace(refused, kept)
            self._health(ts, "previous_start_refused", record=kept.name, first_lines=first)
        self._finalize_leftovers()
        self.started = True
        self._save(ts)

    @staticmethod
    def _copy_atomic(source: Any, target: Any) -> None:
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, target)

    def _refuse(self, error: BaseException) -> None:
        """A failed start: record why, close what was opened, never touch the state; later ticks do nothing."""

        self.refused = repr(error)
        if self.log is not None:
            self.log.close()
            self.log = None
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.bot_dir.mkdir(parents=True, exist_ok=True)
        target = self.bot_dir / "start_refused.txt"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(f"{stamp} start refused; state.json was not changed\n{text}", encoding="utf-8")
        os.replace(temporary, target)

    def _start_detail(self) -> dict[str, Any]:
        s = self.settings
        return {
            "bot": s.bot, "model": s.model, "mode": s.mode, "config_hash": self.config_hash,
            "changed_thresholds": {name: getattr(s, name) for name in self.changed_thresholds},
            "calibration_sha256": self.inputs.calibration_sha256, "policy_sha256": self.inputs.policy_sha256,
            "gamma_file_sha256": self.inputs.gamma_file_sha256, "sigma": self.inputs.sigma,
            "kappa": self.inputs.kappa, "settings": s.as_dict(), "probe": self.probe(),
        }

    def _fresh_start(self, ts: int) -> None:
        s = self.settings
        balances = {asset: self.exchange.balance(asset) for asset in ASSETS}
        expected = {"DOGE": Decimal(s.paper_doge_balance), "USDT": Decimal(s.paper_usdt_balance)}
        if balances != expected:
            raise ValueError(f"a fresh start needs the PROTOCOL paper balances {expected}, the account has {balances}")
        self._open_day(DayClock.day_start_us(ts), None, fresh=True)
        self.ledger.start_day(balances["DOGE"])
        self._health(ts, "start", fresh=True, balances={a: dec(v) for a, v in balances.items()},
                     **self._start_detail())

    def _restore(self, state: Mapping[str, Any], ts: int) -> None:
        s = self.settings
        if state["version"] != STATE_VERSION:
            raise ValueError(f"state version {state['version']!r}, expected {STATE_VERSION}")
        for key in ("bot", "model", "mode"):
            if state[key] != getattr(s, key):
                raise ValueError(f"the state belongs to {key} {state[key]!r}, this bot is {getattr(s, key)!r}")
        if s.mode == "production" and state["config_hash"] != self.config_hash:
            raise ValueError("the production settings changed since the state was saved")
        day = state["day"]
        self.ledger.set_state(state["ledger"])
        self.safety.set_state(state["safety"])
        self.rate.set_state(state["rate"])
        self.counters = new_counters() | state["counters"]
        if state["features"] is not None:
            self.features.set_state(state["features"])
        self._open_day(day["start_us"], day["sizes"], fresh=False)
        seeded = {asset: self.exchange.balance(asset) for asset in ASSETS}
        saved = {asset: Decimal(state["balances"][asset]) for asset in ASSETS}
        for asset in ASSETS:
            self.exchange.set_balance(asset, saved[asset])
        open_ids = self.exchange.open_order_ids()
        lost = []
        for data in state["orders"]:
            info = OrderInfo.from_json(data)
            if info.order_id in open_ids:
                self.orders[info.order_id] = info
            else:
                lost.append(info.order_id)
        self.counters["lost_orders"] += len(lost)
        self.resting = {side: (oid if oid in self.orders else None) for side, oid in state["resting"].items()}
        self.close_order = state["close_order"] if state["close_order"] in self.orders else None
        self.closed, self.carry, self.day_steps = day["closed"], day["carry"], day["steps"]
        self.last_step_us, self.last_mid = state["last_step_us"], state["last_mid"]
        self._health(
            ts, "restart", gap_s=(self.clock.unheld_us() - state["clock_last_us"]) / 1e6, last_step_us=self.last_step_us,
            seeded_balances={a: dec(v) for a, v in seeded.items()},
            restored_balances={a: dec(v) for a, v in saved.items()}, lost_orders=lost,
            truncated=self.log.truncations, short_files=self.log.short_files,
            S0=None if state["features"] is None else state["features"]["initial_mid"],
            q=dec(self.ledger.units), halted=self.safety.halted, halt_reason=self.safety.halt_reason,
            config_changed=state["config_hash"] != self.config_hash, **self._start_detail(),
        )
        if self.log.short_files:
            self._health(ts, "deviation", event="telemetry_shorter_than_state", files=self.log.short_files)
        if not self.closed and self.clock.wall_us() >= self.dayclock.close_start_us(self.day_start_us):  # (k)
            self._health(ts, "deviation", event="missed_close",
                         detail="the bot was down across 23:59:50 UTC; it closes at its first tick (PROTOCOL §5.4)",
                         position=dec(self.ledger.position))

    def _finalize_leftovers(self) -> None:
        """Days before the state's day whose files were not compressed (an interrupted run) are finalized now.
        A later day's files can only come from a process that died before saving: they are set aside when
        that day starts (DayLog fresh=True)."""

        for folder in sorted(path for path in self.bot_dir.iterdir() if path.is_dir()):
            name = folder.name
            if (len(name) == 10 and name[4] == "-" and name < self.date
                    and any(folder.glob(f"*_{name}.csv"))):
                self.finalizer.submit(folder, name)

    # --- connector events (the shell calls these from Hummingbot's hooks) ---------------------------------------

    @_pins_ts
    def on_created(self, order_id: str) -> None:
        info = self.orders.get(order_id)
        ts = self._event_ts()
        if info is None:
            self._log_order(ts, "ack", order_id, None, None, None, None, None, "not_live")
            return
        self._log_order(ts, "ack", order_id, info.side, info.kind, info.price, info.amount, None, info.purpose)

    @_pins_ts
    def on_fill(self, order_id: str, side: str, price: Decimal, amount: Decimal, fee: Decimal, fee_asset: str,
                fee_rate: Decimal, exchange_trade_id: str, mid: float | None) -> None:
        ts = self._event_ts()
        info = self.orders.get(order_id)
        mid_now = self.last_mid if mid is None else mid
        if info is None:
            self.counters["unknown_fills"] += 1
            self._health(ts, "deviation", event="fill_of_unknown_order", order_id=order_id, side=side,
                         price=dec(price), amount=dec(amount))
        elif info.side != side:
            raise ValueError(f"fill side {side} differs from order {order_id}'s side {info.side}")
        close = info is not None and info.kind == "market"
        self.ledger.book_fill(side, price, amount, fee, fee_asset, close=close)
        self.counters["fills_buy" if side == "buy" else "fills_sell"] += 1
        if fee_rate != Decimal(repr(self.settings.fee_rate)):
            self._once(ts, "fee_rate", "deviation", event="fee_rate_differs", fee_rate=dec(fee_rate))
        sign = 1 if side == "buy" else -1
        tick = self.settings.tick_size
        delta_fill = None if mid_now is None else sign * (mid_now - float(price)) / tick
        post_ts = mid_post = delta_post = None
        if info is not None:
            post_ts, mid_post = info.post_ts_us, info.mid_at_post
            delta_post = sign * (mid_post - float(price)) / tick
        self.log.add("fills", [ts, utc_iso(ts), order_id, side, None if info is None else info.kind, price, amount,
                               mid_now, delta_fill, post_ts, mid_post, delta_post, fee, fee_asset, close,
                               exchange_trade_id])
        self._log_order(ts, "fill", order_id, side, None if info is None else info.kind, price, amount, None,
                        None if info is None else info.purpose)
        if info is not None:
            info.filled += amount
            fee_usdt = fee * price if fee_asset == "DOGE" else fee
            info.cost_usdt += Decimal(sign) * (price - Decimal(repr(info.mid_at_post))) * amount + fee_usdt
            if info.kind == "limit" or info.filled >= info.amount:
                self._order_done(ts, info)
        for event in self.safety.check_inventory(self.ledger.units, ts):
            self._apply(event, ts)

    def _order_done(self, ts: int, info: OrderInfo) -> None:
        self.orders.pop(info.order_id, None)
        if self.resting.get(info.side) == info.order_id:
            self.resting[info.side] = None
        if info.kind != "market":
            return
        self.counters["close_fills"] += 1
        if self.close_order == info.order_id:
            self.close_order = None
        wealth_after = None
        if self.last_mid is not None:
            wealth_after = float(self.ledger.cash) + float(self.ledger.position) * self.last_mid
        self._health(ts, "close_done", reason=info.purpose, side=info.side, amount=dec(info.amount),
                     mid_at_submit=info.mid_at_post, cost_usdt=dec(info.cost_usdt), wealth_after=wealth_after,
                     position=dec(self.ledger.position))
        if info.purpose == "day_close" and self.ledger.position == 0:
            self.closed = True

    @_pins_ts
    def on_cancelled(self, order_id: str) -> None:
        ts = self._event_ts()
        info = self.orders.pop(order_id, None)
        if info is None:
            self._log_order(ts, "cancelled", order_id, None, None, None, None, None, "not_live")
            return
        self.counters["cancelled"] += 1
        if not info.cancelling:
            self.counters["exchange_cancels"] += 1
        if self.resting.get(info.side) == order_id:
            self.resting[info.side] = None
        if self.close_order == order_id:
            self.close_order = None
        self._log_order(ts, "cancelled", order_id, info.side, info.kind, info.price, info.amount, None,
                        "requested" if info.cancelling else "by_exchange")

    @_pins_ts
    def on_failed(self, order_id: str, message: str | None) -> None:
        ts = self._event_ts()
        info = self.orders.pop(order_id, None)
        if info is not None:
            if self.resting.get(info.side) == order_id:
                self.resting[info.side] = None
            if self.close_order == order_id:
                self.close_order = None
        self.counters["fails"] += 1
        self._log_order(ts, "fail", order_id, None if info is None else info.side,
                        None if info is None else info.kind, None, None, None, "failed")
        self._health(ts, "order_failed", order_id=order_id, message=None if message is None else str(message)[:500])
        for event in self.safety.reject(ts, "order failed"):
            self._apply(event, ts)

    @_pins_ts
    def on_completed(self, order_id: str, base_amount: Decimal, quote_amount: Decimal) -> None:
        """Hummingbot's completed event: the base amount received or sold (net of a DOGE fee on a buy) in the
        amount column and the quote notional in the reason, never in the price column."""

        ts = self._event_ts()
        info = self.orders.get(order_id)
        self._log_order(ts, "complete", order_id, None if info is None else info.side, None if info is None
                        else info.kind, None, base_amount, None, f"quote_amount={dec(Decimal(quote_amount))}")

    @_pins_ts
    def on_public_trade(self, exchange_ts: float, trade_id: Any, price: float, amount: float, side: str) -> None:
        if self.log is None:
            return
        ts = self._event_ts()
        self.log.add("trades", [ts, utc_iso(ts), exchange_ts, trade_id, price, amount, side])

    @_pins_ts
    def on_clock_check(self, result: Mapping[str, Any]) -> None:
        ts = self._event_ts()
        self.counters["clock_checks"] += 1
        self._health(ts, "clock_check", **result)
        if result["error"] is not None:
            return
        offset = float(result["offset_s"])
        self.counters["max_abs_clock_offset_s"] = max(self.counters["max_abs_clock_offset_s"], abs(offset))
        self.clock_status = {"checked_utc": utc_iso(ts), "offset_s": offset, "rate_step": result.get("rate_step"),
                             "rate_monotonic": result.get("rate_monotonic"), "rate_over_s": result.get("rate_over_s"),
                             "raw_offset_s": result.get("raw_offset_s"), "wall_offset_s": result.get("wall_offset_s")}
        for event in self.safety.clock_offset(offset):
            self._apply(event, ts)

    @_pins_ts
    def on_feed_event(self, kind: str, level: str, message: str) -> None:
        ts = self._event_ts()
        if kind in ("ws_closed", "ws_error"):
            self.counters["reconnects"] += 1
        elif kind == "snapshot_error":  # a failed REST re-sync (clarification h): the book waits for the next one
            self.counters["snapshot_errors"] += 1
        self._health(ts, "feed", kind=kind, level=level, message=message)

    @_pins_ts
    def on_book_snapshot(self, previous_uid: int, uid: int, last_diff_uid: int | None = None) -> None:
        """An applied REST snapshot (clarification h): its update id, and the book's last applied diff id when
        the change is seen. Hummingbot 2.17's restore_from_snapshot_and_diffs replays its whole past-diff window,
        older diffs included (OrderBookMessage ordering makes its bisect return 0), so a snapshot seen with no diff
        newer than it yet (last_diff_uid < uid) is counted: then only older diffs were replayed onto it."""

        ts = self._event_ts()
        self.counters["book_snapshots"] += 1
        ahead = last_diff_uid is not None and last_diff_uid < uid
        self.counters["snapshots_ahead_of_diffs"] += ahead
        self._health(ts, "book_snapshot", previous_uid=previous_uid, uid=uid, last_diff_uid=last_diff_uid,
                     ahead_of_diffs=ahead)

    @_pins_ts
    def fail(self, where: str, error: BaseException) -> None:
        """An unexpected exception in the shell's tick or hooks: halt (cancel, close, post nothing) and log
        it as a deviation; never raises. Before a successful start it does nothing (the start was refused)."""

        if self._failing or not self.started:
            return
        self._failing = True
        try:
            ts = self._event_ts()
            self.counters["exceptions"] += 1
            text = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-4000:]
            self._health(ts, "deviation", event="exception", where=where, error=repr(error), traceback=text)
            for event in self.safety.halt("exception", f"{where}: {error!r}", ts):
                self._apply(event, ts)
        except Exception as nested:  # the halt itself failed: keep the flag, report through Hummingbot's log
            self.safety.halted = True
            self.safety.halt_reason = self.safety.halt_reason or f"exception in {where}: {error!r}; then {nested!r}"
        finally:
            self._failing = False

    @_pins_ts
    def stop(self) -> None:
        """A graceful stop: cancel the resting quotes (so nothing fills after the last save; paper cancels
        are synchronous), save, close the files (the state records their sizes), let compression finish.
        A market close still queued in the paper engine is lost with it and is redone after the restart."""

        if self.stopped or not self.started:
            self.stopped = True
            return
        ts = self._event_ts()  # pinned: the synchronous paper cancels share it (clarification k)
        self._cancel_all(ts, "stop")
        self._health(ts, "stop", q=dec(self.ledger.units), live_orders=list(self.orders))
        self._save(ts)
        if self.log is not None:
            self.log.close()
        self.finalizer.wait(timeout=30.0)
        self.stopped = True

    # --- safety actions ----------------------------------------------------------------------------------------

    def _apply(self, event: SafetyEvent, ts: int) -> None:
        triggers = self.counters["triggers"]
        key = f"{event.trigger}:{event.action}"
        triggers[key] = triggers.get(key, 0) + 1
        kind = "deviation" if event.action == "halt" and event.trigger in ("q_ledger", "residual", "rejects",
                                                                           "exception") else "incident"
        if event.trigger == "drawdown":
            kind = "finding"
        self._health(ts, event.action, trigger=event.trigger, detail=event.detail, kind=kind,
                     q=dec(self.ledger.units), wealth=self.ledger.wealth, drawdown=self.ledger.drawdown)
        if event.action in ("halt", "close_and_pause"):
            self._cancel_all(ts, event.trigger)
            if self.last_mid is not None:
                self._flatten(ts, self.last_mid, "halt" if event.action == "halt" else "drawdown")
            self._save(ts)
        elif event.action == "pause":
            self._cancel_all(ts, event.trigger)

    # --- telemetry and state -----------------------------------------------------------------------------

    def _health(self, ts: int, event: str, /, **detail: Any) -> None:
        if self.log is not None:
            self.log.add_health(ts, event, **detail)

    def _once(self, ts: int, key: str, event: str, /, **detail: Any) -> None:
        tag = f"{self.date}:{key}"
        if tag not in self._logged_once:
            self._logged_once.add(tag)
            self._health(ts, event, **detail)

    def _log_order(self, ts: int, event: str, order_id: str | None, side: str | None, kind: str | None,
                   price: Decimal | None, amount: Any, mid: float | None, reason: str | None) -> None:
        if self.log is not None:
            self.log.add("orders", [ts, utc_iso(ts), event, order_id, side, kind, price, amount, mid, reason])

    def _write_step(self, ts: int, local_us: int, book: Book, phase: str, dt: float | None, t_elapsed: float,
                    T: float, decision: Decision, doge: Decimal, residual: Decimal, quoting: bool) -> None:
        bids, asks = book
        self.log.add("snapshots", snapshot_fields(ts, local_us, asks, bids))
        quotes, rounded, ledger = decision.quotes, decision.rounded, self.ledger
        fields: list[Any] = [ts, utc_iso(ts), phase, dt, t_elapsed, T - t_elapsed, decision.mid]
        if self.settings.model == "B3":
            fields += [float(value) for value in decision.obs] + [float(value) for value in decision.z]
            fields += [float(value) for value in decision.action]
        resting = [None if self.resting[side] is None else self.orders[self.resting[side]].price for side in SIDES]
        fields += [
            quotes["gamma"], quotes["skew"], quotes["bid"], quotes["ask"], rounded.bid, rounded.ask,
            rounded.bid_rounded, rounded.ask_rounded, resting[0], resting[1], ledger.units, ledger.cash,
            ledger.wealth, ledger.peak, ledger.drawdown, ledger.fees["DOGE"], ledger.fees["USDT"], doge, residual,
            quoting, self.safety.stale_paused, self.safety.clock_paused, self.safety.drawdown_paused,
            self.safety.halted,
        ]
        self.log.add("steps", fields)

    def _periodic(self, ts: int) -> None:
        done, errors = self.finalizer.drain()
        for date, lines in done:
            self._health(ts, "day_finalized", date=date, files=lines)
        for date, error in errors:
            self._health(ts, "deviation", event="finalize_failed", date=date, error=error)
        now = self.clock.unheld_us()  # the cadence runs on while ts is held at a day's end (clarification k)
        if self.last_heartbeat_us is None or now - self.last_heartbeat_us >= self.settings.heartbeat_interval_s * US_PER_SECOND:
            self.last_heartbeat_us = now
            self._health(ts, "heartbeat", rss_mb=self.rss_mb(), probe=self.probe(), q=dec(self.ledger.units),
                         wealth=self.ledger.wealth, drawdown=self.ledger.drawdown, steps=self.counters["steps"],
                         ticks=self.counters["ticks"], new_orders=self.rate.day_orders,
                         max_new_orders_10s=self.rate.max_in_window, quoting=self.safety.quoting,
                         book_snapshots=self.counters["book_snapshots"], crossed_books=self.counters["crossed_books"],
                         clock_jumps=self.counters["clock_jumps"], clock_backward=self.counters["clock_backward"],
                         snapshots_ahead_of_diffs=self.counters["snapshots_ahead_of_diffs"],
                         snapshot_errors=self.counters["snapshot_errors"])
        if self.last_save_us is None or now - self.last_save_us >= self.settings.save_interval_s * US_PER_SECOND:
            self._save(ts)

    def _save(self, ts: int) -> None:
        """Flush the files, then save the state with their sizes; a failure of either is counted, logged and
        retried at the next second (the state then still describes the last good flush). The days ended since
        the last good save are finalized only now, when the state on disk no longer names them."""

        self.last_save_us = self.clock.unheld_us()
        self._write_incidents(ts)
        try:
            sizes = None if self.log is None else self.log.flush()
        except OSError as error:  # e.g. a CSV opened with a share lock on the Windows host: buffers are kept
            self.counters["flush_failures"] += 1
            self._health(ts, "flush_failed", error=repr(error))
            return
        try:
            save_state(self.state_path, self.state(sizes))
        except OSError as error:  # a locked or full disk
            self.counters["save_failures"] += 1
            self._health(ts, "state_save_failed", error=repr(error))
            return
        for folder, date in self._pending_finalize:
            self.finalizer.submit(folder, date)
        self._pending_finalize.clear()

    def state(self, sizes: Mapping[str, int] | None) -> dict[str, Any]:
        safety, ledger = self.safety, self.ledger
        return {
            "version": STATE_VERSION,
            "bot": self.settings.bot,
            "model": self.settings.model,
            "mode": self.settings.mode,
            "config_hash": self.config_hash,
            "saved_wall_us": self.clock.wall_us(),
            "clock_last_us": self.clock.last_us,
            "last_step_us": self.last_step_us,
            "last_mid": self.last_mid,
            "day": {"start_us": self.day_start_us, "date": self.date, "closed": self.closed, "carry": self.carry,
                    "steps": self.day_steps, "sizes": None if sizes is None else dict(sizes)},
            "features": self.features.get_state() if self.day_steps else None,
            "ledger": ledger.get_state(),
            "safety": safety.get_state(),
            "rate": self.rate.get_state(),
            "counters": self.counters,
            "balances": {asset: dec(self.exchange.balance(asset)) for asset in ASSETS},
            "orders": [info.to_json() for info in self.orders.values()],
            "resting": dict(self.resting),
            "close_order": self.close_order,
            "status": {
                "q": dec(ledger.units), "wealth": ledger.wealth, "peak": ledger.peak, "drawdown": ledger.drawdown,
                "halted": safety.halted, "halt_reason": safety.halt_reason, "stale_paused": safety.stale_paused,
                "clock_paused": safety.clock_paused, "drawdown_paused": safety.drawdown_paused,
                "new_orders": self.rate.day_orders, "max_new_orders_10s": self.rate.max_in_window,
                "skipped_rate_10s": self.rate.skipped_window, "skipped_rate_day": self.rate.skipped_day,
                "cancels": self.rate.cancels, "book_snapshots": self.counters["book_snapshots"],
                "crossed_books": self.counters["crossed_books"], "clock_jumps": self.counters["clock_jumps"],
                "clock_backward": self.counters["clock_backward"],
                "snapshots_ahead_of_diffs": self.counters["snapshots_ahead_of_diffs"],
                "snapshot_errors": self.counters["snapshot_errors"], "clock": self.clock_status,
            },
        }

    def status_text(self) -> str:
        s, ledger, safety = self.settings, self.ledger, self.safety
        if self.refused is not None:
            return f"{s.bot} {s.model} {s.mode}: start refused ({self.refused}); see start_refused.txt"
        return (f"{s.bot} {s.model} {s.mode} day {self.date} steps {self.counters['steps']} q {dec(ledger.units)} "
                f"W {ledger.wealth:.4f} DD {ledger.drawdown:.4f} quoting {safety.quoting} halted {safety.halted}")
