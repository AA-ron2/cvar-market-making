"""Order execution rules of the Hummingbot forward test (PROTOCOL §5.3–5.5 and §7; WO-HB2 deliverable 1).

Pure Python with no clock of its own: every rule takes the step's time as an argument (int µs, the ts_us of
pre-run clarification c), so the unit tests drive it with a fake clock.
- round_away_from_mid: clarification a (a bid off the tick grid is rounded down, an ask up);
- Requoter: post, re-quote (target moved >= 1 tick and order >= 1 s old) or cancel a side at the cap (§5.3);
- RateLimiter: new orders against 100 per rolling 10 s and 200,000 per UTC day; cancels counted apart (§5.3);
- Ledger: q, cash, fees by asset, W, peak, drawdown and the reconciliation residual, from fills only (§5.1, §7);
- Safety: the §7 triggers and the state they leave (halted, paused);
- DayClock: the 23:59:50 close, the wait until 00:00:00 and the reset (§5.5);
- StepClock and dt_seconds: ts_us and Δt (clarification c; CLOCK_MONOTONIC_RAW since i; the day ceiling, k; the
  rate correction, j);
- save_state / load_state: the per-bot state file, replaced atomically (§5.4).

Runtime module: standard library and hummingbot_adapter.quotes (for the one grid tolerance; it brings numpy and
cvar_mm.gym.formula_as), no Hummingbot import (it runs inside the Hummingbot container).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from hummingbot_adapter.quotes import GRID_TOLERANCE_TICKS

US_PER_SECOND = 1_000_000
DAY_US = 86_400 * US_PER_SECOND
# Clarification j: CLOCK_MONOTONIC_RAW's rate against Binance server time in ppm (+ = fast), the pooled slope of the
# bots' clock checks since 2026-10-21 07:31:18Z (RUN_LOG.md). A decimal string: StepClock builds it exactly.
CLOCK_RAW_RATE_PPM = "13.92"
SIDES = ("buy", "sell")
FEE_ASSETS = ("DOGE", "USDT")

POST, REQUOTE, CANCEL, KEEP, NONE = "post", "requote", "cancel", "keep", "none"
TRADE, CLOSE, WAIT, RESET = "trade", "close", "wait", "reset"


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return side


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


# --- clarification a: rounding away from the mid ------------------------------------------------------------


@dataclass(frozen=True)
class RoundedQuotes:
    bid: float
    ask: float
    bid_rounded: bool
    ask_rounded: bool


def round_away_from_mid(quotes: Mapping[str, Any], mid: float, tick_size: float = 1e-5) -> RoundedQuotes:
    """Clarification a: a bid off the tick grid is rounded down and an ask up, away from the mid.

    A price within GRID_TOLERANCE_TICKS of the grid is on it (float noise of k * tick) and comes back as the
    grid price with its flag False. The frozen mid clamp keeps bid <= S - tick/2 and ask >= S + tick/2, so a
    bid at or above the mid, or an ask at or below it, is refused rather than rounded.
    """

    bid, ask = _finite("bid", quotes["bid"]), _finite("ask", quotes["ask"])
    mid, tick = _finite("mid", mid), _finite("tick_size", tick_size)
    if tick <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")
    if not bid < mid < ask:
        raise ValueError(f"the quotes must straddle the mid: bid {bid!r}, mid {mid!r}, ask {ask!r}")
    grid = Decimal(repr(tick))
    bid_ticks, bid_rounded = _grid_ticks(bid / tick, math.floor)
    ask_ticks, ask_rounded = _grid_ticks(ask / tick, math.ceil)
    return RoundedQuotes(
        bid=float(Decimal(bid_ticks) * grid),
        ask=float(Decimal(ask_ticks) * grid),
        bid_rounded=bid_rounded,
        ask_rounded=ask_rounded,
    )


def _grid_ticks(ticks: float, away: Callable[[float], int]) -> tuple[int, bool]:
    nearest = round(ticks)
    if abs(ticks - nearest) <= GRID_TOLERANCE_TICKS:
        return int(nearest), False
    return int(away(ticks)), True


# --- §5.3: one bid and one ask -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RestingOrder:
    """An order the bot has submitted and not yet seen filled, cancelled or failed."""

    order_id: str
    side: str
    price: Decimal
    post_ts_us: int
    mid_at_post: float

    def __post_init__(self) -> None:
        _side(self.side)
        if not isinstance(self.price, Decimal) or not self.price > 0:
            raise ValueError(f"price must be a positive Decimal, got {self.price!r}")
        _int("post_ts_us", self.post_ts_us)
        _finite("mid_at_post", self.mid_at_post)

    def to_json(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "side": self.side,
            "price": str(self.price),
            "post_ts_us": self.post_ts_us,
            "mid_at_post": self.mid_at_post,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RestingOrder:
        return cls(
            order_id=str(data["order_id"]),
            side=str(data["side"]),
            price=Decimal(str(data["price"])),
            post_ts_us=_int("post_ts_us", data["post_ts_us"]),
            mid_at_post=_finite("mid_at_post", data["mid_at_post"]),
        )


class Requoter:
    """PROTOCOL §5.3 for one side: post if nothing rests; re-quote only if the target moved by at least
    min_move_ticks and the resting order is at least min_age_us old; cancel a side that must not be posted
    (a bid while q >= q_max, an ask while q <= -q_max, or no target because quoting is off)."""

    def __init__(self, q_max: int = 10, min_move_ticks: int = 1, min_age_us: int = US_PER_SECOND,
                 tick: Decimal = Decimal("0.00001")):
        self.q_max = _int("q_max", q_max)
        self.min_move = _int("min_move_ticks", min_move_ticks) * tick
        self.min_age_us = _int("min_age_us", min_age_us)
        if self.q_max < 1 or self.min_move <= 0 or self.min_age_us < 0:
            raise ValueError("q_max and min_move_ticks must be positive and min_age_us non-negative")

    def allowed(self, side: str, q: int) -> bool:
        return q < self.q_max if _side(side) == "buy" else q > -self.q_max

    def decide(self, side: str, target: Decimal | None, resting: RestingOrder | None, q: int, now_us: int) -> str:
        """POST, REQUOTE (cancel the resting order, post the target), CANCEL, KEEP or NONE."""

        if target is None or not self.allowed(side, _int("q", q)):
            return CANCEL if resting is not None else NONE
        if resting is None:
            return POST
        if resting.side != side:
            raise ValueError(f"a {resting.side} order rests on the {side} side")
        moved = abs(target - resting.price) >= self.min_move
        old_enough = now_us - resting.post_ts_us >= self.min_age_us
        return REQUOTE if moved and old_enough else KEEP


class RateLimiter:
    """PROTOCOL §5.3: every new order counts towards max_per_window per rolling window_us and max_per_day per
    UTC day; an order that would exceed either is skipped and counted. Cancels are counted apart. The daily
    counts belong to the UTC day of day_us when it is given (the bot gives the wall clock, clarification k: a
    day's ts stays inside that day, so a position carried past 00:00 has its ts in the old day), else of the
    order's own time: they restart at the first order or cancel of a later UTC day, whatever the bot's ledger
    day (a position carried past 00:00 is closed under the new day's cap)."""

    def __init__(self, max_per_window: int = 100, window_us: int = 10 * US_PER_SECOND, max_per_day: int = 200_000):
        self.max_per_window = _int("max_per_window", max_per_window)
        self.window_us = _int("window_us", window_us)
        self.max_per_day = _int("max_per_day", max_per_day)
        if min(self.max_per_window, self.window_us, self.max_per_day) < 1:
            raise ValueError("the rate caps and the window must be positive")
        self._window: deque[int] = deque()
        self.day: int | None = None  # the UTC day (ts // DAY_US) of the daily counts
        self.day_orders = 0
        self.max_in_window = 0
        self.skipped_window = 0
        self.skipped_day = 0
        self.cancels = 0

    def _roll(self, now_us: int) -> None:
        day = now_us // DAY_US
        if self.day is not None and day < self.day:
            raise ValueError(f"time went back to UTC day {day} from day {self.day}")
        if self.day is not None and day > self.day:
            self.reset_day()
        self.day = day

    def acquire(self, now_us: int, day_us: int | None = None) -> str | None:
        """Count a new order at now_us (in the UTC day of day_us if given) and return None, or return why it is
        skipped ("rate_day", "rate_10s")."""

        now_us = _int("now_us", now_us)
        if self._window and now_us < self._window[-1]:
            raise ValueError(f"time went backwards: {now_us} < {self._window[-1]}")
        self._roll(now_us if day_us is None else _int("day_us", day_us))
        while self._window and self._window[0] <= now_us - self.window_us:
            self._window.popleft()
        if self.day_orders + 1 > self.max_per_day:
            self.skipped_day += 1
            return "rate_day"
        if len(self._window) + 1 > self.max_per_window:
            self.skipped_window += 1
            return "rate_10s"
        self._window.append(now_us)
        self.day_orders += 1
        self.max_in_window = max(self.max_in_window, len(self._window))
        return None

    def count_cancel(self, now_us: int | None = None, day_us: int | None = None) -> None:
        if day_us is not None or now_us is not None:
            self._roll(_int("day_us", day_us) if day_us is not None else _int("now_us", now_us))
        self.cancels += 1

    def reset_day(self) -> None:
        """A new UTC day: the daily counts restart; the rolling window runs on across midnight."""

        self.day_orders = self.max_in_window = self.skipped_window = self.skipped_day = self.cancels = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "window": list(self._window),
            "day": self.day,
            "day_orders": self.day_orders,
            "max_in_window": self.max_in_window,
            "skipped_window": self.skipped_window,
            "skipped_day": self.skipped_day,
            "cancels": self.cancels,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        window = [_int("window", value) for value in state["window"]]
        if window != sorted(window) or len(window) > self.max_per_window:
            raise ValueError("the stored rate window must be sorted and within the cap")
        self._window = deque(window)
        self.day = None if state["day"] is None else _int("day", state["day"])
        for key in ("day_orders", "max_in_window", "skipped_window", "skipped_day", "cancels"):
            setattr(self, key, _int(key, state[key]))


# --- the fill ledger (§5.1, §7, §9) ---------------------------------------------------------------------------


class Ledger:
    """From fill events only: the position in DOGE and q = position / `unit` (PROTOCOL §5.1), cash in USDT
    before fees, fees by asset, the gross wealth W = cash + unit * q * S marked at the mid, its running peak
    and drawdown, and the residual DOGE balance - (day-start DOGE balance + unit * q - DOGE fees booked) (§7).

    A paper market order fills one book level at a time, so between those fills the position need not be a
    whole number of units; q refuses such a position, and every step sees whole units (the levels of one
    market order are filled within one connector tick)."""

    def __init__(self, unit: int = 100):
        self.unit = _int("unit", unit)
        if self.unit < 1:
            raise ValueError(f"unit must be positive, got {unit!r}")
        self.position = Decimal(0)
        self.cash = Decimal(0)
        self.fees = {asset: Decimal(0) for asset in FEE_ASSETS}
        self.doge_reference: Decimal | None = None
        self.wealth = 0.0
        self.peak = 0.0
        self.drawdown = 0.0
        self.fills = 0
        self.fill_notional = Decimal(0)
        self.close_notional = Decimal(0)

    @property
    def units(self) -> Decimal:
        return self.position / self.unit

    @property
    def q(self) -> int:
        units = self.units
        if units != units.to_integral_value():
            raise ValueError(f"the position {self.position} DOGE is not a whole number of {self.unit}-DOGE units")
        return int(units)

    def start_day(self, doge_balance: Decimal) -> None:
        """00:00 UTC, after the close (§5.5): a flat book, zero cash and fees, and the new DOGE reference."""

        if self.position != 0:
            raise ValueError(f"a day starts flat; the position is {self.position} DOGE")
        if not isinstance(doge_balance, Decimal) or not doge_balance.is_finite():
            raise ValueError(f"the DOGE balance must be a finite Decimal, got {doge_balance!r}")
        self.cash = Decimal(0)
        self.fees = {asset: Decimal(0) for asset in FEE_ASSETS}
        self.doge_reference = doge_balance
        self.wealth = self.peak = self.drawdown = 0.0
        self.fills = 0
        self.fill_notional = self.close_notional = Decimal(0)

    def book_fill(self, side: str, price: Decimal, amount: Decimal, fee: Decimal, fee_asset: str,
                  close: bool = False) -> Decimal:
        """Book one fill and return the signed change of the position in DOGE. A close's notional is kept
        apart from the quotes' (§9 Q-C: F and C)."""

        _side(side)
        for name, value in (("price", price), ("amount", amount), ("fee", fee)):
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal, got {value!r}")
        if price <= 0 or amount <= 0 or fee < 0:
            raise ValueError(f"invalid fill: price {price}, amount {amount}, fee {fee}")
        if fee_asset not in FEE_ASSETS:
            raise ValueError(f"fee asset {fee_asset!r} is not one of {FEE_ASSETS}")
        signed = amount if side == "buy" else -amount
        self.position += signed
        self.cash -= signed * price
        self.fees[fee_asset] += fee
        self.fills += 1
        if close:
            self.close_notional += price * amount
        else:
            self.fill_notional += price * amount
        return signed

    def mark(self, mid: float) -> float:
        """W at the mid, and the running peak and drawdown of the day."""

        mid = _finite("mid", mid)
        self.wealth = float(self.cash) + float(self.position) * mid
        self.peak = max(self.peak, self.wealth)
        self.drawdown = self.peak - self.wealth
        return self.wealth

    def residual(self, doge_balance: Decimal) -> Decimal:
        if self.doge_reference is None:
            raise RuntimeError("Ledger.residual() before start_day()")
        return doge_balance - (self.doge_reference + self.position - self.fees["DOGE"])

    def get_state(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "position": str(self.position),
            "cash": str(self.cash),
            "fees": {asset: str(value) for asset, value in self.fees.items()},
            "doge_reference": None if self.doge_reference is None else str(self.doge_reference),
            "wealth": self.wealth,
            "peak": self.peak,
            "drawdown": self.drawdown,
            "fills": self.fills,
            "fill_notional": str(self.fill_notional),
            "close_notional": str(self.close_notional),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        if _int("unit", state["unit"]) != self.unit:
            raise ValueError(f"state unit {state['unit']!r} differs from {self.unit}")
        if set(state["fees"]) != set(FEE_ASSETS):
            raise ValueError(f"state fees {sorted(state['fees'])}, expected {list(FEE_ASSETS)}")
        self.position = Decimal(state["position"])
        self.cash = Decimal(state["cash"])
        self.fees = {asset: Decimal(state["fees"][asset]) for asset in FEE_ASSETS}
        reference = state["doge_reference"]
        self.doge_reference = None if reference is None else Decimal(reference)
        self.wealth = _finite("wealth", state["wealth"])
        self.peak = _finite("peak", state["peak"])
        self.drawdown = _finite("drawdown", state["drawdown"])
        self.fills = _int("fills", state["fills"])
        self.fill_notional = Decimal(state["fill_notional"])
        self.close_notional = Decimal(state["close_notional"])


# --- §7: safety limits -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyEvent:
    """What a trigger did: action is "halt", "close_and_pause", "pause", "resume" or "warn"."""

    action: str
    trigger: str
    detail: str


class Safety:
    """The §7 triggers. halted holds until the author clears it in the state file; the drawdown pause holds
    until 00:00 UTC (new_day); the stale-book and clock pauses lift when their condition clears."""

    def __init__(
        self,
        q_halt: int = 10,
        residual_max_doge: Decimal = Decimal(1),
        stale_pause_us: int = 30 * US_PER_SECOND,
        stale_halt_us: int = 300 * US_PER_SECOND,
        reject_max: int = 10,
        reject_window_us: int = 60 * US_PER_SECOND,
        drawdown_max_usdt: float = 40.0,
        clock_warn_s: float = 0.5,
        clock_pause_s: float = 2.0,
    ):
        self.q_halt = _int("q_halt", q_halt)
        self.residual_max = Decimal(residual_max_doge)
        self.stale_pause_us = _int("stale_pause_us", stale_pause_us)
        self.stale_halt_us = _int("stale_halt_us", stale_halt_us)
        self.reject_max = _int("reject_max", reject_max)
        self.reject_window_us = _int("reject_window_us", reject_window_us)
        self.drawdown_max = _finite("drawdown_max_usdt", drawdown_max_usdt)
        self.clock_warn_s = _finite("clock_warn_s", clock_warn_s)
        self.clock_pause_s = _finite("clock_pause_s", clock_pause_s)
        if not 0 < self.stale_pause_us < self.stale_halt_us:
            raise ValueError("the stale-book pause must come before the stale-book halt")
        self.halted = False
        self.halt_reason: str | None = None
        self.halt_ts_us: int | None = None
        self.drawdown_paused = False
        self.stale_paused = False
        self.clock_paused = False
        self.last_book_change_us: int | None = None
        self._rejects: deque[int] = deque()

    @property
    def quoting(self) -> bool:
        return not (self.halted or self.drawdown_paused or self.stale_paused or self.clock_paused)

    def halt(self, trigger: str, detail: str, now_us: int) -> list[SafetyEvent]:
        if self.halted:
            return []
        self.halted, self.halt_reason, self.halt_ts_us = True, f"{trigger}: {detail}", _int("now_us", now_us)
        return [SafetyEvent("halt", trigger, detail)]

    def check_inventory(self, q: Decimal | int, now_us: int) -> list[SafetyEvent]:
        """|q_ledger| > q_halt halts (q in units; it may be fractional between a market order's level fills)."""

        if abs(Decimal(q)) > self.q_halt:
            return self.halt("q_ledger", f"|q| = {abs(Decimal(q))} > {self.q_halt}", now_us)
        return []

    def check_residual(self, residual: Decimal, now_us: int) -> list[SafetyEvent]:
        if abs(residual) > self.residual_max:
            return self.halt("residual", f"residual {residual} DOGE, limit {self.residual_max}", now_us)
        return []

    def book(self, now_us: int, changed: bool) -> list[SafetyEvent]:
        """Every sampling tick: a changed top 25 lifts a stale pause; none for stale_pause_us pauses quoting,
        and none for stale_halt_us halts."""

        now_us = _int("now_us", now_us)
        if self.last_book_change_us is None:  # the first tick after a start or restore starts the timer
            self.last_book_change_us = now_us
            if not changed:
                return []
        if changed:  # only a real change lifts a pause (also one restored from the state)
            self.last_book_change_us = now_us
            if self.stale_paused:
                self.stale_paused = False
                return [SafetyEvent("resume", "stale_book", "top-25 updates returned")]
            return []
        idle = now_us - self.last_book_change_us
        if idle >= self.stale_halt_us:
            return self.halt("stale_book", f"no top-25 change for {idle / US_PER_SECOND:.1f} s", now_us)
        if idle >= self.stale_pause_us and not self.stale_paused:
            self.stale_paused = True
            return [SafetyEvent("pause", "stale_book", f"no top-25 change for {idle / US_PER_SECOND:.1f} s")]
        return []

    def reject(self, now_us: int, detail: str) -> list[SafetyEvent]:
        """A rejected or failed order; more than reject_max within reject_window_us halts."""

        now_us = _int("now_us", now_us)
        self._rejects.append(now_us)
        while self._rejects and self._rejects[0] <= now_us - self.reject_window_us:
            self._rejects.popleft()
        if len(self._rejects) > self.reject_max:
            return self.halt("rejects", f"{len(self._rejects)} rejects or failures within the window; last: {detail}",
                             now_us)
        return []

    def check_drawdown(self, drawdown: float) -> list[SafetyEvent]:
        if _finite("drawdown", drawdown) > self.drawdown_max and not self.drawdown_paused:
            self.drawdown_paused = True
            return [SafetyEvent("close_and_pause", "drawdown",
                                f"gross drawdown {drawdown!r} USDT > {self.drawdown_max!r}")]
        return []

    def clock_offset(self, offset_s: float) -> list[SafetyEvent]:
        size = abs(_finite("offset_s", offset_s))
        events = []
        if size > self.clock_warn_s:
            events.append(SafetyEvent("warn", "clock_offset", f"offset {offset_s!r} s"))
        if size > self.clock_pause_s and not self.clock_paused:
            self.clock_paused = True
            events.append(SafetyEvent("pause", "clock_offset", f"offset {offset_s!r} s > {self.clock_pause_s!r}"))
        elif size < self.clock_pause_s and self.clock_paused:
            self.clock_paused = False
            events.append(SafetyEvent("resume", "clock_offset", f"offset {offset_s!r} s"))
        return events

    def new_day(self) -> None:
        """00:00 UTC: the drawdown pause ends (§7: pause until 00:00 UTC)."""

        self.drawdown_paused = False

    def get_state(self) -> dict[str, Any]:
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "halt_ts_us": self.halt_ts_us,
            "drawdown_paused": self.drawdown_paused,
            "stale_paused": self.stale_paused,
            "clock_paused": self.clock_paused,
            "rejects": list(self._rejects),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore after a restart. The stale-book clock restarts from the first tick (book() sets it)."""

        self.halted = bool(state["halted"])
        self.halt_reason = state["halt_reason"]
        self.halt_ts_us = None if state["halt_ts_us"] is None else _int("halt_ts_us", state["halt_ts_us"])
        self.drawdown_paused = bool(state["drawdown_paused"])
        self.stale_paused = bool(state["stale_paused"])
        self.clock_paused = bool(state["clock_paused"])
        self._rejects = deque(_int("rejects", value) for value in state["rejects"])
        self.last_book_change_us = None


# --- §5.5: the UTC day ----------------------------------------------------------------------------------------


class DayClock:
    """From the UTC time it is given (the bot gives the wall clock, clarification k): TRADE before 23:59:50, CLOSE
    from 23:59:50 until the close is done, WAIT after it until 00:00:00, and RESET in a later UTC day."""

    def __init__(self, close_before_us: int = 10 * US_PER_SECOND):
        self.close_before_us = _int("close_before_us", close_before_us)
        if not 0 < self.close_before_us < DAY_US:
            raise ValueError(f"close_before_us must be within a day, got {close_before_us!r}")

    @staticmethod
    def day_start_us(ts_us: int) -> int:
        return _int("ts_us", ts_us) - ts_us % DAY_US

    @staticmethod
    def day_id(ts_us: int) -> str:
        return datetime.fromtimestamp(_int("ts_us", ts_us) // US_PER_SECOND, tz=timezone.utc).date().isoformat()

    def close_start_us(self, day_start_us: int) -> int:
        return day_start_us + DAY_US - self.close_before_us

    def phase(self, ts_us: int, day_start_us: int, closed: bool) -> str:
        ts_us, day_start_us = _int("ts_us", ts_us), _int("day_start_us", day_start_us)
        if day_start_us % DAY_US:
            raise ValueError(f"day_start_us {day_start_us} is not 00:00 UTC")
        if ts_us < day_start_us:
            raise ValueError(f"ts_us {ts_us} is before the day's start {day_start_us}")
        if ts_us >= day_start_us + DAY_US:
            return RESET
        if ts_us >= self.close_start_us(day_start_us):
            return WAIT if closed else CLOSE
        return TRADE

    @staticmethod
    def t_elapsed(ts_us: int, day_start_us: int) -> float:
        """Seconds since 00:00 UTC (clarification c)."""

        return (ts_us - day_start_us) / 1e6


def monotonic_raw_ns() -> int:
    """CLOCK_MONOTONIC_RAW in ns (clarification i): the hardware clock without the kernel's NTP slewing, which made
    the Docker VM's CLOCK_MONOTONIC run 0.01-0.03 % fast (RUN_LOG.md, 2026-10-21). Linux only (the container);
    the unit tests on Windows inject a fake clock instead."""

    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class StepClock:
    """Clarification c, with i and k: ts_us = the wall-clock µs at the anchor + the CLOCK_MONOTONIC_RAW µs since the
    anchor, never below the last value returned (non-decreasing across re-anchoring) and below ceiling_us, the end
    of the UTC day whose files are open (so a day's rows stay inside that day). Clarification j: the RAW time since
    the anchor is divided by 1 + c·1e-6 (c = rate_ppm, RAW's rate against Binance in ppm), exactly; the anchor
    itself is the uncorrected wall clock. rate_ppm "0" (the default) is the clock before j, bit for bit."""

    def __init__(self, wall_ns: Callable[[], int] = time.time_ns, monotonic_ns: Callable[[], int] = monotonic_raw_ns,
                 rate_ppm: str = "0"):
        if not isinstance(rate_ppm, str):  # a decimal string, never a binary float: the rate is built exactly
            raise TypeError(f"rate_ppm must be a decimal string, got {rate_ppm!r}")
        rate = 1 + Fraction(rate_ppm) / 10**6
        if rate <= 0:
            raise ValueError(f"rate_ppm must be above -1e6, got {rate_ppm!r}")
        self.rate_ppm = rate_ppm
        self._rate_num, self._rate_den = rate.numerator, rate.denominator
        self._wall_ns = wall_ns
        self._monotonic_ns = monotonic_ns
        self._wall0_us: int | None = None
        self._mono0_ns: int | None = None
        self.last_us: int | None = None
        self.ceiling_us: int | None = None

    @property
    def monotonic_ns(self) -> Callable[[], int]:
        """The step clock's monotonic source; the clock check reads the same one for its rate diagnostic."""

        return self._monotonic_ns

    def anchor(self, floor_us: int | None = None) -> None:
        """Re-anchor on the wall clock (at start-up and at each 00:00 UTC reset); later values stay >= floor_us."""

        self._wall0_us = self._wall_ns() // 1000
        self._mono0_ns = self._monotonic_ns()
        if floor_us is not None:
            floor_us = _int("floor_us", floor_us)
            self.last_us = floor_us if self.last_us is None else max(self.last_us, floor_us)

    def anchor_new_day(self, day_start_us: int) -> None:
        """Clarification k, the 00:00 UTC switch: anchor on the wall clock with no floor, so no lead of the step
        clock carries into the new day (the switch comes once the wall clock is in the new day, and every ts of
        the old day stayed below its end); the anchor is kept inside the new day (a wall clock stepped back
        across 00:00 in between anchors at its start), and ts then stays below the day's end."""

        day_start_us = _int("day_start_us", day_start_us)
        if day_start_us % DAY_US:
            raise ValueError(f"day_start_us {day_start_us} is not 00:00 UTC")
        ceiling_us = day_start_us + DAY_US
        self._wall0_us = min(max(self._wall_ns() // 1000, day_start_us), ceiling_us - 1)
        self._mono0_ns = self._monotonic_ns()
        self.last_us = None
        self.ceiling_us = ceiling_us

    def anchored_us(self) -> int:
        """The wall clock at the anchor + the monotonic time since (corrected, j), without the floor: what to compare
        with the wall clock to detect a suspension or a clock step (right after anchor() the two agree)."""

        if self._wall0_us is None:
            raise RuntimeError("StepClock used before anchor()")
        # j: the RAW ns since the anchor / (1 + c·1e-6), floored to whole µs in integer arithmetic (c = 0: ns // 1000)
        return self._wall0_us + (self._monotonic_ns() - self._mono0_ns) * self._rate_den // (1000 * self._rate_num)

    def uncorrected_us(self) -> int:
        """The wall clock at the anchor + the RAW µs since, without j's correction, the floor or the ceiling: the
        clock check logs its offset (raw_offset_s), RAW's own error against Binance, from which c is estimated."""

        if self._wall0_us is None:
            raise RuntimeError("StepClock used before anchor()")
        return self._wall0_us + (self._monotonic_ns() - self._mono0_ns) // 1000

    def unheld_us(self) -> int:
        """The step clock without the day's ceiling (clarification k): what the §7 check compares with Binance and
        what times the saves, while ts is held at a day's end."""

        ts = self.anchored_us()
        return ts if self.last_us is None else max(ts, self.last_us)

    def peek_us(self) -> int:
        ts = self.unheld_us()
        return ts if self.ceiling_us is None else min(ts, self.ceiling_us - 1)

    def now_us(self) -> int:
        self.last_us = self.peek_us()
        return self.last_us

    def wall_us(self) -> int:
        return self._wall_ns() // 1000


def dt_seconds(previous_us: int, ts_us: int) -> float:
    """Clarification c: Δt = (ts_us - previous ts_us) / 1e6, as the frozen loader derives Δt from a snapshot
    file's timestamp column (the loader's dt is int64 µs differences divided by 1e6)."""

    previous_us, ts_us = _int("previous_us", previous_us), _int("ts_us", ts_us)
    if ts_us < previous_us:
        raise ValueError(f"ts_us went backwards: {ts_us} < {previous_us}")
    return (ts_us - previous_us) / 1e6


# --- §5.4: the state file ---------------------------------------------------------------------------------------


def save_state(path: str | Path, state: Mapping[str, Any]) -> None:
    """Write the state as JSON to a temporary file beside it, then os.replace it (atomic on one file system)."""

    path = Path(path)
    text = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_state(path: str | Path) -> dict[str, Any] | None:
    """The saved state, or None if the bot has never saved one."""

    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
