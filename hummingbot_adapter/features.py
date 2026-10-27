"""B3's seven observation features from live order-book rows (PROTOCOL §5.1–5.2).

FormulaASActionWrapper keeps its rolling window inside a gymnasium wrapper, so `_update_rolling_features`
and `_obs` cannot be imported on their own. They are re-implemented here expression for expression, and the
Jan 7 parity test (hummingbot_adapter/tests/test_hb_feature_parity.py, T2) is the arbiter. spread_ticks and
lob_imbalance come from the imported extract_market_feature_arrays, on a one-row frame.

A row is a mapping with the Tardis column names: the best ask and bid prices and the top `imbalance_depth`
amounts per side. t_elapsed and T are inputs: in the replay, the day's cumulative and total Δt; live, the
seconds since 00:00 UTC and 86,400 (PROTOCOL §5.2). dt is the time since the previous step; the replay's is
never negative (its timestamps are sorted) and update() refuses a negative one, so a live caller takes dt
from a monotonic clock or from the steps' own timestamps, not from differences of wall-clock time.

Runtime module: standard library, numpy, pandas and cvar_mm.gym.formula_as only (it runs inside the Hummingbot
container). Importing cvar_mm.gym.formula_as runs cvar_mm/gym/__init__.py, which also imports its gymnasium-only
modules (the replay environment, plotting with matplotlib and seaborn) when gymnasium is installed: the
container must not have gymnasium (test_hb_import_discipline.py imports this module without it).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES, extract_market_feature_arrays

STATE_VERSION = 1
STATE_KEYS = frozenset({
    "version", "tick_size", "q_max", "window", "imbalance_depth", "initial_mid", "last_mid", "dS", "dt",
})


def row_columns(imbalance_depth: int) -> tuple[str, ...]:
    """The columns a row must hold, in the order of the one-row frame."""

    columns = ["asks[0].price", "bids[0].price"]
    for level in range(imbalance_depth):
        columns += [f"bids[{level}].amount", f"asks[{level}].amount"]
    return tuple(columns)


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _clock_and_inventory(t_elapsed: Any, T: Any, q: Any) -> tuple[float, float, float]:
    t = _finite("t_elapsed", t_elapsed)
    T = _finite("T", T)
    if T <= 0:
        raise ValueError(f"T must be positive, got {T!r}")
    return t, T, _finite("q", q)


class FeatureEngine:
    """The formula observation of FormulaASActionWrapper, one book row at a time."""

    def __init__(
        self,
        tick_size: float = 1e-5,
        q_max: int = 10,
        window: int = 100,
        imbalance_depth: int = 5,
    ):
        self.tick_size = _finite("tick_size", tick_size)
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size!r}")
        if int(q_max) != q_max or q_max < 1:
            raise ValueError(f"q_max must be a positive integer, got {q_max!r}")
        if int(window) != window or window < 2:
            raise ValueError(f"window must be an integer of at least 2, got {window!r}")
        if int(imbalance_depth) != imbalance_depth or imbalance_depth < 1:
            raise ValueError(f"imbalance_depth must be a positive integer, got {imbalance_depth!r}")
        self.q_max = int(q_max)
        self.window = int(window)
        self.imbalance_depth = int(imbalance_depth)
        self._columns = row_columns(self.imbalance_depth)
        self._dS: deque[float] = deque(maxlen=self.window)
        self._dt: deque[float] = deque(maxlen=self.window)
        self._last_mid: float | None = None
        self._initial_mid: float | None = None

    def reset(self, row: Mapping[str, Any], t_elapsed: float, T: float, q: float) -> np.ndarray:
        """Start a day at this row: S0 is its mid, and the window is empty (FormulaASActionWrapper.reset)."""

        clock = _clock_and_inventory(t_elapsed, T, q)
        mid, spread_ticks, lob_imbalance = self._read_row(row)
        self._dS = deque(maxlen=self.window)
        self._dt = deque(maxlen=self.window)
        self._last_mid = mid
        self._initial_mid = max(abs(self._last_mid), 1e-12)
        return self._obs(mid, spread_ticks, lob_imbalance, *clock)

    def update(self, row: Mapping[str, Any], dt: float, t_elapsed: float, T: float, q: float) -> np.ndarray:
        """Append ΔS and max(Δt, 1e-10) to the window, then return the observation
        (FormulaASActionWrapper._update_rolling_features, then _obs). Invalid input changes nothing."""

        if self._initial_mid is None:
            raise RuntimeError("FeatureEngine.update() before reset() or set_state()")
        dt = _finite("dt", dt)
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")
        clock = _clock_and_inventory(t_elapsed, T, q)
        mid, spread_ticks, lob_imbalance = self._read_row(row)
        self._dS.append(mid - self._last_mid)
        self._dt.append(max(dt, 1e-10))
        self._last_mid = mid
        return self._obs(mid, spread_ticks, lob_imbalance, *clock)

    def get_state(self) -> dict[str, Any]:
        """S0, the last mid and the window, for a restart within the same UTC day (PROTOCOL §5.4)."""

        if self._initial_mid is None:
            raise RuntimeError("FeatureEngine.get_state() before reset()")
        return {
            "version": STATE_VERSION,
            "tick_size": self.tick_size,
            "q_max": self.q_max,
            "window": self.window,
            "imbalance_depth": self.imbalance_depth,
            "initial_mid": self._initial_mid,
            "last_mid": self._last_mid,
            "dS": list(self._dS),
            "dt": list(self._dt),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore what get_state() returned; refuses a state of another configuration."""

        keys = set(state)
        if keys != STATE_KEYS:
            raise ValueError(
                f"state keys differ: missing {sorted(STATE_KEYS - keys)}, unexpected {sorted(keys - STATE_KEYS)}"
            )
        if state["version"] != STATE_VERSION:
            raise ValueError(f"state version {state['version']!r}, expected {STATE_VERSION}")
        settings = {
            "tick_size": self.tick_size,
            "q_max": self.q_max,
            "window": self.window,
            "imbalance_depth": self.imbalance_depth,
        }
        for name, value in settings.items():
            if state[name] != value:
                raise ValueError(f"state {name} {state[name]!r} differs from this engine's {value!r}")
        initial_mid = _finite("initial_mid", state["initial_mid"])
        last_mid = _finite("last_mid", state["last_mid"])
        if initial_mid < 1e-12:
            raise ValueError(f"initial_mid must be at least 1e-12, got {initial_mid!r}")
        dS = [_finite("dS", value) for value in state["dS"]]
        dt = [_finite("dt", value) for value in state["dt"]]
        if len(dS) != len(dt) or len(dS) > self.window:
            raise ValueError(f"window lengths {len(dS)} and {len(dt)} (at most {self.window}, equal)")
        if any(value < 1e-10 for value in dt):
            raise ValueError("every stored dt must be at least 1e-10")
        self._initial_mid = initial_mid
        self._last_mid = last_mid
        self._dS = deque(dS, maxlen=self.window)
        self._dt = deque(dt, maxlen=self.window)

    def _read_row(self, row: Mapping[str, Any]) -> tuple[float, float, float]:
        values = []
        for column in self._columns:
            if column not in row:
                raise KeyError(f"row lacks {column!r}")
            values.append(_finite(column, row[column]))
        frame = pd.DataFrame(np.array([values], dtype=np.float64), columns=list(self._columns))
        features = extract_market_feature_arrays(
            frame, tick_size=self.tick_size, imbalance_depth=self.imbalance_depth
        )
        # data_loader.load_single_day_with_features: (asks[0].price + bids[0].price) / 2.
        mid = (values[0] + values[1]) / 2
        return mid, float(features["spread_ticks"][0]), float(features["lob_imbalance"][0])

    def _obs(
        self,
        mid: float,
        spread_ticks: float,
        lob_imbalance: float,
        t: float,
        T: float,
        q: float,
    ) -> np.ndarray:
        """FormulaASActionWrapper._obs for one trajectory, in FORMULA_FEATURE_NAMES order."""

        terminal_time = max(T, 1e-12)

        if len(self._dS) >= 2:
            dt_sum = max(float(np.sum(self._dt)), 1e-12)
            rolling_vol = np.sqrt(float(np.sum(np.square(self._dS))) / dt_sum)
            momentum = float(np.sum(self._dS)) / dt_sum
        else:
            rolling_vol = 0.0
            momentum = 0.0

        shared = np.array([
            np.clip((terminal_time - t) / terminal_time, 0.0, 1.0),
            np.log(max(mid, 1e-12) / self._initial_mid),
            rolling_vol / self.tick_size,
            momentum / self.tick_size,
            spread_ticks,
            lob_imbalance,
        ], dtype=np.float32)

        obs = np.zeros(len(FORMULA_FEATURE_NAMES), dtype=np.float32)
        obs[0] = np.clip(q / max(float(self.q_max), 1.0), -1.0, 1.0)
        obs[1:] = shared
        return obs
