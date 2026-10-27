"""Test helpers: synthetic Tardis book files and a recording agent for the frozen replay loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cvar_mm.agents.agent import Agent
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES
from cvar_mm.gym.index_names import ASSET_PRICE_INDEX, INVENTORY_INDEX, TIME_INDEX

TICK = 1e-5
REPO_ROOT = Path(__file__).resolve().parents[2]
B0_GAMMA = 0.8908584807525282  # PROTOCOL §2


def synthetic_book(
    rng: np.random.Generator,
    rows: int,
    *,
    start_us: int = 1_735_689_600_000_000,  # 2025-01-01 00:00:00 UTC
    start_bid_ticks: int = 8_000,
    mean_dt_us: int = 500_000,
    zero_dt_share: float = 0.05,
    levels: int = 5,
) -> pd.DataFrame:
    """A Tardis-like book_snapshot frame: a random walk of the best bid in ticks, spreads of 1-3 ticks,
    some repeated timestamps (zero dt), and random amounts on `levels` levels per side. The amounts have
    two decimals, so their sums are inexact in floating point (Binance DOGE amounts are whole numbers)."""

    gaps = rng.exponential(mean_dt_us, rows).astype(np.int64) + 1
    gaps[rng.random(rows) < zero_dt_share] = 0
    gaps[0] = 0
    timestamps = start_us + np.cumsum(gaps)
    bid_ticks = start_bid_ticks + np.cumsum(rng.choice([-1, 0, 1], size=rows, p=[0.3, 0.4, 0.3]))
    spread_ticks = rng.choice([1, 2, 3], size=rows, p=[0.8, 0.15, 0.05])
    columns: dict[str, object] = {
        "exchange": "binance",
        "symbol": "DOGEUSDT",
        "timestamp": timestamps,
        "local_timestamp": timestamps + 1_000,
    }
    for level in range(levels):
        columns[f"asks[{level}].price"] = np.round((bid_ticks + spread_ticks + level) * TICK, 5)
        columns[f"asks[{level}].amount"] = np.round(rng.uniform(1.0, 200_000.0, rows), 2)
        columns[f"bids[{level}].price"] = np.round((bid_ticks - level) * TICK, 5)
        columns[f"bids[{level}].amount"] = np.round(rng.uniform(1.0, 200_000.0, rows), 2)
    return pd.DataFrame(columns)


def write_book_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


class RecordingAgent(Agent):
    """Wraps an agent's get_action. generate_trajectory_stats calls it once per step with the wrapper's raw
    observation; it records that observation, the wrapper's state (q, t, mid), the action, optionally the
    normalized observation, and the bid and ask the wrapper then computed from that action."""

    def __init__(
        self,
        env,
        rows: int,
        act: Callable[[np.ndarray], np.ndarray],
        normalize: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.env = env
        self._act = act
        self._normalize = normalize
        self.decisions = 0
        width = len(FORMULA_FEATURE_NAMES)
        self.raw_obs = np.full((rows, width), np.nan, dtype=np.float32)
        self.norm_obs = None if normalize is None else np.full((rows - 1, width), np.nan, dtype=np.float32)
        self.actions = np.full((rows - 1, 2), np.nan)
        self.q = np.full(rows, np.nan)
        self.t = np.full(rows, np.nan)
        self.mid = np.full(rows, np.nan)
        self.bid = np.full(rows - 1, np.nan)
        self.ask = np.full(rows - 1, np.nan)

    def _record_state(self, k: int) -> None:
        state = self.env.metric_state
        self.q[k] = state[0, INVENTORY_INDEX]
        self.t[k] = state[0, TIME_INDEX]
        self.mid[k] = state[0, ASSET_PRICE_INDEX]

    def _record_quotes(self, k: int) -> None:
        diagnostics = self.env.last_formula_diagnostics
        self.bid[k] = diagnostics["bid"][0]
        self.ask[k] = diagnostics["ask"][0]

    def get_action(self, state: np.ndarray) -> np.ndarray:
        k = self.decisions
        if k:
            self._record_quotes(k - 1)
        if state.shape != (1, len(FORMULA_FEATURE_NAMES)) or state.dtype != np.float32:
            raise AssertionError(f"unexpected observation {state.dtype} {state.shape}")
        self.raw_obs[k] = state[0]
        self._record_state(k)
        if self._normalize is not None:
            self.norm_obs[k] = self._normalize(state)[0]
        action = self._act(state)
        self.actions[k] = np.asarray(action, dtype=np.float64).reshape(-1)
        self.decisions += 1
        return action

    def finish(self) -> None:
        """After the loop: the last step's quotes, and the terminal observation and state."""

        k = self.decisions
        if k != len(self.q) - 1:
            raise AssertionError(f"{k} decisions for {len(self.q)} rows")
        self._record_quotes(k - 1)
        self.raw_obs[k] = self.env._obs()[0]
        self._record_state(k)
