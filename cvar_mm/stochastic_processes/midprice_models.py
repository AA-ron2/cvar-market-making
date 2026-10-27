"""
Midprice models.

Provides two concrete midprice models, both conforming to the same
duck-typed interface so that ``TradingEnvironment`` works unchanged:

    • ``BrownianMotionMidpriceModel`` — simulated ABM (A-S eq. 1)
    • ``MarketReplayMidpriceModel``   — replay real Tardis L2 data

Interface contract (shared by both models):
    current_state    : np.ndarray, shape (num_trajectories, 1)
    update()         : advance one step
    reset()          : return to initial state
    n_snapshots      : int   — total observation count (n_steps + 1)
    dt_array         : np.ndarray, shape (n_snapshots,)
    step_index       : int   — current position in the episode
    steps_remaining  : int   — n_snapshots − 1 − step_index
    volatility       : float — σ in price / √time
    min_value / max_value : np.ndarray, shape (1, 1)

Architecture reference
    mbt-gym  StochasticProcessModel  → MidpriceModel  → concrete class
    (Jerome et al., 2023, §3, Fig. 1)
"""

from __future__ import annotations

from math import sqrt

import numpy as np


# ══════════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════════
class MidpriceModel:
    """Minimal base following mbt-gym's StochasticProcessModel contract."""

    def __init__(
        self,
        initial_state: np.ndarray,
        min_value: np.ndarray,
        max_value: np.ndarray,
        num_trajectories: int = 1,
        seed: int | None = None,
    ):
        self.initial_state = initial_state          # (1, 1)
        self.min_value = min_value                  # (1, 1)
        self.max_value = max_value                  # (1, 1)
        self.num_trajectories = num_trajectories
        self.rng = np.random.default_rng(seed)
        self.current_state = self._broadcast(initial_state)

    def reset(self) -> None:
        self.current_state = self._broadcast(self.initial_state)

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
        state: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _broadcast(self, scalar_state: np.ndarray) -> np.ndarray:
        return np.repeat(scalar_state, self.num_trajectories, axis=0)


# ══════════════════════════════════════════════════════════════
# Brownian Motion (simulated)
# ══════════════════════════════════════════════════════════════
class BrownianMotionMidpriceModel(MidpriceModel):
    """
    Arithmetic Brownian motion midprice model (simulated).

        dS = μ dt + σ dW

    Discretised as:

        S_{t+1} = S_t  +  μ Δt  +  σ √Δt Z,    Z ~ N(0,1)

    This is the standard midprice model in Avellaneda & Stoikov (2008),
    eq. (1): dS = σ dW  (with μ = 0).

    Each of the ``num_trajectories`` paths receives an *independent*
    Brownian increment at each step, so N=1000 gives 1000 distinct
    price paths in one vectorised call.

    Parameters
    ----------
    drift : float
        Drift μ  (default 0 — as in A-S).
    volatility : float
        σ in price / √time.  mbt-gym default = 2.0.
    initial_price : float
        S₀.  mbt-gym default = 100.
    terminal_time : float
        T.  mbt-gym default = 1.0.
    n_steps : int
        Number of discrete time steps per episode.
        Step size Δt = T / n_steps.  mbt-gym default = 200.
    num_trajectories : int
        N parallel paths.
    seed : int | None

    References
    ----------
    • Avellaneda & Stoikov (2008), eq. (1).
    • Jerome et al. (2023), ``BrownianMotionMidpriceModel``.
    """

    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        n_steps: int = 200,
        num_trajectories: int = 1,
        seed: int | None = None,
    ):
        self.drift = drift
        self._volatility = volatility
        self._terminal_time = terminal_time
        self._n_steps = n_steps
        self._step_size = terminal_time / n_steps
        self._step_index = 0

        # Observation-space bounds: ±4σ√T around initial price
        # (same as mbt-gym's _get_max_value)
        span = 4.0 * volatility * np.sqrt(terminal_time)

        # Synthetic uniform dt_array for interface parity with MarketReplay
        self.dt_array = np.full(n_steps + 1, self._step_size)
        self.dt_array[0] = 0.0
        self.n_snapshots = n_steps + 1

        super().__init__(
            initial_state=np.array([[initial_price]]),
            min_value=np.array([[initial_price - span]]),
            max_value=np.array([[initial_price + span]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    # ── public interface ──────────────────────────────────────
    def reset(self) -> None:
        self._step_index = 0
        super().reset()

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
        state: np.ndarray | None = None,
    ) -> None:
        """
        Advance one ABM step:  S_{t+1} = S_t + μ Δt + σ √Δt Z.
        """
        self._step_index += 1
        dt = self._step_size
        self.current_state = (
            self.current_state
            + self.drift * dt * np.ones((self.num_trajectories, 1))
            + self._volatility
            * sqrt(dt)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

    # ── properties (interface parity) ─────────────────────────
    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def steps_remaining(self) -> int:
        return self.n_snapshots - 1 - self._step_index

    @property
    def volatility(self) -> float:
        return self._volatility

    @property
    def step_size(self) -> float:
        return self._step_size


# ══════════════════════════════════════════════════════════════
# Market replay (real data)
# ══════════════════════════════════════════════════════════════
class MarketReplayMidpriceModel(MidpriceModel):
    """
    Deterministic midprice replay from historical L2 order-book data.

    The midprice path *S* and the corresponding inter-snapshot intervals
    *dt* are injected at construction.  Each call to ``update()`` simply
    advances the internal index by one snapshot.

    Parameters
    ----------
    midprices : np.ndarray, shape (M,)
        Full session midprice series.
    dt_array : np.ndarray, shape (M,)
        Inter-snapshot time deltas in seconds.  dt[0] is typically 0.
    num_trajectories : int
    seed : int | None
    """

    def __init__(
        self,
        midprices: np.ndarray,
        dt_array: np.ndarray,
        num_trajectories: int = 1,
        seed: int | None = None,
    ):
        self.midprices = midprices.astype(np.float64)
        self.dt_array = dt_array.astype(np.float64)
        self.n_snapshots = len(midprices)
        self._step_index = 0

        initial = np.array([[midprices[0]]])
        super().__init__(
            initial_state=initial,
            min_value=np.array([[midprices.min()]]),
            max_value=np.array([[midprices.max()]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def reset(self) -> None:
        self._step_index = 0
        super().reset()

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
        state: np.ndarray | None = None,
    ) -> None:
        """Advance to the next snapshot."""
        self._step_index += 1
        price = self.midprices[self._step_index]
        self.current_state[:, 0] = price

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def steps_remaining(self) -> int:
        return self.n_snapshots - 1 - self._step_index

    @property
    def volatility(self) -> float:
        """
        Realised σ in price / √second from absolute price differences
        (arithmetic BM convention).  Quadratic-variation estimator:
            σ² = Σ(ΔS²) / Σ(Δt)
        Reference: Avellaneda & Stoikov (2008) eq. (1).
        """
        dS = np.diff(self.midprices)
        dt = self.dt_array[1:]
        valid = dt > 0
        sigma2 = np.sum(dS[valid] ** 2) / np.sum(dt[valid])
        return float(np.sqrt(sigma2))


# ══════════════════════════════════════════════════════════════
# Multi-day market replay
# ══════════════════════════════════════════════════════════════
class MultiDayReplayMidpriceModel(MidpriceModel):
    """
    Multi-day midprice replay.  On each ``reset()``, a new day's data
    is selected, exposing the RL agent to diverse market regimes.

    Parameters
    ----------
    daily_midprices : list[np.ndarray]
    daily_dt_arrays : list[np.ndarray]
    num_trajectories : int
    seed : int | None
    mode : str
        'random'     — sample a random day each reset
        'sequential' — cycle 0, 1, ..., N-1, 0, 1, ...
    """

    def __init__(
        self,
        daily_midprices: list[np.ndarray],
        daily_dt_arrays: list[np.ndarray],
        num_trajectories: int = 1,
        seed: int | None = None,
        mode: str = "random",
    ):
        assert len(daily_midprices) == len(daily_dt_arrays)
        assert mode in ("random", "sequential")
        self.days = [
            (m.astype(np.float64), d.astype(np.float64))
            for m, d in zip(daily_midprices, daily_dt_arrays)
        ]
        self.n_days = len(self.days)
        self.mode = mode
        self._day_index = 0
        self._reset_count = 0
        self._completed_counts = np.zeros(self.n_days, dtype=np.int64)
        self._started_counts = np.zeros(self.n_days, dtype=np.int64)

        self.midprices, self.dt_array = self.days[0]
        self.n_snapshots = len(self.midprices)
        self._step_index = 0

        all_prices = np.concatenate([m for m, _ in self.days])
        super().__init__(
            initial_state=np.array([[self.midprices[0]]]),
            min_value=np.array([[all_prices.min()]]),
            max_value=np.array([[all_prices.max()]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def reset(self) -> None:
        if self._reset_count > 0 and self._step_index >= self.n_snapshots - 1:
            self._completed_counts[self._day_index] += 1
        if self.mode == "random":
            self._day_index = int(self.rng.integers(self.n_days))
        else:
            self._day_index = self._reset_count % self.n_days
        self._reset_count += 1
        self._started_counts[self._day_index] += 1
        self.midprices, self.dt_array = self.days[self._day_index]
        self.n_snapshots = len(self.midprices)
        self._step_index = 0
        self.initial_state = np.array([[self.midprices[0]]])
        super().reset()

    def update(self, arrivals, fills, action, state=None) -> None:
        self._step_index += 1
        self.current_state[:, 0] = self.midprices[self._step_index]

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def steps_remaining(self) -> int:
        return self.n_snapshots - 1 - self._step_index

    @property
    def day_index(self) -> int:
        return self._day_index

    @property
    def coverage_report(self) -> dict[str, list[int]]:
        return {
            "started": self._started_counts.astype(int).tolist(),
            "completed": self._completed_counts.astype(int).tolist(),
        }

    @property
    def volatility(self) -> float:
        dS = np.diff(self.midprices)
        dt = self.dt_array[1:]
        valid = dt > 0
        return float(np.sqrt(np.sum(dS[valid] ** 2) / np.sum(dt[valid])))
