"""
Arrival models.

An arrival model determines whether an exogenous *market* order arrives
during a single time step.  If it does, the agent's resting limit order
may be filled (subject to the fill-probability model).

Convention (mbt-gym, Jerome et al. 2023, §3):
    arrivals[:, 0]  →  sell market order arrives (hits agent's bid / buy side)
    arrivals[:, 1]  →  buy market order arrives  (lifts agent's ask / sell side)

This module provides:
    • ``PoissonArrivalModel`` — constant-intensity Poisson (A-S model).

Two arrival-probability formulas are supported:
    • **Linear** (mbt-gym default):   P = λ · Δt
      Used for fixed small Δt in simulated environments.
    • **Exact CDF**:                  P = 1 − exp(−λ · Δt)
      Required for variable / larger Δt in market-replay environments.
"""

from __future__ import annotations

import numpy as np


# ── base class ────────────────────────────────────────────────
class ArrivalModel:
    """
    Minimal base following mbt-gym's ``ArrivalModel`` contract.
    """

    def __init__(self, num_trajectories: int = 1, seed: int | None = None):
        self.num_trajectories = num_trajectories
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass                            # stateless for Poisson

    def get_arrivals(self, dt: float) -> np.ndarray:
        """Return (num_trajectories, 2) binary array."""
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)


# ── concrete: Poisson ─────────────────────────────────────────
class PoissonArrivalModel(ArrivalModel):
    """
    Constant-intensity Poisson arrival model.

    Parameters
    ----------
    intensity : np.ndarray, shape (2,)
        Arrival rates [λ_bid, λ_ask].
        mbt-gym default = [140, 140].
    num_trajectories : int
    seed : int | None
    use_linear_approximation : bool
        If True (default), arrival probability is  ``P = λ · Δt``  —
        the linear approximation used in mbt-gym.  Valid when λ·Δt ≪ 1.
        If False, uses exact Poisson CDF  ``P = 1 − exp(−λ · Δt)``.
        Use False for market-replay environments with variable / large Δt.

    Notes
    -----
    mbt-gym's ``PoissonArrivalModel.get_arrivals()`` (arrival_models.py:54)
    uses::

        return unif < self.intensity * self.step_size      # linear

    For the standard A-S replication (λ=140, Δt=0.005) this gives
    P=0.70.  The exact CDF would give P=0.50 — a large difference.
    To replicate mbt-gym results, set ``use_linear_approximation=True``.
    """

    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        num_trajectories: int = 1,
        seed: int | None = None,
        use_linear_approximation: bool = True,
    ):
        super().__init__(num_trajectories=num_trajectories, seed=seed)
        self.intensity = np.asarray(intensity, dtype=np.float64)
        self.use_linear_approximation = use_linear_approximation

    def get_arrivals(self, dt: float) -> np.ndarray:
        """
        Sample arrivals for current Δt.

        Returns
        -------
        arrivals : np.ndarray, shape (num_trajectories, 2), dtype int
            Binary: 1 = market order arrived, 0 = no arrival.
        """
        if self.use_linear_approximation:
            probs = self.intensity * dt                      # mbt-gym formula
        else:
            probs = 1.0 - np.exp(-self.intensity * dt)       # exact Poisson CDF

        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return (unif < probs).astype(np.int32)

class HawkesArrivalModel(ArrivalModel):
    """
    Self-exciting Hawkes arrival model.

    The intensity λ_t follows:
        dλ_t = β(λ̄ − λ_t)dt + γ_jump · dN_t

    where:
        λ̄    = baseline arrival rate
        β     = mean-reversion speed
        γ_jump = jump size on arrival (self-excitation)
        N_t   = counting process (arrivals)

    Discretised (Euler):
        λ_{t+1} = λ_t + β(λ̄ − λ_t)Δt + γ_jump · arrivals_t

    The intensity is part of the state (observable by the agent).

    Reference: Jerome et al. (2023), mbt-gym ``HawkesArrivalModel``.
    See also: Hawkes (1971), Bacry et al. (2015, arXiv:1507.02822).

    Parameters
    ----------
    baseline_rate : np.ndarray, shape (2,)
        λ̄ for [bid_side, ask_side].
    jump_size : float
        γ_jump — how much intensity increases on each arrival.
    mean_reversion : float
        β — speed at which intensity reverts to baseline.
    num_trajectories : int
    seed : int | None
    """

    def __init__(
        self,
        baseline_rate: np.ndarray = np.array([10.0, 10.0]),
        jump_size: float = 40.0,
        mean_reversion: float = 60.0,
        num_trajectories: int = 1,
        seed: int | None = None,
    ):
        super().__init__(num_trajectories=num_trajectories, seed=seed)
        self.baseline_rate = np.asarray(baseline_rate, dtype=np.float64)
        self.jump_size = jump_size
        self.mean_reversion = mean_reversion

        # Current intensity (stochastic state)
        self._intensity = np.repeat(
            self.baseline_rate.reshape(1, -1), num_trajectories, axis=0,
        )  # (N, 2)

    def reset(self) -> None:
        self._intensity = np.repeat(
            self.baseline_rate.reshape(1, -1), self.num_trajectories, axis=0,
        )

    def get_arrivals(self, dt: float) -> np.ndarray:
        """Sample arrivals and update intensity (self-excitation)."""
        # Arrival probability from current intensity
        probs = self._intensity * dt
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        arrivals = (unif < probs).astype(np.int32)

        # Update intensity: mean-revert + jump on arrival
        self._intensity = (
            self._intensity
            + self.mean_reversion
            * (self.baseline_rate - self._intensity) * dt
            + self.jump_size * arrivals
        )

        return arrivals

    @property
    def intensity(self) -> np.ndarray:
        """Current intensity, shape (num_trajectories, 2)."""
        return self._intensity