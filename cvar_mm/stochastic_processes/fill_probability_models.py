"""
Fill-probability models.

Given that a market order has *arrived* (from the arrival model),
a fill-probability model determines whether the agent's resting limit
order at depth δ from the midprice is actually executed.

Convention (mbt-gym, Jerome et al. 2023, §3):
    depths[:, 0]  →  bid depth  δ_bid = S − p_bid
    depths[:, 1]  →  ask depth  δ_ask = p_ask − S

This module provides:
    • ``ExponentialFillFunction`` — legacy conditional fill probability.
    • ``DepthDependentPoissonFillFunction`` — exact A-S execution hazard.

Reference: Avellaneda & Stoikov (2008) eq. (9)–(10):
    λ(δ) = A exp(−κ δ)
where A is factored into the arrival model and κ is here.
"""

from __future__ import annotations

import numpy as np


# ── base class ────────────────────────────────────────────────
class FillProbabilityModel:
    """Minimal base following mbt-gym's ``FillProbabilityModel``."""

    def __init__(self, num_trajectories: int = 1, seed: int | None = None):
        self.num_trajectories = num_trajectories
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass                            # stateless

    def get_fills(self, depths: np.ndarray) -> np.ndarray:
        """Return (num_trajectories, 2) binary array."""
        raise NotImplementedError

    @property
    def max_depth(self) -> float:
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)


# ── concrete: exponential ─────────────────────────────────────
class ExponentialFillFunction(FillProbabilityModel):
    """
    Exponential fill-probability model:  P(fill | δ) = exp(−κ δ).

    Parameters
    ----------
    fill_exponent : float
        κ  (kappa) in USDT⁻¹.  Calibrated value for DOGEUSDT ≈ 35 000.
    num_trajectories : int
    seed : int | None

    Notes
    -----
    ``max_depth`` is set so that P(fill) ≥ 1 %  →  δ_max = −ln(0.01)/κ.
    This is used to define the Gymnasium action-space upper bound
    (same convention as mbt-gym's ``ExponentialFillFunction``).
    """

    def __init__(
        self,
        fill_exponent: float = 35_000.0,
        num_trajectories: int = 1,
        seed: int | None = None,
    ):
        super().__init__(num_trajectories=num_trajectories, seed=seed)
        self.fill_exponent = fill_exponent

    def get_fills(self, depths: np.ndarray) -> np.ndarray:
        """
        Sample fills given quote depths.

        Parameters
        ----------
        depths : np.ndarray, shape (num_trajectories, 2)
            Non-negative bid/ask depths in price units (USDT).

        Returns
        -------
        fills : np.ndarray, shape (num_trajectories, 2), dtype int
            Binary: 1 = filled, 0 = not filled.
        """
        fill_probs = np.exp(-self.fill_exponent * depths)
        unif = self.rng.uniform(size=depths.shape)
        return (unif < fill_probs).astype(np.int32)

    @property
    def max_depth(self) -> float:
        """Depth at which fill probability drops to 1 %."""
        return -np.log(0.01) / self.fill_exponent


class DepthDependentPoissonFillFunction(FillProbabilityModel):
    """Exact one-step A-S execution model.

    For intensity ``lambda(delta) = A exp(-kappa delta)``, this samples

    ``P(execution in dt) = 1 - exp(-lambda(delta) * dt)``.

    It deliberately uses a single Bernoulli draw.  Multiplying a separate
    zero-depth arrival Bernoulli by a conditional depth Bernoulli is not the
    CDF of the depth-dependent Poisson process for finite ``dt``.
    """

    uses_depth_dependent_hazard = True

    def __init__(
        self,
        zero_depth_intensity: float,
        decay_coefficient: float,
        num_trajectories: int = 1,
        seed: int | None = None,
    ):
        super().__init__(num_trajectories=num_trajectories, seed=seed)
        if zero_depth_intensity <= 0 or decay_coefficient <= 0:
            raise ValueError("A and kappa must be positive")
        self.zero_depth_intensity = float(zero_depth_intensity)
        self.fill_exponent = float(decay_coefficient)

    def execution_probabilities(self, depths: np.ndarray, dt: float) -> np.ndarray:
        if dt < 0:
            raise ValueError("dt must be non-negative")
        intensity = self.zero_depth_intensity * np.exp(-self.fill_exponent * depths)
        return -np.expm1(-intensity * dt)

    def get_fills(self, depths: np.ndarray, dt: float | None = None) -> np.ndarray:
        if dt is None:
            raise ValueError("The exact depth-dependent hazard requires dt")
        probabilities = self.execution_probabilities(depths, dt)
        return (self.rng.uniform(size=depths.shape) < probabilities).astype(np.int32)

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent
