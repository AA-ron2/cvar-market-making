"""
Model dynamics.

``ModelDynamics`` is the *glue layer* that connects the stochastic
processes (midprice, arrivals, fills) and defines how agent state
(cash, inventory) is updated on each step.

Architecture reference (Jerome et al., 2023, §3, Fig. 1):
    TradingEnvironment  ──owns──▶  ModelDynamics
    ModelDynamics       ──owns──▶  MidpriceModel
                                   ArrivalModel
                                   FillProbabilityModel

This module provides:
    • ``LimitOrderModelDynamics`` — market-making with limit orders
      (bid/ask depths as actions).  Matches the Avellaneda-Stoikov
      problem formulation.

Cash / inventory update (mbt-gym convention):
    fill_multiplier = [[-1, 1]]   (per trajectory)
    • bid fill  → inventory += 1,  cash -= bid_price
    • ask fill  → inventory -= 1,  cash += ask_price
"""

from __future__ import annotations

import gymnasium
import numpy as np

from cvar_mm.gym.index_names import (
    CASH_INDEX, INVENTORY_INDEX, BID_INDEX, ASK_INDEX,
)
from cvar_mm.stochastic_processes.midprice_models import MidpriceModel
from cvar_mm.stochastic_processes.arrival_models import ArrivalModel
from cvar_mm.stochastic_processes.fill_probability_models import FillProbabilityModel


# ── base class ────────────────────────────────────────────────
class ModelDynamics:
    """Minimal base following mbt-gym's ``ModelDynamics``."""

    def __init__(
        self,
        midprice_model: MidpriceModel,
        arrival_model: ArrivalModel | None = None,
        fill_probability_model: FillProbabilityModel | None = None,
        num_trajectories: int = 1,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.num_trajectories = num_trajectories

        # fill_multiplier encodes the cash-flow sign convention:
        #   bid fill  →  buy  →  cash decreases  →  multiplier = -1
        #   ask fill  →  sell →  cash increases  →  multiplier = +1
        ones = np.ones((num_trajectories, 1))
        self.fill_multiplier = np.concatenate((-ones, ones), axis=1)

        self.state: np.ndarray | None = None        # set by env.reset()

    def get_arrivals_and_fills(
        self, action: np.ndarray, dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update_state(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def get_action_space(self) -> gymnasium.spaces.Space:
        raise NotImplementedError

    @property
    def midprice(self) -> np.ndarray:
        """Current midprice, shape (num_trajectories, 1)."""
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)


# ── concrete: limit-order market making ───────────────────────
class LimitOrderModelDynamics(ModelDynamics):
    """
    Dynamics for limit-order market making (the A-S problem).

    Actions are bid/ask depths ``[δ_bid, δ_ask]`` (non-negative, in
    price units).  At each step the model:
        1. Samples whether market orders arrive  (arrival model).
        2. Samples whether those orders fill the agent's quotes
           at the given depths  (fill-probability model).
        3. Updates cash and inventory accordingly.

    This mirrors ``LimitOrderModelDynamics`` in mbt-gym.

    Parameters
    ----------
    midprice_model : MidpriceModel
    arrival_model : ArrivalModel
    fill_probability_model : FillProbabilityModel
    num_trajectories : int
    max_depth : float | None
        Upper bound for depth action space.  If ``None``, derived from
        the fill-probability model (depth at which P(fill) = 1 %).
    """

    def __init__(
        self,
        midprice_model: MidpriceModel,
        arrival_model: ArrivalModel,
        fill_probability_model: FillProbabilityModel,
        num_trajectories: int = 1,
        max_depth: float | None = None,
    ):
        super().__init__(
            midprice_model=midprice_model,
            arrival_model=arrival_model,
            fill_probability_model=fill_probability_model,
            num_trajectories=num_trajectories,
        )
        self.max_depth = max_depth or fill_probability_model.max_depth

    # ── core interface ────────────────────────────────────────
    def get_arrivals_and_fills(
        self, action: np.ndarray, dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample arrivals and fills for the current step.

        Parameters
        ----------
        action : np.ndarray, shape (num_trajectories, 2)
            Bid/ask depths [δ_bid, δ_ask].
        dt : float
            Time step in seconds (variable for market replay).

        Returns
        -------
        arrivals : np.ndarray, shape (N, 2), binary
        fills    : np.ndarray, shape (N, 2), binary
        """
        depths = action[:, BID_INDEX : ASK_INDEX + 1]       # (N, 2)
        if getattr(self.fill_probability_model, "uses_depth_dependent_hazard", False):
            # The exact model represents execution directly, so no independent
            # zero-depth arrival draw is multiplied into this Bernoulli event.
            fills = self.fill_probability_model.get_fills(depths, dt)
            arrivals = np.ones_like(fills)
        else:
            arrivals = self.arrival_model.get_arrivals(dt)
            fills = self.fill_probability_model.get_fills(depths)
        return arrivals, fills

    def update_state(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
    ) -> None:
        """
        Update cash and inventory from fill events.

        Cash update:
            Δcash = Σ_side [ sign · arrival · fill · (S ± δ) ]
        Inventory update:
            Δq    = Σ_side [ arrival · fill · (−sign) ]

        where sign = fill_multiplier  (−1 for bid, +1 for ask).
        """
        depths = action[:, BID_INDEX : ASK_INDEX + 1]       # (N, 2)
        execution_prices = self.midprice + depths * self.fill_multiplier
        # midprice - δ_bid  for bid,  midprice + δ_ask  for ask

        executed = arrivals * fills                          # (N, 2) binary

        self.state[:, INVENTORY_INDEX] += np.sum(
            executed * (-self.fill_multiplier), axis=1,
        )
        self.state[:, CASH_INDEX] += np.sum(
            self.fill_multiplier * executed * execution_prices, axis=1,
        )

    def get_action_space(self) -> gymnasium.spaces.Space:
        """Box(0, max_depth, shape=(2,)) — bid depth & ask depth."""
        return gymnasium.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(self.max_depth),
            shape=(2,),
        )
