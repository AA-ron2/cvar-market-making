"""
Trading environment.

``TradingEnvironment`` is a Gymnasium-compatible ``Env`` that orchestrates
the model-dynamics, reward function, and stochastic processes.

Architecture reference (Jerome et al., 2023, §3, Fig. 1):
    TradingEnvironment
      ├── model_dynamics  (LimitOrderModelDynamics)
      │     ├── midprice_model
      │     ├── arrival_model
      │     └── fill_probability_model
      └── reward_function  (PnLReward)

State vector layout (see ``index_names.py``):
    [cash, inventory, time, midprice]

Adaptation notes vs. standard mbt-gym
--------------------------------------
• mbt-gym uses a fixed ``step_size`` = T / n_steps.
  Here, Δt is *variable* (read from ``MarketReplayMidpriceModel``),
  because Tardis L2 snapshots have irregular timestamps.
• Episode length = number of data snapshots (not a configurable n_steps).
• ``num_trajectories`` defaults to 1.  Vectorisation is a planned
  future extension.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from cvar_mm.gym.index_names import (
    CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX,
)
from cvar_mm.gym.model_dynamics import LimitOrderModelDynamics
from cvar_mm.rewards import PnLReward, RewardFunction
from cvar_mm.stochastic_processes.midprice_models import MarketReplayMidpriceModel


class TradingEnvironment(gymnasium.Env):
    """
    Gymnasium environment for limit-order market making.

    Parameters
    ----------
    model_dynamics : LimitOrderModelDynamics
        Pre-wired dynamics (midprice + arrival + fill models).
    reward_function : RewardFunction
        Per-step reward.  Defaults to ``PnLReward``.
    initial_cash : float
        Starting cash balance.
    initial_inventory : int
        Starting inventory position.
    max_inventory : int
        Hard position limit (±Q_MAX).  Fills that would breach it are
        suppressed — same as mbt-gym's ``_remove_max_inventory_fills``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model_dynamics: LimitOrderModelDynamics,
        reward_function: RewardFunction | None = None,
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        max_inventory: int = 10,
        normalise_action_space: bool = False,
        normalise_observation_space: bool = False,
        normalise_rewards: bool = False,
        reward_scale: float | None = None,
        feature_computer=None,
    ):
        super().__init__()

        self.model_dynamics = model_dynamics
        self.reward_function = reward_function or PnLReward()
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.normalise_action_space_ = normalise_action_space
        self.normalise_observation_space_ = normalise_observation_space
        self.normalise_rewards_ = normalise_rewards
        self.feature_computer = feature_computer

        # Alias for readability
        self._midprice_model: MarketReplayMidpriceModel = (
            model_dynamics.midprice_model
        )
        self.num_trajectories = model_dynamics.num_trajectories

        # Episode geometry — match mbt-gym's env.n_steps / env.terminal_time
        self.n_steps = self._midprice_model.n_snapshots - 1
        self.terminal_time = float(self._midprice_model.dt_array.sum())
        self._seed = None

        # Gymnasium spaces (raw, before normalisation)
        self.observation_space = self._build_observation_space()
        self.action_space = model_dynamics.get_action_space()

        # Normalisation caches (mbt-gym convention: linear map to [-1, 1])
        if self.normalise_observation_space_:
            from copy import copy
            self.original_observation_space = copy(self.observation_space)
            self._obs_low = self.observation_space.low
            self._obs_range = (self.observation_space.high - self.observation_space.low) / 2.0
            self.observation_space = Box(
                low=-np.ones_like(self._obs_low, dtype=np.float32),
                high=np.ones_like(self._obs_low, dtype=np.float32),
            )

        if self.normalise_action_space_:
            from copy import copy
            self.original_action_space = copy(self.action_space)
            self._act_low = self.action_space.low
            self._act_range = (self.action_space.high - self.action_space.low) / 2.0
            self.action_space = Box(
                low=-np.ones_like(self._act_low, dtype=np.float32),
                high=np.ones_like(self._act_low, dtype=np.float32),
            )

        # Reward scaling
        if self.normalise_rewards_:
            self.reward_scale = reward_scale if reward_scale is not None else 1.0
        else:
            self.reward_scale = 1.0

        # Internal state — set properly in reset()
        self.model_dynamics.state = None

    # ── Gymnasium interface ───────────────────────────────────
    def seed(self, seed: int | None = None) -> None:
        """Set RNG seed for arrival and fill models (mbt-gym compat)."""
        self._seed = seed
        if seed is not None:
            self.model_dynamics.arrival_model.seed(seed)
            self.model_dynamics.fill_probability_model.seed(seed + 1)

    # ── Normalisation helpers (mbt-gym convention) ────────────
    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.normalise_observation_space_:
            return (obs - self._obs_low) / self._obs_range - 1.0
        return obs

    def _denormalise_action(self, action: np.ndarray) -> np.ndarray:
        if self.normalise_action_space_:
            return (action + 1.0) * self._act_range + self._act_low
        return action

    def _scale_reward(self, reward: np.ndarray) -> np.ndarray:
        return reward * self.reward_scale

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset to the start of the data session."""
        super().reset(seed=seed)

        # Reset all stochastic processes
        self._midprice_model.reset()
        # A multi-day replay switches day on reset; T and the step count follow it (K9).
        self.n_steps = self._midprice_model.n_snapshots - 1
        self.terminal_time = float(self._midprice_model.dt_array.sum())
        self.model_dynamics.arrival_model.reset()
        self.model_dynamics.fill_probability_model.reset()

        # Re-seed if requested
        if seed is not None:
            self.model_dynamics.arrival_model.seed(seed)
            self.model_dynamics.fill_probability_model.seed(seed + 1)

        # Build initial state: [cash, inventory, time=0, midprice_0]
        state = np.zeros((self.num_trajectories, 4))
        state[:, CASH_INDEX] = self.initial_cash
        state[:, INVENTORY_INDEX] = self.initial_inventory
        state[:, TIME_INDEX] = 0.0
        state[:, ASSET_PRICE_INDEX] = self._midprice_model.current_state[:, 0]
        self.model_dynamics.state = state

        self.reward_function.reset(state.copy())

        # Reset feature computer
        if self.feature_computer is not None:
            self.feature_computer.reset(float(self._midprice_model.current_state[0, 0]))

        return self._obs(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : np.ndarray, shape (2,) or (num_trajectories, 2)
            Bid/ask depths [δ_bid, δ_ask].

        Returns
        -------
        obs       : np.ndarray, shape (4,)
        reward    : float
        terminated: bool   (True at end of data)
        truncated : bool   (always False — no early truncation)
        info      : dict
        """
        # Ensure action is 2-D  (N, 2)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        # Denormalise action from [-1,1] → [0, max_depth] if enabled
        action = self._denormalise_action(action)

        current_state = self.model_dynamics.state.copy()

        # 1. Read Δt from the *next* snapshot
        dt = self._midprice_model.dt_array[
            self._midprice_model.step_index + 1
        ]

        # 2. Skip zero-dt snapshots (duplicate timestamps) — no fills
        #    but midprice may still change, so compute MTM reward.
        if dt <= 0:
            self._advance_midprice()
            self.model_dynamics.state[:, ASSET_PRICE_INDEX] = (
                self._midprice_model.current_state[:, 0]
            )
            if self.feature_computer is not None:
                self.feature_computer.update(
                    float(self._midprice_model.current_state[0, 0]), 0.0,
                )
            next_state = self.model_dynamics.state.copy()
            reward = self._scale_reward(self.reward_function.calculate(
                current_state, action, next_state, self._terminated(),
            ))
            dones = np.full((self.num_trajectories,), self._terminated(), dtype=bool)
            return self._obs(), reward, dones, False, {}

        # 3. Sample arrivals & fills
        arrivals, fills = self.model_dynamics.get_arrivals_and_fills(action, dt)

        # 4. Suppress fills that would breach Q_MAX
        fills = self._enforce_inventory_limits(fills)

        # 5. Update agent state (cash, inventory)
        self.model_dynamics.update_state(arrivals, fills, action)

        # 6. Advance time and midprice
        self._advance_midprice()
        self.model_dynamics.state[:, TIME_INDEX] += dt
        self.model_dynamics.state[:, ASSET_PRICE_INDEX] = (
            self._midprice_model.current_state[:, 0]
        )

        # 6b. Update feature computer with new price
        if self.feature_computer is not None:
            self.feature_computer.update(
                float(self._midprice_model.current_state[0, 0]), float(dt),
            )

        # 7. Reward
        next_state = self.model_dynamics.state.copy()
        reward = self._scale_reward(self.reward_function.calculate(
            current_state, action, next_state, self._terminated(),
        ))

        done = self._terminated()
        dones = np.full((self.num_trajectories,), done, dtype=bool)

        return (
            self._obs(),
            reward,              # (num_trajectories,)
            dones,               # (num_trajectories,) bool
            False,
            {},
        )

    # ── internal helpers ──────────────────────────────────────
    def _obs(self) -> np.ndarray:
        """Return observation, with optional features appended, normalised if enabled."""
        base = self.model_dynamics.state.copy()  # (N, 4)
        if self.feature_computer is not None:
            feats = self.feature_computer.compute()  # (n_features,)
            feats_broadcast = np.repeat(feats.reshape(1, -1), self.num_trajectories, axis=0)
            base = np.concatenate([base, feats_broadcast], axis=1)
        return self._normalise_obs(base)

    def _terminated(self) -> bool:
        return self._midprice_model.steps_remaining <= 0

    def _advance_midprice(self) -> None:
        self._midprice_model.update(
            arrivals=None, fills=None, action=None,
        )

    def _enforce_inventory_limits(self, fills: np.ndarray) -> np.ndarray:
        """
        Suppress fills that would push inventory beyond ±max_inventory.

        Same logic as mbt-gym's ``_remove_max_inventory_fills``:
        if at +Q_MAX  → zero out bid fills (no more buying);
        if at −Q_MAX  → zero out ask fills (no more selling).
        """
        q = self.model_dynamics.state[:, INVENTORY_INDEX]
        at_max = (q >= self.max_inventory).astype(np.int32).reshape(-1, 1)
        at_min = (q <= -self.max_inventory).astype(np.int32).reshape(-1, 1)
        mask = np.concatenate((1 - at_max, 1 - at_min), axis=1)
        return fills * mask

    def _build_observation_space(self) -> gymnasium.spaces.Space:
        """
        Observation space: [cash, inventory, time, midprice, (features...)].

        Bounds follow mbt-gym's ``_get_observation_space`` convention.
        Features are appended if ``feature_computer`` is provided.
        """
        mp = self._midprice_model
        max_cash = float(mp.n_snapshots * mp.max_value[0, 0])
        T_sec = float(mp.dt_array.sum())

        low = np.array(
            [-max_cash, -self.max_inventory, 0.0, mp.min_value[0, 0]],
            dtype=np.float32,
        )
        high = np.array(
            [max_cash, self.max_inventory, T_sec, mp.max_value[0, 0]],
            dtype=np.float32,
        )

        # Append feature bounds if feature_computer is set
        if self.feature_computer is not None:
            feat_low, feat_high = self.feature_computer.get_bounds()
            low = np.concatenate([low, feat_low])
            high = np.concatenate([high, feat_high])

        return Box(low=low, high=high)

    # ── properties ────────────────────────────────────────────
    @property
    def midprice(self) -> float:
        return float(self._midprice_model.current_state[0, 0])

    @property
    def inventory(self) -> float:
        return float(self.model_dynamics.state[0, INVENTORY_INDEX])

    @property
    def cash(self) -> float:
        return float(self.model_dynamics.state[0, CASH_INDEX])

    @property
    def time(self) -> float:
        return float(self.model_dynamics.state[0, TIME_INDEX])
