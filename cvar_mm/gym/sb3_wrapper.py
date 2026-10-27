# Parts adapted from mbt-gym (Copyright (c) 2018, Joseph Jerome; BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
"""
Stable Baselines 3 vectorised-environment wrapper.

Wraps ``TradingEnvironment`` as an SB3 ``VecEnv`` so that PPO / SAC /
etc. can train directly on vectorised market-making rollouts.

API follows mbt-gym's ``StableBaselinesTradingEnvironment`` (Jerome et al., 2023, §4); parts adapted
from mbt-gym (BSD-3-Clause); see THIRD_PARTY_NOTICES.md. The wrapper:

    • ``step_async`` stores the action;
    • ``step_wait`` calls ``env.step()``, and on terminal episodes
      saves the final observation in ``info["terminal_observation"]``
      then auto-resets — the SB3 convention.

Usage::

    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from stable_baselines3 import PPO

    vec_env = StableBaselinesTradingEnvironment(env)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100_000)

Requirements:
    pip install stable-baselines3
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np

from cvar_mm.gym.trading_environment import TradingEnvironment
from cvar_mm.gym.index_names import ASSET_PRICE_INDEX, CASH_INDEX, INVENTORY_INDEX

try:
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.vec_env.base_vec_env import (
        VecEnvObs, VecEnvStepReturn, VecEnvIndices,
    )
    _SB3_AVAILABLE = True
except Exception:
    _SB3_AVAILABLE = False

    # Stub so the module can be imported without SB3 installed
    class VecEnv:
        def __init__(self, *a, **kw): ...


class StableBaselinesTradingEnvironment(VecEnv):
    """
    SB3-compatible ``VecEnv`` wrapper around ``TradingEnvironment``.

    Parameters
    ----------
    trading_env : TradingEnvironment
        Must have ``num_trajectories >= 1``.
    store_terminal_observation_info : bool
        If True (default), saves the final observation in
        ``info["terminal_observation"]`` before auto-reset.
    """

    def __init__(
        self,
        trading_env: TradingEnvironment,
        store_terminal_observation_info: bool = True,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for StableBaselinesTradingEnvironment. "
                "Install it with:  pip install stable-baselines3"
            )

        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info
        self.actions: np.ndarray = self.env.action_space.sample()
        self._wealth_peak: np.ndarray | None = None
        self._running_max_drawdown: np.ndarray | None = None

        super().__init__(
            self.env.num_trajectories,
            self.env.observation_space,
            self.env.action_space,
        )

    # ── VecEnv interface ──────────────────────────────────────
    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._reset_drawdown_tracker()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        obs, rewards, dones, truncated, infos = self.env.step(self.actions)

        # Build per-trajectory info dicts
        if not isinstance(infos, list):
            infos = [{} for _ in range(self.env.num_trajectories)]

        state = self._metric_state()
        wealth = state[:, CASH_INDEX] + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
        if self._wealth_peak is None:
            self._wealth_peak = wealth.copy()
            self._running_max_drawdown = np.zeros_like(wealth)
        self._wealth_peak = np.maximum(self._wealth_peak, wealth)
        assert self._running_max_drawdown is not None
        self._running_max_drawdown = np.maximum(
            self._running_max_drawdown, self._wealth_peak - wealth
        )
        day_index = int(getattr(self.env.model_dynamics.midprice_model, "day_index", 0))
        infos = [info.copy() for info in infos]
        for index, info in enumerate(infos):
            info["trajectory_max_drawdown"] = float(self._running_max_drawdown[index])
            info["replay_day_index"] = day_index

        if dones.min():
            if self.store_terminal_observation_info:
                infos = [info.copy() for info in infos]
                for count, info in enumerate(infos):
                    info["terminal_observation"] = obs[count, :]
            obs, _ = self.env.reset()
            self._reset_drawdown_tracker()

        return obs, rewards, dones, infos

    def _reset_drawdown_tracker(self) -> None:
        state = self._metric_state()
        wealth = state[:, CASH_INDEX] + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
        self._wealth_peak = wealth.copy()
        self._running_max_drawdown = np.zeros_like(wealth)

    def _metric_state(self) -> np.ndarray:
        state = getattr(self.env, "metric_state", None)
        if callable(state):
            state = state()
        if state is None:
            state = self.env.model_dynamics.state
        return np.asarray(state)

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list:
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list:
        return [False for _ in range(self.env.num_trajectories)]

    def seed(self, seed: Optional[int] = None) -> list:
        self.env.seed(seed)
        return [seed] * self.env.num_trajectories

    def get_images(self) -> Sequence[np.ndarray]:
        return []

    # ── convenience properties ────────────────────────────────
    @property
    def num_trajectories(self):
        return self.env.num_trajectories

    @property
    def n_steps(self):
        return self.env.n_steps
