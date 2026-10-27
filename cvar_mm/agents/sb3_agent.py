"""
Stable Baselines 3 trained agent wrapper.

Wraps a trained SB3 model so it conforms to the ``Agent`` interface
and can be used with ``plot_trajectory``, ``backtest_summary``, etc.

Two normalisation modes are supported:

1. **Manual normalisation** (BM stage, nb1):
       agent = Sb3Agent(model, train_env=normalised_trading_env)
   Obs are normalised using fixed bounds from the training env,
   actions are denormalised the same way.

2. **VecNormalize** (market replay stage, nb2/nb3):
       agent = Sb3Agent(model, vecnorm_env=frozen_vecnorm_env)
   Obs are normalised using the VecNormalize running statistics
   (frozen at evaluation time). Actions are NOT denormalised —
   the model was trained on a normalised action space via the
   TradingEnvironment's built-in action normalisation, or on raw
   actions directly (depending on setup).

If both are provided, VecNormalize takes precedence.
"""

from __future__ import annotations

import numpy as np

from cvar_mm.agents.agent import Agent


class Sb3Agent(Agent):
    """
    Wraps a trained SB3 model as an ``Agent``.

    Parameters
    ----------
    model : stable_baselines3 model
        Trained PPO / SAC / etc.
    train_env : TradingEnvironment | None
        The manually-normalised env the model was trained on (nb1 / BM stage).
        If provided, obs are normalised using fixed bounds and actions
        denormalised. Used when normalise_observation_space=True was set.
    vecnorm_env : VecNormalize | None
        A VecNormalize wrapper with training=False and norm_reward=False,
        loaded from the saved .pkl file (nb2/nb3 / market replay stage).
        If provided, obs normalisation uses VecNormalize running statistics.
        Takes precedence over train_env.
    deterministic : bool
        If True (default), uses the mean action (no exploration noise).
    """

    def __init__(
        self,
        model,
        train_env=None,
        vecnorm_env=None,
        deterministic: bool = True,
    ):
        self.model = model
        self.deterministic = deterministic
        self._vecnorm_env = vecnorm_env

        # Manual normalisation parameters (BM stage)
        self._norm_obs = False
        self._norm_act = False

        if vecnorm_env is not None:
            # VecNormalize mode: obs normalised via running stats
            # Actions are raw (VecNormalize doesn't touch actions)
            pass
        elif train_env is not None:
            if getattr(train_env, "normalise_observation_space_", False):
                self._norm_obs = True
                self._obs_low = train_env._obs_low.copy()
                self._obs_range = train_env._obs_range.copy()
            if getattr(train_env, "normalise_action_space_", False):
                self._norm_act = True
                self._act_low = train_env._act_low.copy()
                self._act_range = train_env._act_range.copy()

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply whichever normalisation was used at training time."""
        if self._vecnorm_env is not None:
            # VecNormalize expects shape (1, obs_dim); returns same shape
            norm = self._vecnorm_env.normalize_obs(obs)
            return norm
        if self._norm_obs:
            return (obs - self._obs_low) / self._obs_range - 1.0
        return obs

    def _denormalise_action(self, action: np.ndarray) -> np.ndarray:
        """Undo action normalisation if manual normalisation was used."""
        if self._norm_act:
            return (action + 1.0) * self._act_range + self._act_low
        return action

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        state : np.ndarray, shape (num_trajectories, obs_dim)
            Raw (non-normalised) observation from the environment.

        Returns
        -------
        action : np.ndarray, shape (num_trajectories, 2)
            Raw depths [δ_bid, δ_ask] in price units.
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        N = state.shape[0]
        norm_state = self._normalise_obs(state)

        actions = np.zeros((N, 2))
        for i in range(N):
            act, _ = self.model.predict(
                norm_state[i], deterministic=self.deterministic,
            )
            actions[i] = self._denormalise_action(act)

        return actions
