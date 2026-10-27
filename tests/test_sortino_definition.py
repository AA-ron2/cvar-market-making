"""The reported Sortino is Falces Marin et al. (2022) eq. (20) (DECISIONS D-13)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats


class ScriptedWealthEnv:
    """One trajectory whose wealth moves by the scripted amounts, one 60 s step each."""

    num_trajectories = 1
    max_inventory = 10

    def __init__(self, changes):
        self.wealth = np.concatenate([[0.0], np.cumsum(changes)])
        self.model_dynamics = SimpleNamespace(
            midprice_model=SimpleNamespace(dt_array=np.r_[0.0, np.full(len(changes), 60.0)], step_index=0)
        )

    def seed(self, seed):
        pass

    def _state(self):
        step = self.model_dynamics.midprice_model.step_index
        return np.array([[self.wealth[step], 0.0, 60.0 * step, 1.0]])

    def reset(self):
        self.model_dynamics.midprice_model.step_index = 0
        return self._state(), {}

    def step(self, action):
        model = self.model_dynamics.midprice_model
        model.step_index += 1
        return self._state(), np.zeros(1), np.array([model.step_index == len(self.wealth) - 1]), False, {}


def test_sortino_is_mean_return_over_std_of_negative_returns_about_their_mean():
    changes = [2.0, -1.0, -3.0, 1.0]
    agent = SimpleNamespace(get_action=lambda obs: np.zeros((1, 2)))
    stats = generate_trajectory_stats(ScriptedWealthEnv(changes), agent, seed=0, fixed_clock_seconds=60.0)

    assert stats["fixed_clock_return_count"] == 4
    # eq. (20): mean(r) = -0.25; negatives [-1, -3] have mean -2 and population std 1.
    assert stats["sortino"][0] == pytest.approx(-0.25)
    # Sortino & Price (1994) would divide by sqrt(mean(min(r, 0)^2)) = sqrt(2.5) instead.
    assert stats["sortino"][0] != pytest.approx(-0.25 / np.sqrt(2.5))
