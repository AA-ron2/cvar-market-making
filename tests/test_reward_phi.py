from __future__ import annotations

import numpy as np

from cvar_mm.gym.experiment_config import BMExperimentConfig, ReplayExperimentConfig
from cvar_mm.rewards import CjMmCriterion, PnLReward


def test_replay_reward_has_no_inventory_penalty_and_bm_stage_is_unchanged():
    assert ReplayExperimentConfig().phi == 0.0
    assert BMExperimentConfig().phi == 0.01


def test_cj_criterion_with_zero_phi_equals_pnl_reward_exactly():
    rng = np.random.default_rng(0)
    current = rng.normal(size=(256, 4))
    following = rng.normal(size=(256, 4))
    current[:, 2] = rng.uniform(0.0, 86_400.0, 256)
    following[:, 2] = current[:, 2] + rng.exponential(1.0, 256)
    following[:, 1] = rng.integers(-10, 11, 256)
    action = rng.normal(size=(256, 2))

    expected = PnLReward().calculate(current, action, following)
    for phi in (0.0, ReplayExperimentConfig().phi):
        np.testing.assert_array_equal(
            CjMmCriterion(per_step_inventory_aversion=phi).calculate(current, action, following),
            expected,
        )
