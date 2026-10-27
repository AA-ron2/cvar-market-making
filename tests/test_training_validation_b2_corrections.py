from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvar_mm.gym.drawdown_objectives import (
    CompleteTrajectorySegment,
    add_terminal_maxdd_score_advantage,
)
from cvar_mm.gym.model_selection import choose_b2_alpha
from cvar_mm.rewards import CjMmDrawdownPenalty
from cvar_mm.stochastic_processes.midprice_models import MultiDayReplayMidpriceModel


def test_incremental_drawdown_reward_telescopes_to_terminal_maxdd():
    reward = CjMmDrawdownPenalty(per_step_inventory_aversion=0.0, drawdown_penalty=3.0)
    states = [
        np.array([[0.0, 0.0, 0.0, 100.0]]),
        np.array([[-2.0, 0.0, 1.0, 100.0]]),
        np.array([[-1.0, 0.0, 2.0, 100.0]]),
        np.array([[-4.0, 0.0, 3.0, 100.0]]),
    ]
    reward.reset(states[0])
    values = [
        reward.calculate(states[index], np.zeros((1, 2)), states[index + 1])[0]
        for index in range(3)
    ]
    assert values == pytest.approx([-8.0, 1.0, -9.0])
    assert sum(values) == pytest.approx(-4.0 - 3.0 * 4.0)


def test_b2_risk_score_is_added_to_ppo_advantages_for_every_action():
    segments = [
        CompleteTrajectorySegment(0, 2, 0, 0, 1.0),
        CompleteTrajectorySegment(0, 2, 1, 0, 3.0),
    ]
    advantages = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)  # mu_A = 0, sigma_A = 1
    adjusted = add_terminal_maxdd_score_advantage(advantages, segments, alpha=2.0, reward_scale=0.5).advantages
    # Leave-one-out within the day: Z minus the other path's Z is [-2, 2]; every action of the path
    # gets -alpha * that / (s * sigma_A) = [8, -8] on top of its standardized advantage (D-15a).
    assert adjusted[:, 0] == pytest.approx([9.0, 7.0])
    assert adjusted[:, 1] == pytest.approx([-9.0, -7.0])


def test_sequential_replay_reports_completed_coverage_for_every_day():
    prices = [np.array([1.0, 1.1]), np.array([2.0, 2.1, 2.2])]
    deltas = [np.array([0.0, 1.0]), np.array([0.0, 1.0, 1.0])]
    model = MultiDayReplayMidpriceModel(prices, deltas, mode="sequential")
    dummy = np.zeros((1, 2))
    for day_steps in (1, 2):
        model.reset()
        for _ in range(day_steps):
            model.update(dummy, dummy, dummy)
    model.reset()  # records completion of the second day
    assert model.coverage_report["completed"] == [1, 1]


def test_b2_selection_uses_validation_rule_and_deterministic_ties():
    positive = pd.DataFrame([
        {"alpha": 1.0, "mean_sharpe": 0.2, "mean_maxdd": 2.0},
        {"alpha": 0.1, "mean_sharpe": 0.2, "mean_maxdd": 2.0},
        {"alpha": 10.0, "mean_sharpe": 0.4, "mean_maxdd": 3.0},
    ])
    assert choose_b2_alpha(positive) == 0.1

    fallback = pd.DataFrame([
        {"alpha": 1.0, "mean_sharpe": -0.1, "mean_maxdd": 2.0},
        {"alpha": 0.1, "mean_sharpe": -0.2, "mean_maxdd": 1.0},
    ])
    assert choose_b2_alpha(fallback) == 1.0


@pytest.mark.parametrize("alpha", [0.0, -0.1, float("nan"), float("inf")])
def test_b2_selection_refuses_a_non_positive_alpha(alpha):
    """D-18: alpha = 0 is the B1+ control, which can never be a B2 candidate (F-05); here it would even win."""

    summary = pd.DataFrame([
        {"alpha": alpha, "mean_sharpe": 0.5, "mean_maxdd": 0.5},
        {"alpha": 1.0, "mean_sharpe": 0.2, "mean_maxdd": 2.0},
    ])
    with pytest.raises(ValueError, match="F-05"):
        choose_b2_alpha(summary)


def test_select_b2_refuses_a_non_positive_alpha_before_reading_anything(monkeypatch):
    import importlib

    aggregate = importlib.import_module("scripts.pipeline.aggregate_final_results")
    monkeypatch.setattr(aggregate, "load_context", lambda **_: pytest.fail("select_b2 read the pipeline first"))
    with pytest.raises(ValueError, match="F-05"):
        aggregate.select_b2([0.0, 1.0])
