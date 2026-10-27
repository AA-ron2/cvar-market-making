from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from cvar_mm.gym.formula_as import (
    FormulaASActionConfig,
    FormulaASActionWrapper,
    compute_formula_as_depths,
    map_formula_action,
)


def _cfg() -> FormulaASActionConfig:
    return FormulaASActionConfig(
        sigma=2.5e-5,
        kappa=35_000.0,
        tick_size=1e-5,
        gamma_min=0.01,
        gamma_max=0.9,
        skew_ticks_max=5.0,
    )


def test_formula_action_mapping_hits_bounds() -> None:
    gamma, skew = map_formula_action(np.array([[-1.0, -1.0], [1.0, 1.0]]), _cfg())

    assert np.isclose(gamma[0], 0.01)
    assert np.isclose(gamma[1], 0.9)
    assert np.isclose(skew[0], -5e-5)
    assert np.isclose(skew[1], 5e-5)


def test_formula_quotes_are_non_crossed_and_depths_nonnegative() -> None:
    depths, diagnostics = compute_formula_as_depths(
        midprice=np.array([0.1, 0.1]),
        inventory=np.array([10.0, -10.0]),
        time_elapsed=np.array([0.0, 10.0]),
        terminal_time=60.0,
        action=np.array([[0.0, -1.0], [0.0, 1.0]]),
        config=_cfg(),
    )

    assert depths.shape == (2, 2)
    assert np.all(depths > 0)
    assert np.all(diagnostics["bid"] < diagnostics["ask"])
    assert np.all(diagnostics["bid"] < 0.1)
    assert np.all(diagnostics["ask"] > 0.1)


def test_formula_wrapper_smoke_step() -> None:
    pytest.importorskip("gymnasium")

    from cvar_mm.gym.model_dynamics import LimitOrderModelDynamics
    from cvar_mm.gym.trading_environment import TradingEnvironment
    from cvar_mm.rewards import PnLReward
    from cvar_mm.stochastic_processes import (
        ExponentialFillFunction,
        MarketReplayMidpriceModel,
        PoissonArrivalModel,
    )

    midprices = np.array([0.10000, 0.10001, 0.10002, 0.10001], dtype=float)
    dt = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)
    base_env = TradingEnvironment(
        model_dynamics=LimitOrderModelDynamics(
            midprice_model=MarketReplayMidpriceModel(midprices, dt, num_trajectories=2),
            arrival_model=PoissonArrivalModel(np.array([0.1, 0.1]), num_trajectories=2),
            fill_probability_model=ExponentialFillFunction(35_000.0, num_trajectories=2),
            num_trajectories=2,
        ),
        reward_function=PnLReward(),
        max_inventory=10,
    )
    env = FormulaASActionWrapper(base_env, action_config=_cfg())

    obs, _ = env.reset(seed=123)
    assert obs.shape == (2, 7)

    obs, reward, done, _, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert obs.shape == (2, 7)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert env.last_depth_action is not None
    assert env.last_depth_action.shape == (2, 2)
    assert "depth_action" in info
