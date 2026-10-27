from __future__ import annotations

import numpy as np
import pytest

from cvar_mm.gym.calibration import (
    FirstPassageCalibrationConfig,
    IntensityCurve,
    collect_censored_first_passage,
    contiguous_supported_depth_prefix,
    empirical_monotonicity_fraction,
    fit_exponential_intensity,
)
from cvar_mm.gym.helpers.fast_rollout import fast_simulate
from cvar_mm.stochastic_processes.fill_probability_models import (
    DepthDependentPoissonFillFunction,
)


def _config(**overrides):
    values = dict(
        tick_size=1.0,
        depth_ticks=(1.0,),
        censor_horizon_seconds=5.0,
        execution_proxy="midpoint_first_passage",  # explicit: the default is F-03's best_price_reach
        exposure_start_rule="all_snapshots",
        min_hits_per_depth=1,
    )
    values.update(overrides)
    return FirstPassageCalibrationConfig(**values)


def test_market_end_exposures_are_right_censored_with_available_time():
    mid = np.array([100.0, 100.0, 100.0])
    dt = np.array([0.0, 1.0, 1.0])
    bid, ask = collect_censored_first_passage(mid, dt, config=_config())
    for curve in (bid, ask):
        assert curve.exposures.tolist() == [2]
        assert curve.censored.tolist() == [2]
        assert curve.hits.tolist() == [0]
        assert curve.exposures.tolist() == (curve.hits + curve.censored).tolist()
        assert curve.censoring_fraction.tolist() == pytest.approx([1.0])
        assert curve.total_exposure_seconds.tolist() == pytest.approx([3.0])


def test_proxy_choice_changes_the_explicit_hit_definition():
    mid = np.array([100.0, 100.0, 100.0])
    dt = np.array([0.0, 1.0, 1.0])
    best_bid = np.array([99.5, 99.5, 99.5])
    best_ask = np.array([100.5, 101.5, 100.5])
    _, midpoint_ask = collect_censored_first_passage(mid, dt, config=_config())
    _, l2_ask = collect_censored_first_passage(
        mid,
        dt,
        config=_config(execution_proxy="best_price_reach"),
        best_bid=best_bid,
        best_ask=best_ask,
    )
    assert midpoint_ask.hits.tolist() == [0]
    assert l2_ask.hits.tolist() == [1]


def test_intensity_mle_and_fit_use_hits_over_exposure_seconds():
    depths = np.array([0.0, 1.0, 2.0])
    A, kappa = 4.0, 0.7
    exposure = np.full(3, 10_000.0)
    hits = np.rint(A * np.exp(-kappa * depths) * exposure).astype(np.int64)
    curve = IntensityCurve(
        side="pooled",
        depth_ticks=depths,
        depth_price=depths,
        hits=hits,
        total_exposure_seconds=exposure,
        exposures=np.full(3, 50_000),
        censored=np.full(3, 50_000) - hits,
    )
    fit = fit_exponential_intensity(curve, min_hits_per_depth=10)
    assert curve.intensity_per_second == pytest.approx(hits / exposure)
    assert fit.A == pytest.approx(A, rel=2e-4)
    assert fit.kappa == pytest.approx(kappa, rel=2e-4)
    assert fit.rmse_log < 1e-3
    assert fit.A > 0 and fit.kappa > 0


def test_profile_monotonicity_and_contiguous_support_diagnostics():
    curve = IntensityCurve(
        side="pooled",
        depth_ticks=np.array([1.0, 2.0, 3.0, 4.0]),
        depth_price=np.array([1.0, 2.0, 3.0, 4.0]),
        hits=np.array([100, 80, 5, 90]),
        total_exposure_seconds=np.full(4, 100.0),
        exposures=np.array([120, 100, 20, 110]),
        censored=np.array([20, 20, 15, 20]),
    )

    # Valid depths are 1, 2, and 4: intensity falls once, then rises once.
    assert empirical_monotonicity_fraction(curve, min_hits_per_depth=10) == pytest.approx(0.5)
    assert contiguous_supported_depth_prefix(
        curve, min_hits_per_depth=10
    ).tolist() == [1.0, 2.0]
    assert curve.censoring_fraction == pytest.approx([1 / 6, 0.2, 0.75, 2 / 11])


def test_exact_execution_probability_is_depth_dependent_poisson_cdf():
    model = DepthDependentPoissonFillFunction(2.0, 0.5, seed=7)
    depths = np.array([[0.0, 2.0]])
    expected = 1.0 - np.exp(-2.0 * np.exp(-0.5 * depths) * 0.25)
    assert model.execution_probabilities(depths, 0.25) == pytest.approx(expected)


def test_fast_b0_quotes_at_interval_start_then_marks_endpoint():
    common = dict(
        dt_array=np.array([0.0, 1.0]),
        gamma=0.1,
        sigma=0.0,
        kappa=0.1,
        A=1.0,
        terminal_time=1.0,
        tick_size=0.01,
        Q_MAX=10,
        num_trajectories=1,
    )
    for seed in range(100):
        flat = fast_simulate(midprices=np.array([100.0, 100.0]), seed=seed, **common)
        moved = fast_simulate(midprices=np.array([100.0, 110.0]), seed=seed, **common)
        terminal_q = float(flat["terminal_q"][0])
        if terminal_q != 0:
            assert moved["terminal_q"][0] == terminal_q
            assert moved["total_pnl"][0] - flat["total_pnl"][0] == pytest.approx(10.0 * terminal_q)
            break
    else:  # pragma: no cover - deterministic seeds above produce a one-sided fill
        pytest.fail("No one-sided fill found in deterministic smoke seeds")
