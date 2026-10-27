"""FeatureEngine against the frozen FormulaASActionWrapper on a synthetic day (no data needed), and its
state round trip and refusals. The full Jan 7 day is test_hb_feature_parity.py (T2)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from hummingbot_adapter.features import FeatureEngine, row_columns
from hummingbot_adapter.tests.support import RecordingAgent, synthetic_book, write_book_csv
from cvar_mm.gym.data_loader import (
    load_single_day_with_features,
    read_market_data_csv,
    stable_order_by_exchange_timestamp,
)
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
from cvar_mm.gym.notebook_support import build_formula_replay_env
from cvar_mm.rewards import PnLReward

SIGMA, KAPPA = 2.7568397609849377e-05, 12066.410319238006


@pytest.fixture(scope="module")
def synthetic_replay(tmp_path_factory):
    """The frozen replay of a synthetic 3,000-row day under a random policy, recorded step by step."""

    rng = np.random.default_rng(7)
    path = write_book_csv(
        tmp_path_factory.mktemp("book") / "binance_book_snapshot_25_2025-01-02_DOGEUSDT.csv",
        synthetic_book(rng, 3_000),
    )
    config = replace(ReplayExperimentConfig(), A=2.0, kappa=KAPPA)
    S, dt, _, features = load_single_day_with_features(str(path), tick_size=config.tick_size)
    env = build_formula_replay_env(
        S, dt, config, sigma=SIGMA, market_features=features, reward_fn=PnLReward(), num_trajectories=1
    )
    action_rng = np.random.default_rng(11)
    agent = RecordingAgent(env, len(S), lambda state: action_rng.uniform(-1.0, 1.0, size=(1, 2)))
    generate_trajectory_stats(env, agent, seed=42, fixed_clock_seconds=config.fixed_clock_seconds)
    agent.finish()
    frame = stable_order_by_exchange_timestamp(read_market_data_csv(str(path)))
    rows = frame[list(row_columns(5))].to_dict("records")
    return {"agent": agent, "rows": rows, "dt": dt, "T": float(env.terminal_time), "q_max": config.q_max}


def _engine_observations(replay, engine: FeatureEngine, start: int = 0, stop: int | None = None):
    """The engine's observations and its float64 mid (the S of the quotes) at each row."""

    agent, rows, dt, T = replay["agent"], replay["rows"], replay["dt"], replay["T"]
    stop = len(rows) if stop is None else stop
    out, mids = [], []
    for k in range(start, stop):
        if k == 0:
            out.append(engine.reset(rows[0], agent.t[0], T, agent.q[0]))
        else:
            out.append(engine.update(rows[k], dt[k], agent.t[k], T, agent.q[k]))
        mids.append(engine.get_state()["last_mid"])
    return np.array(out), np.array(mids)


def test_feature_engine_reproduces_the_wrapper_on_a_synthetic_day_bitwise(synthetic_replay):
    agent = synthetic_replay["agent"]
    assert np.any(synthetic_replay["dt"][1:] == 0), "the synthetic day must contain zero-dt rows"
    assert np.unique(agent.q).size > 3, "inventory must move"
    observations, mids = _engine_observations(synthetic_replay, FeatureEngine())
    assert observations.dtype == np.float32
    mismatches = [k for k in range(len(observations)) if not np.array_equal(observations[k], agent.raw_obs[k])]
    assert len(observations) == 3_000
    assert mismatches == []
    assert np.array_equal(mids, agent.mid)  # float64, as the replay's S


def test_state_round_trip_through_json_continues_bitwise(synthetic_replay):
    agent, rows, dt, T = (synthetic_replay[key] for key in ("agent", "rows", "dt", "T"))
    engine = FeatureEngine()
    _engine_observations(synthetic_replay, engine, 0, 1_234)
    restored = FeatureEngine()
    restored.set_state(json.loads(json.dumps(engine.get_state())))
    assert restored.get_state() == engine.get_state()
    for k in range(1_234, len(rows)):
        expected = engine.update(rows[k], dt[k], agent.t[k], T, agent.q[k])
        assert np.array_equal(restored.update(rows[k], dt[k], agent.t[k], T, agent.q[k]), expected)
        assert np.array_equal(expected, agent.raw_obs[k])


def test_window_keeps_the_last_100_steps(synthetic_replay):
    engine = FeatureEngine()
    _engine_observations(synthetic_replay, engine, 0, 250)
    state = engine.get_state()
    assert len(state["dS"]) == len(state["dt"]) == 100
    assert state["dt"] == [max(value, 1e-10) for value in synthetic_replay["dt"][150:250]]


def test_rolling_features_are_zero_until_two_steps(synthetic_replay):
    rows, dt = synthetic_replay["rows"], synthetic_replay["dt"]
    engine = FeatureEngine()
    first = engine.reset(rows[0], 0.0, 100.0, 0)
    second = engine.update(rows[1], dt[1], 1.0, 100.0, 0)
    assert first[3] == first[4] == second[3] == second[4] == 0.0
    assert first[1] == np.float32(1.0) and first[2] == np.float32(0.0)


def test_refusals_leave_the_state_unchanged(synthetic_replay):
    rows = synthetic_replay["rows"]
    engine = FeatureEngine()
    with pytest.raises(RuntimeError):
        engine.update(rows[1], 0.5, 1.0, 100.0, 0)
    with pytest.raises(RuntimeError):
        engine.get_state()
    engine.reset(rows[0], 0.0, 100.0, 0)
    before = engine.get_state()
    incomplete = dict(rows[1])
    del incomplete["asks[4].amount"]
    bad_inputs = [
        (incomplete, 0.5, 1.0, 100.0, 0),
        ({**rows[1], "bids[0].price": float("nan")}, 0.5, 1.0, 100.0, 0),
        (rows[1], -0.5, 1.0, 100.0, 0),
        (rows[1], float("inf"), 1.0, 100.0, 0),
        (rows[1], 0.5, 1.0, 0.0, 0),
        (rows[1], 0.5, float("nan"), 100.0, 0),
        (rows[1], 0.5, 1.0, 100.0, float("nan")),
    ]
    for args in bad_inputs:
        with pytest.raises((KeyError, ValueError)):
            engine.update(*args)
        assert engine.get_state() == before


def test_set_state_refuses_another_configuration(synthetic_replay):
    engine = FeatureEngine()
    engine.reset(synthetic_replay["rows"][0], 0.0, 100.0, 0)
    state = engine.get_state()
    for key, value in (("window", 50), ("tick_size", 1e-4), ("q_max", 5), ("imbalance_depth", 3), ("version", 2)):
        with pytest.raises(ValueError):
            FeatureEngine().set_state({**state, key: value})
    with pytest.raises(ValueError):
        FeatureEngine().set_state({**state, "dS": [0.0] * 3, "dt": [1.0] * 2})
    with pytest.raises(ValueError):
        FeatureEngine().set_state({**state, "dS": [0.0] * 101, "dt": [1.0] * 101})
    with pytest.raises(ValueError):
        FeatureEngine().set_state({key: value for key, value in state.items() if key != "last_mid"})


@pytest.mark.parametrize("kwargs", [{"window": 1}, {"window": 2.5}, {"q_max": 0}, {"tick_size": 0.0}, {"imbalance_depth": 0}])
def test_constructor_refuses_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        FeatureEngine(**kwargs)
