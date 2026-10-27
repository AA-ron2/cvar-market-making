"""Fixtures for the data tests T1-T4. They skip unless HB_ARTIFACTS names the folder of pre-registered inputs
(PROTOCOL §2), and each input is checked against hummingbot_adapter/SHA256SUMS_inputs.txt before use."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hummingbot_adapter.export_policy import RECORD_SHA256, file_sha256
from hummingbot_adapter.features import row_columns
from hummingbot_adapter.policy import NumpyPolicy
from hummingbot_adapter.tests.support import REPO_ROOT, RecordingAgent

ARTIFACTS_ENV = "HB_ARTIFACTS"
INPUT_HASHES = REPO_ROOT / "hummingbot_adapter" / "SHA256SUMS_inputs.txt"
INPUTS = {
    "model": "ppo_b3_doge.zip",
    "vecnorm": "vecnorm_b3.pkl",
    "params": "formula_train_params.json",
    "b0_gamma": "b0_gamma_fixed.txt",
    "jan7": "binance_book_snapshot_25_2025-01-07_DOGEUSDT.csv",
}


@pytest.fixture(scope="session")
def hb_artifacts() -> Path:
    raw = os.environ.get(ARTIFACTS_ENV)
    if not raw:
        pytest.skip(f"{ARTIFACTS_ENV} is not set")
    path = Path(raw)
    if not path.is_dir():
        pytest.fail(f"{ARTIFACTS_ENV}={raw} is not a directory")
    return path


@pytest.fixture(scope="session")
def inputs(hb_artifacts) -> dict[str, Path]:
    """The pre-registered input files, each with the SHA-256 that SHA256SUMS_inputs.txt records."""

    recorded = {}
    for line in INPUT_HASHES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            recorded[name.strip()] = digest
    paths = {}
    for key, name in INPUTS.items():
        path = hb_artifacts / name
        assert file_sha256(path) == recorded[name], f"{name} differs from SHA256SUMS_inputs.txt"
        paths[key] = path
    return paths


@pytest.fixture(scope="session")
def b3_numpy_policy(hb_artifacts, inputs) -> NumpyPolicy:
    """The exported b3_policy.npz (step 3), whose sidecar names the pre-registered inputs and its own hash."""

    npz, sidecar_path = hb_artifacts / "b3_policy.npz", hb_artifacts / "b3_policy.json"
    if not npz.exists() or not sidecar_path.exists():
        pytest.fail("b3_policy.npz or its sidecar is missing: run python -m hummingbot_adapter.export_policy first")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["output"]["sha256"] == file_sha256(npz)
    assert sidecar["inputs"]["model"]["sha256"] == file_sha256(inputs["model"])
    assert sidecar["inputs"]["vecnorm"]["sha256"] == file_sha256(inputs["vecnorm"])
    assert sidecar["inputs"]["record"]["sha256"] == RECORD_SHA256
    return NumpyPolicy.load(npz)


@dataclasses.dataclass(frozen=True)
class Jan7Replay:
    """The frozen B3 replay of Jan 7 (fill seed 42, one path), recorded at every row k: the raw observation
    and the state (q, t, mid) at row k; for k < M - 1 also the normalized observation, the SB3 action and the
    wrapper's bid and ask for the step from k to k + 1."""

    settings: dict
    terminal_time: float
    S: np.ndarray
    dt: np.ndarray
    raw_obs: np.ndarray
    norm_obs: np.ndarray
    actions: np.ndarray
    q: np.ndarray
    t: np.ndarray
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    row_columns: tuple
    row_values: np.ndarray
    stats: dict


@pytest.fixture(scope="session")
def jan7_replay(inputs) -> Jan7Replay:
    """The replay as cvar_mm.gym.notebook_support.evaluate_formula_rl_per_day builds it, with one path."""

    from stable_baselines3 import PPO

    from cvar_mm.agents import Sb3Agent
    from cvar_mm.gym.data_loader import (
        load_single_day_with_features,
        read_market_data_csv,
        stable_order_by_exchange_timestamp,
    )
    from cvar_mm.gym.experiment_config import ReplayExperimentConfig
    from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
    from cvar_mm.gym.notebook_support import build_formula_replay_env, freeze_vecnorm
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from cvar_mm.rewards import PnLReward
    from scripts.pipeline.common import formula_kwargs_from_config

    train_params = json.loads(inputs["params"].read_text(encoding="utf-8"))
    config = dataclasses.replace(
        ReplayExperimentConfig(), A=float(train_params["A_train"]), kappa=float(train_params["kappa_train"])
    )
    formula_kwargs = formula_kwargs_from_config(train_params, config)
    settings = {
        "tick_size": config.tick_size,
        "q_max": config.q_max,
        "feature_window": config.feature_window,
        "gamma_range": (formula_kwargs["gamma_min"], formula_kwargs["gamma_max"]),
        "skew_ticks_max": formula_kwargs["skew_ticks_max"],
        "sigma": formula_kwargs["sigma"],
        "kappa": config.kappa,
        "A": config.A,
    }
    print(f"\nHB-PARITY replay settings: {settings}")
    assert config.tick_size == 1e-5 and config.q_max == 10 and config.feature_window == 100
    assert settings["gamma_range"] == (0.01, 0.9) and settings["skew_ticks_max"] == 5.0
    assert settings["sigma"] == train_params["sigma_train"] and settings["kappa"] == train_params["kappa_train"]

    S, dt, _, features = load_single_day_with_features(str(inputs["jan7"]), tick_size=config.tick_size)

    def build():
        return build_formula_replay_env(
            S, dt, config, market_features=features, reward_fn=PnLReward(), num_trajectories=1, **formula_kwargs
        )

    env_sim, env_vn = build(), build()
    eval_vn = freeze_vecnorm(str(inputs["vecnorm"]), StableBaselinesTradingEnvironment(env_vn), config, norm_reward=False)
    action_config = env_sim.action_config
    assert (action_config.sigma, action_config.kappa, action_config.tick_size) == (
        settings["sigma"], settings["kappa"], 1e-5
    )
    assert (action_config.gamma_min, action_config.gamma_max, action_config.skew_ticks_max) == (0.01, 0.9, 5.0)
    assert env_sim.rolling_window == 100 and env_sim.max_inventory == 10
    assert eval_vn.clip_obs == 10.0 and eval_vn.norm_obs and not eval_vn.training and not eval_vn.norm_reward

    model = PPO.load(str(inputs["model"]), device="cpu")
    agent = RecordingAgent(env_sim, len(S), Sb3Agent(model, vecnorm_env=eval_vn).get_action, eval_vn.normalize_obs)
    stats = generate_trajectory_stats(env_sim, agent, seed=42, fixed_clock_seconds=config.fixed_clock_seconds)
    agent.finish()

    columns = row_columns(5)
    frame = stable_order_by_exchange_timestamp(read_market_data_csv(str(inputs["jan7"])))
    row_values = frame[list(columns)].to_numpy(dtype=np.float64)
    del frame
    assert len(row_values) == len(S)
    return Jan7Replay(
        settings=settings,
        terminal_time=float(env_sim.terminal_time),
        S=S,
        dt=dt,
        raw_obs=agent.raw_obs,
        norm_obs=agent.norm_obs,
        actions=agent.actions,
        q=agent.q,
        t=agent.t,
        mid=agent.mid,
        bid=agent.bid,
        ask=agent.ask,
        row_columns=columns,
        row_values=row_values,
        stats={key: np.asarray(value).tolist() for key, value in stats.items()},
    )


@pytest.fixture(scope="session")
def numpy_actions(jan7_replay, b3_numpy_policy) -> np.ndarray:
    """NumpyPolicy.act on every recorded raw observation that SB3 acted on."""

    return np.array([b3_numpy_policy.act(obs) for obs in jan7_replay.raw_obs[:-1]])
