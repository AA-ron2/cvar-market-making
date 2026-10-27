"""D-17: the B1 rollouts behind B3's threshold d, on synthetic replay days.

The script-level tests (calibrate_cvar_b3.py, B3's first rollout, the profile
hashes) are in test_risk_aware_training.py, next to its synthetic pipeline.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cvar_mm.gym.experiment_config import ReplayExperimentConfig

TICK = 1e-5
FORMULA_KWARGS = {"sigma": 2.7e-5, "gamma_min": 0.01, "gamma_max": 0.9, "skew_ticks_max": 5.0}
SEEDS = (42, 43)
PATHS = 4


def synthetic_day(seed: int, n_snapshots: int = 40):
    rng = np.random.default_rng(seed)
    midprices = 0.35 + TICK * np.cumsum(np.r_[0, rng.integers(-1, 2, n_snapshots - 1)])
    dt = np.ones(n_snapshots)
    dt[0] = 0.0
    dt[7] = 0.0  # duplicate exchange timestamp: the mid may move, nothing fills
    features = {
        "spread_ticks": np.ones(n_snapshots, dtype=np.float32),
        "lob_imbalance": np.zeros(n_snapshots, dtype=np.float32),
    }
    return midprices, dt, features


@pytest.fixture(scope="module")
def b1(tmp_path_factory):
    """A B1 stand-in (sigma about 1) with its saved normalizer, on two 39-step days."""

    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env, make_vecnorm
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from cvar_mm.rewards import PnLReward

    root = tmp_path_factory.mktemp("b1")
    # A = 1: a quote fills with probability of order 0.3 per 1 s step, so inventory builds up, adverse mid
    # moves give drawdowns and the full-day MaxDD differs from path to path.
    cfg = replace(ReplayExperimentConfig(repo_root=root, datasets_subdir=root), A=1.0, kappa=12_000.0)
    days = [synthetic_day(seed) for seed in (20250101, 20250102)]
    env = build_formula_multi_day_replay_env(
        [day[0] for day in days],
        [day[1] for day in days],
        cfg,
        daily_market_features=[day[2] for day in days],
        reward_fn=PnLReward(),
        mode="sequential",
        **FORMULA_KWARGS,
    )
    vecnorm = make_vecnorm(StableBaselinesTradingEnvironment(env), cfg, training=True, norm_reward=True)
    model = PPO("MlpPolicy", vecnorm, n_steps=64, batch_size=32, n_epochs=2, seed=7, device="cpu")
    model.learn(total_timesteps=128)
    vecnorm_path = root / "vecnorm_b1.pkl"
    vecnorm.save(str(vecnorm_path))
    return SimpleNamespace(model=model, vecnorm_path=vecnorm_path, cfg=cfg, days=days)


def collect_kwargs(b1, *, days=None, fill_seeds=SEEDS, rollouts=PATHS) -> dict:
    days = b1.days if days is None else days
    return dict(
        daily_midprices=[day[0] for day in days],
        daily_dt_arrays=[day[1] for day in days],
        daily_market_features=[day[2] for day in days],
        model=b1.model,
        vecnorm_path=b1.vecnorm_path,
        config=b1.cfg,
        formula_kwargs=FORMULA_KWARGS,
        fill_seeds=fill_seeds,
        rollouts_per_seed=rollouts,
    )


def collect(b1, *, deterministic, **kwargs):
    from cvar_mm.gym.cvar_lagrangian import collect_b1_full_day_maxdrawdowns

    return collect_b1_full_day_maxdrawdowns(**collect_kwargs(b1, **kwargs), deterministic=deterministic)


def test_stochastic_maxdd_is_bitwise_reproducible_whatever_the_callers_rng_state(b1):
    import random

    import torch

    torch.manual_seed(0)
    first, first_records = collect(b1, deterministic=False)
    torch.manual_seed(12345)
    torch.randn(1000)
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    second, second_records = collect(b1, deterministic=False)

    assert torch.equal(torch.get_rng_state(), torch_state)  # the caller's CPU generator is restored (forked)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0] and np.array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:] and random.getstate() == python_state
    np.testing.assert_array_equal(first, second)
    assert first_records == second_records
    assert first.dtype == np.float64 and first.shape == (2 * len(SEEDS) * PATHS,)
    assert np.ptp(first) > 0  # the paths' drawdowns differ, so the equalities are informative


def test_another_fill_seed_changes_the_stochastic_maxdd(b1):
    base, _ = collect(b1, deterministic=False, fill_seeds=(42,))
    other, _ = collect(b1, deterministic=False, fill_seeds=(44,))
    assert not np.array_equal(base, other)


def test_each_rollout_depends_only_on_its_day_and_fill_seed(b1):
    full, _ = collect(b1, deterministic=False)
    alone, _ = collect(b1, deterministic=False, days=b1.days[1:], fill_seeds=(43,))
    blocks = full.reshape(len(b1.days), len(SEEDS), PATHS)  # day-major, then fill seed, then path
    np.testing.assert_array_equal(blocks[1, 1], alone)


def test_the_torch_seed_is_the_fill_seed_and_only_the_stochastic_pass_uses_it(b1, monkeypatch):
    """D-17: torch is seeded with the fill seed per (day, fill seed); the mean actions use no noise at all."""

    import torch

    original_seed = torch.manual_seed
    seeds: list[int] = []
    shift = {"value": 0}

    def recording_seed(seed):
        seeds.append(seed)
        return original_seed(seed + shift["value"])

    monkeypatch.setattr(torch, "manual_seed", recording_seed)
    stochastic, _ = collect(b1, deterministic=False)
    deterministic, _ = collect(b1, deterministic=True)
    assert seeds == [seed for _ in b1.days for seed in SEEDS] * 2  # one per rollout, the fill seed, both passes

    shift["value"] = 1000  # another torch state inside every rollout
    np.testing.assert_array_equal(collect(b1, deterministic=True)[0], deterministic)
    assert not np.array_equal(collect(b1, deterministic=False)[0], stochastic)
    assert not np.array_equal(deterministic, stochastic)  # the switch really samples B1's noise


def test_fill_seeds_draw_different_action_noise_from_the_same_start(b1, monkeypatch):
    """Every rollout starts from the same reset observation, so different first actions can only be noise."""

    from cvar_mm.gym import cvar_lagrangian

    original = cvar_lagrangian._BatchedPolicyAgent.get_action
    first_actions: list[np.ndarray] = []

    def recording(self, state):
        actions = original(self, state)
        first_actions.append(np.array(actions))
        return actions

    monkeypatch.setattr(cvar_lagrangian._BatchedPolicyAgent, "get_action", recording)
    collect(b1, deterministic=False, days=b1.days[:1])
    steps = len(b1.days[0][0]) - 1  # one call per step; the fill seeds' rollouts follow each other
    assert len(first_actions) == len(SEEDS) * steps
    assert not np.array_equal(first_actions[0], first_actions[steps])  # first step of fill seeds 42 and 43
    first_actions.clear()
    collect(b1, deterministic=True, days=b1.days[:1])
    np.testing.assert_array_equal(first_actions[0], first_actions[steps])  # no noise: identical first actions


def test_records_are_day_major_with_fill_seed_path_and_sampling_mode(b1):
    values, records = collect(b1, deterministic=False)

    assert [(record["day_index"], record["fill_seed"], record["path"]) for record in records] == [
        (day, seed, path) for day in range(len(b1.days)) for seed in SEEDS for path in range(PATHS)
    ]
    assert all(record["deterministic"] is False for record in records)
    assert all(set(record) == {"day_index", "fill_seed", "path", "deterministic", "max_drawdown"} for record in records)
    np.testing.assert_array_equal([record["max_drawdown"] for record in records], values)


def test_the_collector_refuses_a_missing_sampling_mode_and_bad_inputs(b1):
    import torch

    from cvar_mm.gym.cvar_lagrangian import collect_b1_full_day_maxdrawdowns

    parameter = inspect.signature(collect_b1_full_day_maxdrawdowns).parameters["deterministic"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    kwargs = collect_kwargs(b1)
    with pytest.raises(TypeError, match="deterministic"):
        collect_b1_full_day_maxdrawdowns(**kwargs)
    for value in (None, 0, 1, "False", np.bool_(False)):
        with pytest.raises(TypeError, match="D-17"):
            collect_b1_full_day_maxdrawdowns(**kwargs, deterministic=value)
    with pytest.raises(ValueError, match="unique"):
        collect(b1, deterministic=False, fill_seeds=(42, 42))
    with pytest.raises(ValueError, match="unique"):
        collect(b1, deterministic=False, fill_seeds=())
    with pytest.raises(ValueError, match="at least 1"):
        collect(b1, deterministic=False, rollouts=0)
    with pytest.raises(ValueError, match="same non-zero number"):
        collect(b1, deterministic=False, days=[])
    with pytest.raises(ValueError, match="same non-zero number"):
        collect_b1_full_day_maxdrawdowns(
            **{**kwargs, "daily_dt_arrays": kwargs["daily_dt_arrays"][:1]}, deterministic=False
        )
    with pytest.raises(ValueError, match="CPU"):
        collect_b1_full_day_maxdrawdowns(
            **{**kwargs, "model": SimpleNamespace(device=torch.device("cuda"))}, deterministic=False
        )


def test_the_batched_agent_draws_what_ppo_rollouts_draw(b1):
    """D-17: one predict per step equals PPO's collect_rollouts step under the same seed, bit for bit."""

    import torch

    from cvar_mm.gym.cvar_lagrangian import _BatchedPolicyAgent
    from cvar_mm.gym.notebook_support import build_formula_replay_env, freeze_vecnorm
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment

    prices, dt, features = b1.days[0]

    def env():
        return build_formula_replay_env(
            prices, dt, b1.cfg, market_features=features, num_trajectories=8, **FORMULA_KWARGS
        )

    normalizer = freeze_vecnorm(b1.vecnorm_path, StableBaselinesTradingEnvironment(env()), b1.cfg, norm_reward=False)
    raw, _ = env().reset()
    observations = torch.as_tensor(normalizer.normalize_obs(raw))
    low, high = b1.model.action_space.low, b1.model.action_space.high
    policy = b1.model.policy
    policy.set_training_mode(False)  # as collect_rollouts does

    torch.manual_seed(5)
    drawn = _BatchedPolicyAgent(b1.model, normalizer, deterministic=False).get_action(raw)
    torch.manual_seed(5)
    with torch.no_grad():
        sampled, _, _ = policy(observations)  # what collect_rollouts executes, before its clip
    np.testing.assert_array_equal(drawn, np.clip(sampled.numpy(), low, high))
    assert drawn.shape == (8, 2)

    with torch.no_grad():
        mean = policy.get_distribution(observations).mode()
    mean_actions = _BatchedPolicyAgent(b1.model, normalizer, deterministic=True).get_action(raw)
    np.testing.assert_array_equal(mean_actions, np.clip(mean.numpy(), low, high))
    assert not np.array_equal(drawn, mean_actions)
