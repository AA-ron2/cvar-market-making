"""K9 robustness (per-day terminal time, a seeded B0 tuner) and the D-15a/b/c warm start that replaced K9's critic scale."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.formula_as import compute_formula_as_depths
from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env

DAYS = [np.array([0.1, 0.10001, 0.10002]), np.array([0.1, 0.10001, 0.10002, 0.10001])]
DTS = [np.array([0.0, 1.0, 1.0]), np.array([0.0, 2.0, 2.0, 2.0])]


def test_terminal_time_follows_the_replayed_day(tmp_path):
    env = build_formula_multi_day_replay_env(
        DAYS, DTS, ReplayExperimentConfig(repo_root=tmp_path), sigma=2.5e-5, mode="sequential"
    )
    env.reset()
    assert (env.terminal_time, env.n_steps) == (2.0, 2)
    for _ in range(2):
        env.step(np.zeros((1, 2), dtype=np.float32))

    obs, _ = env.reset()  # sequential replay: the second, longer day
    assert (env.terminal_time, env.n_steps) == (6.0, 3)
    assert obs[0, 1] == 1.0  # time_remaining_frac at the start of the day
    env.step(np.zeros((1, 2), dtype=np.float32))
    _, expected = compute_formula_as_depths(
        midprice=[0.1], inventory=[0.0], time_elapsed=[0.0], terminal_time=6.0,
        action=np.zeros((1, 2)), config=env.action_config,
    )
    np.testing.assert_array_equal(env.last_formula_diagnostics["half_spread"], expected["half_spread"])


def test_tune_gamma_is_reproducible_for_a_seed():
    pytest.importorskip("optuna")
    from cvar_mm.gym.calibration import tune_gamma

    midprices = np.linspace(100.0, 101.0, 11)
    dt = np.full(11, 0.1)
    dt[0] = 0.0
    arguments = dict(
        midprices=midprices, dt_array=dt, sigma=2.0, kappa=1.5, A=140.0, tick_size=0.01, Q_MAX=10,
        gamma_range=(0.01, 0.2), n_trials=12, num_trajectories=2, seed=0, verbose=False,
    )
    first = [trial.params["gamma"] for trial in tune_gamma(**arguments)[1].trials]
    second = [trial.params["gamma"] for trial in tune_gamma(**arguments)[1].trials]
    assert first == second  # more trials than TPE's 10 random start-up trials


def test_warm_start_copies_b1_weights_with_fresh_adam_and_frozen_log_std(tmp_path, monkeypatch):
    """D-15a/D-15b/D-15c test (d): B1's weights exactly, B1's units, fresh Adam moments, log sigma frozen; and
    D-21 test (c): a source B1 whose sigma is not POLICY_STD is refused."""

    pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from cvar_mm.gym.notebook_support import freeze_vecnorm, make_vecnorm
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from cvar_mm.rewards import PnLReward
    from scripts.pipeline.common import fixed_std_ppo, warm_start_from

    monkeypatch.setenv("POLICY_STD", "0.3")  # D-21, as configs/final_pipeline.env exports it
    cfg = replace(ReplayExperimentConfig(repo_root=tmp_path), A=0.5, kappa=1e4)

    def vec_env():
        return StableBaselinesTradingEnvironment(build_formula_multi_day_replay_env(
            DAYS, DTS, cfg, sigma=1e-4, reward_fn=PnLReward(), mode="sequential"
        ))

    source_normalizer = make_vecnorm(vec_env(), cfg, training=True, norm_reward=True)
    source = fixed_std_ppo(source_normalizer, {"n_steps": 64, "batch_size": 32, "n_epochs": 1}, seed=7, device="cpu")
    source.learn(total_timesteps=128)
    source.save(str(tmp_path / "b1.zip"))
    source_normalizer.save(str(tmp_path / "b1.pkl"))
    assert source.policy.optimizer.state  # B1 has Adam moments that must not be carried over

    frozen = freeze_vecnorm(tmp_path / "b1.pkl", vec_env(), cfg, norm_reward=True)
    target = PPO("MlpPolicy", frozen, n_steps=64, batch_size=32, learning_rate=cfg.ppo_learning_rate,
                 seed=42, device="cpu")
    scale = warm_start_from(target, tmp_path / "b1.zip", frozen)

    assert scale == float(np.sqrt(source_normalizer.ret_rms.var + source_normalizer.epsilon)) != 1.0
    # Every weight of the B1 held in memory, value head included (no K9 rescale) ...
    assert target.policy.state_dict().keys() == source.policy.state_dict().keys()
    for key, value in source.policy.state_dict().items():
        assert torch.equal(target.policy.state_dict()[key], value), key
    # ... with a fresh optimizer (D-15c) and log sigma frozen (D-15b).
    assert len(target.policy.optimizer.state) == 0
    (group,) = target.policy.optimizer.param_groups
    assert (group["lr"], group["betas"], group["eps"]) == (cfg.ppo_learning_rate, (0.9, 0.999), 1e-5)
    assert target.policy.log_std.requires_grad is False
    assert all(parameter.requires_grad for name, parameter in target.policy.named_parameters() if name != "log_std")

    untrained = make_vecnorm(vec_env(), cfg, training=False, norm_reward=True)
    with pytest.raises(ValueError, match="never updated"):
        warm_start_from(PPO("MlpPolicy", untrained, n_steps=64, batch_size=32, device="cpu"),
                        tmp_path / "b1.zip", untrained)
    raw_rewards = freeze_vecnorm(tmp_path / "b1.pkl", vec_env(), cfg, norm_reward=False)
    still_training = VecNormalize.load(str(tmp_path / "b1.pkl"), vec_env())
    for normalizer in (raw_rewards, still_training):
        with pytest.raises(ValueError, match="norm_reward=True and training=False"):
            warm_start_from(PPO("MlpPolicy", normalizer, n_steps=64, batch_size=32, device="cpu"),
                            tmp_path / "b1.zip", normalizer)

    # exact_match=False loads with strict=False: a source whose keys differ must still be refused.
    deeper = PPO("MlpPolicy", source_normalizer, n_steps=64, batch_size=32, seed=7, device="cpu",
                 policy_kwargs={"net_arch": {"pi": [64, 64, 64], "vf": [64, 64]}, "log_std_init": np.log(0.3)})
    deeper.save(str(tmp_path / "deeper.zip"))
    with pytest.raises(ValueError, match="differs from the source B1"):
        warm_start_from(PPO("MlpPolicy", frozen, n_steps=64, batch_size=32, device="cpu"),
                        tmp_path / "deeper.zip", frozen)

    # D-21: a B1 that learned its sigma (SB3's default, log_std_init = 0) is refused before anything is copied.
    learned = PPO("MlpPolicy", source_normalizer, n_steps=64, batch_size=32, seed=7, device="cpu")
    learned.save(str(tmp_path / "learned_sigma.zip"))
    fresh = PPO("MlpPolicy", frozen, n_steps=64, batch_size=32, device="cpu")
    before = {key: value.clone() for key, value in fresh.policy.state_dict().items()}
    with pytest.raises(ValueError, match="D-21"):
        warm_start_from(fresh, tmp_path / "learned_sigma.zip", frozen)
    assert all(torch.equal(fresh.policy.state_dict()[key], value) for key, value in before.items())
    assert fresh.policy.log_std.requires_grad  # nothing was frozen either
