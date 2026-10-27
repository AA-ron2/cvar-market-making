"""export_policy.py and NumpyPolicy on a synthetic PPO with B3's architecture (no data needed). The B3 parity
over Jan 7 is test_hb_policy_parity.py (T1)."""

from __future__ import annotations

import json
from datetime import datetime

import gymnasium
import numpy as np
import pytest
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hummingbot_adapter import export_policy
from hummingbot_adapter.policy import NPZ_KEYS, NumpyPolicy
from hummingbot_adapter.tests.support import REPO_ROOT


class _SpacesOnly(gymnasium.Env):
    def __init__(self, obs_dim=7, action_bound=1.0):
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(-action_bound, action_bound, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}


def _model_and_normalizer(
    net_arch=(64, 64), seed=0, obs_dim=7, action_bound=1.0, activation_fn=None, use_sde=False, norm_obs=True
):
    venv = DummyVecEnv([lambda: _SpacesOnly(obs_dim, action_bound)])
    policy_kwargs = {"net_arch": {"pi": list(net_arch), "vf": [64, 64]}, "log_std_init": float(np.log(0.3))}
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    model = PPO("MlpPolicy", venv, device="cpu", seed=seed, use_sde=use_sde, policy_kwargs=policy_kwargs)
    # Random weights at the trained B3's scale (rms 0.24, 0.21 and 0.07 for W1, W2, W3; biases 0.08), so that
    # tanh saturates and the action clip binds as in B3. Torch's float32 rounding grows with the weights: at
    # 3-4 times this scale it alone exceeds the 1e-6 parity bound, against 1.9e-7 for the real B3.
    generator = torch.Generator().manual_seed(seed)
    actor = model.policy
    with torch.no_grad():
        for parameter, scale in (
            (actor.mlp_extractor.policy_net[0].weight, 0.25), (actor.mlp_extractor.policy_net[0].bias, 0.08),
            (actor.mlp_extractor.policy_net[2].weight, 0.25), (actor.mlp_extractor.policy_net[2].bias, 0.08),
            (actor.action_net.weight, 0.15), (actor.action_net.bias, 0.08),
        ):
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)
        actor.log_std.fill_(float(np.log(0.3)))
    normalizer = VecNormalize(venv, norm_obs=norm_obs, norm_reward=False, clip_obs=10.0)
    if norm_obs:
        rng = np.random.default_rng(seed)
        normalizer.obs_rms.mean = rng.normal(0.0, 2.0, obs_dim)
        normalizer.obs_rms.var = rng.uniform(0.001, 5.0, obs_dim)
    normalizer.training = False
    return model, normalizer


def _observations(rng, count=4_000):
    scale = np.array([1.0, 0.5, 0.05, 5.0, 5.0, 3.0, 1.0])
    obs = rng.normal(0.0, 1.0, (count, 7)) * scale * rng.choice([1.0, 30.0], size=(count, 1), p=[0.9, 0.1])
    return obs.astype(np.float32)


def test_numpy_policy_matches_sb3_on_synthetic_b3_shaped_actor():
    model, normalizer = _model_and_normalizer()
    policy = NumpyPolicy({**export_policy.actor_arrays(model), **export_policy.vecnorm_arrays(normalizer)})
    rng = np.random.default_rng(1)
    clipped_normalized = clipped_actions = 0
    worst = 0.0
    for obs in _observations(rng):
        expected_normalized = normalizer.normalize_obs(obs.reshape(1, -1))[0]
        assert np.array_equal(policy.normalize(obs), expected_normalized)
        expected, _ = model.predict(expected_normalized, deterministic=True)
        worst = max(worst, float(np.max(np.abs(policy.act(obs) - expected))))
        clipped_normalized += bool(np.any(np.abs(expected_normalized) == 10.0))
        clipped_actions += bool(np.any(np.abs(expected) == 1.0))
    assert worst <= 1e-6
    assert clipped_normalized > 0 and clipped_actions > 0


def _write_inputs(tmp_path, clip_obs=10.0, **model_kwargs):
    model, normalizer = _model_and_normalizer(**model_kwargs)
    normalizer.clip_obs = clip_obs
    model_path, vecnorm_path = tmp_path / "ppo_b3_doge.zip", tmp_path / "vecnorm_b3.pkl"
    model.save(model_path)
    normalizer.save(vecnorm_path)
    record = tmp_path / "profile_stage.sha256"
    record.write_text(
        f"{'0' * 64}  models/ppo_b1_doge.zip\n"
        f"{export_policy.file_sha256(model_path)}  models/ppo_b3_doge.zip\n"
        f"{export_policy.file_sha256(vecnorm_path)}  models/vecnorm_b3.pkl\n",
        encoding="utf-8", newline="\n",
    )
    return model, normalizer, model_path, vecnorm_path, record


def _args(model_path, vecnorm_path, record, out):
    return ["--model", str(model_path), "--vecnorm", str(vecnorm_path), "--record", str(record), "--out", str(out)]


def test_export_writes_npz_and_sidecar_that_reproduce_sb3(tmp_path, monkeypatch, capsys):
    model, normalizer, model_path, vecnorm_path, record = _write_inputs(tmp_path)
    monkeypatch.setattr(export_policy, "RECORD_SHA256", export_policy.file_sha256(record))
    out = tmp_path / "out" / "b3_policy.npz"
    out.parent.mkdir()
    assert export_policy.main(_args(model_path, vecnorm_path, record, out)) == 0
    assert "equals float32 ln 0.3: True" in capsys.readouterr().out
    with np.load(out, allow_pickle=False) as data:
        assert list(data.files) == list(NPZ_KEYS)
    sidecar_bytes = out.with_suffix(".json").read_bytes()
    assert b"\r" not in sidecar_bytes
    sidecar = json.loads(sidecar_bytes)
    assert sidecar["output"]["sha256"] == export_policy.file_sha256(out)
    assert sidecar["inputs"]["model"]["sha256"] == export_policy.file_sha256(model_path)
    assert sidecar["inputs"]["vecnorm"]["sha256"] == export_policy.file_sha256(vecnorm_path)
    assert sidecar["inputs"]["record"]["sha256"] == export_policy.file_sha256(record)
    assert sidecar["versions"]["torch"] == torch.__version__
    assert sidecar["versions"]["stable_baselines3"] == stable_baselines3.__version__
    datetime.strptime(sidecar["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
    assert sidecar["log_std_equals_float32_ln_0_3"] is True
    assert sidecar["vecnormalize"]["clip_obs"] == 10.0 and len(sidecar["git_commit"]) == 40
    loaded = NumpyPolicy.load(out)
    for obs in _observations(np.random.default_rng(2), 500):
        normalized = normalizer.normalize_obs(obs.reshape(1, -1))[0]
        expected, _ = model.predict(normalized, deterministic=True)
        assert np.max(np.abs(loaded.act(obs) - expected)) <= 1e-6
    assert export_policy.main(_args(model_path, vecnorm_path, record, out)) == 2  # never overwritten
    assert "exists" in capsys.readouterr().err


def test_export_refuses_wrong_hashes_before_loading(tmp_path, monkeypatch, capsys):
    _, _, model_path, vecnorm_path, record = _write_inputs(tmp_path)
    out = tmp_path / "b3_policy.npz"
    assert export_policy.main(_args(model_path, vecnorm_path, record, out)) == 2  # not the pre-registered record
    assert "pre-registered" in capsys.readouterr().err

    loads = []
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: loads.append(args))
    tampered = tmp_path / "tampered.zip"
    tampered.write_bytes(model_path.read_bytes() + b"\0")
    monkeypatch.setattr(export_policy, "RECORD_SHA256", export_policy.file_sha256(record))
    assert export_policy.main(_args(tampered, vecnorm_path, record, out)) == 2
    assert "records" in capsys.readouterr().err
    assert export_policy.main(_args(model_path, model_path, record, out)) == 2  # the zip as the normalizer
    without_line = tmp_path / "without_line.sha256"
    without_line.write_text(record.read_text().replace("models/vecnorm_b3.pkl", "models/vecnorm_b2.pkl"))
    monkeypatch.setattr(export_policy, "RECORD_SHA256", export_policy.file_sha256(without_line))
    assert export_policy.main(_args(model_path, vecnorm_path, without_line, out)) == 2
    assert "no line" in capsys.readouterr().err
    assert loads == [] and not out.exists()


def test_export_refuses_another_architecture_or_clip(tmp_path, monkeypatch, capsys):
    for case, kwargs, message in (
        ("arch", {"net_arch": (32, 32)}, "policy_net"),
        ("deep", {"net_arch": (64, 64, 64)}, "policy_net"),
        ("relu", {"activation_fn": torch.nn.ReLU}, "policy_net"),
        ("sde", {"use_sde": True}, "action distribution"),
        ("obs8", {"obs_dim": 8}, "observation space"),
        ("bounds", {"action_bound": 2.0}, "action space"),
        ("clip", {"clip_obs": 5.0}, "clip_obs"),
        ("raw_obs", {"norm_obs": False}, "norm_obs"),
    ):
        directory = tmp_path / case
        directory.mkdir()
        _, _, model_path, vecnorm_path, record = _write_inputs(directory, **kwargs)
        monkeypatch.setattr(export_policy, "RECORD_SHA256", export_policy.file_sha256(record))
        out = directory / "b3_policy.npz"
        assert export_policy.main(_args(model_path, vecnorm_path, record, out)) == 2
        assert message in capsys.readouterr().err
        assert not out.exists()


def test_export_refuses_an_output_in_the_repository_or_a_missing_folder(tmp_path, monkeypatch, capsys):
    _, _, model_path, vecnorm_path, record = _write_inputs(tmp_path)
    monkeypatch.setattr(export_policy, "RECORD_SHA256", export_policy.file_sha256(record))
    for out, message in (
        (REPO_ROOT / "hummingbot_adapter" / "b3_policy.npz", "inside the repository"),
        (REPO_ROOT / "b3_policy.npz", "inside the repository"),
        (tmp_path / "missing" / "b3_policy.npz", "does not exist"),
    ):
        assert export_policy.main(_args(model_path, vecnorm_path, record, out)) == 2
        assert message in capsys.readouterr().err
        assert not out.exists() and not out.with_suffix(".json").exists()


def test_a_failed_export_removes_what_it_wrote(tmp_path, monkeypatch):
    _, _, model_path, vecnorm_path, record = _write_inputs(tmp_path)
    monkeypatch.setattr(export_policy, "RECORD_SHA256", export_policy.file_sha256(record))
    out = tmp_path / "b3_policy.npz"

    def half_written(handle, **arrays):
        handle.write(b"PK\x03\x04partial")
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(export_policy.np, "savez", half_written)
        with pytest.raises(OSError):
            export_policy.main(_args(model_path, vecnorm_path, record, out))
    assert not out.exists() and not out.with_suffix(".json").exists()

    def no_sidecar(*args, **kwargs):
        raise RuntimeError("sidecar failed")

    with monkeypatch.context() as patch:
        patch.setattr(export_policy, "_sidecar_payload", no_sidecar)
        with pytest.raises(RuntimeError):
            export_policy.main(_args(model_path, vecnorm_path, record, out))
    assert not out.exists() and not out.with_suffix(".json").exists()
    assert export_policy.main(_args(model_path, vecnorm_path, record, out)) == 0  # nothing left to refuse


def test_numpy_policy_refuses_malformed_arrays_and_observations(tmp_path):
    model, normalizer = _model_and_normalizer()
    arrays = {**export_policy.actor_arrays(model), **export_policy.vecnorm_arrays(normalizer)}
    policy = NumpyPolicy(arrays)
    with pytest.raises(ValueError):
        policy.act(np.zeros(7, dtype=np.float64))
    with pytest.raises(ValueError):
        policy.act(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError):
        policy.act(np.full(7, np.nan, dtype=np.float32))
    for key, value in (("W1", arrays["W1"].astype(np.float64)), ("obs_var", -arrays["obs_var"]),
                       ("b3", np.zeros(3, dtype=np.float32)), ("obs_mean", arrays["obs_mean"].astype(np.float32))):
        with pytest.raises(ValueError):
            NumpyPolicy({**arrays, key: value})
    with pytest.raises(ValueError):
        NumpyPolicy({key: value for key, value in arrays.items() if key != "epsilon"})
    path = tmp_path / "extra.npz"
    np.savez(path, **arrays, extra=np.zeros(1))
    with pytest.raises(ValueError):
        NumpyPolicy.load(path)
