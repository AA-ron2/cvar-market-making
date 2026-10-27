"""Export B3's frozen actor and normalizer to numpy arrays for the Hummingbot container (PROTOCOL §2, §5.2).

    python -m hummingbot_adapter.export_policy --model <zip> --vecnorm <pkl> --record <profile_stage.sha256> --out <npz>

Refuses unless the record's SHA-256 is the pre-registered one (PROTOCOL §2) and the model and normalizer hash
to the record's lines for models/ppo_b3_doge.zip and models/vecnorm_b3.pkl. Nothing is unpickled before these
checks. Then it asserts the actor architecture that policy.NumpyPolicy implements and the VecNormalize settings
of evaluation, and writes the .npz with a sidecar JSON (same name, .json). It never overwrites either file, never
writes inside the repository, and removes what it wrote if it fails before both files are complete.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hummingbot_adapter.policy import ACTION_DIM, HIDDEN, NPZ_KEYS, OBS_DIM, NumpyPolicy
from hummingbot_adapter.provenance import REPO_ROOT, file_sha256, git_output, is_inside

RECORD_SHA256 = "f0b870b532e6727ee7b7728ab6e186460131a9b10aa7de8bf925f3f7edd629f8"  # PROTOCOL §2
RECORD_MODEL = "models/ppo_b3_doge.zip"
RECORD_VECNORM = "models/vecnorm_b3.pkl"
CLIP_OBS = 10.0
POLICY_STD = 0.3  # D-21: log sigma fixed at ln 0.3


class ExportRefused(RuntimeError):
    """An input fails a provenance or architecture check."""


def read_record(path: str | Path) -> dict[str, str]:
    """sha256sum lines ("<hex>  <path>" or "<hex> *<path>") as {path: hex}; refuses a malformed line."""

    entries: dict[str, str] = {}
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        digest = parts[0].lower()
        if len(parts) != 2 or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ExportRefused(f"{path}: line {number} is not a sha256sum line: {raw!r}")
        name = parts[1].strip().removeprefix("*")
        if name in entries:
            raise ExportRefused(f"{path}: {name} is listed twice")
        entries[name] = digest
    return entries


def check_inputs(model: Path, vecnorm: Path, record: Path) -> dict[str, str]:
    """The hash checks; returns the three SHA-256 values."""

    hashes = {"record": file_sha256(record), "model": file_sha256(model), "vecnorm": file_sha256(vecnorm)}
    if hashes["record"] != RECORD_SHA256:
        raise ExportRefused(
            f"{record} has SHA-256 {hashes['record']}, not the pre-registered {RECORD_SHA256} (PROTOCOL §2)"
        )
    entries = read_record(record)
    for key, name in (("model", RECORD_MODEL), ("vecnorm", RECORD_VECNORM)):
        if name not in entries:
            raise ExportRefused(f"{record} has no line for {name}")
        if hashes[key] != entries[name]:
            raise ExportRefused(
                f"{model if key == 'model' else vecnorm} has SHA-256 {hashes[key]}, but {record} records "
                f"{entries[name]} for {name}"
            )
    return hashes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportRefused(message)


def actor_arrays(model: Any) -> dict[str, np.ndarray]:
    """Assert the SB3 actor that NumpyPolicy implements; return its float32 parameters."""

    from gymnasium import spaces
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from torch import nn

    policy = model.policy
    observation_space, action_space = model.observation_space, model.action_space
    _require(
        isinstance(observation_space, spaces.Box) and observation_space.shape == (OBS_DIM,),
        f"observation space {observation_space}, expected shape ({OBS_DIM},)",
    )
    _require(
        isinstance(action_space, spaces.Box)
        and action_space.shape == (ACTION_DIM,)
        and np.all(action_space.low == -1.0)
        and np.all(action_space.high == 1.0),
        f"action space {action_space}, expected Box(-1, 1, ({ACTION_DIM},))",
    )
    _require(policy.squash_output is False, f"squash_output is {policy.squash_output}, expected False")
    _require(
        isinstance(policy.action_dist, DiagGaussianDistribution) and not policy.use_sde,
        f"action distribution {type(policy.action_dist).__name__} (use_sde {policy.use_sde}), "
        "expected a DiagGaussianDistribution without SDE",
    )
    _require(
        isinstance(policy.pi_features_extractor, FlattenExtractor),
        f"actor features extractor {type(policy.pi_features_extractor).__name__}, expected FlattenExtractor",
    )
    net = policy.mlp_extractor.policy_net
    expected = [(nn.Linear, OBS_DIM, HIDDEN), (nn.Tanh,), (nn.Linear, HIDDEN, HIDDEN), (nn.Tanh,)]
    _require(
        isinstance(net, nn.Sequential) and len(net) == len(expected),
        f"mlp_extractor.policy_net is {net}, expected Linear(7,64), Tanh, Linear(64,64), Tanh",
    )
    for layer, spec in zip(net, expected):
        _require(type(layer) is spec[0], f"policy_net layer {layer}, expected {spec[0].__name__}")
        if spec[0] is nn.Linear:
            _require(
                (layer.in_features, layer.out_features) == spec[1:] and layer.bias is not None,
                f"policy_net layer {layer}, expected Linear({spec[1]}, {spec[2]}) with bias",
            )
    head = policy.action_net
    _require(
        type(head) is nn.Linear
        and (head.in_features, head.out_features) == (HIDDEN, ACTION_DIM)
        and head.bias is not None,
        f"action_net is {head}, expected Linear(64, 2) with bias",
    )
    arrays = {
        "W1": net[0].weight, "b1": net[0].bias,
        "W2": net[2].weight, "b2": net[2].bias,
        "W3": head.weight, "b3": head.bias,
    }
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in arrays.items()}
    for key, value in arrays.items():
        _require(value.dtype == np.float32, f"{key} is {value.dtype}, expected float32")
    return arrays


def vecnorm_arrays(vecnorm: Any) -> dict[str, np.ndarray]:
    """The observation statistics that VecNormalize.normalize_obs uses; asserts norm_obs and clip_obs 10."""

    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import VecNormalize

    _require(isinstance(vecnorm, VecNormalize), f"normalizer is {type(vecnorm).__name__}, expected VecNormalize")
    _require(vecnorm.norm_obs is True, f"norm_obs is {vecnorm.norm_obs!r}, expected True")
    _require(float(vecnorm.clip_obs) == CLIP_OBS, f"clip_obs is {vecnorm.clip_obs!r}, expected {CLIP_OBS}")
    obs_rms = vecnorm.obs_rms
    _require(isinstance(obs_rms, RunningMeanStd), f"obs_rms is {type(obs_rms).__name__}, expected RunningMeanStd")
    arrays = {
        "obs_mean": np.asarray(obs_rms.mean, dtype=np.float64).copy(),
        "obs_var": np.asarray(obs_rms.var, dtype=np.float64).copy(),
        "epsilon": np.asarray(vecnorm.epsilon, dtype=np.float64),
        "clip_obs": np.asarray(vecnorm.clip_obs, dtype=np.float64),
    }
    for key in ("obs_mean", "obs_var"):
        _require(np.asarray(getattr(obs_rms, key.removeprefix("obs_"))).dtype == np.float64, f"obs_rms {key} is not float64")
        _require(arrays[key].shape == (OBS_DIM,), f"{key} has shape {arrays[key].shape}, expected ({OBS_DIM},)")
        _require(bool(np.all(np.isfinite(arrays[key]))), f"{key} is not finite")
    _require(bool(np.all(arrays["obs_var"] >= 0)), "obs_var is negative")
    _require(float(arrays["epsilon"]) > 0, f"epsilon is {vecnorm.epsilon!r}, expected > 0")
    return arrays


def _git(*args: str) -> str:
    result = git_output(REPO_ROOT, *args)
    if result.returncode != 0:
        raise ExportRefused(f"git {' '.join(args)} failed in {REPO_ROOT}: {result.stderr.strip()}")
    return result.stdout.rstrip("\n")


def export(model_path: Path, vecnorm_path: Path, record_path: Path, out: Path) -> dict[str, Any]:
    """Check, load, assert and write; returns the sidecar payload."""

    if out.suffix != ".npz":
        raise ExportRefused(f"--out must end in .npz, got {out}")
    if is_inside(out, REPO_ROOT):
        raise ExportRefused(f"{out} is inside the repository; the .npz goes to HB_ARTIFACTS (never committed)")
    sidecar = out.with_suffix(".json")
    for path in (out, sidecar):
        if path.exists():
            raise ExportRefused(f"{path} exists; move it aside first (never overwritten)")
    if not out.parent.is_dir():
        raise ExportRefused(f"the folder of --out does not exist: {out.parent}")
    hashes = check_inputs(model_path, vecnorm_path, record_path)
    git_commit = _git("rev-parse", "HEAD")
    git_status = _git("status", "--porcelain", "--untracked-files=all", "--", "hummingbot_adapter")

    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    arrays = actor_arrays(model)
    with open(vecnorm_path, "rb") as handle:
        vecnorm = pickle.load(handle)
    arrays.update(vecnorm_arrays(vecnorm))
    NumpyPolicy(arrays)  # the arrays load as the runtime will load them

    created: list[Path] = []
    try:
        with open(out, "xb") as handle:
            created.append(out)
            np.savez(handle, **{key: arrays[key] for key in NPZ_KEYS})
        payload = _sidecar_payload(
            model, vecnorm, arrays, hashes, git_commit, git_status, model_path, vecnorm_path, record_path, out
        )
        with open(sidecar, "x", encoding="utf-8", newline="\n") as handle:
            created.append(sidecar)
            handle.write(json.dumps(payload, indent=2) + "\n")
    except BaseException:
        for path in created:  # only files this run created (both are opened exclusively)
            path.unlink(missing_ok=True)
        raise
    return payload


def _sidecar_payload(
    model: Any,
    vecnorm: Any,
    arrays: dict[str, np.ndarray],
    hashes: dict[str, str],
    git_commit: str,
    git_status: str,
    model_path: Path,
    vecnorm_path: Path,
    record_path: Path,
    out: Path,
) -> dict[str, Any]:
    import stable_baselines3
    import torch

    log_std = model.policy.log_std.detach().cpu().numpy().copy()
    expected_log_std = np.float32(np.log(POLICY_STD))
    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "git_status_hummingbot_adapter": git_status,
        "inputs": {
            "model": {"path": str(model_path), "sha256": hashes["model"], "record_name": RECORD_MODEL},
            "vecnorm": {"path": str(vecnorm_path), "sha256": hashes["vecnorm"], "record_name": RECORD_VECNORM},
            "record": {"path": str(record_path), "sha256": hashes["record"]},
        },
        "output": {"path": str(out), "sha256": file_sha256(out), "keys": list(NPZ_KEYS)},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "architecture": {
            "observation_space": str(model.observation_space),
            "action_space": str(model.action_space),
            "squash_output": bool(model.policy.squash_output),
            "policy_net": str(model.policy.mlp_extractor.policy_net),
            "action_net": str(model.policy.action_net),
        },
        "log_std": log_std.tolist(),
        "policy_std": np.exp(log_std.astype(np.float64)).tolist(),
        "log_std_equals_float32_ln_0_3": bool(np.all(log_std == expected_log_std)),
        "vecnormalize": {
            "norm_obs": bool(vecnorm.norm_obs),
            "clip_obs": float(vecnorm.clip_obs),
            "epsilon": float(vecnorm.epsilon),
            "obs_mean": arrays["obs_mean"].tolist(),
            "obs_var": arrays["obs_var"].tolist(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vecnorm", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = export(args.model, args.vecnorm, args.record, args.out)
    except ExportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    inputs = payload["inputs"]
    print(f"record   {inputs['record']['sha256']}  (pre-registered, PROTOCOL §2)")
    print(f"model    {inputs['model']['sha256']}  = record line for {RECORD_MODEL}")
    print(f"vecnorm  {inputs['vecnorm']['sha256']}  = record line for {RECORD_VECNORM}")
    architecture = payload["architecture"]
    print(f"observation space {architecture['observation_space']}; action space {architecture['action_space']}; "
          f"squash_output {architecture['squash_output']}")
    print(f"policy_net {architecture['policy_net']}")
    print(f"action_net {architecture['action_net']}")
    print(f"log_std {payload['log_std']} (sigma {payload['policy_std']}); equals float32 ln 0.3: "
          f"{payload['log_std_equals_float32_ln_0_3']}")
    vecnormalize = payload["vecnormalize"]
    print(f"VecNormalize norm_obs {vecnormalize['norm_obs']}, clip_obs {vecnormalize['clip_obs']}, "
          f"epsilon {vecnormalize['epsilon']}")
    print(f"wrote {payload['output']['path']} (SHA-256 {payload['output']['sha256']}) and its .json sidecar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
