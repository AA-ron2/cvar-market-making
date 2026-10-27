"""B3's deterministic actor in numpy (PROTOCOL §5.2).

The arrays come from export_policy.py: the frozen VecNormalize observation statistics (clip 10, as
freeze_vecnorm uses them in evaluation) and the SB3 actor, Linear(7, 64)-Tanh-Linear(64, 64)-Tanh, then
action_net Linear(64, 2). The deterministic action is the Gaussian mean clipped to the action space [-1, 1],
as ActorCriticPolicy.predict(deterministic=True) returns it; the mean is computed in float64 from the float32
weights.

Runtime module: standard library and numpy only (it runs inside the Hummingbot container).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NPZ_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3", "obs_mean", "obs_var", "epsilon", "clip_obs")
OBS_DIM = 7
HIDDEN = 64
ACTION_DIM = 2
SHAPES = {
    "W1": (HIDDEN, OBS_DIM),
    "b1": (HIDDEN,),
    "W2": (HIDDEN, HIDDEN),
    "b2": (HIDDEN,),
    "W3": (ACTION_DIM, HIDDEN),
    "b3": (ACTION_DIM,),
    "obs_mean": (OBS_DIM,),
    "obs_var": (OBS_DIM,),
    "epsilon": (),
    "clip_obs": (),
}


class NumpyPolicy:
    """VecNormalize.normalize_obs, then the SB3 actor's mean, clipped to [-1, 1]."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        if set(arrays) != set(NPZ_KEYS):
            raise ValueError(f"policy arrays {sorted(arrays)}, expected {sorted(NPZ_KEYS)}")
        for key, shape in SHAPES.items():
            value = np.asarray(arrays[key])
            if value.shape != shape:
                raise ValueError(f"{key} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key} is not finite")
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            if np.asarray(arrays[key]).dtype != np.float32:
                raise ValueError(f"{key} must be float32 (the SB3 parameters), got {np.asarray(arrays[key]).dtype}")
        for key in ("obs_mean", "obs_var", "epsilon", "clip_obs"):
            if np.asarray(arrays[key]).dtype != np.float64:
                raise ValueError(f"{key} must be float64 (the VecNormalize statistics), got {np.asarray(arrays[key]).dtype}")
        if np.any(np.asarray(arrays["obs_var"]) < 0):
            raise ValueError("obs_var must be non-negative")
        if float(arrays["epsilon"]) <= 0 or float(arrays["clip_obs"]) <= 0:
            raise ValueError("epsilon and clip_obs must be positive")
        self.W1 = np.asarray(arrays["W1"], dtype=np.float64)
        self.b1 = np.asarray(arrays["b1"], dtype=np.float64)
        self.W2 = np.asarray(arrays["W2"], dtype=np.float64)
        self.b2 = np.asarray(arrays["b2"], dtype=np.float64)
        self.W3 = np.asarray(arrays["W3"], dtype=np.float64)
        self.b3 = np.asarray(arrays["b3"], dtype=np.float64)
        self.obs_mean = np.asarray(arrays["obs_mean"], dtype=np.float64)
        self.obs_var = np.asarray(arrays["obs_var"], dtype=np.float64)
        self.epsilon = float(arrays["epsilon"])
        self.clip_obs = float(arrays["clip_obs"])

    @classmethod
    def load(cls, path: str | Path) -> NumpyPolicy:
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) != set(NPZ_KEYS):
                raise ValueError(f"{path} holds {sorted(data.files)}, expected {sorted(NPZ_KEYS)}")
            arrays = {key: data[key] for key in NPZ_KEYS}
        return cls(arrays)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """clip((obs - mean) / sqrt(var + eps), -clip, clip) in float64, then float32, as
        VecNormalize.normalize_obs computes it from the float32 observation."""

        obs = np.asarray(obs)
        if obs.shape != (OBS_DIM,) or obs.dtype != np.float32:
            raise ValueError(f"expected a float32 observation of shape ({OBS_DIM},), got {obs.dtype} {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"observation is not finite: {obs}")
        normalized = np.clip(
            (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon), -self.clip_obs, self.clip_obs
        )
        return normalized.astype(np.float32)

    def act(self, raw_obs: np.ndarray) -> np.ndarray:
        """The deterministic action, float64 of shape (2,): clip(W3 tanh(W2 tanh(W1 x + b1) + b2) + b3, -1, 1)."""

        x = self.normalize(raw_obs).astype(np.float64)
        hidden = np.tanh(self.W1 @ x + self.b1)
        hidden = np.tanh(self.W2 @ hidden + self.b2)
        mean = self.W3 @ hidden + self.b3
        return np.clip(mean, -1.0, 1.0)
