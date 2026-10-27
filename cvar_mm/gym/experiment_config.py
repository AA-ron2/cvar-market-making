from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default: the repository's data/ folder (data/README.md); the DATA_DIR environment variable overrides it.
DATA_DIR = os.environ.get("DATA_DIR", (Path(__file__).resolve().parents[2] / "data").as_posix() + "/")


def find_repo_root(start: Path | None = None) -> Path:
    """
    Resolve the repository root by walking upward until ``cvar_mm/`` exists.

    This keeps notebook imports/path handling stable whether the current
    working directory is the repo root or ``notebooks/``.
    """
    candidates = [start or Path.cwd()]
    candidates.extend(candidates[0].parents)
    for candidate in candidates:
        if (candidate / "cvar_mm").exists() and (candidate / "notebooks").exists():
            return candidate.resolve()
    return Path(__file__).resolve().parents[2]


def _tuple(value: tuple[float, float] | list[float]) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _resolve_dir(repo_root: Path, value: str | Path) -> Path:
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = repo_root / directory
    return directory.resolve()


@dataclass(frozen=True)
class BMExperimentConfig:
    repo_root: Path = field(default_factory=find_repo_root)
    models_subdir: str = "models"
    results_subdir: str = "results"

    s0: float = 100.0
    terminal_time: float = 1.0
    sigma: float = 2.0
    n_steps: int = 200
    A: float = 140.0
    kappa: float = 1.5
    tick_size: float = 0.01
    q_max: int = 10
    n_train: int = 1000
    phi: float = 0.01

    as_gamma_range: tuple[float, float] = (0.01, 1.0)
    as_gamma_trials: int = 50
    as_gamma_num_trajectories: int = 50
    reward_scale_num_trajectories: int = 1000

    ppo_gamma_bm: float = 0.99
    ppo_n_epochs: int = 10
    ppo_learning_rate: float = 3e-4
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_total_timesteps_multiplier: int = 100

    evaluation_rollouts: int = 20
    evaluation_seed: int = 42
    fixed_clock_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(self, "as_gamma_range", _tuple(self.as_gamma_range))

    @property
    def models_dir(self) -> Path:
        return self.repo_root / self.models_subdir

    @property
    def results_dir(self) -> Path:
        return self.repo_root / self.results_subdir

    @property
    def tensorboard_dir(self) -> Path:
        return self.repo_root / "tb_logs"

    @property
    def ppo_batch_size(self) -> int:
        return self.n_train * self.n_steps // 4

    @property
    def ppo_total_timesteps(self) -> int:
        return self.n_train * self.n_steps * self.ppo_total_timesteps_multiplier

    def ensure_artifact_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, stem: str) -> Path:
        return self.models_dir / stem

    def result_path(self, filename: str) -> Path:
        return self.results_dir / filename

    def ppo_kwargs(self) -> dict[str, float | int]:
        return {
            "n_steps": self.n_steps,
            "batch_size": self.ppo_batch_size,
            "n_epochs": self.ppo_n_epochs,
            "learning_rate": self.ppo_learning_rate,
            "gamma": self.ppo_gamma_bm,
            "gae_lambda": self.ppo_gae_lambda,
            "clip_range": self.ppo_clip_range,
            "ent_coef": self.ppo_ent_coef,
        }


@dataclass(frozen=True)
class ReplayExperimentConfig:
    repo_root: Path = field(default_factory=find_repo_root)
    models_subdir: str = "models"
    results_subdir: str = "results"
    datasets_subdir: str | Path = field(
        default_factory=lambda: os.environ.get("DATA_DIR", DATA_DIR)
    )

    pair: str = "DOGEUSDT"
    replay_date: str = "2025-01-01"

    tick_size: float = 0.00001
    q_max: int = 10
    phi: float = 0.0  # D-10: B1-B3 reward is mark-to-market P&L
    kappa: float = 35_000.0
    A: float = 0.8
    alpha_dd: float = 1.0
    feature_window: int = 100

    # Primary B0 tuning and the formula-policy action mapping share this range.
    # Wider analytical ranges remain available by passing an explicit override
    # to ``tune_gamma`` in a separately labelled sensitivity analysis.
    as_gamma_range: tuple[float, float] = (0.01, 0.9)
    as_gamma_trials: int = 20
    as_gamma_num_trajectories: int = 50
    reward_scale_num_trajectories: int = 50

    ppo_gamma_replay: float = 0.999
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 512
    ppo_n_epochs: int = 10
    ppo_learning_rate: float = 3e-4
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_total_timesteps_multiplier: int = 50

    vecnorm_clip_obs: float = 10.0
    evaluation_rollouts: int = 20
    evaluation_seed: int = 42
    fixed_clock_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(self, "as_gamma_range", _tuple(self.as_gamma_range))
        if self.as_gamma_range[0] <= 0 or self.as_gamma_range[0] >= self.as_gamma_range[1]:
            raise ValueError("as_gamma_range must contain positive increasing bounds")

    @property
    def datasets_dir(self) -> Path:
        return _resolve_dir(self.repo_root, self.datasets_subdir)

    @property
    def models_dir(self) -> Path:
        return self.repo_root / self.models_subdir

    @property
    def results_dir(self) -> Path:
        return self.repo_root / self.results_subdir

    def ensure_artifact_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def data_path(self, replay_date: str | None = None) -> Path:
        date = replay_date or self.replay_date
        return self.datasets_dir / f"binance_book_snapshot_25_{date}_{self.pair}.csv"

    def available_data_files(self, max_days: int | None = None) -> list[Path]:
        files = sorted(self.datasets_dir.glob(f"binance_book_snapshot_25_*_{self.pair}.csv"))
        if max_days is not None:
            files = files[:max_days]
        return files

    @property
    def ppo_total_timesteps(self) -> int:
        return self.ppo_n_steps * self.ppo_total_timesteps_multiplier

    def model_path(self, stem: str) -> Path:
        return self.models_dir / stem

    def vecnorm_path(self, name: str) -> Path:
        suffix = ".pkl" if not name.endswith(".pkl") else ""
        return self.models_dir / f"{name}{suffix}"

    def result_path(self, filename: str) -> Path:
        return self.results_dir / filename

    def ppo_kwargs(self) -> dict[str, float | int]:
        return {
            "n_steps": self.ppo_n_steps,
            "batch_size": self.ppo_batch_size,
            "n_epochs": self.ppo_n_epochs,
            "learning_rate": self.ppo_learning_rate,
            "gamma": self.ppo_gamma_replay,
            "gae_lambda": self.ppo_gae_lambda,
            "clip_range": self.ppo_clip_range,
            "ent_coef": self.ppo_ent_coef,
        }
