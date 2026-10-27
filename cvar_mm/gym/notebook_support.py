from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.formula_as import FormulaASActionConfig, FormulaASActionWrapper
from cvar_mm.gym.helpers.fast_rollout import fast_simulate
from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
from cvar_mm.gym.features import FeatureComputer, RollingVolatility
from cvar_mm.gym.model_dynamics import LimitOrderModelDynamics
from cvar_mm.gym.trading_environment import TradingEnvironment
from cvar_mm.rewards import PnLReward
from cvar_mm.stochastic_processes import (
    DepthDependentPoissonFillFunction,
    MarketReplayMidpriceModel,
    MultiDayReplayMidpriceModel,
    PoissonArrivalModel,
)

SUMMARY_METRICS = (
    "Sharpe",
    "Sortino",
    "Max DD",
    "P&L-to-MAP",
    "Final PnL",
)


def build_replay_feature_computer(window: int = 100) -> FeatureComputer:
    return FeatureComputer([RollingVolatility(window)])


def build_replay_env(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    config: ReplayExperimentConfig,
    reward_fn=None,
    *,
    include_features: bool = True,
    manual_normalise: bool = False,
    reward_scale: float | None = None,
    num_trajectories: int = 1,
) -> TradingEnvironment:
    """
    Build a replay trading environment using one consistent notebook path.

    Replay notebooks default to raw observations + VecNormalize. Manual
    normalisation remains available only for controlled ablations.
    """
    if reward_fn is None:
        reward_fn = PnLReward()

    feature_computer = (
        build_replay_feature_computer(config.feature_window)
        if include_features else None
    )
    return TradingEnvironment(
        model_dynamics=LimitOrderModelDynamics(
            midprice_model=MarketReplayMidpriceModel(midprices, dt_array, num_trajectories),
            arrival_model=PoissonArrivalModel(
                np.array([config.A, config.A]),
                num_trajectories,
                use_linear_approximation=False,
            ),
            fill_probability_model=DepthDependentPoissonFillFunction(
                config.A, config.kappa, num_trajectories
            ),
            num_trajectories=num_trajectories,
        ),
        reward_function=reward_fn,
        max_inventory=config.q_max,
        normalise_observation_space=manual_normalise,
        normalise_action_space=manual_normalise,
        normalise_rewards=manual_normalise,
        reward_scale=reward_scale if manual_normalise else None,
        feature_computer=feature_computer,
    )


def build_multi_day_replay_env(
    daily_midprices: list[np.ndarray],
    daily_dt_arrays: list[np.ndarray],
    config: ReplayExperimentConfig,
    reward_fn=None,
    *,
    mode: str = "random",
    include_features: bool = True,
    manual_normalise: bool = False,
    reward_scale: float | None = None,
    num_trajectories: int = 1,
) -> TradingEnvironment:
    """
    Build a training environment that samples one complete replay day per reset.

    This avoids concatenating independent trading days into one artificial
    episode, which would carry inventory across day boundaries and introduce
    zero-time overnight price jumps.
    """
    if reward_fn is None:
        reward_fn = PnLReward()

    feature_computer = (
        build_replay_feature_computer(config.feature_window)
        if include_features else None
    )
    return TradingEnvironment(
        model_dynamics=LimitOrderModelDynamics(
            midprice_model=MultiDayReplayMidpriceModel(
                daily_midprices,
                daily_dt_arrays,
                num_trajectories,
                mode=mode,
            ),
            arrival_model=PoissonArrivalModel(
                np.array([config.A, config.A]),
                num_trajectories,
                use_linear_approximation=False,
            ),
            fill_probability_model=DepthDependentPoissonFillFunction(
                config.A, config.kappa, num_trajectories
            ),
            num_trajectories=num_trajectories,
        ),
        reward_function=reward_fn,
        max_inventory=config.q_max,
        normalise_observation_space=manual_normalise,
        normalise_action_space=manual_normalise,
        normalise_rewards=manual_normalise,
        reward_scale=reward_scale if manual_normalise else None,
        feature_computer=feature_computer,
    )


def build_formula_replay_env(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    config: ReplayExperimentConfig,
    *,
    sigma: float,
    market_features: dict[str, np.ndarray] | None = None,
    reward_fn=None,
    gamma_min: float = 0.01,
    gamma_max: float = 0.9,
    skew_ticks_max: float = 5.0,
    enabled_features: tuple[str, ...] | list[str] | None = None,
    num_trajectories: int = 1,
) -> FormulaASActionWrapper:
    """Build a one-day replay env where actions are A-S parameters."""
    base_env = build_replay_env(
        midprices,
        dt_array,
        config,
        reward_fn=reward_fn,
        include_features=False,
        manual_normalise=False,
        num_trajectories=num_trajectories,
    )
    return FormulaASActionWrapper(
        base_env,
        action_config=FormulaASActionConfig(
            sigma=sigma,
            kappa=config.kappa,
            tick_size=config.tick_size,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            skew_ticks_max=skew_ticks_max,
        ),
        market_features=market_features,
        rolling_window=config.feature_window,
        enabled_features=enabled_features,
    )


def build_formula_multi_day_replay_env(
    daily_midprices: list[np.ndarray],
    daily_dt_arrays: list[np.ndarray],
    config: ReplayExperimentConfig,
    *,
    sigma: float,
    daily_market_features: list[dict[str, np.ndarray]] | None = None,
    reward_fn=None,
    mode: str = "random",
    gamma_min: float = 0.01,
    gamma_max: float = 0.9,
    skew_ticks_max: float = 5.0,
    enabled_features: tuple[str, ...] | list[str] | None = None,
    num_trajectories: int = 1,
) -> FormulaASActionWrapper:
    """Build a multi-day replay env where PPO controls A-S parameters."""
    base_env = build_multi_day_replay_env(
        daily_midprices,
        daily_dt_arrays,
        config,
        reward_fn=reward_fn,
        mode=mode,
        include_features=False,
        manual_normalise=False,
        num_trajectories=num_trajectories,
    )
    return FormulaASActionWrapper(
        base_env,
        action_config=FormulaASActionConfig(
            sigma=sigma,
            kappa=config.kappa,
            tick_size=config.tick_size,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            skew_ticks_max=skew_ticks_max,
        ),
        daily_market_features=daily_market_features,
        rolling_window=config.feature_window,
        enabled_features=enabled_features,
    )


def make_vecnorm(sb3_env, config: ReplayExperimentConfig, *, training: bool = True, norm_reward: bool = True):
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except Exception as exc:  # pragma: no cover - depends on local SB3 install
        raise ImportError("stable-baselines3 is required for VecNormalize support.") from exc

    return VecNormalize(
        sb3_env,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=config.vecnorm_clip_obs,
        training=training,
    )


def freeze_vecnorm(source, eval_sb3_env, config: ReplayExperimentConfig, *, norm_reward: bool = False):
    """
    Freeze VecNormalize statistics for evaluation.

    ``source`` may be either a saved ``.pkl`` path or an in-memory training
    VecNormalize wrapper.
    """
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except Exception as exc:  # pragma: no cover - depends on local SB3 install
        raise ImportError("stable-baselines3 is required for VecNormalize support.") from exc

    if isinstance(source, (str, Path)):
        eval_vn = VecNormalize.load(str(source), eval_sb3_env)
    else:
        eval_vn = VecNormalize(
            eval_sb3_env,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=config.vecnorm_clip_obs,
            training=False,
        )
        eval_vn.obs_rms = copy.deepcopy(source.obs_rms)
        if hasattr(source, "ret_rms"):
            eval_vn.ret_rms = copy.deepcopy(source.ret_rms)

    eval_vn.training = False
    eval_vn.norm_reward = norm_reward
    return eval_vn


def stats_dict_to_frame(stats: dict[str, np.ndarray], *, seed: int | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "Sharpe": np.asarray(stats["sharpe"], dtype=float),
        "Sortino": np.asarray(stats["sortino"], dtype=float),
        "Max DD": np.asarray(stats["max_drawdown"], dtype=float),
        "P&L-to-MAP": np.asarray(stats["pnl_to_map"], dtype=float),
        "Final PnL": np.asarray(stats["total_pnl"], dtype=float),
        "Mean |q|": np.asarray(stats["mean_abs_q"], dtype=float),
        "Terminal q": np.asarray(stats["terminal_q"], dtype=float),
        "Mean spread": np.asarray(stats["mean_spread"], dtype=float),
    })
    if "snapshot_sharpe" in stats:
        frame["Sharpe Snapshot (supplementary)"] = np.asarray(
            stats["snapshot_sharpe"], dtype=float
        )
    if "snapshot_sortino" in stats:
        frame["Sortino Snapshot (supplementary)"] = np.asarray(
            stats["snapshot_sortino"], dtype=float
        )
    if "near_cap_fraction" in stats:
        frame["Near Cap Fraction"] = np.asarray(stats["near_cap_fraction"], dtype=float)
    if seed is not None:
        frame["Seed"] = seed
    return frame


def evaluate_agent_over_seeds(
    env_factory: Callable[[], TradingEnvironment],
    agent,
    *,
    seeds: list[int] | range,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        env = env_factory()
        frames.append(stats_dict_to_frame(generate_trajectory_stats(env, agent, seed=seed), seed=seed))
    return pd.concat(frames, ignore_index=True)


def evaluate_as_fast(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    *,
    gamma: float,
    sigma: float,
    kappa: float,
    A: float,
    terminal_time: float,
    tick_size: float,
    q_max: int,
    seeds: list[int] | range,
    use_linear_approximation: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        stats = fast_simulate(
            midprices=midprices,
            dt_array=dt_array,
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            A=A,
            terminal_time=terminal_time,
            tick_size=tick_size,
            Q_MAX=q_max,
            num_trajectories=1,
            seed=seed,
            use_linear_approximation=use_linear_approximation,
        )
        frames.append(stats_dict_to_frame(stats, seed=seed))
    return pd.concat(frames, ignore_index=True)


def summarise_agent_frames(agent_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for agent_name, frame in agent_frames.items():
        row: dict[str, float] = {"Samples": float(len(frame))}
        for metric in SUMMARY_METRICS:
            row[f"{metric} Mean"] = float(frame[metric].mean())
            row[f"{metric} Std"] = float(frame[metric].std(ddof=0))
            row[f"{metric} Median"] = float(frame[metric].median())
        if "Mean |q|" in frame.columns:
            row["Mean |q| Mean"] = float(frame["Mean |q|"].mean())
        if "Near Cap Fraction" in frame.columns:
            row["Near Cap Fraction Mean"] = float(frame["Near Cap Fraction"].mean())
        rows[agent_name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate_rl_per_day(
    model,
    vecnorm_path,
    test_S: list,
    test_dt: list,
    test_dates: list,
    config: ReplayExperimentConfig,
    *,
    reward_fn=None,
    seed: int = 42,
    num_rollouts: int | None = None,
) -> pd.DataFrame:
    """Evaluate a trained SB3 model independently on each test day.

    Returns a DataFrame indexed by date string with one row per test day.
    Two separate env instances are created per day: one for simulation
    and one as the VecNormalize observation-space reference, so that
    the running of the simulation env never interferes with the normaliser.
    """
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from cvar_mm.agents import Sb3Agent

    rollouts = num_rollouts or config.evaluation_rollouts
    rows = []
    for S, dt, date in zip(test_S, test_dt, test_dates):
        env_sim = build_replay_env(
            S,
            dt,
            config,
            reward_fn=reward_fn or PnLReward(),
            num_trajectories=rollouts,
        )
        env_vn = build_replay_env(
            S,
            dt,
            config,
            reward_fn=reward_fn or PnLReward(),
            num_trajectories=rollouts,
        )
        sb3_env = StableBaselinesTradingEnvironment(env_vn)
        eval_vn = freeze_vecnorm(vecnorm_path, sb3_env, config, norm_reward=False)
        agent = Sb3Agent(model, vecnorm_env=eval_vn)
        stats = generate_trajectory_stats(
            env_sim, agent, seed=seed, fixed_clock_seconds=config.fixed_clock_seconds
        )
        frame = stats_dict_to_frame(stats)
        row = frame.mean(numeric_only=True).to_dict()
        row["Day"] = str(date)
        row["Rollouts"] = float(len(frame))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Day")


def evaluate_formula_rl_per_day(
    model,
    vecnorm_path,
    test_S: list,
    test_dt: list,
    test_dates: list,
    config: ReplayExperimentConfig,
    *,
    sigma: float,
    test_market_features: list[dict[str, np.ndarray]] | None = None,
    reward_fn=None,
    seed: int = 42,
    num_rollouts: int | None = None,
    gamma_min: float = 0.01,
    gamma_max: float = 0.9,
    skew_ticks_max: float = 5.0,
    enabled_features: tuple[str, ...] | list[str] | None = None,
    return_episodes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a formula-AS parameter policy independently on each test day.

    With ``return_episodes`` it also returns one row per full-day path
    (Day, Rollout, metrics); the per-day means are unchanged.
    """
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from cvar_mm.agents import Sb3Agent

    rollouts = num_rollouts or config.evaluation_rollouts
    feature_list = test_market_features or [None] * len(test_S)
    rows = []
    episodes = []
    for S, dt, date, features in zip(test_S, test_dt, test_dates, feature_list):
        env_sim = build_formula_replay_env(
            S,
            dt,
            config,
            sigma=sigma,
            market_features=features,
            reward_fn=reward_fn or PnLReward(),
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            skew_ticks_max=skew_ticks_max,
            enabled_features=enabled_features,
            num_trajectories=rollouts,
        )
        env_vn = build_formula_replay_env(
            S,
            dt,
            config,
            sigma=sigma,
            market_features=features,
            reward_fn=reward_fn or PnLReward(),
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            skew_ticks_max=skew_ticks_max,
            enabled_features=enabled_features,
            num_trajectories=rollouts,
        )
        sb3_env = StableBaselinesTradingEnvironment(env_vn)
        eval_vn = freeze_vecnorm(vecnorm_path, sb3_env, config, norm_reward=False)
        agent = Sb3Agent(model, vecnorm_env=eval_vn)
        stats = generate_trajectory_stats(
            env_sim, agent, seed=seed, fixed_clock_seconds=config.fixed_clock_seconds
        )
        frame = stats_dict_to_frame(stats)
        row = frame.mean(numeric_only=True).to_dict()
        row["Day"] = str(date)
        row["Rollouts"] = float(len(frame))
        rows.append(row)
        episodes.append(frame.assign(Day=str(date), Rollout=np.arange(len(frame))))
    per_day = pd.DataFrame(rows).set_index("Day")
    if return_episodes:
        return per_day, pd.concat(episodes, ignore_index=True)
    return per_day


def run_qmax_sensitivity(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    *,
    gamma: float,
    sigma: float,
    kappa: float,
    A: float,
    terminal_time: float,
    tick_size: float,
    qmax_candidates: tuple[int, ...] = (10, 20, 50),
    num_trajectories: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for q_max in qmax_candidates:
        stats = fast_simulate(
            midprices=midprices,
            dt_array=dt_array,
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            A=A,
            terminal_time=terminal_time,
            tick_size=tick_size,
            Q_MAX=q_max,
            num_trajectories=num_trajectories,
            seed=seed,
            use_linear_approximation=False,
        )
        rows.append({
            "Q_MAX": q_max,
            "Sharpe Mean": float(np.mean(stats["sharpe"])),
            "Max DD Mean": float(np.mean(stats["max_drawdown"])),
            "Final PnL Mean": float(np.mean(stats["total_pnl"])),
            "P&L-to-MAP Mean": float(np.mean(stats["pnl_to_map"])),
            "Mean |q| Mean": float(np.mean(stats["mean_abs_q"])),
            "Near Cap Fraction Mean": float(np.mean(stats.get("near_cap_fraction", np.zeros(num_trajectories)))),
        })
    return pd.DataFrame(rows).sort_values("Q_MAX").reset_index(drop=True)
