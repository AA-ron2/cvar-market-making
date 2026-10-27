"""Full-day Rockafellar--Uryasev CVaR constraint for PPO.

The constrained cost is the trajectory functional

``Z = max_t(max_{u<=t} W_u - W_t)``,

not a 2048-step window or a sum of snapshot drawdowns.  For tail mass
``rho=0.20`` the auxiliary objective is

``H_rho(theta, nu) = nu + E[(Z-nu)_+] / rho``.

PPO maximizes economic return minus ``lambda * (H_rho-d)``.  The risk score
coefficient is inserted into PPO advantages before optimization; rewards are
never edited after return/advantage computation.

Per batch (DECISIONS D-12): nu is the batch's empirical (1 - rho)-quantile of
Z, the exact inner minimizer of H; lambda takes the normalized projected dual
step ``clip(lambda + eta (H - d) / d, 0, lambda_max)`` after the batch; and the
score term carries the per-day leave-one-out baseline shared with B2. The score
term enters the batch-standardized advantages in B1's units (D-15a), through
the arithmetic B2 uses.

The threshold d is 0.8 x the empirical CVaR_0.20 of B1's Z on the training days
from stochastic rollouts, the sampling that produces H (D-17).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cvar_mm.gym.drawdown_objectives import (
    CompleteDayRiskCallback,
    CompleteTrajectorySegment,
    RiskAdjustedAdvantages,
    standardized_risk_advantages,
)


B3_TAIL_MASS = 0.20
B3_THRESHOLD_MULTIPLIER = 0.80
B3_LAMBDA_MAX = 500.0
B3_THRESHOLD_ESTIMATOR = "stochastic (D-17)"
B3_POLICY_SAMPLING_SEED_RULE = (
    "torch.manual_seed(fill seed) before each (day, fill seed) rollout, in a forked CPU RNG state (D-17)"
)


@dataclass(frozen=True)
class RUCVaREstimate:
    nu: float
    value: float
    excesses: np.ndarray


def rockafellar_uryasev_value(costs: np.ndarray, *, nu: float, rho: float = B3_TAIL_MASS) -> float:
    """Evaluate ``nu + mean((Z-nu)+)/rho`` on complete trajectory costs."""

    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("costs must be a non-empty finite one-dimensional array")
    if not 0 < rho < 1:
        raise ValueError("rho must lie strictly between zero and one")
    return float(nu + np.mean(np.maximum(values - nu, 0.0)) / rho)


def empirical_ru_cvar(costs: np.ndarray, *, rho: float = B3_TAIL_MASS) -> RUCVaREstimate:
    """Return an empirical RU minimizer and its upper-tail CVaR value."""

    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("costs must be a non-empty one-dimensional array")
    try:
        nu = float(np.quantile(values, 1.0 - rho, method="higher"))
    except TypeError:  # NumPy < 1.22
        nu = float(np.quantile(values, 1.0 - rho, interpolation="higher"))
    excesses = np.maximum(values - nu, 0.0)
    return RUCVaREstimate(
        nu=nu,
        value=rockafellar_uryasev_value(values, nu=nu, rho=rho),
        excesses=excesses,
    )


def add_ru_cvar_score_advantage(
    advantages: np.ndarray,
    segments: Sequence[CompleteTrajectorySegment],
    *,
    nu: float,
    lambda_value: float,
    rho: float = B3_TAIL_MASS,
    reward_scale: float,
) -> RiskAdjustedAdvantages:
    """B3's score term with ``c = (Z - nu)+ / rho`` and weight lambda, in B1's units (D-12, D-15a).

    Every action of path i receives ``-lambda (c_i - mean of c_j over the other
    paths of the same day) / (s sigma_A)`` on top of its standardized advantage.

    Uses the same arithmetic as B2's score term, so rho = 1, nu = 0,
    lambda = alpha reproduces B2 exactly (D-11).
    """

    coefficients = np.asarray(
        [max(segment.max_drawdown - nu, 0.0) / rho for segment in segments],
        dtype=np.float64,
    )
    return standardized_risk_advantages(
        advantages, segments, coefficients, weight=lambda_value, reward_scale=reward_scale
    )


class FullDayRUCVaRCallback(CompleteDayRiskCallback):
    """Complete-day PPO/CVaR optimizer: batch-quantile ``nu``, normalized dual ``lambda`` (D-12)."""

    label = "B3"

    def __init__(
        self,
        *,
        dd_threshold: float,
        expected_day_indices: tuple[int, ...],
        reward_scale: float,
        rho: float = B3_TAIL_MASS,
        lambda_learning_rate: float,
        lambda_init: float = 0.0,
        lambda_max: float = B3_LAMBDA_MAX,
        verbose: int = 1,
    ):
        super().__init__(expected_day_indices=expected_day_indices, reward_scale=reward_scale, verbose=verbose)
        if rho != B3_TAIL_MASS:
            raise ValueError(f"B3 tail mass is frozen at {B3_TAIL_MASS}")
        if dd_threshold <= 0 or lambda_learning_rate <= 0:
            raise ValueError("threshold and dual learning rate must be positive")
        self.dd_threshold = float(dd_threshold)
        self.rho = float(rho)
        self.lambda_learning_rate = float(lambda_learning_rate)
        self.lambda_ = float(lambda_init)
        self.lambda_max = float(lambda_max)
        self.nu_history: list[float] = []
        self.lambda_used_history: list[float] = []
        self.lambda_history: list[float] = []
        self.cvar_history: list[float] = []
        self.violation_history: list[float] = []
        self.mean_maxdd_history: list[float] = []
        self.segment_history: list[list[CompleteTrajectorySegment]] = []
        # elapsed_history (the base class) ends each update after its PPO.train(), so it gives the wall time per update.

    def _adjust_batch(
        self, advantages: np.ndarray, segments: list[CompleteTrajectorySegment]
    ) -> RiskAdjustedAdvantages:
        costs = np.asarray([segment.max_drawdown for segment in segments])
        # D-12(a): nu is the batch (1 - rho)-quantile, the exact minimizer of H.
        estimate = empirical_ru_cvar(costs, rho=self.rho)
        lambda_used = self.lambda_
        violation = estimate.value - self.dd_threshold
        result = add_ru_cvar_score_advantage(
            advantages,
            segments,
            nu=estimate.nu,
            lambda_value=lambda_used,
            rho=self.rho,
            reward_scale=self.reward_scale,
        )

        # D-12(b): normalized projected dual ascent; the new lambda acts from the next batch.
        self.lambda_ = float(np.clip(
            lambda_used + self.lambda_learning_rate * violation / self.dd_threshold,
            0.0,
            self.lambda_max,
        ))

        self.nu_history.append(estimate.nu)
        self.lambda_used_history.append(lambda_used)
        self.lambda_history.append(self.lambda_)
        self.cvar_history.append(estimate.value)
        self.violation_history.append(violation)
        self.mean_maxdd_history.append(float(np.mean(costs)))
        self.segment_history.append(list(segments))
        self.logger.record("risk/ru_h", estimate.value)
        self.logger.record("risk/nu", estimate.nu)
        self.logger.record("risk/lambda", self.lambda_)
        self.logger.record("risk/violation", violation)
        if self.verbose:
            print(
                f"RU-CVaR={estimate.value:.6f} d={self.dd_threshold:.6f} "
                f"nu={estimate.nu:.6f} lambda={lambda_used:.6f}->{self.lambda_:.6f}"
            )
        return result


def training_feasibility(
    costs_by_update: Sequence[np.ndarray],
    threshold: float,
    *,
    rho: float = B3_TAIL_MASS,
) -> dict:
    """B3 training diagnostic (K6): pooled CVaR_rho of Z over the last K rollouts vs d.

    Rollout j samples the policy before update j, so rollout 1 is the
    warm-start B1 itself (lambda_0 = 0). The window is the last
    K = min(5, k - 1) rollouts, i.e. policies after at least one B3 update; the
    final policy is not rolled out. Training actions and d both come from
    stochastic rollouts with B1's sigma (D-17), so they compare like with like.
    """

    updates = len(costs_by_update)
    if updates < 2:
        raise ValueError("B3 feasibility needs at least 2 updates: rollout 1 is the warm-start B1")
    window = min(5, updates - 1)
    recent = [np.asarray(costs, dtype=np.float64) for costs in costs_by_update[-window:]]
    pooled = empirical_ru_cvar(np.concatenate(recent), rho=rho)
    return {
        "rollouts": list(range(updates - window + 1, updates + 1)),
        "paths_pooled": int(sum(costs.size for costs in recent)),
        "pooled_cvar": pooled.value,
        "mean_batch_cvar": float(np.mean([empirical_ru_cvar(costs, rho=rho).value for costs in recent])),
        "threshold_d": float(threshold),
        "feasible": bool(pooled.value <= threshold),
        "scope": "training diagnostic: stochastic training rollouts; d from stochastic B1 rollouts (D-17)",
    }


class _BatchedPolicyAgent:
    """B1's actions for all fill paths of a step from one ``predict`` call (D-17).

    PPO's ``collect_rollouts`` passes the observations of all paths of a step
    through the policy at once, which draws their Gaussian noise in one call;
    one call here draws it with the same torch routine, so for the same number
    of paths and the same torch state the stochastic actions are bitwise PPO's.
    ``Sb3Agent`` (one call per path, used by evaluation) draws the same law
    with another routine. Observations go through B1's frozen normalizer;
    ``predict`` clips actions to the action box, as PPO does before
    ``env.step``. It has the ``cvar_mm.agents.agent.Agent`` interface.
    """

    def __init__(self, model, vecnorm, *, deterministic: bool):
        self.model = model
        self.vecnorm = vecnorm
        self.deterministic = deterministic

    def get_action(self, state: np.ndarray) -> np.ndarray:
        actions, _ = self.model.predict(self.vecnorm.normalize_obs(state), deterministic=self.deterministic)
        return actions


def _b1_rollout_maxdrawdowns(
    prices: np.ndarray,
    dt: np.ndarray,
    features: dict[str, np.ndarray] | None,
    *,
    model,
    vecnorm_path,
    config,
    formula_kwargs: dict,
    fill_seed: int,
    rollouts: int,
    deterministic: bool,
) -> np.ndarray:
    """Full-day MaxDD of ``rollouts`` fill paths of B1 on one day: a function of (day, fill seed) alone."""

    import torch

    from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
    from cvar_mm.gym.notebook_support import build_formula_replay_env, freeze_vecnorm
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from cvar_mm.rewards import PnLReward

    env = build_formula_replay_env(
        prices, dt, config, market_features=features, reward_fn=PnLReward(),
        num_trajectories=rollouts, **formula_kwargs,
    )
    normalizer_env = build_formula_replay_env(
        prices, dt, config, market_features=features, reward_fn=PnLReward(),
        num_trajectories=rollouts, **formula_kwargs,
    )
    frozen = freeze_vecnorm(
        vecnorm_path,
        StableBaselinesTradingEnvironment(normalizer_env),
        config,
        norm_reward=False,
    )
    agent = _BatchedPolicyAgent(model, frozen, deterministic=deterministic)
    # D-17: the action noise comes from torch's CPU generator, seeded with the fill seed in a forked
    # state, as env.seed(fill seed) seeds the fill draws; the caller's CPU generator state is restored.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fill_seed)
        stats = generate_trajectory_stats(env, agent, seed=fill_seed)
    drawdowns = np.asarray(stats["max_drawdown"], dtype=np.float64)
    if drawdowns.shape != (rollouts,) or not np.all(np.isfinite(drawdowns)) or np.any(drawdowns < 0):
        raise RuntimeError(f"Invalid B1 full-day MaxDD for fill seed {fill_seed}: {drawdowns}")
    return drawdowns


def collect_b1_full_day_maxdrawdowns(
    *,
    daily_midprices: list[np.ndarray],
    daily_dt_arrays: list[np.ndarray],
    daily_market_features: list[dict[str, np.ndarray] | None],
    model,
    vecnorm_path,
    config,
    formula_kwargs: dict,
    fill_seeds: Sequence[int],
    rollouts_per_seed: int,
    deterministic: bool,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    """Full-day MaxDD of B1 with its saved normalizer on every training day (D-17).

    ``deterministic=False`` samples B1's actions with its own sigma through
    PPO's own sampling routine, so on each day these Z have the law of the Z
    that B3's H is measured on at warm start; d comes from them.
    ``deterministic=True`` uses the mean action, the evaluation policy (F-17);
    its CVaR is recorded, not used. Each (day, fill seed) rollout runs
    ``rollouts_per_seed`` fill paths with torch seeded by the fill seed, so it
    depends on (day, fill seed) alone and the action noise repeats across days,
    as the fill draws do. The fill model's uniform draws do not depend on the
    actions, so both modes use the same ones path by path. Values and records
    are day-major, then fill seed, then path.
    """

    import torch

    if not isinstance(deterministic, bool):
        raise TypeError(f"deterministic must be True or False (D-17), got {deterministic!r}")
    if torch.device(model.device).type != "cpu":
        raise ValueError("Load B1 on CPU: the D-17 seed governs torch's CPU generator only")
    days = len(daily_midprices)
    if days == 0 or len(daily_dt_arrays) != days or len(daily_market_features) != days:
        raise ValueError("Need the same non-zero number of daily midprice, dt and feature arrays")
    seeds = [int(seed) for seed in fill_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Fill seeds must be non-empty and unique: a repeated seed repeats its rollout, got {seeds}")
    if rollouts_per_seed < 1:
        raise ValueError(f"rollouts_per_seed must be at least 1, got {rollouts_per_seed}")

    mode = "deterministic" if deterministic else "stochastic"
    values: list[float] = []
    records: list[dict[str, float | int | bool]] = []
    for day_index, (prices, dt, features) in enumerate(zip(
        daily_midprices, daily_dt_arrays, daily_market_features
    )):
        for seed in seeds:
            start = time.perf_counter()
            drawdowns = _b1_rollout_maxdrawdowns(
                prices, dt, features,
                model=model,
                vecnorm_path=vecnorm_path,
                config=config,
                formula_kwargs=formula_kwargs,
                fill_seed=seed,
                rollouts=rollouts_per_seed,
                deterministic=deterministic,
            )
            print(
                f"B1 {mode} rollout: day {day_index}, fill seed {seed}, "
                f"{rollouts_per_seed} paths, {time.perf_counter() - start:.1f} s",
                flush=True,
            )
            for path, drawdown in enumerate(drawdowns):
                values.append(float(drawdown))
                records.append({
                    "day_index": day_index,
                    "fill_seed": seed,
                    "path": path,
                    "deterministic": deterministic,
                    "max_drawdown": float(drawdown),
                })
    return np.asarray(values, dtype=np.float64), records


def b3_threshold_from_b1(costs: np.ndarray) -> tuple[float, RUCVaREstimate]:
    """Frozen design: ``d = 0.8 * empirical CVaR_0.20(Z_B1)``, Z from stochastic B1 rollouts (D-17)."""

    estimate = empirical_ru_cvar(costs, rho=B3_TAIL_MASS)
    return B3_THRESHOLD_MULTIPLIER * estimate.value, estimate
