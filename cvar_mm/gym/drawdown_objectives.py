"""Full-trajectory drawdown terms that enter PPO's policy surrogate, in B1's units (D-15a)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - permits documentation-only imports
    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, verbose: int = 0):
            self.verbose = verbose


@dataclass(frozen=True)
class CompleteTrajectorySegment:
    start: int
    stop: int
    env_index: int
    day_index: int
    max_drawdown: float


@dataclass(frozen=True)
class RiskAdjustedAdvantages:
    """The advantages PPO optimizes for one batch, and the batch statistics behind them (D-15a)."""

    advantages: np.ndarray  # float32, the rollout buffer's (n_steps, n_envs) shape
    mean: float  # mu_A of the GAE advantages of the normalized rewards
    std: float  # sigma_A, their population standard deviation


def vecnormalize_reward_scale(vecnorm) -> float:
    """s = sqrt(ret_rms.var + epsilon), the divisor ``VecNormalize.normalize_reward`` applies."""

    return float(np.sqrt(vecnorm.ret_rms.var + vecnorm.epsilon))


def leave_one_out_by_day(values: np.ndarray, day_indices: Sequence[int]) -> np.ndarray:
    """Return ``c_i - mean(c_j : j != i, same replay day)`` for every fill path i.

    For n paths on a day this is ``(n c_i - sum_day c) / (n - 1)``, so it sums
    to zero within each day.  The baseline leaves out path i itself (Kool, van
    Hoof & Welling 2019); it needs at least two fill paths per day (D-12).
    """

    values = np.asarray(values, dtype=np.float64)
    days = np.asarray(day_indices)
    if values.ndim != 1 or values.shape != days.shape:
        raise ValueError("values and day_indices must be matching one-dimensional arrays")
    _, groups, counts = np.unique(days, return_inverse=True, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("The leave-one-out baseline needs at least two fill paths per day")
    paths = counts[groups]
    return (paths * values - np.bincount(groups, weights=values)[groups]) / (paths - 1)


def standardized_risk_advantages(
    advantages: np.ndarray,
    segments: Sequence[CompleteTrajectorySegment],
    coefficients: np.ndarray,
    *,
    weight: float,
    reward_scale: float,
) -> RiskAdjustedAdvantages:
    """D-15a: ``Ahat = (A - mu_A) / sigma_A - weight * (c_i - c_bar_i) / (reward_scale * sigma_A)``.

    A are SB3's GAE advantages of B1's normalized rewards r/s, and mu_A and
    sigma_A their mean and population std over the whole batch. c_i - c_bar_i
    is path i's score in USDT minus the mean over the other paths of the same
    day (leave-one-out), subtracted from every action of that path. With
    A_raw = s A this is
    ``[(A_raw - mean(A_raw)) - weight * (c_i - c_bar_i)] / (s sigma_A)``: the trade-off
    between the P&L objective and the risk score is exactly ``weight``, and Ahat
    does not depend on the reward unit. B2 (c = Z, weight = alpha) and B3
    (c = (Z - nu)+ / rho, weight = lambda) share this arithmetic, so B3 at
    rho = 1, nu = 0, lambda = alpha reproduces B2 bit for bit (D-11).
    Computed in float64 and returned as float32, the rollout buffer's dtype.
    """

    if not segments:
        raise ValueError("complete segments must be non-empty")
    if not (math.isfinite(weight) and weight >= 0):
        raise ValueError(f"the risk weight must be finite and non-negative, got {weight}")
    if not (math.isfinite(reward_scale) and reward_scale > 0):
        raise ValueError(f"the reward scale s must be finite and positive, got {reward_scale}")
    values = np.asarray(advantages, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("advantages must have the rollout buffer's (n_steps, n_envs) shape")
    for segment in segments:
        if (
            segment.start < 0
            or segment.stop <= segment.start
            or segment.stop > values.shape[0]
            or not 0 <= segment.env_index < values.shape[1]
        ):
            raise ValueError(f"Invalid complete-trajectory segment: {segment}")
    costs = np.asarray(coefficients, dtype=np.float64)
    if not np.all(np.isfinite(costs)):
        raise ValueError("the risk coefficients must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("the PPO advantages must be finite")
    mean, std = float(values.mean()), float(values.std())
    if not (math.isfinite(std) and std > 0):
        raise ValueError(f"the advantage std sigma_A must be finite and positive, got {std}")
    baselined = leave_one_out_by_day(costs, [segment.day_index for segment in segments])
    adjusted = values - mean
    adjusted /= std
    step = weight / (reward_scale * std)
    for coefficient, segment in zip(baselined, segments):
        adjusted[segment.start : segment.stop, segment.env_index] -= step * coefficient
    return RiskAdjustedAdvantages(adjusted.astype(np.float32), mean, std)


def add_terminal_maxdd_score_advantage(
    advantages: np.ndarray,
    segments: Sequence[CompleteTrajectorySegment],
    *,
    alpha: float,
    reward_scale: float,
) -> RiskAdjustedAdvantages:
    """B2: the likelihood-ratio term of ``-alpha E[MaxDD]`` in B1's units (D-15a).

    Every action of complete trajectory i receives
    ``-alpha (Z_i - mean of Z_j over the other fill paths of the same day) / (s sigma_A)``
    on top of its standardized advantage. The baseline excludes Z_i, so the
    score estimator of the gradient of E[Z] stays unbiased.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    costs = np.asarray([segment.max_drawdown for segment in segments], dtype=np.float64)
    return standardized_risk_advantages(advantages, segments, costs, weight=alpha, reward_scale=reward_scale)


def ppo_update_summary(adam_steps: int, minibatches_per_epoch: int, n_epochs: int) -> tuple[int, bool]:
    """Epochs completed, and whether SB3's KL early stop fired, from the Adam steps of one PPO.train().

    SB3 checks the approximate KL of each minibatch before its optimizer step
    and stops the remaining epochs once it exceeds 1.5 x target_kl, so a full
    update takes exactly ``n_epochs * minibatches_per_epoch`` steps.
    """

    if minibatches_per_epoch < 1 or n_epochs < 1 or not 0 <= adam_steps <= n_epochs * minibatches_per_epoch:
        raise ValueError(
            f"{adam_steps} Adam steps cannot come from {n_epochs} epochs of {minibatches_per_epoch} minibatches"
        )
    return adam_steps // minibatches_per_epoch, adam_steps < n_epochs * minibatches_per_epoch


UPDATE_DIAGNOSTICS = (
    "reward_scale_s",
    "advantage_mean",
    "advantage_std",
    "reward_clip_fraction",
    "adam_steps",
    "minibatches_per_epoch",
    "epochs_completed",
    "kl_stopped",
    "approx_kl",  # this and the next three: SB3's train/* values of the update's PPO.train()
    "clip_fraction",
    "value_loss",  # D-23
    "explained_variance",  # D-23
    "mean_log_std",
    "adam_damping_ratio",
    "action_shift_vs_b1",
    "elapsed_seconds",
)
# D-21: action_shift_vs_b1 is measured on this many observations of the first rollout, one value per action
# (the policy's action order, common.write_metadata's "policy_action"); histories give each its own column.
ACTION_SHIFT_SAMPLE = 10_000
ACTION_DIMENSIONS = ("gamma", "skew")
ACTION_SHIFT_COLUMNS = tuple(f"action_shift_vs_b1_{action}" for action in ACTION_DIMENSIONS)
UPDATE_DIAGNOSTIC_COLUMNS = tuple(
    column
    for key in UPDATE_DIAGNOSTICS
    for column in (ACTION_SHIFT_COLUMNS if key == "action_shift_vs_b1" else (key,))
)


class CompleteDayRiskCallback(BaseCallback):
    """What B2 and B3 share (D-11, D-15): complete-day batches, the D-15a advantages, frozen log sigma.

    A subclass implements ``_adjust_batch(advantages, segments)``, which returns
    the batch's ``RiskAdjustedAdvantages``. ``diagnostics`` holds one dict per
    update with the ``UPDATE_DIAGNOSTICS`` (D-15): the first four are recorded
    when the batch is adjusted, the rest after its PPO.train(), i.e. at the
    next rollout start or at training end.

    ``action_shift_vs_b1`` (D-21) is the mean |clip(mu_now) - clip(mu_B1)| per
    action on a fixed sample of the first rollout's normalized observations,
    taken before the first update, when the policy is still the warm-start B1.
    The sample is drawn by a generator seeded with the training seed, and the
    deterministic actions draw no noise, so no RNG that training uses moves.
    """

    label = "risk-aware"

    def __init__(self, *, expected_day_indices: tuple[int, ...], reward_scale: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        if not (math.isfinite(reward_scale) and reward_scale > 0):
            raise ValueError("the reward scale s must be finite and positive (D-15a)")
        self.expected_day_indices = tuple(expected_day_indices)
        self.reward_scale = float(reward_scale)
        self.diagnostics: list[dict] = []
        self._starts: np.ndarray | None = None
        self._segments: list[CompleteTrajectorySegment] = []
        self._frozen_log_std = None
        self._clip_reward = math.inf
        self._clipped_rewards = 0
        self._rewards_seen = 0
        self._actor_parameters: frozenset = frozenset()
        self._step_hook = None
        self._adam_steps = 0
        self._adam_steps_before_update = 0
        self._damping_sum = 0.0
        self._damping_count = 0
        self._start_time = time.perf_counter()
        self._shift_observations: np.ndarray | None = None
        self._b1_actions: np.ndarray | None = None
        self.action_shift_before_updates: list[float] | None = None  # 0 by construction (D-21)

    @property
    def elapsed_history(self) -> list[float]:
        """Cumulative wall time at the end of each update, after its PPO.train()."""

        return [row["elapsed_seconds"] for row in self.diagnostics if "elapsed_seconds" in row]

    def _adjust_batch(
        self, advantages: np.ndarray, segments: list[CompleteTrajectorySegment]
    ) -> RiskAdjustedAdvantages:
        raise NotImplementedError

    def _on_training_start(self) -> None:
        from stable_baselines3.common.vec_env import VecNormalize

        env = self.training_env
        if not isinstance(env, VecNormalize) or not env.norm_reward or env.training:
            raise RuntimeError(
                f"{self.label} must train on B1's normalizer frozen with norm_reward=True and training=False (D-15a)"
            )
        if vecnormalize_reward_scale(env) != self.reward_scale:
            raise RuntimeError(f"{self.label}: the reward scale s differs from the training normalizer's (D-15a)")
        policy = self.model.policy
        if policy.log_std.requires_grad:
            raise RuntimeError(f"{self.label} must keep log sigma frozen at B1's value (D-15b)")
        self._frozen_log_std = policy.log_std.detach().clone()
        self._actor_parameters = frozenset(
            id(parameter)
            for name, parameter in policy.named_parameters()
            if name.startswith(("mlp_extractor.policy_net.", "action_net."))
        )
        self._step_hook = policy.optimizer.register_step_post_hook(self._after_adam_step)
        self._start_time = time.perf_counter()

    def _after_adam_step(self, optimizer, args, kwargs) -> None:
        """Count Adam steps and accumulate ``adam_damping_ratio`` over the policy-mean parameters.

        The ratio is mean |grad| over the mean bias-corrected sqrt(v_hat) that
        Adam divides by: about 1 when the second moments match the current
        gradients, far below 1 when stale moments damp the steps (D-15c).
        """

        self._adam_steps += 1
        gradients = roots = 0.0
        for group in optimizer.param_groups:
            beta2 = group["betas"][1]
            for parameter in group["params"]:
                state = optimizer.state.get(parameter)
                if id(parameter) not in self._actor_parameters or parameter.grad is None or not state:
                    continue
                v_hat = state["exp_avg_sq"] / (1.0 - beta2 ** float(state["step"]))
                gradients += float(parameter.grad.abs().sum())
                roots += float(v_hat.sqrt().sum())
        if roots > 0:
            self._damping_sum += gradients / roots
            self._damping_count += 1

    def _deterministic_actions(self, observations: np.ndarray) -> np.ndarray:
        """clip(mu) on these normalized observations, exactly as SB3's predict(deterministic=True) returns the
        evaluation action; it draws no noise."""

        actions, _ = self.model.policy.predict(observations, deterministic=True)
        return np.asarray(actions, dtype=np.float64)

    def _take_action_shift_sample(self, observations: np.ndarray) -> None:
        """D-21: up to ACTION_SHIFT_SAMPLE of the first rollout's observations, drawn without replacement by a
        generator seeded with the training seed, and B1's deterministic actions on them."""

        seed = self.model.seed
        if not isinstance(seed, int):
            raise RuntimeError(f"{self.label}: action_shift_vs_b1 samples with the training seed, got {seed!r} (D-21)")
        flat = observations.reshape(-1, *observations.shape[2:])
        rows = np.random.default_rng(seed).choice(len(flat), size=min(ACTION_SHIFT_SAMPLE, len(flat)), replace=False)
        self._shift_observations = flat[rows].copy()
        self._b1_actions = self._deterministic_actions(self._shift_observations)
        self.action_shift_before_updates = self._action_shift()

    def _action_shift(self) -> list[float]:
        """Mean |clip(mu_now) - clip(mu_B1)| per action dimension on the D-21 sample."""

        shift = np.abs(self._deterministic_actions(self._shift_observations) - self._b1_actions).mean(axis=0)
        return [float(value) for value in shift]

    def _check_log_std_frozen(self) -> None:
        """D-15b: B2/B3 train only the policy mean and the critic."""

        if self._frozen_log_std is None:
            return
        import torch

        if not torch.equal(self.model.policy.log_std.detach(), self._frozen_log_std):
            raise RuntimeError(f"{self.label}: log sigma changed during training (D-15b)")

    def _close_update(self) -> None:
        """After the last batch's PPO.train(): the D-15b check and that update's PPO diagnostics."""

        self._check_log_std_frozen()
        if not self.diagnostics or "adam_steps" in self.diagnostics[-1]:
            return
        model = self.model
        steps = self._adam_steps - self._adam_steps_before_update
        minibatches = -(-model.n_steps * self.training_env.num_envs // model.batch_size)
        epochs, stopped = ppo_update_summary(steps, minibatches, model.n_epochs)
        # .get only: indexing SB3's defaultdict would add a key and break its next dump.
        logged = {
            key: self.logger.name_to_value.get(f"train/{key}")
            for key in ("approx_kl", "clip_fraction", "value_loss", "explained_variance")  # the last two: D-23
        }
        missing = [key for key, value in logged.items() if value is None]
        if missing:
            raise RuntimeError(f"{self.label}: PPO.train() recorded no {' or '.join(missing)} for this update")
        row = self.diagnostics[-1]
        row.update(
            adam_steps=steps,
            minibatches_per_epoch=minibatches,
            epochs_completed=epochs,
            kl_stopped=stopped,
            # SB3's values as they are (plain floats for the JSON metadata); explained_variance is NaN, SB3's
            # value, only if the batch's returns have zero variance.
            **{key: float(value) for key, value in logged.items()},
            mean_log_std=float(model.policy.log_std.mean()),
            adam_damping_ratio=self._damping_sum / self._damping_count if self._damping_count else None,
            action_shift_vs_b1=self._action_shift(),
            elapsed_seconds=time.perf_counter() - self._start_time,
        )
        for key in ("adam_steps", "epochs_completed", "kl_stopped", "mean_log_std", "adam_damping_ratio"):
            if row[key] is not None:
                self.logger.record(f"risk/{key}", float(row[key]))
        for column, value in zip(ACTION_SHIFT_COLUMNS, row["action_shift_vs_b1"]):
            self.logger.record(f"risk/{column}", value)

    def _on_rollout_start(self) -> None:
        self._close_update()
        env = self.training_env
        self._clip_reward = float(env.clip_reward)
        self._clipped_rewards = self._rewards_seen = 0
        self._starts = np.zeros(env.num_envs, dtype=np.int64)
        self._segments = []

    def _on_step(self) -> bool:
        assert self._starts is not None
        # D-15a: VecNormalize clips r/s at +-clip_reward; the expected fraction is 0.
        raw = np.asarray(self.training_env.get_original_reward(), dtype=np.float64)
        self._clipped_rewards += int(np.count_nonzero(np.abs(raw) / self.reward_scale > self._clip_reward))
        self._rewards_seen += raw.size
        step_index = int(self.locals["n_steps"])
        for env_index, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][env_index]
                self._segments.append(
                    CompleteTrajectorySegment(
                        start=int(self._starts[env_index]),
                        stop=step_index + 1,
                        env_index=env_index,
                        day_index=int(info["replay_day_index"]),
                        max_drawdown=float(info["trajectory_max_drawdown"]),
                    )
                )
                self._starts[env_index] = step_index + 1
        return True

    def _on_rollout_end(self) -> None:
        assert self._starts is not None
        buffer = self.model.rollout_buffer
        if np.any(self._starts != buffer.buffer_size):
            raise RuntimeError(f"{self.label} PPO rollout ended with an incomplete market-day trajectory")
        observed = sorted(segment.day_index for segment in self._segments)
        expected = sorted(list(self.expected_day_indices) * self.training_env.num_envs)
        if observed != expected:
            raise RuntimeError(
                f"{self.label} batch is not balanced over complete days: observed={observed}, expected={expected}"
            )
        if self._shift_observations is None:  # the first rollout, before any update: the policy is B1 (D-21)
            self._take_action_shift_sample(buffer.observations)
        # PPO.train() consumes this array, so the risk gradient is in the clipped policy loss
        # itself; the critic's targets (returns) were fixed before this callback ran.
        result = self._adjust_batch(buffer.advantages, self._segments)
        buffer.advantages = result.advantages
        row = {
            "reward_scale_s": self.reward_scale,
            "advantage_mean": result.mean,
            "advantage_std": result.std,
            "reward_clip_fraction": self._clipped_rewards / self._rewards_seen,
        }
        self.diagnostics.append(row)
        for key, value in row.items():
            self.logger.record(f"risk/{key}", value)
        self._adam_steps_before_update = self._adam_steps
        self._damping_sum, self._damping_count = 0.0, 0

    def _on_training_end(self) -> None:
        self._close_update()
        if self._step_hook is not None:
            self._step_hook.remove()
            self._step_hook = None
        # SB3 writes update u's train/* with update u + 1's dump, at step (u + 1) x batch, and never
        # writes the last update's; dump them, with this update's risk/* values, at that step.
        self.logger.dump(step=self.model.num_timesteps + self.model.n_steps * self.training_env.num_envs)


class FullDayMaxDrawdownPPOCallback(CompleteDayRiskCallback):
    """B2: inject the full-day terminal-MaxDD score term into PPO advantages, in B1's units."""

    label = "B2"

    def __init__(
        self,
        alpha: float,
        expected_day_indices: tuple[int, ...],
        *,
        reward_scale: float,
        verbose: int = 0,
    ):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        super().__init__(expected_day_indices=expected_day_indices, reward_scale=reward_scale, verbose=verbose)
        self.alpha = float(alpha)

    def _adjust_batch(
        self, advantages: np.ndarray, segments: list[CompleteTrajectorySegment]
    ) -> RiskAdjustedAdvantages:
        result = add_terminal_maxdd_score_advantage(
            advantages, segments, alpha=self.alpha, reward_scale=self.reward_scale
        )
        self.logger.record("risk/b2_mean_full_day_maxdd", np.mean([segment.max_drawdown for segment in segments]))
        return result
