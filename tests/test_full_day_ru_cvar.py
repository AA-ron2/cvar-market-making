from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cvar_mm.gym.cvar_lagrangian import (
    B3_LAMBDA_MAX,
    B3_TAIL_MASS,
    FullDayRUCVaRCallback,
    add_ru_cvar_score_advantage,
    b3_threshold_from_b1,
    empirical_ru_cvar,
    rockafellar_uryasev_value,
)
from cvar_mm.gym.drawdown_objectives import (
    CompleteTrajectorySegment,
    add_terminal_maxdd_score_advantage,
    leave_one_out_by_day,
    ppo_update_summary,
    standardized_risk_advantages,
    vecnormalize_reward_scale,
)

S = 2.5e-4  # a B1 reward scale of the size measured on the DOGE stand-in


def test_empirical_ru_value_matches_worst_twenty_percent_mean():
    costs = np.arange(10.0)
    estimate = empirical_ru_cvar(costs, rho=B3_TAIL_MASS)
    assert estimate.nu == 8.0
    assert estimate.value == pytest.approx(8.5)
    assert rockafellar_uryasev_value(costs, nu=8.0, rho=.2) == pytest.approx(8.5)


def test_threshold_is_frozen_twenty_percent_tighter_than_b1_tail_risk():
    threshold, estimate = b3_threshold_from_b1(np.arange(10.0))
    assert threshold == pytest.approx(0.8 * estimate.value)
    assert threshold < estimate.value  # B1 is deliberately not initially feasible


TWO_PATHS = [CompleteTrajectorySegment(0, 2, 0, 0, 1.0), CompleteTrajectorySegment(0, 2, 1, 0, 3.0)]
# mu_A = 1 and sigma_A = 2 (population), so the standardized advantages are [[1, -1], [1, -1]].
A_KNOWN = np.array([[3.0, -1.0], [3.0, -1.0]], dtype=np.float32)


def test_risk_term_is_divided_by_s_and_sigma_a():
    """D-15a in closed form: +-9. Without any division it would be +-5, with s only +-17, with sigma_A only +-3."""

    # Z = [1, 3]; leave-one-out gives [-2, 2]; the term is alpha * that / (s sigma_A) = 2 * [-2, 2] / (0.25 * 2).
    b2 = add_terminal_maxdd_score_advantage(A_KNOWN, TWO_PATHS, alpha=2.0, reward_scale=0.25)
    b3 = add_ru_cvar_score_advantage(A_KNOWN, TWO_PATHS, nu=0.0, lambda_value=2.0, rho=1.0, reward_scale=0.25)
    for result in (b2, b3):
        assert (result.mean, result.std) == (1.0, 2.0)
        np.testing.assert_array_equal(result.advantages, [[9.0, -9.0], [9.0, -9.0]])
        assert result.advantages.dtype == np.float32


def test_policy_score_uses_explicit_excess_loss_with_leave_one_out_baseline():
    result = add_ru_cvar_score_advantage(
        A_KNOWN, TWO_PATHS, nu=1.0, lambda_value=2.0, rho=0.2, reward_scale=0.25
    )
    # c = (Z - nu)+ / rho = [0, 10]; leave-one-out gives [-10, 10]; the term is lambda * that / (s sigma_A).
    np.testing.assert_allclose(result.advantages, [[41.0, -41.0], [41.0, -41.0]], rtol=1e-6)


def test_incomplete_segments_are_rejected_before_ppo_optimization():
    with pytest.raises(ValueError, match="Invalid complete-trajectory"):
        add_ru_cvar_score_advantage(
            np.arange(4.0).reshape(4, 1),
            [CompleteTrajectorySegment(0, 5, 0, 0, 1.0)],
            nu=0.0,
            lambda_value=1.0,
            reward_scale=S,
        )


def test_zero_weight_gives_the_batch_standardized_advantages():
    """D-15a test (a): with w = 0 PPO sees exactly (A - mu_A) / sigma_A over the whole batch."""

    rng = np.random.default_rng(3)
    advantages = rng.normal(0.4, 3.0, size=(6, 4)).astype(np.float32)
    segments = [
        CompleteTrajectorySegment(day * 3, day * 3 + 3, path, day, float(rng.gamma(2.0, 0.01)))
        for day in range(2)
        for path in range(4)
    ]
    drawdowns = np.array([segment.max_drawdown for segment in segments])
    result = standardized_risk_advantages(advantages, segments, drawdowns, weight=0.0, reward_scale=S)
    exact = advantages.astype(np.float64)
    np.testing.assert_array_equal(result.advantages, ((exact - exact.mean()) / exact.std()).astype(np.float32))
    assert (result.mean, result.std) == (exact.mean(), exact.std())
    weighted = standardized_risk_advantages(advantages, segments, drawdowns, weight=1.0, reward_scale=S)
    assert not np.array_equal(weighted.advantages, result.advantages)  # the weight is not ignored


def test_risk_advantages_reject_degenerate_batches():
    finite = np.arange(4.0).reshape(2, 2)
    costs = np.array([1.0, 3.0])
    for advantages, coefficients, weight, scale, segments, message in (
        (np.ones((2, 2)), costs, 1.0, S, TWO_PATHS, "sigma_A"),
        (np.array([[np.nan, 1.0], [0.0, 2.0]]), costs, 1.0, S, TWO_PATHS, "advantages must be finite"),
        (finite, np.array([1.0, np.inf]), 1.0, S, TWO_PATHS, "coefficients must be finite"),
        (finite, costs, 1.0, 0.0, TWO_PATHS, "reward scale"),
        (finite, costs, -1.0, S, TWO_PATHS, "risk weight"),
        (finite, costs, 1.0, S, [], "non-empty"),
    ):
        with pytest.raises(ValueError, match=message):
            standardized_risk_advantages(advantages, segments, coefficients, weight=weight, reward_scale=scale)
    with pytest.raises(ValueError, match="coefficients must be finite"):  # B2 has no other check on Z
        add_terminal_maxdd_score_advantage(
            finite, [TWO_PATHS[0], CompleteTrajectorySegment(0, 2, 1, 0, float("nan"))], alpha=1.0, reward_scale=S
        )


def sb3_gae(rewards, values, episode_starts, last_values, dones, gamma=0.999, gae_lambda=0.95):
    """SB3's RolloutBuffer.compute_returns_and_advantage, in float32 as in the buffer."""

    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            non_terminal, next_values = 1.0 - dones, last_values
        else:
            non_terminal, next_values = 1.0 - episode_starts[step + 1], values[step + 1]
        delta = rewards[step] + gamma * next_values * non_terminal - values[step]
        last = delta + gamma * gae_lambda * non_terminal * last
        advantages[step] = last
    return advantages


def test_risk_advantages_do_not_depend_on_the_reward_unit():
    """D-15a test (b): scaling raw rewards, Z and s by c > 0 leaves what PPO optimizes unchanged."""

    rng = np.random.default_rng(17)
    days, paths, steps = 2, 3, 40
    raw = rng.normal(0.0, 1.0, size=(days * steps, paths)) * S  # float64, as the env returns it
    values = rng.normal(0.0, 0.5, size=(days * steps, paths)).astype(np.float32)  # the critic, in r/s units
    episode_starts = np.zeros((days * steps, paths), dtype=np.float32)
    episode_starts[steps] = 1.0
    drawdowns = rng.gamma(2.0, 0.01, size=(days, paths))
    variance, epsilon = 0.8 * S**2, 0.2 * S**2  # B1's ret_rms.var and VecNormalize's epsilon, s**2 = var + eps

    def ppo_advantages(c, weight):
        # B1's normalizer in units scaled by c: var and epsilon scale by c**2, so s scales by c.
        normalizer = SimpleNamespace(ret_rms=SimpleNamespace(var=np.float64(c**2 * variance)), epsilon=c**2 * epsilon)
        s = vecnormalize_reward_scale(normalizer)
        normalized = np.clip((c * raw) / s, -10.0, 10.0).astype(np.float32)  # what the frozen VecNormalize emits
        gae = sb3_gae(normalized, values, episode_starts, np.zeros(paths, np.float32), np.ones(paths, np.float32))
        segments = [
            CompleteTrajectorySegment(day * steps, (day + 1) * steps, path, day, c * drawdowns[day, path])
            for day in range(days)
            for path in range(paths)
        ]
        return add_terminal_maxdd_score_advantage(gae, segments, alpha=weight, reward_scale=s).advantages

    reference = ppo_advantages(1.0, 10.0)
    for c in (1e-3, 37.0):
        np.testing.assert_allclose(ppo_advantages(c, 10.0), reference, rtol=1e-6, atol=1e-6)
    assert np.abs(reference - ppo_advantages(1.0, 0.0)).max() > 1e-2  # the risk term is really there


def test_leave_one_out_baseline_has_zero_mean_within_each_day():
    values, days = np.array([1.0, 2.0, 6.0, 0.0, 4.0]), np.array([0, 0, 0, 1, 1])
    baselined = leave_one_out_by_day(values, days)
    np.testing.assert_allclose(baselined, [-3.0, -1.5, 4.5, -4.0, 4.0])
    for day in (0, 1):
        assert baselined[days == day].sum() == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="at least two fill paths"):
        leave_one_out_by_day(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]))


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_b2_equals_b3_at_rho_one_nu_zero_lambda_alpha_with_leave_one_out(alpha):
    rng = np.random.default_rng(11)
    days, paths, steps = 3, 4, 5
    advantages = rng.normal(size=(days * steps, paths)).astype(np.float32)
    segments = [
        CompleteTrajectorySegment(day * steps, (day + 1) * steps, path, day, float(rng.gamma(2.0, 0.01)))
        for day in range(days)
        for path in range(paths)
    ]
    b2 = add_terminal_maxdd_score_advantage(advantages, segments, alpha=alpha, reward_scale=S)
    b3 = add_ru_cvar_score_advantage(advantages, segments, nu=0.0, lambda_value=alpha, rho=1.0, reward_scale=S)
    np.testing.assert_array_equal(b2.advantages, b3.advantages)  # max |diff| = 0
    assert (b2.mean, b2.std) == (b3.mean, b3.std)
    no_risk = standardized_risk_advantages(advantages, segments, np.zeros(len(segments)), weight=alpha, reward_scale=S)
    assert not np.array_equal(b2.advantages, no_risk.advantages)


def drive_one_batch(
    callback, costs: np.ndarray, steps: int = 3, raw_rewards=None, clip_reward: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Feed the callback one balanced rollout (days x fill paths) as SB3 would; return (A, Ahat).

    ``raw_rewards`` (one value per path) is what the fake VecNormalize reports as the original
    reward at every step.
    """

    days, paths = costs.shape
    gae = np.random.default_rng(days * 100 + paths).normal(size=(days * steps, paths)).astype(np.float32)
    observations = np.zeros((days * steps, paths, 7), dtype=np.float32)  # sampled for action_shift_vs_b1 (D-21)
    buffer = SimpleNamespace(buffer_size=days * steps, advantages=gae.copy(), observations=observations)
    raw = np.zeros(paths) if raw_rewards is None else np.asarray(raw_rewards, dtype=np.float64)
    env = SimpleNamespace(num_envs=paths, clip_reward=clip_reward, get_original_reward=lambda: raw.copy())
    callback.model = SimpleNamespace(
        rollout_buffer=buffer,
        seed=0,
        policy=SimpleNamespace(predict=lambda observation, deterministic: (np.zeros((len(observation), 2)), None)),
        logger=SimpleNamespace(record=lambda *args, **kwargs: None),
        get_env=lambda: env,
    )
    callback._on_rollout_start()
    for step in range(days * steps):
        day = step // steps
        callback.locals = {
            "n_steps": step,
            "dones": np.full(paths, (step + 1) % steps == 0),
            "infos": [
                {"replay_day_index": day, "trajectory_max_drawdown": float(costs[day, path])}
                for path in range(paths)
            ],
        }
        callback._on_step()
    callback._on_rollout_end()
    return gae, buffer.advantages


def test_b3_nu_is_the_batch_quantile_and_lambda_takes_the_normalized_step():
    pytest.importorskip("stable_baselines3")
    costs = np.random.default_rng(5).gamma(2.0, 0.01, size=(2, 4))
    d, eta = 0.005, 0.7
    callback = FullDayRUCVaRCallback(
        dd_threshold=d, expected_day_indices=(0, 1), reward_scale=S, lambda_learning_rate=eta, lambda_init=1.0,
        verbose=0,
    )
    gae, advantages = drive_one_batch(callback, costs)

    estimate = empirical_ru_cvar(costs.ravel())
    assert callback.nu_history == [np.quantile(costs.ravel(), 1.0 - B3_TAIL_MASS, method="higher")]
    assert callback.cvar_history[0] == pytest.approx(estimate.value, rel=1e-12)
    assert callback.lambda_used_history == [1.0]
    assert callback.lambda_history[0] == pytest.approx(1.0 + eta * (estimate.value - d) / d, rel=1e-12)
    # The batch uses lambda before its own dual step (lambda_used), not the updated lambda.
    np.testing.assert_array_equal(
        advantages,
        add_ru_cvar_score_advantage(
            gae, callback._segments, nu=estimate.nu, lambda_value=1.0, reward_scale=S
        ).advantages,
    )


def test_batch_diagnostics_record_s_sigma_a_and_the_clipped_reward_fraction():
    """D-15a: a reward is clipped when |r_raw| / s > clip_reward (here 3, so a hard-coded 10 would fail)."""

    pytest.importorskip("stable_baselines3")
    costs = np.random.default_rng(8).gamma(2.0, 0.01, size=(2, 4))
    callback = FullDayRUCVaRCallback(
        dd_threshold=0.01, expected_day_indices=(0, 1), reward_scale=S, lambda_learning_rate=0.5, verbose=0
    )
    gae, _ = drive_one_batch(callback, costs, raw_rewards=[2.0 * S, -4.0 * S, 0.0, 3.5 * S], clip_reward=3.0)

    (row,) = callback.diagnostics
    assert row["reward_scale_s"] == S
    assert row["reward_clip_fraction"] == 0.5  # -4 s and 3.5 s exceed 3 s; 2 s and 0 do not
    exact = gae.astype(np.float64)
    assert (row["advantage_mean"], row["advantage_std"]) == (exact.mean(), exact.std())


def test_ppo_update_summary_reads_epochs_and_the_kl_stop_from_the_adam_steps():
    assert ppo_update_summary(320, 32, 10) == (10, False)  # a full update
    assert ppo_update_summary(100, 32, 10) == (3, True)  # stopped in the fourth epoch
    assert ppo_update_summary(31, 32, 10) == (0, True)  # stopped in the first epoch
    assert ppo_update_summary(0, 32, 10) == (0, True)  # stopped before the first step
    for impossible in (321, -1):
        with pytest.raises(ValueError, match="cannot come from"):
            ppo_update_summary(impossible, 32, 10)


@pytest.mark.parametrize("missing", ["value_loss", "explained_variance"])
def test_an_update_whose_train_logged_no_value_loss_or_explained_variance_is_not_recorded(missing):
    """D-23: closing an update reads SB3's train/* values of that update's PPO.train() with .get, and raises,
    naming the missing one, rather than complete the update's diagnostics row without it."""

    pytest.importorskip("stable_baselines3")
    costs = np.random.default_rng(9).gamma(2.0, 0.01, size=(2, 4))
    callback = FullDayRUCVaRCallback(
        dd_threshold=0.01, expected_day_indices=(0, 1), reward_scale=S, lambda_learning_rate=0.5, verbose=0
    )
    drive_one_batch(callback, costs)  # opens update 1's row
    logged = {"train/approx_kl": 0.0, "train/clip_fraction": 0.0, "train/value_loss": 0.5,
              "train/explained_variance": 0.25}
    del logged[f"train/{missing}"]
    callback.model.n_steps, callback.model.batch_size, callback.model.n_epochs = 2 * 3, 1, 10
    callback.model.logger = SimpleNamespace(record=lambda *args, **kwargs: None, name_to_value=logged)
    with pytest.raises(RuntimeError, match=f"PPO.train\\(\\) recorded no {missing} for this update"):
        callback._on_rollout_start()  # closes update 1, as at the next rollout's start
    assert "adam_steps" not in callback.diagnostics[-1]


def test_b3_lambda_is_projected_onto_zero_and_lambda_max():
    pytest.importorskip("stable_baselines3")
    costs = np.random.default_rng(6).gamma(2.0, 0.01, size=(2, 4))
    upper = FullDayRUCVaRCallback(
        dd_threshold=1e-6, expected_day_indices=(0, 1), reward_scale=S, lambda_learning_rate=1e9, verbose=0
    )
    drive_one_batch(upper, costs)
    assert upper.lambda_history == [B3_LAMBDA_MAX]
    lower = FullDayRUCVaRCallback(
        dd_threshold=1e3, expected_day_indices=(0, 1), reward_scale=S, lambda_learning_rate=2.0, lambda_init=0.5,
        verbose=0,
    )
    drive_one_batch(lower, costs)
    assert lower.lambda_history == [0.0]
