"""T1: NumpyPolicy (the exported b3_policy.npz) against SB3's deterministic predict over every Jan 7 step."""

from __future__ import annotations

import numpy as np


def test_t1_numpy_policy_matches_sb3_predict_over_jan7(jan7_replay, numpy_actions):
    steps = len(jan7_replay.actions)
    assert steps >= 10_000
    assert np.all(np.isfinite(jan7_replay.actions)) and numpy_actions.shape == jan7_replay.actions.shape
    differences = np.abs(numpy_actions - jan7_replay.actions)
    worst = float(differences.max())
    print(f"\nHB-PARITY T1: {steps} steps, max |NumpyPolicy.act - SB3 predict| = {worst!r} "
          f"(gamma {float(differences[:, 0].max())!r}, skew {float(differences[:, 1].max())!r}); "
          f"exactly equal on {int(np.sum(np.all(differences == 0, axis=1)))} steps")
    assert worst <= 1e-6


def test_numpy_normalization_is_vecnormalize_bitwise_over_jan7(jan7_replay, b3_numpy_policy):
    mismatches = sum(
        not np.array_equal(b3_numpy_policy.normalize(obs), expected)
        for obs, expected in zip(jan7_replay.raw_obs[:-1], jan7_replay.norm_obs)
    )
    clipped = int(np.sum(np.any(np.abs(jan7_replay.norm_obs) == 10.0, axis=1)))
    print(f"\nHB-PARITY normalization: {len(jan7_replay.norm_obs)} steps, {mismatches} mismatches, "
          f"{clipped} steps with a feature clipped at +-10")
    assert mismatches == 0
