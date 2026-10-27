"""T2: FeatureEngine against the frozen replay's observations over the full Jan 7 day, bitwise."""

from __future__ import annotations

import numpy as np

from hummingbot_adapter.features import FeatureEngine


def test_t2_feature_engine_reproduces_every_jan7_observation_bitwise(jan7_replay):
    replay = jan7_replay
    T = replay.terminal_time
    engine = FeatureEngine(tick_size=1e-5, q_max=10, window=100, imbalance_depth=5)
    columns = replay.row_columns
    mismatches, mid_mismatches = [], []
    for k, values in enumerate(replay.row_values):
        row = dict(zip(columns, values))
        if k == 0:
            obs = engine.reset(row, replay.t[0], T, replay.q[0])
        else:
            obs = engine.update(row, replay.dt[k], replay.t[k], T, replay.q[k])
        if obs.dtype != np.float32 or not np.array_equal(obs, replay.raw_obs[k]):
            mismatches.append(k)
        if engine.get_state()["last_mid"] != replay.mid[k]:  # float64: the S that the quotes use
            mid_mismatches.append(k)
    print(f"\nHB-PARITY T2: {len(replay.row_values)} observations compared (reset + {len(replay.row_values) - 1} "
          f"steps), {len(mismatches)} mismatches; float64 mid mismatches {len(mid_mismatches)}; T = {T!r} s; "
          f"zero-dt rows {int(np.sum(replay.dt[1:] == 0))}; q range [{replay.q.min():.0f}, {replay.q.max():.0f}]")
    assert len(replay.row_values) == len(replay.raw_obs) == len(replay.S)
    assert mismatches == []
    assert mid_mismatches == []
