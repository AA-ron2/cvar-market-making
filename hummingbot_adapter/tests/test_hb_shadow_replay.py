"""WO-HB3: the shadow replay (hummingbot_adapter.shadow_replay), its refusals and its parity check, and the import
discipline of the two analysis modules. The smoke runs on a 2,000-row slice of the Jan 7 2025 file need HB_ARTIFACTS
(conftest.py); everything else is synthetic."""

from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd
import pytest

from hummingbot_adapter import analysis, shadow_replay
from hummingbot_adapter.provenance import file_sha256
from hummingbot_adapter.tests.support import B0_GAMMA, REPO_ROOT, synthetic_book, write_book_csv

# The header of the pipeline's B0 episode file (results/b0_test_episodes.csv), as the author read it on Snellius
# on 2026-10-23
B0_EPISODES_HEADER = ("Day,Fill Seed,Rollout,Sharpe,Sortino,Max DD,P&L-to-MAP,Final PnL,Mean |q|,Terminal q,"
                      "Mean spread,Sharpe Snapshot (supplementary),Sortino Snapshot (supplementary),Near Cap Fraction")
SLICE_ROWS = 2_000
JAN7 = "2025-01-07"


def export_pipeline_env(monkeypatch) -> None:
    """configs/final_pipeline.env's exports into the environment, as `source configs/final_pipeline.env` would."""

    for raw in (REPO_ROOT / "configs" / "final_pipeline.env").read_bytes().decode("utf-8").split("\n"):
        line = raw.strip()
        if line.startswith("export "):
            key, _, value = line[len("export "):].partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            monkeypatch.setenv(key.strip(), value)


@pytest.fixture
def run_inputs(tmp_path):
    """A day file (never read: every refusal runs first), hb_calibration-style params and a frozen-value gamma file."""

    day = tmp_path / "hb_data" / "hb_b0" / "2026-10-22"
    day.mkdir(parents=True)
    day_file = day / "snapshots_2026-10-22.csv.gz"
    day_file.write_bytes(b"not read")
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"sigma": 7.5e-06, "A": 0.095, "kappa": 41_000.0}), encoding="utf-8")
    gamma = tmp_path / "b0_gamma_fixed.txt"
    gamma.write_text(repr(B0_GAMMA), encoding="utf-8")
    return {"day_file": day_file, "params": params, "gamma": gamma, "out": tmp_path / "hb_analysis" / "shadow"}


def run_args(inputs, model="b0", **override) -> list[str]:
    args = {"--model": model, "--day-file": inputs["day_file"], "--date": "2026-10-22", "--params": inputs["params"],
            "--out-dir": inputs["out"]}
    if model == "b0":
        args["--gamma-file"] = inputs["gamma"]
    args |= override
    return ["run"] + [str(item) for pair in args.items() for item in pair if pair[1] is not None]


# --- inputs and refusals -----------------------------------------------------------------------------------------


def test_params_files_of_both_layouts_and_only_those(tmp_path):
    good = tmp_path / "hb.json"
    good.write_text(json.dumps({"sigma": 1e-5, "A": 0.1, "kappa": 4e4, "r_squared_log": 0.99}), encoding="utf-8")
    assert shadow_replay.read_params(good) | {"path": None, "sha256": None} == {
        "path": None, "sha256": None, "kind": "hb_calibration", "keys": ["sigma", "A", "kappa"],
        "sigma": 1e-5, "A": 0.1, "kappa": 4e4}
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"sigma_train": 2e-5, "A_train": 0.2, "kappa_train": 1.2e4, "per_day": [
        {"sigma": 1, "A": 1, "kappa": 1}]}), encoding="utf-8")
    assert shadow_replay.read_params(train)["kind"] == "formula_train_params"
    assert shadow_replay.read_params(train)["sha256"] == file_sha256(train)
    for content in ({"sig": 1e-5, "A": 0.1, "kappa": 4e4}, {"sigma": 1e-5, "A": 0.1, "kappa": 4e4, "A_train": 0.1},
                    {"sigma": 1e-5, "A": -0.1, "kappa": 4e4}, {"sigma": True, "A": 0.1, "kappa": 4e4}, [1, 2, 3]):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(shadow_replay.ShadowRefused):
            shadow_replay.read_params(bad)


def test_params_with_unknown_keys_are_refused(run_inputs, monkeypatch, capsys):
    export_pipeline_env(monkeypatch)
    run_inputs["params"].write_text(json.dumps({"s": 1e-5, "a": 0.1, "k": 4e4}), encoding="utf-8")
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "hb_calibration.json" in capsys.readouterr().err
    assert not run_inputs["out"].exists()


def test_a_gamma_file_other_than_the_frozen_one_is_refused(run_inputs, monkeypatch, capsys):
    export_pipeline_env(monkeypatch)
    run_inputs["gamma"].write_text("0.975143028728294", encoding="utf-8")  # the stale repository copy (PROTOCOL §2)
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "not the frozen gamma_B0" in capsys.readouterr().err
    run_inputs["gamma"].write_text(repr(B0_GAMMA) + "\n", encoding="utf-8")  # the frozen value, other bytes
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "SHA-256" in capsys.readouterr().err
    assert not run_inputs["out"].exists()


def test_a_model_or_normalizer_hash_mismatch_is_refused(run_inputs, tmp_path, monkeypatch, capsys):
    export_pipeline_env(monkeypatch)
    model, vecnorm = tmp_path / "ppo_b3_doge.zip", tmp_path / "vecnorm_b3.pkl"
    model.write_bytes(b"a stale policy")
    vecnorm.write_bytes(b"a stale normalizer")
    arguments = run_args(run_inputs, "b3", **{"--model-zip": model, "--vecnorm": vecnorm})
    assert shadow_replay.main(arguments) == 2
    assert "B3 policy" in capsys.readouterr().err
    # a policy that passes, a stale normalizer: the normalizer's own check refuses
    monkeypatch.setattr(shadow_replay, "MODEL_SHA256", file_sha256(model))
    assert shadow_replay.main(arguments) == 2
    assert "B3 normalizer" in capsys.readouterr().err
    assert not run_inputs["out"].exists()


@pytest.mark.parametrize("name, value", [("FORMULA_SKEW_TICKS_MAX", "4"), ("EVALUATION_FILL_SEEDS", "42"),
                                         ("EVALUATION_ROLLOUTS", "10"), ("FORMULA_SKEW_TICKS_MAX", None)])
def test_the_pipeline_environment_is_required(run_inputs, monkeypatch, capsys, name, value):
    export_pipeline_env(monkeypatch)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    assert shadow_replay.main(run_args(run_inputs)) == 2
    error = capsys.readouterr().err
    assert name in error and "source configs/final_pipeline.env" in error


def test_missing_or_unreadable_inputs_are_refusals(run_inputs, tmp_path, monkeypatch, capsys):
    export_pipeline_env(monkeypatch)
    assert shadow_replay.main(run_args(run_inputs, **{"--gamma-file": tmp_path / "none.txt"})) == 2
    assert "no such input file" in capsys.readouterr().err
    run_inputs["params"].write_text("{not json", encoding="utf-8")
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "not a readable JSON file" in capsys.readouterr().err
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert shadow_replay.main(["compare", "--ours", str(empty), "--reference", str(empty), "--day", JAN7]) == 2


def test_the_out_dir_holds_shadow_outputs_only(run_inputs, monkeypatch, capsys):
    """Never a bot, day or analysis folder: their SHA256SUMS would be rebuilt over the wrong files."""

    export_pipeline_env(monkeypatch)
    run_inputs["out"].mkdir(parents=True)
    (run_inputs["out"] / "state.json").write_text("{}", encoding="utf-8")
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "holds other things than shadow-replay outputs" in capsys.readouterr().err
    (run_inputs["out"] / "state.json").unlink()
    (run_inputs["out"] / "shadow_b3_2026-10-22_episodes.csv").write_text("x", encoding="utf-8")
    (run_inputs["out"] / "SHA256SUMS").write_text("", encoding="utf-8")
    monkeypatch.setenv("EVALUATION_ROLLOUTS", "10")  # refused by the next check, the environment's
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "EVALUATION_ROLLOUTS" in capsys.readouterr().err  # past the out-dir check: other shadow outputs are fine


def test_arguments_of_the_other_model_and_existing_outputs_are_refused(run_inputs, monkeypatch, capsys):
    export_pipeline_env(monkeypatch)
    assert shadow_replay.main(run_args(run_inputs, "b0", **{"--vecnorm": run_inputs["gamma"]})) == 2
    assert shadow_replay.main(run_args(run_inputs, **{"--date": "2026-10-23"})) == 2  # not the file's date
    run_inputs["out"].mkdir(parents=True)
    (run_inputs["out"] / "shadow_b0_2026-10-22_episodes.csv").write_text("x", encoding="utf-8")
    assert shadow_replay.main(run_args(run_inputs)) == 2
    assert "never overwritten" in capsys.readouterr().err
    inside = run_inputs["day_file"].parent / "shadow"
    assert shadow_replay.main(run_args(run_inputs, **{"--out-dir": inside})) == 2  # never into the day's folder


# --- compare -------------------------------------------------------------------------------------------------------


def episodes(day: str = JAN7, seeds=(42, 43), rollouts: int = 3, shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for rollout in range(rollouts):
            base = seed + rollout / 10
            rows.append({"Day": day, "Fill Seed": seed, "Rollout": rollout, "Sharpe": base, "Sortino": -base,
                         "Max DD": base / 100, "P&L-to-MAP": 0.0, "Final PnL": base / 1000 + shift, "Mean |q|": 3.0,
                         "Terminal q": -1.0, "Mean spread": 2e-5, "Sharpe Snapshot (supplementary)": 0.1,
                         "Sortino Snapshot (supplementary)": np.nan, "Near Cap Fraction": 0.25})
    return pd.DataFrame(rows, columns=list(shadow_replay.EPISODE_COLUMNS))


def test_compare_matches_on_fill_seed_and_rollout_over_every_numeric_column():
    reference = pd.concat([episodes(), episodes(day="2025-01-08")], ignore_index=True)
    ours = episodes().sample(frac=1.0, random_state=1)  # another row order
    result = shadow_replay.compare_frames(ours, reference, JAN7)
    assert result.ok and result.problems == []
    assert result.summary["metric"].tolist() == list(shadow_replay.EPISODE_COLUMNS[3:])
    assert result.summary["max_abs"].max() == 0.0
    nearly = episodes()
    nearly["Final PnL"] *= 1 + 5e-10  # within 1e-9 relative
    assert shadow_replay.compare_frames(nearly, reference, JAN7).ok
    off = episodes()
    off.loc[off["Rollout"] == 2, "Near Cap Fraction"] += 1e-6
    result = shadow_replay.compare_frames(off, reference, JAN7)
    assert not result.ok
    assert result.failures[["Fill Seed", "Rollout", "metric"]].values.tolist() == [
        [42, 2, "Near Cap Fraction"], [43, 2, "Near Cap Fraction"]]
    nan = episodes()
    nan.loc[0, "Sortino Snapshot (supplementary)"] = 1.0  # a value where the reference has NaN
    assert not shadow_replay.compare_frames(nan, reference, JAN7).ok


def test_compare_refuses_other_columns_or_keys():
    reference = episodes()
    assert not shadow_replay.compare_frames(reference.drop(columns=["Near Cap Fraction"]), reference, JAN7).ok
    swapped = reference[[*reference.columns[:3], "Sortino", "Sharpe", *reference.columns[5:]]]
    assert "the columns differ" in shadow_replay.compare_frames(swapped, reference, JAN7).problems[0]
    assert not shadow_replay.compare_frames(episodes(rollouts=2), reference, JAN7).ok
    assert not shadow_replay.compare_frames(episodes(day="2025-01-08"), reference, JAN7).ok


def test_compare_command_exit_codes(tmp_path, capsys):
    ours, reference = tmp_path / "ours.csv", tmp_path / "reference.csv"
    episodes().to_csv(ours, index=False)
    episodes().to_csv(reference, index=False)
    assert shadow_replay.main(["compare", "--ours", str(ours), "--reference", str(reference), "--day", JAN7]) == 0
    assert "compare PASSED" in capsys.readouterr().out
    episodes(shift=1e-6).to_csv(ours, index=False)
    assert shadow_replay.main(["compare", "--ours", str(ours), "--reference", str(reference), "--day", JAN7]) == 1
    assert "compare FAILED" in capsys.readouterr().out
    assert shadow_replay.main(["compare", "--ours", str(tmp_path / "none.csv"), "--reference", str(reference),
                               "--day", JAN7]) == 2


def test_reference_paths_follow_the_pipeline_layout():
    """eval_part_dir / eval_episode_path and seeded_result_name of scripts.pipeline.common, under any results root."""

    paths = shadow_replay.reference_episode_paths("/results-root")
    assert paths == {"b0": "/results-root/b0_test_episodes.csv",
                     "b3": "/results-root/eval_parts/retrospective_holdout/b3/episodes_day_2025-01-07.csv"}


# --- import discipline ---------------------------------------------------------------------------------------------

FROZEN_IMPORTS = {
    "shadow_replay.py": {
        "cvar_mm.gym.helpers.fast_rollout": {"fast_simulate"},
        "cvar_mm.gym.notebook_support": {"evaluate_formula_rl_per_day", "stats_dict_to_frame"},
        "cvar_mm.gym.data_loader": {"load_single_day", "load_single_day_with_features"},
        "scripts.pipeline.common": {"build_config", "formula_kwargs_from_config", "eval_episode_path",
                                    "seeded_result_name", "parse_seed_list"},
    },
    "analysis.py": {
        "cvar_mm.gym.calibration": {"_sigma_from_day"},
        "cvar_mm.gym.data_loader": {"load_single_day_calibration_arrays"},
        "hummingbot_adapter.telemetry": {"FILL_HEADER", "HEALTH_HEADER", "KINDS", "ORDER_HEADER", "SNAPSHOT_HEADER",
                                         "SUMS_NAME", "TRADE_HEADER", "headers"},
    },
}
FROZEN_NAMES = {"fast_simulate", "evaluate_formula_rl_per_day", "stats_dict_to_frame", "load_single_day",
                "load_single_day_with_features", "load_single_day_calibration_arrays", "build_config",
                "formula_kwargs_from_config", "_sigma_from_day", "generate_trajectory_stats", "freeze_vecnorm",
                "build_formula_replay_env", "avellaneda_stoikov_quotes", "calibrate_from_days", "eval_part_dir",
                "eval_episode_path", "seeded_result_name", "read_market_data_csv", "extract_market_feature_arrays"}


def _tree(name: str) -> ast.Module:
    path = REPO_ROOT / "hummingbot_adapter" / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("name", sorted(FROZEN_IMPORTS))
def test_the_new_modules_import_the_frozen_functions_and_define_none(name):
    tree = _tree(name)
    imported: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import at line {node.lineno}"
            imported.setdefault(node.module, set()).update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            assert called not in {"__import__", "import_module", "exec", "eval"}, f"line {node.lineno}: {called}"
    for module, names in FROZEN_IMPORTS[name].items():
        assert names <= imported.get(module, set()), f"{name} does not import {names} from {module}"
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    defined |= {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign) for target in node.targets
                if isinstance(target, ast.Name)}
    assert defined & FROZEN_NAMES == set(), f"{name} defines a frozen name"


def test_the_imported_functions_are_the_frozen_objects():
    from cvar_mm.gym import calibration, data_loader, notebook_support
    from cvar_mm.gym.helpers import fast_rollout
    from scripts.pipeline import common

    assert shadow_replay.fast_simulate is fast_rollout.fast_simulate
    assert shadow_replay.evaluate_formula_rl_per_day is notebook_support.evaluate_formula_rl_per_day
    assert shadow_replay.stats_dict_to_frame is notebook_support.stats_dict_to_frame
    assert shadow_replay.load_single_day is data_loader.load_single_day
    assert shadow_replay.load_single_day_with_features is data_loader.load_single_day_with_features
    assert shadow_replay.build_config is common.build_config
    assert shadow_replay.formula_kwargs_from_config is common.formula_kwargs_from_config
    assert analysis._sigma_from_day is calibration._sigma_from_day
    assert analysis.load_single_day_calibration_arrays is data_loader.load_single_day_calibration_arrays


@pytest.mark.parametrize("name", ["analysis.py", "shadow_replay.py"])
def test_no_hard_coded_windows_or_snellius_path(name):
    text = (REPO_ROOT / "hummingbot_adapter" / name).read_text(encoding="utf-8")
    assert re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text) is None
    for marker in ("/gpfs", "/home", "\\Users", "/Users"):
        assert marker not in text, marker


def test_b0_episode_header_without_artifacts(run_inputs, tmp_path, monkeypatch):
    """The addendum's header without HB_ARTIFACTS: B0 on a 300-row synthetic book writes the pipeline's header, one row
    per fill seed x rollout."""

    export_pipeline_env(monkeypatch)
    (tmp_path / "book").mkdir()
    day_file = write_book_csv(tmp_path / "book" / "synthetic_book_2025-01-01.csv",
                              synthetic_book(np.random.default_rng(7), 300))
    out = tmp_path / "out"
    assert shadow_replay.main(["run", "--model", "b0", "--day-file", str(day_file), "--date", "2025-01-01",
                               "--params", str(run_inputs["params"]), "--out-dir", str(out),
                               "--gamma-file", str(run_inputs["gamma"]), "--rollouts", "2"]) == 0
    path = out / "shadow_b0_2025-01-01_episodes.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == B0_EPISODES_HEADER
    assert ",".join(shadow_replay.EPISODE_COLUMNS) == B0_EPISODES_HEADER
    assert shadow_replay.read_episodes(path)[["Fill Seed", "Rollout"]].values.tolist() == [[42, 0], [42, 1], [43, 0],
                                                                                          [43, 1]]


# --- smoke runs on a slice of the Jan 7 2025 file (HB_ARTIFACTS) -------------------------------------------------


@pytest.fixture(scope="module")
def jan7_slice(inputs, tmp_path_factory):
    """The header and the first 2,000 rows of the Jan 7 file (its date in the name)."""

    path = tmp_path_factory.mktemp("jan7") / f"binance_book_snapshot_25_{JAN7}_DOGEUSDT_first{SLICE_ROWS}.csv"
    with open(inputs["jan7"], encoding="utf-8", newline="") as source, \
            open(path, "w", encoding="utf-8", newline="") as target:
        for index, line in enumerate(source):
            if index > SLICE_ROWS:
                break
            target.write(line)
    return path


def test_b0_shadow_replay_smoke_on_a_jan7_slice(inputs, jan7_slice, tmp_path, monkeypatch):
    """Seeds 42 and 43 x 2 rollouts with the Jan 1-6 params: 4 rows, the pipeline's header, identical over two calls,
    and equal to run_b0_fair.py's own loop on the same slice."""

    export_pipeline_env(monkeypatch)
    outs = []
    for name in ("first", "second"):
        out = tmp_path / name
        assert shadow_replay.main(["run", "--model", "b0", "--day-file", str(jan7_slice), "--date", JAN7,
                                   "--params", str(inputs["params"]), "--out-dir", str(out),
                                   "--gamma-file", str(inputs["b0_gamma"]), "--fill-seeds", "42 43",
                                   "--rollouts", "2"]) == 0
        outs.append(out)
    first, second = (out / f"shadow_b0_{JAN7}_episodes.csv" for out in outs)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").splitlines()[0] == B0_EPISODES_HEADER
    frame = shadow_replay.read_episodes(first)
    assert frame[["Day", "Fill Seed", "Rollout"]].values.tolist() == [[JAN7, 42, 0], [JAN7, 42, 1], [JAN7, 43, 0],
                                                                       [JAN7, 43, 1]]
    record = json.loads((outs[0] / f"shadow_b0_{JAN7}.json").read_text(encoding="utf-8"))
    params = json.loads(inputs["params"].read_text(encoding="utf-8"))
    assert record["inputs"]["day_file"]["sha256"] == file_sha256(jan7_slice)
    assert record["inputs"]["params"]["kind"] == "formula_train_params"
    assert record["inputs"]["gamma_file"]["gamma"] == B0_GAMMA
    assert (record["parameters"]["sigma"], record["parameters"]["A"], record["parameters"]["kappa"]) == (
        params["sigma_train"], params["A_train"], params["kappa_train"])
    assert record["parameters"]["Q_MAX"] == 10 and record["pipeline_settings"] is False
    assert record["episodes"] == {"file": first.name, "rows": 4, "sha256": file_sha256(first)}
    sums = (outs[0] / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ")[1] for line in sums] == sorted([first.name, f"shadow_b0_{JAN7}.json"])

    # run_b0_fair.py's test-day loop, written out here on the same slice
    from cvar_mm.gym.data_loader import load_single_day
    from cvar_mm.gym.helpers.fast_rollout import fast_simulate
    from cvar_mm.gym.notebook_support import stats_dict_to_frame

    S, dt, _ = load_single_day(str(jan7_slice))
    frames = []
    for seed in (42, 43):
        stats = fast_simulate(midprices=S, dt_array=dt, gamma=B0_GAMMA, sigma=params["sigma_train"],
                              kappa=params["kappa_train"], A=params["A_train"], terminal_time=float(dt.sum()),
                              tick_size=1e-5, Q_MAX=10, num_trajectories=2, seed=seed, use_linear_approximation=False,
                              fixed_clock_seconds=60.0)
        frames.append(stats_dict_to_frame(stats).assign(Day=JAN7, Rollout=np.arange(2), **{"Fill Seed": seed}))
    expected = pd.concat(frames, ignore_index=True)
    leading = ["Day", "Fill Seed", "Rollout"]
    expected = expected[leading + [column for column in expected.columns if column not in leading]]
    assert shadow_replay.compare_frames(frame, expected, JAN7).summary["max_abs"].max() == 0.0


def test_b3_shadow_replay_smoke_on_a_jan7_slice(inputs, jan7_slice, tmp_path, monkeypatch):
    """The B3 path (evaluate_formula_model.py's loop) on the slice: 4 rows with the same header as B0's file."""

    export_pipeline_env(monkeypatch)
    assert shadow_replay.main(["run", "--model", "b3", "--day-file", str(jan7_slice), "--date", JAN7,
                               "--params", str(inputs["params"]), "--out-dir", str(tmp_path),
                               "--model-zip", str(inputs["model"]), "--vecnorm", str(inputs["vecnorm"]),
                               "--rollouts", "2"]) == 0
    path = tmp_path / f"shadow_b3_{JAN7}_episodes.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == B0_EPISODES_HEADER
    frame = pd.read_csv(path, dtype={"Day": str})
    assert len(frame) == 4 and frame["Fill Seed"].tolist() == [42, 42, 43, 43]
    record = json.loads((tmp_path / f"shadow_b3_{JAN7}.json").read_text(encoding="utf-8"))
    assert record["inputs"]["model_zip"]["sha256"] == file_sha256(inputs["model"])
    assert record["parameters"]["formula_kwargs"]["skew_ticks_max"] == 5.0
