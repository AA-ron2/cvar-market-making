"""T6: the calibration wrapper on synthetic files (it never opens a 2026 Tardis file)."""

from __future__ import annotations

import gzip
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from hummingbot_adapter import calibrate as hb_calibrate
from hummingbot_adapter.tests.support import REPO_ROOT, synthetic_book, write_book_csv
from cvar_mm.gym import calibration, data_loader
from cvar_mm.gym.calibration import FirstPassageCalibrationConfig

ENV_FILE = REPO_ROOT / "configs" / "final_pipeline.env"
NAME = "binance_book_snapshot_25_{date}_DOGEUSDT.csv"


@pytest.fixture(autouse=True)
def _isolated_calibration_env(monkeypatch):
    """Each test starts without CALIBRATION_* values, and the values the wrapper writes into os.environ are
    undone afterwards (setenv records the original state, so teardown restores it)."""

    for key in hb_calibrate.CALIBRATION_KEYS:
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)


def _lf_env_text() -> str:
    return ENV_FILE.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _env_file(tmp_path: Path, text: str, name: str = "final_pipeline.env", newline: str = "\n") -> Path:
    path = tmp_path / name
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


def _day(tmp_path: Path, day: str, seed: int, rows: int = 30_000, gz: bool = False) -> Path:
    path = write_book_csv(tmp_path / NAME.format(date=day), synthetic_book(np.random.default_rng(seed), rows))
    if not gz:
        return path
    gz_path = path.with_name(path.name + ".gz")
    gz_path.write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    return gz_path


def test_crlf_env_file_gives_the_same_config_as_the_lf_original(tmp_path, monkeypatch):
    lf = _env_file(tmp_path, _lf_env_text(), "lf.env")
    crlf = _env_file(tmp_path, _lf_env_text(), "crlf.env", newline="\r\n")
    assert b"\r\n" in crlf.read_bytes() and b"\r" not in lf.read_bytes()
    assert hb_calibrate.read_calibration_env(lf) == hb_calibrate.read_calibration_env(crlf)
    config_lf, values_lf = hb_calibrate.calibration_config(lf)
    for key in hb_calibrate.CALIBRATION_KEYS:
        monkeypatch.delenv(key)
    config_crlf, values_crlf = hb_calibrate.calibration_config(crlf)
    assert config_crlf == config_lf and values_crlf == values_lf
    assert all("\r" not in value for value in values_crlf.values())
    assert config_lf == FirstPassageCalibrationConfig(
        tick_size=0.00001,
        depth_ticks=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0),
        censor_horizon_seconds=15.0,
        execution_proxy="best_price_reach",
        exposure_start_rule="fixed_stride",
        exposure_stride_seconds=15.0,
        min_hits_per_depth=10,
        volatility_window_seconds=600.0,
    )
    assert asdict(config_lf) == {**asdict(config_crlf)}


@pytest.mark.parametrize("key", hb_calibrate.CALIBRATION_KEYS)
def test_a_missing_calibration_key_is_refused(tmp_path, key):
    text = "\n".join(line for line in _lf_env_text().split("\n") if not line.startswith(f"export {key}="))
    env = _env_file(tmp_path, text)
    with pytest.raises(hb_calibrate.CalibrationRefused, match=key):
        hb_calibrate.read_calibration_env(env)
    commented = _env_file(tmp_path, text + f"\n# export {key}=1\n", "commented.env")
    with pytest.raises(hb_calibrate.CalibrationRefused, match=key):
        hb_calibrate.read_calibration_env(commented)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("export CALIBRATION_MIN_HITS=10", "twice"),
        ("export CALIBRATION_UNKNOWN=1", "unknown"),
        ('export CALIBRATION_MIN_HITS="$MIN_HITS"', "does not accept"),
        ("export CALIBRATION_MIN_HITS=10 # ten", "does not accept"),
        ("export CALIBRATION_MIN_HITS=", "does not accept"),
    ],
)
def test_repeated_unknown_or_shell_valued_keys_are_refused(tmp_path, line, match):
    text = _lf_env_text()
    if match != "twice":
        text = "\n".join(entry for entry in text.split("\n") if not entry.startswith("export CALIBRATION_MIN_HITS="))
    env = _env_file(tmp_path, text + "\n" + line + "\n")
    with pytest.raises(hb_calibrate.CalibrationRefused, match=match):
        hb_calibrate.read_calibration_env(env)


def test_run_mode_passes_the_loader_arrays_unchanged_to_calibrate_from_days(tmp_path, monkeypatch):
    days = [_day(tmp_path, "2025-01-02", 1), _day(tmp_path, "2025-01-03", 2, gz=True)]
    loaded, calls = [], []
    real_load, real_fit = data_loader.load_single_day_calibration_arrays, calibration.calibrate_from_days

    def spy_load(path):
        arrays = real_load(path)
        loaded.append((path, arrays))
        return arrays

    def spy_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(data_loader, "load_single_day_calibration_arrays", spy_load)
    monkeypatch.setattr(calibration, "calibrate_from_days", spy_fit)
    payload = hb_calibrate.calibrate(days, _env_file(tmp_path, _lf_env_text()))

    csvs = [str(days[0]), str(days[1].with_name(days[1].name[: -len(".gz")]))]
    assert [path for path, _ in loaded] == csvs
    assert len(calls) == 1
    (mids, dts), kwargs = calls[0]
    assert set(kwargs) == {"config", "daily_best_bid", "daily_best_ask"}
    for index, (_, (mid, dt, _, bid, ask)) in enumerate(loaded):
        assert mids[index] is mid and dts[index] is dt
        assert kwargs["daily_best_bid"][index] is bid and kwargs["daily_best_ask"][index] is ask
    assert len(mids) == len(dts) == len(kwargs["daily_best_bid"]) == len(kwargs["daily_best_ask"]) == 2
    assert kwargs["config"] == hb_calibrate.calibration_config(_env_file(tmp_path, _lf_env_text(), "again.env"))[0]

    result = real_fit(mids, dts, config=kwargs["config"], daily_best_bid=kwargs["daily_best_bid"],
                      daily_best_ask=kwargs["daily_best_ask"])
    assert (payload["sigma"], payload["A"], payload["kappa"]) == (result.sigma, result.A, result.kappa)
    assert payload["pooled_fit"] == asdict(result.diagnostics.pooled_fit)
    assert payload["r_squared_log"] == result.diagnostics.pooled_fit.r_squared_log
    assert payload["pooled_curve"]["hits"] == result.diagnostics.pooled_curve.hits.tolist()
    assert payload["bid_curve"]["side"] == "bid" and payload["ask_curve"]["side"] == "ask"
    assert payload["calibration_config"] == asdict(kwargs["config"])
    for info, (_, (mid, *_)) in zip(payload["files"], loaded):
        assert info["rows"] == len(mid)
        assert (info["mid_mean"], info["mid_min"], info["mid_max"]) == (np.mean(mid), np.min(mid), np.max(mid))
    assert payload["files"][0]["input_sha256"] == payload["files"][0]["csv_sha256"] == hb_calibrate.file_sha256(days[0])
    assert payload["files"][1]["input_sha256"] == hb_calibrate.file_sha256(days[1])
    assert payload["files"][1]["csv_sha256"] == hb_calibrate.file_sha256(csvs[1])
    assert len(payload["protocol_commit"]) >= 7 and len(payload["git_commit"]) == 40
    json.dumps(payload)


def test_the_decompressed_csv_is_reused_only_if_identical(tmp_path):
    gz_path = _day(tmp_path, "2025-01-02", 3, rows=2_000, gz=True)
    first = hb_calibrate.materialize_csv(gz_path)
    csv = Path(first["csv"])
    assert csv.read_bytes() == gzip.decompress(gz_path.read_bytes())
    assert hb_calibrate.materialize_csv(gz_path) == first
    csv.write_bytes(csv.read_bytes() + b"\n")
    with pytest.raises(hb_calibrate.CalibrationRefused, match="exists"):
        hb_calibrate.materialize_csv(gz_path)
    assert not list(tmp_path.glob("*.partial"))
    stale = csv.with_name(csv.name + ".partial")
    stale.write_bytes(b"interrupted")
    with pytest.raises(hb_calibrate.CalibrationRefused, match="interrupted"):
        hb_calibrate.materialize_csv(gz_path)
    assert stale.read_bytes() == b"interrupted"


def test_a_failed_decompression_leaves_no_partial_and_propagates(tmp_path):
    gz_path = _day(tmp_path, "2025-01-02", 6, rows=3_000, gz=True)
    good = gz_path.read_bytes()
    gz_path.write_bytes(good[: len(good) // 2])  # truncated download
    with pytest.raises(EOFError):
        hb_calibrate.materialize_csv(gz_path)
    assert not list(tmp_path.glob("*.partial")) and not list(tmp_path.glob("*.csv"))
    gz_path.write_bytes(good)
    assert Path(hb_calibrate.materialize_csv(gz_path)["csv"]).is_file()


def test_decompression_never_writes_into_the_repository(tmp_path, monkeypatch):
    inside = REPO_ROOT / "hummingbot_adapter" / "not_there" / (NAME.format(date="2025-01-02") + ".gz")
    with pytest.raises(hb_calibrate.CalibrationRefused, match="inside the repository"):
        hb_calibrate.materialize_csv(inside)
    assert not inside.parent.exists()
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(hb_calibrate.CalibrationRefused, match="inside the repository"):
        hb_calibrate.materialize_csv(Path(NAME.format(date="2025-01-02") + ".gz"))

    # calibrate() refuses before it opens anything, in a repository whose PROTOCOL.md is committed
    repo = tmp_path / "repo"
    (repo / "hummingbot_adapter").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "hummingbot_adapter" / "PROTOCOL.md").write_bytes(b"# protocol\n")
    _git(repo, "add", "hummingbot_adapter/PROTOCOL.md")
    _git(repo, "commit", "-q", "-m", "protocol")
    gz_path = _day(repo, "2025-01-02", 7, rows=500, gz=True)
    opened = []
    monkeypatch.setattr(data_loader, "load_single_day_calibration_arrays", lambda path: opened.append(path))
    with pytest.raises(hb_calibrate.CalibrationRefused, match="inside the repository"):
        hb_calibrate.calibrate([gz_path], _env_file(tmp_path, _lf_env_text()), repo)
    assert opened == [] and not list(repo.glob("*.csv")) and not list(repo.glob("*.partial"))


def test_out_folder_is_checked_before_anything_is_opened_and_a_write_failure_exits_3(tmp_path, monkeypatch, capsys):
    day = _day(tmp_path, "2025-01-02", 8)
    env = _env_file(tmp_path, _lf_env_text())
    opened = []
    real_load = data_loader.load_single_day_calibration_arrays

    def spy_load(path):
        opened.append(path)
        return real_load(path)

    monkeypatch.setattr(data_loader, "load_single_day_calibration_arrays", spy_load)
    missing_folder = tmp_path / "no_such_folder" / "hb_calibration.json"
    assert hb_calibrate.main(["--inputs", str(day), "--env-file", str(env), "--out", str(missing_folder)]) == 2
    assert "does not exist" in capsys.readouterr().err and opened == []

    def failing_write(payload, out):
        raise PermissionError(13, "Permission denied", str(out))

    monkeypatch.setattr(hb_calibrate, "write_json", failing_write)
    out = tmp_path / "hb_calibration.json"
    assert hb_calibrate.main(["--inputs", str(day), "--env-file", str(env), "--out", str(out)]) == 3
    captured = capsys.readouterr()
    assert "WRITE FAILED" in captured.err and "sigma" in captured.out and opened == [str(day)]


def test_git_output_with_utf8_commit_subjects_does_not_crash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "x.txt").write_bytes(b"x\n")
    _git(repo, "add", "x.txt")
    subject = "B1⁺ control, rho ρ, quote ”x”"  # bytes 0x81 and 0x9d fail under cp1252
    _git(repo, "commit", "-q", "-m", subject)
    assert hb_calibrate._git(repo, "log", "-1", "--format=%s") == subject


def test_a_missing_input_is_refused_before_any_is_opened(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(hb_calibrate, "materialize_csv", lambda *args: opened.append(args))
    present = _day(tmp_path, "2025-01-02", 4, rows=500)
    with pytest.raises(hb_calibrate.CalibrationRefused, match="no such input"):
        hb_calibrate.calibrate([present, tmp_path / NAME.format(date="2025-01-03")], _env_file(tmp_path, _lf_env_text()))
    assert opened == []


def test_a_run_day_file_is_refused_before_anything_is_opened(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(data_loader, "load_single_day_calibration_arrays", lambda path: opened.append(path))
    monkeypatch.setattr(hb_calibrate, "materialize_csv", lambda *args: opened.append(args))
    env = _env_file(tmp_path, _lf_env_text())
    early = tmp_path / NAME.format(date="2026-10-19")
    for day in ("2026-10-20", "2026-10-21", "2027-01-01"):
        late = tmp_path / (NAME.format(date=day) + ".gz")
        late.write_bytes(b"not opened")
        with pytest.raises(hb_calibrate.CalibrationRefused, match="run day"):
            hb_calibrate.calibrate([late], env)
        with pytest.raises(hb_calibrate.CalibrationRefused, match="run day"):
            hb_calibrate.calibrate([early, late], env)
    assert opened == []
    for bad in ("binance_book_snapshot_25_DOGEUSDT.csv", "x_2026-09-01_2026-10-01.csv", "x_2026-13-01.csv"):
        with pytest.raises(hb_calibrate.CalibrationRefused):
            hb_calibrate.calibrate([tmp_path / bad], env)
    with pytest.raises(hb_calibrate.CalibrationRefused, match="same date"):
        hb_calibrate.calibrate([early, early.with_name(early.name + ".gz")], env)
    assert opened == []


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout


def test_protocol_must_be_committed_and_unmodified(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    protocol = repo / "hummingbot_adapter" / "PROTOCOL.md"
    protocol.parent.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "README.md").write_bytes(b"x\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "first")
    protocol.write_bytes(b"# protocol\n")
    with pytest.raises(hb_calibrate.CalibrationRefused, match="not committed"):
        hb_calibrate.require_committed_protocol(repo)
    _git(repo, "add", "hummingbot_adapter/PROTOCOL.md")
    _git(repo, "commit", "-q", "-m", "protocol")
    commit = hb_calibrate.require_committed_protocol(repo)
    assert commit == _git(repo, "log", "-1", "--format=%h").strip()

    protocol.write_bytes(b"# protocol, edited\n")
    with pytest.raises(hb_calibrate.CalibrationRefused, match="differs"):
        hb_calibrate.require_committed_protocol(repo)
    opened = []
    monkeypatch.setattr(data_loader, "load_single_day_calibration_arrays", lambda path: opened.append(path))
    with pytest.raises(hb_calibrate.CalibrationRefused, match="differs"):
        hb_calibrate.calibrate([tmp_path / NAME.format(date="2025-01-02")], _env_file(tmp_path, _lf_env_text()), repo)
    assert opened == []
    protocol.unlink()
    with pytest.raises(hb_calibrate.CalibrationRefused, match="differs"):
        hb_calibrate.require_committed_protocol(repo)
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    with pytest.raises(hb_calibrate.CalibrationRefused, match="git"):
        hb_calibrate.require_committed_protocol(not_a_repo)


def test_this_branch_protocol_is_committed_and_unmodified():
    assert hb_calibrate.require_committed_protocol() == subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", "hummingbot_adapter/PROTOCOL.md"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_verify_mode_compares_with_the_frozen_values(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index, day in enumerate(hb_calibrate.FROZEN_DAYS):
        _day(data_dir, day, 100 + index, rows=12_000)
    env = _env_file(tmp_path, _lf_env_text())
    reference = hb_calibrate.calibrate([data_dir / NAME.format(date=day) for day in hb_calibrate.FROZEN_DAYS], env)
    expected = tmp_path / "formula_train_params.json"
    expected.write_text(json.dumps({
        "sigma_train": reference["sigma"], "A_train": reference["A"], "kappa_train": reference["kappa"],
    }), encoding="utf-8")
    args = ["--verify-frozen", "--data-dir", str(data_dir), "--expected", str(expected), "--env-file", str(env)]

    assert hb_calibrate.main([*args, "--out", str(tmp_path / "refused.json")]) == 2  # not the frozen file
    assert "not the frozen" in capsys.readouterr().err
    monkeypatch.setattr(hb_calibrate, "FROZEN_PARAMS_SHA256", hb_calibrate.file_sha256(expected))
    assert hb_calibrate.main([*args, "--out", str(tmp_path / "pass.json")]) == 0
    printed = capsys.readouterr().out
    assert "verify PASSED" in printed and "relative difference 0.0" in printed
    verify = json.loads((tmp_path / "pass.json").read_text(encoding="utf-8"))["verify"]
    assert verify["passed"] and all(item["relative_difference"] == 0.0 for item in verify["comparisons"].values())
    assert hb_calibrate.main([*args, "--out", str(tmp_path / "pass.json")]) == 2  # never overwritten

    assert hb_calibrate.VERIFY_TOLERANCE == 1e-12
    for name, key, factor, code in (
        ("fail_1e-9", "sigma", 1 + 1e-9, 1),
        ("fail_2e-12", "kappa", 1 + 2e-12, 1),  # just above the bound
        ("pass_5e-13", "A", 1 + 5e-13, 0),  # just below it
    ):
        values = {"sigma_train": reference["sigma"], "A_train": reference["A"], "kappa_train": reference["kappa"]}
        values[f"{key}_train"] *= factor
        expected.write_text(json.dumps(values), encoding="utf-8")
        monkeypatch.setattr(hb_calibrate, "FROZEN_PARAMS_SHA256", hb_calibrate.file_sha256(expected))
        out = tmp_path / f"{name}.json"
        assert hb_calibrate.main([*args, "--out", str(out)]) == code, name
        printed = capsys.readouterr().out
        assert ("verify PASSED" if code == 0 else "verify FAILED") in printed
        assert json.loads(out.read_text(encoding="utf-8"))["verify"]["passed"] is (code == 0)


def test_run_mode_cli_writes_lf_json(tmp_path, capsys):
    day = _day(tmp_path, "2025-01-02", 5)
    out = tmp_path / "hb_calibration.json"
    assert hb_calibrate.main(["--inputs", str(day), "--env-file", str(_env_file(tmp_path, _lf_env_text())),
                              "--out", str(out)]) == 0
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    payload = json.loads(raw)
    assert {"sigma", "A", "kappa", "pooled_fit", "pooled_curve", "bid_curve", "ask_curve", "files",
            "calibration_config", "git_commit", "created_utc"} <= set(payload)
    assert "sigma" in capsys.readouterr().out
