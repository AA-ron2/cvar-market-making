"""WO-HB2 deliverable 3 without Hummingbot: bot_config (BotSettings against PROTOCOL.md, check_settings,
from_mapping, config_hash, bot_dir, verify_inputs); hb_forward_bot.py by AST (it imports hummingbot, so it is
never imported here); the deploy files; the import discipline of the new runtime modules; and monitor
(ClockChecker with fake clocks and a fake fetch, ReconnectWatcher, memory_rss_mb).

The PROTOCOL.md values below are typed in from the pre-registered text, with their sections, and never read from
bot_config. verify_inputs runs on the real artifacts only when HB_ARTIFACTS is set (conftest.py); its refusals
also run without them, on temporary copies of the committed calibration and of the §2 gamma file rebuilt from
its bytes (whose SHA-256 is checked against PROTOCOL §2).
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import http.client
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path, PureWindowsPath

import pytest

from hummingbot_adapter import bot_config, monitor
from hummingbot_adapter.bot_config import (
    BotSettings,
    check_settings,
    config_hash,
    from_mapping,
    protocol_default,
    verify_inputs,
)
from hummingbot_adapter.execution import StepClock, monotonic_raw_ns
from hummingbot_adapter.monitor import ClockChecker, ReconnectWatcher, memory_rss_mb
from hummingbot_adapter.tests.support import REPO_ROOT

ADAPTER = REPO_ROOT / "hummingbot_adapter"
DEPLOY = ADAPTER / "deploy"
SCRIPT_CONFIGS = DEPLOY / "conf" / "scripts"
BOT_MODULE = ADAPTER / "hb_forward_bot.py"
COMMITTED_CALIBRATION = ADAPTER / "config" / "hb_calibration.json"

# PROTOCOL.md, typed in from the pre-registered text.
PROTOCOL_MD_CONSTANTS = {
    "exchange": "binance_paper_trade",  # §3: the connector
    "trading_pair": "DOGE-USDT",  # §3: Binance Spot DOGE-USDT
    "tick_size": 0.00001,  # §3: price tick 0.00001 USDT
    "paper_doge_balance": 1_000_000,  # §3: paper balances DOGE 1,000,000
    "paper_usdt_balance": 100_000,  # §3: and USDT 100,000
    "fee_rate": 0.001,  # §3: 0.1 % for maker and taker orders
    "gamma_b0": 0.8908584807525282,  # §2: B0
    "order_amount_doge": 100,  # §5.1: 1 unit = 100 DOGE (§5.3: 100 DOGE per order)
    "q_max": 10,  # §5.1: q_max = 10 units
    "book_levels": 25,  # §5.1: the top-25 book
    "sampling_interval_s": 0.1,  # §5.1: every 0.1 s clock tick
    "feature_window": 100,  # §5.2: over the last 100 steps
    "imbalance_depth": 5,  # §5.2: lob_imbalance over the top 5 levels per side
    "day_seconds": 86_400,  # §5.2: T = 86,400 s
    "gamma_min": 0.01,  # §5.2: gamma log-linearly in [0.01, 0.9]
    "gamma_max": 0.9,  # §5.2
    "skew_ticks_max": 5.0,  # §5.2: skew = 5 ticks x u
    "requote_min_ticks": 1,  # §5.3: the target moved by at least 1 tick
    "requote_min_age_s": 1.0,  # §5.3: and the resting order is at least 1 s old
    "rate_window_s": 10,  # §5.3: 100 per rolling 10 s
    "rate_max_per_window": 100,  # §5.3
    "rate_max_per_day": 200_000,  # §5.3: 200,000 per UTC day
    "close_seconds_before_midnight": 10,  # §5.5: the close at 23:59:50 UTC
}
PROTOCOL_MD_THRESHOLDS = {  # §7
    "halt_q_max": 10,  # |q_ledger| > q_max: halt
    "residual_max_doge": 1.0,  # |reconciliation residual| > 1 DOGE: halt
    "stale_pause_s": 30.0,  # no top-25 change for 30 s: cancel and pause
    "stale_halt_s": 300.0,  # halt after 5 min
    "reject_max": 10,  # more than 10 rejects within 60 s: halt
    "reject_window_s": 60.0,
    "drawdown_limit_usdt": 40.0,  # daily drawdown of gross equity > 40 USDT: close, then pause
    "clock_warn_s": 0.5,  # clock offset > 0.5 s: warn
    "clock_pause_s": 2.0,  # > 2 s: pause quoting
    "clock_check_interval_s": 3_600.0,  # GET /api/v3/time, hourly
}
PROTOCOL_MD = {**PROTOCOL_MD_CONSTANTS, **PROTOCOL_MD_THRESHOLDS}

MODEL_SHA256 = "05ee4c97524456a6445f54cfe0a4bf353ccd539c20060f1176b83455cebe4e53"  # §2: models/ppo_b3_doge.zip
VECNORM_SHA256 = "d2be858e002d3b1e5a32bbe65822de3a4686cdbb91a81b8636241460a9b009b9"  # §2: models/vecnorm_b3.pkl
B0_GAMMA_SHA256 = "34cbb018dd15a9112f8c6f4f2cbb2874bea4efefbc1de3ad7c1895b6abe226c4"  # §2: b0_gamma_fixed.txt
B0_GAMMA_BYTES = b"0.8908584807525282"  # that file's bytes (test_the_pinned_hashes_... checks its SHA-256)
STALE_GAMMA = 0.975143028728294  # §2: the stale repo copy, never used
IMAGE = "hummingbot-local:2.17.0-9af100d"  # §2: built locally from v2.17.0 (9af100d)
# RUN_LOG.md 2026-10-19T21:26:01Z: the 2026-10-01 calibration (PROTOCOL §4)
CALIBRATION = {"sigma": 7.556448594347906e-06, "A": 0.09500929388592372, "kappa": 41813.40928239842}

# A different valid value for every BotSettings field (each passes BotSettings' own checks).
CHANGED = {
    "exchange": "binance",  # the real exchange
    "trading_pair": "BTC-USDT",
    "tick_size": 0.0001,
    "paper_doge_balance": 2_000_000,
    "paper_usdt_balance": 50_000,
    "fee_rate": 0.00075,
    "gamma_b0": STALE_GAMMA,
    "order_amount_doge": 200,
    "q_max": 5,
    "book_levels": 20,
    "sampling_interval_s": 1.0,
    "feature_window": 50,
    "imbalance_depth": 3,
    "day_seconds": 3_600,
    "gamma_min": 0.02,
    "gamma_max": 0.8,
    "skew_ticks_max": 4.0,
    "requote_min_ticks": 2,
    "requote_min_age_s": 0.5,
    "rate_window_s": 20,
    "rate_max_per_window": 50,
    "rate_max_per_day": 100_000,
    "close_seconds_before_midnight": 30,
    # the §7 thresholds, made more sensitive as a smoke run may (a longer reject window counts more rejects)
    "halt_q_max": 1,
    "residual_max_doge": 0.5,
    "stale_pause_s": 10.0,
    "stale_halt_s": 60.0,
    "reject_max": 5,
    "reject_window_s": 120.0,
    "drawdown_limit_usdt": 0.0001,
    "clock_warn_s": 0.25,
    "clock_pause_s": 1.0,
    "clock_check_interval_s": 600.0,
    # operation and identity
    "artifacts_dir": "/elsewhere/artifacts",
    "repo_dir": "/elsewhere/repo",
    "data_root": "/elsewhere/hb_data",
    "save_interval_s": 0.5,
    "heartbeat_interval_s": 30.0,
    "model": "B0",
    "mode": "smoke",
}
HB_CONFIG_FIELDS = {"script_file_name", "controllers_config"}  # StrategyV2ConfigBase's own fields
SETTINGS_FIELDS = {field.name: field for field in dataclasses.fields(BotSettings)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flip_digit(path: Path, after: bytes) -> None:
    """Change one byte: the first digit after `after` becomes its neighbour (0<->1, 2<->3, ...)."""

    data = bytearray(path.read_bytes())
    start = data.index(after) + len(after)
    index = next(i for i in range(start, len(data)) if 0x30 <= data[i] <= 0x39)
    data[index] ^= 0x01
    path.write_bytes(bytes(data))


# --- BotSettings against PROTOCOL.md ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PROTOCOL_MD))
def test_protocol_default_equals_protocol_md(name):
    assert protocol_default(name) == PROTOCOL_MD[name]
    for model in ("B0", "B3"):
        for mode in ("production", "smoke"):
            assert getattr(BotSettings(model=model, mode=mode), name) == PROTOCOL_MD[name]


def test_settings_fields_are_the_protocol_constants_the_section7_thresholds_and_operation():
    """Only the §7 thresholds may differ in a smoke run, so the split must be PROTOCOL's; every field is in
    exactly one group (a new field must be classified)."""

    assert set(bot_config.PROTOCOL_CONSTANTS) == set(PROTOCOL_MD_CONSTANTS)
    assert set(bot_config.THRESHOLDS) == set(PROTOCOL_MD_THRESHOLDS)
    groups = [("model", "mode"), bot_config.PROTOCOL_CONSTANTS, bot_config.THRESHOLDS, bot_config.OPERATION]
    assert sorted(name for group in groups for name in group) == sorted(SETTINGS_FIELDS)
    assert set(CHANGED) == set(SETTINGS_FIELDS)


def test_every_default_has_its_declared_type_and_model_and_mode_have_none():
    for name, field in SETTINGS_FIELDS.items():
        if name in ("model", "mode"):
            assert field.default is dataclasses.MISSING, name
        else:
            assert type(field.default).__name__ == field.type, name


def test_state_and_telemetry_are_saved_at_least_once_per_second():
    """WO-HB2: the state is written and the telemetry flushed at least once per second (one interval for both)."""

    assert protocol_default("save_interval_s") <= 1.0


@pytest.mark.parametrize("model, mode", [("B1", "production"), ("B2", "smoke"), ("b3", "production"),
                                         ("B3", "paper"), ("B0", "")])
def test_settings_refuse_another_model_or_mode(model, mode):
    """PROTOCOL §2: B0 and B3 only."""

    with pytest.raises(ValueError):
        BotSettings(model=model, mode=mode)


# --- check_settings ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["B0", "B3"])
@pytest.mark.parametrize("mode", ["production", "smoke"])
def test_check_settings_accepts_the_protocol(model, mode):
    assert check_settings(BotSettings(model=model, mode=mode)) == []


@pytest.mark.parametrize("name", sorted(PROTOCOL_MD))
def test_production_refuses_any_changed_protocol_constant_or_threshold(name):
    settings = dataclasses.replace(BotSettings(model="B3", mode="production"), **{name: CHANGED[name]})
    with pytest.raises(ValueError, match=re.escape(f"{name}=")):
        check_settings(settings)


@pytest.mark.parametrize("name", sorted(PROTOCOL_MD_THRESHOLDS))
def test_smoke_returns_a_changed_section7_threshold(name):
    settings = dataclasses.replace(BotSettings(model="B3", mode="smoke"), **{name: CHANGED[name]})
    assert check_settings(settings) == [name]


@pytest.mark.parametrize("name", sorted(PROTOCOL_MD_THRESHOLDS))
def test_smoke_refuses_a_section7_threshold_made_less_sensitive(name):
    """WO-HB2: smoke mode may LOWER the §7 thresholds only; for the reject window, 'more sensitive' is longer."""

    default = getattr(BotSettings(model="B3", mode="smoke"), name)
    laxer = default * 0.5 if name == "reject_window_s" else default * 2
    if name == "stale_pause_s":
        laxer = 100.0  # still below stale_halt_s (300 s), so only the sensitivity rule can refuse it
    if isinstance(default, int):
        laxer = int(laxer)
    settings = dataclasses.replace(BotSettings(model="B3", mode="smoke"), **{name: laxer})
    with pytest.raises(ValueError, match="more sensitive"):
        check_settings(settings)


def test_the_state_and_telemetry_are_saved_at_least_once_per_second_in_every_mode():
    for mode in ("production", "smoke"):
        assert check_settings(dataclasses.replace(BotSettings(model="B0", mode=mode), save_interval_s=0.5)) == []
        with pytest.raises(ValueError, match="once per second"):
            check_settings(dataclasses.replace(BotSettings(model="B0", mode=mode), save_interval_s=1.5))


@pytest.mark.parametrize("name", sorted(PROTOCOL_MD_CONSTANTS))
def test_smoke_refuses_a_changed_protocol_constant(name):
    settings = dataclasses.replace(BotSettings(model="B3", mode="smoke"), **{name: CHANGED[name]})
    with pytest.raises(ValueError, match=re.escape(f"{name}=")):
        check_settings(settings)


# --- from_mapping ----------------------------------------------------------------------------------------------


def test_from_mapping_drops_hummingbots_own_fields():
    values = {"script_file_name": "hb_forward_bot.py", "controllers_config": [], "model": "B0", "mode": "production"}
    assert from_mapping(values) == BotSettings(model="B0", mode="production")


@pytest.mark.parametrize("key", ["tick_sise", "markets", "candles_config", "Model"])
def test_from_mapping_refuses_unknown_keys(key):
    with pytest.raises(ValueError, match=key):
        from_mapping({"model": "B3", "mode": "production", key: 1})


def test_from_mapping_coerces_numbers_to_the_declared_types():
    """An int field written as 10.0 and a float field written as 40 (as a yml may write them) come back as the
    declared types, so the settings and their hash equal the defaults'."""

    values: dict[str, object] = {"model": "B3", "mode": "production"}
    for name, field in SETTINGS_FIELDS.items():
        if field.type == "int":
            values[name] = float(field.default)
        elif field.type == "float" and float(field.default).is_integer():
            values[name] = int(field.default)
    assert any(SETTINGS_FIELDS[name].type == "int" and type(value) is float for name, value in values.items())
    assert any(SETTINGS_FIELDS[name].type == "float" and type(value) is int for name, value in values.items())
    settings = from_mapping(values)
    defaults = BotSettings(model="B3", mode="production")
    assert settings == defaults
    for name, field in SETTINGS_FIELDS.items():
        assert type(getattr(settings, name)).__name__ == field.type, name
    assert config_hash(settings) == config_hash(defaults)


@pytest.mark.parametrize("name, value", [
    ("q_max", 10.5),  # not an integer: refused, never truncated
    ("q_max", True),
    ("q_max", math.nan),
    ("drawdown_limit_usdt", math.nan),
    ("drawdown_limit_usdt", math.inf),
    ("drawdown_limit_usdt", -math.inf),
    ("drawdown_limit_usdt", -40.0),
    ("tick_size", 0.0),
    ("stale_pause_s", 300.0),  # must be below stale_halt_s
    ("gamma_min", 0.9),  # must be below gamma_max
])
def test_from_mapping_refuses_invalid_values(name, value):
    with pytest.raises(ValueError):
        from_mapping({"model": "B3", "mode": "smoke", name: value})


# --- config_hash and bot_dir -------------------------------------------------------------------------------------


def test_config_hash_is_stable():
    digest = config_hash(BotSettings(model="B3", mode="production"))
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert config_hash(BotSettings(model="B3", mode="production")) == digest
    assert config_hash(from_mapping({"model": "B3", "mode": "production", "drawdown_limit_usdt": 40,
                                     "q_max": 10.0})) == digest
    code = ("from hummingbot_adapter.bot_config import BotSettings, config_hash\n"
            "print(config_hash(BotSettings(model='B3', mode='production')))\n")
    for seed in ("0", "4242"):  # another process and another str hash seed
        env = {**os.environ, "PYTHONHASHSEED": seed,
               "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == digest


@pytest.mark.parametrize("name", sorted(CHANGED))
def test_config_hash_changes_with_every_field(name):
    base = BotSettings(model="B3", mode="production")
    assert config_hash(dataclasses.replace(base, **{name: CHANGED[name]})) != config_hash(base)


@pytest.mark.parametrize("model, bot", [("B0", "hb_b0"), ("B3", "hb_b3")])
def test_bot_dir_is_under_smoke_only_in_smoke_mode(tmp_path, model, bot):
    production = BotSettings(model=model, mode="production", data_root=str(tmp_path))
    smoke = BotSettings(model=model, mode="smoke", data_root=str(tmp_path))
    assert production.bot == smoke.bot == bot
    assert production.bot_dir == tmp_path / bot
    assert smoke.bot_dir == tmp_path / "smoke" / bot
    assert BotSettings(model=model, mode="production").bot_dir == Path("/hb_data") / bot


# --- verify_inputs -----------------------------------------------------------------------------------------------


def test_the_pinned_hashes_are_protocols_and_the_committed_calibrations():
    assert (bot_config.MODEL_SHA256, bot_config.VECNORM_SHA256, bot_config.B0_GAMMA_SHA256) == (
        MODEL_SHA256, VECNORM_SHA256, B0_GAMMA_SHA256)
    assert hashlib.sha256(B0_GAMMA_BYTES).hexdigest() == B0_GAMMA_SHA256  # the rebuilt file is the §2 file
    assert float(B0_GAMMA_BYTES) == PROTOCOL_MD["gamma_b0"]
    assert bot_config.CALIBRATION_SHA256 == _sha256(COMMITTED_CALIBRATION)
    committed = json.loads(COMMITTED_CALIBRATION.read_text(encoding="utf-8"))
    assert {key: committed[key] for key in CALIBRATION} == CALIBRATION


def _json_leaf_differences(a, b, path=""):
    """The paths of the leaves where two JSON values of the same structure differ."""

    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        return [found for key in a for found in _json_leaf_differences(a[key], b[key], f"{path}/{key}")]
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        return [found for i, (x, y) in enumerate(zip(a, b)) for found in _json_leaf_differences(x, y, f"{path}/{i}")]
    return [] if a == b else [path]


@pytest.mark.parametrize("model", ["B0", "B3"])
def test_verify_inputs_accepts_the_frozen_inputs(hb_artifacts, tmp_path, model):
    """The run's hb_calibration.json (HB_ARTIFACTS) has CALIBRATION_SHA256_RUN and differs from the published copy
    only in two local paths; verify_inputs accepts the published copy beside the run's other inputs."""

    run_calibration = hb_artifacts / "hb_calibration.json"
    assert _sha256(run_calibration) == bot_config.CALIBRATION_SHA256_RUN
    run_values = json.loads(run_calibration.read_text(encoding="utf-8"))
    published = json.loads(COMMITTED_CALIBRATION.read_text(encoding="utf-8"))
    assert _json_leaf_differences(run_values, published) == ["/files/0/input", "/files/0/csv"]
    for key in ("input", "csv"):
        assert PureWindowsPath(run_values["files"][0][key]).name == PureWindowsPath(published["files"][0][key]).name

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    shutil.copyfile(COMMITTED_CALIBRATION, artifacts / "hb_calibration.json")
    for name in ("b3_policy.npz", "b3_policy.json", "b0_gamma_fixed.txt"):
        shutil.copyfile(hb_artifacts / name, artifacts / name)
    settings = BotSettings(model=model, mode="production", artifacts_dir=str(artifacts), repo_dir=str(REPO_ROOT))
    inputs = verify_inputs(settings)
    assert (inputs.sigma, inputs.kappa, inputs.A) == (CALIBRATION["sigma"], CALIBRATION["kappa"], CALIBRATION["A"])
    assert inputs.calibration_sha256 == _sha256(COMMITTED_CALIBRATION) == _sha256(artifacts / "hb_calibration.json")
    if model == "B3":
        npz = artifacts / "b3_policy.npz"
        sidecar = json.loads((artifacts / "b3_policy.json").read_text(encoding="utf-8"))
        assert inputs.policy_path == npz
        assert inputs.policy_sha256 == _sha256(npz) == sidecar["output"]["sha256"]
        assert (sidecar["inputs"]["model"]["sha256"], sidecar["inputs"]["vecnorm"]["sha256"]) == (
            MODEL_SHA256, VECNORM_SHA256)
        assert inputs.gamma_file_sha256 is None
    else:
        assert inputs.gamma_file_sha256 == B0_GAMMA_SHA256 == _sha256(artifacts / "b0_gamma_fixed.txt")
        assert inputs.policy_path is None and inputs.policy_sha256 is None


@pytest.fixture
def frozen_dirs(tmp_path):
    """Copies of the frozen inputs that need no HB_ARTIFACTS: an artifacts folder with the committed calibration
    and the §2 gamma file (rebuilt from its bytes), and a repo folder with the committed calibration."""

    artifacts = tmp_path / "artifacts"
    repo = tmp_path / "repo"
    committed = repo / "hummingbot_adapter" / "config" / "hb_calibration.json"
    artifacts.mkdir()
    committed.parent.mkdir(parents=True)
    shutil.copyfile(COMMITTED_CALIBRATION, artifacts / "hb_calibration.json")
    shutil.copyfile(COMMITTED_CALIBRATION, committed)
    (artifacts / "b0_gamma_fixed.txt").write_bytes(B0_GAMMA_BYTES)
    return artifacts, repo


def _settings(model: str, artifacts: Path, repo: Path, **changes) -> BotSettings:
    return BotSettings(model=model, mode="production", artifacts_dir=str(artifacts), repo_dir=str(repo), **changes)


def test_verify_inputs_accepts_b0_on_copies_of_the_frozen_files(frozen_dirs):
    """The control for the refusals below: the unchanged copies pass."""

    inputs = verify_inputs(_settings("B0", *frozen_dirs))
    assert (inputs.sigma, inputs.kappa, inputs.A) == (CALIBRATION["sigma"], CALIBRATION["kappa"], CALIBRATION["A"])
    assert inputs.calibration_sha256 == _sha256(COMMITTED_CALIBRATION)
    assert inputs.gamma_file_sha256 == B0_GAMMA_SHA256
    assert inputs.policy_path is None and inputs.policy_sha256 is None


@pytest.mark.parametrize("which", ["artifacts", "committed"])
def test_verify_inputs_refuses_a_calibration_with_one_byte_changed(frozen_dirs, which):
    artifacts, repo = frozen_dirs
    target = artifacts / "hb_calibration.json" if which == "artifacts" else (
        repo / "hummingbot_adapter" / "config" / "hb_calibration.json")
    _flip_digit(target, b'"sigma": ')
    changed = json.loads(target.read_text(encoding="utf-8"))  # still a plausible calibration
    assert changed["sigma"] != CALIBRATION["sigma"] and changed["kappa"] == CALIBRATION["kappa"]
    with pytest.raises(ValueError, match="hb_calibration.json"):
        verify_inputs(_settings("B0", artifacts, repo))


@pytest.mark.parametrize("content", [b"0.8908584807525283", b"0.8908584807525282\n", repr(STALE_GAMMA).encode()],
                         ids=["one_byte_changed", "trailing_newline", "stale_repo_copy"])
def test_verify_inputs_refuses_another_gamma_file(frozen_dirs, content):
    artifacts, repo = frozen_dirs
    (artifacts / "b0_gamma_fixed.txt").write_bytes(content)
    with pytest.raises(ValueError, match="b0_gamma_fixed.txt"):
        verify_inputs(_settings("B0", artifacts, repo))


@pytest.mark.parametrize("gamma", [STALE_GAMMA, math.nextafter(PROTOCOL_MD["gamma_b0"], 1.0)])
def test_verify_inputs_refuses_a_gamma_b0_that_differs_from_the_frozen_file(frozen_dirs, gamma):
    with pytest.raises(ValueError, match="gamma_b0"):
        verify_inputs(_settings("B0", *frozen_dirs, gamma_b0=gamma))


def test_verify_inputs_refuses_a_policy_that_only_its_own_sidecar_vouches_for(frozen_dirs):
    """A foreign npz with a sidecar that records its hash and the PROTOCOL §2 sources is still refused."""

    artifacts, repo = frozen_dirs
    npz = artifacts / "b3_policy.npz"
    npz.write_bytes(b"not the exported B3 actor")
    sidecar = {"output": {"sha256": _sha256(npz)},
               "inputs": {"model": {"sha256": MODEL_SHA256}, "vecnorm": {"sha256": VECNORM_SHA256}}}
    (artifacts / "b3_policy.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match=_sha256(npz)):
        verify_inputs(_settings("B3", artifacts, repo))


@pytest.fixture
def b3_copies(hb_artifacts, tmp_path):
    """Copies of the real B3 inputs (HB_ARTIFACTS needed), beside the published calibration."""

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    shutil.copyfile(COMMITTED_CALIBRATION, artifacts / "hb_calibration.json")
    for name in ("b3_policy.npz", "b3_policy.json"):
        shutil.copyfile(hb_artifacts / name, artifacts / name)
    return artifacts


@pytest.mark.parametrize("change", ["none", "npz_one_byte", "sidecar_npz_hash", "sidecar_model_hash",
                                    "sidecar_vecnorm_hash"])
def test_verify_inputs_refuses_a_b3_hash_mismatch(b3_copies, change):
    npz, sidecar_path = b3_copies / "b3_policy.npz", b3_copies / "b3_policy.json"
    settings = _settings("B3", b3_copies, REPO_ROOT)
    if change == "none":  # the control: the unchanged copies pass
        assert verify_inputs(settings).policy_sha256 == _sha256(npz)
        return
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    wrong = "0" * 64
    if change == "npz_one_byte":
        data = bytearray(npz.read_bytes())
        data[len(data) // 2] ^= 0x01
        npz.write_bytes(bytes(data))
        wrong = _sha256(npz)
    elif change == "sidecar_npz_hash":
        sidecar["output"]["sha256"] = wrong
    elif change == "sidecar_model_hash":
        sidecar["inputs"]["model"]["sha256"] = wrong
    else:
        sidecar["inputs"]["vecnorm"]["sha256"] = wrong
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match=wrong):
        verify_inputs(settings)


# --- hb_forward_bot.py by AST (it imports hummingbot) -------------------------------------------------------------


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def test_shell_config_declares_every_setting_with_its_protocol_default():
    tree = _tree(BOT_MODULE)
    config = _classes(tree)["HbForwardBotConfig"]
    fields = [node for node in config.body if isinstance(node, ast.AnnAssign)]
    names = [node.target.id for node in fields]
    assert len(names) == len(set(names)), "a field is declared twice"
    assert set(names) == set(SETTINGS_FIELDS) | HB_CONFIG_FIELDS
    for node in fields:
        name = node.target.id
        if name in HB_CONFIG_FIELDS:
            continue
        assert ast.unparse(node.annotation) == SETTINGS_FIELDS[name].type, name
        if name in ("model", "mode"):
            assert node.value is None, f"{name} must have no default"
        else:
            assert node.value is not None and ast.unparse(node.value) == f"protocol_default({name!r})", name
    sources = {alias.asname or alias.name: node.module for node in tree.body if isinstance(node, ast.ImportFrom)
               for alias in node.names}
    assert sources["protocol_default"] == "hummingbot_adapter.bot_config"


def test_shell_defines_exactly_one_strategy_class_and_one_config_class():
    """Hummingbot 2.17.0 takes the first StrategyV2Base and the first config subclass among the module's names,
    imported ones included, so the module defines one of each and imports no other strategy or config class."""

    tree = _tree(BOT_MODULE)
    bases = {name: [ast.unparse(base) for base in node.bases] for name, node in _classes(tree).items()}
    assert [name for name, found in bases.items() if "StrategyV2Base" in found] == ["HbForwardBot"]
    assert [name for name, found in bases.items() if "StrategyV2ConfigBase" in found] == ["HbForwardBotConfig"]
    imported = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "hummingbot" for alias in node.names}
    assert {name for name in imported if "Strategy" in name or "Config" in name} == {
        "StrategyV2Base", "StrategyV2ConfigBase"}


def test_shell_checks_the_settings_and_the_frozen_inputs_before_it_builds_the_core():
    bot = _classes(_tree(BOT_MODULE))["HbForwardBot"]
    init = next(node for node in bot.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    first: dict[str, int] = {}
    for node in sorted((node for node in ast.walk(init) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)), key=lambda node: (node.lineno, node.col_offset)):
        first.setdefault(node.func.id, node.lineno)
    order = ("from_mapping", "check_settings", "verify_inputs", "BotCore")
    assert [name for name in order if name not in first] == []
    assert [first[name] for name in order] == sorted(first[name] for name in order)
    assert len({first[name] for name in order}) == len(order)


def test_deploy_scripts_package_imports_exactly_the_two_bot_classes():
    package = DEPLOY / "scripts"
    assert (package / "__init__.py").is_file()  # a regular package: the repo's own scripts/ cannot shadow it
    tree = _tree(package / "hb_forward_bot.py")
    statements = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                                                     and isinstance(node.value.value, str))]
    assert len(statements) == 1
    (node,) = statements
    assert isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "hummingbot_adapter.hb_forward_bot"
    assert sorted((alias.name, alias.asname) for alias in node.names) == [("HbForwardBot", None),
                                                                         ("HbForwardBotConfig", None)]
    assert {"HbForwardBot", "HbForwardBotConfig"} <= set(_classes(_tree(BOT_MODULE)))


# --- the deploy files ----------------------------------------------------------------------------------------------

SCRIPT_CONFIG_FILES = {  # file: (model, mode, the §7 thresholds it changes)
    "conf_hb_b0.yml": ("B0", "production", {}),
    "conf_hb_b3.yml": ("B3", "production", {}),
    "conf_hb_b0_smoke.yml": ("B0", "smoke", {}),
    "conf_hb_b3_smoke.yml": ("B3", "smoke", {}),
    "conf_hb_b3_smoke_drawdown.yml": ("B3", "smoke", {"drawdown_limit_usdt": 0.0001}),
    "conf_hb_b3_smoke_qcap.yml": ("B3", "smoke", {"halt_q_max": 1}),
}
SERVICES = {"hb_b0": "B0", "hb_b3": "B3"}
COMPOSE_ENV = {"HBOT_PASSWORD": "from-the-env", "HB_DATA": "/host/hb_data", "HB_ARTIFACTS": "/host/hb_artifacts"}
_VARIABLE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:(?P<op>:?[-?])(?P<arg>[^}]*))?\}")


@pytest.fixture(scope="module")
def yaml_module():
    return pytest.importorskip("yaml", reason="PyYAML reads the deploy files (Hummingbot uses yaml.safe_load)")


def _load(yaml_module, path: Path):
    return yaml_module.safe_load(path.read_text(encoding="utf-8"))


def _interpolate(text: str, env: dict[str, str]) -> str:
    """Docker Compose's ${NAME}, ${NAME:-default}, ${NAME-default}, ${NAME:?error} and ${NAME?error}."""

    def value(match: re.Match) -> str:
        name, op, arg = match["name"], match["op"], match["arg"]
        found = env.get(name)
        missing = found is None or (found == "" and op is not None and op.startswith(":"))
        if op in (":-", "-"):
            return arg if missing else found
        if op in (":?", "?"):
            if missing:
                raise ValueError(f"{name} is required: {arg}")
            return found
        return "" if found is None else found

    resolved = _VARIABLE.sub(value, text)
    assert "$" not in resolved, f"unresolved interpolation in {text!r}"
    return resolved


@pytest.fixture(scope="module")
def compose(yaml_module):
    return _load(yaml_module, DEPLOY / "docker-compose.yml")


def test_the_script_configs_are_the_six_expected_files():
    assert sorted(path.name for path in SCRIPT_CONFIGS.iterdir()) == sorted(SCRIPT_CONFIG_FILES)


@pytest.mark.parametrize("name", sorted(SCRIPT_CONFIG_FILES))
def test_script_config_loads_and_passes_check_settings(yaml_module, name):
    model, mode, thresholds = SCRIPT_CONFIG_FILES[name]
    values = _load(yaml_module, SCRIPT_CONFIGS / name)
    assert values["script_file_name"] == "hb_forward_bot.py"
    assert (DEPLOY / "scripts" / values["script_file_name"]).is_file()  # Hummingbot imports scripts.<stem>
    assert values["controllers_config"] == []
    settings = from_mapping(values)
    assert (settings.model, settings.mode) == (model, mode)
    assert check_settings(settings) == list(thresholds)
    assert {key: getattr(settings, key) for key in thresholds} == thresholds
    root = Path(settings.data_root)
    assert settings.bot_dir == (root / settings.bot if mode == "production" else root / "smoke" / settings.bot)


@pytest.mark.parametrize("name", ["conf_hb_b0.yml", "conf_hb_b3.yml"])
def test_production_script_config_lists_every_field_at_its_default(yaml_module, name):
    model = SCRIPT_CONFIG_FILES[name][0]
    values = _load(yaml_module, SCRIPT_CONFIGS / name)
    assert set(values) == set(SETTINGS_FIELDS) | HB_CONFIG_FIELDS
    defaults = BotSettings(model=model, mode="production")
    assert from_mapping(values) == defaults
    for key, field in SETTINGS_FIELDS.items():
        assert values[key] == getattr(defaults, key), key
        assert type(values[key]).__name__ == field.type, key  # written as declared, not rescued by coercion
    for key, value in PROTOCOL_MD.items():
        assert values[key] == value, key


@pytest.mark.parametrize("name", ["conf_hb_b0_smoke.yml", "conf_hb_b3_smoke.yml", "conf_hb_b3_smoke_drawdown.yml",
                                  "conf_hb_b3_smoke_qcap.yml"])
def test_smoke_script_config_differs_from_production_only_in_mode_and_its_thresholds(yaml_module, name):
    model, _, thresholds = SCRIPT_CONFIG_FILES[name]
    smoke = _load(yaml_module, SCRIPT_CONFIGS / name)
    production = _load(yaml_module, SCRIPT_CONFIGS / f"conf_hb_{model.lower()}.yml")
    assert set(smoke) == set(production)
    differ = {key: smoke[key] for key in smoke
              if smoke[key] != production[key] or type(smoke[key]) is not type(production[key])}
    assert differ == {"mode": "smoke", **thresholds}


def test_client_configs_tick_at_the_sampling_interval_with_the_protocol_paper_account(yaml_module):
    instance_ids = []
    for bot in SERVICES:
        client = _load(yaml_module, DEPLOY / "conf" / bot / "conf_client.yml")
        assert client["tick_size"] == PROTOCOL_MD["sampling_interval_s"] == 0.1  # §5.1
        paper = client["paper_trade"]
        assert paper["paper_trade_exchanges"] == ["binance"]  # binance_paper_trade (§3)
        assert paper["paper_trade_account_balance"] == {"DOGE": 1_000_000, "USDT": 100_000}  # §3
        assert paper["paper_trade_account_balance"] == {"DOGE": protocol_default("paper_doge_balance"),
                                                        "USDT": protocol_default("paper_usdt_balance")}
        assert isinstance(client["instance_id"], str) and client["instance_id"]
        instance_ids.append(client["instance_id"])
    assert len(set(instance_ids)) == len(SERVICES)


def test_compose_runs_each_bot_from_the_local_image_with_its_production_config(yaml_module, compose):
    services = compose["services"]
    assert set(services) == set(SERVICES) == {BotSettings(model=m, mode="production").bot for m in ("B0", "B3")}
    for name, service in services.items():
        model = SERVICES[name]
        assert service["image"] == IMAGE
        assert service["init"] is True
        assert service["restart"] == "unless-stopped"
        assert service["container_name"] == name
        command = [_interpolate(part, COMPOSE_ENV) for part in service["command"]]
        assert command == ["hbot", "start", f"conf_hb_{model.lower()}.yml", "--foreground"]
        config = _load(yaml_module, SCRIPT_CONFIGS / command[2])
        assert (config["model"], config["mode"]) == (model, "production")
        environment = service["environment"]
        assert re.fullmatch(r"\$\{HBOT_PASSWORD(:?\?[^}]*)?\}", environment["HBOT_PASSWORD"])
        assert _interpolate(environment["HBOT_PASSWORD"], COMPOSE_ENV) == COMPOSE_ENV["HBOT_PASSWORD"]
        assert environment["PYTHONPATH"] == "/repo"


def test_compose_mounts_the_code_inputs_and_configs_read_only(compose):
    for name, service in compose["services"].items():
        mounts: dict[str, tuple[str, set[str]]] = {}
        for raw in service["volumes"]:
            parts = _interpolate(raw, COMPOSE_ENV).split(":")
            assert len(parts) in (2, 3), raw
            assert parts[1] not in mounts, f"{name}: {parts[1]} is mounted twice"
            mounts[parts[1]] = (parts[0], set(parts[2].split(",")) if len(parts) == 3 else set())
        settings = BotSettings(model=SERVICES[name], mode="production")
        read_only = {
            settings.repo_dir: REPO_ROOT,  # /repo: the repository root
            "/home/hummingbot/scripts": DEPLOY / "scripts",
            "/home/hummingbot/conf/scripts": SCRIPT_CONFIGS,
        }
        for target, folder in read_only.items():
            source, options = mounts[target]
            assert (DEPLOY / source).resolve() == folder.resolve(), (name, target)
            assert "ro" in options, (name, target)
        assert mounts[settings.artifacts_dir] == (COMPOSE_ENV["HB_ARTIFACTS"], {"ro"})
        source, options = mounts[settings.data_root]
        assert source == COMPOSE_ENV["HB_DATA"] and "ro" not in options
        for target in ("/home/hummingbot/conf", "/home/hummingbot/data", "/home/hummingbot/logs"):
            assert "ro" not in mounts[target][1], (name, target)  # Hummingbot writes these


def test_gitignore_ignores_the_deploy_env_file():
    lines = [line.strip() for line in (ADAPTER / ".gitignore").read_text(encoding="utf-8").splitlines()]
    patterns = [line for line in lines if line and not line.startswith("#")]
    assert "deploy/.env" in patterns
    assert "!deploy/.env" not in patterns


def test_status_script_is_short():
    assert len((DEPLOY / "status.ps1").read_text(encoding="utf-8").splitlines()) < 30


# --- import discipline of the new runtime modules -----------------------------------------------------------------

NEW_RUNTIME_MODULES = ("execution", "telemetry", "bot_config", "bot_core", "monitor", "clear_halt", "run_log")
ALLOWED_ADAPTER = frozenset(f"hummingbot_adapter.{name}" for name in (
    "features", "quotes", "policy", "execution", "telemetry", "bot_config", "bot_core", "monitor", "run_log"))
THIRD_PARTY = frozenset({"numpy", "pandas"})
PROJECT = frozenset({"cvar_mm.gym.formula_as"})


def _imports(path: Path) -> tuple[list[str], list[str]]:
    found, problems = [], []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"relative import at line {node.lineno}")
            if node.module == "hummingbot_adapter":  # from hummingbot_adapter import x imports the module x
                found += [f"hummingbot_adapter.{alias.name}" for alias in node.names]
            else:
                found.append(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in {"__import__", "import_module", "exec", "eval"}:
                problems.append(f"dynamic import or evaluation at line {node.lineno}: {name}")
    return found, problems


def _allowed(module: str) -> bool:
    root = module.split(".")[0]
    return (module == "__future__" or root in sys.stdlib_module_names or root in THIRD_PARTY or module in PROJECT
            or module in ALLOWED_ADAPTER)


@pytest.mark.parametrize("name", NEW_RUNTIME_MODULES)
def test_new_runtime_module_imports_only_the_allowed_modules(name):
    found, problems = _imports(ADAPTER / f"{name}.py")
    assert found, "the AST walk found no import"
    assert problems == []
    assert [module for module in found if not _allowed(module)] == []
    assert [module for module in found if module.split(".")[0] == "hummingbot"] == []


def test_the_allowlist_rejects_what_the_rule_forbids():
    for module in ("hummingbot", "hummingbot.core.clock", "torch", "stable_baselines3.common", "gymnasium",
                   "scipy.stats", "yaml", "cvar_mm", "cvar_mm.gym", "cvar_mm.gym.calibration", "scripts.pipeline.common",
                   "hummingbot_adapter", "hummingbot_adapter.calibrate", "hummingbot_adapter.export_policy",
                   "hummingbot_adapter.hb_forward_bot", "hummingbot_adapter.clear_halt"):
        assert not _allowed(module), module
    for module in ("__future__", "json", "urllib.request", "collections.abc", "numpy", "pandas",
                   "cvar_mm.gym.formula_as", "hummingbot_adapter.execution", "hummingbot_adapter.monitor"):
        assert _allowed(module), module


def test_shell_imports_only_hummingbot_and_the_allowed_modules():
    """hb_forward_bot.py may import hummingbot, but nothing that would not load in the container (for example
    hummingbot_adapter.calibrate, whose scripts.pipeline import Hummingbot's scripts package shadows)."""

    found, problems = _imports(BOT_MODULE)
    assert problems == []
    assert any(module.split(".")[0] == "hummingbot" for module in found)
    assert [module for module in found if module.split(".")[0] != "hummingbot" and not _allowed(module)] == []


def test_new_runtime_modules_import_without_hummingbot_torch_sb3_gymnasium_or_scipy():
    """As in a container without them: None in sys.modules makes an import fail."""

    code = (
        "import json, sys\n"
        "for name in ('hummingbot', 'torch', 'stable_baselines3', 'gymnasium', 'scipy'):\n"
        "    sys.modules[name] = None\n"
        + "".join(f"import hummingbot_adapter.{name}\n" for name in NEW_RUNTIME_MODULES)
        + "heavy = {'hummingbot', 'torch', 'stable_baselines3', 'gymnasium', 'scipy', 'matplotlib', 'seaborn',\n"
        "         'sklearn', 'shap'}\n"
        "print(json.dumps({\n"
        "    'heavy': sorted(n for n, m in sys.modules.items() if m is not None and n.split('.')[0] in heavy),\n"
        "    'cvar_mm': sorted(n for n in sys.modules if n.split('.')[0] == 'cvar_mm'),\n"
        "    'adapter': sorted(n for n in sys.modules if n.split('.')[0] == 'hummingbot_adapter'),\n"
        "}))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded["heavy"] == []
    assert loaded["cvar_mm"] == ["cvar_mm", "cvar_mm.gym", "cvar_mm.gym.experiment_config", "cvar_mm.gym.formula_as",
                               "cvar_mm.gym.index_names"]
    assert loaded["adapter"] == sorted({"hummingbot_adapter"} | {f"hummingbot_adapter.{name}" for name in (
        *NEW_RUNTIME_MODULES, "features", "quotes", "policy")})


# --- monitor: ClockChecker ----------------------------------------------------------------------------------------

T0_US = 1_792_454_400_000_000  # 2026-10-20 00:00:00 UTC


class _Requests:
    """Scripted GET /api/v3/time requests for ClockChecker. During each fetch the true time (the server's clock)
    advances by the request's round trip, and the server stamps it at the middle, lag_ms behind; the bot clock runs
    1 s ahead of the true time and the wall clock 0.25 s ahead, plus wall_steps_us applied inside the requests. An
    exception in rtts_ms is raised by that fetch."""

    def __init__(self, rtts_ms: list, lags_ms: list[int], wall_steps_us: list[int] | None = None):
        self.true_us, self.wall_shift_us, self.mono_ns = T0_US, 0, 9_000_000_000
        self.script = list(zip(rtts_ms, lags_ms, wall_steps_us or [0] * len(rtts_ms), strict=True))
        self.fetches = 0

    def peek_us(self) -> int:
        return self.true_us + 1_000_000

    def wall_us(self) -> int:
        return self.true_us + 250_000 + self.wall_shift_us

    def monotonic_ns(self) -> int:
        return self.mono_ns

    def _advance(self, us: int) -> None:
        self.true_us += us
        self.mono_ns += us * 1_000

    def fetch_ms(self) -> int:
        rtt_ms, lag_ms, wall_step_us = self.script[self.fetches]
        self.fetches += 1
        if isinstance(rtt_ms, BaseException):
            raise rtt_ms
        self._advance(rtt_ms * 500)
        self.wall_shift_us += wall_step_us
        server_ms = self.true_us // 1_000 - lag_ms
        self._advance(rtt_ms * 500)
        return server_ms

    def checker(self, **kwargs) -> ClockChecker:
        return ClockChecker(self.peek_us, self.wall_us, fetch_ms=self.fetch_ms, monotonic_ns=self.monotonic_ns,
                            **kwargs)


def test_clock_check_uses_the_smallest_round_trip_of_5_at_its_midpoint_and_discards_round_trips_over_1_s():
    """Clarification i: five requests per check; round trips above 1 s are discarded, and the request with the
    smallest round trip is used at its midpoint (RTT/2 bounds its error)."""

    requests = _Requests(rtts_ms=[1_500, 400, 2_000, 300, 1_200], lags_ms=[10, 20, 30, 40, 50])
    result = requests.checker().check_once()
    assert requests.fetches == 5 and result["error"] is None and result["fetch_errors"] == 0
    assert result["rtts_s"] == [1.5, 0.4, 2.0, 0.3, 1.2]
    assert result["rtt_s"] == 0.3  # 0.4 s and 0.3 s are kept; the fourth request has the smallest round trip
    assert result["server_ms"] == T0_US // 1000 + 4_010  # stamped 4.05 s after T0, 40 ms behind
    assert result["offset_s"] == 1.04 and result["wall_offset_s"] == 0.29  # 1 s and 0.25 s ahead, + the 40 ms


def test_a_check_with_no_round_trip_up_to_1_s_is_an_error_without_an_offset_or_a_rate_baseline():
    requests = _Requests(rtts_ms=[1_100, 1_500, 2_000, 1_001, 3_000], lags_ms=[0] * 5)
    checker = requests.checker()
    result = checker.check_once()
    assert result["error"] == "no round trip <= 1.0 s among 5 responses"
    assert result["rtts_s"] == [1.1, 1.5, 2.0, 1.001, 3.0] and result["fetch_errors"] == 0
    assert "offset_s" not in result and not checker._history  # nothing for §7, and no baseline for a rate
    failing = _Requests([OSError("down"), 1_500, ConnectionResetError(104, "reset"), 2_000, 1_200], [0] * 5)
    result = failing.checker().check_once()
    assert result["error"] == ("no round trip <= 1.0 s among 3 responses; 2 failed, the last "
                               "ConnectionResetError(104, 'reset')") and "offset_s" not in result
    partly = _Requests([OSError("down"), 1_500, 350, OSError("down"), 1_200], [0, 0, 5, 0, 0])
    result = partly.checker().check_once()  # one usable response is enough
    assert result["error"] is None and result["fetch_errors"] == 2 and result["rtt_s"] == 0.35
    assert result["offset_s"] == 1.005


def test_clock_check_logs_the_uncorrected_offset_read_around_the_same_request():
    """Clarification j: with raw_us (the step clock without its rate correction) a check also reports raw_offset_s,
    from the same request and midpoint as offset_s, so their difference carries no network error; without raw_us it
    is None."""

    requests = _Requests(rtts_ms=[1_500, 400, 2_000, 300, 1_200], lags_ms=[10, 20, 30, 40, 50])
    result = requests.checker(raw_us=lambda: requests.true_us + 1_200_000).check_once()
    assert (result["offset_s"], result["raw_offset_s"], result["wall_offset_s"]) == (1.04, 1.24, 0.29)
    plain = _Requests(rtts_ms=[300], lags_ms=[0])
    assert plain.checker(samples=1).check_once()["raw_offset_s"] is None


def test_round_trips_are_timed_on_the_monotonic_clock_so_a_wall_clock_step_cannot_win_the_minimum():
    """A wall-clock step back of 3 s inside a 0.6 s request would give a round trip of -2.4 s on the wall clock;
    timed on the monotonic clock it is 0.6 s, and the 0.3 s request is used."""

    requests = _Requests(rtts_ms=[600, 300, 350, 400, 450], lags_ms=[0, 7, 0, 0, 0],
                         wall_steps_us=[-3_000_000, 0, 0, 0, 0])
    result = requests.checker().check_once()
    assert result["rtts_s"] == [0.6, 0.3, 0.35, 0.4, 0.45] and result["rtt_s"] == 0.3
    assert result["offset_s"] == 1.007 and result["wall_offset_s"] == -2.743  # the wall clock now 2.75 s behind


@pytest.mark.parametrize("error", [
    urllib.error.URLError("unreachable"),
    TimeoutError("timed out"),
    ConnectionResetError(104, "reset by peer"),
    http.client.RemoteDisconnected("closed without response"),
    json.JSONDecodeError("Expecting value", "<html>", 0),
    KeyError("serverTime"),
    http.client.IncompleteRead(b'{"serverTi'),  # the connection dropped inside the body
    http.client.BadStatusLine("garbage"),  # a malformed status line (e.g. from a proxy)
], ids=lambda error: type(error).__name__)
def test_clock_check_error_path_returns_the_error_and_no_offset(error):
    """Every way GET /api/v3/time can fail is a result for the health log, never an exception: an exception
    would end the checker's thread, and with it every later hourly check (and the lifting of a clock pause)."""

    def fetch() -> int:
        raise error

    checker = ClockChecker(lambda: T0_US, lambda: T0_US, fetch_ms=fetch)
    result = checker.check_once()
    assert isinstance(result["error"], str) and type(error).__name__ in result["error"]
    assert "offset_s" not in result and "wall_offset_s" not in result


class _RecordingStop:
    """Stands in for ClockChecker's stop event: records each wait's timeout and stops after `waits` waits."""

    def __init__(self, waits: int):
        self.timeouts: list[float] = []
        self._waits = waits

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        return len(self.timeouts) >= self._waits

    def set(self) -> None:
        raise AssertionError("not used here")


def test_clock_checker_retries_until_settled_then_checks_hourly():
    """A failed check or an offset above the warning level repeats after retry_s (so a clock pause can lift);
    a settled check (|offset| <= warn_s, PROTOCOL §7 warns only above 0.5 s) waits the hourly interval. Each
    check sends 5 requests (clarification i)."""

    outcomes = [OSError("down"), 1_000, -2_500, 200, 500]  # per check: the bot clock's lead on the server, in ms
    calls = []

    def fetch() -> int:
        outcome = outcomes[len(calls) // 5]
        calls.append(outcome)
        if isinstance(outcome, BaseException):
            raise outcome
        return T0_US // 1000 - outcome

    checker = ClockChecker(lambda: T0_US, lambda: T0_US, interval_s=3_600.0, retry_s=60.0, warn_s=0.5,
                           fetch_ms=fetch)
    stop = _RecordingStop(waits=5)
    checker._stop = stop  # no sleeping: the stop event decides when _run returns
    checker._run()
    assert stop.timeouts == [60.0, 60.0, 60.0, 3_600.0, 3_600.0] and len(calls) == 25
    results = checker.drain()
    assert [result["error"] is not None for result in results] == [True, False, False, False, False]
    assert [result.get("offset_s") for result in results] == [None, 1.0, -2.5, 0.2, 0.5]
    assert checker.drain() == []


def test_clock_checker_checks_at_once_in_its_thread_and_stops():
    checker = ClockChecker(lambda: T0_US, lambda: T0_US, fetch_ms=lambda: T0_US // 1000)
    checker.start()
    with pytest.raises(RuntimeError):
        checker.start()
    checker.stop()
    checker._thread.join(timeout=10.0)
    assert not checker._thread.is_alive()
    assert [result["offset_s"] for result in checker.drain()] == [0.0]


@pytest.mark.parametrize("interval_s, retry_s", [(3_600.0, 0.0), (3_600.0, -1.0), (60.0, 120.0)])
def test_clock_checker_refuses_a_retry_outside_the_interval(interval_s, retry_s):
    with pytest.raises(ValueError):
        ClockChecker(lambda: T0_US, lambda: T0_US, interval_s=interval_s, retry_s=retry_s,
                     fetch_ms=lambda: T0_US // 1000)


@pytest.mark.parametrize("setting", [{"samples": 0}, {"max_rtt_s": 0.0}, {"rate_window_s": 0.0}, {"rate_window_s": -1.0}])
def test_clock_checker_refuses_no_samples_and_non_positive_limits(setting):
    """A zero rate window would divide by a zero span and end the checker's thread; refused at construction."""

    with pytest.raises(ValueError):
        ClockChecker(lambda: T0_US, lambda: T0_US, fetch_ms=lambda: T0_US // 1000, **setting)


def test_clock_checker_retries_after_a_check_whose_round_trips_all_exceed_1_s():
    """An all-too-slow check gives no offset, so it is not settled: the next check comes after retry_s."""

    requests = _Requests(rtts_ms=[1_500] * 5 + [300] * 5, lags_ms=[0] * 5 + [-1_000] * 5)  # then 0 s offset
    checker = requests.checker(interval_s=3_600.0, retry_s=60.0, warn_s=0.5)
    stop = _RecordingStop(waits=2)
    checker._stop = stop
    checker._run()
    assert stop.timeouts == [60.0, 3_600.0] and requests.fetches == 10
    first, second = checker.drain()
    assert first["error"] == "no round trip <= 1.0 s among 5 responses" and second["offset_s"] == 0.0


class _Response:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_fetch_binance_server_ms_reads_server_time_from_api_v3_time(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return _Response(b'{"serverTime":1792454400123}')

    monkeypatch.setattr(monitor.urllib.request, "urlopen", urlopen)
    assert monitor.fetch_binance_server_ms(timeout_s=3.0) == 1_792_454_400_123
    assert calls == [("https://api.binance.com/api/v3/time", 3.0)]  # PROTOCOL §7: GET /api/v3/time


# --- monitor: ReconnectWatcher and memory ---------------------------------------------------------------------------

# Hummingbot 2.17.0 logger names: the data source's logger_name_for_class (module.qualname), under
# hummingbot.connector (level NETWORK = 16, no propagation), and the order-book tracker's module logger.
DATA_SOURCE_LOGGER = ("hummingbot.connector.exchange.binance.binance_api_order_book_data_source."
                      "BinanceAPIOrderBookDataSource")
TRACKER_LOGGER = "hummingbot.core.data_type.order_book_tracker"


@pytest.fixture
def feed_loggers():
    """The feed's loggers set up as in the container; levels, propagation and handlers are restored after."""

    names = ("hummingbot.connector", TRACKER_LOGGER, DATA_SOURCE_LOGGER)
    saved = {name: (logging.getLogger(name).level, logging.getLogger(name).propagate,
                    list(logging.getLogger(name).handlers)) for name in names}
    connector = logging.getLogger("hummingbot.connector")
    connector.setLevel(16)
    connector.propagate = False
    logging.getLogger(TRACKER_LOGGER).setLevel(logging.INFO)
    yield logging.getLogger(DATA_SOURCE_LOGGER), logging.getLogger(TRACKER_LOGGER)
    for name, (level, propagate, handlers) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers


def test_reconnect_watcher_maps_the_feed_messages_to_kinds(feed_loggers):
    source, tracker = feed_loggers
    watcher = ReconnectWatcher()
    watcher.attach()
    try:
        source.info("Connecting to wss://stream.binance.com:9443/ws")  # not a feed message
        source.warning("The websocket connection was closed (The WS connection was closed unexpectedly. "
                       "Close code = 1000 msg data: None)")
        try:
            raise RuntimeError("stream failed")
        except RuntimeError:
            source.exception("Unexpected error occurred when listening to order book streams. Retrying in 5 seconds...")
        source.info("Subscribed to public order book and trade channels...")
        tracker.info("Initialized order book for DOGE-USDT. 1/1 completed.")
        tracker.info("Order book tracker started")  # not a feed message
        logging.getLogger("hb_test_not_attached").warning("The websocket connection was closed (elsewhere)")
        events = watcher.drain()
    finally:
        watcher.detach()
    assert [(kind, level) for kind, level, _ in events] == [
        ("ws_closed", "WARNING"), ("ws_error", "ERROR"), ("subscribed", "INFO"), ("book_initialized", "INFO")]
    assert events[2][2] == "Subscribed to public order book and trade channels..."
    assert watcher.drain() == []


def test_reconnect_watcher_hears_nothing_once_detached(feed_loggers):
    source, _ = feed_loggers
    watcher = ReconnectWatcher()
    watcher.attach(("hummingbot.connector",))
    source.info("Subscribed to public order book and trade channels...")
    watcher.detach(("hummingbot.connector",))
    source.info("Subscribed to public order book and trade channels...")
    assert [kind for kind, _, _ in watcher.drain()] == ["subscribed"]


def test_memory_rss_mb_reads_vmrss_and_is_none_without_proc(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  401234 kB\nVmSize:\t  398765 kB\nVmHWM:\t  130000 kB\n"
                      "VmRSS:\t  123456 kB\nRssAnon:\t  100000 kB\n", encoding="utf-8")
    assert memory_rss_mb(status) == 123456 / 1024
    assert memory_rss_mb(tmp_path / "missing" / "status") is None


# --- round 2: clarification h (hb_forward_bot imports Hummingbot, so its source is read, not imported) ------------


def test_hb_forward_bot_sets_the_binance_book_resync_to_60_s_at_import_and_checks_it():
    path = REPO_ROOT / "hummingbot_adapter" / "hb_forward_bot.py"
    tree = _tree(path)
    constants = {node.targets[0].id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)}
    assert constants["BOOK_RESYNC_S"] == 60
    [resync] = [node for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                and ast.unparse(node.targets[0]) == "BinanceAPIOrderBookDataSource.FULL_ORDER_BOOK_RESET_DELTA_SECONDS"]
    assert ast.unparse(resync.value) == "BOOK_RESYNC_S"
    # module level, before the classes: in place when Hummingbot imports the script, before the connector starts
    assert resync.lineno < min(node.lineno for node in tree.body if isinstance(node, ast.ClassDef))
    [origin] = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                and any(alias.name == "BinanceAPIOrderBookDataSource" for alias in node.names)]
    assert origin.module == "hummingbot.connector.exchange.binance.binance_api_order_book_data_source"
    source = path.read_text(encoding="utf-8")  # the strategy refuses to start unless the data source sees 60 s
    assert "order_book_tracker.data_source" in source and "FULL_ORDER_BOOK_RESET_DELTA_SECONDS != BOOK_RESYNC_S" in source


def test_clock_checker_reports_the_step_and_monotonic_rates_over_the_past_hour():
    """A diagnostic, no trading rule: Δstep / Δserver - 1 and Δmonotonic / Δserver - 1, from the latest check at
    least an hour old."""

    server = {"us": 1_792_500_000_000_000}
    clock = {"mono_ns": 5_000_000_000, "step_us": 1_792_500_000_000_000}
    checker = ClockChecker(lambda: clock["step_us"], lambda: server["us"], fetch_ms=lambda: server["us"] // 1000,
                           monotonic_ns=lambda: clock["mono_ns"])
    first = checker.check_once()
    assert (first["rate_step"], first["rate_monotonic"], first["rate_over_s"]) == (None, None, None)
    # an hour later: the monotonic clock 2 % slow; the step clock 2 s behind after a re-anchor caught the rest
    server["us"] += 3_600_000_000
    clock["mono_ns"] += 3_528_000_000_000
    clock["step_us"] += 3_598_000_000
    second = checker.check_once()
    assert second["rate_over_s"] == 3_600.0
    assert second["rate_monotonic"] == pytest.approx(-0.02, abs=1e-12)
    assert second["rate_step"] == pytest.approx(-2 / 3_600, abs=1e-12)
    # a minute later: still over at least an hour (from the first check), not over the last minute
    server["us"] += 60_000_000
    clock["mono_ns"] += 58_800_000_000
    clock["step_us"] += 60_000_000
    third = checker.check_once()
    assert third["rate_over_s"] == 3_660.0
    assert third["rate_monotonic"] == pytest.approx(-0.02, abs=1e-12)
    # two hours after the first: the reference is the LATEST check at least an hour old (the third), not the oldest
    server["us"] += 3_640_000_000
    clock["mono_ns"] += 3_567_200_000_000
    clock["step_us"] += 3_640_000_000
    fourth = checker.check_once()
    assert fourth["rate_over_s"] == 3_640.0
    assert fourth["rate_monotonic"] == pytest.approx(-0.02, abs=1e-12)
    assert len(checker._history) == 3  # the first check, more than two hours old, was dropped


def test_clock_checker_reports_no_rate_until_a_check_at_least_an_hour_old_exists():
    """Clarification i: the rates need a baseline of at least an hour. Before one exists (as for a check a minute
    after start-up) they are None, never a rate over a minute."""

    server = {"us": T0_US}
    checker = ClockChecker(lambda: server["us"] + 100_000, lambda: server["us"], fetch_ms=lambda: server["us"] // 1000,
                           monotonic_ns=lambda: server["us"] * 1_000)
    for minute in range(60):  # one check a minute for 59 minutes
        result = checker.check_once()
        assert result["error"] is None and result["offset_s"] == 0.1
        assert (result["rate_step"], result["rate_monotonic"], result["rate_over_s"]) == (None, None, None), minute
        server["us"] += 60_000_000
    result = checker.check_once()  # an hour after the first
    assert (result["rate_step"], result["rate_monotonic"], result["rate_over_s"]) == (0.0, 0.0, 3_600.0)


def test_hb_forward_bot_runs_its_step_clock_and_clock_check_on_clock_monotonic_raw():
    """Clarification i: the shell builds its StepClock with the default source, CLOCK_MONOTONIC_RAW, and hands the
    clock check that same source, so the rate diagnostic measures the clock that advances ts. Clarification k: the
    check reads the step clock without the day's ceiling (unheld_us), never the held ts. Clarification j: the step
    clock gets the rate correction CLOCK_RAW_RATE_PPM, and the check logs the uncorrected clock beside it."""

    tree = _tree(REPO_ROOT / "hummingbot_adapter" / "hb_forward_bot.py")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    [step_clock] = [node for node in calls if ast.unparse(node.func) == "StepClock"]
    [checker] = [node for node in calls if ast.unparse(node.func) == "ClockChecker"]
    assert step_clock.args == []
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in step_clock.keywords} == {
        "rate_ppm": "CLOCK_RAW_RATE_PPM"}
    [origin] = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and any(alias.name == "CLOCK_RAW_RATE_PPM" for alias in node.names)]
    assert origin.module == "hummingbot_adapter.execution"
    assert [ast.unparse(arg) for arg in checker.args] == ["self._clock.unheld_us", "self._clock.wall_us"]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in checker.keywords}
    assert keywords["monotonic_ns"] == "self._clock.monotonic_ns"
    assert keywords["raw_us"] == "self._clock.uncorrected_us"
    assert inspect.signature(StepClock).parameters["monotonic_ns"].default is monotonic_raw_ns
    # the checks keep their defaults: the smallest round trip of 5, round trips above 1 s discarded, a 1 h baseline
    assert not {"samples", "max_rtt_s", "rate_window_s", "fetch_ms"} & set(keywords)
    defaults = {name: parameter.default for name, parameter in inspect.signature(ClockChecker).parameters.items()}
    assert (defaults["samples"], defaults["max_rtt_s"], defaults["rate_window_s"]) == (5, 1.0, 3_600.0)



def test_reconnect_watcher_counts_a_failed_rest_resync_as_a_snapshot_error():
    """Clarification h: Hummingbot logs these two messages when a REST re-sync fails; the bot counts them."""

    watcher = ReconnectWatcher()
    name = "hummingbot.connector.exchange.binance.binance_api_order_book_data_source.BinanceAPIOrderBookDataSource"
    for message in ("Unexpected error fetching order book snapshot for DOGE-USDT.",
                    "Unexpected error when processing public order book snapshots from exchange"):
        watcher.emit(logging.LogRecord(name, logging.ERROR, __file__, 1, message, None, None))
    assert [kind for kind, _, _ in watcher.drain()] == ["snapshot_error", "snapshot_error"]
