"""Settings of the forward-test bot (WO-HB2 deliverable 3): every PROTOCOL.md constant with its default, the
production check, the config hash, and the check of the frozen inputs.

hb_forward_bot.HbForwardBotConfig declares the same fields with these defaults (test_hb_bot_config.py checks
that they agree) and becomes a BotSettings through from_mapping(). In production every protocol constant must
equal PROTOCOL.md; in smoke mode only the §7 thresholds may differ, and the outputs go under <data_root>/smoke/.

Runtime module: standard library only (it runs inside the Hummingbot container).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MODELS = ("B0", "B3")
MODES = ("production", "smoke")
BOT_NAMES = {"B0": "hb_b0", "B3": "hb_b3"}

# Frozen inputs (PROTOCOL §2; WO-HB1): the committed calibration, the exported B3 actor and its sources, gamma_B0.
CALIBRATION_FILE = "hb_calibration.json"
COMMITTED_CALIBRATION = PurePosixPath("hummingbot_adapter") / "config" / CALIBRATION_FILE
CALIBRATION_SHA256 = "ebbe62470773dc1b27c24fa013444c10a6a9af10c77acfdd8b4d2397cd24fd6d"
# The forward-test run used the calibration file with this hash. The published copy (CALIBRATION_SHA256) differs from
# it only in two local paths (files[0] "input" and "csv"); every value the bots read is the same.
CALIBRATION_SHA256_RUN = "d71af506725da1e6dc047d82f58a30dcd3beb34c8cef4abc7f693e809282048c"
POLICY_FILE, POLICY_SIDECAR = "b3_policy.npz", "b3_policy.json"
POLICY_SHA256 = "15a4bf62df295e0b4e00a02203c27d18ecc626ac1f40de5dae4fd22f43058db2"
MODEL_SHA256 = "05ee4c97524456a6445f54cfe0a4bf353ccd539c20060f1176b83455cebe4e53"
VECNORM_SHA256 = "d2be858e002d3b1e5a32bbe65822de3a4686cdbb91a81b8636241460a9b009b9"
B0_GAMMA_FILE = "b0_gamma_fixed.txt"
B0_GAMMA_SHA256 = "34cbb018dd15a9112f8c6f4f2cbb2874bea4efefbc1de3ad7c1895b6abe226c4"


@dataclass(frozen=True)
class BotSettings:
    model: str
    mode: str
    exchange: str = "binance_paper_trade"
    trading_pair: str = "DOGE-USDT"
    artifacts_dir: str = "/artifacts"
    repo_dir: str = "/repo"
    data_root: str = "/hb_data"
    # PROTOCOL §2-§5 (market, model and order management)
    tick_size: float = 1e-05
    order_amount_doge: int = 100
    q_max: int = 10
    feature_window: int = 100
    imbalance_depth: int = 5
    book_levels: int = 25
    day_seconds: int = 86_400
    sampling_interval_s: float = 0.1
    gamma_b0: float = 0.8908584807525282
    gamma_min: float = 0.01
    gamma_max: float = 0.9
    skew_ticks_max: float = 5.0
    requote_min_ticks: int = 1
    requote_min_age_s: float = 1.0
    rate_window_s: int = 10
    rate_max_per_window: int = 100
    rate_max_per_day: int = 200_000
    close_seconds_before_midnight: int = 10
    paper_doge_balance: int = 1_000_000
    paper_usdt_balance: int = 100_000
    fee_rate: float = 0.001
    # PROTOCOL §7 thresholds (smoke mode may change these)
    halt_q_max: int = 10
    residual_max_doge: float = 1.0
    stale_pause_s: float = 30.0
    stale_halt_s: float = 300.0
    reject_max: int = 10
    reject_window_s: float = 60.0
    drawdown_limit_usdt: float = 40.0
    clock_warn_s: float = 0.5
    clock_pause_s: float = 2.0
    clock_check_interval_s: float = 3_600.0
    # operation (not in PROTOCOL.md)
    save_interval_s: float = 1.0
    heartbeat_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type in ("int", "float"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite number, got {value!r}")
                if field.type == "int" and value != int(value):
                    raise ValueError(f"{field.name} must be an integer, got {value!r}")
                if value <= 0:
                    raise ValueError(f"{field.name} must be positive, got {value!r}")
        if not self.stale_pause_s < self.stale_halt_s:
            raise ValueError("stale_pause_s must be below stale_halt_s")
        if not self.gamma_min < self.gamma_max:
            raise ValueError("gamma_min must be below gamma_max")

    @property
    def bot(self) -> str:
        return BOT_NAMES[self.model]

    @property
    def bot_dir(self) -> Path:
        """<data_root>/<bot> in production, <data_root>/smoke/<bot> in smoke mode (WO-HB2 deliverable 3)."""

        root = Path(self.data_root)
        return root / self.bot if self.mode == "production" else root / "smoke" / self.bot

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PROTOCOL_CONSTANTS = (
    "exchange", "trading_pair", "tick_size", "order_amount_doge", "q_max", "feature_window", "imbalance_depth",
    "book_levels", "day_seconds", "sampling_interval_s", "gamma_b0", "gamma_min", "gamma_max", "skew_ticks_max",
    "requote_min_ticks", "requote_min_age_s", "rate_window_s", "rate_max_per_window", "rate_max_per_day",
    "close_seconds_before_midnight", "paper_doge_balance", "paper_usdt_balance", "fee_rate",
)
THRESHOLDS = (
    "halt_q_max", "residual_max_doge", "stale_pause_s", "stale_halt_s", "reject_max", "reject_window_s",
    "drawdown_limit_usdt", "clock_warn_s", "clock_pause_s", "clock_check_interval_s",
)
OPERATION = ("artifacts_dir", "repo_dir", "data_root", "save_interval_s", "heartbeat_interval_s")
_DEFAULTS = {field.name: field.default for field in dataclasses.fields(BotSettings)
             if field.default is not dataclasses.MISSING}


def protocol_default(name: str) -> Any:
    return _DEFAULTS[name]


def from_mapping(values: Mapping[str, Any]) -> BotSettings:
    """BotSettings from the Hummingbot config's fields; refuses unknown keys. Hummingbot's own fields
    (script_file_name, controllers_config) are dropped."""

    names = {field.name for field in dataclasses.fields(BotSettings)}
    data = {key: value for key, value in values.items() if key not in ("script_file_name", "controllers_config")}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown settings: {unknown}")
    converted = {}
    for field in dataclasses.fields(BotSettings):
        if field.name not in data:
            continue
        value = data[field.name]
        if field.type == "float" and not isinstance(value, bool):
            value = float(value)
        elif (field.type == "int" and isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(value) and float(value).is_integer()):
            value = int(value)
        converted[field.name] = value
    return BotSettings(**converted)


MORE_SENSITIVE_WHEN_HIGHER = ("reject_window_s",)  # a longer window counts more rejects


def check_settings(settings: BotSettings) -> list[str]:
    """Raise if a production setting differs from PROTOCOL.md, or a smoke setting other than a §7 threshold
    does, or a smoke threshold is made less sensitive than PROTOCOL.md (smoke may only lower thresholds), or
    the state and telemetry would be saved less than once per second. Returns the thresholds a smoke run
    changed (for the health log)."""

    differ = [name for name in PROTOCOL_CONSTANTS if getattr(settings, name) != _DEFAULTS[name]]
    if differ:
        raise ValueError(f"settings differ from PROTOCOL.md in {settings.mode} mode: "
                         + ", ".join(f"{name}={getattr(settings, name)!r}" for name in differ))
    changed = [name for name in THRESHOLDS if getattr(settings, name) != _DEFAULTS[name]]
    if settings.mode == "production" and changed:
        raise ValueError("production thresholds differ from PROTOCOL.md: "
                         + ", ".join(f"{name}={getattr(settings, name)!r}" for name in changed))
    laxer = [name for name in changed if (getattr(settings, name) < _DEFAULTS[name]) == (name in MORE_SENSITIVE_WHEN_HIGHER)]
    if laxer:
        raise ValueError("smoke mode may only make §7 thresholds more sensitive: "
                         + ", ".join(f"{name}={getattr(settings, name)!r} (PROTOCOL {_DEFAULTS[name]!r})" for name in laxer))
    if settings.save_interval_s > 1.0:
        raise ValueError(f"save_interval_s {settings.save_interval_s!r}: the state and telemetry are written at "
                         "least once per second (WO-HB2)")
    return changed


def config_hash(settings: BotSettings) -> str:
    """SHA-256 of the settings as canonical JSON (sorted keys, repr floats)."""

    text = json.dumps(settings.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedInputs:
    sigma: float
    kappa: float
    A: float
    calibration_sha256: str
    policy_path: Path | None
    policy_sha256: str | None
    gamma_file_sha256: str | None


def verify_inputs(settings: BotSettings) -> VerifiedInputs:
    """Refuse to start unless the calibration equals the committed config/hb_calibration.json (and its pinned
    hash), and, for B3, b3_policy.npz matches the hash in b3_policy.json, whose sources are the PROTOCOL §2
    model and normalizer; for B0, b0_gamma_fixed.txt has its PROTOCOL §2 hash and holds gamma_b0."""

    artifacts = Path(settings.artifacts_dir)
    calibration = artifacts / CALIBRATION_FILE
    committed = Path(settings.repo_dir) / COMMITTED_CALIBRATION
    for path in (calibration, committed):
        digest = file_sha256(path)
        if digest != CALIBRATION_SHA256:
            raise ValueError(f"{path} has SHA-256 {digest}, expected {CALIBRATION_SHA256} (the committed calibration)")
    values = json.loads(calibration.read_text(encoding="utf-8"))
    sigma, kappa, A = (float(values[key]) for key in ("sigma", "kappa", "A"))
    if not all(math.isfinite(value) and value > 0 for value in (sigma, kappa, A)):
        raise ValueError(f"calibration values must be positive: sigma {sigma}, kappa {kappa}, A {A}")
    policy_path = policy_digest = gamma_digest = None
    if settings.model == "B3":
        policy_path = artifacts / POLICY_FILE
        sidecar = json.loads((artifacts / POLICY_SIDECAR).read_text(encoding="utf-8"))
        policy_digest = file_sha256(policy_path)
        expected = {
            "npz": (policy_digest, sidecar["output"]["sha256"]),
            "pinned npz": (policy_digest, POLICY_SHA256),
            "model": (sidecar["inputs"]["model"]["sha256"], MODEL_SHA256),
            "normalizer": (sidecar["inputs"]["vecnorm"]["sha256"], VECNORM_SHA256),
        }
        for name, (found, wanted) in expected.items():
            if found != wanted:
                raise ValueError(f"B3 {name} hash {found} differs from {wanted}")
    else:
        gamma_path = artifacts / B0_GAMMA_FILE
        gamma_digest = file_sha256(gamma_path)
        if gamma_digest != B0_GAMMA_SHA256:
            raise ValueError(f"{gamma_path} has SHA-256 {gamma_digest}, expected {B0_GAMMA_SHA256}")
        gamma = float(gamma_path.read_text(encoding="utf-8").strip())
        if gamma != settings.gamma_b0:
            raise ValueError(f"gamma_b0 {settings.gamma_b0!r} differs from {gamma_path}: {gamma!r}")
    return VerifiedInputs(sigma, kappa, A, CALIBRATION_SHA256, policy_path, policy_digest, gamma_digest)
