"""Run outputs of the forward test, per bot and UTC day (PROTOCOL §11; WO-HB2 deliverable 2).

Each day's folder <bot dir>/<date>/ holds six CSV files, written with newline="\\n" through in-memory buffers
(the 0.1 s tick only appends strings; flush() writes them, at least once per second):
- snapshots_<date>.csv: the Tardis book_snapshot_25 layout, column for column (SNAPSHOT_HEADER is the header of
  the Jan 7 2025 Tardis file); prices and amounts as Tardis prints them (shortest decimal, no ".0"); a missing
  level is two empty fields;
- steps_<date>.csv, orders_<date>.csv, fills_<date>.csv, trades_<date>.csv, health_<date>.csv.
At the day's end finalize_day() gzips each file (deterministic gzip header), deletes the plain file and writes
SHA256SUMS in the day's folder; the bot folder's SHA256SUMS lists every day's files. A finalized day is never
written again: DayLog refuses to open it and finalize_day never replaces a .csv.gz with other content. After a
restart, DayLog truncates each file to the size the state recorded at its last save, so the files hold exactly
the steps that the restored state has seen (PROTOCOL §5.4); a day started afresh sets aside (".orphan-<n>") any
file a dead process left for it.

Runtime module: standard library, numpy and cvar_mm.gym.formula_as only (it runs inside the Hummingbot container).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES

KINDS = ("snapshots", "steps", "orders", "fills", "trades", "health")
SNAPSHOT_LEVELS = 25
SNAPSHOT_HEADER = ("exchange", "symbol", "timestamp", "local_timestamp") + tuple(
    f"{side}[{level}].{field}"
    for level in range(SNAPSHOT_LEVELS)
    for side in ("asks", "bids")
    for field in ("price", "amount")
)
EXCHANGE, SYMBOL = "binance", "DOGEUSDT"

STEP_HEAD = ("ts_us", "utc", "phase", "dt", "t_elapsed", "tau", "mid")
B3_STEP_FIELDS = (
    tuple(FORMULA_FEATURE_NAMES)
    + tuple(f"z_{name}" for name in FORMULA_FEATURE_NAMES)
    + ("action_gamma", "action_skew")
)
STEP_TAIL = (
    "gamma", "skew", "bid_target", "ask_target", "bid_quote", "ask_quote", "bid_rounded", "ask_rounded",
    "resting_bid", "resting_ask", "q", "cash", "wealth", "peak", "drawdown", "fees_doge", "fees_usdt",
    "doge_balance", "residual_doge", "quoting", "stale_paused", "clock_paused", "drawdown_paused", "halted",
)
ORDER_HEADER = ("ts_us", "utc", "event", "order_id", "side", "order_type", "price", "amount", "mid_at_post", "reason")
FILL_HEADER = (
    "ts_us", "utc", "order_id", "side", "order_type", "price", "amount", "mid_at_fill", "delta_fill_ticks",
    "post_ts_us", "mid_at_post", "delta_post_ticks", "fee", "fee_asset", "close", "exchange_trade_id",
)
TRADE_HEADER = ("ts_us", "utc", "exchange_ts", "trade_id", "price", "amount", "side")
HEALTH_HEADER = ("ts_us", "utc", "event", "detail")
SUMS_NAME = "SHA256SUMS"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def step_header(model: str) -> tuple[str, ...]:
    """B3's steps add the 7 raw observations, their z-scores and the raw action (WO-HB2 deliverable 2)."""

    if model == "B3":
        return STEP_HEAD + B3_STEP_FIELDS + STEP_TAIL
    if model == "B0":
        return STEP_HEAD + STEP_TAIL
    raise ValueError(f"model must be B0 or B3, got {model!r}")


def headers(model: str) -> dict[str, tuple[str, ...]]:
    return {
        "snapshots": SNAPSHOT_HEADER,
        "steps": step_header(model),
        "orders": ORDER_HEADER,
        "fills": FILL_HEADER,
        "trades": TRADE_HEADER,
        "health": HEALTH_HEADER,
    }


# --- formatting --------------------------------------------------------------------------------------------------


def _int_us(name: str, value: Any) -> int:
    """A timestamp in integer µs; refuses bools and non-integral values (never truncates)."""

    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError(f"{name} must be integer microseconds, got {value!r}")
    return int(value)


def utc_iso(ts_us: int) -> str:
    """ISO 8601 UTC with microseconds, exact for int µs."""

    return (_EPOCH + timedelta(microseconds=_int_us("ts_us", ts_us))).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def fmt_book(value: float) -> str:
    """A book price or amount as Tardis prints it: whole numbers without ".0", others as the shortest
    round-trip decimal, so the frozen loader reads back the identical float64."""

    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"book values must be finite and non-negative, got {value!r}")
    if number.is_integer() and abs(number) < 2**53:
        return str(int(number))
    text = repr(number)
    if "e" in text or "E" in text:
        raise ValueError(f"{value!r} would print in exponent notation")
    return text


def fmt_float(value: Any) -> str:
    """A float as its shortest round-trip repr; float32 values are widened exactly first. None prints empty."""

    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"telemetry values must be finite, got {value!r}")
    return repr(number)


def dec_text(value: Decimal) -> str:
    """A Decimal as fixed-point text without trailing zeros, every digit kept (Hummingbot quantizes amounts to
    1e-7: 100.0000000 -> 100, 0E-7 -> 0; a market fill's binary-expansion price keeps all its digits).
    Decimal.normalize() is not used: it rounds to the context's 28 digits."""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"expected a finite Decimal, got {value!r}")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def fmt_value(value: Any) -> str:
    """Strings, ints and Decimals as text (Decimals via dec_text), bools as 1/0, floats (numpy's too) via
    fmt_float, None empty. Refuses text that would need CSV quoting."""

    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return fmt_float(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, Decimal):
        return dec_text(value)
    text = str(value)
    if any(character in text for character in ',"\n\r'):
        raise ValueError(f"field {text!r} would need CSV quoting")
    return text


def snapshot_fields(ts_us: int, local_us: int, asks: Sequence[tuple[float, float]],
                    bids: Sequence[tuple[float, float]]) -> list[str]:
    """One Tardis book_snapshot_25 row: best level first on each side; a missing level is two empty fields."""

    if len(asks) > SNAPSHOT_LEVELS or len(bids) > SNAPSHOT_LEVELS:
        raise ValueError(f"at most {SNAPSHOT_LEVELS} levels per side")
    fields = [EXCHANGE, SYMBOL, str(_int_us("ts_us", ts_us)), str(_int_us("local_us", local_us))]
    for level in range(SNAPSHOT_LEVELS):
        for side in (asks, bids):
            if level < len(side):
                price, amount = side[level]
                fields += [fmt_book(price), fmt_book(amount)]
            else:
                fields += ["", ""]
    return fields


# --- one day's files ---------------------------------------------------------------------------------------------


def is_finalized(folder: str | Path, date: str) -> bool:
    folder = Path(folder)
    return (folder / SUMS_NAME).exists() or any(folder.glob(f"*_{date}.csv.gz"))


class DayLog:
    """The six CSV files of one bot and UTC day, appended through buffers. A file that exists must start with
    the expected header. With `sizes` (the state's record at its last save), each file is first truncated to
    its recorded size; with fresh=True (a day that starts now), a non-empty file a dead process left is set
    aside as <name>.orphan-<n>. A finalized day (SHA256SUMS or a .csv.gz present) is refused.
    truncations, short_files and orphaned list what was found, for the health log."""

    def __init__(self, folder: str | Path, date: str, file_headers: Mapping[str, Sequence[str]],
                 sizes: Mapping[str, int] | None = None, fresh: bool = False):
        if set(file_headers) != set(KINDS):
            raise ValueError(f"headers for {sorted(file_headers)}, expected {sorted(KINDS)}")
        if sizes is not None and fresh:
            raise ValueError("a fresh day has no recorded sizes")
        self.folder = Path(folder)
        self.date = date
        if is_finalized(self.folder, date):
            raise ValueError(f"day {date} in {self.folder} is finalized; its files are never written again")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.paths = {kind: self.folder / f"{kind}_{date}.csv" for kind in KINDS}
        self._widths = {kind: len(file_headers[kind]) for kind in KINDS}
        self.truncations: list[tuple[str, int, int]] = []
        self.short_files: list[tuple[str, int, int]] = []
        self.orphaned: list[str] = []
        self._buffers: dict[str, list[str]] = {kind: [] for kind in KINDS}
        self._handles: dict[str, io.TextIOWrapper] = {}
        try:
            for kind in KINDS:
                if fresh:
                    self._set_aside(kind)
                self._open(kind, tuple(file_headers[kind]), None if sizes is None else sizes.get(kind))
        except BaseException:
            self.close()
            raise

    def _set_aside(self, kind: str) -> None:
        path = self.paths[kind]
        if not path.exists() or path.stat().st_size == 0:
            return
        number = 1
        while path.with_name(f"{path.name}.orphan-{number}").exists():
            number += 1
        target = path.with_name(f"{path.name}.orphan-{number}")
        os.replace(path, target)
        self.orphaned.append(target.name)

    def _open(self, kind: str, header: tuple[str, ...], recorded: int | None) -> None:
        path = self.paths[kind]
        header_line = ",".join(header) + "\n"
        size = path.stat().st_size if path.exists() else 0
        if recorded is not None and size > recorded:
            with open(path, "r+b") as handle:
                handle.truncate(recorded)
            self.truncations.append((kind, size, recorded))
            size = recorded
        elif recorded is not None and size < recorded:
            self.short_files.append((kind, size, recorded))
        if size:
            with open(path, encoding="utf-8", newline="\n") as handle:
                first = handle.readline()
            if first != header_line:
                raise ValueError(f"{path} starts with another header")
        handle = open(path, "a", encoding="utf-8", newline="\n")
        if not size:
            handle.write(header_line)
        self._handles[kind] = handle

    def add(self, kind: str, fields: Sequence[Any]) -> None:
        if len(fields) != self._widths[kind]:
            raise ValueError(f"{kind} row has {len(fields)} fields, the header {self._widths[kind]}")
        self._buffers[kind].append(",".join(fmt_value(field) for field in fields) + "\n")

    def add_health(self, ts_us: int, event: str, /, **detail: Any) -> None:
        """A health row; detail is one JSON object in a quoted CSV field (any key, "event" included)."""

        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(
            [str(_int_us("ts_us", ts_us)), utc_iso(ts_us), event, json.dumps(detail, sort_keys=True, default=str)]
        )
        self._buffers["health"].append(buffer.getvalue())

    def flush(self) -> dict[str, int]:
        """Write the buffers and return each file's size in bytes (what the state records). A buffer is
        cleared only after its write succeeded, so a failed flush can be retried."""

        sizes = {}
        for kind in KINDS:
            handle = self._handles[kind]
            if self._buffers[kind]:
                handle.write("".join(self._buffers[kind]))
                self._buffers[kind].clear()
            handle.flush()
            sizes[kind] = handle.tell()
        return sizes

    def close(self) -> None:
        for kind, handle in list(self._handles.items()):
            if self._buffers.get(kind):
                handle.write("".join(self._buffers[kind]))
                self._buffers[kind].clear()
            handle.close()
        self._handles.clear()


# --- the day's end ---------------------------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _same_content(plain: Path, packed: Path, chunk_size: int = 1 << 20) -> bool:
    with open(plain, "rb") as left, gzip.open(packed, "rb") as right:
        while True:
            a, b = left.read(chunk_size), right.read(chunk_size)
            if a != b:
                return False
            if not a:
                return True


def finalize_day(folder: str | Path, date: str) -> list[str]:
    """Gzip each of the day's CSV files (mtime 0, no file name in the header; data fsynced before the rename),
    delete the plain file, write the day's SHA256SUMS over every file in the folder, and rebuild the bot
    folder's SHA256SUMS. Safe to run again after an interrupted run. Never replaces an existing .csv.gz: if a
    plain file and its .csv.gz both exist, the plain file is deleted when the two hold the same bytes, and a
    FileExistsError is raised when they differ. Returns the day's SHA256SUMS lines."""

    folder = Path(folder)
    for kind in KINDS:
        plain = folder / f"{kind}_{date}.csv"
        if not plain.exists():
            continue
        packed = folder / f"{kind}_{date}.csv.gz"
        if packed.exists():
            if not _same_content(plain, packed):
                raise FileExistsError(f"{packed} exists with other content than {plain}; neither is changed")
            plain.unlink()
            continue
        temporary = folder / f"{kind}_{date}.csv.gz.tmp"
        with open(plain, "rb") as source, open(temporary, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as target:
                while chunk := source.read(1 << 20):
                    target.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, packed)
        plain.unlink()
    names = sorted(path.name for path in folder.iterdir()
                   if path.is_file() and path.name != SUMS_NAME and not path.name.endswith(".tmp"))
    lines = [f"{file_sha256(folder / name)}  {name}" for name in names]
    _replace_text(folder / SUMS_NAME, "".join(line + "\n" for line in lines))
    rebuild_bot_sums(folder.parent)
    return lines


def rebuild_bot_sums(bot_folder: str | Path) -> None:
    """The bot folder's SHA256SUMS: every finalized day's lines, with the day folder in the path."""

    bot_folder = Path(bot_folder)
    lines = []
    for sums in sorted(bot_folder.glob(f"*/{SUMS_NAME}")):
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            lines.append(f"{digest}  {sums.parent.name}/{name}")
    _replace_text(bot_folder / SUMS_NAME, "".join(line + "\n" for line in lines))


class Finalizer:
    """Runs finalize_day off the event loop: submit() queues a day and returns at once; one worker thread
    finalizes the queued days in order. drain() hands the results and errors to the bot's health log."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[Path, str]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._done: list[tuple[str, list[str]]] = []
        self._errors: list[tuple[str, str]] = []

    def submit(self, folder: str | Path, date: str) -> None:
        with self._lock:
            self._pending.append((Path(folder), date))
            if not self._running:  # set and cleared under the lock, so no queued day is ever left behind
                self._running = True
                self._thread = threading.Thread(target=self._work, name="finalize", daemon=False)
                self._thread.start()

    def _work(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._running = False
                    return
                folder, date = self._pending.pop(0)
            try:
                lines = finalize_day(folder, date)
            except Exception as error:  # reported through the bot's health log, never silent
                with self._lock:
                    self._errors.append((date, repr(error)))
            else:
                with self._lock:
                    self._done.append((date, lines))

    def drain(self) -> tuple[list[tuple[str, list[str]]], list[tuple[str, str]]]:
        """The results and errors since the last drain (taken under the lock, so none is lost)."""

        with self._lock:
            done, errors = self._done, self._errors
            self._done, self._errors = [], []
        return done, errors

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
