"""Health probes of the forward-test bot that run beside the 0.1 s tick (WO-HB2 deliverables 2-3).

- ClockChecker: PROTOCOL §7's clock offset against Binance server time (GET /api/v3/time), hourly, in a
  daemon thread; results wait in a queue that the bot drains at its tick. Each check uses the sample with the
  smallest round trip of 5 (clarification i). While a check fails or the offset exceeds the warning level it
  repeats every retry_s, so a clock pause can lift once the offset is below 2 s.
- ReconnectWatcher: a logging handler for the order-book feed's disconnect, error and (re)subscribe messages
  (Hummingbot 2.17.0 binance_api_order_book_data_source / order_book_tracker_data_source), so the health
  log records the reconnects the bot observes.
- memory_rss_mb: the process's resident memory from /proc (None where /proc is missing).

Runtime module: standard library only (it runs inside the Hummingbot container).
"""

from __future__ import annotations

import http.client
import json
import logging
import queue
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"
FEED_LOGGERS = ("hummingbot.connector", "hummingbot.core.data_type.order_book_tracker")
FEED_MESSAGES = {
    "The websocket connection was closed": "ws_closed",
    "Unexpected error occurred when listening to order book streams": "ws_error",
    "Subscribed to public order book and trade channels": "subscribed",
    "Initialized order book for": "book_initialized",
    # a failed REST re-sync (clarification h; order_book_tracker_data_source.py): retried at the next 60 s wait
    "Unexpected error fetching order book snapshot": "snapshot_error",
    "Unexpected error when processing public order book snapshots": "snapshot_error",
}


def fetch_binance_server_ms(timeout_s: float = 10.0) -> int:
    with urllib.request.urlopen(BINANCE_TIME_URL, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict) or isinstance(body.get("serverTime"), bool) or not isinstance(body.get("serverTime"), int):
        raise ValueError(f"unexpected server-time body {body!r:.200}")
    return body["serverTime"]


class ClockChecker:
    """Offsets of the bot's clock (peek_us, the ts_us clock) and of the wall clock against the server. A check
    sends `samples` requests, discards those whose round trip exceeds max_rtt_s (timed on the monotonic clock,
    which a wall-clock step cannot distort), and uses the one with the smallest round trip at its midpoint;
    RTT/2 bounds its error (clarification i). A diagnostic, no trading rule: the step clock's rate against the
    server (rate_step = Δstep / Δserver - 1; with re-anchors it includes their corrections) and the monotonic
    clock's (rate_monotonic, without them; in the bot the step clock's own source, CLOCK_MONOTONIC_RAW), from
    the latest check at least rate_window_s before this one, and none until such a check exists (rate_over_s
    says over how long). With raw_us (clarification j: the step clock without its rate correction), the result
    also holds raw_offset_s, read at the same midpoint as offset_s, so their difference carries no network error."""

    def __init__(
        self,
        peek_us: Callable[[], int],
        wall_us: Callable[[], int],
        interval_s: float = 3_600.0,
        retry_s: float = 60.0,
        warn_s: float = 0.5,
        fetch_ms: Callable[[], int] = fetch_binance_server_ms,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        rate_window_s: float = 3_600.0,
        samples: int = 5,
        max_rtt_s: float = 1.0,
        raw_us: Callable[[], int] | None = None,
    ):
        if not 0 < retry_s <= interval_s:
            raise ValueError("retry_s must be positive and at most interval_s")
        if samples < 1 or not max_rtt_s > 0 or not rate_window_s > 0:
            raise ValueError("a check needs at least one sample, a positive max_rtt_s and a positive rate_window_s")
        self._peek_us, self._wall_us, self._fetch_ms, self._raw_us = peek_us, wall_us, fetch_ms, raw_us
        self._monotonic_ns = monotonic_ns
        self.interval_s, self.retry_s, self.warn_s, self.rate_window_s = interval_s, retry_s, warn_s, rate_window_s
        self.samples, self.max_rtt_s = samples, max_rtt_s
        self.results: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._history: deque[tuple[int, float, float]] = deque()  # (server µs, step-clock µs, monotonic µs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self) -> dict[str, Any]:
        responses, errors = [], []
        for _ in range(self.samples):
            bot0, raw0, wall0, mono0 = self._peek_us(), self._raw(), self._wall_us(), self._monotonic_ns()
            try:
                server_ms = self._fetch_ms()
            except (OSError, ValueError, KeyError, http.client.HTTPException) as error:
                # URLError and timeouts are OSErrors, a bad body a ValueError, a broken HTTP response an HTTPException
                errors.append(error)
                continue
            bot1, raw1, wall1, mono1 = self._peek_us(), self._raw(), self._wall_us(), self._monotonic_ns()
            responses.append({"server_ms": server_ms, "step_us": (bot0 + bot1) / 2, "wall_mid_us": (wall0 + wall1) / 2,
                              "raw_mid_us": None if raw0 is None else (raw0 + raw1) / 2,
                              "mono_us": (mono0 + mono1) / 2000, "rtt_s": (mono1 - mono0) / 1e9, "wall_us": wall1})
        rtts = [round(response["rtt_s"], 6) for response in responses]
        kept = [response for response in responses if response["rtt_s"] <= self.max_rtt_s]
        if not kept:
            if responses:
                error = f"no round trip <= {self.max_rtt_s} s among {len(responses)} responses"
                error += f"; {len(errors)} failed, the last {errors[-1]!r}" if errors else ""
            else:
                error = repr(errors[-1])
            return {"error": error, "wall_us": self._wall_us(), "rtts_s": rtts, "fetch_errors": len(errors)}
        best = min(kept, key=lambda response: response["rtt_s"])
        server_us = best["server_ms"] * 1000
        step_us, mono_us = best["step_us"], best["mono_us"]
        result = {
            "error": None,
            "server_ms": best["server_ms"],
            "offset_s": (step_us - server_us) / 1e6,
            "wall_offset_s": (best["wall_mid_us"] - server_us) / 1e6,
            "raw_offset_s": None if best["raw_mid_us"] is None else (best["raw_mid_us"] - server_us) / 1e6,
            "rtt_s": best["rtt_s"],
            "rtts_s": rtts,
            "fetch_errors": len(errors),
            "wall_us": best["wall_us"],
            "rate_step": None,
            "rate_monotonic": None,
            "rate_over_s": None,
        }
        reference = None
        for sample in self._history:  # oldest first: keep the latest one at least rate_window_s old
            if server_us - sample[0] >= self.rate_window_s * 1e6:
                reference = sample
        if reference is not None:
            span = server_us - reference[0]
            result.update(rate_step=(step_us - reference[1]) / span - 1, rate_monotonic=(mono_us - reference[2]) / span - 1,
                          rate_over_s=span / 1e6)
        self._history.append((server_us, step_us, mono_us))
        while self._history and server_us - self._history[0][0] > 2 * self.rate_window_s * 1e6:
            self._history.popleft()
        return result

    def _raw(self) -> int | None:
        return None if self._raw_us is None else self._raw_us()

    def _run(self) -> None:
        while True:
            result = self.check_once()
            self.results.put(result)
            settled = result["error"] is None and abs(result["offset_s"]) <= self.warn_s
            if self._stop.wait(self.interval_s if settled else self.retry_s):
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ClockChecker already started")
        self._thread = threading.Thread(target=self._run, name="clock-check", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def drain(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out


class ReconnectWatcher(logging.Handler):
    """Queues (kind, level, message) for the feed messages in FEED_MESSAGES; attach() adds it to the feed's
    loggers (hummingbot.connector does not propagate to the root logger)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: queue.SimpleQueue[tuple[str, str, str]] = queue.SimpleQueue()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for prefix, kind in FEED_MESSAGES.items():
            if message.startswith(prefix):
                self.events.put((kind, record.levelname, message[:300]))
                return

    def attach(self, names: tuple[str, ...] = FEED_LOGGERS) -> None:
        for name in names:
            logging.getLogger(name).addHandler(self)

    def detach(self, names: tuple[str, ...] = FEED_LOGGERS) -> None:
        for name in names:
            logging.getLogger(name).removeHandler(self)

    def drain(self) -> list[tuple[str, str, str]]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out


def memory_rss_mb(status: str | Path = "/proc/self/status") -> float | None:
    path = Path(status)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None
