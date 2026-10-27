"""Docker VM clock logger for the clarification-j drift estimate (runs beside the bots, touches nothing of theirs).
Every 60 s one JSON line with CLOCK_REALTIME, CLOCK_MONOTONIC_RAW and CLOCK_MONOTONIC (ns, read back to back);
every 600 s also a Binance GET /api/v3/time check: 5 requests on one kept-alive HTTPS connection (warmed by a
discarded request), the one with the smallest round trip at its midpoint (round trip timed on RAW). Appends to
/log/clocklog.jsonl; standard library only."""
import http.client
import json
import time

PATH = "/log/clocklog.jsonl"


def clocks():
    w0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
    raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    w1 = time.clock_gettime_ns(time.CLOCK_REALTIME)
    return {"wall_ns": (w0 + w1) // 2, "raw_ns": raw, "mono_ns": mono}


def binance():
    conn = http.client.HTTPSConnection("api.binance.com", 443, timeout=10)
    best = None
    try:
        for i in range(6):
            r0, w0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.time_ns()
            conn.request("GET", "/api/v3/time")
            response = conn.getresponse()
            body = response.read()
            r1, w1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.time_ns()
            if response.status != 200 or i == 0:  # the first request opens TCP and TLS: discarded
                continue
            sample = {"server_ms": json.loads(body)["serverTime"], "rtt_ns": r1 - r0,
                      "raw_mid_ns": (r0 + r1) // 2, "wall_mid_ns": (w0 + w1) // 2}
            if best is None or sample["rtt_ns"] < best["rtt_ns"]:
                best = sample
    finally:
        conn.close()
    return best


last_check = 0.0
while True:
    row = {"kind": "clocks", **clocks()}
    if time.monotonic() - last_check >= 600 or last_check == 0.0:
        last_check = time.monotonic()
        try:
            row["binance"] = binance()
        except (OSError, ValueError, KeyError, http.client.HTTPException) as error:
            row["binance_error"] = repr(error)
    with open(PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    time.sleep(60)
