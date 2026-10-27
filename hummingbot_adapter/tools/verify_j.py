"""Clarification j verification (read-only), the author's rule of 2026-10-21 (~11:00Z), for every clock check of both
bots since a restart onto j:
  (i)  mechanism: |(raw-offset slope - corrected-offset slope) - c| <= 0.1 ppm between consecutive checks (the two
       offsets come from the same request and midpoint, so the network error cancels);
  (ii) 0 re-anchors, 0 clock_backward, |step offset| <= 0.5 s at every check.
For information only (not pass/fail): the hourly residual rate (raw rate - c, against Binance) and its projection in
s/day, the wall clock's offset and the step-wall slope (valid while the host clock is disciplined, clarification l).
Usage: python -I verify_j.py <hb_data> <restart utc, e.g. 2026-10-21T11:35:00Z> <c ppm, e.g. 13.92>"""
import csv
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root, since, c = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
OFFSET_MAX, MECHANISM_MAX = 0.5, 0.1


def rows(path):
    for _ in range(50):
        try:
            with open(path, encoding="utf-8", newline="") as handle:
                return list(csv.reader(handle))[1:]
        except PermissionError:
            time.sleep(0.1)
    raise PermissionError(path)


verdict = True
worst = None
for bot in ("hb_b0", "hb_b3"):
    health = []
    for day_dir in sorted(p for p in (root / bot).iterdir() if p.is_dir() and p.name[:2] == "20"):
        health += [r for r in rows(day_dir / f"health_{day_dir.name}.csv") if r[1] >= since]
    checks = [json.loads(r[3]) | {"utc": r[1]} for r in health if r[2] == "clock_check"]
    jumps = [r for r in health if r[2] == "deviation" and '"clock_jump"' in r[3]]
    backward = [r for r in health if r[2] == "incident" and '"clock_backward"' in r[3]]
    warns = [r for r in health if r[2] in ("warn", "pause") and "clock_offset" in r[3]]
    starts = [r[1] for r in health if r[2] in ("start", "restart")]
    good = [check for check in checks if check["error"] is None]
    print(f"== {bot}: starts {[s[11:19] + 'Z' for s in starts]}; {len(checks)} clock checks ({len(good)} good);"
          f" re-anchors {len(jumps)}; clock_backward {len(backward)}; clock warn/pause rows {len(warns)}")
    verdict &= not jumps and not backward and len(starts) <= 1
    previous = None
    for check in checks:
        if check["error"] is not None:
            print(f"   {check['utc'][11:19]}Z ERROR {check['error']}")
            continue
        raw = check.get("raw_offset_s")
        ok = abs(check["offset_s"]) <= OFFSET_MAX and raw is not None
        verdict &= ok
        text = (f"   {check['utc'][11:19]}Z offset {check['offset_s']:+.4f} s (uncorrected {raw:+.4f} s, wall"
                f" {check['wall_offset_s']:+.4f} s, step-wall {check['offset_s'] - check['wall_offset_s']:+.4f} s),"
                f" rtt {check['rtt_s']:.3f} s; {'OK' if ok else 'FAIL'}")
        if previous is not None:
            span = (check["server_ms"] - previous["server_ms"]) / 1000
            raw_rate = (raw - previous["raw_offset_s"]) / span * 1e6
            step_rate = (check["offset_s"] - previous["offset_s"]) / span * 1e6
            step_wall = ((check["offset_s"] - check["wall_offset_s"])
                         - (previous["offset_s"] - previous["wall_offset_s"])) / span * 1e6
            mechanism = raw_rate - step_rate - c
            residual = raw_rate - c
            mech_ok = abs(mechanism) <= MECHANISM_MAX
            verdict &= mech_ok
            if span >= 1800 and (worst is None or abs(residual) > abs(worst[0])):
                worst = (residual, bot, previous["utc"][11:19], check["utc"][11:19])
            text += (f"\n      over {span:.0f} s: raw rate {raw_rate:+.2f} ppm, corrected {step_rate:+.2f} ppm,"
                     f" (i) raw - corrected - c = {mechanism:+.4f} ppm {'OK' if mech_ok else 'FAIL'};"
                     f" residual {residual:+.2f} ppm = {residual * 0.0864:+.3f} s/day; step-wall slope {step_wall:+.2f} ppm")
        print(text)
        previous = check | {"raw_offset_s": raw}
if worst is not None:
    print(f"worst hour (>= 30 min): residual {worst[0]:+.2f} ppm = {worst[0] * 0.0864:+.3f} s/day ({worst[1]},"
          f" {worst[2]}Z-{worst[3]}Z), information only")
print("VERDICT so far ((i) and (ii)):", "PASS" if verdict else "FAIL")
