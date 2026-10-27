"""Clear a bot's halt flag by hand (PROTOCOL §7: a halted bot posts nothing until the author clears the flag; a
container restart does not clear it). Stop the bot first, then:

    python -m hummingbot_adapter.clear_halt <bot dir>/state.json --by "<name>" [--note "<why>"]

It refuses while the state is younger than min_age_s (the bot saves it every second), appends the clearance to
halt_clearances.csv beside the state, and changes nothing else. Exit codes: 0 cleared, 1 not halted, 2 no state,
3 the bot looks alive.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hummingbot_adapter.execution import load_state, save_state

MIN_AGE_S = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state", type=Path)
    parser.add_argument("--by", required=True, help="who clears the flag (recorded)")
    parser.add_argument("--note", default="", help="why (recorded)")
    parser.add_argument("--min-age-s", type=float, default=MIN_AGE_S)
    args = parser.parse_args(argv)
    state = load_state(args.state)
    if state is None:
        print(f"no state file at {args.state}", file=sys.stderr)
        return 2
    safety = state["safety"]
    if not safety["halted"]:
        print(f"{state['bot']} is not halted")
        return 1
    age_s = (time.time_ns() // 1000 - state["saved_wall_us"]) / 1e6
    if age_s < args.min_age_s:
        print(f"refusing: the state was saved {age_s:.1f} s ago; stop the bot first", file=sys.stderr)
        return 3
    record = args.state.with_name("halt_clearances.csv")
    new = not record.exists()
    with open(record, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new:
            writer.writerow(["cleared_utc", "bot", "halt_reason", "halt_ts_us", "by", "note"])
        writer.writerow([datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), state["bot"],
                         safety["halt_reason"], safety["halt_ts_us"], args.by, args.note])
    reason = safety["halt_reason"]
    safety.update({"halted": False, "halt_reason": None, "halt_ts_us": None})
    state["status"].update({"halted": False, "halt_reason": None})  # what status.ps1 prints
    save_state(args.state, state)
    print(f"{state['bot']}: halt cleared ({reason}); recorded in {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
