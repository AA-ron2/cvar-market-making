"""Append the bots' clock incidents to RUN_LOG.md (clarification g: every re-anchor of the step clock is a
RUN_LOG.md incident with its UTC time, bot and minutes affected; clarification i: so is every clock_backward
episode, a step clock more than 2 s ahead of the wall clock; PROTOCOL §8).

The repository is mounted read-only in the containers, so each bot writes its incident lines to
<bot dir>/run_log_incidents.md (bot_core.BotCore._incident). This tool appends to RUN_LOG.md, in time order and
each as its own entry, every such line that RUN_LOG.md does not hold yet, and changes nothing else:

    python -m hummingbot_adapter.run_log --data-root <HB_DATA> [--smoke] [--dry-run]

It prints the lines it appended (or would append). Commit RUN_LOG.md afterwards ([documentation]).
Runtime module: standard library only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

INCIDENTS_FILE = "run_log_incidents.md"
BOTS = ("hb_b0", "hb_b3")
RUN_LOG = Path(__file__).resolve().parent / "RUN_LOG.md"


def incident_lines(data_root: str | Path, smoke: bool = False) -> list[str]:
    """Every bot's incident lines, in time order (each line starts with its UTC time)."""

    root = Path(data_root) / "smoke" if smoke else Path(data_root)
    lines = []
    for bot in BOTS:
        path = root / bot / INCIDENTS_FILE
        if path.exists():
            text = path.read_text(encoding="utf-8")
            complete = text.splitlines()
            if not text.endswith("\n") and complete:  # a line the bot is still writing waits for the next run
                complete.pop()
            lines += [line for line in complete if line.strip()]
    return sorted(lines)


def pending(lines: list[str], run_log: str | Path) -> list[str]:
    present = set(Path(run_log).read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line not in present]


def append(lines: list[str], run_log: str | Path) -> None:
    """Each line as its own entry, after a blank line, as RUN_LOG.md separates its entries."""

    path = Path(run_log)
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError(f"{path} does not end with a newline")
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(f"\n{line}\n" for line in lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="read <data root>/smoke (never needed for the run)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-log", type=Path, default=RUN_LOG)
    args = parser.parse_args(argv)
    lines = pending(incident_lines(args.data_root, args.smoke), args.run_log)
    if lines and not args.dry_run:
        append(lines, args.run_log)
    for line in lines:
        print(("would append: " if args.dry_run else "appended: ") + line)
    print(f"{len(lines)} incident line(s) {'pending' if args.dry_run else 'appended'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
