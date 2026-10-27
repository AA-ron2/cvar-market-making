"""run_log.py: the bots' re-anchor incidents appended to RUN_LOG.md (clarification g), once each, in time order."""

from __future__ import annotations

import pytest

from hummingbot_adapter import run_log

HEADER = "# Run log (UTC)\n\n2026-10-18T21:55:11Z: PROTOCOL.md committed.\n"
B0 = "2026-10-21T03:12:05Z: hb_b0 (production): incident: clock re-anchor (clarification g): x; minutes affected 0.04."
B3_EARLY = "2026-10-21T01:00:00Z: hb_b3 (production): incident: clock re-anchor (clarification g): y; minutes affected 0.05."
B3_LATE = "2026-10-21T04:00:00Z: hb_b3 (production): incident: clock re-anchor (clarification g): z; minutes affected 1.20."


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _setup(tmp_path, root=None):
    root = tmp_path / "hb_data" if root is None else root
    _write(root / "hb_b0" / run_log.INCIDENTS_FILE, B0 + "\n")
    _write(root / "hb_b3" / run_log.INCIDENTS_FILE, B3_LATE + "\n" + B3_EARLY + "\n")
    return tmp_path / "hb_data", _write(tmp_path / "RUN_LOG.md", HEADER)


def test_incidents_of_both_bots_are_appended_in_time_order_each_as_its_own_entry(tmp_path, capsys):
    data, log = _setup(tmp_path)
    assert run_log.main(["--data-root", str(data), "--run-log", str(log)]) == 0
    assert log.read_text(encoding="utf-8") == HEADER + f"\n{B3_EARLY}\n\n{B0}\n\n{B3_LATE}\n"
    assert "3 incident line(s) appended" in capsys.readouterr().out


def test_a_second_run_appends_nothing_and_a_new_incident_only_once(tmp_path):
    data, log = _setup(tmp_path)
    run_log.main(["--data-root", str(data), "--run-log", str(log)])
    before = log.read_text(encoding="utf-8")
    run_log.main(["--data-root", str(data), "--run-log", str(log)])
    assert log.read_text(encoding="utf-8") == before
    new = "2026-10-21T05:00:00Z: hb_b0 (production): incident: clock re-anchor (clarification g): w; minutes affected 0.03."
    with open(data / "hb_b0" / run_log.INCIDENTS_FILE, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(new + "\n")
    run_log.main(["--data-root", str(data), "--run-log", str(log)])
    assert log.read_text(encoding="utf-8") == before + f"\n{new}\n"


def test_dry_run_changes_nothing_and_smoke_reads_only_the_smoke_folders(tmp_path, capsys):
    data, log = _setup(tmp_path, root=tmp_path / "hb_data" / "smoke")
    assert run_log.main(["--data-root", str(data), "--run-log", str(log), "--dry-run"]) == 0
    assert "0 incident line(s) pending" in capsys.readouterr().out  # production folders: nothing
    assert run_log.main(["--data-root", str(data), "--smoke", "--run-log", str(log), "--dry-run"]) == 0
    assert "3 incident line(s) pending" in capsys.readouterr().out
    assert log.read_text(encoding="utf-8") == HEADER


def test_a_run_log_without_a_final_newline_is_refused(tmp_path):
    data, log = _setup(tmp_path)
    log.write_text(HEADER.rstrip("\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="newline"):
        run_log.main(["--data-root", str(data), "--run-log", str(log)])


def test_a_line_still_being_written_waits_for_the_next_run(tmp_path):
    data, log = _setup(tmp_path)
    partial = "2026-10-21T06:00:00Z: hb_b0 (production): incident: clock re-anchor (clarification g): half"
    with open(data / "hb_b0" / run_log.INCIDENTS_FILE, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(partial)  # no newline yet: the bot is mid-write
    run_log.main(["--data-root", str(data), "--run-log", str(log)])
    assert partial not in log.read_text(encoding="utf-8")
    with open(data / "hb_b0" / run_log.INCIDENTS_FILE, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("; minutes affected 0.01.\n")
    run_log.main(["--data-root", str(data), "--run-log", str(log)])
    assert log.read_text(encoding="utf-8").endswith(f"\n{partial}; minutes affected 0.01.\n")
