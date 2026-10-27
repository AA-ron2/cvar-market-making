"""D-14a size study: the null designs, the tables and the script's outputs, on a few replications."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pytest

from scripts.analysis import cvar_test_size_study as study


def test_the_eight_designs_are_the_work_order_s():
    assert study.DESIGNS == tuple(
        (phi, correlation, sigma) for phi in (0.0, 0.5) for correlation in (0.5, 0.9) for sigma in (0.5, 1.0)
    )
    assert (study.DAYS, study.ALPHA, study.BLOCK_LENGTH, study.SEED) == (23, 0.05, 3, 42)
    assert (study.REPLICATIONS, study.RANDOMIZATION_DRAWS, study.BOOTSTRAP_DRAWS) == (2_000, 2_000, 2_000)


@pytest.mark.parametrize("design", [(0.0, 0.5, 0.5), (0.5, 0.9, 1.0)])
def test_a_design_has_its_ar1_day_effect_and_within_day_correlation(design):
    """log X^m / sigma = D + e^m with D a unit-variance AR(1) (phi) and corr(e^A, e^B) = r: so corr(log X^A,
    log X^B) = (1 + r)/2, and the lag-1 autocorrelation of D + (e^A + e^B)/2 is phi / (1 + (1 + r)/2)."""

    phi, correlation, sigma = design
    first, second = study.simulate_pair(np.random.default_rng(5), phi, correlation, sigma, days=200_000)
    log_a, log_b = np.log(first) / sigma, np.log(second) / sigma
    assert np.var(log_a) == pytest.approx(2.0, abs=0.03) and np.var(log_b) == pytest.approx(2.0, abs=0.03)
    assert np.corrcoef(log_a, log_b)[0, 1] == pytest.approx((1.0 + correlation) / 2.0, abs=0.01)
    average = (log_a + log_b) / 2.0
    lag1 = np.corrcoef(average[:-1], average[1:])[0, 1]
    assert lag1 == pytest.approx(phi / (1.0 + (1.0 + correlation) / 2.0), abs=0.01)


RATES = (
    ("randomization_rejection_rate", "randomization_rejection_mcse"),
    ("bootstrap_rejection_rate", "bootstrap_rejection_mcse"),
    ("bootstrap_ci_coverage", "bootstrap_ci_coverage_mcse"),
)


def test_the_rates_do_not_depend_on_the_chunking():
    """Each replication draws from its own seeds (SEED, design, replication), so chunks change nothing."""

    one = study.size_study(3, 19, 39, workers=1, chunk=1)  # 39 bootstrap draws: the fewest for a 95 % interval is 20
    three = study.size_study(3, 19, 39, workers=1, chunk=3)
    assert one == three and len(one) == 8
    for row in one:
        assert tuple(row) == study.COLUMNS
        for value, mcse in RATES:
            assert 0.0 <= row[value] <= 1.0
            assert row[mcse] == pytest.approx(np.sqrt(row[value] * (1.0 - row[value]) / 3))
        # the bootstrap rejects at p < 0.05, the test its interval inverts (D-14)
        assert row["bootstrap_rejection_rate"] == pytest.approx(1.0 - row["bootstrap_ci_coverage"])


def test_the_script_writes_the_csv_the_booktabs_table_and_its_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvar_test_size_study.py", "--replications", "2", "--draws", "19", "--bootstrap-draws", "39",
        "--workers", "1", "--output-dir", str(tmp_path),
    ])
    study.main()
    with (tmp_path / "size_study.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [tuple(row) for row in rows] == [study.COLUMNS] * 8
    assert [(float(row["phi"]), float(row["within_day_correlation"]), float(row["sigma"])) for row in rows] == list(study.DESIGNS)
    assert {(row["replications"], row["randomization_draws"], row["bootstrap_draws"]) for row in rows} == {("2", "19", "39")}
    table = (tmp_path / "size_study_table.tex").read_text(encoding="utf-8")
    assert table.count(r"\toprule") == table.count(r"\midrule") == table.count(r"\bottomrule") == 1
    body = table.split("\\midrule\n")[1].split("    \\bottomrule")[0].strip().splitlines()
    assert len(body) == 8 and body[0].startswith("0.0 & 0.5 & 0.5 & ") and body[0].endswith(r"\\")
    metadata = json.loads((tmp_path / "size_study_metadata.json").read_text(encoding="utf-8"))
    assert (metadata["replications"], metadata["randomization_draws"], metadata["seed"]) == (2, 19, 42)
    assert metadata["rejection_rule"].startswith("randomization: p <= alpha") and metadata["wall_seconds"] >= 0
