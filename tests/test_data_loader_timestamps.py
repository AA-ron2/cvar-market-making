from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

from cvar_mm.gym import calibration, data_loader
from cvar_mm.gym.calibration import ASParameters
from cvar_mm.gym.data_loader import (
    load_single_day,
    load_single_day_calibration_arrays,
    load_single_day_with_features,
    read_market_data_csv,
    stable_order_by_exchange_timestamp,
    timestamps_us_to_datetime_and_deltas,
)


BASE_TIMESTAMP_US = 1_735_689_601_703_988


def _write_market_csv(tmp_path, timestamps: list[object]):
    path = tmp_path / "synthetic.csv"
    row_count = len(timestamps)
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "asks[0].price": 100.01 + np.arange(row_count) * 0.01,
            "bids[0].price": 99.99 + np.arange(row_count) * 0.01,
            "asks[0].amount": np.ones(row_count),
            "bids[0].amount": np.ones(row_count),
        }
    ).to_csv(path, index=False)
    return path


def _load_all_variants(path):
    _, plain_dt, plain_index = load_single_day(str(path))
    _, calibration_dt, calibration_index, _, _ = (
        load_single_day_calibration_arrays(str(path))
    )
    _, feature_dt, feature_index, _ = load_single_day_with_features(str(path))
    return (
        (plain_dt, plain_index),
        (calibration_dt, calibration_index),
        (feature_dt, feature_index),
    )


def test_ordered_frame_is_returned_without_reordering_or_warning() -> None:
    frame = pd.DataFrame(
        {"marker": ["first", "second", "third"]},
        index=pd.Index([1_000_000, 2_000_000, 3_000_000], name="timestamp"),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ordered = stable_order_by_exchange_timestamp(frame)

    assert ordered is frame
    assert ordered["marker"].tolist() == ["first", "second", "third"]
    assert caught == []
    _, dt = timestamps_us_to_datetime_and_deltas(ordered.index)
    assert dt.tolist() == [0.0, 1.0, 1.0]


def test_adjacent_inversion_stably_reorders_complete_rows() -> None:
    frame = pd.DataFrame(
        {"marker": ["one", "three", "two", "four"]},
        index=pd.Index(
            [1_000_000, 3_000_000, 2_000_000, 4_000_000],
            name="timestamp",
        ),
    )

    with pytest.warns(
        RuntimeWarning,
        match=(
            r"Detected 1 out-of-order exchange timestamp transition "
            r"\(max backward step 1000000 us\)"
        ),
    ):
        ordered = stable_order_by_exchange_timestamp(frame)

    assert ordered.index.tolist() == [1_000_000, 2_000_000, 3_000_000, 4_000_000]
    assert ordered["marker"].tolist() == ["one", "two", "three", "four"]
    _, dt = timestamps_us_to_datetime_and_deltas(ordered.index)
    assert dt.tolist() == [0.0, 1.0, 1.0, 1.0]


def test_duplicate_timestamps_keep_relative_order_and_zero_delta(tmp_path) -> None:
    frame = pd.DataFrame(
        {"marker": ["three", "first duplicate", "one", "second duplicate"]},
        index=pd.Index(
            [3_000_000, 2_000_000, 1_000_000, 2_000_000],
            name="timestamp",
        ),
    )

    with pytest.warns(RuntimeWarning):
        ordered = stable_order_by_exchange_timestamp(frame)

    assert ordered["marker"].tolist() == [
        "one",
        "first duplicate",
        "second duplicate",
        "three",
    ]
    path = _write_market_csv(tmp_path, frame.index.tolist())
    with pytest.warns(RuntimeWarning):
        _, dt, _ = load_single_day(str(path))
    assert dt.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_small_exchange_timestamp_inversion_is_sorted_not_clipped(tmp_path) -> None:
    timestamps = [
        BASE_TIMESTAMP_US,
        BASE_TIMESTAMP_US + 2_000,
        BASE_TIMESTAMP_US + 1_263,
        BASE_TIMESTAMP_US + 3_000,
    ]
    path = _write_market_csv(tmp_path, timestamps)

    with pytest.warns(RuntimeWarning, match=r"max backward step 737 us"):
        _, dt, index = load_single_day(str(path))

    expected_index = pd.to_datetime(sorted(timestamps), unit="us")
    assert index.equals(expected_index)
    assert dt.tolist() == [0.0, 0.001263, 0.000737, 0.001]


def test_all_loader_paths_agree_after_stable_timestamp_ordering(tmp_path) -> None:
    timestamps = [1_000_000, 3_000_000, 2_000_000, 4_000_000]
    path = _write_market_csv(tmp_path, timestamps)

    with pytest.warns(RuntimeWarning) as caught:
        plain_mid, plain_dt, plain_index = load_single_day(str(path))
        calibration_mid, calibration_dt, calibration_index, bid, ask = (
            load_single_day_calibration_arrays(str(path))
        )
        feature_mid, feature_dt, feature_index, _ = load_single_day_with_features(
            str(path)
        )

    assert len(caught) == 3
    expected_dt = [0.0, 1.0, 1.0, 1.0]
    expected_mid = [100.0, 100.02, 100.01, 100.03]
    expected_bid = [99.99, 100.01, 100.0, 100.02]
    expected_ask = [100.01, 100.03, 100.02, 100.04]
    assert plain_dt.tolist() == expected_dt
    np.testing.assert_array_equal(calibration_dt, plain_dt)
    np.testing.assert_array_equal(feature_dt, plain_dt)
    np.testing.assert_array_equal(calibration_index, plain_index)
    np.testing.assert_array_equal(feature_index, plain_index)
    assert plain_mid.tolist() == pytest.approx(expected_mid)
    assert calibration_mid.tolist() == pytest.approx(expected_mid)
    assert feature_mid.tolist() == pytest.approx(expected_mid)
    assert bid.tolist() == pytest.approx(expected_bid)
    assert ask.tolist() == pytest.approx(expected_ask)


def test_one_second_raw_microsecond_delta_is_not_compressed(tmp_path) -> None:
    path = _write_market_csv(
        tmp_path,
        [BASE_TIMESTAMP_US, BASE_TIMESTAMP_US + 1_000_000],
    )

    for dt, _ in _load_all_variants(path):
        assert dt.tolist() == [0.0, 1.0]
        assert dt[1] != 0.001


def test_nearly_24_hour_span_and_loader_deltas_agree(tmp_path) -> None:
    span_us = 86_397_972_303
    timestamps = [
        BASE_TIMESTAMP_US,
        BASE_TIMESTAMP_US + 123_456,
        BASE_TIMESTAMP_US + 1_123_456,
        BASE_TIMESTAMP_US + span_us,
    ]
    path = _write_market_csv(tmp_path, timestamps)

    variants = _load_all_variants(path)
    expected = np.diff(timestamps, prepend=timestamps[0]) / 1_000_000.0
    for dt, _ in variants:
        np.testing.assert_array_equal(dt, expected)
        assert dt.sum() == pytest.approx(86_397.972303)
    np.testing.assert_array_equal(variants[0][0], variants[1][0])
    np.testing.assert_array_equal(variants[0][0], variants[2][0])


def test_elapsed_seconds_are_independent_of_datetime_microsecond_resolution(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_market_csv(
        tmp_path,
        [BASE_TIMESTAMP_US, BASE_TIMESTAMP_US + 1_000_000],
    )
    pandas_to_datetime = pd.to_datetime

    def to_datetime_us(*args, **kwargs):
        return pandas_to_datetime(*args, **kwargs).as_unit("us")

    monkeypatch.setattr(data_loader.pd, "to_datetime", to_datetime_us)

    for dt, index in _load_all_variants(path):
        assert str(index.dtype) == "datetime64[us]"
        assert dt.tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ([BASE_TIMESTAMP_US, "not-a-timestamp"], "numeric integer microseconds"),
        ([1_000_000, "1_000_000"], "numeric integer microseconds"),
        ([1_000_000, 1_000_000.5], "finite integer microseconds"),
        ([BASE_TIMESTAMP_US, f"{BASE_TIMESTAMP_US}.1"], "finite integer microseconds"),
        ([1_000_000, np.iinfo(np.int64).max + 1], "int64 microsecond range"),
        ([1_000_000, np.iinfo(np.uint64).max], "int64 microsecond range"),
        ([1_000_000, np.iinfo(np.int64).min - 1], "int64 microsecond range"),
        ([1_000_000, "-9.223372036854776e18"], "int64 microsecond range"),
    ],
)
def test_invalid_timestamps_are_rejected_without_sorting(
    tmp_path, timestamps: list[object], message: str
) -> None:
    path = _write_market_csv(tmp_path, timestamps)

    with pytest.raises(ValueError, match=message):
        load_single_day(str(path))


@pytest.mark.parametrize(
    "raw_timestamp",
    [
        BASE_TIMESTAMP_US,
        np.iinfo(np.int64).max,
        np.iinfo(np.int64).max + 1,
        np.iinfo(np.uint64).max,
        np.iinfo(np.int64).min - 1,
    ],
)
def test_market_csv_read_preserves_exact_timestamp_text(tmp_path, raw_timestamp) -> None:
    path = _write_market_csv(tmp_path, [str(raw_timestamp)])

    frame = read_market_data_csv(str(path))

    assert frame.index.tolist() == [str(raw_timestamp)]
    assert isinstance(frame.index[0], str)
    if np.iinfo(np.int64).min <= raw_timestamp <= np.iinfo(np.int64).max:
        assert data_loader._validated_timestamps_us(frame.index).tolist() == [raw_timestamp]
    else:
        with pytest.raises(ValueError, match="int64 microsecond range"):
            data_loader._validated_timestamps_us(frame.index)


def test_all_market_csv_paths_reject_unsigned_overflow(tmp_path) -> None:
    path = _write_market_csv(tmp_path, [1_000_000, np.iinfo(np.uint64).max])

    for loader in (
        load_single_day,
        load_single_day_calibration_arrays,
        load_single_day_with_features,
    ):
        with pytest.raises(ValueError, match="int64 microsecond range"):
            loader(str(path))
    with pytest.raises(ValueError, match="int64 microsecond range"):
        calibration.calibrate_as_parameters(str(path), plot=False, verbose=False)


def test_integer_valued_timestamp_text_preserves_single_microsecond(tmp_path) -> None:
    raw_timestamp = "9007199254740993.0"
    path = _write_market_csv(tmp_path, [raw_timestamp])

    frame = read_market_data_csv(str(path))

    assert frame.index.tolist() == [raw_timestamp]
    assert data_loader._validated_timestamps_us(frame.index).tolist() == [
        9_007_199_254_740_993
    ]


def test_uint64_timestamps_are_checked_before_signed_conversion() -> None:
    signed_max = np.iinfo(np.int64).max
    valid = pd.Index(
        np.array([BASE_TIMESTAMP_US, np.uint64(signed_max)], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        data_loader._validated_timestamps_us(valid),
        np.array([BASE_TIMESTAMP_US, signed_max], dtype=np.int64),
    )

    for invalid in (np.uint64(signed_max + 1), np.uint64(np.iinfo(np.uint64).max)):
        with pytest.raises(ValueError, match="int64 microsecond range"):
            data_loader._validated_timestamps_us(
                pd.Index(np.array([BASE_TIMESTAMP_US, invalid], dtype=np.uint64))
            )


def test_signed_and_integer_valued_float_timestamps_remain_valid() -> None:
    signed_min = np.iinfo(np.int64).min
    assert data_loader._validated_timestamps_us(
        pd.Index(np.array([signed_min, BASE_TIMESTAMP_US], dtype=np.int64))
    ).tolist() == [signed_min, BASE_TIMESTAMP_US]
    assert data_loader._validated_timestamps_us(
        pd.Index([1_000_000.0, 2_000_000.0])
    ).tolist() == [1_000_000, 2_000_000]


def test_timestamp_delta_helper_still_rejects_unsorted_input() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        timestamps_us_to_datetime_and_deltas(
            pd.Index([BASE_TIMESTAMP_US, BASE_TIMESTAMP_US - 1])
        )


def test_file_calibration_wrapper_uses_raw_microsecond_clock(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_market_csv(
        tmp_path,
        [
            BASE_TIMESTAMP_US,
            BASE_TIMESTAMP_US + 3_000_000,
            BASE_TIMESTAMP_US + 2_000_000,
            BASE_TIMESTAMP_US + 4_000_000,
        ],
    )
    captured = {}

    def fake_calibrate(mid, dt, **kwargs):
        captured["mid"] = mid.copy()
        captured["dt"] = dt.copy()
        captured["bid"] = kwargs["best_bid"].copy()
        captured["ask"] = kwargs["best_ask"].copy()
        return SimpleNamespace(parameters=ASParameters(1.0, 2.0, 3.0))

    monkeypatch.setattr(calibration, "calibrate_from_arrays", fake_calibrate)

    with pytest.warns(RuntimeWarning):
        calibration.calibrate_as_parameters(str(path), plot=False, verbose=False)

    assert captured["dt"].tolist() == [0.0, 2.0, 1.0, 1.0]
    assert captured["mid"].tolist() == pytest.approx([100.0, 100.02, 100.01, 100.03])
    assert captured["bid"].tolist() == pytest.approx([99.99, 100.01, 100.0, 100.02])
    assert captured["ask"].tolist() == pytest.approx([100.01, 100.03, 100.02, 100.04])
