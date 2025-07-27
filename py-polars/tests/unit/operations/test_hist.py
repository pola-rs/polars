from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit

inf = float("inf")


def test_hist_empty_data_no_inputs() -> None:
    s = pl.Series([], dtype=pl.UInt8)

    # No bins or edges specified: 10 bins around unit interval
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=pl.Float64
            ),
            "category": pl.Series(
                [
                    "[0.0, 0.1]",
                    "(0.1, 0.2]",
                    "(0.2, 0.3]",
                    "(0.3, 0.4]",
                    "(0.4, 0.5]",
                    "(0.5, 0.6]",
                    "(0.6, 0.7]",
                    "(0.7, 0.8]",
                    "(0.8, 0.9]",
                    "(0.9, 1.0]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=pl.UInt32),
        }
    )
    result = s.hist()
    assert_frame_equal(result, expected)


def test_hist_empty_data_empty_bins() -> None:
    s = pl.Series([], dtype=pl.UInt8)

    # No bins or edges specified: 10 bins around unit interval
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([], dtype=pl.Float64),
            "category": pl.Series([], dtype=pl.Categorical),
            "count": pl.Series([], dtype=pl.UInt32),
        }
    )
    result = s.hist(bins=[])
    assert_frame_equal(result, expected)


def test_hist_empty_data_single_bin_edge() -> None:
    s = pl.Series([], dtype=pl.UInt8)

    # No bins or edges specified: 10 bins around unit interval
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([], dtype=pl.Float64),
            "category": pl.Series([], dtype=pl.Categorical),
            "count": pl.Series([], dtype=pl.UInt32),
        }
    )
    result = s.hist(bins=[2])
    assert_frame_equal(result, expected)


def test_hist_empty_data_valid_edges() -> None:
    s = pl.Series([], dtype=pl.UInt8)

    # No bins or edges specified: 10 bins around unit interval
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([2.0, 3.0], dtype=pl.Float64),
            "category": pl.Series(["[1.0, 2.0]", "(2.0, 3.0]"], dtype=pl.Categorical),
            "count": pl.Series([0, 0], dtype=pl.UInt32),
        }
    )
    result = s.hist(bins=[1, 2, 3])
    assert_frame_equal(result, expected)


def test_hist_empty_data_invalid_edges() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    with pytest.raises(ComputeError, match="bins must increase monotonically"):
        s.hist(bins=[1, 0])  # invalid order


def test_hist_empty_data_bad_bin_count() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        s.hist(bin_count=-1)  # invalid order


def test_hist_empty_data_zero_bin_count() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([], dtype=pl.Float64),
            "category": pl.Series([], dtype=pl.Categorical),
            "count": pl.Series([], dtype=pl.UInt32),
        }
    )
    result = s.hist(bin_count=0)
    assert_frame_equal(result, expected)


def test_hist_empty_data_single_bin_count() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([1.0], dtype=pl.Float64),
            "category": pl.Series(["[0.0, 1.0]"], dtype=pl.Categorical),
            "count": pl.Series([0], dtype=pl.UInt32),
        }
    )
    result = s.hist(bin_count=1)
    assert_frame_equal(result, expected)


def test_hist_empty_data_valid_bin_count() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([0.2, 0.4, 0.6, 0.8, 1.0], dtype=pl.Float64),
            "category": pl.Series(
                [
                    "[0.0, 0.2]",
                    "(0.2, 0.4]",
                    "(0.4, 0.6]",
                    "(0.6, 0.8]",
                    "(0.8, 1.0]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([0, 0, 0, 0, 0], dtype=pl.UInt32),
        }
    )
    result = s.hist(bin_count=5)
    assert_frame_equal(result, expected)


def test_hist_invalid_bin_count() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        s.hist(bin_count=-1)  # invalid order


def test_hist_invalid_bins() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    with pytest.raises(ComputeError, match="bins must increase monotonically"):
        s.hist(bins=[1, 0])  # invalid order


def test_hist_bin_outside_data() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bins=[-10, -9])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([-9.0], dtype=pl.Float64),
            "category": pl.Series(["[-10.0, -9.0]"], dtype=pl.Categorical),
            "count": pl.Series([0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_bins_between_data() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bins=[4.5, 10.5])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([10.5], dtype=pl.Float64),
            "category": pl.Series(["[4.5, 10.5]"], dtype=pl.Categorical),
            "count": pl.Series([0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_bins_first_edge() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bins=[2, 3, 4])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([3.0, 4.0], dtype=pl.Float64),
            "category": pl.Series(["[2.0, 3.0]", "(3.0, 4.0]"], dtype=pl.Categorical),
            "count": pl.Series([1, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_bins_last_edge() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bins=[-4, 0, 99, 100])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([0.0, 99.0, 100.0], dtype=pl.Float64),
            "category": pl.Series(
                [
                    "[-4.0, 0.0]",
                    "(0.0, 99.0]",
                    "(99.0, 100.0]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_single_value_single_bin_count() -> None:
    s = pl.Series([1], dtype=pl.Int32)
    result = s.hist(bin_count=1)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([1.5], dtype=pl.Float64),
            "category": pl.Series(["[0.5, 1.5]"], dtype=pl.Categorical),
            "count": pl.Series([1], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_single_bin_count() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bin_count=1)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([99.0], dtype=pl.Float64),
            "category": pl.Series(["[-5.0, 99.0]"], dtype=pl.Categorical),
            "count": pl.Series([5], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_partial_covering() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bins=[-1.5, 2.5, 50, 105])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([2.5, 50.0, 105.0], dtype=pl.Float64),
            "category": pl.Series(
                ["[-1.5, 2.5]", "(2.5, 50.0]", "(50.0, 105.0]"], dtype=pl.Categorical
            ),
            "count": pl.Series([3, 0, 1], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_full_covering() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bins=[-5.5, 2.5, 50, 105])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([2.5, 50.0, 105.0], dtype=pl.Float64),
            "category": pl.Series(
                ["[-5.5, 2.5]", "(2.5, 50.0]", "(50.0, 105.0]"], dtype=pl.Categorical
            ),
            "count": pl.Series([4, 0, 1], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_more_bins_than_data() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    result = s.hist(bin_count=8)

    # manually compute breaks
    span = 99 - (-5)
    width = span / 8
    breaks = [-5 + width * i for i in range(8 + 1)]
    categories = [f"({breaks[i]}, {breaks[i + 1]}]" for i in range(8)]
    categories[0] = f"[{categories[0][1:]}"

    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(breaks[1:], dtype=pl.Float64),
            "category": pl.Series(categories, dtype=pl.Categorical),
            "count": pl.Series([4, 0, 0, 0, 0, 0, 0, 1], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist() -> None:
    s = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
    out = s.hist(bin_count=4)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([2.75, 4.5, 6.25, 8.0], dtype=pl.Float64),
            "category": pl.Series(
                ["[1.0, 2.75]", "(2.75, 4.5]", "(4.5, 6.25]", "(6.25, 8.0]"],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([3, 2, 0, 2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(out, expected)


def test_hist_all_null() -> None:
    s = pl.Series([None], dtype=pl.Float64)
    out = s.hist()
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=pl.Float64
            ),
            "category": pl.Series(
                [
                    "[0.0, 0.1]",
                    "(0.1, 0.2]",
                    "(0.2, 0.3]",
                    "(0.3, 0.4]",
                    "(0.4, 0.5]",
                    "(0.5, 0.6]",
                    "(0.6, 0.7]",
                    "(0.7, 0.8]",
                    "(0.8, 0.9]",
                    "(0.9, 1.0]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([0] * 10, dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("n_null", [0, 5])
@pytest.mark.parametrize("n_values", [3, 10, 250])
def test_hist_rand(n_values: int, n_null: int) -> None:
    s_rand = pl.Series([None] * n_null, dtype=pl.Int64)
    s_values = pl.Series(np.random.randint(0, 100, n_values), dtype=pl.Int64)
    if s_values.n_unique() == 1:
        pytest.skip("Identical values not tested.")
    s = pl.concat((s_rand, s_values))
    out = s.hist(bin_count=10)

    bp = out["breakpoint"]
    count = out["count"]
    min_edge = s.min() - (s.max() - s.min()) * 0.001  # type: ignore[operator]
    for i in range(out.height):
        if i == 0:
            lower = min_edge
        else:
            lower = bp[i - 1]
        upper = bp[i]

        assert ((s <= upper) & (s > lower)).sum() == count[i]


def test_hist_floating_point() -> None:
    # This test hits the specific floating point case where the bin width should be
    # 5.7, but it is represented by 5.6999999. The result is that an item equal to the
    # upper bound (72) exceeds the maximum bins. This tests the code path that catches
    # this case.
    n_values = 3
    n_null = 50

    np.random.seed(2)
    s_rand = pl.Series([None] * n_null, dtype=pl.Int64)
    s_values = pl.Series(np.random.randint(0, 100, n_values), dtype=pl.Int64)
    s = pl.concat((s_rand, s_values))
    out = s.hist(bin_count=10)
    min_edge = s.min() - (s.max() - s.min()) * 0.001  # type: ignore[operator]

    bp = out["breakpoint"]
    count = out["count"]
    for i in range(out.height):
        if i == 0:
            lower = min_edge
        else:
            lower = bp[i - 1]
        upper = bp[i]

        assert ((s <= upper) & (s > lower)).sum() == count[i]


def test_hist_max_boundary_19998() -> None:
    s = pl.Series(
        [
            9514.988509739183,
            30738.098872148617,
            41400.15705103004,
            49093.06982022727,
        ]
    )
    result = s.hist(bin_count=50)
    assert result["count"].sum() == 4


def test_hist_max_boundary_20133() -> None:
    # Given a set of values that result in bin index to be a floating point number that
    # is represented as 5.000000000000001
    s = pl.Series(
        [
            6.197601318359375,
            83.5203145345052,
        ]
    )

    # When histogram is calculated
    result = s.hist(bin_count=5)

    # Then there is no exception (previously was possible to get index out of bounds
    # here) and all the numbers fit into at least one of the bins
    assert result["count"].sum() == 2


def test_hist_same_values_20030() -> None:
    out = pl.Series([1, 1]).hist(bin_count=2)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([1.0, 1.5], dtype=pl.Float64),
            "category": pl.Series(["[0.5, 1.0]", "(1.0, 1.5]"], dtype=pl.Categorical),
            "count": pl.Series([2, 0], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(out, expected)


def test_hist_breakpoint_accuracy() -> None:
    s = pl.Series([1, 2, 3, 4])
    out = s.hist(bin_count=3)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([2.0, 3.0, 4.0], dtype=pl.Float64),
            "category": pl.Series(
                ["[1.0, 2.0]", "(2.0, 3.0]", "(3.0, 4.0]"], dtype=pl.Categorical
            ),
            "count": pl.Series([2, 1, 1], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(out, expected)


def test_hist_ensure_max_value_20879() -> None:
    s = pl.Series([-1 / 3, 0, 1, 2, 3, 7])
    result = s.hist(bin_count=3)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [
                    2.0 + 1 / 9,
                    4.0 + 5 / 9,
                    7.0,
                ],
                dtype=pl.Float64,
            ),
            "category": pl.Series(
                [
                    "[-0.333333, 2.111111]",
                    "(2.111111, 4.555556]",
                    "(4.555556, 7.0]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([4, 1, 1], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_ignore_nans_21082() -> None:
    s = pl.Series([0.0, float("nan"), 0.5, float("nan"), 1.0])
    result = s.hist(bins=[-0.001, 0.25, 0.5, 0.75, 1.0])
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([0.25, 0.5, 0.75, 1.0], dtype=pl.Float64),
            "category": pl.Series(
                [
                    "[-0.001, 0.25]",
                    "(0.25, 0.5]",
                    "(0.5, 0.75]",
                    "(0.75, 1.0]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 1, 0, 1], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_include_lower_22056() -> None:
    s = pl.Series("a", [1, 5])
    result = s.hist(bins=[1, 5], include_category=True)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([5.0], dtype=pl.Float64),
            "category": pl.Series(["[1.0, 5.0]"], dtype=pl.Categorical),
            "count": pl.Series([2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_ulp_edge_22234() -> None:
    # Uniform path
    s = pl.Series([1.0, 1e-16, 1.3e-16, -1.0])
    result = s.hist(bin_count=2)
    assert result["count"].to_list() == [1, 3]

    # Manual path
    result = s.hist(bins=[-1, 0, 1])
    assert result["count"].to_list() == [1, 3]


def test_hist_temporal_date_bins() -> None:
    s = pl.Series(
        [
            date(2024, 12, 26),  # -5
            date(2025, 1, 2),  # 2
            date(2024, 12, 31),  # 0
            date(2025, 1, 1),  # 1
            date(2025, 4, 10),  # 99
        ],
        dtype=pl.Date,
    )
    result = s.hist(
        bins=[
            date(2024, 12, 27),  # -4
            date(2024, 12, 31),  # 0
            date(2025, 4, 10),  # 99
            date(2025, 4, 11),  # 100
        ]
    )
    expected = pl.DataFrame(
        {
            "breakpoint": [date(2024, 12, 31), date(2025, 4, 10), date(2025, 4, 11)],
            "category": pl.Series(
                [
                    "[2024-12-27, 2024-12-31]",
                    "(2024-12-31, 2025-04-10]",
                    "(2025-04-10, 2025-04-11]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_temporal_date_bin_count() -> None:
    s = pl.Series(
        [
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 4),
            date(2025, 1, 8),
            date(2025, 1, 9),
        ]
    )
    result = s.hist(bin_count=4)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [date(2025, 1, 3), date(2025, 1, 5), date(2025, 1, 7), date(2025, 1, 9)]
            ),
            "category": pl.Series(
                [
                    "[2025-01-01, 2025-01-03]",
                    "(2025-01-03, 2025-01-05]",
                    "(2025-01-05, 2025-01-07]",
                    "(2025-01-07, 2025-01-09]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([2, 1, 0, 2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_temporal_datetime_bins() -> None:
    s = pl.Series(
        [
            datetime(2024, 12, 26),  # -5
            datetime(2025, 1, 2),  # 2
            datetime(2024, 12, 31),  # 0
            datetime(2025, 1, 1),  # 1
            datetime(2025, 4, 10),  # 99
        ],
        dtype=pl.Datetime(time_unit="ms", time_zone="EST"),
    )
    result = s.hist(
        bins=pl.Series(
            [
                datetime(2024, 12, 27),  # -4
                datetime(2024, 12, 31),  # 0
                datetime(2025, 4, 10),  # 99
                datetime(2025, 4, 11),  # 100
            ],
            dtype=pl.Datetime(time_unit="ns", time_zone="EST"),
        )
    )
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [datetime(2024, 12, 31), datetime(2025, 4, 10), datetime(2025, 4, 11)],
                dtype=pl.Datetime(time_unit="ms", time_zone="EST"),
            ),
            "category": pl.Series(
                [
                    "[2024-12-26 19:00:00 EST, 2024-12-30 19:00:00 EST]",
                    "(2024-12-30 19:00:00 EST, 2025-04-09 19:00:00 EST]",
                    "(2025-04-09 19:00:00 EST, 2025-04-10 19:00:00 EST]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
def test_hist_temporal_datetime_bin_count(tu: TimeUnit) -> None:
    s = pl.Series(
        [
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            datetime(2025, 1, 4),
            datetime(2025, 1, 8),
            datetime(2025, 1, 9),
        ],
        dtype=pl.Datetime(time_unit=tu),
    )
    result = s.hist(bin_count=4)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [
                    datetime(2025, 1, 3),
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 9),
                ],
                dtype=pl.Datetime(time_unit=tu),
            ),
            "category": pl.Series(
                [
                    "[2025-01-01 00:00:00, 2025-01-03 00:00:00]",
                    "(2025-01-03 00:00:00, 2025-01-05 00:00:00]",
                    "(2025-01-05 00:00:00, 2025-01-07 00:00:00]",
                    "(2025-01-07 00:00:00, 2025-01-09 00:00:00]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([2, 1, 0, 2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
def test_hist_temporal_duration_bins(tu: TimeUnit) -> None:
    s = pl.Series(
        [
            timedelta(days=-5),
            timedelta(days=2),
            timedelta(days=0),
            timedelta(days=1),
            timedelta(days=99),
        ],
        dtype=pl.Duration(time_unit=tu),
    )
    result = s.hist(
        bins=pl.Series(
            [
                timedelta(days=-4),
                timedelta(days=0),
                timedelta(days=99),
                timedelta(days=100),
            ],
            dtype=pl.Duration(time_unit=tu),
        )
    )
    # The rust side always outputs the 'µ' character.
    units = "µs" if tu == "us" else tu
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [timedelta(days=0), timedelta(days=99), timedelta(days=100)],
                dtype=pl.Duration(time_unit=tu),
            ),
            "category": pl.Series(
                [
                    f"[-4d, 0{units}]",
                    f"(0{units}, 99d]",
                    "(99d, 100d]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
def test_hist_temporal_duration_bin_count(tu: TimeUnit) -> None:
    s = pl.Series(
        [
            timedelta(days=1),
            timedelta(days=2),
            timedelta(days=4),
            timedelta(days=8),
            timedelta(days=9),
        ],
        dtype=pl.Duration(time_unit=tu),
    )
    result = s.hist(bin_count=4)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [
                    timedelta(days=3),
                    timedelta(days=5),
                    timedelta(days=7),
                    timedelta(days=9),
                ],
                dtype=pl.Duration(time_unit=tu),
            ),
            "category": pl.Series(
                [
                    "[1d, 3d]",
                    "(3d, 5d]",
                    "(5d, 7d]",
                    "(7d, 9d]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([2, 1, 0, 2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_temporal_time_bins() -> None:
    s = pl.Series(
        [
            time(minute=0),
            time(minute=2),
            time(minute=0),
            time(minute=1),
            time(minute=58),
        ]
    )
    result = s.hist(bins=[time(minute=0), time(minute=10), time(minute=15)])
    expected = pl.DataFrame(
        {
            "breakpoint": [time(minute=10), time(minute=15)],
            "category": pl.Series(
                [
                    "[00:00:00, 00:10:00]",
                    "(00:10:00, 00:15:00]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([4, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_temporal_time_bin_count() -> None:
    s = pl.Series(
        [
            time(minute=1),
            time(minute=2),
            time(minute=4),
            time(minute=8),
            time(minute=9),
        ],
    )
    result = s.hist(bin_count=4)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [time(minute=3), time(minute=5), time(minute=7), time(minute=9)]
            ),
            "category": pl.Series(
                [
                    "[00:01:00, 00:03:00]",
                    "(00:03:00, 00:05:00]",
                    "(00:05:00, 00:07:00]",
                    "(00:07:00, 00:09:00]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([2, 1, 0, 2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_temporal_values_upcast() -> None:
    s = pl.Series(
        [
            datetime(2024, 12, 26),  # -5
            datetime(2025, 1, 2),  # 2
            datetime(2024, 12, 31),  # 0
            datetime(2025, 1, 1),  # 1
            datetime(2025, 4, 10),  # 99
        ],
        dtype=pl.Datetime(time_unit="ns", time_zone="EST"),
    )
    result = s.hist(
        bins=pl.Series(
            [
                datetime(2024, 12, 27),  # -4
                datetime(2024, 12, 31),  # 0
                datetime(2025, 4, 10),  # 99
                datetime(2025, 4, 11),  # 100
            ],
            dtype=pl.Datetime(time_unit="ms", time_zone="EST"),
        )
    )
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series(
                [datetime(2024, 12, 31), datetime(2025, 4, 10), datetime(2025, 4, 11)],
                dtype=pl.Datetime(time_unit="ms", time_zone="EST"),
            ),
            "category": pl.Series(
                [
                    "[2024-12-26 19:00:00 EST, 2024-12-30 19:00:00 EST]",
                    "(2024-12-30 19:00:00 EST, 2025-04-09 19:00:00 EST]",
                    "(2025-04-09 19:00:00 EST, 2025-04-10 19:00:00 EST]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


def test_hist_temporal_too_many_bins() -> None:
    s = pl.Series([date(2025, 1, 1), date(2025, 1, 8)])
    with pytest.raises(
        ComputeError,
        match="data type is too coarse to create 8 bins in specified data range",
    ):
        s.hist(bin_count=8)

    s = pl.Series(
        [datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 0, 1)], dtype=pl.Datetime("ms")
    )
    with pytest.raises(
        ComputeError,
        match="data type is too coarse to create 1001 bins in specified data range",
    ):
        s.hist(bin_count=1_001)

    s = pl.Series(
        [datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 0, 0, 100)],
        dtype=pl.Datetime("us"),
    )
    with pytest.raises(
        ComputeError,
        match="data type is too coarse to create 101 bins in specified data range",
    ):
        s.hist(bin_count=101)

    s = pl.Series([timedelta(0), timedelta(seconds=1)], dtype=pl.Duration("ms"))
    with pytest.raises(
        ComputeError,
        match="data type is too coarse to create 1001 bins in specified data range",
    ):
        s.hist(bin_count=1001)

    s = pl.Series([time(0), time(microsecond=1)], dtype=pl.Time)
    with pytest.raises(
        ComputeError,
        match="data type is too coarse to create 1001 bins in specified data range",
    ):
        s.hist(bin_count=1001)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "Asia/Kathmandu"])
def test_hist_temporal_date_upcast_values(tu: TimeUnit, tz: str | None) -> None:
    s = pl.Series(
        [
            date(2024, 12, 26),  # -5
            date(2025, 1, 2),  # 2
            date(2024, 12, 31),  # 0
            date(2025, 1, 1),  # 1
            date(2025, 4, 10),  # 99
        ],
        dtype=pl.Date,
    )
    bins = pl.Series(
        [
            date(2024, 12, 27),  # -4
            date(2024, 12, 31),  # 0
            date(2025, 4, 10),  # 99
            date(2025, 4, 11),  # 100
        ],
        dtype=pl.Datetime(tu),
    )
    expected_breakpoint = pl.Series(
        [date(2024, 12, 31), date(2025, 4, 10), date(2025, 4, 11)],
        dtype=pl.Datetime(tu),
    )
    if tz is not None:
        bins = bins.dt.replace_time_zone(tz)
        expected_breakpoint = expected_breakpoint.dt.replace_time_zone(tz)
        tz_str = " +0545"
    else:
        tz_str = ""
    result = s.hist(bins=bins)
    expected = pl.DataFrame(
        {
            "breakpoint": expected_breakpoint,
            "category": pl.Series(
                [
                    f"[2024-12-27 00:00:00{tz_str}, 2024-12-31 00:00:00{tz_str}]",
                    f"(2024-12-31 00:00:00{tz_str}, 2025-04-10 00:00:00{tz_str}]",
                    f"(2025-04-10 00:00:00{tz_str}, 2025-04-11 00:00:00{tz_str}]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "Asia/Kathmandu"])
def test_hist_temporal_date_upcast_bins(tu: TimeUnit, tz: str | None) -> None:
    s = pl.Series(
        [
            date(2024, 12, 26),  # -5
            date(2025, 1, 2),  # 2
            date(2024, 12, 31),  # 0
            date(2025, 1, 1),  # 1
            date(2025, 4, 10),  # 99
        ],
        dtype=pl.Datetime(tu),
    )
    bins = pl.Series(
        [
            date(2024, 12, 27),  # -4
            date(2024, 12, 31),  # 0
            date(2025, 4, 10),  # 99
            date(2025, 4, 11),  # 100
        ],
        dtype=pl.Date,
    )
    expected_breakpoint = pl.Series(
        [date(2024, 12, 31), date(2025, 4, 10), date(2025, 4, 11)],
        dtype=pl.Datetime(tu),
    )
    if tz is not None:
        s = s.dt.replace_time_zone(tz)
        expected_breakpoint = expected_breakpoint.dt.replace_time_zone(tz)
        tz_str = " +0545"
    else:
        tz_str = ""
    result = s.hist(bins=bins)
    expected = pl.DataFrame(
        {
            "breakpoint": expected_breakpoint,
            "category": pl.Series(
                [
                    f"[2024-12-27 00:00:00{tz_str}, 2024-12-31 00:00:00{tz_str}]",
                    f"(2024-12-31 00:00:00{tz_str}, 2025-04-10 00:00:00{tz_str}]",
                    f"(2025-04-10 00:00:00{tz_str}, 2025-04-11 00:00:00{tz_str}]",
                ],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([1, 3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(result, expected)
