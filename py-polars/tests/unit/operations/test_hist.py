from __future__ import annotations

import numpy as np
import pytest

import polars as pl
from polars import StringCache
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

inf = float("inf")


@StringCache()
def test_hist_empty_data_no_inputs() -> None:
    with pl.StringCache():
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
def test_hist_empty_data_invalid_edges() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    with pytest.raises(ComputeError, match="bins must increase monotonically"):
        s.hist(bins=[1, 0])  # invalid order


@StringCache()
def test_hist_empty_data_bad_bin_count() -> None:
    s = pl.Series([], dtype=pl.UInt8)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        s.hist(bin_count=-1)  # invalid order


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
def test_hist_invalid_bin_count() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        s.hist(bin_count=-1)  # invalid order


@StringCache()
def test_hist_invalid_bins() -> None:
    s = pl.Series([-5, 2, 0, 1, 99], dtype=pl.Int32)
    with pytest.raises(ComputeError, match="bins must increase monotonically"):
        s.hist(bins=[1, 0])  # invalid order


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@StringCache()
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


@pytest.mark.may_fail_auto_streaming
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
@pytest.mark.may_fail_auto_streaming
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


@pytest.mark.may_fail_auto_streaming
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


@pytest.mark.may_fail_auto_streaming
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
    with pl.StringCache():
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
    with pl.StringCache():
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
    with pl.StringCache():
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
