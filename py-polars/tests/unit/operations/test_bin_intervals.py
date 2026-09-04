from __future__ import annotations

import datetime
from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_bin_intervals() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_intervals([-1, 1], labels=["a", "b", "c"])

    expected = pl.Series("a", ["a", "b", "b", "c", "c"], dtype=pl.Enum(["a", "b", "c"]))
    assert_series_equal(result, expected)


def test_bin_intervals_right_closed() -> None:
    s = pl.Series("a", [0, 1, 2, 3, 4])

    left = s.bin_intervals([1, 2, 3], labels=False)
    right = s.bin_intervals([1, 2, 3], labels=False, right_closed=True)

    # A value sitting exactly on a breakpoint belongs to the upper bin when
    # left-closed and to the lower bin when right-closed.
    assert left.to_list() == [0, 1, 2, 3, 3]
    assert right.to_list() == [0, 0, 1, 2, 3]


def test_bin_intervals_single_bin() -> None:
    s = pl.Series("a", [1, 2, 3])

    result = s.bin_intervals([], labels=["only"])

    assert result.to_list() == ["only"] * 3


def test_bin_intervals_include_intervals() -> None:
    s = pl.Series("a", [-2, 0, 2])

    result = s.bin_intervals([-1, 1], labels=False, include_intervals=True)

    expected = pl.Series(
        "a",
        [
            {"bin": 0, "left": None, "right": -1},
            {"bin": 1, "left": -1, "right": 1},
            {"bin": 2, "left": 1, "right": None},
        ],
        dtype=pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}),
    )
    assert_series_equal(result, expected)


def test_bin_intervals_null_breakpoint_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="cannot contain nulls"):
        lf.select(pl.col("a").bin_intervals([None], labels=False)).collect_schema()


@pytest.mark.parametrize(
    ("dtype", "breaks", "expected_bound", "expected_bins"),
    [
        # Both numeric, so the column and the breakpoints meet at their supertype rather
        # than the breakpoints being forced into the column's dtype.
        (pl.Int64, [1.5], pl.Float64, [0, 1, 1]),
        (pl.Int64, [2.0], pl.Float64, [0, 1, 1]),
        (pl.Float32, [0.1], pl.Float64, [1, 1, 1]),
        (pl.UInt8, [2], pl.Int64, [0, 1, 1]),
        (pl.Int8, [2], pl.Int64, [0, 1, 1]),
    ],
)
def test_bin_intervals_numeric_breakpoints_promote_to_supertype(
    dtype: pl.DataType,
    breaks: list[Any],
    expected_bound: pl.DataType,
    expected_bins: list[int],
) -> None:
    s = pl.Series("a", [1, 2, 3]).cast(dtype)

    result = s.bin_intervals(breaks, labels=False, include_intervals=True)

    assert result.struct["bin"].to_list() == expected_bins
    assert result.struct["left"].dtype == expected_bound


def test_bin_intervals_breakpoints_of_the_input_dtype_do_not_promote() -> None:
    s = pl.Series("a", [1, 2, 3], dtype=pl.Int8)

    # Passing a Series rather than a Python list keeps the narrow dtype on both sides.
    result = s.bin_intervals(
        pl.Series([2], dtype=pl.Int8), labels=False, include_intervals=True
    )

    assert result.struct["left"].dtype == pl.Int8


def test_bin_intervals_series_breakpoints() -> None:
    s = pl.Series("a", [1, 2, 3])

    from_series = s.bin_intervals(pl.Series([2]), labels=False)
    from_list = s.bin_intervals([2], labels=False)

    assert_series_equal(from_series, from_list)


def test_bin_intervals_enum_uses_declaration_order() -> None:
    # Declaration order deliberately disagrees with lexical order.
    dtype = pl.Enum(["zebra", "apple", "mango"])
    s = pl.Series("a", ["mango", "zebra", "apple"], dtype=dtype)

    result = s.bin_intervals(pl.Series(["apple"], dtype=dtype), labels=False)

    # zebra < apple < mango, so only zebra falls below the breakpoint.
    assert result.to_list() == [1, 0, 1]


def test_bin_intervals_enum_boundaries_keep_dtype() -> None:
    dtype = pl.Enum(["zebra", "apple", "mango"])
    s = pl.Series("a", ["mango", "zebra"], dtype=dtype)

    result = s.bin_intervals(
        pl.Series(["apple"], dtype=dtype), labels=False, include_intervals=True
    )

    assert result.struct["left"].dtype == dtype
    assert result.struct["right"].dtype == dtype


def test_bin_intervals_enum_string_breakpoint_is_cast() -> None:
    dtype = pl.Enum(["zebra", "apple", "mango"])
    s = pl.Series("a", ["mango", "zebra"], dtype=dtype)

    assert s.bin_intervals(["apple"], labels=False).to_list() == [1, 0]


def test_bin_intervals_enum_unknown_breakpoint_raises() -> None:
    dtype = pl.Enum(["zebra", "apple", "mango"])
    s = pl.Series("a", ["mango", "zebra"], dtype=dtype)

    # Non-numeric breakpoints are cast down to the column, and "nope" is not a category.
    with pytest.raises(InvalidOperationError, match="conversion from `str` to `enum`"):
        s.bin_intervals(["nope"], labels=False)


def test_bin_intervals_categorical_uses_lexical_order() -> None:
    s = pl.Series("a", ["mango", "zebra", "apple"], dtype=pl.Categorical)

    result = s.bin_intervals(pl.Series(["mango"], dtype=pl.Categorical), labels=False)

    # Categorical sorts lexically: apple < mango < zebra.
    assert result.to_list() == [1, 1, 0]


def test_bin_intervals_over_is_a_noop() -> None:
    df = pl.DataFrame({"a": [1, 5, 9, 2, 6], "g": ["x", "x", "x", "y", "y"]})
    expr = pl.col("a").bin_intervals([4, 7], labels=False)

    # Explicit breakpoints make this elementwise, so grouping cannot change the result.
    assert_series_equal(
        df.select(expr.over("g")).to_series(), df.select(expr).to_series()
    )


def test_bin_intervals_serde_temporal_breakpoints() -> None:
    lf = pl.LazyFrame({"d": [datetime.date(2020, 1, 1), datetime.date(2022, 1, 1)]})

    q = lf.select(pl.col("d").bin_intervals([datetime.date(2021, 1, 1)], labels=False))

    assert_frame_equal(pl.LazyFrame.deserialize(q.serialize()).collect(), q.collect())


def test_bin_intervals_uniform() -> None:
    s = pl.Series("a", [0, 10])

    # Breakpoints are `min + (i + 1)/n * (max - min)`, so a single break at 5.0.
    assert s.bin_intervals(2, labels=["low", "high"]).to_list() == ["low", "high"]


def test_bin_intervals_uniform_boundaries_keep_input_dtype() -> None:
    s = pl.Series("a", [0, 100], dtype=pl.Int64)

    result = s.bin_intervals(4, labels=False, include_intervals=True)

    # Equal-width breakpoints are computed in the input dtype, not via Float64.
    assert result.struct["left"].dtype == pl.Int64
    assert result.struct["right"].to_list() == [25, None]


def test_bin_intervals_uniform_is_exact_beyond_float64_precision() -> None:
    base = 2**62
    s = pl.Series("a", [base + i for i in range(4)])

    result = s.bin_intervals(4, labels=False, include_intervals=True)

    # Going through f64 would round all four values to the same double, collapsing
    # min and max and dumping the whole column into the last bin.
    assert result.struct["bin"].to_list() == [0, 1, 2, 3]
    assert result.struct["right"].to_list() == [
        base + 1,
        base + 2,
        base + 3,
        None,
    ]


def test_bin_intervals_uniform_repeats_breakpoints_in_a_narrow_range() -> None:
    s = pl.Series("a", [0, 1, 2, 3])

    result = s.bin_intervals(10, labels=False)

    # Ten equal-width Int64 breakpoints do not exist between 0 and 3, so they repeat and
    # the bins they delimit stay empty. The price of boundaries in the input dtype.
    assert result.to_list() == [0, 3, 6, 9]


@pytest.mark.parametrize(
    ("right_closed", "expected_right"),
    [
        (False, [1, 2, 3, None]),
        (True, [0, 1, 2, None]),
    ],
)
def test_bin_intervals_uniform_integer_thresholds_follow_closure(
    right_closed: bool, expected_right: list[int | None]
) -> None:
    s = pl.Series("a", [0, 1, 2, 3])

    result = s.bin_intervals(
        4,
        labels=False,
        include_intervals=True,
        right_closed=right_closed,
    )

    assert result.struct["bin"].to_list() == [0, 1, 2, 3]
    assert result.struct["right"].to_list() == expected_right


def test_bin_intervals_uniform_float_extremes() -> None:
    max_float = float.fromhex("0x1.fffffffffffffp+1023")
    s = pl.Series("a", [-max_float, max_float])

    result = s.bin_intervals(2, labels=False, include_intervals=True)

    assert result.struct["bin"].to_list() == [0, 1]
    assert result.struct["right"].to_list() == [0.0, None]


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int8, [-128, 0, 127]),
        (pl.Int64, [-(2**63), 0, 2**63 - 1]),
        (pl.Int128, [-(2**127), 0, 2**127 - 1]),
        (pl.UInt64, [0, 2**63, 2**64 - 1]),
        (pl.UInt128, [0, 2**127, 2**128 - 1]),
    ],
)
def test_bin_intervals_uniform_spans_the_full_width(
    dtype: pl.DataType, values: list[int]
) -> None:
    s = pl.Series("a", values, dtype=dtype)

    result = s.bin_intervals(3, labels=False)

    # `max - min` overflows the input's own signed width here, so the span has to be
    # measured in the unsigned domain. Getting that wrong collapses the whole column
    # into a single bin.
    assert result.to_list() == [0, 1, 2]


def test_bin_intervals_uniform_with_no_span() -> None:
    s = pl.Series("a", [7, 7, 7])

    result = s.bin_intervals(4, labels=False, include_intervals=True)

    # Every threshold lands on `min`, so the whole column falls into the last bin.
    assert result.struct["bin"].to_list() == [3, 3, 3]
    assert result.struct["left"].to_list() == [7, 7, 7]
    assert result.struct["right"].to_list() == [None, None, None]
