from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal
from polars.testing.parametric import series

if TYPE_CHECKING:
    from polars._typing import IntoExpr, PolarsDataType
    from polars.datatypes import IntegerType


def isnan(value: object) -> bool:
    if isinstance(value, int):
        return False
    if not isinstance(value, (np.number, float)):
        return False
    return np.isnan(value)  # type: ignore[no-any-return]


def assert_index_of(
    series: pl.Series,
    value: IntoExpr,
    convert_to_literal: bool = False,
) -> None:
    """``Series.index_of()`` returns the index, or ``None`` if it can't be found."""
    if isnan(value):
        expected_index = None
        for i, o in enumerate(series.to_list()):
            if o is not None and np.isnan(o):
                expected_index = i
                break
    else:
        try:
            expected_index = series.to_list().index(value)
        except ValueError:
            expected_index = None
    if expected_index == -1:
        expected_index = None

    if convert_to_literal:
        value = pl.lit(value, dtype=series.dtype)

    # Eager API:
    assert series.index_of(value) == expected_index
    # Lazy API:
    assert pl.LazyFrame({"series": series}).select(
        pl.col("series").index_of(value)
    ).collect().get_column("series").to_list() == [expected_index]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_float(dtype: pl.DataType) -> None:
    values = [1.5, np.nan, np.inf, 3.0, None, -np.inf, 0.0, -0.0, -np.nan]
    if dtype == pl.Float32:
        # Can't pass Python literals to index_of() for Float32
        values = [(None if v is None else np.float32(v)) for v in values]  # type: ignore[misc]

    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat([pl.Series([1, 7], dtype=dtype), series], rechunk=False)

    extra_values = [
        np.int8(3),
        np.float32(1.5),
        np.float32(2**10),
    ]
    if dtype == pl.Float64:
        extra_values.extend([np.int32(2**10), np.float64(2**10), np.float64(1.5)])
    for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
        for value in values:
            assert_index_of(s, value, convert_to_literal=True)
            assert_index_of(s, value, convert_to_literal=False)
        for value in extra_values:  # type: ignore[assignment]
            assert_index_of(s, value)

    # -np.nan should match np.nan:
    assert series.index_of(-np.float32("nan")) == 1  # type: ignore[arg-type]
    # -0.0 should match 0.0:
    assert series.index_of(-np.float32(0.0)) == 6  # type: ignore[arg-type]


def test_null() -> None:
    series = pl.Series([None, None], dtype=pl.Null)
    assert_index_of(series, None)


def test_empty() -> None:
    series = pl.Series([], dtype=pl.Null)
    assert_index_of(series, None)
    series = pl.Series([], dtype=pl.Int64)
    assert_index_of(series, None)
    assert_index_of(series, 12)
    assert_index_of(series.sort(descending=True), 12)
    assert_index_of(series.sort(descending=False), 12)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.UInt128,
    ],
)
def test_integer(dtype: IntegerType) -> None:
    print(dtype)
    dtype_min = dtype.min()
    dtype_max = pl.Int128.max() if dtype == pl.UInt128 else dtype.max()

    values = [
        51,
        3,
        None,
        4,
        pl.select(dtype_max).item(),
        pl.select(dtype_min).item(),
    ]
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat(
        [pl.Series([100, 7], dtype=dtype), series], rechunk=False
    )

    extra_values = [pl.select(v).item() for v in [dtype_max - 1, dtype_min + 1]]
    for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
        value: IntoExpr
        for value in values:
            assert_index_of(s, value, convert_to_literal=True)
            assert_index_of(s, value, convert_to_literal=False)
        for value in extra_values:
            assert_index_of(s, value, convert_to_literal=True)
            assert_index_of(s, value, convert_to_literal=False)

        # Can't cast floats:
        for f in [np.float32(3.1), np.float64(3.1), 50.9]:
            with pytest.raises(InvalidOperationError, match=r"cannot cast.*"):
                s.index_of(f)  # type: ignore[arg-type]


def test_integer_upcast() -> None:
    series = pl.Series([0, 123, 456, 789], dtype=pl.Int64)
    for should_work in [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16, pl.Int32, pl.UInt32]:
        assert series.index_of(pl.lit(123, dtype=should_work)) == 1


def test_groupby() -> None:
    df = pl.DataFrame(
        {"label": ["a", "b", "a", "b", "a", "b"], "value": [10, 3, 20, 2, 40, 20]}
    )
    expected = pl.DataFrame(
        {"label": ["a", "b"], "value": [1, 2]},
        schema={"label": pl.String, "value": pl.get_index_type()},
    )
    assert_frame_equal(
        df.group_by("label", maintain_order=True).agg(pl.col("value").index_of(20)),
        expected,
    )
    assert_frame_equal(
        df.lazy()
        .group_by("label", maintain_order=True)
        .agg(pl.col("value").index_of(20))
        .collect(),
        expected,
    )


LISTS_STRATEGY = st.lists(
    st.one_of(st.none(), st.integers(min_value=10, max_value=50)), max_size=10
)


@given(
    list1=LISTS_STRATEGY,
    list2=LISTS_STRATEGY,
    list3=LISTS_STRATEGY,
)
# The examples are cases where this test previously caught bugs:
@example([], [], [None])
@pytest.mark.slow
def test_randomized(
    list1: list[int | None], list2: list[int | None], list3: list[int | None]
) -> None:
    series = pl.concat(
        [pl.Series(values, dtype=pl.Int8) for values in [list1, list2, list3]],
        rechunk=False,
    )
    sorted_series = series.sort(descending=False)
    sorted_series2 = series.sort(descending=True)

    # Values are between 10 and 50, plus add None and max/min range values:
    for i in set(range(10, 51)) | {-128, 127, None}:
        assert_index_of(series, i)
        assert_index_of(sorted_series, i)
        assert_index_of(sorted_series2, i)


ENUM = pl.Enum(["a", "b", "c"])


@pytest.mark.parametrize(
    ("series", "extra_values", "sortable"),
    [
        (pl.Series(["abc", None, "bb"]), ["", "🚲"], True),
        (pl.Series([True, None, False, True, False]), [], True),
        (
            pl.Series([datetime(1997, 12, 31), datetime(1996, 1, 1)]),
            [datetime(2023, 12, 12, 16, 12, 39)],
            True,
        ),
        (
            pl.Series([date(1997, 12, 31), None, date(1996, 1, 1)]),
            [date(2023, 12, 12)],
            True,
        ),
        (
            pl.Series([time(16, 12, 31), None, time(11, 10, 53)]),
            [time(11, 12, 16)],
            True,
        ),
        (
            pl.Series(
                [timedelta(hours=12), None, timedelta(minutes=3)],
            ),
            [timedelta(minutes=17)],
            True,
        ),
        (pl.Series([[1, 2], None, [4, 5], [6], [None, 3, 5]]), [[5, 7], []], True),
        (
            pl.Series([[[1, 2]], None, [[4, 5]], [[6]], [[None, 3, 5]], [None]]),
            [[[5, 7]], []],
            True,
        ),
        (
            pl.Series([[1, 2], None, [4, 5], [None, 3]], dtype=pl.Array(pl.Int64(), 2)),
            [[5, 7], [None, None]],
            True,
        ),
        (
            pl.Series(
                [[[1, 2]], [None], [[4, 5]], None, [[None, 3]]],
                dtype=pl.Array(pl.Array(pl.Int64(), 2), 1),
            ),
            [[[5, 7]], [[None, None]]],
            True,
        ),
        (
            pl.Series(
                [{"a": 1, "b": 2}, None, {"a": 3, "b": 4}, {"a": None, "b": 2}],
                dtype=pl.Struct({"a": pl.Int64(), "b": pl.Int64()}),
            ),
            [{"a": 7, "b": None}, {"a": 6, "b": 4}],
            False,
        ),
        (pl.Series([b"abc", None, b"xxx"]), [b"\x0025"], True),
        (
            pl.Series(
                [Decimal(12), None, Decimal(3), Decimal(-12), Decimal(1) / Decimal(10)],
                dtype=pl.Decimal(20, 4),
            ),
            [Decimal(4), Decimal(-2), Decimal(1) / Decimal(4), Decimal(1) / Decimal(8)],
            True,
        ),
    ],
)
def test_other_types(
    series: pl.Series, extra_values: list[Any], sortable: bool
) -> None:
    expected_values = series.to_list()
    series_variants = [series, series.drop_nulls()]
    if sortable:
        series_variants.extend(
            [
                series.sort(descending=False),
                series.sort(descending=True),
            ]
        )
    for s in series_variants:
        for value in expected_values:
            assert_index_of(s, value, convert_to_literal=True)
            assert_index_of(s, value, convert_to_literal=False)
        # Extra values may not be expressible as literal of correct dtype, so
        # don't try:
        for value in extra_values:
            assert_index_of(s, value)


# Before the output type would be list[idx-type] when no item was found
def test_non_found_correct_type() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [0, 1], pl.Int32),
            pl.Series("b", [1, 2], pl.Int32),
        ]
    )

    assert_frame_equal(
        df.group_by("a", maintain_order=True).agg(pl.col.b.index_of(1)),
        pl.DataFrame({"a": [0, 1], "b": [0, None]}),
        check_dtypes=False,
    )


def test_error_on_multiple_values() -> None:
    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="needle of `index_of` can only contain",
    ):
        pl.Series("a", [1, 2, 3]).index_of(pl.Series([2, 3]))


@pytest.mark.parametrize(
    "convert_to_literal",
    [
        True,
        False,
    ],
)
def test_enum(convert_to_literal: bool) -> None:
    series = pl.Series(["a", "c", None, "b"], dtype=pl.Enum(["c", "b", "a"]))
    expected_values = series.to_list()
    for s in [
        series,
        series.drop_nulls(),
        series.sort(descending=False),
        series.sort(descending=True),
    ]:
        for value in expected_values:
            assert_index_of(s, value, convert_to_literal=convert_to_literal)


@pytest.mark.parametrize(
    "convert_to_literal",
    [True, False],
)
def test_categorical(convert_to_literal: bool) -> None:
    series = pl.Series(["a", "c", None, "b"], dtype=pl.Categorical)
    expected_values = series.to_list()
    for s in [
        series,
        series.drop_nulls(),
        series.sort(descending=False),
        series.sort(descending=True),
    ]:
        for value in expected_values:
            assert_index_of(s, value, convert_to_literal=convert_to_literal)


@pytest.mark.parametrize("value", [0, 0.1])
def test_categorical_wrong_type_keys_dont_work(value: int | float) -> None:
    series = pl.Series(["a", "c", None, "b"], dtype=pl.Categorical)
    msg = "cannot cast.*losslessly.*"
    with pytest.raises(InvalidOperationError, match=msg):
        series.index_of(value)
    df = pl.DataFrame({"s": series})
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(pl.col("s").index_of(value))


@given(s=series(name="s", allow_chunks=True, max_size=10))
def test_index_of_null_parametric(s: pl.Series) -> None:
    idx_null = s.index_of(None)
    if s.len() == 0:
        assert idx_null is None
    elif s.null_count() == 0:
        assert idx_null is None
    elif s.null_count() == len(s):
        assert idx_null == 0


def test_out_of_range_integers() -> None:
    series = pl.Series([0, 100, None, 1, 2], dtype=pl.Int8)
    with pytest.raises(InvalidOperationError, match="cannot cast 128 losslessly to i8"):
        assert series.index_of(128)
    with pytest.raises(
        InvalidOperationError, match="cannot cast -200 losslessly to i8"
    ):
        assert series.index_of(-200)


def test_out_of_range_decimal() -> None:
    # Up to 34 digits of integers:
    series = pl.Series([1, None], dtype=pl.Decimal(36, 2))
    assert series.index_of(10**34 - 1) is None
    assert series.index_of(-(10**34 - 1)) is None
    out_of_range = 10**34
    with pytest.raises(
        InvalidOperationError, match=f"cannot cast {out_of_range} losslessly"
    ):
        assert series.index_of(out_of_range)
    with pytest.raises(
        InvalidOperationError, match=f"cannot cast {-out_of_range} losslessly"
    ):
        assert series.index_of(-out_of_range)


def test_out_of_range_float64() -> None:
    series = pl.Series([0, 255, None], dtype=pl.Float64)
    # Small numbers are fine:
    assert series.index_of(1_000_000) is None
    assert series.index_of(-1_000_000) is None
    with pytest.raises(
        InvalidOperationError, match=f"cannot cast {2**53} losslessly to f64"
    ):
        assert series.index_of(2**53)
    with pytest.raises(
        InvalidOperationError, match=f"cannot cast {-(2**53)} losslessly to f64"
    ):
        assert series.index_of(-(2**53))


def test_out_of_range_float32() -> None:
    series = pl.Series([0, 255, None], dtype=pl.Float32)
    # Small numbers are fine:
    assert series.index_of(1_000_000) is None
    assert series.index_of(-1_000_000) is None
    with pytest.raises(
        InvalidOperationError, match=f"cannot cast {2**24} losslessly to f32"
    ):
        assert series.index_of(2**24)
    with pytest.raises(
        InvalidOperationError, match=f"cannot cast {-(2**24)} losslessly to f32"
    ):
        assert series.index_of(-(2**24))


def assert_lossy_cast_rejected(
    series_dtype: PolarsDataType, value: Any, value_dtype: PolarsDataType
) -> None:
    # We create a Series with a null because previously lossless casts would
    # sometimes get turned into nulls and you'd get an answer.
    series = pl.Series([None], dtype=series_dtype)
    with pytest.raises(InvalidOperationError, match="cannot cast losslessly"):
        series.index_of(pl.lit(value, dtype=value_dtype))


@pytest.mark.parametrize(
    ("series_dtype", "value", "value_dtype"),
    [
        # Completely incompatible:
        (pl.String, 1, pl.UInt8),
        (pl.UInt8, "1", pl.String),
        # Larger integer doesn't fit in smaller integer:
        (pl.UInt8, 17, pl.UInt16),
        # Can't find negative numbers in unsigned integers:
        (pl.UInt16, -1, pl.Int8),
        # Values after the decimal point that can't be represented:
        (pl.Decimal(3, 1), 1, pl.Decimal(4, 2)),
        # Can't fit in Decimal:
        (pl.Decimal(3, 0), 1, pl.Decimal(4, 0)),
        (pl.Decimal(5, 2), 1, pl.Decimal(5, 1)),
        (pl.Decimal(5, 2), 1, pl.UInt16),
        # Can't fit nanoseconds in milliseconds:
        (pl.Duration("ms"), 1, pl.Duration("ns")),
        # Arrays that are the wrong length:
        (pl.Array(pl.Int64, 2), [1], pl.Array(pl.Int64, 1)),
        # Struct with wrong number of fields:
        (
            pl.Struct({"a": pl.Int64, "b": pl.Int64}),
            {"a": 1},
            pl.Struct({"a": pl.Int64}),
        ),
        # Struct with different field name:
        (pl.Struct({"a": pl.Int64}), {"x": 1}, pl.Struct({"x": pl.Int64})),
    ],
)
def test_lossy_casts_are_rejected(
    series_dtype: PolarsDataType, value: Any, value_dtype: PolarsDataType
) -> None:
    assert_lossy_cast_rejected(series_dtype, value, value_dtype)


def test_lossy_casts_are_rejected_nested_dtypes() -> None:
    # Make sure casting rules are applied recursively for Lists, Arrays,
    # Struct:
    series_dtype, value, value_dtype = pl.UInt8, 17, pl.UInt16
    assert_lossy_cast_rejected(pl.List(series_dtype), [value], pl.List(value_dtype))
    assert_lossy_cast_rejected(
        pl.Array(series_dtype, 1), [value], pl.Array(value_dtype, 1)
    )
    assert_lossy_cast_rejected(
        pl.Struct({"key": series_dtype}),
        {"key": value},
        pl.Struct({"key": value_dtype}),
    )


def test_decimal_search_for_int() -> None:
    values = [Decimal(-12), Decimal(12), Decimal(30)]
    series = pl.Series(values, dtype=pl.Decimal(4, 1))
    for i, value in enumerate(values):
        assert series.index_of(value) == i
        assert series.index_of(int(value)) == i
        assert series.index_of(np.int8(value)) == i  # type: ignore[arg-type]
    # Decimal's integer range is 3 digits (3 == 4 - 1), so int8 fits:
    assert series.index_of(np.int8(127)) is None  # type: ignore[arg-type]
    assert series.index_of(np.int8(-128)) is None  # type: ignore[arg-type]
