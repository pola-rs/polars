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

if TYPE_CHECKING:
    from polars._typing import IntoExpr
from polars.testing import assert_frame_equal


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
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat([pl.Series([1, 7], dtype=dtype), series], rechunk=False)

    extra_values = [
        np.int8(3),
        np.int64(2**42),
        np.float64(1.5),
        np.float32(1.5),
        np.float32(2**37),
        np.float64(2**100),
    ]
    for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
        for value in values:
            assert_index_of(s, value, convert_to_literal=True)
            assert_index_of(s, value, convert_to_literal=False)
        for value in extra_values:  # type: ignore[assignment]
            assert_index_of(s, value)

    # Explicitly check some extra-tricky edge cases:
    assert series.index_of(-np.nan) == 1  # -np.nan should match np.nan
    assert series.index_of(-0.0) == 6  # -0.0 should match 0.0


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
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int128,
    ],
)
def test_integer(dtype: pl.DataType) -> None:
    values = [
        51,
        3,
        None,
        4,
        pl.select(dtype.max()).item(),  # type: ignore[attr-defined]
        pl.select(dtype.min()).item(),  # type: ignore[attr-defined]
    ]
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat(
        [pl.Series([100, 7], dtype=dtype), series], rechunk=False
    )

    extra_values = [pl.select(v).item() for v in [dtype.max() - 1, dtype.min() + 1]]  # type: ignore[attr-defined]
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
            with pytest.raises(InvalidOperationError, match="cannot cast lossless"):
                s.index_of(f)  # type: ignore[arg-type]


def test_groupby() -> None:
    df = pl.DataFrame(
        {"label": ["a", "b", "a", "b", "a", "b"], "value": [10, 3, 20, 2, 40, 20]}
    )
    expected = pl.DataFrame(
        {"label": ["a", "b"], "value": [1, 2]},
        schema={"label": pl.String, "value": pl.UInt32},
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
        (pl.Series([Decimal(12), None, Decimal(3)]), [Decimal(4)], True),
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
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/20318"
            ),
        ),
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/20171"
            ),
        ),
    ],
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
