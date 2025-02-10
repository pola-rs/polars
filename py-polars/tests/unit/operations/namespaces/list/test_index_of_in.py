"""Tests for ``.list.index_of_in()``."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import FLOAT_DTYPES, INTEGER_DTYPES
from tests.unit.operations.test_index_of import get_expected_index

if TYPE_CHECKING:
    from polars._typing import IntoExpr, PythonLiteral

IdxType = pl.get_index_type()


def assert_index_of_in_from_scalar(
    list_series: pl.Series, value: PythonLiteral
) -> None:
    expected_indexes = [
        None if sub_series is None else get_expected_index(sub_series, value)
        for sub_series in list_series
    ]

    original_value = value
    del value
    for updated_value in (original_value, pl.lit(original_value)):
        # Eager API:
        assert_series_equal(
            list_series.list.index_of_in(updated_value),
            pl.Series(list_series.name, expected_indexes, dtype=IdxType),
        )
        # Lazy API:
        assert_frame_equal(
            pl.LazyFrame({"lists": list_series})
            .select(pl.col("lists").list.index_of_in(updated_value))
            .collect(),
            pl.DataFrame({"lists": expected_indexes}, schema={"lists": IdxType}),
        )


def assert_index_of_in_from_series(
    list_series: pl.Series,
    values: pl.Series,
) -> None:
    expected_indexes = [
        None if sub_series is None else get_expected_index(sub_series, value)
        for (sub_series, value) in zip(list_series, values)
    ]

    # Eager API:
    assert_series_equal(
        list_series.list.index_of_in(values),
        pl.Series(list_series.name, expected_indexes, dtype=IdxType),
    )
    # Lazy API:
    assert_frame_equal(
        pl.LazyFrame({"lists": list_series, "values": values})
        .select(pl.col("lists").list.index_of_in(pl.col("values")))
        .collect(),
        pl.DataFrame({"lists": expected_indexes}, schema={"lists": IdxType}),
    )


def test_index_of_in_from_scalar() -> None:
    list_series = pl.Series([[3, 1], [2, 4], [5, 3, 1]])
    assert_index_of_in_from_scalar(list_series, 1)


def test_index_of_in_from_series() -> None:
    list_series = pl.Series([[3, 1], [2, 4], [5, 3, 1]])
    values = pl.Series([1, 2, 6])
    assert_index_of_in_from_series(list_series, values)


def to_int(expr: pl.Expr) -> int:
    return pl.select(expr).item()


@pytest.mark.parametrize("lists_dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("values_dtype", INTEGER_DTYPES)
def test_integer(lists_dtype: pl.DataType, values_dtype: pl.DataType) -> None:
    lists = [
        [51, 3],
        [None, 4],
        None,
        [to_int(lists_dtype.max()), 3],  # type: ignore[attr-defined]
        [6, to_int(lists_dtype.min())],  # type: ignore[attr-defined]
    ]
    lists_series = pl.Series(lists, dtype=pl.List(lists_dtype))
    chunked_series = pl.concat(
        [pl.Series([[100, 7]], dtype=pl.List(lists_dtype)), lists_series], rechunk=False
    )
    values = [
        to_int(v) for v in [lists_dtype.max() - 1, lists_dtype.min() + 1]
    ]  # type: ignore[attr-defined]
    for sublist in lists:
        if sublist is None:
            values.append(None)
        else:
            values.extend(sublist)

    # Scalars:
    for s in [lists_series, chunked_series]:
        value: IntoExpr
        for value in values:
            assert_index_of_in_from_scalar(s, value)

    # Series
    search_series = pl.Series([3, 4, 7, None, 6], dtype=values_dtype)
    assert_index_of_in_from_series(lists_series, search_series)
    search_series = pl.Series([17, 3, 4, 7, None, 6], dtype=values_dtype)
    assert_index_of_in_from_series(chunked_series, search_series)


def test_no_lossy_numeric_casts() -> None:
    list_series = pl.Series([[3]], dtype=pl.List(pl.Int8()))
    for will_be_lossy in [np.float32(3.1), np.float64(3.1), 50.9]:
        with pytest.raises(InvalidOperationError, match="cannot cast lossless"):
            list_series.list.index_of_in(will_be_lossy)  # type: ignore[arg-type]

    for will_be_lossy in [300, -300, pl.lit(300, dtype=pl.Int16)]:
        with pytest.raises(InvalidOperationError, match="conversion from"):
            list_series.list.index_of_in(will_be_lossy)  # type: ignore[arg-type]


def test_multichunk_needles() -> None:
    series = pl.Series([[1, 3], [3, 2], [4, 5, 3]])
    needles = pl.concat([pl.Series([3, 1]), pl.Series([3])])
    assert series.list.index_of_in(needles).to_list() == [1, None, 2]


def test_mismatched_length() -> None:
    """
    Mismatched lengths result in an error.

    Unfortunately a length 1 Series will be treated as a _scalar_, which seems
    weird, but that's how e.g. list.contains() works so maybe that's
    intentional.
    """
    series = pl.Series([[1, 3], [3, 2], [4, 5, 3]])
    needles = pl.Series([3, 2])
    with pytest.raises(ComputeError, match="shapes don't match"):
        series.list.index_of_in(pl.Series(needles))


def all_values(list_series: pl.Series) -> list:
    values = []
    for subseries in list_series.to_list():
        if subseries is not None:
            values.extend(subseries)
    return values


@pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
def test_float(float_dtype: pl.DataType) -> None:
    lists = [
        [1.5, np.nan, np.inf],
        [3.0, None, -np.inf],
        [0.0, -0.0, -np.nan],
        None,
        [None, None],
    ]
    lists_series = pl.Series(lists, dtype=pl.List(float_dtype))

    # Scalar
    for value in all_values(lists_series) + [
        None,
        3.5,
        np.float64(1.5),
        np.float32(3.0),
    ]:
        assert_index_of_in_from_scalar(lists_series, value)

    # Series
    assert_index_of_in_from_series(
        lists_series, pl.Series([1.5, -np.inf, -np.nan, 3, None], dtype=float_dtype)
    )


@pytest.mark.parametrize(
    ("list_series", "extra_values"),
    [
        (pl.Series([["abc", "def"], ["ghi", "zzz", "X"], ["Y"]]), ["foo"]),
        (pl.Series([[b"abc", b"def"], [b"ghi", b"zzz", b"X"], [b"Y"]]), [b"foo"]),
        (pl.Series([[True, None, False], [True, False]]), []),
        (
            pl.Series(
                [
                    [datetime(1997, 12, 31), datetime(1996, 1, 1)],
                    [datetime(1997, 12, 30), datetime(1996, 1, 2)],
                ]
            ),
            [datetime(2003, 1, 1)],
        ),
        (
            pl.Series(
                [
                    [date(1997, 12, 31), date(1996, 1, 1)],
                    [date(1997, 12, 30), date(1996, 1, 2)],
                ]
            ),
            [date(2003, 1, 1)],
        ),
        (
            pl.Series(
                [
                    [time(16, 12, 31), None, time(11, 10, 53)],
                    [time(16, 11, 31), time(11, 10, 54)],
                ]
            ),
            [time(12, 6, 7)],
        ),
        (
            pl.Series(
                [
                    [timedelta(hours=12), None, timedelta(minutes=3)],
                    [timedelta(hours=3), None, timedelta(hours=1)],
                ],
            ),
            [timedelta(minutes=7)],
        ),
        (
            pl.Series(
                [[Decimal(12), None, Decimal(3)], [Decimal(500), None, Decimal(16)]]
            ),
            [Decimal(4)],
        ),
        (
            pl.Series([[[1, 2], None], [[4, 5], [6]], None, [[None, 3, 5]], [None]]),
            [[5, 7], []],
        ),
        (
            pl.Series(
                [
                    [[[1, 2], None], [[4, 5], [6]]],
                    [[[None, 3, 5]]],
                    None,
                    [None],
                    [[None]],
                    [[[None]]],
                ]
            ),
            [[[5, 7]], [[]], [None]],
        ),
        (
            pl.Series(
                [[[1, 2]], [[4, 5]], [[None, 3]], [None], None],
                dtype=pl.List(pl.Array(pl.Int64(), 2)),
            ),
            [[5, 7]],
        ),
        (
            pl.Series(
                [
                    [{"a": 1, "b": 2}, None],
                    [{"a": 3, "b": 4}, {"a": None, "b": 2}],
                    None,
                ],
                dtype=pl.List(pl.Struct({"a": pl.Int64(), "b": pl.Int64()})),
            ),
            [{"a": 7, "b": None}, {"a": 6, "b": 4}],
        ),
        (
            pl.Series(
                [["a", "c"], [None, "b"], ["b", "a", "a", "c"], None, [None]],
                dtype=pl.List(pl.Enum(["c", "b", "a"])),
            ),
            [],
        ),
    ],
)
def test_other_types(list_series: pl.Series, extra_values: list[PythonLiteral]) -> None:
    needles_series = pl.Series(
        [
            None if sublist is None else sublist[i % len(sublist)]
            for (i, sublist) in enumerate(list_series)
        ],
        dtype=list_series.dtype.inner,
    )
    assert_index_of_in_from_series(list_series, needles_series)

    values = all_values(list_series) + extra_values + [None]
    for value in values:
        assert_index_of_in_from_scalar(list_series, value)


@pytest.mark.xfail(reason="Depends on Series.index_of supporting Categoricals")
def test_categorical() -> None:
    # When this starts passing, convert to test_other_types entry above.
    series = pl.Series(
        [["a", "c"], [None, "b"], ["b", "a", "a", "c"], None, [None]],
        dtype=pl.List(pl.Categorical),
    )
    assert series.list.index_of_in("b").to_list() == [None, 1, 0, None, None]


def test_nulls() -> None:
    series = pl.Series([[None, None], None], dtype=pl.List(pl.Null))
    assert series.list.index_of_in(None).to_list() == [0, None]

    series = pl.Series([None, [None, None]], dtype=pl.List(pl.Int64))
    assert series.list.index_of_in(None).to_list() == [None, 0]
    assert series.list.index_of_in(1).to_list() == [None, None]


def test_wrong_type() -> None:
    series = pl.Series([[1, 2, 3], [4, 5]])
    with pytest.raises(
        ComputeError,
        match=r"dtypes didn't match: series values have dtype i64 and needle has dtype list\[i64\]",
    ):
        # Searching for a list won't work:
        series.list.index_of_in([1, 2])
