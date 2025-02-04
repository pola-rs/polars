"""Tests for ``.list.index_of_in()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

from tests.unit.conftest import INTEGER_DTYPES
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


# Testing plan:
# D All integers
# - Both floats, with nans
# - Strings
# - datetime, date, time, timedelta
# - nested lists
# - something with hypothesis
# - error case: non-matching lengths


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
    for will_be_lossy in [
        np.float32(3.1),
        np.float64(3.1),
        50.9,
        300,
        -300,
        pl.lit(300, dtype=pl.Int16),
    ]:
        with pytest.raises(InvalidOperationError, match="cannot cast lossless"):
            list_series.list.index_of_in(will_be_lossy)  # type: ignore[arg-type]
