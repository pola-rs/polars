from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_abs() -> None:
    # ints
    s = pl.Series([1, -2, 3, -4])
    assert_series_equal(s.abs(), pl.Series([1, 2, 3, 4]))
    assert_series_equal(cast(pl.Series, np.abs(s)), pl.Series([1, 2, 3, 4]))

    # floats
    s = pl.Series([1.0, -2.0, 3, -4.0])
    assert_series_equal(s.abs(), pl.Series([1.0, 2.0, 3.0, 4.0]))
    assert_series_equal(cast(pl.Series, np.abs(s)), pl.Series([1.0, 2.0, 3.0, 4.0]))
    assert_series_equal(
        pl.select(pl.lit(s).abs()).to_series(), pl.Series([1.0, 2.0, 3.0, 4.0])
    )


def test_abs_series_duration() -> None:
    s = pl.Series([timedelta(hours=1), timedelta(hours=-1)])
    assert s.abs().to_list() == [timedelta(hours=1), timedelta(hours=1)]


def test_abs_expr() -> None:
    df = pl.DataFrame({"x": [-1, 0, 1]})
    out = df.select(abs(pl.col("x")))

    assert out["x"].to_list() == [1, 0, 1]


def test_builtin_abs() -> None:
    s = pl.Series("s", [-1, 0, 1, None])
    assert abs(s).to_list() == [1, 0, 1, None]


@pytest.mark.parametrize(
    "dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
)
def test_abs_builtin(dtype: PolarsDataType) -> None:
    lf = pl.LazyFrame({"a": [-1, 0, 1, None]}, schema={"a": dtype})
    result = lf.select(abs(pl.col("a")))
    expected = pl.LazyFrame({"a": [1, 0, 1, None]}, schema={"a": dtype})
    assert_frame_equal(result, expected)


def test_abs_method() -> None:
    lf = pl.LazyFrame({"a": [-1, 0, 1, None]})
    result_op = lf.select(abs(pl.col("a")))
    result_method = lf.select(pl.col("a").abs())
    assert_frame_equal(result_op, result_method)


def test_abs_decimal() -> None:
    lf = pl.LazyFrame({"a": [D("-1.5"), D("0.0"), D("5.0"), None]})
    result = lf.select(pl.col("a").abs())
    expected = pl.LazyFrame({"a": [D("1.5"), D("0.0"), D("5.0"), None]})
    assert_frame_equal(result, expected)


def test_abs_duration() -> None:
    lf = pl.LazyFrame({"a": [timedelta(hours=2), timedelta(days=-2), None]})
    result = lf.select(pl.col("a").abs())
    expected = pl.LazyFrame({"a": [timedelta(hours=2), timedelta(days=2), None]})
    assert_frame_equal(result, expected)


def test_abs_overflow_wrapping() -> None:
    df = pl.DataFrame({"a": [-128]}, schema={"a": pl.Int8})
    result = df.select(pl.col("a").abs())
    assert_frame_equal(result, df)


def test_abs_unsigned_int() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8})
    result = df.select(pl.col("a").abs())
    assert_frame_equal(result, df)


def test_abs_non_numeric() -> None:
    df = pl.DataFrame({"a": ["p", "q", "r"]})
    with pytest.raises(
        InvalidOperationError, match="`abs` operation not supported for dtype `str`"
    ):
        df.select(pl.col("a").abs())


def test_abs_date() -> None:
    df = pl.DataFrame({"date": [date(1960, 1, 1), date(1970, 1, 1), date(1980, 1, 1)]})

    with pytest.raises(
        InvalidOperationError, match="`abs` operation not supported for dtype `date`"
    ):
        df.select(pl.col("date").abs())


def test_abs_series_builtin() -> None:
    s = pl.Series("a", [-1, 0, 1, None])
    result = abs(s)
    expected = pl.Series("a", [1, 0, 1, None])
    assert_series_equal(result, expected)
