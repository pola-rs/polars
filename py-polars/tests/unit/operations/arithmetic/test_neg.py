from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    "dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
)
def test_neg_operator(dtype: PolarsDataType) -> None:
    lf = pl.LazyFrame({"a": [-1, 0, 1, None]}, schema={"a": dtype})
    result = lf.select(-pl.col("a"))
    expected = pl.LazyFrame({"a": [1, 0, -1, None]}, schema={"a": dtype})
    assert_frame_equal(result, expected)


def test_neg_method() -> None:
    lf = pl.LazyFrame({"a": [-1, 0, 1, None]})
    result_op = lf.select(-pl.col("a"))
    result_method = lf.select(pl.col("a").neg())
    assert_frame_equal(result_op, result_method)


def test_neg_decimal() -> None:
    lf = pl.LazyFrame({"a": [D("-1.5"), D("0.0"), D("5.0"), None]})
    result = lf.select(-pl.col("a"))
    expected = pl.LazyFrame({"a": [D("1.5"), D("0.0"), D("-5.0"), None]})
    assert_frame_equal(result, expected)


def test_neg_duration() -> None:
    lf = pl.LazyFrame({"a": [timedelta(hours=2), timedelta(days=-2), None]})
    result = lf.select(-pl.col("a"))
    expected = pl.LazyFrame({"a": [timedelta(hours=-2), timedelta(days=2), None]})
    assert_frame_equal(result, expected)


def test_neg_overflow_wrapping() -> None:
    df = pl.DataFrame({"a": [-128]}, schema={"a": pl.Int8})
    result = df.select(-pl.col("a"))
    assert_frame_equal(result, df)


def test_neg_unsigned_int() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8})
    with pytest.raises(
        InvalidOperationError, match="`neg` operation not supported for dtype `u8`"
    ):
        df.select(-pl.col("a"))


def test_neg_non_numeric() -> None:
    df = pl.DataFrame({"a": ["p", "q", "r"]})
    with pytest.raises(
        InvalidOperationError, match="`neg` operation not supported for dtype `str`"
    ):
        df.select(-pl.col("a"))


def test_neg_series_operator() -> None:
    s = pl.Series("a", [-1, 0, 1, None])
    result = -s
    expected = pl.Series("a", [1, 0, -1, None])
    assert_series_equal(result, expected)
