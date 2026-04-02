from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.fixture
def clip_exprs() -> list[pl.Expr]:
    return [
        pl.col("a").clip(pl.col("min"), pl.col("max")).alias("clip"),
        pl.col("a").clip(lower_bound=pl.col("min")).alias("clip_min"),
        pl.col("a").clip(upper_bound=pl.col("max")).alias("clip_max"),
    ]


def test_clip_int(clip_exprs: list[pl.Expr]) -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, None],
            "min": [0, -1, 4, None, 4, -10],
            "max": [2, 1, 8, 5, None, 10],
        }
    )
    result = lf.select(clip_exprs)
    expected = pl.LazyFrame(
        {
            "clip": [1, 1, 4, 4, 5, None],
            "clip_min": [1, 2, 4, 4, 5, None],
            "clip_max": [1, 1, 3, 4, 5, None],
        }
    )
    assert_frame_equal(result, expected)


def test_clip_float(clip_exprs: list[pl.Expr]) -> None:
    lf = pl.LazyFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "min": [0.0, -1.0, 4.0, None, 4.0, None],
            "max": [2.0, 1.0, 8.0, 5.0, None, None],
        }
    )
    result = lf.select(clip_exprs)
    expected = pl.LazyFrame(
        {
            "clip": [1.0, 1.0, 4.0, 4.0, 5.0, None],
            "clip_min": [1.0, 2.0, 4.0, 4.0, 5.0, None],
            "clip_max": [1.0, 1.0, 3.0, 4.0, 5.0, None],
        }
    )
    assert_frame_equal(result, expected)


def test_clip_datetime(clip_exprs: list[pl.Expr]) -> None:
    lf = pl.LazyFrame(
        {
            "a": [
                datetime(1995, 6, 5, 10, 30),
                datetime(1995, 6, 5),
                datetime(2023, 10, 20, 18, 30, 6),
                None,
                datetime(2023, 9, 24),
                datetime(2000, 1, 10),
            ],
            "min": [
                datetime(1995, 6, 5, 10, 29),
                datetime(1996, 6, 5),
                datetime(2020, 9, 24),
                datetime(2020, 1, 1),
                None,
                datetime(2000, 1, 1),
            ],
            "max": [
                datetime(1995, 7, 21, 10, 30),
                datetime(2000, 1, 1),
                datetime(2023, 9, 20, 18, 30, 6),
                datetime(2000, 1, 1),
                datetime(1993, 3, 13),
                None,
            ],
        }
    )
    result = lf.select(clip_exprs)
    expected = pl.LazyFrame(
        {
            "clip": [
                datetime(1995, 6, 5, 10, 30),
                datetime(1996, 6, 5),
                datetime(2023, 9, 20, 18, 30, 6),
                None,
                datetime(1993, 3, 13),
                datetime(2000, 1, 10),
            ],
            "clip_min": [
                datetime(1995, 6, 5, 10, 30),
                datetime(1996, 6, 5),
                datetime(2023, 10, 20, 18, 30, 6),
                None,
                datetime(2023, 9, 24),
                datetime(2000, 1, 10),
            ],
            "clip_max": [
                datetime(1995, 6, 5, 10, 30),
                datetime(1995, 6, 5),
                datetime(2023, 9, 20, 18, 30, 6),
                None,
                datetime(1993, 3, 13),
                datetime(2000, 1, 10),
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_clip_non_numeric_dtype_fails() -> None:
    msg = "`clip` only supports physical numeric types"

    s = pl.Series(["a", "b", "c"])
    with pytest.raises(InvalidOperationError, match=msg):
        s.clip(pl.lit("b"), pl.lit("z"))


def test_clip_string_input() -> None:
    df = pl.DataFrame({"a": [0, 1, 2], "min": [1, None, 1]})
    result = df.select(pl.col("a").clip("min"))
    expected = pl.DataFrame({"a": [1, 1, 2]})
    assert_frame_equal(result, expected)


def test_clip_bound_invalid_for_original_dtype() -> None:
    s = pl.Series([1, 2, 3, 4], dtype=pl.UInt32)
    with pytest.raises(
        InvalidOperationError, match="conversion from `i32` to `u32` failed"
    ):
        s.clip(-1, 5)


def test_clip_decimal() -> None:
    ser = pl.Series("a", ["1.1", "2.2", "3.3"], pl.Decimal(21, 1))

    result = ser.clip(lower_bound=Decimal("1.5"), upper_bound=Decimal("2.5"))
    expected = pl.Series("a", ["1.5", "2.2", "2.5"], pl.Decimal(21, 1))
    assert_series_equal(result, expected)

    result = ser.clip(lower_bound=Decimal("1.5"))
    expected = pl.Series("a", ["1.5", "2.2", "3.3"], pl.Decimal(21, 1))
    assert_series_equal(result, expected)

    result = ser.clip(upper_bound=Decimal("2.5"))
    expected = pl.Series("a", ["1.1", "2.2", "2.5"], pl.Decimal(21, 1))
    assert_series_equal(result, expected)


def test_clip_unequal_lengths_22018() -> None:
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([1, 2, 3]).clip(lower_bound=pl.Series([1, 2]))
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([1, 2, 3]).clip(upper_bound=pl.Series([1, 2]))
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([1, 2, 3]).clip(pl.Series([1, 2]), pl.Series([1, 2, 3]))
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([1, 2, 3]).clip(pl.Series([1, 2, 3]), pl.Series([1, 2]))


def test_clip_mixed_scalar_series_bound_with_nulls_27086() -> None:
    s = pl.Series([0, 5, 8])

    result = s.clip(lower_bound=2, upper_bound=pl.Series([None, 6, 7]))
    assert_series_equal(result, pl.Series([2, 5, 7]))

    result = pl.Series([8, 5, 8]).clip(
        lower_bound=pl.Series([None, 1, 3]), upper_bound=6
    )
    assert_series_equal(result, pl.Series([6, 5, 6]))

    s_with_nulls = pl.Series([None, 5, 8], dtype=pl.Int64)
    result = s_with_nulls.clip(lower_bound=2, upper_bound=pl.Series([None, 6, 7]))
    assert_series_equal(result, pl.Series([None, 5, 7], dtype=pl.Int64))

    result = pl.Series([None, 5, 8], dtype=pl.Int64).clip(
        lower_bound=pl.Series([None, 1, 3]), upper_bound=6
    )
    assert_series_equal(result, pl.Series([None, 5, 6], dtype=pl.Int64))

    null_scalar = pl.Series([None], dtype=pl.Int64)

    assert_series_equal(
        s.clip(lower_bound=null_scalar, upper_bound=pl.Series([3, 4, 9])),
        pl.Series([0, 4, 8]),
    )

    assert_series_equal(
        s.clip(lower_bound=pl.Series([1, 6, 3]), upper_bound=null_scalar),
        pl.Series([1, 6, 8]),
    )

    assert_series_equal(
        s.clip(lower_bound=null_scalar, upper_bound=null_scalar),
        s,
    )

    assert_series_equal(
        pl.Series([0, 5, 8]).clip(lower_bound=pl.Series([None, 3, 3])),
        pl.Series([0, 5, 8]),
    )
    assert_series_equal(
        pl.Series([0, 5, 8]).clip(upper_bound=pl.Series([None, 4, 4])),
        pl.Series([0, 4, 4]),
    )


def test_clip_mixed_scalar_series_bound_with_nulls_lazy_27086() -> None:
    lf = pl.LazyFrame({"a": [0, 5, 8], "upper": [None, 6, 7]})
    result = lf.select(pl.col("a").clip(lower_bound=2, upper_bound=pl.col("upper")))
    assert_frame_equal(result, pl.LazyFrame({"a": [2, 5, 7]}))

    lf = pl.LazyFrame({"a": [8, 5, 8], "lower": [None, 1, 3]})
    result = lf.select(pl.col("a").clip(lower_bound=pl.col("lower"), upper_bound=6))
    assert_frame_equal(result, pl.LazyFrame({"a": [6, 5, 6]}))

    lf = pl.LazyFrame({"a": [None, 5, 8], "upper": [None, 6, 7]})
    result = lf.select(pl.col("a").clip(lower_bound=2, upper_bound=pl.col("upper")))
    assert_frame_equal(result, pl.LazyFrame({"a": [None, 5, 7]}))


def test_clip_bound_nan() -> None:
    assert_series_equal(
        pl.Series([1.0, 2.0]).clip(float("nan"), float("nan")),
        pl.Series([1.0, 2.0]),
    )

    assert_series_equal(
        pl.Series([1.0, 2.0]).clip(float("nan"), None),
        pl.Series([1.0, 2.0]),
    )

    assert_series_equal(
        pl.Series([1.0, 2.0]).clip(None, float("nan")),
        pl.Series([1.0, 2.0]),
    )
