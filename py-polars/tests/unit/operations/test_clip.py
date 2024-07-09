from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal


@pytest.fixture()
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
