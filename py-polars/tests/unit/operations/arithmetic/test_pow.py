import pytest

import polars as pl
from polars.exceptions import InvalidOperationError


def test_pow_dtype() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5],
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 5],
            "d": [1, 2, 3, 4, 5],
            "e": [1, 2, 3, 4, 5],
            "f": [1, 2, 3, 4, 5],
            "g": [1, 2, 1, 2, 1],
            "h": [1, 2, 1, 2, 1],
        },
        schema_overrides={
            "a": pl.Int64,
            "b": pl.UInt64,
            "c": pl.Int32,
            "d": pl.UInt32,
            "e": pl.Int16,
            "f": pl.UInt16,
            "g": pl.Int8,
            "h": pl.UInt8,
        },
    ).lazy()

    df = (
        df.with_columns([pl.col("foo").cast(pl.UInt32)])
        .with_columns(
            (pl.col("foo") * 2**2).alias("scaled_foo"),
            (pl.col("foo") * 2**2.1).alias("scaled_foo2"),
            (pl.col("a") ** pl.col("h")).alias("a_pow_h"),
            (pl.col("b") ** pl.col("h")).alias("b_pow_h"),
            (pl.col("c") ** pl.col("h")).alias("c_pow_h"),
            (pl.col("d") ** pl.col("h")).alias("d_pow_h"),
            (pl.col("e") ** pl.col("h")).alias("e_pow_h"),
            (pl.col("f") ** pl.col("h")).alias("f_pow_h"),
            (pl.col("g") ** pl.col("h")).alias("g_pow_h"),
            (pl.col("h") ** pl.col("h")).alias("h_pow_h"),
        )
        .drop(["a", "b", "c", "d", "e", "f", "g", "h"])
    )
    expected = [
        pl.UInt32,
        pl.UInt32,
        pl.Float64,
        pl.Int64,
        pl.UInt64,
        pl.Int32,
        pl.UInt32,
        pl.Int16,
        pl.UInt16,
        pl.Int8,
        pl.UInt8,
    ]
    assert df.collect().dtypes == expected
    assert df.collect_schema().dtypes() == expected


def test_pow_collect_schema_raises_non_numeric_base() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        pl.col("a") ** 2
    )
    with pytest.raises(
        InvalidOperationError,
        match="`pow` operation not supported for dtype `str` as base",
    ):
        lf.collect_schema()


def test_pow_collect_schema_raises_non_numeric_exponent() -> None:
    lf = pl.LazyFrame({"a": [1], "b": [None]}, schema={"a": pl.Int64, "b": pl.String}).select(
        pl.col("a") ** pl.col("b")
    )
    with pytest.raises(
        InvalidOperationError,
        match="`pow` operation not supported for dtype `str` as exponent",
    ):
        lf.collect_schema()


def test_sqrt_collect_schema_raises_non_numeric() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        pl.col("a").sqrt()
    )
    with pytest.raises(
        InvalidOperationError, match="`sqrt` operation not supported for dtype `str`"
    ):
        lf.collect_schema()


def test_sqrt_collect_raises_non_numeric() -> None:
    # Previously this silently auto-cast String -> Float64; now matches `pow` and raises.
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        pl.col("a").sqrt()
    )
    with pytest.raises(
        InvalidOperationError, match="`sqrt` operation not supported for dtype `str`"
    ):
        lf.collect()


def test_cbrt_collect_schema_raises_non_numeric() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        pl.col("a").cbrt()
    )
    with pytest.raises(
        InvalidOperationError, match="`cbrt` operation not supported for dtype `str`"
    ):
        lf.collect_schema()


def test_cbrt_collect_raises_non_numeric() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        pl.col("a").cbrt()
    )
    with pytest.raises(
        InvalidOperationError, match="`cbrt` operation not supported for dtype `str`"
    ):
        lf.collect()
