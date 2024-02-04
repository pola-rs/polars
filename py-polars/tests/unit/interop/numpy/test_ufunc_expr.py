from __future__ import annotations

from typing import Any, cast

import numpy as np

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_ufunc() -> None:
    df = pl.DataFrame([pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt8)])
    out = df.select(
        [
            np.power(pl.col("a"), 2).alias("power_uint8"),  # type: ignore[call-overload]
            np.power(pl.col("a"), 2.0).alias("power_float64"),  # type: ignore[call-overload]
            np.power(pl.col("a"), 2, dtype=np.uint16).alias("power_uint16"),  # type: ignore[call-overload]
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power_uint8", [1, 4, 9, 16], dtype=pl.UInt8),
            pl.Series("power_float64", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
            pl.Series("power_uint16", [1, 4, 9, 16], dtype=pl.UInt16),
        ]
    )
    assert_frame_equal(out, expected)
    assert out.dtypes == expected.dtypes


def test_ufunc_expr_not_first() -> None:
    """Check numpy ufunc expressions also work if expression not the first argument."""
    df = pl.DataFrame([pl.Series("a", [1, 2, 3], dtype=pl.Float64)])
    out = df.select(
        [
            np.power(2.0, cast(Any, pl.col("a"))).alias("power"),
            (2.0 / cast(Any, pl.col("a"))).alias("divide_scalar"),
            (np.array([2, 2, 2]) / cast(Any, pl.col("a"))).alias("divide_array"),
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power", [2**1, 2**2, 2**3], dtype=pl.Float64),
            pl.Series("divide_scalar", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
            pl.Series("divide_array", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(out, expected)


def test_lazy_ufunc() -> None:
    ldf = pl.LazyFrame([pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt8)])
    out = ldf.select(
        [
            np.power(cast(Any, pl.col("a")), 2).alias("power_uint8"),
            np.power(cast(Any, pl.col("a")), 2.0).alias("power_float64"),
            np.power(cast(Any, pl.col("a")), 2, dtype=np.uint16).alias("power_uint16"),
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power_uint8", [1, 4, 9, 16], dtype=pl.UInt8),
            pl.Series("power_float64", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
            pl.Series("power_uint16", [1, 4, 9, 16], dtype=pl.UInt16),
        ]
    )
    assert_frame_equal(out.collect(), expected)


def test_lazy_ufunc_expr_not_first() -> None:
    """Check numpy ufunc expressions also work if expression not the first argument."""
    ldf = pl.LazyFrame([pl.Series("a", [1, 2, 3], dtype=pl.Float64)])
    out = ldf.select(
        [
            np.power(2.0, cast(Any, pl.col("a"))).alias("power"),
            (2.0 / cast(Any, pl.col("a"))).alias("divide_scalar"),
            (np.array([2, 2, 2]) / cast(Any, pl.col("a"))).alias("divide_array"),
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power", [2**1, 2**2, 2**3], dtype=pl.Float64),
            pl.Series("divide_scalar", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
            pl.Series("divide_array", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(out.collect(), expected)


def test_ufunc_recognition() -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1.1, 2.2, 3.3, 4.4]})
    assert_frame_equal(df.select(np.exp(pl.col("b"))), df.select(pl.col("b").exp()))


# https://github.com/pola-rs/polars/issues/6770
def test_ufunc_multiple_expressions() -> None:
    df = pl.DataFrame(
        {
            "v": [
                -4.293,
                -2.4659,
                -1.8378,
                -0.2821,
                -4.5649,
                -3.8128,
                -7.4274,
                3.3443,
                3.8604,
                -4.2200,
            ],
            "u": [
                -11.2268,
                6.3478,
                7.1681,
                3.4986,
                2.7320,
                -1.0695,
                -10.1408,
                11.2327,
                6.6623,
                -8.1412,
            ],
        }
    )
    expected = np.arctan2(df.get_column("v"), df.get_column("u"))
    result = df.select(np.arctan2(pl.col("v"), pl.col("u")))[:, 0]  # type: ignore[call-overload]
    assert_series_equal(expected, result)  # type: ignore[arg-type]


def test_grouped_ufunc() -> None:
    df = pl.DataFrame({"id": ["a", "a", "b", "b"], "values": [0.1, 0.1, -0.1, -0.1]})
    df.group_by("id").agg(pl.col("values").log1p().sum().pipe(np.expm1))
