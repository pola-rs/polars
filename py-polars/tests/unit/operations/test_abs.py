from datetime import timedelta
from typing import cast

import numpy as np

import polars as pl
from polars.testing import assert_series_equal


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


def test_abs_logical_type() -> None:
    s = pl.Series([timedelta(hours=1), timedelta(hours=-1)])
    assert s.abs().to_list() == [timedelta(hours=1), timedelta(hours=1)]


def test_abs_expr() -> None:
    df = pl.DataFrame({"x": [-1, 0, 1]})
    out = df.select(abs(pl.col("x")))

    assert out["x"].to_list() == [1, 0, 1]


def test_builtin_abs() -> None:
    s = pl.Series("s", [-1, 0, 1, None])
    assert abs(s).to_list() == [1, 0, 1, None]
