from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_pos() -> None:
    df = pl.LazyFrame({"x": [1, 2]})
    result = df.select(+pl.col("x"))
    assert_frame_equal(result, df)


def test_pos_string() -> None:
    a = pl.Series("a", [""])
    assert_series_equal(+a, a)


def test_pos_datetime() -> None:
    a = pl.Series("a", [datetime(2022, 1, 1)])
    assert_series_equal(+a, a)
