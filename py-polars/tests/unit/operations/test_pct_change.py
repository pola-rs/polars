from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_pct_change() -> None:
    s = pl.Series("a", [1, 2, 4, 8, 16, 32, 64])
    expected = pl.Series("a", [None, None, 3.0, 3.0, 3.0, 3.0, 3.0])
    assert_series_equal(s.pct_change(2), expected)
    assert_series_equal(s.pct_change(pl.Series([2])), expected)
    # negative
    assert pl.Series(range(5)).pct_change(-1).to_list() == [
        -1.0,
        -0.5,
        -0.3333333333333333,
        -0.25,
        None,
    ]


def test_pct_change_nulls() -> None:
    df = pl.DataFrame(
        {
            "a": [10, 11, 12, None, 12, 24],
        }
    )
    result = df.select(pl.col("a").pct_change().alias("pct_change"))
    expected = pl.DataFrame({"pct_change": [None, 0.1, 0.090909, None, None, 1.0]})
    assert_frame_equal(result, expected)
