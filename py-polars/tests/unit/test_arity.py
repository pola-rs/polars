import polars as pl
from polars.testing import assert_frame_equal


def test_expression_literal_series_order() -> None:
    s = pl.Series([1, 2, 3])
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = df.select(pl.col("a") + s)
    expected = pl.DataFrame({"a": [2, 4, 6]})
    assert_frame_equal(result, expected)

    result = df.select(pl.lit(s) + pl.col("a"))
    expected = pl.DataFrame({"": [2, 4, 6]})
    assert_frame_equal(result, expected)
