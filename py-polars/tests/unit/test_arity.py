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


def test_when_then_broadcast_nulls_12665() -> None:
    df = pl.DataFrame(
        {
            "val": [1, 2, 3, 4],
            "threshold": [4, None, None, 1],
        }
    )

    assert df.select(
        when=pl.when(pl.col("val") > pl.col("threshold")).then(1).otherwise(0),
    ).to_dict(as_series=False) == {"when": [0, 0, 0, 1]}
