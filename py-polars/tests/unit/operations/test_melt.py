import polars as pl
from polars.testing import assert_frame_equal


def test_melt_projection_pd_7747() -> None:
    df = pl.LazyFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "age": [40, 30, 21, 33, 45],
            "weight": [100, 103, 95, 90, 110],
        }
    )
    result = (
        df.with_columns(pl.col("age").alias("wgt"))
        .melt(id_vars="number", value_vars="wgt")
        .select("number", "value")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "value": [40, 30, 21, 33, 45],
        }
    )
    assert_frame_equal(result, expected)
