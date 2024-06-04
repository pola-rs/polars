import polars as pl
from polars.testing.asserts.frame import assert_frame_equal


def test_rle() -> None:
    values = [1, 1, 2, 1, None, 1, 3, 3]
    lf = pl.LazyFrame({"a": values})

    expected = pl.LazyFrame(
        {"len": [2, 1, 1, 1, 1, 2], "value": [1, 2, 1, None, 1, 3]},
        schema_overrides={"len": pl.get_index_type()},
    )

    result_expr = lf.select(pl.col("a").rle()).unnest("a")
    assert_frame_equal(result_expr, expected)

    result_series = lf.collect().to_series().rle().struct.unnest()
    assert_frame_equal(result_series, expected.collect())


def test_rle_id() -> None:
    values = [1, 1, 2, 1, None, 1, 3, 3]
    lf = pl.LazyFrame({"a": values})

    expected = pl.LazyFrame(
        {"a": [0, 0, 1, 2, 3, 4, 5, 5]}, schema={"a": pl.get_index_type()}
    )

    result_expr = lf.select(pl.col("a").rle_id())
    assert_frame_equal(result_expr, expected)

    result_series = lf.collect().to_series().rle_id()
    assert_frame_equal(result_series.to_frame(), expected.collect())
