import polars as pl


def test_quantile_expr_input() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0, 0, 0.3, 0.2, 0]})

    assert df.select([pl.col("a").quantile(pl.col("b").sum() + 0.1)]).frame_equal(
        df.select(pl.col("a").quantile(0.6))
    )
