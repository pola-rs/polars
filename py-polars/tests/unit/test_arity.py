import polars as pl


def test_expression_literal_series_order() -> None:
    s = pl.Series([1, 2, 3])
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert df.select(pl.col("a") + s).to_dict(False) == {"a": [2, 4, 6]}
    assert df.select(pl.lit(s) + pl.col("a")).to_dict(False) == {"": [2, 4, 6]}
