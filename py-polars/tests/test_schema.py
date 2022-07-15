import polars as pl


def test_schema_on_agg() -> None:
    df = pl.DataFrame({"a": ["x", "x", "y", "n"], "b": [1, 2, 3, 4]})

    assert (
        df.lazy()
        .groupby("a")
        .agg(
            [
                pl.col("b").min().alias("min"),
                pl.col("b").max().alias("max"),
                pl.col("b").sum().alias("sum"),
                pl.col("b").first().alias("first"),
                pl.col("b").last().alias("last"),
            ]
        )
    ).schema == {
        "a": pl.Utf8,
        "min": pl.Int64,
        "max": pl.Int64,
        "sum": pl.Int64,
        "first": pl.Int64,
        "last": pl.Int64,
    }
