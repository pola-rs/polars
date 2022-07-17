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


def test_fill_null_minimal_upcast_4056() -> None:
    df = pl.DataFrame({"a": [-1, 2, None]})
    df = df.with_columns(pl.col("a").cast(pl.Int8))
    assert df.with_column(pl.col(pl.Int8).fill_null(-1)).dtypes[0] == pl.Int8
    assert df.with_column(pl.col(pl.Int8).fill_null(-1000)).dtypes[0] == pl.Int32
