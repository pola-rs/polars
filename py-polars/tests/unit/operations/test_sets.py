import polars as pl


def test_set_intersection_13765() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([[1], [1]], dtype=pl.List(pl.UInt32)),
            "f": pl.Series([1, 2], dtype=pl.UInt32),
        }
    )

    df = df.join(df, how="cross", suffix="_other")
    df = df.filter(pl.col("f") == 1)

    df.select(pl.col("a").list.set_intersection("a_other")).to_dict(as_series=False)
