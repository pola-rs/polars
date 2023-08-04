import polars as pl


def test_nan_in_groupby_agg() -> None:
    df = pl.DataFrame(
        {
            "key": ["a", "a", "a", "a"],
            "value": [18.58, 18.78, float("NaN"), 18.63],
            "bar": [0, 0, 0, 0],
        }
    )

    assert df.groupby("bar", "key").agg(pl.col("value").max())["value"].item() == 18.78
    assert df.groupby("bar", "key").agg(pl.col("value").min())["value"].item() == 18.58
