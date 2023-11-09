import polars as pl


def test_nan_in_group_by_agg() -> None:
    df = pl.DataFrame(
        {
            "key": ["a", "a", "a", "a"],
            "value": [18.58, 18.78, float("nan"), 18.63],
            "bar": [0, 0, 0, 0],
        }
    )

    assert df.group_by("bar", "key").agg(pl.col("value").max())["value"].item() == 18.78
    assert df.group_by("bar", "key").agg(pl.col("value").min())["value"].item() == 18.58


def test_nan_aggregations() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 2.0, 3.0], "b": [1, 1, 1, 1]})

    aggs = [
        pl.col("a").max().alias("max"),
        pl.col("a").min().alias("min"),
        pl.col("a").nan_max().alias("nan_max"),
        pl.col("a").nan_min().alias("nan_min"),
    ]

    assert (
        str(df.select(aggs).to_dict(as_series=False))
        == "{'max': [3.0], 'min': [1.0], 'nan_max': [nan], 'nan_min': [nan]}"
    )
    assert (
        str(df.group_by("b").agg(aggs).to_dict(as_series=False))
        == "{'b': [1], 'max': [3.0], 'min': [1.0], 'nan_max': [nan], 'nan_min': [nan]}"
    )
