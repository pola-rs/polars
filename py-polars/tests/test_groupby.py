import polars as pl


def test_groupby_sorted_empty_dataframe_3680() -> None:
    assert (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .lazy()
        .sort("key")
        .groupby("key")
        .tail(1)
        .collect()
    ).shape == (0, 2)


def test_groupby_custom_agg_empty_list() -> None:
    assert (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .groupby("key")
        .agg(
            [
                pl.col("val").mean().alias("mean"),
                pl.col("val").std().alias("std"),
                pl.col("val").skew().alias("skew"),
                pl.col("val").kurtosis().alias("kurt"),
            ]
        )
    ).dtypes == [pl.Categorical, pl.Float64, pl.Float64, pl.Float64, pl.Float64]
