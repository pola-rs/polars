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
