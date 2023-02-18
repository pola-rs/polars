import polars as pl


def test_comparison_nulls_single() -> None:
    df1 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.Utf8),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    df2 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.Utf8),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    assert (df1 == df2).row(0) == (True, True, True)
    assert (df1 != df2).row(0) == (False, False, False)
