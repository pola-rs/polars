import polars as pl


def test_equals() -> None:
    # Values are checked
    df1 = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    df2 = pl.DataFrame(
        {
            "foo": [3, 2, 1],
            "bar": [8.0, 7.0, 6.0],
            "ham": ["c", "b", "a"],
        }
    )

    assert df1.equals(df1) is True
    assert df1.equals(df2) is False

    # Column names are checked
    df3 = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [6.0, 7.0, 8.0],
            "c": ["a", "b", "c"],
        }
    )
    assert df1.equals(df3) is False

    # Datatypes are NOT checked
    df = pl.DataFrame(
        {
            "foo": [1, 2, None],
            "bar": [6.0, 7.0, None],
            "ham": ["a", "b", None],
        }
    )
    assert df.equals(df.with_columns(pl.col("foo").cast(pl.Int8))) is True
    assert df.equals(df.with_columns(pl.col("ham").cast(pl.Categorical))) is True

    # The null_equal parameter determines if None values are considered equal
    assert df.equals(df) is True
    assert df.equals(df, null_equal=False) is False
