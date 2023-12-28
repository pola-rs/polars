"""
Test some basic methods of the dataframe consortium standard.

Full testing is done at https://github.com/data-apis/dataframe-api-compat,
this is just to check that the entry point works as expected.
"""

import polars as pl


def test_dataframe() -> None:
    df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df_pl.__dataframe_consortium_standard__()
    result = df.get_column_names()
    expected = ["a", "b"]
    assert result == expected


def test_lazyframe() -> None:
    df_pl = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df_pl.__dataframe_consortium_standard__()
    result = df.get_column_names()
    expected = ["a", "b"]
    assert result == expected


def test_series() -> None:
    ser = pl.Series("a", [1, 2, 3])
    col = ser.__column_consortium_standard__()
    assert col.name == "a"
