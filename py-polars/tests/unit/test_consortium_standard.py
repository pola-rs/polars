"""
Test some basic methods of the dataframe consortium standard.

Full testing is done at https://github.com/data-apis/dataframe-api-compat,
this is just to check that the entry point works as expected.
"""
import pytest

import polars as pl

pytest.importorskip("dataframe-api-compat")


def test_dataframe() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.__dataframe_consortium_standard__()
    result = df.get_column_names()
    expected = ["a", "b"]
    assert result == expected


def test_lazyframe() -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.__dataframe_consortium_standard__()
    result = df.get_column_names()
    expected = ["a", "b"]
    assert result == expected


def test_series() -> None:
    ser = pl.Series([1, 2, 3])
    ser = ser.__column_consortium_standard__()
    result = ser.get_value(1)
    expected = 2
    assert result == expected
