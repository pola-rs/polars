from __future__ import annotations

import pandas as pd
import pytest

import polars as pl


def test_from_pandas_exclude_index() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.Index([5, 6], name="c"))
    df = pl.from_pandas(data, include_index=False)
    assert df.columns == ["a", "b"]
    assert df.rows() == [(1, 3), (2, 4)]


def test_from_pandas_include_index() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.Index([5, 6], name="c"))
    df = pl.from_pandas(data, include_index=True)
    assert df.columns == ["c", "a", "b"]
    assert df.rows() == [(5, 1, 3), (6, 2, 4)]


def test_from_pandas_exclude_dup_index() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.Index([5, 6], name="a"))
    df = pl.from_pandas(data, include_index=False)
    assert df.columns == ["a", "b"]
    assert df.rows() == [(1, 3), (2, 4)]


def test_from_pandas_include_dup_index() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.Index([5, 6], name="a"))

    with pytest.raises(ValueError):
        pl.from_pandas(data, include_index=True)
