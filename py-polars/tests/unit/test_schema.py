from __future__ import annotations

import polars as pl


def test_contains_dtype_top_level() -> None:
    s = pl.Schema({"x": pl.Int64(), "y": pl.String()})
    assert s.contains_dtype(pl.Int64(), recursive=False)
    assert not s.contains_dtype(pl.Float64(), recursive=False)


def test_contains_dtype_recursive_nested() -> None:
    s = pl.Schema({"x": pl.Int64(), "y": pl.List(pl.Float64)})
    assert not s.contains_dtype(pl.Float64(), recursive=False)
    assert s.contains_dtype(pl.Float64(), recursive=True)


def test_contains_dtype_recursive_struct() -> None:
    s = pl.Schema({"x": pl.Struct({"a": pl.Int32, "b": pl.List(pl.String)})})
    assert s.contains_dtype(pl.String(), recursive=True)
    assert not s.contains_dtype(pl.Float64(), recursive=True)
