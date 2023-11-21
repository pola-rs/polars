import pytest

import polars as pl


def test_enum_creation() -> None:
    s = pl.Series([None, "a", "b"], dtype=pl.Enum(categories=["a", "b"]))
    assert s.null_count() == 1
    assert s.len() == 3
    assert s.dtype == pl.Enum(categories=["a", "b"])


def test_enum_non_existent() -> None:
    with pytest.raises(
        pl.OutOfBoundsError,
        match=("in string column not found in fixed set of categories"),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Enum(categories=["a", "b"]))


def test_nested_enum_creation() -> None:
    dtype = pl.List(pl.Enum(["a", "b", "c"]))
    s = pl.Series([[None, "a"], ["b", "c"]], dtype=dtype)
    assert s.len() == 2
    assert s.dtype == dtype
