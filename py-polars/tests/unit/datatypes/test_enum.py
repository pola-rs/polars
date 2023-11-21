import pytest

import polars as pl
from polars.testing import assert_series_equal


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


def test_enum_from_schema_argument() -> None:
    df = pl.DataFrame(
        {"col1": ["a", "b", "c"]}, schema={"col1": pl.Enum(["a", "b", "c"])}
    )
    assert df.get_column("col1").dtype == pl.Enum


def test_equality_of_two_separately_constructed_enums() -> None:
    s = pl.Series([None, "a", "b"], dtype=pl.Enum(categories=["a", "b"]))
    s2 = pl.Series([None, "a", "b"], dtype=pl.Enum(categories=["a", "b"]))
    assert_series_equal(s, s2)


def test_nested_enum_creation() -> None:
    dtype = pl.List(pl.Enum(["a", "b", "c"]))
    s = pl.Series([[None, "a"], ["b", "c"]], dtype=dtype)
    assert s.len() == 2
    assert s.dtype == dtype
