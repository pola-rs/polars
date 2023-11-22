import pytest

import polars as pl
from polars import StringCache
from polars.testing import assert_series_equal


def test_enum_creation() -> None:
    s = pl.Series([None, "a", "b"], dtype=pl.Enum(categories=["a", "b"]))
    assert s.null_count() == 1
    assert s.len() == 3
    assert s.dtype == pl.Enum(categories=["a", "b"])


def test_enum_non_existent() -> None:
    with pytest.raises(
        pl.OutOfBoundsError,
        match=("value c is not present in Enum"),
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


def test_casting_to_an_enum_from_utf() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"])
    s2 = s.cast(dtype)
    assert s2.dtype == dtype
    assert s2.null_count() == 1


def test_casting_to_an_enum_from_categorical() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    s2 = s.cast(dtype)
    assert s2.dtype == dtype
    assert s2.null_count() == 1
    expected = pl.Series([None, "a", "b", "c"], dtype=dtype)
    assert_series_equal(s2, expected)


def test_casting_to_an_enum_from_categorical_nonexistent() -> None:
    with pytest.raises(
        pl.OutOfBoundsError,
        match=("value c is not present in Enum"),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Categorical).cast(pl.Enum(["a", "b"]))


@StringCache()
def test_casting_to_an_enum_from_global_categorical() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    s2 = s.cast(dtype)
    assert s2.dtype == dtype
    assert s2.null_count() == 1
    expected = pl.Series([None, "a", "b", "c"], dtype=dtype)
    assert_series_equal(s2, expected)


@StringCache()
def test_casting_to_an_enum_from_global_categorical_nonexistent() -> None:
    with pytest.raises(
        pl.OutOfBoundsError,
        match=("value c is not present in Enum"),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Categorical).cast(pl.Enum(["a", "b"]))


def test_casting_from_an_enum_to_local() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=dtype)
    s2 = s.cast(pl.Categorical)
    expected = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(s2, expected)


@StringCache()
def test_casting_from_an_enum_to_global() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=dtype)
    s2 = s.cast(pl.Categorical)
    expected = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(s2, expected)
