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
        pl.ComputeError,
        match=("value 'c' is not present in Enum"),
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


def test_casting_to_an_enum_from_integer() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    expected = pl.Series([None, "b", "a", "c"], dtype=dtype)
    s = pl.Series([None, 1, 0, 2], dtype=pl.UInt32)
    s_enum = s.cast(dtype)
    assert s_enum.dtype == dtype
    assert s_enum.null_count() == 1
    assert_series_equal(s_enum, expected)


def test_casting_to_an_enum_oob_from_integer() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, 1, 0, 5], dtype=pl.UInt32)
    with pytest.raises(
        pl.OutOfBoundsError, match=("index 5 is bigger than the number of categories 3")
    ):
        s.cast(dtype)


def test_casting_to_an_enum_from_categorical_nonexistent() -> None:
    with pytest.raises(
        pl.ComputeError,
        match=("value 'c' is not present in Enum"),
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
        pl.ComputeError,
        match=("value 'c' is not present in Enum"),
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


def test_append_to_an_enum() -> None:
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s2 = pl.Series(["c", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s.append(s2)
    assert s.len() == 8


def test_append_to_an_enum_with_new_category() -> None:
    with pytest.raises(
        pl.ComputeError,
        match=("enum is not compatible with other categorical / enum"),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"])).append(
            pl.Series(["d", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c", "d"]))
        )


def test_extend_to_an_enum() -> None:
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s2 = pl.Series(["c", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s.extend(s2)
    assert s.len() == 8
    assert s.null_count() == 1


def test_series_init_uninstantiated_enum() -> None:
    with pytest.raises(
        TypeError, match="Enum types must be instantiated with a list of categories"
    ):
        pl.Series(["a", "b", "a"], dtype=pl.Enum)
