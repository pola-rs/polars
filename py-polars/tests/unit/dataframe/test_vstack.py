import pytest

import polars as pl
from polars.exceptions import SchemaError, ShapeError
from polars.testing import assert_frame_equal


@pytest.fixture()
def df1() -> pl.DataFrame:
    return pl.DataFrame({"foo": [1, 2], "bar": [6, 7], "ham": ["a", "b"]})


@pytest.fixture()
def df2() -> pl.DataFrame:
    return pl.DataFrame({"foo": [3, 4], "bar": [8, 9], "ham": ["c", "d"]})


def test_vstack(df1: pl.DataFrame, df2: pl.DataFrame) -> None:
    result = df1.vstack(df2)
    expected = pl.DataFrame(
        {"foo": [1, 2, 3, 4], "bar": [6, 7, 8, 9], "ham": ["a", "b", "c", "d"]}
    )
    assert_frame_equal(result, expected)


def test_vstack_in_place(df1: pl.DataFrame, df2: pl.DataFrame) -> None:
    df1.vstack(df2, in_place=True)
    expected = pl.DataFrame(
        {"foo": [1, 2, 3, 4], "bar": [6, 7, 8, 9], "ham": ["a", "b", "c", "d"]}
    )
    assert_frame_equal(df1, expected)


def test_vstack_self(df1: pl.DataFrame) -> None:
    result = df1.vstack(df1)
    expected = pl.DataFrame(
        {"foo": [1, 2, 1, 2], "bar": [6, 7, 6, 7], "ham": ["a", "b", "a", "b"]}
    )
    assert_frame_equal(result, expected)


def test_vstack_self_in_place(df1: pl.DataFrame) -> None:
    df1.vstack(df1, in_place=True)
    expected = pl.DataFrame(
        {"foo": [1, 2, 1, 2], "bar": [6, 7, 6, 7], "ham": ["a", "b", "a", "b"]}
    )
    assert_frame_equal(df1, expected)


def test_vstack_column_number_mismatch(df1: pl.DataFrame) -> None:
    df2 = df1.drop("ham")

    with pytest.raises(ShapeError):
        df1.vstack(df2)


def test_vstack_column_name_mismatch(df1: pl.DataFrame) -> None:
    df2 = df1.with_columns(pl.col("foo").alias("oof"))

    with pytest.raises(ShapeError):
        df1.vstack(df2)


def test_vstack_with_null_column() -> None:
    df1 = pl.DataFrame({"x": [3.5]}, schema={"x": pl.Float64})
    df2 = pl.DataFrame({"x": [None]}, schema={"x": pl.Null})

    result = df1.vstack(df2)
    expected = pl.DataFrame({"x": [3.5, None]}, schema={"x": pl.Float64})

    assert_frame_equal(result, expected)

    with pytest.raises(SchemaError):
        df2.vstack(df1)


def test_vstack_with_nested_nulls() -> None:
    a = pl.DataFrame({"x": [[3.5]]}, schema={"x": pl.List(pl.Float32)})
    b = pl.DataFrame({"x": [[None]]}, schema={"x": pl.List(pl.Null)})

    out = a.vstack(b)
    expected = pl.DataFrame({"x": [[3.5], [None]]}, schema={"x": pl.List(pl.Float32)})
    assert_frame_equal(out, expected)
