import pytest

import polars as pl
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

    with pytest.raises(pl.ShapeError):
        df1.vstack(df2)


def test_vstack_column_name_mismatch(df1: pl.DataFrame) -> None:
    df2 = df1.with_columns(pl.col("foo").alias("oof"))

    with pytest.raises(pl.ShapeError):
        df1.vstack(df2)
