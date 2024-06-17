from datetime import datetime

import pytest

import polars as pl
from polars.exceptions import ShapeError
from polars.testing import assert_frame_equal


def test_extend_various_dtypes() -> None:
    with pl.StringCache():
        df1 = pl.DataFrame(
            {
                "foo": [1, 2],
                "bar": [True, False],
                "ham": ["a", "b"],
                "cat": ["A", "B"],
                "dates": [datetime(2021, 1, 1), datetime(2021, 2, 1)],
            },
            schema_overrides={"cat": pl.Categorical},
        )
        df2 = pl.DataFrame(
            {
                "foo": [3, 4],
                "bar": [True, None],
                "ham": ["c", "d"],
                "cat": ["C", "B"],
                "dates": [datetime(2022, 9, 1), datetime(2021, 2, 1)],
            },
            schema_overrides={"cat": pl.Categorical},
        )

        df1.extend(df2)

        expected = pl.DataFrame(
            {
                "foo": [1, 2, 3, 4],
                "bar": [True, False, True, None],
                "ham": ["a", "b", "c", "d"],
                "cat": ["A", "B", "C", "B"],
                "dates": [
                    datetime(2021, 1, 1),
                    datetime(2021, 2, 1),
                    datetime(2022, 9, 1),
                    datetime(2021, 2, 1),
                ],
            },
            schema_overrides={"cat": pl.Categorical},
        )
        assert_frame_equal(df1, expected)


def test_extend_slice_offset_8745() -> None:
    df = pl.DataFrame([{"age": 1}, {"age": 2}, {"age": 3}])

    df = df[:-1]
    tail = pl.DataFrame([{"age": 8}])
    result = df.extend(tail)

    expected = pl.DataFrame({"age": [1, 2, 8]})
    assert_frame_equal(result, expected)


def test_extend_self() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [True, False]})

    df.extend(df)

    expected = pl.DataFrame({"a": [1, 2, 1, 2], "b": [True, False, True, False]})
    assert_frame_equal(df, expected)


def test_extend_column_number_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [True, False]})
    df2 = df1.drop("a")

    with pytest.raises(ShapeError):
        df1.extend(df2)


def test_extend_column_name_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [True, False]})
    df2 = df1.with_columns(pl.col("a").alias("c"))

    with pytest.raises(ShapeError):
        df1.extend(df2)
