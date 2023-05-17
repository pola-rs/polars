from datetime import date, datetime
from typing import Iterator

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


def test_transpose_supertype() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "ham"]})
    result = df.transpose()
    expected = pl.DataFrame(
        {
            "column_0": ["1", "foo"],
            "column_1": ["2", "bar"],
            "column_2": ["3", "ham"],
        }
    )
    assert_frame_equal(result, expected)


def test_transpose_tz_naive_and_tz_aware() -> None:
    df = pl.DataFrame(
        {
            "a": [datetime(2020, 1, 1)],
            "b": [datetime(2020, 1, 1)],
        }
    )
    df = df.with_columns(pl.col("b").dt.replace_time_zone("Asia/Kathmandu"))
    with pytest.raises(
        ComputeError,
        match=r"failed to determine supertype of datetime\[μs\] and datetime\[μs, Asia/Kathmandu\]",
    ):
        df.transpose()


def test_transpose_struct() -> None:
    df = pl.DataFrame(
        {
            "a": ["foo", "bar", "ham"],
            "b": [
                {"a": date(2022, 1, 1), "b": True},
                {"a": date(2022, 1, 2), "b": False},
                {"a": date(2022, 1, 3), "b": False},
            ],
        }
    )
    result = df.transpose()
    expected = pl.DataFrame(
        {
            "column_0": ["foo", "{2022-01-01,true}"],
            "column_1": ["bar", "{2022-01-02,false}"],
            "column_2": ["ham", "{2022-01-03,false}"],
        }
    )
    assert_frame_equal(result, expected)

    df = pl.DataFrame(
        {
            "b": [
                {"a": date(2022, 1, 1), "b": True},
                {"a": date(2022, 1, 2), "b": False},
                {"a": date(2022, 1, 3), "b": False},
            ]
        }
    )
    result = df.transpose()
    expected = pl.DataFrame(
        {
            "column_0": [{"a": date(2022, 1, 1), "b": True}],
            "column_1": [{"a": date(2022, 1, 2), "b": False}],
            "column_2": [{"a": date(2022, 1, 3), "b": False}],
        }
    )
    assert_frame_equal(result, expected)


def test_transpose_arguments() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    expected = pl.DataFrame(
        {
            "column": ["a", "b"],
            "column_0": [1, 1],
            "column_1": [2, 2],
            "column_2": [3, 3],
        }
    )
    out = df.transpose(include_header=True)
    assert_frame_equal(expected, out)

    out = df.transpose(include_header=False, column_names=["a", "b", "c"])
    expected = pl.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert_frame_equal(expected, out)

    out = df.transpose(
        include_header=True, header_name="foo", column_names=["a", "b", "c"]
    )
    expected = pl.DataFrame(
        {
            "foo": ["a", "b"],
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert_frame_equal(expected, out)

    def name_generator() -> Iterator[str]:
        base_name = "my_column_"
        count = 0
        while True:
            yield f"{base_name}{count}"
            count += 1

    out = df.transpose(include_header=False, column_names=name_generator())
    expected = pl.DataFrame(
        {
            "my_column_0": [1, 1],
            "my_column_1": [2, 2],
            "my_column_2": [3, 3],
        }
    )
    assert_frame_equal(expected, out)


def test_transpose_logical_data() -> None:
    df = pl.DataFrame(
        {
            "a": [date(2022, 2, 1), date(2022, 2, 2), date(2022, 1, 3)],
            "b": [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
        }
    )
    result = df.transpose()
    expected = pl.DataFrame(
        {
            "column_0": [datetime(2022, 2, 1, 0, 0), datetime(2022, 1, 1, 0, 0)],
            "column_1": [datetime(2022, 2, 2, 0, 0), datetime(2022, 1, 2, 0, 0)],
            "column_2": [datetime(2022, 1, 3, 0, 0), datetime(2022, 1, 3, 0, 0)],
        }
    )
    assert_frame_equal(result, expected)
