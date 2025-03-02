from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

import polars as pl
from polars.exceptions import (
    InvalidOperationError,
    SchemaError,
)
from polars.testing import assert_frame_equal


def test_list_pad_start_with_expr_fill() -> None:
    df = pl.DataFrame(
        {"a": [[1], [], [1, 2, 3]], "int": [0, 999, 2], "float": [0.0, 999, 2]}
    )
    result = df.select(
        filled_int=pl.col("a").list.pad_start(fill_value=pl.col("int"), length=3),
        filled_float=pl.col("a").list.pad_start(fill_value=pl.col("float"), length=1),
    )
    expected = pl.DataFrame(
        {
            "filled_int": [[0, 0, 1], [999, 999, 999], [1, 2, 3]],
            "filled_float": [[1.0], [999.0], [1.0, 2.0, 3.0]],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "fill_value", "expect"),
    [
        ([[1], [], [1, 2, 3]], 0, [[0, 0, 1], [0, 0, 0], [1, 2, 3]]),
        (
            [[1.0], [], [1.0, 2.0, 3.0]],
            0.0,
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        ),
        (
            [["a"], [], ["a", "b", "b"]],
            "foo",
            [["foo", "foo", "a"], ["foo", "foo", "foo"], ["a", "b", "b"]],
        ),
        (
            [[True], [], [False, False, True]],
            True,
            [[True, True, True], [True, True, True], [False, False, True]],
        ),
    ],
)
def test_list_pad_start_with_lit_fill(data: Any, fill_value: Any, expect: Any) -> None:
    df = pl.DataFrame({"a": data})
    result = df.select(pl.col("a").list.pad_start(fill_value, length=3))
    expected = pl.DataFrame({"a": expect})
    assert_frame_equal(result, expected)


def test_list_pad_start_no_slice() -> None:
    df = pl.DataFrame({"a": [[1], [2, 3, 4, 5]]})
    result = df.select(pl.col("a").list.pad_start(1, length=2))
    expected = pl.DataFrame(
        {"a": [[1, 1], [2, 3, 4, 5]]}, schema={"a": pl.List(pl.Int64)}
    )
    assert_frame_equal(result, expected)


def test_list_pad_start_with_expr_length() -> None:
    df = pl.DataFrame({"a": [[1], [], [1, 2, 3]], "length": [2, 2, 4]})
    result = df.select(
        length_expr=pl.col("a").list.pad_start(
            fill_value=999, length=pl.col("a").list.len().max()
        ),
        length_col=pl.col("a").list.pad_start(fill_value=999, length=pl.col("length")),
    )
    expected = pl.DataFrame(
        {
            "length_expr": [[999, 999, 1], [999, 999, 999], [1, 2, 3]],
            "length_col": [[999, 1], [999, 999], [999, 1, 2, 3]],
        }
    )
    assert_frame_equal(result, expected)


def test_list_pad_start_zero_length() -> None:
    df = pl.DataFrame({"a": [[1], [2, 3]]})
    result = df.select(pl.col("a").list.pad_start(1, length=0))
    expected = pl.DataFrame({"a": [[1], [2, 3]]}, schema={"a": pl.List(pl.Int64)})
    assert_frame_equal(result, expected)


def test_list_pad_start_casts_to_supertype() -> None:
    df = pl.DataFrame({"a": [["a"], ["a", "b"]]})
    result = df.select(pl.col("a").list.pad_start(1, length=2))
    expected = pl.DataFrame({"a": [["1", "a"], ["a", "b"]]})
    assert_frame_equal(result, expected)

    with pytest.raises(SchemaError, match="failed to determine supertype"):
        pl.DataFrame({"a": [[]]}, schema={"a": pl.List(pl.Categorical)}).select(
            pl.col("a").list.pad_start(True, length=2)
        )


def test_list_pad_start_errors() -> None:
    df = pl.DataFrame({"a": [["a"], ["a", "b"]]})

    with pytest.raises(TypeError, match="fill_value"):
        df.select(pl.col("a").list.pad_start(length=2))  # type: ignore[call-arg]
    with pytest.raises(InvalidOperationError, match="to String not supported"):
        df.select(pl.col("a").list.pad_start(timedelta(days=1), length=2))
    with pytest.raises(
        InvalidOperationError, match="conversion from `i32` to `u64` failed"
    ):
        df.select(pl.col("a").list.pad_start("foo", length=-1))


@pytest.mark.parametrize(
    ("fill_value", "type"),
    [
        (timedelta(days=1), pl.Duration),
        (date(2022, 1, 1), pl.Date),
        (datetime(2022, 1, 1, 23), pl.Datetime),
    ],
)
def test_list_pad_start_unsupported_type(fill_value: Any, type: Any) -> None:
    df = pl.DataFrame({"a": [[], []]}, schema={"a": pl.List(type)})
    with pytest.raises(InvalidOperationError, match="doesn't work on data type"):
        df.select(pl.col("a").list.pad_start(fill_value, length=2))
