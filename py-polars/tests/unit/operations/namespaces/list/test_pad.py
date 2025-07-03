from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_list_pad_start_with_expr_fill() -> None:
    df = pl.DataFrame({"a": [[1], [], [1, 2, 3]], "int": [0, 999, 2]})
    result = df.select(
        filled_int=pl.col("a").list.pad_start(3, pl.col("int")),
    )
    expected = pl.DataFrame(
        {
            "filled_int": [[0, 0, 1], [999, 999, 999], [1, 2, 3]],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "fill_value", "expect"),
    [
        ([[1], [], [1, 2, 3]], 0, [[0, 0, 1], [0, 0, 0], [1, 2, 3]]),
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
    result = df.select(pl.col("a").list.pad_start(3, fill_value))
    expected = pl.DataFrame({"a": expect})
    assert_frame_equal(result, expected)


def test_list_pad_start_no_slice() -> None:
    df = pl.DataFrame({"a": [[1], [2, 3, 4, 5]]})
    result = df.select(pl.col("a").list.pad_start(2, 1))
    expected = pl.DataFrame(
        {"a": [[1, 1], [2, 3, 4, 5]]}, schema={"a": pl.List(pl.Int64)}
    )
    assert_frame_equal(result, expected)


def test_list_pad_start_with_expr_length() -> None:
    df = pl.DataFrame(
        {"a": [[1], [], [1, 2, 3]], "length": [2, 2, 4], "fill": [0, 1, 2]}
    )
    result = (
        df.lazy()
        .select(
            length_expr=pl.col("a").list.pad_start(pl.col("a").list.len().max(), 999),
            length_col=pl.col("a").list.pad_start(pl.col("length"), pl.col("fill")),
        )
        .collect()
    )
    expected = pl.DataFrame(
        {
            "length_expr": [[999, 999, 1], [999, 999, 999], [1, 2, 3]],
            "length_col": [[0, 1], [1, 1], [2, 1, 2, 3]],
        }
    )
    assert_frame_equal(result, expected)


def test_list_pad_start_zero_length() -> None:
    df = pl.DataFrame({"a": [[1], [2, 3]]})
    result = df.select(pl.col("a").list.pad_start(0, 1))
    expected = pl.DataFrame({"a": [[1], [2, 3]]}, schema={"a": pl.List(pl.Int64)})
    assert_frame_equal(result, expected)
