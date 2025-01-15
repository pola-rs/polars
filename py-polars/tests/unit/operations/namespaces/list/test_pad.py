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


def test_list_pad_start_with_expr() -> None:
    df = pl.DataFrame(
        {"a": [[1], [], [1, 2, 3]], "int": [0, 999, 2], "float": [0.0, 999, 2]}
    )
    result = df.select(
        filled_int=pl.col("a").list.pad_start(fill_value=pl.col("int")),
        filled_float=pl.col("a").list.pad_start(fill_value=pl.col("float")),
    )
    expected = pl.DataFrame(
        {
            "filled_int": [[0, 0, 1], [999, 999, 999], [1, 2, 3]],
            "filled_float": [[0.0, 0.0, 1.0], [999.0, 999.0, 999.0], [1.0, 2.0, 3.0]],
        }
    )
    assert_frame_equal(result, expected)


def test_list_pad_start_with_lit() -> None:
    df = pl.DataFrame({"a": [[1], [], [1, 2, 3]]})
    result = df.select(pl.col("a").list.pad_start(fill_value=0))
    expected = pl.DataFrame({"a": [[0, 0, 1], [0, 0, 0], [1, 2, 3]]})
    assert_frame_equal(result, expected)


def test_list_pad_start_zero_length() -> None:
    df = pl.DataFrame({"a": [[], []]})
    result = df.select(pl.col("a").list.pad_start("a"))
    expected = pl.DataFrame({"a": [[], []]}, schema={"a": pl.List(pl.String)})
    assert_frame_equal(result, expected)


def test_list_pad_start_casts_to_supertype() -> None:
    df = pl.DataFrame({"a": [["a"], ["a", "b"]]})
    result = df.select(pl.col("a").list.pad_start(1))
    expected = pl.DataFrame({"a": [["1", "a"], ["a", "b"]]})
    assert_frame_equal(result, expected)

    with pytest.raises(SchemaError, match="failed to determine supertype"):
        pl.DataFrame({"a": [[]]}, schema={"a": pl.List(pl.Categorical)}).select(
            pl.col("a").list.pad_start(True)
        )


def test_list_pad_start_errors() -> None:
    df = pl.DataFrame({"a": [["a"], ["a", "b"]]})

    with pytest.raises(TypeError, match="fill_value"):
        df.select(pl.col("a").list.pad_start())
    with pytest.raises(InvalidOperationError, match="to String not supported"):
        df.select(pl.col("a").list.pad_start(timedelta(days=1)))


@pytest.mark.parametrize(
    ("fill_value", "type"),
    [
        (True, pl.Boolean),
        (timedelta(days=1), pl.Duration),
        (date(2022, 1, 1), pl.Date),
        (datetime(2022, 1, 1, 23), pl.Datetime),
    ],
)
def test_list_pad_start_unsupported_type(fill_value: Any, type: Any) -> None:
    df = pl.DataFrame({"a": [[], []]}, schema={"a": pl.List(type)})
    with pytest.raises(InvalidOperationError, match="doesn't work on data type"):
        df.select(pl.col("a").list.pad_start(fill_value))
