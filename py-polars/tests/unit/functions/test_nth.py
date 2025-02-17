from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import DuplicateError
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("expr", "expected_cols"),
    [
        (pl.nth(0), "a"),
        (pl.nth(-1), "c"),
        (pl.nth(2, 1), ["c", "b"]),
        (pl.nth([2, -2, 0]), ["c", "b", "a"]),
    ],
)
def test_nth(expr: pl.Expr, expected_cols: list[str]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df.select(expr)
    expected = df.select(expected_cols)
    assert_frame_equal(result, expected)


def test_nth_duplicate() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(DuplicateError, match="a"):
        df.select(pl.nth(0, 0))
