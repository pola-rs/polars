from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_null_index() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4], [5, 6]], "b": [[1, 2], [1, 2], [4, 5]]})

    result = df.with_columns(pl.lit(None).alias("null_col"))[-1]

    expected = pl.DataFrame(
        {"a": [[5, 6]], "b": [[4, 5]], "null_col": [None]},
        schema_overrides={"null_col": pl.Null},
    )
    assert_frame_equal(result, expected)


def test_null_grouping_12950() -> None:
    assert pl.DataFrame({"x": None}).unique().to_dict(as_series=False) == {"x": [None]}
    assert pl.DataFrame({"x": [None, None]}).unique().to_dict(as_series=False) == {
        "x": [None]
    }
    assert pl.DataFrame({"x": None}).slice(0, 0).unique().to_dict(as_series=False) == {
        "x": []
    }


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (pl.Expr.gt, [None, None]),
        (pl.Expr.lt, [None, None]),
        (pl.Expr.ge, [None, None]),
        (pl.Expr.le, [None, None]),
        (pl.Expr.eq, [None, None]),
        (pl.Expr.eq_missing, [True, True]),
        (pl.Expr.ne, [None, None]),
        (pl.Expr.ne_missing, [False, False]),
    ],
)
def test_null_comp_14118(op: Any, expected: list[None | bool]) -> None:
    df = pl.DataFrame(
        {
            "a": [None, None],
            "b": [None, None],
        }
    )

    output_df = df.select(
        cmp=op(pl.col("a"), pl.col("b")),
        broadcast_lhs=op(pl.lit(None), pl.col("b")),
        broadcast_rhs=op(pl.col("a"), pl.lit(None)),
    )

    expected_df = pl.DataFrame(
        {
            "cmp": expected,
            "broadcast_lhs": expected,
            "broadcast_rhs": expected,
        },
        schema={
            "cmp": pl.Boolean,
            "broadcast_lhs": pl.Boolean,
            "broadcast_rhs": pl.Boolean,
        },
    )
    assert_frame_equal(output_df, expected_df)


def test_null_hash_rows_14100() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [None, None, None, None]})
    assert df.hash_rows().dtype == pl.UInt64
    assert df["b"].hash().dtype == pl.UInt64
    assert df.select([pl.col("b").hash().alias("foo")])["foo"].dtype == pl.UInt64


def test_null_lit_filter_16664() -> None:
    assert pl.DataFrame({"x": []}).filter(pl.lit(True)).shape == (0, 1)
