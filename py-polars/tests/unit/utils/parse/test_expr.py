from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import polars as pl
from polars._utils.parse.expr import parse_into_expression
from polars._utils.wrap import wrap_expr
from polars.testing import assert_frame_equal


def assert_expr_equal(result: pl.Expr, expected: pl.Expr) -> None:
    """
    Evaluate the given expressions in a simple context to assert equality.

    WARNING: This is not a fully featured function - it's just to evaluate the tests in
    this module. Do not use it elsewhere.
    """
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_frame_equal(df.select(result), df.select(expected))


@pytest.mark.parametrize(
    "input", [5, 2.0, pl.Series([1, 2, 3]), date(2022, 1, 1), b"hi"]
)
def test_parse_into_expression_lit(input: Any) -> None:
    result = wrap_expr(parse_into_expression(input))
    expected = pl.lit(input)
    assert_expr_equal(result, expected)


def test_parse_into_expression_col() -> None:
    result = wrap_expr(parse_into_expression("a"))
    expected = pl.col("a")
    assert_expr_equal(result, expected)


@pytest.mark.parametrize("input", [pl.lit(4), pl.col("a")])
def test_parse_into_expression_expr(input: pl.Expr) -> None:
    result = wrap_expr(parse_into_expression(input))
    expected = input
    assert_expr_equal(result, expected)


@pytest.mark.parametrize(
    "input", [pl.when(True).then(1), pl.when(True).then(1).when(False).then(0)]
)
def test_parse_into_expression_whenthen(input: Any) -> None:
    result = wrap_expr(parse_into_expression(input))
    expected = input.otherwise(None)
    assert_expr_equal(result, expected)


@pytest.mark.parametrize("input", [[1, 2, 3], (1, 2)])
def test_parse_into_expression_list(input: Any) -> None:
    result = wrap_expr(parse_into_expression(input))
    expected = pl.lit(pl.Series("literal", [input]))
    assert_expr_equal(result, expected)


def test_parse_into_expression_str_as_lit() -> None:
    result = wrap_expr(parse_into_expression("a", str_as_lit=True))
    expected = pl.lit("a")
    assert_expr_equal(result, expected)


def test_parse_into_expression_structify() -> None:
    result = wrap_expr(parse_into_expression(pl.col("a", "b"), structify=True))
    expected = pl.struct("a", "b")
    assert_expr_equal(result, expected)


def test_parse_into_expression_structify_multiple_outputs() -> None:
    # note: this only works because assert_expr_equal evaluates on a dataframe with
    # columns "a" and "b"
    result = wrap_expr(parse_into_expression(pl.col("*"), structify=True))
    expected = pl.struct("a", "b")
    assert_expr_equal(result, expected)
