from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def assert_expr_equal(
    left: pl.Expr,
    right: pl.Expr,
    context: pl.DataFrame | pl.LazyFrame | None = None,
) -> None:
    """
    Evaluate expressions in a context to determine equality.

    Parameters
    ----------
    left
        The expression to compare.
    right
        The other expression the compare.
    context
        The context in which the expressions will be evaluated. Defaults to an empty
        context.

    """
    if context is None:
        context = pl.DataFrame()
    assert_frame_equal(context.select(left), context.select(right))


def test_all_expr() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3, 4, 5, None]})
    assert_frame_equal(df.select(pl.all()), df)


def test_any_expr(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.with_columns(pl.col("A").cast(bool)).select(pl.any("A")).item()


@pytest.mark.parametrize("function", ["all", "any"])
@pytest.mark.parametrize("input", ["a", "^a|b$"])
def test_alias_for_col_agg_bool(function: str, input: str) -> None:
    result = getattr(pl, function)(input)  # e.g. pl.all(input)
    expected = getattr(pl.col(input), function)()  # e.g. pl.col(input).all()
    context = pl.DataFrame({"a": [True, False], "b": [True, True]})
    assert_expr_equal(result, expected, context)


@pytest.mark.parametrize("function", ["min", "max", "sum", "cumsum"])
@pytest.mark.parametrize("input", ["a", "^a|b$"])
def test_alias_for_col_agg(function: str, input: str) -> None:
    result = getattr(pl, function)(input)  # e.g. pl.min(input)
    expected = getattr(pl.col(input), function)()  # e.g. pl.col(input).min()
    context = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    assert_expr_equal(result, expected, context)


@pytest.mark.parametrize("function", ["all", "any"])
def test_deprecated_alias_for_series_agg_bool(function: str) -> None:
    s = pl.Series([True, True, False])
    with pytest.deprecated_call():
        result = getattr(pl, function)(s)  # e.g. pl.all(s)
    expected = getattr(s, function)()  # e.g. s.all()
    assert result == expected


@pytest.mark.parametrize("function", ["min", "max", "sum"])
def test_deprecated_alias_for_series_agg_numeric(function: str) -> None:
    s = pl.Series([1, 2, 3])
    with pytest.deprecated_call():
        result = getattr(pl, function)(s)  # e.g. pl.max(s)
    expected = getattr(s, function)()  # e.g. s.max()
    assert result == expected


def test_deprecated_alias_for_series_agg_cumsum() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.deprecated_call():
        result = pl.cumsum(s)
    expected = s.cumsum()
    assert_series_equal(result, expected)


@pytest.mark.parametrize("function", ["all", "any", "min", "max", "sum", "cumsum"])
def test_deprecated_horizontal(function: str) -> None:
    with pytest.deprecated_call():
        getattr(pl, function)(pl.col("a"))  # e.g. pl.all(pl.col("a"))
