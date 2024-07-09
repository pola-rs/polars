from __future__ import annotations

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


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


@pytest.mark.parametrize("function", ["min", "max", "sum", "cum_sum"])
@pytest.mark.parametrize("input", ["a", "^a|b$"])
def test_alias_for_col_agg(function: str, input: str) -> None:
    result = getattr(pl, function)(input)  # e.g. pl.min(input)
    expected = getattr(pl.col(input), function)()  # e.g. pl.col(input).min()
    context = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    assert_expr_equal(result, expected, context)


@pytest.mark.release()
def test_mean_overflow() -> None:
    np.random.seed(1)
    expected = 769.5607652

    df = pl.DataFrame(np.random.randint(500, 1040, 5000000), schema=["value"])

    result = df.with_columns(pl.mean("value"))[0, 0]
    assert np.isclose(result, expected)

    result = df.with_columns(pl.col("value").cast(pl.Int32)).with_columns(
        pl.mean("value")
    )[0, 0]
    assert np.isclose(result, expected)

    result = df.with_columns(pl.col("value").cast(pl.Int32)).get_column("value").mean()
    assert np.isclose(result, expected)


def test_deep_subexpression_f32_schema_7129() -> None:
    df = pl.DataFrame({"a": [1.1, 2.3, 3.4, 4.5]}, schema={"a": pl.Float32()})
    assert df.with_columns(pl.col("a") - pl.col("a").median()).dtypes == [pl.Float32]
    assert df.with_columns(
        (pl.col("a") - pl.col("a").mean()) / (pl.col("a").std() + 0.001)
    ).dtypes == [pl.Float32]
