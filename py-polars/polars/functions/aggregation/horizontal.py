from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable

import polars.functions as F
from polars.datatypes import UInt32
from polars.utils._parse_expr_input import parse_as_list_of_expressions
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import IntoExpr


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Evaluate a bitwise AND operation.

    Otherwise, this function computes the bitwise AND horizontally across multiple
    columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Examples
    --------
    Selecting all columns and calculating the sum:

    >>> df = pl.DataFrame(
    ...     {"a": [1, 2, 3], "b": ["hello", "foo", "bar"], "c": [1, 1, 1]}
    ... )
    >>> df.select(pl.all().sum())
    shape: (1, 3)
    ┌─────┬──────┬─────┐
    │ a   ┆ b    ┆ c   │
    │ --- ┆ ---  ┆ --- │
    │ i64 ┆ str  ┆ i64 │
    ╞═════╪══════╪═════╡
    │ 6   ┆ null ┆ 3   │
    └─────┴──────┴─────┘

    Bitwise AND across multiple columns:

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [True, False, True],
    ...         "b": [True, False, False],
    ...         "c": [False, True, False],
    ...     }
    ... )
    >>> df.select(pl.all("a", "b"))
    shape: (3, 1)
    ┌───────┐
    │ all   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    │ false │
    └───────┘

    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    return F.fold(
        F.lit(True), lambda a, b: a.cast(bool) & b.cast(bool), exprs_wrapped
    ).alias("all")


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Compute bitwise OR horizontally across multiple columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [True, False, True],
    ...         "b": [False, False, False],
    ...         "c": [False, True, False],
    ...     }
    ... )
    >>> df.with_columns(pl.any("a", "b", "c"))
    shape: (3, 4)
    ┌───────┬───────┬───────┬──────┐
    │ a     ┆ b     ┆ c     ┆ any  │
    │ ---   ┆ ---   ┆ ---   ┆ ---  │
    │ bool  ┆ bool  ┆ bool  ┆ bool │
    ╞═══════╪═══════╪═══════╪══════╡
    │ true  ┆ false ┆ false ┆ true │
    │ false ┆ false ┆ true  ┆ true │
    │ true  ┆ false ┆ false ┆ true │
    └───────┴───────┴───────┴──────┘

    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    return F.fold(
        F.lit(False), lambda a, b: a.cast(bool) | b.cast(bool), exprs_wrapped
    ).alias("any")


def max_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Get the maximum value horizontally across columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, 2],
    ...         "c": ["foo", "bar", "foo"],
    ...     }
    ... )
    >>> df.select(pl.max("a", "b"))
    shape: (3, 1)
    ┌─────┐
    │ max │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 4   │
    │ 8   │
    │ 3   │
    └─────┘

    """
    exprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.max_exprs(exprs))


def min_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Get the minimum value horizontally across columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, 2],
    ...         "c": ["foo", "bar", "foo"],
    ...     }
    ... )
    >>> df.select(pl.min("a", "b"))
    shape: (3, 1)
    ┌─────┐
    │ min │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 5   │
    │ 2   │
    └─────┘

    """
    exprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.min_exprs(exprs))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Sum all values horizontally across columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2],
    ...         "b": [3, 4],
    ...         "c": [5, 6],
    ...     }
    ... )
    >>> df.with_columns(pl.sum("a", "c"))
    shape: (2, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   ┆ sum │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   ┆ 6   │
    │ 2   ┆ 4   ┆ 6   ┆ 8   │
    └─────┴─────┴─────┴─────┘

    """
    exprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.sum_exprs(exprs))


def cumsum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Cumulatively sum all values horizontally across columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2],
    ...         "b": [3, 4],
    ...         "c": [5, 6],
    ...     }
    ... )
    >>> df.with_columns(pl.cumsum("a", "c"))
    shape: (2, 4)
    ┌─────┬─────┬─────┬───────────┐
    │ a   ┆ b   ┆ c   ┆ cumsum    │
    │ --- ┆ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64 ┆ struct[2] │
    ╞═════╪═════╪═════╪═══════════╡
    │ 1   ┆ 3   ┆ 5   ┆ {1,6}     │
    │ 2   ┆ 4   ┆ 6   ┆ {2,8}     │
    └─────┴─────┴─────┴───────────┘

    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    # (Expr): use u32 as that will not cast to float as eagerly
    return F.cumfold(F.lit(0).cast(UInt32), lambda a, b: a + b, exprs_wrapped).alias(
        "cumsum"
    )
