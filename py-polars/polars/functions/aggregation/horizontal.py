from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable

import polars.functions as F
from polars.datatypes import UInt32
from polars.utils._parse_expr_input import parse_as_list_of_expressions
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import deprecate_renamed_function

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import IntoExpr


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Compute the bitwise AND horizontally across columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Notes
    -----
    `Kleene logic`_ is used to deal with nulls: if the column contains any null values
    and no `False` values, the output is null.

    .. _Kleene logic: https://en.wikipedia.org/wiki/Three-valued_logic

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [False, False, True, True, False, None],
    ...         "b": [False, True, True, None, None, None],
    ...         "c": ["u", "v", "w", "x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(all=pl.all_horizontal("a", "b"))
    shape: (6, 4)
    ┌───────┬───────┬─────┬───────┐
    │ a     ┆ b     ┆ c   ┆ all   │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ bool  ┆ bool  ┆ str ┆ bool  │
    ╞═══════╪═══════╪═════╪═══════╡
    │ false ┆ false ┆ u   ┆ false │
    │ false ┆ true  ┆ v   ┆ false │
    │ true  ┆ true  ┆ w   ┆ true  │
    │ true  ┆ null  ┆ x   ┆ null  │
    │ false ┆ null  ┆ y   ┆ false │
    │ null  ┆ null  ┆ z   ┆ null  │
    └───────┴───────┴─────┴───────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.all_horizontal(pyexprs))


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Compute the bitwise OR horizontally across columns.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.

    Notes
    -----
    `Kleene logic`_ is used to deal with nulls: if the column contains any null values
    and no `True` values, the output is null.

    .. _Kleene logic: https://en.wikipedia.org/wiki/Three-valued_logic

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [False, False, True, True, False, None],
    ...         "b": [False, True, True, None, None, None],
    ...         "c": ["u", "v", "w", "x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(any=pl.any_horizontal("a", "b"))
    shape: (6, 4)
    ┌───────┬───────┬─────┬───────┐
    │ a     ┆ b     ┆ c   ┆ any   │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ bool  ┆ bool  ┆ str ┆ bool  │
    ╞═══════╪═══════╪═════╪═══════╡
    │ false ┆ false ┆ u   ┆ false │
    │ false ┆ true  ┆ v   ┆ true  │
    │ true  ┆ true  ┆ w   ┆ true  │
    │ true  ┆ null  ┆ x   ┆ true  │
    │ false ┆ null  ┆ y   ┆ null  │
    │ null  ┆ null  ┆ z   ┆ null  │
    └───────┴───────┴─────┴───────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.any_horizontal(pyexprs))


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
    ...         "b": [4, 5, None],
    ...         "c": ["x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(max=pl.max_horizontal("a", "b"))
    shape: (3, 4)
    ┌─────┬──────┬─────┬─────┐
    │ a   ┆ b    ┆ c   ┆ max │
    │ --- ┆ ---  ┆ --- ┆ --- │
    │ i64 ┆ i64  ┆ str ┆ i64 │
    ╞═════╪══════╪═════╪═════╡
    │ 1   ┆ 4    ┆ x   ┆ 4   │
    │ 8   ┆ 5    ┆ y   ┆ 8   │
    │ 3   ┆ null ┆ z   ┆ 3   │
    └─────┴──────┴─────┴─────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.max_horizontal(pyexprs))


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
    ...         "b": [4, 5, None],
    ...         "c": ["x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(min=pl.min_horizontal("a", "b"))
    shape: (3, 4)
    ┌─────┬──────┬─────┬─────┐
    │ a   ┆ b    ┆ c   ┆ min │
    │ --- ┆ ---  ┆ --- ┆ --- │
    │ i64 ┆ i64  ┆ str ┆ i64 │
    ╞═════╪══════╪═════╪═════╡
    │ 1   ┆ 4    ┆ x   ┆ 1   │
    │ 8   ┆ 5    ┆ y   ┆ 5   │
    │ 3   ┆ null ┆ z   ┆ 3   │
    └─────┴──────┴─────┴─────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.min_horizontal(pyexprs))


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
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, None],
    ...         "c": ["x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(sum=pl.sum_horizontal("a", "b"))
    shape: (3, 4)
    ┌─────┬──────┬─────┬─────┐
    │ a   ┆ b    ┆ c   ┆ sum │
    │ --- ┆ ---  ┆ --- ┆ --- │
    │ i64 ┆ i64  ┆ str ┆ i64 │
    ╞═════╪══════╪═════╪═════╡
    │ 1   ┆ 4    ┆ x   ┆ 5   │
    │ 8   ┆ 5    ┆ y   ┆ 13  │
    │ 3   ┆ null ┆ z   ┆ 3   │
    └─────┴──────┴─────┴─────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.sum_horizontal(pyexprs))


def mean_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Compute the mean of all values horizontally across columns.

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
    ...         "b": [4, 5, None],
    ...         "c": ["x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(mean=pl.mean_horizontal("a", "b"))
    shape: (3, 4)
    ┌─────┬──────┬─────┬──────┐
    │ a   ┆ b    ┆ c   ┆ mean │
    │ --- ┆ ---  ┆ --- ┆ ---  │
    │ i64 ┆ i64  ┆ str ┆ f64  │
    ╞═════╪══════╪═════╪══════╡
    │ 1   ┆ 4    ┆ x   ┆ 2.5  │
    │ 8   ┆ 5    ┆ y   ┆ 6.5  │
    │ 3   ┆ null ┆ z   ┆ 3.0  │
    └─────┴──────┴─────┴──────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    return wrap_expr(plr.mean_horizontal(pyexprs))


def cum_sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
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
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, None],
    ...         "c": ["x", "y", "z"],
    ...     }
    ... )
    >>> df.with_columns(pl.cum_sum_horizontal("a", "b"))
    shape: (3, 4)
    ┌─────┬──────┬─────┬───────────┐
    │ a   ┆ b    ┆ c   ┆ cum_sum   │
    │ --- ┆ ---  ┆ --- ┆ ---       │
    │ i64 ┆ i64  ┆ str ┆ struct[2] │
    ╞═════╪══════╪═════╪═══════════╡
    │ 1   ┆ 4    ┆ x   ┆ {1,5}     │
    │ 8   ┆ 5    ┆ y   ┆ {8,13}    │
    │ 3   ┆ null ┆ z   ┆ {3,null}  │
    └─────┴──────┴─────┴───────────┘
    """
    pyexprs = parse_as_list_of_expressions(*exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    # (Expr): use u32 as that will not cast to float as eagerly
    return F.cum_fold(F.lit(0).cast(UInt32), lambda a, b: a + b, exprs_wrapped).alias(
        "cum_sum"
    )


@deprecate_renamed_function("cum_sum_horizontal", version="0.19.14")
def cumsum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Cumulatively sum all values horizontally across columns.

    .. deprecated:: 0.19.14
        This function has been renamed to :func:`cum_sum_horizontal`.

    Parameters
    ----------
    *exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    """
    return cum_sum_horizontal(*exprs).alias("cumsum")
