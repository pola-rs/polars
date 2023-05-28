from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Iterable

import polars._reexport as pl
from polars import functions as F
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from polars import Expr
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr


def selection_to_pyexpr_list(
    exprs: IntoExpr | Iterable[IntoExpr],
    structify: bool = False,
) -> list[PyExpr]:
    if exprs is None:
        return []

    if isinstance(
        exprs, (str, pl.Expr, pl.Series, F.whenthen.WhenThen, F.whenthen.WhenThenThen)
    ) or not isinstance(exprs, Iterable):
        return [
            parse_single_expression_input(
                exprs, str_as_lit=False, structify=structify
            )._pyexpr,
        ]

    return [
        parse_single_expression_input(e, str_as_lit=False, structify=structify)._pyexpr
        for e in exprs  # type: ignore[union-attr]
    ]


def parse_single_expression_input(
    input: IntoExpr,
    *,
    str_as_lit: bool = True,
    structify: bool = False,
) -> Expr:
    """
    Parse a single input into an expression.

    Parameters
    ----------
    input
        The input to be parsed as an expression.
    str_as_lit
        Interpret string input as a string literal. If set to ``False``, strings are
        parsed as column names.
    structify
        Convert multi-column expressions to a single struct expression.

    """
    if isinstance(input, pl.Expr):
        expr = input
    elif isinstance(input, str) and not str_as_lit:
        expr = F.col(input)
        structify = False
    elif (
        isinstance(input, (int, float, str, pl.Series, datetime, date, time, timedelta))
        or input is None
    ):
        expr = F.lit(input)
        structify = False
    elif isinstance(input, list):
        expr = F.lit(pl.Series("", [input]))
        structify = False
    elif isinstance(input, (F.whenthen.WhenThen, F.whenthen.WhenThenThen)):
        expr = input.otherwise(None)  # implicitly add the null branch.
    else:
        raise TypeError(
            f"did not expect value {input!r} of type {type(input)}, maybe disambiguate with"
            " pl.lit or pl.col"
        )

    if structify:
        expr = _structify_expression(expr)

    return expr


def _structify_expression(expr: Expr) -> Expr:
    unaliased_expr = expr.meta.undo_aliases()
    if unaliased_expr.meta.has_multiple_outputs():
        try:
            expr_name = expr.meta.output_name()
        except ComputeError:
            expr = F.struct(expr)
        else:
            expr = F.struct(unaliased_expr).alias(expr_name)
    return expr
