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


def parse_as_list_of_expressions(
    inputs: IntoExpr | Iterable[IntoExpr] | None = None,
    *more_inputs: IntoExpr,
    structify: bool = False,
    **named_inputs: IntoExpr,
) -> list[PyExpr]:
    """
    Parse multiple inputs into a list of expressions.

    Parameters
    ----------
    inputs
        Inputs to be parsed as expressions.
    *more_inputs
        Additional inputs to be parsed as expressions, specified as positional
        arguments.
    **named_inputs
        Additional inputs to be parsed as expressions, specified as keyword arguments.
        The expressions will be renamed to the keyword used.
    structify
        Convert multi-column expressions to a single struct expression.

    """
    exprs = _parse_regular_inputs(inputs, more_inputs, structify=structify)

    if named_inputs:
        named_exprs = _parse_named_inputs(named_inputs, structify=structify)
        exprs.extend(named_exprs)

    return exprs


def _parse_regular_inputs(
    inputs: IntoExpr | Iterable[IntoExpr] | None,
    more_inputs: tuple[IntoExpr, ...],
    structify: bool = False,
) -> list[PyExpr]:
    inputs = _inputs_to_list(inputs)
    if more_inputs:
        inputs.extend(more_inputs)
    return [parse_as_expression(e, structify=structify)._pyexpr for e in inputs]


def _inputs_to_list(inputs: IntoExpr | Iterable[IntoExpr] | None) -> list[IntoExpr]:
    if inputs is None:
        return []
    elif not isinstance(inputs, Iterable) or isinstance(inputs, (str, pl.Series)):
        return [inputs]
    else:
        return list(inputs)


def _parse_named_inputs(
    named_inputs: dict[str, IntoExpr], structify: bool = False
) -> Iterable[PyExpr]:
    return (
        parse_as_expression(input, structify=structify).alias(name)._pyexpr
        for name, input in named_inputs.items()
    )


def parse_as_expression(
    input: IntoExpr, *, str_as_lit: bool = False, structify: bool = False
) -> Expr:
    """
    Parse a single input into an expression.

    Parameters
    ----------
    input
        The input to be parsed as an expression.
    str_as_lit
        Interpret string input as a string literal. If set to ``False`` (default),
        strings are parsed as column names.
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
