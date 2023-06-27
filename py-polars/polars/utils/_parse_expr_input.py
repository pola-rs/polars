from __future__ import annotations

import warnings
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Iterable

import polars._reexport as pl
from polars import functions as F
from polars.exceptions import ComputeError
from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    from polars import Expr
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr


def parse_as_list_of_expressions(
    *inputs: IntoExpr | Iterable[IntoExpr],
    __structify: bool = False,
    **named_inputs: IntoExpr,
) -> list[PyExpr]:
    """
    Parse multiple inputs into a list of expressions.

    Parameters
    ----------
    *inputs
        Inputs to be parsed as expressions, specified as positional arguments.
    **named_inputs
        Additional inputs to be parsed as expressions, specified as keyword arguments.
        The expressions will be renamed to the keyword used.
    __structify
        Convert multi-column expressions to a single struct expression.

    """
    exprs = _parse_regular_inputs(inputs, structify=__structify)
    if named_inputs:
        named_exprs = _parse_named_inputs(named_inputs, structify=__structify)
        exprs.extend(named_exprs)

    return exprs


def _parse_regular_inputs(
    inputs: tuple[IntoExpr | Iterable[IntoExpr], ...],
    *,
    structify: bool = False,
) -> list[PyExpr]:
    if not inputs:
        return []

    if (
        len(inputs) > 1
        and isinstance(inputs[0], Iterable)
        and not isinstance(inputs[0], (str, pl.Series))
    ):
        warnings.warn(
            "In the next breaking release, combining list input and positional input will result in an error."
            " To silence this warning, either unpack the list , or append the positional inputs to the list first."
            " The resulting behavior will be identical",
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    input_list = _first_input_to_list(inputs[0])
    input_list.extend(inputs[1:])  # type: ignore[arg-type]
    return [parse_as_expression(e, structify=structify) for e in input_list]


def _first_input_to_list(
    inputs: IntoExpr | Iterable[IntoExpr],
) -> list[IntoExpr]:
    if inputs is None:
        warnings.warn(
            "In the next breaking release, passing `None` as the first expression input will evaluate to `lit(None)`,"
            " rather than be ignored."
            " To silence this warning, either pass no arguments or an empty list to retain the current behavior,"
            " or pass `lit(None)` to opt into the new behavior.",
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )
        return []
    elif not isinstance(inputs, Iterable) or isinstance(inputs, (str, pl.Series)):
        return [inputs]
    else:
        return list(inputs)


def _parse_named_inputs(
    named_inputs: dict[str, IntoExpr], *, structify: bool = False
) -> Iterable[PyExpr]:
    return (
        parse_as_expression(input, structify=structify).alias(name)
        for name, input in named_inputs.items()
    )


def parse_as_expression(
    input: IntoExpr,
    *,
    str_as_lit: bool = False,
    structify: bool = False,
) -> PyExpr | Expr:
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
    wrap
        Return an ``Expr`` object rather than a ``PyExpr`` object.

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

    return expr._pyexpr


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
