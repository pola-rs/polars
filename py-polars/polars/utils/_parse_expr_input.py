from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Iterable

import polars._reexport as pl
from polars import functions as F
from polars.exceptions import ComputeError
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import issue_deprecation_warning

if TYPE_CHECKING:
    from polars import Expr
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr

from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np


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

    Returns
    -------
    list of PyExpr
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

    inputs_iter: Iterable[IntoExpr]
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        inputs_iter = inputs[0]  # type: ignore[assignment]
    else:
        inputs_iter = inputs  # type: ignore[assignment]

    return [parse_as_expression(e, structify=structify) for e in inputs_iter]


def _is_iterable(input: IntoExpr | Iterable[IntoExpr]) -> bool:
    return isinstance(input, Iterable) and not isinstance(
        input, (str, bytes, pl.Series)
    )


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
) -> PyExpr:
    """
    Parse a single input into an expression.

    Parameters
    ----------
    input
        The input to be parsed as an expression.
    str_as_lit
        Interpret string input as a string literal. If set to `False` (default),
        strings are parsed as column names.
    structify
        Convert multi-column expressions to a single struct expression.

    Returns
    -------
    PyExpr
    """
    if isinstance(input, pl.Expr):
        expr = input
    elif isinstance(input, str) and not str_as_lit:
        expr = F.col(input)
        structify = False
    elif (
        isinstance(
            input, (int, float, str, bytes, pl.Series, datetime, date, time, timedelta)
        )
        or input is None
    ):
        expr = F.lit(input)
        structify = False
    elif isinstance(input, (list, tuple)):
        expr = F.lit(pl.Series("literal", [input]))
        structify = False
    elif _check_for_numpy(input) and isinstance(input, np.ndarray):
        expr = F.lit(pl.Series("literal", input))
        structify = False
    else:
        raise TypeError(
            f"did not expect value {input!r} of type {type(input).__name__!r}"
            "\n\nTry disambiguating with `lit` or `col`."
        )

    if structify:
        expr = _structify_expression(expr)

    return expr._pyexpr


def parse_when_constraint_expressions(
    *predicates: IntoExpr | Iterable[IntoExpr],
    **constraints: Any,
) -> PyExpr:
    all_predicates: list[pl.Expr] = []
    for p in predicates:
        all_predicates.extend(wrap_expr(x) for x in parse_as_list_of_expressions(p))

    if "condition" in constraints:
        if isinstance(constraints["condition"], pl.Expr):
            all_predicates.append(constraints.pop("condition"))
            issue_deprecation_warning(
                "`when` no longer takes a 'condition' parameter.\n"
                "To silence this warning you should omit the keyword and pass "
                "as a positional argument instead.",
                version="0.19.16",
            )

    all_predicates.extend(F.col(name).eq(value) for name, value in constraints.items())
    if not all_predicates:
        raise ValueError("No predicates or constraints provided to `when`.")

    return (
        F.all_horizontal(*all_predicates)
        if len(all_predicates) > 1
        else all_predicates[0]
    )._pyexpr


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
