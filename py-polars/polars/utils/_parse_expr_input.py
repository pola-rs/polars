from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import polars._reexport as pl
from polars import functions as F
from polars.exceptions import ComputeError
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import issue_deprecation_warning

if TYPE_CHECKING:
    from polars import Expr
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr, PolarsDataType


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
    exprs = _parse_positional_inputs(inputs, structify=__structify)  # type: ignore[arg-type]
    if named_inputs:
        named_exprs = _parse_named_inputs(named_inputs, structify=__structify)
        exprs.extend(named_exprs)

    return exprs


def _parse_positional_inputs(
    inputs: tuple[IntoExpr, ...] | tuple[Iterable[IntoExpr]],
    *,
    structify: bool = False,
) -> list[PyExpr]:
    inputs_iter = _parse_inputs_as_iterable(inputs)
    return [parse_as_expression(e, structify=structify) for e in inputs_iter]


def _parse_inputs_as_iterable(
    inputs: tuple[Any, ...] | tuple[Iterable[Any]],
) -> Iterable[Any]:
    if not inputs:
        return []

    # Treat input of a single iterable as separate inputs
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        return inputs[0]

    return inputs


def _is_iterable(input: Any | Iterable[Any]) -> bool:
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
    list_as_lit: bool = True,
    structify: bool = False,
    dtype: PolarsDataType | None = None,
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
    list_as_lit
        Interpret list input as a lit literal, If set to `False`,
        lists are parsed as `Series` literals.
    structify
        Convert multi-column expressions to a single struct expression.
    dtype
        If the input is expected to resolve to a literal with a known dtype, pass
        this to the `lit` constructor.

    Returns
    -------
    PyExpr
    """
    if isinstance(input, pl.Expr):
        expr = input
    elif isinstance(input, str) and not str_as_lit:
        expr = F.col(input)
        structify = False
    elif isinstance(input, list) and not list_as_lit:
        expr = F.lit(pl.Series(input), dtype=dtype)
        structify = False
    else:
        expr = F.lit(input, dtype=dtype)
        structify = False

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


def parse_when_constraint_expressions(
    *predicates: IntoExpr | Iterable[IntoExpr],
    **constraints: Any,
) -> PyExpr:
    all_predicates: list[Expr] = [
        wrap_expr(e)
        for e in _parse_positional_inputs(predicates)  # type: ignore[arg-type]
    ]

    if "condition" in constraints:
        if isinstance(constraints["condition"], pl.Expr):
            issue_deprecation_warning(
                "`when` no longer takes a 'condition' parameter.\n"
                "To silence this warning you should omit the keyword and pass "
                "as a positional argument instead.",
                version="0.19.16",
            )
            all_predicates.append(constraints.pop("condition"))

    if constraints:
        all_predicates.extend(
            F.col(name).eq(value) for name, value in constraints.items()
        )

    if not all_predicates:
        raise TypeError("no predicates or constraints provided to `when`")

    if len(all_predicates) == 1:
        return all_predicates[0]._pyexpr

    return F.all_horizontal(all_predicates)._pyexpr
