from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterable

import polars._reexport as pl
from polars import functions as F
from polars._utils.deprecation import issue_deprecation_warning
from polars.exceptions import ComputeError

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

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

    # Treat elements of a single iterable as separate inputs
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
    for name, input in named_inputs.items():
        yield parse_as_expression(input, structify=structify).alias(name)


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


def parse_predicates_constraints_as_expression(
    *predicates: IntoExpr | Iterable[IntoExpr],
    **constraints: Any,
) -> PyExpr:
    """
    Parse predicates and constraints into a single expression.

    The result is an AND-reduction of all inputs.

    Parameters
    ----------
    *predicates
        Predicates to be parsed, specified as positional arguments.
    **constraints
        Constraints to be parsed, specified as keyword arguments.
        These will be converted to predicates of the form "keyword equals input value".

    Returns
    -------
    PyExpr
    """
    all_predicates = _parse_positional_inputs(predicates)  # type: ignore[arg-type]

    if constraints:
        constraint_predicates = _parse_constraints(constraints)
        all_predicates.extend(constraint_predicates)

    return _combine_predicates(all_predicates)


def _parse_constraints(constraints: dict[str, IntoExpr]) -> Iterable[PyExpr]:
    for name, value in constraints.items():
        yield F.col(name).eq(value)._pyexpr


def _combine_predicates(predicates: list[PyExpr]) -> PyExpr:
    if not predicates:
        msg = "at least one predicate or constraint must be provided"
        raise TypeError(msg)

    if len(predicates) == 1:
        return predicates[0]

    return plr.all_horizontal(predicates)


def parse_when_inputs(
    *predicates: IntoExpr | Iterable[IntoExpr],
    **constraints: Any,
) -> PyExpr:
    if "condition" in constraints:
        if isinstance(constraints["condition"], pl.Expr):
            issue_deprecation_warning(
                "`when` no longer takes a 'condition' parameter."
                " To silence this warning, omit the keyword and pass"
                " as a positional argument instead.",
                version="0.19.16",
            )
            predicates = (*predicates, constraints.pop("condition"))

    return parse_predicates_constraints_as_expression(*predicates, **constraints)
