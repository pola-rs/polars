from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Iterable

from polars import functions as F
from polars import internals as pli

if TYPE_CHECKING:
    from polars.expr import Expr
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr


def selection_to_pyexpr_list(
    exprs: IntoExpr | Iterable[IntoExpr],
    structify: bool = False,
) -> list[PyExpr]:
    if exprs is None:
        return []

    if isinstance(
        exprs, (str, pli.Expr, pli.Series, F.whenthen.WhenThen, F.whenthen.WhenThenThen)
    ) or not isinstance(exprs, Iterable):
        return [
            expr_to_lit_or_expr(exprs, str_to_lit=False, structify=structify)._pyexpr,
        ]

    return [
        expr_to_lit_or_expr(e, str_to_lit=False, structify=structify)._pyexpr
        for e in exprs  # type: ignore[union-attr]
    ]


def expr_to_lit_or_expr(
    expr: IntoExpr | Iterable[IntoExpr],
    str_to_lit: bool = True,
    structify: bool = False,
    name: str | None = None,
) -> Expr:
    """
    Convert args to expressions.

    Parameters
    ----------
    expr
        Any argument.
    str_to_lit
        If True string argument `"foo"` will be converted to `lit("foo")`.
        If False it will be converted to `col("foo")`.
    structify
        If the final unaliased expression has multiple output names,
        automatically convert it to struct.
    name
        Apply the given name as an alias to the resulting expression.

    Returns
    -------
    Expr

    """
    if isinstance(expr, pli.Expr):
        pass
    elif isinstance(expr, str) and not str_to_lit:
        expr = F.col(expr)
    elif (
        isinstance(expr, (int, float, str, pli.Series, datetime, date, time, timedelta))
        or expr is None
    ):
        expr = F.lit(expr)
        structify = False
    elif isinstance(expr, list):
        expr = F.lit(pli.Series("", [expr]))
        structify = False
    elif isinstance(expr, (F.whenthen.WhenThen, F.whenthen.WhenThenThen)):
        expr = expr.otherwise(None)  # implicitly add the null branch.
    else:
        raise TypeError(
            f"did not expect value {expr} of type {type(expr)}, maybe disambiguate with"
            " pl.lit or pl.col"
        )

    if structify:
        unaliased_expr = expr.meta.undo_aliases()
        if unaliased_expr.meta.has_multiple_outputs():
            expr_name = _expr_output_name(expr)
            expr = F.struct(expr if expr_name is None else unaliased_expr)
            name = name or expr_name

    return expr if name is None else expr.alias(name)


def _expr_output_name(expr: Expr) -> str | None:
    try:
        return expr.meta.output_name()
    except Exception:
        return None
