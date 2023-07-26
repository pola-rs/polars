from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars.functions as F
from polars.expr.expr import Expr
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr


class When:
    """Utility class. See the `when` function."""

    def __init__(self, when: Any):
        self._when = when

    def then(self, expr: IntoExpr) -> Then:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        return Then(self._when.then(expr))


class Then(Expr):
    """Utility class. See the `when` function."""

    def __init__(self, then: Any):
        self._then = then

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:  # type: ignore[override]
        return wrap_expr(pyexpr)

    @property
    def _pyexpr(self) -> PyExpr:
        return self._then.otherwise(F.lit(None)._pyexpr)

    def when(self, predicate: IntoExpr) -> ChainedWhen:
        """Start another "when, then, otherwise" layer."""
        predicate = parse_as_expression(predicate)
        return ChainedWhen(self._then.when(predicate))

    def otherwise(self, expr: IntoExpr) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        return wrap_expr(self._then.otherwise(expr))


class ChainedWhen(Expr):
    """Utility class. See the `when` function."""

    def __init__(self, chained_when: Any):
        self._chained_when = chained_when

    def then(self, predicate: IntoExpr) -> ChainedThen:
        """Start another "when, then, otherwise" layer."""
        predicate = parse_as_expression(predicate, str_as_lit=True)
        return ChainedThen(self._chained_when.then(predicate))


class ChainedThen(Expr):
    """Utility class. See the `when` function."""

    def __init__(self, chained_then: Any):
        self._chained_then = chained_then

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:  # type: ignore[override]
        return wrap_expr(pyexpr)

    @property
    def _pyexpr(self) -> PyExpr:
        return self._chained_then.otherwise(F.lit(None)._pyexpr)

    def when(self, predicate: IntoExpr) -> ChainedWhen:
        """Start another "when, then, otherwise" layer."""
        predicate = parse_as_expression(predicate)
        return ChainedWhen(self._chained_then.when(predicate))

    def otherwise(self, expr: IntoExpr) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        return wrap_expr(self._chained_then.otherwise(expr))
