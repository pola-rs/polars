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

    def __init__(self, pywhen: Any):
        self._pywhen = pywhen

    def then(self, expr: IntoExpr) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        pywhenthen = self._pywhen.then(expr)
        return WhenThen(pywhenthen)


class WhenThen(Expr):
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:  # type: ignore[override]
        return wrap_expr(pyexpr)

    @property
    def _pyexpr(self) -> PyExpr:
        return self._pywhenthen.otherwise(F.lit(None)._pyexpr)

    def when(self, predicate: IntoExpr) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = parse_as_expression(predicate)
        return WhenThenThen(self._pywhenthen.when(predicate))

    def otherwise(self, expr: IntoExpr) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        return wrap_expr(self._pywhenthen.otherwise(expr))


class WhenThenThen(Expr):
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthenthen: Any):
        self._pywhenthenthen = pywhenthenthen

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:  # type: ignore[override]
        return wrap_expr(pyexpr)

    @property
    def _pyexpr(self) -> PyExpr:
        return self._pywhenthenthen.otherwise(F.lit(None)._pyexpr)

    # @_pyexpr.setter
    # def _pyexpr(self) -> PyExpr:

    def when(self, predicate: IntoExpr) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = parse_as_expression(predicate)
        return WhenThenThen(self._pywhenthenthen.when(predicate))

    def then(self, expr: IntoExpr) -> WhenThenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        return WhenThenThen(self._pywhenthenthen.then(expr))

    def otherwise(self, expr: IntoExpr) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = parse_as_expression(expr, str_as_lit=True)
        return wrap_expr(self._pywhenthenthen.otherwise(expr))
