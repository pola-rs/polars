from __future__ import annotations

import contextlib
import typing
from typing import TYPE_CHECKING, Any, Iterable

from polars.utils._parse_expr_input import expr_to_lit_or_expr
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import when as pywhen

if TYPE_CHECKING:
    from polars.expr.expr import Expr
    from polars.series import Series
    from polars.type_aliases import PolarsExprType, PythonLiteral


class WhenThenThen:
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthenthen: Any):
        self.pywhenthenthen = pywhenthenthen

    def when(self, predicate: Expr | bool) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = expr_to_lit_or_expr(predicate)
        return WhenThenThen(self.pywhenthenthen.when(predicate._pyexpr))

    def then(
        self,
        expr: (PolarsExprType | PythonLiteral | Series | None),
    ) -> WhenThenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        otherwise : Values to return in case of the predicate being `False`.

        """
        expr_ = expr_to_lit_or_expr(expr)
        return WhenThenThen(self.pywhenthenthen.then(expr_._pyexpr))

    def otherwise(
        self,
        expr: (PolarsExprType | PythonLiteral | Series | None),
    ) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        then : Values to return in case of the predicate being `True`.

        """
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self.pywhenthenthen.otherwise(expr._pyexpr))

    @typing.no_type_check
    def __getattr__(self, item) -> Expr:
        expr = self.otherwise(None)  # noqa: F841
        return eval(f"expr.{item}")


class WhenThen:
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    def when(self, predicate: Expr | bool | Series) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = expr_to_lit_or_expr(predicate)
        return WhenThenThen(self._pywhenthen.when(predicate._pyexpr))

    def otherwise(
        self,
        expr: (
            PolarsExprType
            | PythonLiteral
            | Series
            | Iterable[PolarsExprType | PythonLiteral | Series]
            | None
        ),
    ) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        then : Values to return in case of the predicate being `True`.

        """
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))

    @typing.no_type_check
    def __getattr__(self, item) -> Expr:
        expr = self.otherwise(None)  # noqa: F841
        return eval(f"expr.{item}")


class When:
    """Utility class. See the `when` function."""

    def __init__(self, pywhen: pywhen):
        self._pywhen = pywhen

    def then(
        self,
        expr: (
            PolarsExprType
            | PythonLiteral
            | Series
            | Iterable[PolarsExprType | PythonLiteral | Series]
            | None
        ),
    ) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        otherwise : Values to return in case of the predicate being `False`.

        """
        expr = expr_to_lit_or_expr(expr)
        pywhenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(pywhenthen)


def when(expr: Expr | bool | Series) -> When:
    """
    Start a "when, then, otherwise" expression.

    Examples
    --------
    Below we add a column with the value 1, where column "foo" > 2 and the value -1
    where it isn't.

    >>> df = pl.DataFrame({"foo": [1, 3, 4], "bar": [3, 4, 0]})
    >>> df.with_columns(
    ...     pl.when(pl.col("foo") > 2).then(pl.lit(1)).otherwise(pl.lit(-1))
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────────┐
    │ foo ┆ bar ┆ literal │
    │ --- ┆ --- ┆ ---     │
    │ i64 ┆ i64 ┆ i32     │
    ╞═════╪═════╪═════════╡
    │ 1   ┆ 3   ┆ -1      │
    │ 3   ┆ 4   ┆ 1       │
    │ 4   ┆ 0   ┆ 1       │
    └─────┴─────┴─────────┘

    Or with multiple `when, thens` chained:

    >>> df.with_columns(
    ...     pl.when(pl.col("foo") > 2)
    ...     .then(1)
    ...     .when(pl.col("bar") > 2)
    ...     .then(4)
    ...     .otherwise(-1)
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────────┐
    │ foo ┆ bar ┆ literal │
    │ --- ┆ --- ┆ ---     │
    │ i64 ┆ i64 ┆ i32     │
    ╞═════╪═════╪═════════╡
    │ 1   ┆ 3   ┆ 4       │
    │ 3   ┆ 4   ┆ 1       │
    │ 4   ┆ 0   ┆ 1       │
    └─────┴─────┴─────────┘

    See Also
    --------
    then : Values to return in case of the predicate being `True`.
    otherwise : Values to return in case of the predicate being `False`.

    """
    expr = expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)
