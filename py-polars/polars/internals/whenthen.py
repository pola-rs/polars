from __future__ import annotations

import typing
from typing import Any, Sequence

try:
    from polars.polars import when as pywhen

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

from polars import internals as pli


class WhenThenThen:
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthenthen: Any):
        self.pywhenthenthen = pywhenthenthen

    def when(self, predicate: pli.Expr | bool) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = pli.expr_to_lit_or_expr(predicate)
        return WhenThenThen(self.pywhenthenthen.when(predicate._pyexpr))

    def then(
        self,
        expr: (
            pli.Expr
            | int
            | float
            | str
            | None
            | pli.Series
            | Sequence[(int | float | str | None)]
        ),
    ) -> WhenThenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        otherwise : Values to return in case of the predicate being `False`.

        """
        expr_ = pli.expr_to_lit_or_expr(expr)
        return WhenThenThen(self.pywhenthenthen.then(expr_._pyexpr))

    def otherwise(
        self,
        expr: (
            pli.Expr | int | float | str | None | Sequence[(int | float | str | None)]
        ),
    ) -> pli.Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        then : Values to return in case of the predicate being `True`.

        """
        expr = pli.expr_to_lit_or_expr(expr)
        return pli.wrap_expr(self.pywhenthenthen.otherwise(expr._pyexpr))

    @typing.no_type_check
    def __getattr__(self, item) -> pli.Expr:
        expr = self.otherwise(None)  # noqa: F841
        return eval(f"expr.{item}")


class WhenThen:
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    def when(self, predicate: pli.Expr | bool) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = pli.expr_to_lit_or_expr(predicate)
        return WhenThenThen(self._pywhenthen.when(predicate._pyexpr))

    def otherwise(self, expr: pli.Expr | int | float | str | None) -> pli.Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        then : Values to return in case of the predicate being `True`.

        """
        expr = pli.expr_to_lit_or_expr(expr)
        return pli.wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))

    @typing.no_type_check
    def __getattr__(self, item) -> pli.Expr:
        expr = self.otherwise(None)  # noqa: F841
        return eval(f"expr.{item}")


class When:
    """Utility class. See the `when` function."""

    def __init__(self, pywhen: pywhen):
        self._pywhen = pywhen

    def then(
        self,
        expr: (
            pli.Expr
            | pli.Series
            | int
            | float
            | str
            | None
            | Sequence[None | int | float | str]
        ),
    ) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        when : Start another when, then, otherwise layer.
        otherwise : Values to return in case of the predicate being `False`.

        """
        expr = pli.expr_to_lit_or_expr(expr)
        pywhenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(pywhenthen)


def when(expr: pli.Expr | bool) -> When:
    """
    Start a "when, then, otherwise" expression.

    Examples
    --------
    Below we add a column with the value 1, where column "foo" > 2 and the value -1
    where it isn't.

    >>> df = pl.DataFrame({"foo": [1, 3, 4], "bar": [3, 4, 0]})
    >>> df.with_column(pl.when(pl.col("foo") > 2).then(pl.lit(1)).otherwise(pl.lit(-1)))
    shape: (3, 3)
    ┌─────┬─────┬─────────┐
    │ foo ┆ bar ┆ literal │
    │ --- ┆ --- ┆ ---     │
    │ i64 ┆ i64 ┆ i32     │
    ╞═════╪═════╪═════════╡
    │ 1   ┆ 3   ┆ -1      │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ 4   ┆ 1       │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    │ 4   ┆ 0   ┆ 1       │
    └─────┴─────┴─────────┘

    Or with multiple `when, thens` chained:

    >>> df.with_column(
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
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ 4   ┆ 1       │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
    │ 4   ┆ 0   ┆ 1       │
    └─────┴─────┴─────────┘

    See Also
    --------
    then : Values to return in case of the predicate being `True`.
    otherwise : Values to return in case of the predicate being `False`.

    """
    expr = pli.expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)
