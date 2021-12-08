from typing import Any, Union

try:
    from polars.polars import when as pywhen

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

from polars import internals as pli


class WhenThenThen:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhenthenthen: Any):
        self.pywenthenthen = pywhenthenthen

    def when(self, predicate: pli.Expr) -> "WhenThenThen":
        """
        Start another when, then, otherwise layer.
        """
        return WhenThenThen(self.pywenthenthen.when(predicate._pyexpr))

    def then(self, expr: Union[pli.Expr, int, float, str]) -> "WhenThenThen":
        """
        Values to return in case of the predicate being `True`.

        See Also: the `when` function.
        """
        expr_ = pli.expr_to_lit_or_expr(expr)
        return WhenThenThen(self.pywenthenthen.then(expr_._pyexpr))

    def otherwise(self, expr: Union[pli.Expr, int, float, str]) -> pli.Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also: the `when` function.
        """
        expr = pli.expr_to_lit_or_expr(expr)
        return pli.wrap_expr(self.pywenthenthen.otherwise(expr._pyexpr))


class WhenThen:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    def when(self, predicate: pli.Expr) -> WhenThenThen:
        """
        Start another when, then, otherwise layer.
        """
        return WhenThenThen(self._pywhenthen.when(predicate._pyexpr))

    def otherwise(self, expr: Union[pli.Expr, int, float, str]) -> pli.Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also: the `when` function.
        """
        expr = pli.expr_to_lit_or_expr(expr)
        return pli.wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))


class When:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhen: "pywhen"):
        self._pywhen = pywhen

    def then(self, expr: Union[pli.Expr, int, float, str]) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also: the `when` function.
        """
        expr = pli.expr_to_lit_or_expr(expr)
        pywhenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(pywhenthen)


def when(expr: pli.Expr) -> When:
    """
    Start a when, then, otherwise expression.

    Examples
    --------

    Below we add a column with the value 1, where column "foo" > 2 and the value -1 where it isn't.

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

    """
    expr = pli.expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)
