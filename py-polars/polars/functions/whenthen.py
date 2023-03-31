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
    from polars.type_aliases import IntoExpr


def when(expr: IntoExpr | Iterable[IntoExpr]) -> When:
    """
    Start a "when, then, otherwise" expression.

    Expression similar to an `if-else` statement in Python. Always initiated by a
    `pl.when(<condition>).then(<value if condition>)`. Optionally followed by chaining
    one or more `.when(<condition>).then(<value>)` statements. If none of the conditions
    are `True`, an optional `.otherwise(<value if all statements are false>)` can be
    appended at the end. If not appended, and none of the conditions are `True`, `None`
    will be returned.

    Examples
    --------
    Below we add a column with the value 1, where column "foo" > 2 and the value -1
    where it isn't.

    >>> df = pl.DataFrame({"foo": [1, 3, 4], "bar": [3, 4, 0]})
    >>> df.with_columns(
    ...     pl.when(pl.col("foo") > 2)
    ...     .then(pl.lit(1))
    ...     .otherwise(pl.lit(-1))
    ...     .alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i32 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ -1  │
    │ 3   ┆ 4   ┆ 1   │
    │ 4   ┆ 0   ┆ 1   │
    └─────┴─────┴─────┘

    Or with multiple `when, thens` chained:

    >>> df.with_columns(
    ...     pl.when(pl.col("foo") > 2)
    ...     .then(1)
    ...     .when(pl.col("bar") > 2)
    ...     .then(4)
    ...     .otherwise(-1)
    ...     .alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i32 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 4   │
    │ 3   ┆ 4   ┆ 1   │
    │ 4   ┆ 0   ┆ 1   │
    └─────┴─────┴─────┘

    Chained `when, thens` should be read as `if, elif, ... elif`, not
    as `if, if, ... if`, i.e. the first condition that evaluates to True will
    be picked. Note how in the example above for the second row in the dataframe,
    where `foo=3` and `bar=4`, the first `when` evaluates to `True`, and therefore
    the second `when`, which is also `True`, is not evaluated.

    The `otherwise` at the end is optional. If left out, any rows where none
    of the `when` expressions evaluate to True, are set to `null`:

    >>> df.with_columns(pl.when(pl.col("foo") > 2).then(pl.lit(1)).alias("val"))
    shape: (3, 3)
    ┌─────┬─────┬──────┐
    │ foo ┆ bar ┆ val  │
    │ --- ┆ --- ┆ ---  │
    │ i64 ┆ i64 ┆ i32  │
    ╞═════╪═════╪══════╡
    │ 1   ┆ 3   ┆ null │
    │ 3   ┆ 4   ┆ 1    │
    │ 4   ┆ 0   ┆ 1    │
    └─────┴─────┴──────┘


    """
    expr = expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)


class When:
    """Utility class. See the `when` function."""

    def __init__(self, pywhen: pywhen):
        self._pywhen = pywhen

    def then(self, expr: IntoExpr | Iterable[IntoExpr]) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = expr_to_lit_or_expr(expr)
        pywhenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(pywhenthen)


class WhenThen:
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    def when(self, predicate: IntoExpr | Iterable[IntoExpr]) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = expr_to_lit_or_expr(predicate)
        return WhenThenThen(self._pywhenthen.when(predicate._pyexpr))

    def otherwise(self, expr: IntoExpr | Iterable[IntoExpr]) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))

    @typing.no_type_check
    def __getattr__(self, item) -> Expr:
        expr = self.otherwise(None)  # noqa: F841
        return eval(f"expr.{item}")


class WhenThenThen:
    """Utility class. See the `when` function."""

    def __init__(self, pywhenthenthen: Any):
        self.pywhenthenthen = pywhenthenthen

    def when(self, predicate: IntoExpr | Iterable[IntoExpr]) -> WhenThenThen:
        """Start another "when, then, otherwise" layer."""
        predicate = expr_to_lit_or_expr(predicate)
        return WhenThenThen(self.pywhenthenthen.when(predicate._pyexpr))

    def then(self, expr: IntoExpr | Iterable[IntoExpr]) -> WhenThenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr_ = expr_to_lit_or_expr(expr)
        return WhenThenThen(self.pywhenthenthen.then(expr_._pyexpr))

    def otherwise(self, expr: IntoExpr | Iterable[IntoExpr]) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also
        --------
        pl.when : Documentation for `when, then, otherwise`

        """
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self.pywhenthenthen.otherwise(expr._pyexpr))

    @typing.no_type_check
    def __getattr__(self, item) -> Expr:
        expr = self.otherwise(None)  # noqa: F841
        return eval(f"expr.{item}")
