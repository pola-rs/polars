import copy
import typing as tp
from datetime import datetime
from typing import Any, Callable, Optional, Sequence, Type, Union

import numpy as np

import polars as pl

try:
    from polars.polars import PyExpr

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

from ..datatypes import Boolean, DataType, Date, Datetime, Float64, Int64, Utf8
from .functions import col, lit

__all__ = [
    "Expr",
    "ExprStringNameSpace",
    "ExprDateTimeNameSpace",
    "expr_to_lit_or_expr",
]


def _selection_to_pyexpr_list(
    exprs: Union[str, "Expr", Sequence[str], Sequence["Expr"]]
) -> tp.List["PyExpr"]:
    pyexpr_list: tp.List[PyExpr]
    if isinstance(exprs, str):
        pyexpr_list = [col(exprs)._pyexpr]
    elif isinstance(exprs, (bool, int, float)):
        pyexpr_list = [lit(exprs)._pyexpr]
    elif isinstance(exprs, Expr):
        pyexpr_list = [exprs._pyexpr]
    else:
        pyexpr_list = []
        for expr in exprs:
            if isinstance(expr, str):
                expr = col(expr)
            elif isinstance(expr, (bool, int, float)):
                expr = lit(expr)
            pyexpr_list.append(expr._pyexpr)
    return pyexpr_list


def wrap_expr(pyexpr: "PyExpr") -> "Expr":
    return Expr._from_pyexpr(pyexpr)


class Expr:
    """
    Expressions that can be used in various contexts.
    """

    def __init__(self) -> None:
        self._pyexpr: PyExpr

    @staticmethod
    def _from_pyexpr(pyexpr: "PyExpr") -> "Expr":
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __to_pyexpr(self, other: Any) -> "PyExpr":
        if isinstance(other, PyExpr):
            return other
        elif isinstance(other, Expr):
            return other._pyexpr
        else:
            return lit(other)._pyexpr

    def __to_expr(self, other: Any) -> "Expr":
        if isinstance(other, Expr):
            return other
        return lit(other)

    def __bool__(self) -> "Expr":
        raise ValueError(
            "Since Expr are lazy, the truthiness of an Expr is ambiguous. \
            Hint: use '&' or '|' to chain Expr together, not and/or."
        )

    def __invert__(self) -> "Expr":
        return self.is_not()

    def __xor__(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr._xor(self.__to_pyexpr(other)))

    def __rxor__(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr._xor(self.__to_pyexpr(other)))

    def __and__(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr._and(self.__to_pyexpr(other)))

    def __rand__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr._and(self.__to_pyexpr(other)))

    def __or__(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr._or(self.__to_pyexpr(other)))

    def __ror__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other)._or(self._pyexpr))

    def __add__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr + self.__to_pyexpr(other))

    def __radd__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other) + self._pyexpr)

    def __sub__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr - self.__to_pyexpr(other))

    def __rsub__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other) - self._pyexpr)

    def __mul__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr * self.__to_pyexpr(other))

    def __rmul__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other) * self._pyexpr)

    def __truediv__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr / self.__to_pyexpr(other))

    def __rtruediv__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other) / self._pyexpr)

    def __floordiv__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr // self.__to_pyexpr(other))

    def __rfloordiv__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other) // self._pyexpr)

    def __mod__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr % self.__to_pyexpr(other))

    def __rmod__(self, other: Any) -> "Expr":
        return wrap_expr(self.__to_pyexpr(other) % self._pyexpr)

    def __pow__(self, power: float, modulo: None = None) -> "Expr":
        return self.pow(power)

    def __ge__(self, other: Any) -> "Expr":
        return self.gt_eq(self.__to_expr(other))

    def __le__(self, other: Any) -> "Expr":
        return self.lt_eq(self.__to_expr(other))

    def __eq__(self, other: Any) -> "Expr":  # type: ignore[override]
        return self.eq(self.__to_expr(other))

    def __ne__(self, other: Any) -> "Expr":  # type: ignore[override]
        return self.neq(self.__to_expr(other))

    def __lt__(self, other: Any) -> "Expr":
        return self.lt(self.__to_expr(other))

    def __gt__(self, other: Any) -> "Expr":
        return self.gt(self.__to_expr(other))

    def eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.eq(other._pyexpr))

    def neq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.neq(other._pyexpr))

    def gt(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.gt(other._pyexpr))

    def gt_eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.gt_eq(other._pyexpr))

    def lt_eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.lt_eq(other._pyexpr))

    def lt(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.lt(other._pyexpr))

    def __neg__(self) -> "Expr":
        return pl.lit(0) - self  # type: ignore

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> "Expr":
        """
        Numpy universal functions.
        """
        out_type = ufunc(np.array([1])).dtype
        if "float" in str(out_type):
            dtype = pl.Float64  # type: ignore
        else:
            dtype = None  # type: ignore

        args = []
        for inp in inputs:
            if not isinstance(inp, Expr):
                args.append(inp)

        def function(s: "pl.Series") -> "pl.Series":
            return ufunc(s, *args, **kwargs)

        if "dtype" in kwargs:
            return self.map(function, return_dtype=kwargs["dtype"])
        return self.map(function, return_dtype=dtype)

    def sqrt(self) -> "Expr":
        """
        Compute the square root of the elements
        """
        return self ** 0.5

    def alias(self, name: str) -> "Expr":
        """
        Rename the output of an expression.

        Parameters
        ----------
        name
            New name.

        Examples
        --------

        >>> df = pl.DataFrame({
        >>>     "a": [1, 2, 3],
        >>>     "b": ["a", "b", None]
        >>> })
        >>> df
        shape: (3, 2)
        ╭─────┬──────╮
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ "a"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ "b"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null │
        ╰─────┴──────╯
        >>> df.select([
        >>>     col("a").alias("bar"),
        >>>     col("b").alias("foo")
        >>> ])
        shape: (3, 2)
        ╭─────┬──────╮
        │ bar ┆ foo  │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ "a"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ "b"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null │
        ╰─────┴──────╯

        """
        return wrap_expr(self._pyexpr.alias(name))

    def exclude(self, columns: Union[str, tp.List[str]]) -> "Expr":
        """
         Exclude certain columns from a wildcard/regex selection.

         You may also use regexes int he exclude list. They must start with `^` and end with `$`.

         Parameters
         ----------
         columns
             Column(s) to exclude from selection

         Examples
         --------

         >>> df = pl.DataFrame({
         >>>     "a": [1, 2, 3],
         >>>     "b": ["a", "b", None],
         >>>     "c": [None, 2, 1]
         >>> })
         >>> df
         shape: (3, 3)
         ╭─────┬──────┬──────╮
         │ a   ┆ b    ┆ c    │
         │ --- ┆ ---  ┆ ---  │
         │ i64 ┆ str  ┆ i64  │
         ╞═════╪══════╪══════╡
         │ 1   ┆ "a"  ┆ null │
         ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
         │ 2   ┆ "b"  ┆ 2    │
         ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
         │ 3   ┆ null ┆ 1    │
         ╰─────┴──────┴──────╯
         >>> df.select(col("*").exclude("b"))
        shape: (3, 2)
         ╭─────┬──────╮
         │ a   ┆ c    │
         │ --- ┆ ---  │
         │ i64 ┆ i64  │
         ╞═════╪══════╡
         │ 1   ┆ null │
         ├╌╌╌╌╌┼╌╌╌╌╌╌┤
         │ 2   ┆ 2    │
         ├╌╌╌╌╌┼╌╌╌╌╌╌┤
         │ 3   ┆ 1    │
         ╰─────┴──────╯
        """
        if isinstance(columns, str):
            columns = [columns]
        return wrap_expr(self._pyexpr.exclude(columns))

    def keep_name(self) -> "Expr":
        """
        Keep the original root name of the expression.

        Examples
        --------

        A groupby aggregation often changes the name of a column.
        With `keep_name` we can keep the original name of the column

        >>> df = pl.DataFrame({
        >>> "a": [1, 2, 3],
        >>> "b": ["a", "b", None]
        >>> })
        >>> (df.groupby("a")
        >>> .agg(col("b").list())
        >>> .sort(by="a")
        >>> )
        shape: (3, 2)
        ╭─────┬────────────╮
        │ a   ┆ b_agg_list │
        │ --- ┆ ---        │
        │ i64 ┆ list [str] │
        ╞═════╪════════════╡
        │ 1   ┆ [a]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ [b]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]     │
        ╰─────┴────────────╯
        >>> # keep the original column name
        >>> (df.groupby("a")
        >>> .agg(col("b").list().keep_name())
        >>> .sort(by="a")
        >>> )
        shape: (3, 2)
        ╭─────┬────────────╮
        │ a   ┆ b          │
        │ --- ┆ ---        │
        │ i64 ┆ list [str] │
        ╞═════╪════════════╡
        │ 1   ┆ [a]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ [b]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]     │
        ╰─────┴────────────╯

        """

        return wrap_expr(self._pyexpr.keep_name())

    def prefix(self, prefix: str) -> "Expr":
        """
        Add a prefix the to root column name of the expression.

        Examples
        --------

        >>> df = pl.DataFrame(
        >>> {
        >>>     "A": [1, 2, 3, 4, 5],
        >>>     "fruits": ["banana", "banana", "apple", "apple", "banana"],
        >>>     "B": [5, 4, 3, 2, 1],
        >>>     "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        >>> })
        shape: (5, 4)
        ╭─────┬──────────┬─────┬──────────╮
        │ A   ┆ fruits   ┆ B   ┆ cars     │
        │ --- ┆ ---      ┆ --- ┆ ---      │
        │ i64 ┆ str      ┆ i64 ┆ str      │
        ╞═════╪══════════╪═════╪══════════╡
        │ 1   ┆ "banana" ┆ 5   ┆ "beetle" │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ "banana" ┆ 4   ┆ "audi"   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ "apple"  ┆ 3   ┆ "beetle" │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ "apple"  ┆ 2   ┆ "beetle" │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ "banana" ┆ 1   ┆ "beetle" │
        ╰─────┴──────────┴─────┴──────────╯
        >>> (df.select([
        >>> pl.all(),
        >>> pl.all().reverse().suffix("_reverse")
        >>> ]))
        shape: (5, 8)
        ╭─────┬──────────┬─────┬──────────┬───────────┬────────────────┬───────────┬──────────────╮
        │ A   ┆ fruits   ┆ B   ┆ cars     ┆ A_reverse ┆ fruits_reverse ┆ B_reverse ┆ cars_reverse │
        │ --- ┆ ---      ┆ --- ┆ ---      ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str      ┆ i64 ┆ str      ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪══════════╪═════╪══════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ "banana" ┆ 5   ┆ "beetle" ┆ 5         ┆ "banana"       ┆ 1         ┆ "beetle"     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ "banana" ┆ 4   ┆ "audi"   ┆ 4         ┆ "apple"        ┆ 2         ┆ "beetle"     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ "apple"  ┆ 3   ┆ "beetle" ┆ 3         ┆ "apple"        ┆ 3         ┆ "beetle"     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ "apple"  ┆ 2   ┆ "beetle" ┆ 2         ┆ "banana"       ┆ 4         ┆ "audi"       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ "banana" ┆ 1   ┆ "beetle" ┆ 1         ┆ "banana"       ┆ 5         ┆ "beetle"     │
        ╰─────┴──────────┴─────┴──────────┴───────────┴────────────────┴───────────┴──────────────╯

        """
        return wrap_expr(self._pyexpr.prefix(prefix))

    def suffix(self, suffix: str) -> "Expr":
        """
        Add a suffix the to root column name of the expression.

        Examples
        --------

        >>> df = pl.DataFrame(
        >>> {
        >>>     "A": [1, 2, 3, 4, 5],
        >>>     "fruits": ["banana", "banana", "apple", "apple", "banana"],
        >>>     "B": [5, 4, 3, 2, 1],
        >>>     "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        >>> })
        shape: (5, 4)
        ╭─────┬──────────┬─────┬──────────╮
        │ A   ┆ fruits   ┆ B   ┆ cars     │
        │ --- ┆ ---      ┆ --- ┆ ---      │
        │ i64 ┆ str      ┆ i64 ┆ str      │
        ╞═════╪══════════╪═════╪══════════╡
        │ 1   ┆ "banana" ┆ 5   ┆ "beetle" │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ "banana" ┆ 4   ┆ "audi"   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ "apple"  ┆ 3   ┆ "beetle" │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ "apple"  ┆ 2   ┆ "beetle" │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ "banana" ┆ 1   ┆ "beetle" │
        ╰─────┴──────────┴─────┴──────────╯
        >>> (df.select([
        >>> pl.all(),
        >>> pl.all().reverse().prefix("reverse_")
        >>> ]))
        shape: (5, 8)
        ╭─────┬──────────┬─────┬──────────┬───────────┬────────────────┬───────────┬──────────────╮
        │ A   ┆ fruits   ┆ B   ┆ cars     ┆ reverse_A ┆ reverse_fruits ┆ reverse_B ┆ reverse_cars │
        │ --- ┆ ---      ┆ --- ┆ ---      ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str      ┆ i64 ┆ str      ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪══════════╪═════╪══════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ "banana" ┆ 5   ┆ "beetle" ┆ 5         ┆ "banana"       ┆ 1         ┆ "beetle"     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ "banana" ┆ 4   ┆ "audi"   ┆ 4         ┆ "apple"        ┆ 2         ┆ "beetle"     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ "apple"  ┆ 3   ┆ "beetle" ┆ 3         ┆ "apple"        ┆ 3         ┆ "beetle"     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ "apple"  ┆ 2   ┆ "beetle" ┆ 2         ┆ "banana"       ┆ 4         ┆ "audi"       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ "banana" ┆ 1   ┆ "beetle" ┆ 1         ┆ "banana"       ┆ 5         ┆ "beetle"     │
        ╰─────┴──────────┴─────┴──────────┴───────────┴────────────────┴───────────┴──────────────╯

        """
        return wrap_expr(self._pyexpr.suffix(suffix))

    def is_not(self) -> "Expr":
        """
        Negate a boolean expression.

        Examples
        --------

        >>> df = pl.DataFrame({
        >>>     "a": [True, False, False],
        >>>     "b": ["a", "b", None],
        >>> })
        shape: (3, 2)
        ╭───────┬──────╮
        │ a     ┆ b    │
        │ ---   ┆ ---  │
        │ bool  ┆ str  │
        ╞═══════╪══════╡
        │ true  ┆ "a"  │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ false ┆ "b"  │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ false ┆ null │
        ╰───────┴──────╯
        >>> df.select(col("a").is_not())
        shape: (3, 1)
        ╭───────╮
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        ╰───────╯

        """
        return wrap_expr(self._pyexpr.is_not())

    def is_null(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression contains null values.
        """
        return wrap_expr(self._pyexpr.is_null())

    def is_not_null(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression does not contain null values.
        """
        return wrap_expr(self._pyexpr.is_not_null())

    def is_finite(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are finite.
        """
        return wrap_expr(self._pyexpr.is_finite())

    def is_infinite(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are infinite.
        """
        return wrap_expr(self._pyexpr.is_infinite())

    def is_nan(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are NaN (Not A Number).
        """
        return wrap_expr(self._pyexpr.is_nan())

    def is_not_nan(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are not NaN (Not A Number).
        """
        return wrap_expr(self._pyexpr.is_not_nan())

    def agg_groups(self) -> "Expr":
        """
        Get the group indexes of the group by operation.
        Should be used in aggregation context only.
        """
        return wrap_expr(self._pyexpr.agg_groups())

    def count(self) -> "Expr":
        """Count the number of values in this expression"""
        return wrap_expr(self._pyexpr.count())

    def slice(self, offset: int, length: int) -> "Expr":
        """
        Slice the Series.

        Parameters
        ----------
        offset
            Start index.
        length
            Length of the slice.
        """
        return wrap_expr(self._pyexpr.slice(offset, length))

    def drop_nulls(self) -> "Expr":
        """
        Syntactic sugar for:

        >>> col("foo").filter(col("foo").is_not_null())
        """
        return self.filter(self.is_not_null())

    def cumsum(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cumsum(reverse))

    def cummin(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cummin(reverse))

    def cummax(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cummax(reverse))

    def round(self, decimals: int) -> "Expr":
        """
        Round underlying floating point data by `decimals` digits.

        Parameters
        ----------
        decimals
            Number of decimals to round by.
        """
        return wrap_expr(self._pyexpr.round(decimals))

    def dot(self, other: "Expr") -> "Expr":
        """
        Compute the dot/inner product between two Expressions

        Parameters
        ----------
        other
            Expression to compute dot product with
        """
        other = expr_to_lit_or_expr(other, str_to_lit=False)
        return wrap_expr(self._pyexpr.dot(other._pyexpr))

    def mode(self) -> "Expr":
        """
        Compute the most occurring value(s). Can return multiple Values
        """
        return wrap_expr(self._pyexpr.mode())

    def cast(self, dtype: Type[Any], strict: bool = True) -> "Expr":
        """
        Cast between data types.

        Parameters
        ----------
        data_type
            DataType to cast to
        strict
            Throw an error if a cast could not be done for instance due to an overflow
        """
        if dtype == str:
            dtype = Utf8
        elif dtype == bool:
            dtype = Boolean
        elif dtype == float:
            dtype = Float64
        elif dtype == int:
            dtype = Int64
        return wrap_expr(self._pyexpr.cast(dtype, strict))

    def sort(self, reverse: bool = False) -> "Expr":
        """
        Sort this column. In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        reverse
            False -> order from small to large.
            True -> order from large to small.
        """
        return wrap_expr(self._pyexpr.sort(reverse))

    def arg_sort(self, reverse: bool = False) -> "Expr":
        """
        Get the index values that would sort this column.

        Parameters
        ----------
        reverse
            False -> order from small to large.
            True -> order from large to small.

        Returns
        -------
        out
            Series of type UInt32
        """
        return wrap_expr(self._pyexpr.arg_sort(reverse))

    def arg_max(self) -> "Expr":
        """
        Get the index of the maximal value.
        """
        return wrap_expr(self._pyexpr.arg_max())

    def arg_min(self) -> "Expr":
        """
        Get the index of the minimal value.
        """
        return wrap_expr(self._pyexpr.arg_min())

    def sort_by(self, by: Union["Expr", str], reverse: bool = False) -> "Expr":
        """
        Sort this column by the ordering of another column.
        In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        by
            The column used for sorting.
        reverse
            False -> order from small to large.
            True -> order from large to small.
        """
        if isinstance(by, str):
            by = col(by)

        return wrap_expr(self._pyexpr.sort_by(by._pyexpr, reverse))

    def take(self, index: Union[tp.List[int], "Expr", "pl.Series"]) -> "Expr":
        """
        Take values by index.

        Parameters
        ----------
        index
            An expression that leads to a UInt32 dtyped Series.

        Returns
        -------
        Values taken by index
        """
        if isinstance(index, (list, np.ndarray)):
            index = pl.lit(pl.Series("", index, dtype=pl.UInt32))  # type: ignore
        elif isinstance(index, pl.Series):
            index = pl.lit(index)  # type: ignore
        else:
            index = expr_to_lit_or_expr(index, str_to_lit=False)  # type: ignore
        return wrap_expr(self._pyexpr.take(index._pyexpr))  # type: ignore

    def shift(self, periods: int = 1) -> "Expr":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_expr(self._pyexpr.shift(periods))

    def shift_and_fill(self, periods: int, fill_value: "Expr") -> "Expr":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            Fill None values with the result of this expression.
        """
        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.shift_and_fill(periods, fill_value._pyexpr))

    def fill_null(self, fill_value: Union[str, int, float, "Expr"]) -> "Expr":
        """
        Fill none value with a fill value or strategy

        fill_value
            Fill null strategy or a value
                   * "backward"
                   * "forward"
                   * "min"
                   * "max"
                   * "mean"
                   * "one"
                   * "zero"
        """
        if fill_value in ["backward", "forward", "min", "max", "mean", "zero", "one"]:
            return wrap_expr(self._pyexpr.fill_null_with_strategy(fill_value))

        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.fill_null(fill_value._pyexpr))

    def fill_nan(self, fill_value: Union[str, int, float, "Expr"]) -> "Expr":
        """
        Fill none value with a fill value
        """
        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.fill_nan(fill_value._pyexpr))

    def forward_fill(self) -> "Expr":
        """
        Fill missing values with the latest seen values
        """
        return wrap_expr(self._pyexpr.forward_fill())

    def backward_fill(self) -> "Expr":
        """
        Fill missing values with the next to be seen values
        """
        return wrap_expr(self._pyexpr.backward_fill())

    def reverse(self) -> "Expr":
        """
        Reverse the selection.
        """
        return wrap_expr(self._pyexpr.reverse())

    def std(self) -> "Expr":
        """
        Get standard deviation.
        """
        return wrap_expr(self._pyexpr.std())

    def var(self) -> "Expr":
        """
        Get variance.
        """
        return wrap_expr(self._pyexpr.var())

    def max(self) -> "Expr":
        """
        Get maximum value.
        """
        return wrap_expr(self._pyexpr.max())

    def min(self) -> "Expr":
        """
        Get minimum value.
        """
        return wrap_expr(self._pyexpr.min())

    def sum(self) -> "Expr":
        """
        Get sum value.
        """
        return wrap_expr(self._pyexpr.sum())

    def mean(self) -> "Expr":
        """
        Get mean value.
        """
        return wrap_expr(self._pyexpr.mean())

    def median(self) -> "Expr":
        """
        Get median value.
        """
        return wrap_expr(self._pyexpr.median())

    def n_unique(self) -> "Expr":
        """Count unique values."""
        return wrap_expr(self._pyexpr.n_unique())

    def arg_unique(self) -> "Expr":
        """Get index of first unique value."""
        return wrap_expr(self._pyexpr.arg_unique())

    def unique(self) -> "Expr":
        """Get unique values."""
        return wrap_expr(self._pyexpr.unique())

    def first(self) -> "Expr":
        """
        Get the first value.
        """
        return wrap_expr(self._pyexpr.first())

    def last(self) -> "Expr":
        """
        Get the last value.
        """
        return wrap_expr(self._pyexpr.last())

    def list(self) -> "Expr":
        """
        Aggregate to list.
        """
        return wrap_expr(self._pyexpr.list())

    def over(self, expr: Union[str, "Expr", tp.List["Expr"]]) -> "Expr":
        """
        Apply window function over a subgroup.
        This is similar to a groupby + aggregation + self join.
        Or similar to [window functions in Postgres](https://www.postgresql.org/docs/9.1/tutorial-window.html)

        Parameters
        ----------
        expr
            Column(s) to group by.

        Examples
        --------

        >>> df = DataFrame({
        >>>    "groups": [1, 1, 2, 2, 1, 2, 3, 3, 1],
        >>>    "values": [1, 2, 3, 4, 5, 6, 7, 8, 8]
        >>>})
        >>> (df.lazy()
        >>>    .select([
        >>>       col("groups")
        >>>       sum("values").over("groups"))
        >>>   ]).collect())
            ╭────────┬────────╮
            │ groups ┆ values │
            │ ---    ┆ ---    │
            │ i32    ┆ i32    │
            ╞════════╪════════╡
            │ 1      ┆ 16     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 1      ┆ 16     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 2      ┆ 13     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 2      ┆ 13     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ ...    ┆ ...    │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 1      ┆ 16     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 2      ┆ 13     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 3      ┆ 15     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 3      ┆ 15     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 1      ┆ 16     │
            ╰────────┴────────╯

        """

        pyexprs = _selection_to_pyexpr_list(expr)

        return wrap_expr(self._pyexpr.over(pyexprs))

    def is_unique(self) -> "Expr":
        """
        Get mask of unique values.
        """
        return wrap_expr(self._pyexpr.is_unique())

    def is_first(self) -> "Expr":
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series
        """
        return wrap_expr(self._pyexpr.is_first())

    def is_duplicated(self) -> "Expr":
        """
        Get mask of duplicated values.
        """
        return wrap_expr(self._pyexpr.is_duplicated())

    def quantile(self, quantile: float) -> "Expr":
        """
        Get quantile value.
        """
        return wrap_expr(self._pyexpr.quantile(quantile))

    def filter(self, predicate: "Expr") -> "Expr":
        """
        Filter a single column.
        Mostly useful in in aggregation context. If you want to filter on a DataFrame level, use `LazyFrame.filter`.

        Parameters
        ----------
        predicate
            Boolean expression.
        """
        return wrap_expr(self._pyexpr.filter(predicate._pyexpr))

    def where(self, predicate: "Expr") -> "Expr":
        "alias for filter"
        return self.filter(predicate)

    def map(
        self,
        f: Callable[["pl.Series"], "pl.Series"],
        return_dtype: Optional[Type[DataType]] = None,
        agg_list: bool = False,
    ) -> "Expr":
        """
        Apply a custom python function. This function must produce a `Series`. Any other value will be stored as
        null/missing. If you want to apply a function over single values, consider using `apply`.

        [read more in the book](https://ritchie46.github.io/polars-book/how_can_i/use_custom_functions.html#lazy)

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        """
        if return_dtype == str:
            return_dtype = Utf8
        elif return_dtype == int:
            return_dtype = Int64
        elif return_dtype == float:
            return_dtype = Float64
        elif return_dtype == bool:
            return_dtype = Boolean
        return wrap_expr(self._pyexpr.map(f, return_dtype, agg_list))

    def apply(
        self,
        f: Union[Callable[["pl.Series"], "pl.Series"], Callable[[Any], Any]],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "Expr":
        """
        Apply a custom function in a GroupBy or Projection context.

        Depending on the context it has the following behavior:

        ## Context

        * Select/Project
            expected type `f`: Callable[[Any], Any]
            Applies a python function over each individual value in the column.
        * GroupBy
            expected type `f`: Callable[[Series], Series]
            Applies a python function over each group.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.

        Examples
        --------

        >>> df = pl.DataFrame({"a": [1,  2,  1,  1],
                   "b": ["a", "b", "c", "c"]})
        >>> (df
         .lazy()
         .groupby("b")
         .agg([col("a").apply(lambda x: x.sum())])
         .collect()
        )
        shape: (3, 2)
        ╭─────┬─────╮
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 2   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ c   ┆ 2   │
        ╰─────┴─────╯

        """

        # input x: Series of type list containing the group values
        def wrap_f(x: "pl.Series") -> "pl.Series":
            return x.apply(f, return_dtype=return_dtype)

        return self.map(wrap_f, agg_list=True)

    def flatten(self) -> "Expr":
        """
        Alias for explode.

        Explode a list or utf8 Series. This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype
        """

        return wrap_expr(self._pyexpr.explode())

    def explode(self) -> "Expr":
        """
        Explode a list or utf8 Series. This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype
        """
        return wrap_expr(self._pyexpr.explode())

    def take_every(self, n: int) -> "Expr":
        """
        Take every nth value in the Series and return as a new Series.
        """
        return wrap_expr(self._pyexpr.take_every(n))

    def head(self, n: Optional[int] = None) -> "Expr":
        """
        Take the first n values.
        """
        return wrap_expr(self._pyexpr.head(n))

    def tail(self, n: Optional[int] = None) -> "Expr":
        """
        Take the last n values.
        """
        return wrap_expr(self._pyexpr.tail(n))

    def pow(self, exponent: float) -> "Expr":
        """
        Raise expression to the power of exponent.
        """
        return wrap_expr(self._pyexpr.pow(exponent))

    def is_in(self, other: Union["Expr", tp.List[Any]]) -> "Expr":
        """
        Check if elements of this Series are in the right Series, or List values of the right Series.

        Parameters
        ----------
        other
            Series of primitive type or List type.

        Returns
        -------
        Expr that evaluates to a Boolean Series.
        """
        if isinstance(other, list):
            other = pl.lit(pl.Series(other))
        else:
            other = expr_to_lit_or_expr(other, str_to_lit=False)
        return wrap_expr(self._pyexpr.is_in(other._pyexpr))

    def repeat_by(self, by: "Expr") -> "Expr":
        """
        Repeat the elements in this Series `n` times by dictated by the number given by `by`.
        The elements are expanded into a `List`

        Parameters
        ----------
        by
            Numeric column that determines how often the values will be repeated.
            The column will be coerced to UInt32. Give this dtype to make the coercion a no-op.

        Returns
        -------
        Series of type List
        """
        by = expr_to_lit_or_expr(by, False)
        return wrap_expr(self._pyexpr.repeat_by(by._pyexpr))

    def is_between(
        self, start: Union["Expr", datetime], end: Union["Expr", datetime]
    ) -> "Expr":
        """
        Check if this expression is between start and end.
        """
        cast_to_datetime = False
        if isinstance(start, datetime):
            start = lit(start)
            cast_to_datetime = True
        if isinstance(end, datetime):
            end = lit(end)
            cast_to_datetime = True
        if cast_to_datetime:
            expr = self.cast(Datetime)
        else:
            expr = self
        return ((expr > start) & (expr < end)).alias("is_between")

    @property
    def dt(self) -> "ExprDateTimeNameSpace":
        """
        Create an object namespace of all datetime related methods.
        """
        return ExprDateTimeNameSpace(self)

    @property
    def str(self) -> "ExprStringNameSpace":
        """
        Create an object namespace of all string related methods.
        """
        return ExprStringNameSpace(self)

    @property
    def arr(self) -> "ExprListNameSpace":
        """
        Create an object namespace of all datetime related methods.
        """
        return ExprListNameSpace(self)

    def hash(self, k0: int = 0, k1: int = 1, k2: int = 2, k3: int = 3) -> "pl.Expr":
        """
        Hash the Series.

        The hash value is of type `Datetime`

        Parameters
        ----------
        k0
            seed parameter
        k1
            seed parameter
        k2
            seed parameter
        k3
            seed parameter
        """
        return wrap_expr(self._pyexpr.hash(k0, k1, k2, k3))

    def reinterpret(self, signed: bool) -> "pl.Expr":
        """
        Reinterpret the underlying bits as a signed/unsigned integer.
        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            True -> pl.Int64
            False -> pl.UInt64
        """
        return wrap_expr(self._pyexpr.reinterpret(signed))

    def inspect(self, fmt: str = "{}") -> "pl.Expr":  # type: ignore
        """
        Prints the value that this expression evaluates to and passes on the value.

        >>> df.select(col("foo").cumsum().inspect("value is: {}").alias("bar"))
        """

        def inspect(s: "pl.Series") -> "pl.Series":
            print(fmt.format(s))  # type: ignore
            return s

        return self.map(inspect, return_dtype=None, agg_list=True)

    def interpolate(self) -> "pl.Expr":
        """
        Interpolate intermediate values. The interpolation method is linear.
        """
        return wrap_expr(self._pyexpr.interpolate())

    def rolling_min(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        apply a rolling min (moving min) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_min(window_size, weights, min_periods, center)
        )

    def rolling_max(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Apply a rolling max (moving max) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_max(window_size, weights, min_periods, center)
        )

    def rolling_mean(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Apply a rolling mean (moving mean) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------

        >>> df = pl.DataFrame(
        >>>     {
        >>>         "A": [1.0, 8.0, 6.0, 2.0, 16.0, 10.0]
        >>>     }
        >>> )
        >>> df.select([
        >>>     col("A").rolling_mean(window_size=2)
        >>> ])
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 4.5  │
        ├╌╌╌╌╌╌┤
        │ 7    │
        ├╌╌╌╌╌╌┤
        │ 4    │
        ├╌╌╌╌╌╌┤
        │ 9    │
        ├╌╌╌╌╌╌┤
        │ 13   │
        └──────┘

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_mean(window_size, weights, min_periods, center)
        )

    def rolling_sum(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Apply a rolling sum (moving sum) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_sum(window_size, weights, min_periods, center)
        )

    def rolling_std(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Compute a rolling std dev

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_std(window_size, weights, min_periods, center)
        )

    def rolling_var(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_var(window_size, weights, min_periods, center)
        )

    def rolling_apply(
        self, window_size: int, function: Callable[["pl.Series"], Any]
    ) -> "Expr":
        """
        Allows a custom rolling window function.
        Prefer the specific rolling window fucntions over this one, as they are faster.

        Prefer:
            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum

        Parameters
        ----------
        window_size
            Size of the rolling window
        function
            Aggregation function


        Examples
        --------

        >>> df = pl.DataFrame(
        >>>     {
        >>>         "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        >>>     }
        >>> )
        >>> df.select([
        >>>     col("A").rolling_apply(3, lambda s: s.std())
        >>> ])
        shape: (5, 1)
        ┌────────────────────┐
        │ A                  │
        │ ---                │
        │ f64                │
        ╞════════════════════╡
        │ null               │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null               │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4.358898943540674  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4.041451884327381  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5.5677643628300215 │
        └────────────────────┘

        """
        return wrap_expr(self._pyexpr.rolling_apply(window_size, function))

    def rolling_median(self, window_size: int) -> "Expr":
        """
        Compute a rolling median

        Parameters
        ----------
        window_size
            Size of the rolling window
        """
        return wrap_expr(self._pyexpr.rolling_median(window_size))

    def rolling_quantile(self, window_size: int, quantile: float) -> "Expr":
        """
        Compute a rolling quantile

        Parameters
        ----------
        window_size
            Size of the rolling window
        quantile
            quantile to compute
        """
        return wrap_expr(self._pyexpr.rolling_quantile(window_size, quantile))

    def rolling_skew(self, window_size: int, bias: bool = True) -> "Expr":
        """
        Compute a rolling skew
        window_size
            Size of the rolling window
        bias
            If False, then the calculations are corrected for statistical bias.
        """
        return wrap_expr(self._pyexpr.rolling_skew(window_size, bias))

    def abs(self) -> "Expr":
        """
        Take absolute values
        """
        return self.map(lambda s: s.abs())

    def argsort(self, reverse: bool = False) -> "Expr":
        """
        Index location of the sorted variant of this Series.
        Parameters
        ----------
        reverse
            Reverse the ordering. Default is from low to high.
        """
        return pl.argsort_by([self], [reverse])  # type: ignore

    def rank(self, method: str = "average") -> "Expr":  # type: ignore
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method
            {'average', 'min', 'max', 'dense', 'ordinal'}, optional
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):
              * 'average': The average of the ranks that would have been assigned to
                all the tied values is assigned to each value.
              * 'min': The minimum of the ranks that would have been assigned to all
                the tied values is assigned to each value.  (This is also
                referred to as "competition" ranking.)
              * 'max': The maximum of the ranks that would have been assigned to all
                the tied values is assigned to each value.
              * 'dense': Like 'min', but the rank of the next highest element is
                assigned the rank immediately after those assigned to the tied
                elements.
              * 'ordinal': All values are given a distinct rank, corresponding to
                the order that the values occur in `a`.
        """
        return wrap_expr(self._pyexpr.rank(method))

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> "Expr":  # type: ignore
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            number of slots to shift
        null_behavior
            {'ignore', 'drop'}
        """
        return wrap_expr(self._pyexpr.diff(n, null_behavior))

    def skew(self, bias: bool = True) -> "Expr":
        """Compute the sample skewness of a data set.
        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.


        See scipy.stats for more information.

        Parameters
        ----------
        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        .. math::
            g_1=\frac{m_3}{m_2^{3/2}}
        where
        .. math::
            m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i
        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.
        .. math::
            G_1=\frac{k_3}{k_2^{3/2}}=
                \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.
        """
        return wrap_expr(self._pyexpr.skew(bias))

    def kurtosis(self, fisher: bool = True, bias: bool = True) -> "Expr":
        """Compute the kurtosis (Fisher or Pearson) of a dataset.
        Kurtosis is the fourth central moment divided by the square of the
        variance. If Fisher's definition is used, then 3.0 is subtracted from
        the result to give 0.0 for a normal distribution.
        If bias is False then the kurtosis is calculated using k statistics to
        eliminate bias coming from biased moment estimators

        See scipy.stats for more information

        Parameters
        ----------
        fisher : bool, optional
            If True, Fisher's definition is used (normal ==> 0.0). If False,
            Pearson's definition is used (normal ==> 3.0).
        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.
        """
        return wrap_expr(self._pyexpr.kurtosis(fisher, bias))

    def clip(self, min_val: Union[int, float], max_val: Union[int, float]) -> "Expr":
        """
        Clip (limit) the values in an array.

        Parameters
        ----------
        min_val, max_val
            Minimum and maximum value.
        """
        min_val = pl.lit(min_val)  # type: ignore
        max_val = pl.lit(max_val)  # type: ignore
        return (
            pl.when(self < min_val)  # type: ignore
            .then(min_val)
            .when(self > max_val)
            .then(max_val)
            .otherwise(self)
        ).keep_name()

    def str_concat(self, delimiter: str = "-") -> "Expr":  # type: ignore
        """
        Vertically concat the values in the Series to a single string value.

        Returns
        -------
        Series of dtype Utf8

        Examples
        >>> df = pl.DataFrame({"foo": [1, None, 2]})
        >>> df = df.select(col("foo").str_concat("-"))
        shape: (1, 1)
        ┌──────────┐
        │ foo      │
        │ ---      │
        │ str      │
        ╞══════════╡
        │ 1-null-2 │
        └──────────┘
        """
        return wrap_expr(self._pyexpr.str_concat(delimiter))


class ExprListNameSpace:
    """
    Namespace for list related expressions
    """

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def lengths(self) -> Expr:
        """
        Get the length of the arrays as UInt32.
        """
        return wrap_expr(self._pyexpr.arr_lengths())

    def sum(self) -> "Expr":
        """
        Sum all the arrays in the list
        """
        return wrap_expr(self._pyexpr.lst_sum())

    def max(self) -> "Expr":
        """
        Compute the max value of the arrays in the list
        """
        return wrap_expr(self._pyexpr.lst_max())

    def min(self) -> "Expr":
        """
        Compute the min value of the arrays in the list
        """
        return wrap_expr(self._pyexpr.lst_min())

    def mean(self) -> "Expr":
        """
        Compute the mean value of the arrays in the list
        """
        return wrap_expr(self._pyexpr.lst_mean())

    def sort(self, reverse: bool) -> "Expr":
        """
        Sort the arrays in the list
        """
        return wrap_expr(self._pyexpr.lst_sort(reverse))

    def reverse(self) -> "Expr":
        """
        Reverse the arrays in the list
        """
        return wrap_expr(self._pyexpr.lst_reverse())

    def unique(self) -> "Expr":
        """
        Get the unique/distinct values in the list
        """
        return wrap_expr(self._pyexpr.lst_unique())

    def concat(self, other: Union[tp.List[Expr], Expr, str, tp.List[str]]) -> "Expr":
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series
        """
        if not isinstance(other, list):
            other = [other]  # type: ignore
        else:
            other = copy.copy(other)
        # mypy does not understand we have a list by now
        other.insert(0, wrap_expr(self._pyexpr))  # type: ignore
        return pl.concat_list(other)  # type: ignore


class ExprStringNameSpace:
    """
    Namespace for string related expressions
    """

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def strptime(
        self,
        datatype: Union[Date, Datetime],
        fmt: Optional[str] = None,
    ) -> Expr:
        """
        Parse utf8 expression as a Date/Datetimetype.

        Parameters
        ----------
        datatype
            Date | Datetime.
        fmt
            "yyyy-mm-dd".
        """
        if datatype == Date:
            return wrap_expr(self._pyexpr.str_parse_date(fmt))
        elif datatype == Datetime:
            return wrap_expr(self._pyexpr.str_parse_datetime(fmt))
        else:
            raise NotImplementedError

    def lengths(self) -> Expr:
        """
        Get the length of the Strings as UInt32.
        """
        return wrap_expr(self._pyexpr.str_lengths())

    def to_uppercase(self) -> Expr:
        """
        Transform to uppercase variant.
        """
        return wrap_expr(self._pyexpr.str_to_uppercase())

    def to_lowercase(self) -> Expr:
        """
        Transform to lowercase variant.
        """
        return wrap_expr(self._pyexpr.str_to_lowercase())

    def contains(self, pattern: str) -> Expr:
        """
        Check if string contains regex.

        Parameters
        ----------
        pattern
            Regex pattern.
        """
        return wrap_expr(self._pyexpr.str_contains(pattern))

    def json_path_match(self, json_path: str) -> Expr:
        """
        Extract the first match of json string with provided JSONPath expression.
        Throw errors if encounter invalid json strings.
        All return value will be casted to Utf8 regardless of the original value.
        Documentation on JSONPath standard: https://goessner.net/articles/JsonPath/

        Parameters
        ----------
        json_path
            A valid JSON path query string

        Returns
        -------
        Utf8 array. Contain null if original value is null or the json_path return nothing.

        Examples
        --------

        >>> df = pl.DataFrame({
        'json_val' = ['{"a":"1"}',None,'{"a":2}', '{"a":2.1}', '{"a":true}']
        })
        >>> df.select(pl.col('json_val').str.json_path_match('$.a')
        shape: (5,)
        Series: 'json_val' [str]
        [
            "1"
            null
            "2"
            "2.1"
            "true"
        ]
        """
        return wrap_expr(self._pyexpr.str_json_path_match(json_path))

    def replace(self, pattern: str, value: str) -> Expr:
        """
        Replace first regex match with a string value.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        """
        return wrap_expr(self._pyexpr.str_replace(pattern, value))

    def replace_all(self, pattern: str, value: str) -> Expr:
        """
        Replace substring on all regex pattern matches.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        """
        return wrap_expr(self._pyexpr.str_replace_all(pattern, value))

    def slice(self, start: int, length: Optional[int] = None) -> Expr:
        """
        Create subslices of the string values of a Utf8 Series.

        Parameters
        ----------
        start
            Start of the slice (negative indexing may be used).
        length
            Optional length of the slice.

        Returns
        -------
        Series of Utf8 type
        """
        return wrap_expr(self._pyexpr.str_slice(start, length))


class ExprDateTimeNameSpace:
    """
    Namespace for datetime related expressions.
    """

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def strftime(self, fmt: str) -> Expr:
        """
        Format Date/datetime with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
        """
        return wrap_expr(self._pyexpr.strftime(fmt))

    def year(self) -> Expr:
        """
        Extract year from underlying Date representation.
        Can be performed on Date and Datetime.

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32
        """
        return wrap_expr(self._pyexpr.year())

    def month(self) -> Expr:
        """
        Extract month from underlying Date representation.
        Can be performed on Date and Datetime.

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32
        """
        return wrap_expr(self._pyexpr.month())

    def week(self) -> Expr:
        """
        Extract the week from the underlying Date representation.
        Can be performed on Date and Datetime

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32
        """
        return wrap_expr(self._pyexpr.week())

    def weekday(self) -> Expr:
        """
        Extract the week day from the underlying Date representation.
        Can be performed on Date and Datetime.

        Returns the weekday number where monday = 0 and sunday = 6

        Returns
        -------
        Week day as UInt32
        """
        return wrap_expr(self._pyexpr.weekday())

    def day(self) -> Expr:
        """
        Extract day from underlying Date representation.
        Can be performed on Date and Datetime.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_expr(self._pyexpr.day())

    def ordinal_day(self) -> Expr:
        """
        Extract ordinal day from underlying Date representation.
        Can be performed on Date and Datetime.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_expr(self._pyexpr.ordinal_day())

    def hour(self) -> Expr:
        """
        Extract hour from underlying DateTime representation.
        Can be performed on Datetime.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32
        """
        return wrap_expr(self._pyexpr.hour())

    def minute(self) -> Expr:
        """
        Extract minutes from underlying DateTime representation.
        Can be performed on Datetime.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32
        """
        return wrap_expr(self._pyexpr.minute())

    def second(self) -> Expr:
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Datetime.

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32
        """
        return wrap_expr(self._pyexpr.second())

    def nanosecond(self) -> Expr:
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Datetime.

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32
        """
        return wrap_expr(self._pyexpr.nanosecond())

    def round(self, rule: str, n: int) -> Expr:
        """
        Round the datetime.

        Parameters
        ----------
        rule
            Units of the downscaling operation.

            Any of:
                - "month"
                - "week"
                - "day"
                - "hour"
                - "minute"
                - "second"

        n
            Number of units (e.g. 5 "day", 15 "minute".
        """
        return wrap_expr(self._pyexpr).map(lambda s: s.dt.round(rule, n), None)

    def to_python_datetime(self) -> Expr:
        """
        Go from Date/Datetime to python DateTime objects
        """
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.to_python_datetime(), return_dtype=pl.Object
        )

    def timestamp(self) -> Expr:
        """Return timestamp in ms as Int64 type."""
        return wrap_expr(self._pyexpr.timestamp())


def expr_to_lit_or_expr(
    expr: Union[Expr, int, float, str, tp.List[Expr], "pl.Series"],
    str_to_lit: bool = True,
) -> Expr:
    """
    Helper function that converts args to expressions.

    Parameters
    ----------
    expr
        Any argument.
    str_to_lit
        If True string argument `"foo"` will be converted to `lit("foo")`,
        If False it will be converted to `col("foo")`

    Returns
    -------

    """
    if isinstance(expr, str) and not str_to_lit:
        return col(expr)
    elif isinstance(expr, (int, float, str, pl.Series)) or expr is None:
        return lit(expr)
    elif isinstance(expr, list):
        return [expr_to_lit_or_expr(e, str_to_lit=str_to_lit) for e in expr]  # type: ignore[return-value]
    else:
        return expr
