import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from polars.utils import _timedelta_to_pl_duration

try:
    from polars.polars import PyExpr

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

from polars import internals as pli
from polars.datatypes import (
    DataType,
    Date,
    Datetime,
    Float64,
    Int32,
    Object,
    UInt32,
    py_type_to_dtype,
)


def selection_to_pyexpr_list(
    exprs: Union[str, "Expr", Sequence[Union[str, "Expr", "pli.Series"]], "pli.Series"]
) -> List["PyExpr"]:
    if isinstance(exprs, (str, Expr, pli.Series)):
        exprs = [exprs]

    return [expr_to_lit_or_expr(e, str_to_lit=False)._pyexpr for e in exprs]


def wrap_expr(pyexpr: "PyExpr") -> "Expr":
    return Expr._from_pyexpr(pyexpr)


class Expr:
    """
    Expressions that can be used in various contexts.
    """

    def __init__(self) -> None:
        self._pyexpr: PyExpr  # pragma: no cover

    @staticmethod
    def _from_pyexpr(pyexpr: "PyExpr") -> "Expr":
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __to_pyexpr(self, other: Any) -> "PyExpr":
        return self.__to_expr(other)._pyexpr

    def __to_expr(self, other: Any) -> "Expr":
        if isinstance(other, Expr):
            return other
        return pli.lit(other)

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
        return pli.lit(0) - self

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> "Expr":
        """
        Numpy universal functions.
        """
        out_type = ufunc(np.array([1])).dtype
        dtype: Optional[Type[DataType]]
        if "float" in str(out_type):
            dtype = Float64
        else:
            dtype = None

        args = [inp for inp in inputs if not isinstance(inp, Expr)]

        def function(s: "pli.Series") -> "pli.Series":
            return ufunc(s, *args, **kwargs)  # pragma: no cover

        if "dtype" in kwargs:
            dtype = kwargs["dtype"]

        return self.map(function, return_dtype=dtype)

    def to_physical(self) -> "Expr":
        """
        Cast to physical representation of the logical dtype.

        Date -> Int32
        Datetime -> Int64
        Time -> Int64
        other -> other
        """
        return wrap_expr(self._pyexpr.to_physical())

    def any(self) -> "Expr":
        """
        Check if any boolean value in the column is `True`

        Returns
        -------
        Boolean literal
        """
        return wrap_expr(self._pyexpr.any())

    def all(self) -> "Expr":
        """
        Check if all boolean values in the column are `True`

        Returns
        -------
        Boolean literal
        """
        return wrap_expr(self._pyexpr.all())

    def sqrt(self) -> "Expr":
        """
        Compute the square root of the elements
        """
        return self ** 0.5

    def log(self) -> "Expr":
        """
        Natural logarithm, element-wise.

        The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
        The natural logarithm is logarithm in base e.
        """
        return np.log(self)  # type: ignore

    def log10(self) -> "Expr":
        """
        Return the base 10 logarithm of the input array, element-wise.
        """
        return np.log10(self)  # type: ignore

    def exp(self) -> "Expr":
        """
        Return the exponential element-wise
        """
        return np.exp(self)  # type: ignore

    def alias(self, name: str) -> "Expr":
        """
        Rename the output of an expression.

        Parameters
        ----------
        name
            New name.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["a", "b", None],
        ...     }
        ... )
        >>> df
        shape: (3, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ a    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ b    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null │
        └─────┴──────┘
        >>> df.select(
        ...     [
        ...         pl.col("a").alias("bar"),
        ...         pl.col("b").alias("foo"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────┬──────┐
        │ bar ┆ foo  │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ a    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ b    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null │
        └─────┴──────┘

        """
        return wrap_expr(self._pyexpr.alias(name))

    def exclude(
        self,
        columns: Union[str, List[str], Type[DataType], Sequence[Type[DataType]]],
    ) -> "Expr":
        """
        Exclude certain columns from a wildcard/regex selection.

        You may also use regexes in the exclude list. They must start with `^` and end with `$`.

        Parameters
        ----------
        columns
            Column(s) to exclude from selection.
            This can be:
            - a column name, or multiple names
            - a regular expression starting with `^` and ending with `$`
            - a dtype or multiple dtypes

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["a", "b", None],
        ...         "c": [None, 2, 1],
        ...     }
        ... )
        >>> df
        shape: (3, 3)
        ┌─────┬──────┬──────┐
        │ a   ┆ b    ┆ c    │
        │ --- ┆ ---  ┆ ---  │
        │ i64 ┆ str  ┆ i64  │
        ╞═════╪══════╪══════╡
        │ 1   ┆ a    ┆ null │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ b    ┆ 2    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null ┆ 1    │
        └─────┴──────┴──────┘
        >>> df.select(
        ...     pl.col("*").exclude("b"),
        ... )
        shape: (3, 2)
        ┌─────┬──────┐
        │ a   ┆ c    │
        │ --- ┆ ---  │
        │ i64 ┆ i64  │
        ╞═════╪══════╡
        │ 1   ┆ null │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ 2    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ 1    │
        └─────┴──────┘

        """
        if isinstance(columns, str):
            columns = [columns]
            return wrap_expr(self._pyexpr.exclude(columns))
        elif not isinstance(columns, list) and issubclass(columns, DataType):  # type: ignore
            columns = [columns]  # type: ignore
            return wrap_expr(self._pyexpr.exclude_dtype(columns))

        if not all(
            [
                isinstance(a, str) or issubclass(a, DataType)
                for a in columns  # type: ignore
            ]
        ):
            raise ValueError("input should be all string or all DataType")

        if isinstance(columns[0], str):  # type: ignore
            return wrap_expr(self._pyexpr.exclude(columns))
        else:
            return wrap_expr(self._pyexpr.exclude_dtype(columns))

    def keep_name(self) -> "Expr":
        """
        Keep the original root name of the expression.

        Examples
        --------

        A groupby aggregation often changes the name of a column.
        With `keep_name` we can keep the original name of the column

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["a", "b", None],
        ...     }
        ... )
        >>> df.groupby("a").agg(pl.col("b").list()).sort(by="a")
        shape: (3, 2)
        ┌─────┬────────────┐
        │ a   ┆ b_agg_list │
        │ --- ┆ ---        │
        │ i64 ┆ list [str] │
        ╞═════╪════════════╡
        │ 1   ┆ ["a"]      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ ["b"]      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]     │
        └─────┴────────────┘

        Keep the original column name:

        >>> df.groupby("a").agg(pl.col("b").list().keep_name()).sort(by="a")
        shape: (3, 2)
        ┌─────┬────────────┐
        │ a   ┆ b          │
        │ --- ┆ ---        │
        │ i64 ┆ list [str] │
        ╞═════╪════════════╡
        │ 1   ┆ ["a"]      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ ["b"]      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]     │
        └─────┴────────────┘

        """

        return wrap_expr(self._pyexpr.keep_name())

    def prefix(self, prefix: str) -> "Expr":
        """
        Add a prefix the to root column name of the expression.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...     }
        ... )
        >>> df
        shape: (5, 4)
        ┌─────┬────────┬─────┬────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   │
        │ --- ┆ ---    ┆ --- ┆ ---    │
        │ i64 ┆ str    ┆ i64 ┆ str    │
        ╞═════╪════════╪═════╪════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2   ┆ banana ┆ 4   ┆ audi   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 3   ┆ apple  ┆ 3   ┆ beetle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 4   ┆ apple  ┆ 2   ┆ beetle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 5   ┆ banana ┆ 1   ┆ beetle │
        └─────┴────────┴─────┴────────┘
        >>> df.select(
        ...     [
        ...         pl.all(),
        ...         pl.all().reverse().suffix("_reverse"),
        ...     ]
        ... )
        shape: (5, 8)
        ┌─────┬────────┬─────┬────────┬───────────┬────────────────┬───────────┬──────────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ A_reverse ┆ fruits_reverse ┆ B_reverse ┆ cars_reverse │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪════════╪═════╪════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 5         ┆ banana         ┆ 1         ┆ beetle       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.prefix(prefix))

    def suffix(self, suffix: str) -> "Expr":
        """
        Add a suffix the to root column name of the expression.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...     }
        ... )
        >>> df
        shape: (5, 4)
        ┌─────┬────────┬─────┬────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   │
        │ --- ┆ ---    ┆ --- ┆ ---    │
        │ i64 ┆ str    ┆ i64 ┆ str    │
        ╞═════╪════════╪═════╪════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2   ┆ banana ┆ 4   ┆ audi   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 3   ┆ apple  ┆ 3   ┆ beetle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 4   ┆ apple  ┆ 2   ┆ beetle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 5   ┆ banana ┆ 1   ┆ beetle │
        └─────┴────────┴─────┴────────┘
        >>> df.select(
        ...     [
        ...         pl.all(),
        ...         pl.all().reverse().prefix("reverse_"),
        ...     ]
        ... )
        shape: (5, 8)
        ┌─────┬────────┬─────┬────────┬───────────┬────────────────┬───────────┬──────────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ reverse_A ┆ reverse_fruits ┆ reverse_B ┆ reverse_cars │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪════════╪═════╪════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 5         ┆ banana         ┆ 1         ┆ beetle       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.suffix(suffix))

    def is_not(self) -> "Expr":
        """
        Negate a boolean expression.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [True, False, False],
        ...         "b": ["a", "b", None],
        ...     }
        ... )
        >>> df
        shape: (3, 2)
        ┌───────┬──────┐
        │ a     ┆ b    │
        │ ---   ┆ ---  │
        │ bool  ┆ str  │
        ╞═══════╪══════╡
        │ true  ┆ a    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ false ┆ b    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ false ┆ null │
        └───────┴──────┘
        >>> df.select(pl.col("a").is_not())
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        └───────┘

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

    def len(self) -> "Expr":
        """
        Alias for count
        Count the number of values in this expression
        """
        return self.count()

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

        >>> pl.col("foo").filter(pl.col("foo").is_not_null())  # doctest: +IGNORE_RESULT

        """
        return self.filter(self.is_not_null())

    def cumsum(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.
        """
        return wrap_expr(self._pyexpr.cumsum(reverse))

    def cumprod(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative product computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.
        """
        return wrap_expr(self._pyexpr.cumprod(reverse))

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

    def cumcount(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative count computed at every element.
        Counting from 0 to len

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cumcount(reverse))

    def floor(self) -> "Expr":
        """
        Floor underlying floating point array to the lowest integers smaller or equal to the float value.

        Only works on floating point Series
        """
        return wrap_expr(self._pyexpr.floor())

    def ceil(self) -> "Expr":
        """
        Ceil underlying floating point array to the heighest integers smaller or equal to the float value.

        Only works on floating point Series
        """
        return wrap_expr(self._pyexpr.ceil())

    def round(self, decimals: int) -> "Expr":
        """
        Round underlying floating point data by `decimals` digits.

        Parameters
        ----------
        decimals
            Number of decimals to round by.
        """
        return wrap_expr(self._pyexpr.round(decimals))

    def dot(self, other: Union["Expr", str]) -> "Expr":
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
        dtype
            DataType to cast to
        strict
            Throw an error if a cast could not be done for instance due to an overflow
        """
        dtype = py_type_to_dtype(dtype)
        return wrap_expr(self._pyexpr.cast(dtype, strict))

    def sort(self, reverse: bool = False, nulls_last: bool = False) -> "Expr":
        """
        Sort this column. In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        reverse
            False -> order from small to large.
            True -> order from large to small.
        nulls_last
            If True nulls are considered to be larger than any valid value
        """
        return wrap_expr(self._pyexpr.sort_with(reverse, nulls_last))

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

    def sort_by(
        self,
        by: Union["Expr", str, List[Union["Expr", str]]],
        reverse: Union[bool, List[bool]] = False,
    ) -> "Expr":
        """
        Sort this column by the ordering of another column, or multiple other columns.
        In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        by
            The column(s) used for sorting.
        reverse
            False -> order from small to large.
            True -> order from large to small.
        """
        if not isinstance(by, list):
            by = [by]
        if not isinstance(reverse, list):
            reverse = [reverse]
        by = selection_to_pyexpr_list(by)

        return wrap_expr(self._pyexpr.sort_by(by, reverse))

    def take(self, index: Union[List[int], "Expr", "pli.Series", np.ndarray]) -> "Expr":
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
            index_lit = pli.lit(pli.Series("", index, dtype=UInt32))
        else:
            index_lit = pli.expr_to_lit_or_expr(index, str_to_lit=False)
        return pli.wrap_expr(self._pyexpr.take(index_lit._pyexpr))

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

    def shift_and_fill(
        self, periods: int, fill_value: Union[int, float, bool, str, "Expr"]
    ) -> "Expr":
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

    def fill_null(self, fill_value: Union[int, float, bool, str, "Expr"]) -> "Expr":
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
        # we first must check if it is not an expr, as expr does not implement __bool__
        # and thus leads to a value error in the second comparisson.
        if not isinstance(fill_value, Expr) and fill_value in [
            "backward",
            "forward",
            "min",
            "max",
            "mean",
            "zero",
            "one",
        ]:
            return wrap_expr(self._pyexpr.fill_null_with_strategy(fill_value))

        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.fill_null(fill_value._pyexpr))

    def fill_nan(self, fill_value: Union[str, int, float, bool, "Expr"]) -> "Expr":
        """
        Fill floating point NaN value with a fill value
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

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.
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

    def product(self) -> "Expr":
        """
        Compute the product of an expression
        """
        return wrap_expr(self._pyexpr.product())

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

    def over(self, expr: Union[str, "Expr", List[Union["Expr", str]]]) -> "Expr":
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

        >>> df = pl.DataFrame(
        ...     {
        ...         "groups": [1, 1, 2, 2, 1, 2, 3, 3, 1],
        ...         "values": [1, 2, 3, 4, 5, 6, 7, 8, 8],
        ...     }
        ... )
        >>> (
        ...     df.lazy()
        ...     .select(
        ...         [
        ...             pl.col("groups").sum().over("groups"),
        ...         ]
        ...     )
        ...     .collect()
        ... )
        shape: (9, 1)
        ┌────────┐
        │ groups │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 4      │
        ├╌╌╌╌╌╌╌╌┤
        │ 4      │
        ├╌╌╌╌╌╌╌╌┤
        │ 6      │
        ├╌╌╌╌╌╌╌╌┤
        │ 6      │
        ├╌╌╌╌╌╌╌╌┤
        │ ...    │
        ├╌╌╌╌╌╌╌╌┤
        │ 6      │
        ├╌╌╌╌╌╌╌╌┤
        │ 6      │
        ├╌╌╌╌╌╌╌╌┤
        │ 6      │
        ├╌╌╌╌╌╌╌╌┤
        │ 4      │
        └────────┘

        """

        pyexprs = selection_to_pyexpr_list(expr)

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

    def quantile(self, quantile: float, interpolation: str = "nearest") -> "Expr":
        """
        Get quantile value.


        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']
        """
        return wrap_expr(self._pyexpr.quantile(quantile, interpolation))

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
        """
        Alias for filter

        Parameters
        ----------
        predicate
            Boolean expression.
        """
        return self.filter(predicate)

    def map(
        self,
        f: Callable[["pli.Series"], "pli.Series"],
        return_dtype: Optional[Type[DataType]] = None,
        agg_list: bool = False,
    ) -> "Expr":
        """
        Apply a custom python function. This function must produce a `Series`. Any other value will be stored as
        null/missing. If you want to apply a function over single values, consider using `apply`.

        [read more in the book](https://pola-rs.github.io/polars-book/user-guide/howcani/apply/udfs.html)

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        agg_list

        """
        if return_dtype is not None:
            return_dtype = py_type_to_dtype(return_dtype)
        return wrap_expr(self._pyexpr.map(f, return_dtype, agg_list))

    def apply(
        self,
        f: Union[Callable[["pli.Series"], "pli.Series"], Callable[[Any], Any]],
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

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 1, 1],
        ...         "b": ["a", "b", "c", "c"],
        ...     }
        ... )
        >>> (
        ...     df.lazy()
        ...     .groupby("b", maintain_order=True)
        ...     .agg(
        ...         [
        ...             pl.col("a").apply(lambda x: x.sum()),
        ...         ]
        ...     )
        ...     .collect()
        ... )
        shape: (3, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 2   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ c   ┆ 2   │
        └─────┴─────┘

        """

        # input x: Series of type list containing the group values
        def wrap_f(x: "pli.Series") -> "pli.Series":  # pragma: no cover
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

    def is_in(self, other: Union["Expr", List[Any]]) -> "Expr":
        """
        Check if elements of this Series are in the right Series, or List values of the right Series.

        Parameters
        ----------
        other
            Series of primitive type or List type.

        Returns
        -------
        Expr that evaluates to a Boolean Series.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"sets": [[1, 2, 3], [1, 2], [9, 10]], "optional_members": [1, 2, 3]}
        ... )
        >>> df.select([pl.col("optional_members").is_in("sets").alias("contains")])
        shape: (3, 1)
        ┌──────────┐
        │ contains │
        │ ---      │
        │ bool     │
        ╞══════════╡
        │ true     │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ true     │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ false    │
        └──────────┘

        """
        if isinstance(other, list):
            other = pli.lit(pli.Series(other))
        else:
            other = expr_to_lit_or_expr(other, str_to_lit=False)
        return wrap_expr(self._pyexpr.is_in(other._pyexpr))

    def repeat_by(self, by: Union["Expr", str]) -> "Expr":
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
        self,
        start: Union["Expr", datetime],
        end: Union["Expr", datetime],
        include_bounds: Union[bool, Sequence[bool]] = False,
    ) -> "Expr":
        """
        Check if this expression is between start and end.

        Parameters
        ----------
        start
            Lower bound as primitive type or datetime.
        end
            Upper bound as primitive type or datetime.
        include_bounds
           False:           Exclude both start and end (default).
           True:            Include both start and end.
           [False, False]:  Exclude start and exclude end.
           [True, True]:    Include start and include end.
           [False, True]:   Exclude start and include end.
           [True, False]:   Include start and exclude end.

        Returns
        -------
        Expr that evaluates to a Boolean Series.
        """
        cast_to_datetime = False
        if isinstance(start, datetime):
            start = pli.lit(start)
            cast_to_datetime = True
        if isinstance(end, datetime):
            end = pli.lit(end)
            cast_to_datetime = True
        if cast_to_datetime:
            expr = self.cast(Datetime)
        else:
            expr = self
        if include_bounds is False or include_bounds == [False, False]:
            return ((expr > start) & (expr < end)).alias("is_between")
        elif include_bounds is True or include_bounds == [True, True]:
            return ((expr >= start) & (expr <= end)).alias("is_between")
        elif include_bounds == [False, True]:
            return ((expr > start) & (expr <= end)).alias("is_between")
        elif include_bounds == [True, False]:
            return ((expr >= start) & (expr < end)).alias("is_between")
        else:
            raise ValueError(
                "include_bounds should be a boolean or [boolean, boolean]."
            )

    def hash(self, k0: int = 0, k1: int = 1, k2: int = 2, k3: int = 3) -> "Expr":
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

    def reinterpret(self, signed: bool) -> "Expr":
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

    def inspect(self, fmt: str = "{}") -> "Expr":
        """
        Prints the value that this expression evaluates to and passes on the value.

        >>> df = pl.DataFrame({"foo": [1, 1, 2]})
        >>> df.select(pl.col("foo").cumsum().inspect("value is: {}").alias("bar"))
        value is: shape: (3,)
        Series: 'foo' [i64]
        [
            1
            2
            4
        ]
        shape: (3, 1)
        ┌─────┐
        │ bar │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 4   │
        └─────┘

        """

        def inspect(s: "pli.Series") -> "pli.Series":  # pragma: no cover
            print(fmt.format(s))
            return s

        return self.map(inspect, return_dtype=None, agg_list=True)

    def interpolate(self) -> "Expr":
        """
        Interpolate intermediate values. The interpolation method is linear.
        """
        return wrap_expr(self._pyexpr.interpolate())

    def rolling_min(
        self,
        window_size: int,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        apply a rolling min (moving min) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Apply a rolling max (moving max) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Apply a rolling mean (moving mean) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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

        >>> df = pl.DataFrame({"A": [1.0, 8.0, 6.0, 2.0, 16.0, 10.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_mean(window_size=2),
        ...     ]
        ... )
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Apply a rolling sum (moving sum) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Compute a rolling std dev

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        self,
        function: Callable[["pli.Series"], Any],
        window_size: int,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Allows a custom rolling window function.
        Prefer the specific rolling window functions over this one, as they are faster.

        Prefer:

            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum

        Parameters
        ----------
        function
            Aggregation function
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
        ...     {
        ...         "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_apply(lambda s: s.std(), window_size=3),
        ...     ]
        ... )
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
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_apply(
                function, window_size, weights, min_periods, center
            )
        )

    def rolling_median(
        self,
        window_size: int,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Compute a rolling median

        Parameters
        ----------
        window_size
            Size of the rolling window
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        """
        return wrap_expr(
            self._pyexpr.rolling_median(window_size, weights, min_periods, center)
        )

    def rolling_quantile(
        self,
        quantile: float,
        interpolation: str = "nearest",
        window_size: int = 2,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Expr":
        """
        Compute a rolling quantile

        Parameters
        ----------
        quantile
            quantile to compute
        interpolation
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']
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
        return wrap_expr(
            self._pyexpr.rolling_quantile(
                quantile, interpolation, window_size, weights, min_periods, center
            )
        )

    def rolling_skew(self, window_size: int, bias: bool = True) -> "Expr":
        """
        Compute a rolling skew

        Parameters
        ----------
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
        return wrap_expr(self._pyexpr.abs())

    def argsort(self, reverse: bool = False) -> "Expr":
        """
        Index location of the sorted variant of this Series.

        Parameters
        ----------
        reverse
            Reverse the ordering. Default is from low to high.
        """
        return pli.argsort_by([self], [reverse])

    def rank(self, method: str = "average", reverse: bool = False) -> "Expr":
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method
            {'average', 'min', 'max', 'dense', 'ordinal', 'random'}, optional
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):
            - 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
            - 'min': The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
            - 'max': The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
            - 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
            - 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
            - 'random': Like 'ordinal', but the rank for ties is not dependent
            on the order that the values occur in `a`.
        reverse
            reverse the operation
        """
        return wrap_expr(self._pyexpr.rank(method, reverse))

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> "Expr":
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

    def pct_change(self, n: int = 1) -> "Expr":
        """
        Percentage change (as fraction) between current element and most-recent
        non-null element at least n period(s) before the current element.

        Computes the change from the previous row by default.

        Parameters
        ----------
        n
            periods to shift for forming percent change.
        """
        return wrap_expr(self._pyexpr.pct_change(n))

    def skew(self, bias: bool = True) -> "Expr":
        r"""Compute the sample skewness of a data set.
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

        .. math:: g_1=\frac{m_3}{m_2^{3/2}}

        where

        .. math:: m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math:: G_1=\frac{k_3}{k_2^{3/2}}= \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}

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
        Clip (limit) the values in an array to any value that fits in 64 floating poitns range.

        Only works for the following dtypes: {Int32, Int64, Float32, Float64, UInt32}.

        If you want to clip other dtypes, consider writing a when -> then -> otherwise expression

        Parameters
        ----------
        min_val
            Minimum value.
        max_val
            Maximum value.
        """
        return wrap_expr(self._pyexpr.clip(min_val, max_val))

    def lower_bound(self) -> "Expr":
        """
        Returns a unit Series with the lowest value possible for the dtype of this expression.
        """
        return wrap_expr(self._pyexpr.lower_bound())

    def upper_bound(self) -> "Expr":
        """
        Returns a unit Series with the highest value possible for the dtype of this expression.
        """
        return wrap_expr(self._pyexpr.upper_bound())

    def str_concat(self, delimiter: str = "-") -> "Expr":
        """
        Vertically concat the values in the Series to a single string value.

        Returns
        -------
        Series of dtype Utf8

        Examples
        --------

        >>> df = pl.DataFrame({"foo": [1, None, 2]})
        >>> df = df.select(pl.col("foo").str_concat("-"))
        >>> df
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

    def sin(self) -> "Expr":
        """
        Compute the element-wise value for Trigonometric sine on an array

        Returns
        -------
        Series of dtype Float64

        Examples
        --------

        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").sin())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
        return np.sin(self)  # type: ignore

    def cos(self) -> "Expr":
        """
        Compute the element-wise value for Trigonometric cosine on an array

        Returns
        -------
        Series of dtype Float64

        Examples
        --------

        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").cos())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1   │
        └─────┘

        """
        return np.cos(self)  # type: ignore

    def tan(self) -> "Expr":
        """
        Compute the element-wise value for Trigonometric tangent on an array

        Returns
        -------
        Series of dtype Float64

        Examples
        --------

        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").tan().round(2))
        shape: (1, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ 1.56 │
        └──────┘

        """
        return np.tan(self)  # type: ignore

    def arcsin(self) -> "Expr":
        """
        Compute the element-wise value for Trigonometric sine on an array

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arcsin())
        shape: (1, 1)
        ┌────────────────────┐
        │ a                  │
        │ ---                │
        │ f64                │
        ╞════════════════════╡
        │ 1.5707963267948966 │
        └────────────────────┘

        """
        return np.arcsin(self)  # type: ignore

    def arccos(self) -> "Expr":
        """
        Compute the element-wise value for Trigonometric cosine on an array

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").arccos())
        shape: (1, 1)
        ┌────────────────────┐
        │ a                  │
        │ ---                │
        │ f64                │
        ╞════════════════════╡
        │ 1.5707963267948966 │
        └────────────────────┘

        """
        return np.arccos(self)  # type: ignore

    def arctan(self) -> "Expr":
        """
        Compute the element-wise value for Trigonometric tangent on an array

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arctan())
        shape: (1, 1)
        ┌────────────────────┐
        │ a                  │
        │ ---                │
        │ f64                │
        ╞════════════════════╡
        │ 0.7853981633974483 │
        └────────────────────┘

        """
        return np.arctan(self)  # type: ignore

    def reshape(self, dims: Tuple[int, ...]) -> "Expr":
        """
        Reshape this Expr to a flat series, shape: (len,)
        or a List series, shape: (rows, cols)

        if a -1 is used in any of the dimensions, that dimension is inferred.

        Parameters
        ----------
        dims
            Tuple of the dimension sizes

        Returns
        -------
        Expr
        """
        return wrap_expr(self._pyexpr.reshape(dims))

    def shuffle(self, seed: int = 0) -> "Expr":
        """
        Shuffle the contents of this expr.

        Parameters
        ----------
        seed
            Seed initialization
        """
        return wrap_expr(self._pyexpr.shuffle(seed))

    def ewm_mean(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        half_life: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> "Expr":
        r"""
        Exponential moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span, :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life, :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value (otherwise result is Null).

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(self._pyexpr.ewm_mean(alpha, adjust, min_periods))

    def ewm_std(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        half_life: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> "Expr":
        r"""
        Exponential moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span, :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life, :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value (otherwise result is Null).

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(self._pyexpr.ewm_std(alpha, adjust, min_periods))

    def ewm_var(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        half_life: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> "Expr":
        r"""
        Exponential moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span, :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life, :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value (otherwise result is Null).

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(self._pyexpr.ewm_var(alpha, adjust, min_periods))

    def extend(self, value: Optional[Union[int, float, str, bool]], n: int) -> "Expr":
        """
        Extend the Series with given number of values.

        Parameters
        ----------
        value
            The value to extend the Series with. This value may be None to fill with nulls.
        n
            The number of values to extend.
        """
        return wrap_expr(self._pyexpr.extend(value, n))

    # Below are the namespaces defined. Keep these at the end of the definition of Expr, as to not confuse mypy with
    # the type annotation `str` with the namespace "str"

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

    def sort(self, reverse: bool = False) -> "Expr":
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

    def concat(
        self, other: Union[List[Union[Expr, str]], Expr, str, "pli.Series", List[Any]]
    ) -> "Expr":
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series
        """
        if isinstance(other, list) and (
            not isinstance(other[0], (Expr, str, pli.Series))
        ):
            return self.concat(pli.Series([other]))

        other_list: List[Union[Expr, str, "pli.Series"]]
        if not isinstance(other, list):
            other_list = [other]
        else:
            other_list = copy.copy(other)  # type: ignore

        other_list.insert(0, wrap_expr(self._pyexpr))
        return pli.concat_list(other_list)

    def get(self, index: int) -> "Expr":
        """
        Get the value by index in the sublists.
        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sublist
        """
        return wrap_expr(self._pyexpr.lst_get(index))

    def first(self) -> "Expr":
        """
        Get the first value of the sublists.
        """
        return self.get(0)

    def last(self) -> "Expr":
        """
        Get the last value of the sublists.
        """
        return self.get(-1)

    def contains(self, item: Union[float, str, bool, int, date, datetime]) -> "Expr":
        """
        Check if sublists contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Boolean mask
        """
        return wrap_expr(self._pyexpr).map(lambda s: s.arr.contains(item))


class ExprStringNameSpace:
    """
    Namespace for string related expressions
    """

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def strptime(
        self,
        datatype: Union[Type[Date], Type[Datetime]],
        fmt: Optional[str] = None,
        strict: bool = True,
        exact: bool = True,
    ) -> Expr:
        """
        Parse utf8 expression as a Date/Datetimetype.

        Parameters
        ----------
        datatype
            Date | Datetime.
        fmt
            format to use, see the following link for examples:
            https://docs.rs/chrono/latest/chrono/format/strftime/index.html

            example: "%y-%m-%d".
        strict
            raise an error if any conversion fails
        exact
            - If True, require an exact format match.
            - If False, allow the format to match anywhere in the target string.
        """
        if not issubclass(datatype, DataType):
            raise ValueError(
                f"expected: {DataType} got: {datatype}"
            )  # pragma: no cover
        if datatype == Date:
            return wrap_expr(self._pyexpr.str_parse_date(fmt, strict, exact))
        elif datatype == Datetime:
            return wrap_expr(self._pyexpr.str_parse_datetime(fmt, strict, exact))
        else:
            raise ValueError(
                "dtype should be of type {Date, Datetime}"
            )  # pragma: no cover

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

    def strip(self) -> Expr:
        """
        Remove leading and trailing whitespace.
        """
        return wrap_expr(self._pyexpr.str_strip())

    def lstrip(self) -> Expr:
        """
        Remove leading whitespace.
        """
        return wrap_expr(self._pyexpr.str_lstrip())

    def rstrip(self) -> Expr:
        """
        Remove trailing whitespace.
        """
        return wrap_expr(self._pyexpr.str_rstrip())

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

        >>> df = pl.DataFrame(
        ...     {"json_val": ['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}']}
        ... )
        >>> df.select(pl.col("json_val").str.json_path_match("$.a"))
        shape: (5, 1)
        ┌──────────┐
        │ json_val │
        │ ---      │
        │ str      │
        ╞══════════╡
        │ 1        │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ null     │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 2        │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 2.1      │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ true     │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.str_json_path_match(json_path))

    def decode(self, encoding: str, strict: bool = False) -> Expr:
        """
        Decodes a value using the provided encoding

        Parameters
        ----------
        encoding
            'hex' or 'base64'
        strict
            how to handle invalid inputs
            - True: method will throw error if unable to decode a value
            - False: unhandled values will be replaced with `None`

        Examples
        --------
        >>> df = pl.DataFrame({"encoded": ["666f6f", "626172", None]})
        >>> df.select(pl.col("encoded").str.decode("hex"))
        shape: (3, 1)
        ┌─────────┐
        │ encoded │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ foo     │
        ├╌╌╌╌╌╌╌╌╌┤
        │ bar     │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        └─────────┘
        """
        if encoding == "hex":
            return wrap_expr(self._pyexpr.str_hex_decode(strict))
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.str_base64_decode(strict))
        else:
            raise ValueError("supported encodings are 'hex' and 'base64'")

    def encode(self, encoding: str) -> Expr:
        """
        Encodes a value using the provided encoding

        Parameters
        ----------
        encoding
            'hex' or 'base64'

        Returns
        -------
        Utf8 array with values encoded using provided encoding

        Examples
        --------
        >>> df = pl.DataFrame({"strings": ["foo", "bar", None]})
        >>> df.select(pl.col("strings").str.encode("hex"))
        shape: (3, 1)
        ┌─────────┐
        │ strings │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 666f6f  │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 626172  │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        └─────────┘
        """
        if encoding == "hex":
            return wrap_expr(self._pyexpr.str_hex_encode())
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.str_base64_encode())
        else:
            raise ValueError("supported encodings are 'hex' and 'base64'")

    def extract(self, pattern: str, group_index: int = 1) -> Expr:
        r"""
        Extract the target capture group from provided patterns.

        Parameters
        ----------
        pattern
            A valid regex pattern
        group_index
            Index of the targeted capture group.
            Group 0 mean the whole pattern, first group begin at index 1
            Default to the first capture group

        Returns
        -------
        Utf8 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [
        ...             "http://vote.com/ballon_dor?candidate=messi&ref=polars",
        ...             "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
        ...             "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ...         ]
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("a").str.extract(r"candidate=(\w+)", 1),
        ...     ]
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ messi   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        ├╌╌╌╌╌╌╌╌╌┤
        │ ronaldo │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.str_extract(pattern, group_index))

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

    def truncate(
        self,
        every: Union[str, timedelta],
        offset: Optional[Union[str, timedelta]] = None,
    ) -> Expr:
        """
        .. warning::
            This API is experimental and may change without it being considered a breaking change.

        Divide the date/ datetime range into buckets.
        Data must be sorted, if not the output does not make sense.

        The `every` and `offset` arguments are created with
        the following string language:

        1ns # 1 nanosecond
        1us # 1 microsecond
        1ms # 1 millisecond
        1s  # 1 second
        1m  # 1 minute
        1h  # 1 hour
        1d  # 1 day
        1w  # 1 week
        1mo # 1 calendar month
        1y  # 1 calendar year

        3d12h4m25s # 3 days, 12 hours, 4 minutes, and 25 seconds

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

        Returns
        -------
        Date/Datetime series

        Examples
        --------

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> s = pl.date_range(start, stop, timedelta(minutes=30), name="dates")
        >>> s
        shape: (49,)
        Series: 'dates' [datetime[ns]]
        [
            2001-01-01 00:00:00
            2001-01-01 00:30:00
            2001-01-01 01:00:00
            2001-01-01 01:30:00
            2001-01-01 02:00:00
            2001-01-01 02:30:00
            2001-01-01 03:00:00
            2001-01-01 03:30:00
            2001-01-01 04:00:00
            2001-01-01 04:30:00
            2001-01-01 05:00:00
            2001-01-01 05:30:00
            ...
            2001-01-01 18:30:00
            2001-01-01 19:00:00
            2001-01-01 19:30:00
            2001-01-01 20:00:00
            2001-01-01 20:30:00
            2001-01-01 21:00:00
            2001-01-01 21:30:00
            2001-01-01 22:00:00
            2001-01-01 22:30:00
            2001-01-01 23:00:00
            2001-01-01 23:30:00
            2001-01-02 00:00:00
        ]
        >>> s.dt.truncate("1h")
        shape: (49,)
        Series: 'dates' [datetime[ns]]
        [
            2001-01-01 00:00:00
            2001-01-01 00:00:00
            2001-01-01 01:00:00
            2001-01-01 01:00:00
            2001-01-01 02:00:00
            2001-01-01 02:00:00
            2001-01-01 03:00:00
            2001-01-01 03:00:00
            2001-01-01 04:00:00
            2001-01-01 04:00:00
            2001-01-01 05:00:00
            2001-01-01 05:00:00
            ...
            2001-01-01 18:00:00
            2001-01-01 19:00:00
            2001-01-01 19:00:00
            2001-01-01 20:00:00
            2001-01-01 20:00:00
            2001-01-01 21:00:00
            2001-01-01 21:00:00
            2001-01-01 22:00:00
            2001-01-01 22:00:00
            2001-01-01 23:00:00
            2001-01-01 23:00:00
            2001-01-02 00:00:00
        ]
        >>> assert s.dt.truncate("1h") == s.dt.truncate(timedelta(hours=1))

        """
        if offset is None:
            offset = "0ns"
        if isinstance(every, timedelta):
            every = _timedelta_to_pl_duration(every)
        if isinstance(offset, timedelta):
            offset = _timedelta_to_pl_duration(offset)
        return wrap_expr(self._pyexpr.date_truncate(every, offset))

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

    def to_python_datetime(self) -> Expr:
        """
        Go from Date/Datetime to python DateTime objects
        """
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.to_python_datetime(), return_dtype=Object
        )

    def epoch_days(self) -> Expr:
        """
        Get the number of days since the unix EPOCH.
        If the date is before the unix EPOCH, the number of days will be negative.

        Returns
        -------
        Days as Int32
        """
        return wrap_expr(self._pyexpr).cast(Date).cast(Int32)

    def epoch_milliseconds(self) -> Expr:
        """
        Get the number of milliseconds since the unix EPOCH
        If the date is before the unix EPOCH, the number of milliseconds will be negative.

        Returns
        -------
        Milliseconds as Int64
        """
        return self.timestamp()

    def epoch_seconds(self) -> Expr:
        """
        Get the number of seconds since the unix EPOCH
        If the date is before the unix EPOCH, the number of seconds will be negative.

        Returns
        -------
        Milliseconds as Int64
        """
        return wrap_expr(self._pyexpr.dt_epoch_seconds())

    def timestamp(self) -> Expr:
        """Return timestamp in milliseconds as Int64 type."""
        return wrap_expr(self._pyexpr.timestamp())

    def and_time_unit(self, tu: str, dtype: Type[DataType] = Datetime) -> Expr:
        """
        Set time unit a Series of type Datetime

        Parameters
        ----------
        tu
            Time unit for the `Datetime` Series: any of {"ns", "ms"}
        dtype
            Output data type.
        """
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.and_time_unit(tu), return_dtype=dtype
        )

    def and_time_zone(self, tz: Optional[str]) -> Expr:
        """
        Set time zone a Series of type Datetime

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series: any of {"ns", "ms"}

        """
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.and_time_zone(tz), return_dtype=Datetime
        )

    def days(self) -> Expr:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return wrap_expr(self._pyexpr.duration_days())

    def hours(self) -> Expr:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return wrap_expr(self._pyexpr.duration_hours())

    def seconds(self) -> Expr:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return wrap_expr(self._pyexpr.duration_seconds())

    def milliseconds(self) -> Expr:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return wrap_expr(self._pyexpr.duration_milliseconds())

    def nanoseconds(self) -> Expr:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return wrap_expr(self._pyexpr.duration_nanoseconds())


def expr_to_lit_or_expr(
    expr: Union[Expr, bool, int, float, str, "pli.Series"],
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
        return pli.col(expr)
    elif (
        isinstance(expr, (int, float, str, pli.Series, datetime, date)) or expr is None
    ):
        return pli.lit(expr)
    elif isinstance(expr, Expr):
        return expr
    else:
        raise ValueError(
            f"did not expect value {expr} of type {type(expr)}, maybe disambiguate with pl.lit or pl.col"
        )


def _prepare_alpha(
    com: Optional[float] = None,
    span: Optional[float] = None,
    half_life: Optional[float] = None,
    alpha: Optional[float] = None,
) -> float:
    if com is not None and alpha is None:
        assert com >= 0.0
        alpha = 1.0 / (1.0 + com)
    if span is not None and alpha is None:
        assert span >= 1.0
        alpha = 2.0 / (span + 1.0)
    if half_life is not None and alpha is None:
        assert half_life > 0.0
        alpha = 1.0 - np.exp(-np.log(2.0) / half_life)
    if alpha is None:
        raise ValueError("at least one of {com, span, half_life, alpha} should be set")
    return alpha
