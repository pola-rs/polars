from __future__ import annotations

import copy
import math
import random
import sys
from datetime import date, datetime, timedelta
from typing import Any, Callable, List, Sequence

from polars import internals as pli
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    DataType,
    Date,
    Datetime,
    Float64,
    Int32,
    Object,
    Time,
    UInt32,
    py_type_to_dtype,
)
from polars.utils import _timedelta_to_pl_duration

try:
    from polars.polars import PyExpr

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


def selection_to_pyexpr_list(
    exprs: str | Expr | Sequence[str | Expr | pli.Series] | pli.Series,
) -> List[PyExpr]:
    if isinstance(exprs, (str, Expr, pli.Series)):
        exprs = [exprs]

    return [expr_to_lit_or_expr(e, str_to_lit=False)._pyexpr for e in exprs]


def wrap_expr(pyexpr: PyExpr) -> Expr:
    return Expr._from_pyexpr(pyexpr)


class Expr:
    """Expressions that can be used in various contexts."""

    def __init__(self) -> None:
        self._pyexpr: PyExpr  # pragma: no cover

    def __str__(self) -> str:
        return self._pyexpr.to_str()

    def _repr_html_(self) -> str:
        return self._pyexpr.to_str()

    @staticmethod
    def _from_pyexpr(pyexpr: PyExpr) -> Expr:
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __to_pyexpr(self, other: Any) -> PyExpr:
        return self.__to_expr(other)._pyexpr

    def __to_expr(self, other: Any) -> Expr:
        if isinstance(other, Expr):
            return other
        return pli.lit(other)

    def __bool__(self) -> Expr:
        raise ValueError(
            "Since Expr are lazy, the truthiness of an Expr is ambiguous. "
            "Hint: use '&' or '|' to chain Expr together, not and/or."
        )

    def __invert__(self) -> Expr:
        return self.is_not()

    def __xor__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._xor(self.__to_pyexpr(other)))

    def __rxor__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._xor(self.__to_pyexpr(other)))

    def __and__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._and(self.__to_pyexpr(other)))

    def __rand__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr._and(self.__to_pyexpr(other)))

    def __or__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._or(self.__to_pyexpr(other)))

    def __ror__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other)._or(self._pyexpr))

    def __add__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr + self.__to_pyexpr(other))

    def __radd__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other) + self._pyexpr)

    def __sub__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr - self.__to_pyexpr(other))

    def __rsub__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other) - self._pyexpr)

    def __mul__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr * self.__to_pyexpr(other))

    def __rmul__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other) * self._pyexpr)

    def __truediv__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr / self.__to_pyexpr(other))

    def __rtruediv__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other) / self._pyexpr)

    def __floordiv__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr // self.__to_pyexpr(other))

    def __rfloordiv__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other) // self._pyexpr)

    def __mod__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr % self.__to_pyexpr(other))

    def __rmod__(self, other: Any) -> Expr:
        return wrap_expr(self.__to_pyexpr(other) % self._pyexpr)

    def __pow__(self, power: int | float | pli.Series | Expr) -> Expr:
        return self.pow(power)

    def __rpow__(self, base: int | float | Expr) -> Expr:
        return pli.expr_to_lit_or_expr(base) ** self

    def __ge__(self, other: Any) -> Expr:
        return self.gt_eq(self.__to_expr(other))

    def __le__(self, other: Any) -> Expr:
        return self.lt_eq(self.__to_expr(other))

    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return self.eq(self.__to_expr(other))

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return self.neq(self.__to_expr(other))

    def __lt__(self, other: Any) -> Expr:
        return self.lt(self.__to_expr(other))

    def __gt__(self, other: Any) -> Expr:
        return self.gt(self.__to_expr(other))

    def eq(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr.eq(other._pyexpr))

    def neq(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr.neq(other._pyexpr))

    def gt(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr.gt(other._pyexpr))

    def gt_eq(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr.gt_eq(other._pyexpr))

    def lt_eq(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr.lt_eq(other._pyexpr))

    def lt(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr.lt(other._pyexpr))

    def __neg__(self) -> Expr:
        return pli.lit(0) - self

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Expr:
        """Numpy universal functions."""
        if not _NUMPY_AVAILABLE:
            raise ImportError("'numpy' is required for this functionality.")
        out_type = ufunc(np.array([1])).dtype
        dtype: type[DataType] | None
        if "float" in str(out_type):
            dtype = Float64
        else:
            dtype = None

        args = [inp for inp in inputs if not isinstance(inp, Expr)]

        def function(s: pli.Series) -> pli.Series:  # pragma: no cover
            return ufunc(s, *args, **kwargs)

        dtype = kwargs.get("dtype", dtype)

        return self.map(function, return_dtype=dtype)

    def __getstate__(self) -> Any:
        return self._pyexpr.__getstate__()

    def __setstate__(self, state: Any) -> None:
        # init with a dummy
        self._pyexpr = pli.lit(0)._pyexpr
        self._pyexpr.__setstate__(state)

    def to_physical(self) -> Expr:
        """
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - Other data types will be left unchanged.

        Examples
        --------
        Replicating the pandas
        `pd.factorize
        <https://pandas.pydata.org/docs/reference/api/pandas.factorize.html>`_
        function.

        >>> pl.DataFrame({"vals": ["a", "x", None, "a"]}).with_columns(
        ...     [
        ...         pl.col("vals").cast(pl.Categorical),
        ...         pl.col("vals")
        ...         .cast(pl.Categorical)
        ...         .to_physical()
        ...         .alias("vals_physical"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌──────┬───────────────┐
        │ vals ┆ vals_physical │
        │ ---  ┆ ---           │
        │ cat  ┆ u32           │
        ╞══════╪═══════════════╡
        │ a    ┆ 0             │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ x    ┆ 1             │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ null          │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a    ┆ 0             │
        └──────┴───────────────┘

        """
        return wrap_expr(self._pyexpr.to_physical())

    def any(self) -> Expr:
        """
        Check if any boolean value in a Boolean column is `True`.

        Returns
        -------
        Boolean literal

        Examples
        --------
        >>> df = pl.DataFrame({"TF": [True, False], "FF": [False, False]})
        >>> df.select(pl.all().any())
        shape: (1, 2)
        ┌──────┬───────┐
        │ TF   ┆ FF    │
        │ ---  ┆ ---   │
        │ bool ┆ bool  │
        ╞══════╪═══════╡
        │ true ┆ false │
        └──────┴───────┘

        """
        return wrap_expr(self._pyexpr.any())

    def all(self) -> Expr:
        """
        Check if all boolean values in a Boolean column are `True`.

        This method is an expression - not to be confused with
        :func:`polars.all` which is a function to select all columns.

        Returns
        -------
        Boolean literal

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"TT": [True, True], "TF": [True, False], "FF": [False, False]}
        ... )
        >>> df.select(pl.col("*").all())
        shape: (1, 3)
        ┌──────┬───────┬───────┐
        │ TT   ┆ TF    ┆ FF    │
        │ ---  ┆ ---   ┆ ---   │
        │ bool ┆ bool  ┆ bool  │
        ╞══════╪═══════╪═══════╡
        │ true ┆ false ┆ false │
        └──────┴───────┴───────┘

        """
        return wrap_expr(self._pyexpr.all())

    def sqrt(self) -> Expr:
        """Compute the square root of the elements."""
        return self**0.5

    def log10(self) -> Expr:
        """Compute the base 10 logarithm of the input array, element-wise."""
        return self.log(10.0)

    def exp(self) -> Expr:
        """Compute the exponential, element-wise."""
        return wrap_expr(self._pyexpr.exp())

    def alias(self, name: str) -> Expr:
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
        columns: (
            str
            | List[str]
            | DataType
            | type[DataType]
            | DataType
            | Sequence[DataType | type[DataType]]
        ),
    ) -> Expr:
        """
        Exclude certain columns from a wildcard/regex selection.

        You may also use regexes in the exclude list. They must start with `^` and end
        with `$`.

        Parameters
        ----------
        columns
            Column(s) to exclude from selection.
            This can be:

            - a column name, or multiple column names
            - a regular expression starting with `^` and ending with `$`
            - a dtype or multiple dtypes

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "aa": [1, 2, 3],
        ...         "ba": ["a", "b", None],
        ...         "cc": [None, 2.5, 1.5],
        ...     }
        ... )
        >>> df
        shape: (3, 3)
        ┌─────┬──────┬──────┐
        │ aa  ┆ ba   ┆ cc   │
        │ --- ┆ ---  ┆ ---  │
        │ i64 ┆ str  ┆ f64  │
        ╞═════╪══════╪══════╡
        │ 1   ┆ a    ┆ null │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ b    ┆ 2.5  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null ┆ 1.5  │
        └─────┴──────┴──────┘

        Exclude by column name(s):

        >>> df.select(pl.all().exclude("ba"))
        shape: (3, 2)
        ┌─────┬──────┐
        │ aa  ┆ cc   │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ null │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ 2.5  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ 1.5  │
        └─────┴──────┘

        Exclude by regex, e.g. removing all columns whose names end with the letter "a":

        >>> df.select(pl.all().exclude("^.*a$"))
        shape: (3, 1)
        ┌──────┐
        │ cc   │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 2.5  │
        ├╌╌╌╌╌╌┤
        │ 1.5  │
        └──────┘

        Exclude by dtype(s), e.g. removing all columns of type Int64 or Float64:

        >>> df.select(pl.all().exclude([pl.Int64, pl.Float64]))
        shape: (3, 1)
        ┌──────┐
        │ ba   │
        │ ---  │
        │ str  │
        ╞══════╡
        │ a    │
        ├╌╌╌╌╌╌┤
        │ b    │
        ├╌╌╌╌╌╌┤
        │ null │
        └──────┘

        """
        if isinstance(columns, str):
            columns = [columns]
            return wrap_expr(self._pyexpr.exclude(columns))
        elif not isinstance(columns, Sequence) or isinstance(columns, DataType):
            columns = [columns]
            return wrap_expr(self._pyexpr.exclude_dtype(columns))

        if not all(
            [
                isinstance(a, str) or (type(a) is type and issubclass(a, DataType))
                for a in columns
            ]
        ):
            raise ValueError("input should be all string or all DataType")

        if isinstance(columns[0], str):
            return wrap_expr(self._pyexpr.exclude(columns))
        else:
            return wrap_expr(self._pyexpr.exclude_dtype(columns))

    def keep_name(self) -> Expr:
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
        ┌─────┬───────────┐
        │ a   ┆ b         │
        │ --- ┆ ---       │
        │ i64 ┆ list[str] │
        ╞═════╪═══════════╡
        │ 1   ┆ ["a"]     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ ["b"]     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]    │
        └─────┴───────────┘

        Keep the original column name:

        >>> df.groupby("a").agg(pl.col("b").list().keep_name()).sort(by="a")
        shape: (3, 2)
        ┌─────┬───────────┐
        │ a   ┆ b         │
        │ --- ┆ ---       │
        │ i64 ┆ list[str] │
        ╞═════╪═══════════╡
        │ 1   ┆ ["a"]     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ ["b"]     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]    │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.keep_name())

    def prefix(self, prefix: str) -> Expr:
        """
        Add a prefix to the root column name of the expression.

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

        """  # noqa: E501
        return wrap_expr(self._pyexpr.prefix(prefix))

    def suffix(self, suffix: str) -> Expr:
        """
        Add a suffix to the root column name of the expression.

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

        """  # noqa: E501
        return wrap_expr(self._pyexpr.suffix(suffix))

    def map_alias(self, f: Callable[[str], str]) -> Expr:
        """
        Rename the output of an expression by mapping a function over the root name.

        Parameters
        ----------
        f
            Function that maps root name to new name.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2],
        ...         "B": [3, 4],
        ...     }
        ... )
        >>> df.select(
        ...     pl.all().reverse().map_alias(lambda colName: colName + "_reverse")
        ... )
        shape: (2, 2)
        ┌───────────┬───────────┐
        │ A_reverse ┆ B_reverse │
        │ ---       ┆ ---       │
        │ i64       ┆ i64       │
        ╞═══════════╪═══════════╡
        │ 2         ┆ 4         │
        ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1         ┆ 3         │
        └───────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.map_alias(f))

    def is_not(self) -> Expr:
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

    def is_null(self) -> Expr:
        """
        Create a boolean expression returning `True` where the expression contains null
        values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_column(pl.all().is_null().suffix("_isnull"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬──────────┬──────────┐
        │ a    ┆ b   ┆ a_isnull ┆ b_isnull │
        │ ---  ┆ --- ┆ ---      ┆ ---      │
        │ i64  ┆ f64 ┆ bool     ┆ bool     │
        ╞══════╪═════╪══════════╪══════════╡
        │ 1    ┆ 1.0 ┆ false    ┆ false    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2    ┆ 2.0 ┆ false    ┆ false    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ NaN ┆ true     ┆ false    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 1.0 ┆ false    ┆ false    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 5    ┆ 5.0 ┆ false    ┆ false    │
        └──────┴─────┴──────────┴──────────┘

        """
        return wrap_expr(self._pyexpr.is_null())

    def is_not_null(self) -> Expr:
        """
        Create a boolean expression returning `True` where the expression does not
        contain null values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_column(pl.all().is_not_null().suffix("_not_null"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬────────────┬────────────┐
        │ a    ┆ b   ┆ a_not_null ┆ b_not_null │
        │ ---  ┆ --- ┆ ---        ┆ ---        │
        │ i64  ┆ f64 ┆ bool       ┆ bool       │
        ╞══════╪═════╪════════════╪════════════╡
        │ 1    ┆ 1.0 ┆ true       ┆ true       │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2    ┆ 2.0 ┆ true       ┆ true       │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ NaN ┆ false      ┆ true       │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 1.0 ┆ true       ┆ true       │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5    ┆ 5.0 ┆ true       ┆ true       │
        └──────┴─────┴────────────┴────────────┘

        """
        return wrap_expr(self._pyexpr.is_not_null())

    def is_finite(self) -> Expr:
        """
        Create a boolean expression returning `True` where the expression values are
        finite.

        Returns
        -------
        out
            Series of type Boolean

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2],
        ...         "B": [3.0, np.inf],
        ...     }
        ... )
        >>> df.select(pl.all().is_finite())
        shape: (2, 2)
        ┌──────┬───────┐
        │ A    ┆ B     │
        │ ---  ┆ ---   │
        │ bool ┆ bool  │
        ╞══════╪═══════╡
        │ true ┆ true  │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ true ┆ false │
        └──────┴───────┘

        """
        return wrap_expr(self._pyexpr.is_finite())

    def is_infinite(self) -> Expr:
        """
        Create a boolean expression returning `True` where the expression values are
        infinite.

        Returns
        -------
        out
            Series of type Boolean

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2],
        ...         "B": [3.0, np.inf],
        ...     }
        ... )
        >>> df.select(pl.all().is_infinite())
        shape: (2, 2)
        ┌───────┬───────┐
        │ A     ┆ B     │
        │ ---   ┆ ---   │
        │ bool  ┆ bool  │
        ╞═══════╪═══════╡
        │ false ┆ false │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ false ┆ true  │
        └───────┴───────┘

        """
        return wrap_expr(self._pyexpr.is_infinite())

    def is_nan(self) -> Expr:
        """
        Create a boolean expression returning `True` where the expression values are NaN
        (Not A Number).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_column(pl.all().is_nan().suffix("_isnan"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬─────────┬─────────┐
        │ a    ┆ b   ┆ a_isnan ┆ b_isnan │
        │ ---  ┆ --- ┆ ---     ┆ ---     │
        │ i64  ┆ f64 ┆ bool    ┆ bool    │
        ╞══════╪═════╪═════════╪═════════╡
        │ 1    ┆ 1.0 ┆ false   ┆ false   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ 2    ┆ 2.0 ┆ false   ┆ false   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ null ┆ NaN ┆ false   ┆ true    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 1.0 ┆ false   ┆ false   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ 5    ┆ 5.0 ┆ false   ┆ false   │
        └──────┴─────┴─────────┴─────────┘

        """
        return wrap_expr(self._pyexpr.is_nan())

    def is_not_nan(self) -> Expr:
        """
        Create a boolean expression returning `True` where the expression values are not
        NaN (Not A Number).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_column(pl.all().is_not_nan().suffix("_is_not_nan"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬──────────────┬──────────────┐
        │ a    ┆ b   ┆ a_is_not_nan ┆ b_is_not_nan │
        │ ---  ┆ --- ┆ ---          ┆ ---          │
        │ i64  ┆ f64 ┆ bool         ┆ bool         │
        ╞══════╪═════╪══════════════╪══════════════╡
        │ 1    ┆ 1.0 ┆ true         ┆ true         │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2    ┆ 2.0 ┆ true         ┆ true         │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ NaN ┆ true         ┆ false        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 1.0 ┆ true         ┆ true         │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5    ┆ 5.0 ┆ true         ┆ true         │
        └──────┴─────┴──────────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.is_not_nan())

    def agg_groups(self) -> Expr:
        """
        Get the group indexes of the group by operation.
        Should be used in aggregation context only.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": [
        ...             "one",
        ...             "one",
        ...             "one",
        ...             "two",
        ...             "two",
        ...             "two",
        ...         ],
        ...         "value": [94, 95, 96, 97, 97, 99],
        ...     }
        ... )
        >>> df.groupby("group", maintain_order=True).agg(pl.col("value").agg_groups())
        shape: (2, 2)
        ┌───────┬───────────┐
        │ group ┆ value     │
        │ ---   ┆ ---       │
        │ str   ┆ list[u32] │
        ╞═══════╪═══════════╡
        │ one   ┆ [0, 1, 2] │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ two   ┆ [3, 4, 5] │
        └───────┴───────────┘

        """
        return wrap_expr(self._pyexpr.agg_groups())

    def count(self) -> Expr:
        """
        Count the number of values in this expression

        Examples
        --------
        >>> df = pl.DataFrame({"a": [8, 9, 10], "b": [None, 4, 4]})
        >>> df.select(pl.all().count())  # counts nulls
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 3   ┆ 3   │
        └─────┴─────┘

        """
        return wrap_expr(self._pyexpr.count())

    def len(self) -> Expr:
        """
        Alias for :func:`count`.
        Count the number of values in this expression

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10],
        ...         "b": [None, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.all().len())  # counts nulls
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 3   ┆ 3   │
        └─────┴─────┘

        """
        return self.count()

    def slice(self, offset: int | Expr, length: int | Expr) -> Expr:
        """
        Slice the Series.

        Parameters
        ----------
        offset
            Start index.
        length
            Length of the slice.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [8, 9, 10], "b": [None, 4, 4]})
        >>> df.select(pl.all().slice(1, 2))
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 9   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 10  ┆ 4   │
        └─────┴─────┘

        """
        if isinstance(offset, int):
            offset = pli.lit(offset)
        if isinstance(length, int):
            length = pli.lit(length)
        return wrap_expr(self._pyexpr.slice(offset._pyexpr, length._pyexpr))

    def append(self, other: Expr, upcast: bool = True) -> Expr:
        """
        Append expressions. This is done by adding the chunks of `other` to this
        `Series`.

        Parameters
        ----------
        other
            Expression to append.
        upcast
            Cast both `Series` to the same supertype.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [8, 9, 10], "b": [None, 4, 4]})
        >>> df.select(pl.all().head(1).append(pl.all().tail(1)))
        shape: (2, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ i64  │
        ╞═════╪══════╡
        │ 8   ┆ null │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 10  ┆ 4    │
        └─────┴──────┘

        """
        other = expr_to_lit_or_expr(other)
        return wrap_expr(self._pyexpr.append(other._pyexpr, upcast))

    def rechunk(self) -> Expr:
        """Create a single chunk of memory for this Series."""
        return wrap_expr(self._pyexpr.rechunk())

    def drop_nulls(self) -> Expr:
        """
        Drop null values.

        .. warning::
            Note that null values are not floating point NaN values!
            To drop NaN values, use :func:`drop_nans`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [8, 9, 10, 11], "b": [None, 4.0, 4.0, float("nan")]}
        ... )
        >>> df.select(pl.col("b").drop_nulls())
        shape: (3, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        ├╌╌╌╌╌┤
        │ 4.0 │
        ├╌╌╌╌╌┤
        │ NaN │
        └─────┘

        """
        return wrap_expr(self._pyexpr.drop_nulls())

    def drop_nans(self) -> Expr:
        """
        Drop floating point NaN values

        .. warning::
            Note that NaN values are not null values!
            To drop null values, use :func:`drop_nulls`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [8, 9, 10, 11], "b": [None, 4.0, 4.0, float("nan")]}
        ... )
        >>> df.select(pl.col("b").drop_nans())
        shape: (3, 1)
        ┌──────┐
        │ b    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 4.0  │
        ├╌╌╌╌╌╌┤
        │ 4.0  │
        └──────┘

        """
        return wrap_expr(self._pyexpr.drop_nans())

    def cumsum(self, reverse: bool = False) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cumsum(),
        ...         pl.col("a").cumsum(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 10        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 9         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 6   ┆ 7         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 10  ┆ 4         │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.cumsum(reverse))

    def cumprod(self, reverse: bool = False) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cumprod(),
        ...         pl.col("a").cumprod(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 24        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ 24        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 6   ┆ 12        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 24  ┆ 4         │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.cumprod(reverse))

    def cummin(self, reverse: bool = False) -> Expr:
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cummin(),
        ...         pl.col("a").cummin(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 1         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ 2         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ 3         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ 4         │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.cummin(reverse))

    def cummax(self, reverse: bool = False) -> Expr:
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cummax(),
        ...         pl.col("a").cummax(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 4         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ 4         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 4         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ 4         │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.cummax(reverse))

    def cumcount(self, reverse: bool = False) -> Expr:
        """
        Get an array with the cumulative count computed at every element.
        Counting from 0 to len

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cumcount(),
        ...         pl.col("a").cumcount(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ u32 ┆ u32       │
        ╞═════╪═══════════╡
        │ 0   ┆ 3         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ 2         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ 1         │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 0         │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.cumcount(reverse))

    def floor(self) -> Expr:
        """
        Floor underlying floating point array to the lowest integers smaller or equal to
        the float value.

        Only works on floating point Series

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.3, 0.5, 1.0, 1.1]})
        >>> df.select(pl.col("a").floor())
        shape: (4, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        ├╌╌╌╌╌┤
        │ 0.0 │
        ├╌╌╌╌╌┤
        │ 1.0 │
        ├╌╌╌╌╌┤
        │ 1.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.floor())

    def ceil(self) -> Expr:
        """
        Ceil underlying floating point array to the highest integers smaller or equal to
        the float value.

        Only works on floating point Series

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.3, 0.5, 1.0, 1.1]})
        >>> df.select(pl.col("a").ceil())
        shape: (4, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        ├╌╌╌╌╌┤
        │ 1.0 │
        ├╌╌╌╌╌┤
        │ 1.0 │
        ├╌╌╌╌╌┤
        │ 2.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.ceil())

    def round(self, decimals: int) -> Expr:
        """
        Round underlying floating point data by `decimals` digits.

        Parameters
        ----------
        decimals
            Number of decimals to round by.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.33, 0.52, 1.02, 1.17]})
        >>> df.select(pl.col("a").round(1))
        shape: (4, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.3 │
        ├╌╌╌╌╌┤
        │ 0.5 │
        ├╌╌╌╌╌┤
        │ 1.0 │
        ├╌╌╌╌╌┤
        │ 1.2 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.round(decimals))

    def dot(self, other: Expr | str) -> Expr:
        """
        Compute the dot/inner product between two Expressions

        Parameters
        ----------
        other
            Expression to compute dot product with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.select(pl.col("a").dot(pl.col("b")))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 44  │
        └─────┘

        """
        other = expr_to_lit_or_expr(other, str_to_lit=False)
        return wrap_expr(self._pyexpr.dot(other._pyexpr))

    def mode(self) -> Expr:
        """
        Compute the most occurring value(s).
        Can return multiple Values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3],
        ...         "b": [1, 1, 2, 2],
        ...     }
        ... )
        >>> df.select(pl.all().mode())  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 1   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 2   │
        └─────┴─────┘

        """
        return wrap_expr(self._pyexpr.mode())

    def cast(self, dtype: type[Any] | DataType, strict: bool = True) -> Expr:
        """
        Cast between data types.

        Parameters
        ----------
        dtype
            DataType to cast to.
        strict
            Throw an error if a cast could not be done.
            For instance, due to an overflow.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "6"]})
        >>> df.with_columns(
        ...     [
        ...         pl.col("a").cast(pl.Float64),
        ...         pl.col("b").cast(pl.Int32),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ i32 │
        ╞═════╪═════╡
        │ 1.0 ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2.0 ┆ 5   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3.0 ┆ 6   │
        └─────┴─────┘

        """
        dtype = py_type_to_dtype(dtype)
        return wrap_expr(self._pyexpr.cast(dtype, strict))

    def sort(self, reverse: bool = False, nulls_last: bool = False) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": [
        ...             "one",
        ...             "one",
        ...             "one",
        ...             "two",
        ...             "two",
        ...             "two",
        ...         ],
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.select(pl.col("value").sort())
        shape: (6, 1)
        ┌───────┐
        │ value │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 1     │
        ├╌╌╌╌╌╌╌┤
        │ 2     │
        ├╌╌╌╌╌╌╌┤
        │ 3     │
        ├╌╌╌╌╌╌╌┤
        │ 4     │
        ├╌╌╌╌╌╌╌┤
        │ 98    │
        ├╌╌╌╌╌╌╌┤
        │ 99    │
        └───────┘
        >>> df.select(pl.col("value").sort())
        shape: (6, 1)
        ┌───────┐
        │ value │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 1     │
        ├╌╌╌╌╌╌╌┤
        │ 2     │
        ├╌╌╌╌╌╌╌┤
        │ 3     │
        ├╌╌╌╌╌╌╌┤
        │ 4     │
        ├╌╌╌╌╌╌╌┤
        │ 98    │
        ├╌╌╌╌╌╌╌┤
        │ 99    │
        └───────┘
        >>> df.groupby("group").agg(pl.col("value").sort())  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌───────┬────────────┐
        │ group ┆ value      │
        │ ---   ┆ ---        │
        │ str   ┆ list[i64]  │
        ╞═══════╪════════════╡
        │ two   ┆ [3, 4, 99] │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ one   ┆ [1, 2, 98] │
        └───────┴────────────┘

        """
        return wrap_expr(self._pyexpr.sort_with(reverse, nulls_last))

    def arg_sort(self, reverse: bool = False) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_sort())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 0   │
        ├╌╌╌╌╌┤
        │ 2   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arg_sort(reverse))

    def arg_max(self) -> Expr:
        """
        Get the index of the maximal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arg_max())

    def arg_min(self) -> Expr:
        """
        Get the index of the minimal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_min())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arg_min())

    def sort_by(
        self,
        by: Expr | str | List[Expr | str],
        reverse: bool | List[bool] = False,
    ) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": [
        ...             "one",
        ...             "one",
        ...             "one",
        ...             "two",
        ...             "two",
        ...             "two",
        ...         ],
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.select(pl.col("group").sort_by("value"))
        shape: (6, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ one   │
        ├╌╌╌╌╌╌╌┤
        │ one   │
        ├╌╌╌╌╌╌╌┤
        │ two   │
        ├╌╌╌╌╌╌╌┤
        │ two   │
        ├╌╌╌╌╌╌╌┤
        │ one   │
        ├╌╌╌╌╌╌╌┤
        │ two   │
        └───────┘

        """
        if not isinstance(by, list):
            by = [by]
        if not isinstance(reverse, list):
            reverse = [reverse]
        by = selection_to_pyexpr_list(by)

        return wrap_expr(self._pyexpr.sort_by(by, reverse))

    def take(self, index: List[int] | Expr | pli.Series | np.ndarray) -> Expr:
        """
        Take values by index.

        Parameters
        ----------
        index
            An expression that leads to a UInt32 dtyped Series.

        Returns
        -------
        Values taken by index

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": [
        ...             "one",
        ...             "one",
        ...             "one",
        ...             "two",
        ...             "two",
        ...             "two",
        ...         ],
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.groupby("group", maintain_order=True).agg(pl.col("value").take(1))
        shape: (2, 2)
        ┌───────┬───────┐
        │ group ┆ value │
        │ ---   ┆ ---   │
        │ str   ┆ i64   │
        ╞═══════╪═══════╡
        │ one   ┆ 98    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ two   ┆ 99    │
        └───────┴───────┘

        """
        if isinstance(index, list) or (
            _NUMPY_AVAILABLE and isinstance(index, np.ndarray)
        ):
            index_lit = pli.lit(pli.Series("", index, dtype=UInt32))
        else:
            index_lit = pli.expr_to_lit_or_expr(
                index,  # type: ignore[arg-type]
                str_to_lit=False,
            )
        return pli.wrap_expr(self._pyexpr.take(index_lit._pyexpr))

    def shift(self, periods: int = 1) -> Expr:
        """
        Shift the values by a given period and fill the parts that will be empty due to
        this operation with nulls.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").shift(1))
        shape: (4, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 1    │
        ├╌╌╌╌╌╌┤
        │ 2    │
        ├╌╌╌╌╌╌┤
        │ 3    │
        └──────┘

        """
        return wrap_expr(self._pyexpr.shift(periods))

    def shift_and_fill(
        self, periods: int, fill_value: int | float | bool | str | Expr
    ) -> Expr:
        """
        Shift the values by a given period and fill the parts that will be empty due to
        this operation with the result of the ``fill_value`` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            Fill None values with the result of this expression.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").shift_and_fill(1, "a"))
        shape: (4, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ a   │
        ├╌╌╌╌╌┤
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 3   │
        └─────┘

        """
        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.shift_and_fill(periods, fill_value._pyexpr))

    def fill_null(
        self,
        fill_value: int | float | bool | str | Expr,
        limit: int | None = None,
    ) -> Expr:
        """
        Fill null values using a filling strategy, literal, or Expr.

        Parameters
        ----------
        fill_value
            One of {"backward", "forward", "min", "max", "mean", "one", "zero"}
            or an expression.
        limit
            The number of consecutive null values to forward/backward fill.
            Only valid if ``fill_value`` is 'forward' or 'backward'.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, None], "b": [4, None, 6]})
        >>> df.fill_null("zero")
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 0   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 0   ┆ 6   │
        └─────┴─────┘
        >>> df.fill_null(99)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 99  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 99  ┆ 6   │
        └─────┴─────┘

        """
        # we first must check if it is not an expr, as expr does not implement __bool__
        # and thus leads to a value error in the second comparison.
        if not isinstance(fill_value, Expr) and fill_value in [
            "backward",
            "forward",
            "min",
            "max",
            "mean",
            "zero",
            "one",
        ]:
            return wrap_expr(self._pyexpr.fill_null_with_strategy(fill_value, limit))

        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.fill_null(fill_value._pyexpr))

    def fill_nan(self, fill_value: str | int | float | bool | Expr) -> Expr:
        """
        Fill floating point NaN value with a fill value

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [1.0, None, float("nan")], "b": [4.0, float("nan"), 6]}
        ... )
        >>> df.fill_nan("zero")
        shape: (3, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ str  ┆ str  │
        ╞══════╪══════╡
        │ 1.0  ┆ 4.0  │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ zero │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ zero ┆ 6.0  │
        └──────┴──────┘

        """
        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.fill_nan(fill_value._pyexpr))

    def forward_fill(self, limit: int | None = None) -> Expr:
        """
        Fill missing values with the latest seen values.

        Parameters
        ----------
        limit
            The number of consecutive null values to forward fill.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, None], "b": [4, None, 6]})
        >>> df.select(pl.all().forward_fill())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 6   │
        └─────┴─────┘

        """
        return wrap_expr(self._pyexpr.forward_fill(limit))

    def backward_fill(self, limit: int | None = None) -> Expr:
        """
        Fill missing values with the next to be seen values.

        Parameters
        ----------
        limit
            The number of consecutive null values to backward fill.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, None], "b": [4, None, 6]})
        >>> df.select(pl.all().backward_fill())
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 1    ┆ 4   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2    ┆ 6   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ null ┆ 6   │
        └──────┴─────┘

        """
        return wrap_expr(self._pyexpr.backward_fill(limit))

    def reverse(self) -> Expr:
        """
        Reverse the selection.

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

        """  # noqa: E501
        return wrap_expr(self._pyexpr.reverse())

    def std(self) -> Expr:
        """
        Get standard deviation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").std())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.std())

    def var(self) -> Expr:
        """
        Get variance.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").var())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.var())

    def max(self) -> Expr:
        """
        Get maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.max())

    def min(self) -> Expr:
        """
        Get minimum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").min())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ -1  │
        └─────┘

        """
        return wrap_expr(self._pyexpr.min())

    def sum(self) -> Expr:
        """
        Get sum value.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").sum())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │  0  │
        └─────┘

        """
        return wrap_expr(self._pyexpr.sum())

    def mean(self) -> Expr:
        """
        Get mean value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").mean())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.mean())

    def median(self) -> Expr:
        """
        Get median value using linear interpolation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").median())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.median())

    def product(self) -> Expr:
        """
        Compute the product of an expression

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").product())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.product())

    def n_unique(self) -> "Expr":
        """
        Count unique values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").n_unique())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.n_unique())

    def null_count(self) -> "Expr":
        """
        Count null values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [None, 1, None], "b": [1, 2, 3]})
        >>> df.select(pl.all().null_count())
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 2   ┆ 0   │
        └─────┴─────┘

        """
        return wrap_expr(self._pyexpr.null_count())

    def arg_unique(self) -> Expr:
        """
        Get index of first unique value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [8, 9, 10], "b": [None, 4, 4]})
        >>> df.select(pl.col("a").arg_unique())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        ├╌╌╌╌╌┤
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        └─────┘
        >>> df.select(pl.col("b").arg_unique())
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        ├╌╌╌╌╌┤
        │ 1   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arg_unique())

    def unique(self, maintain_order: bool = False) -> Expr:
        """
        Get unique values of this expression.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").unique())  # doctest: +IGNORE_RESULT
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        ├╌╌╌╌╌┤
        │ 1   │
        └─────┘
        >>> df.select(pl.col("a").unique(maintain_order=True))
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        └─────┘

        """
        if maintain_order:
            return wrap_expr(self._pyexpr.unique_stable())
        return wrap_expr(self._pyexpr.unique())

    def first(self) -> Expr:
        """
        Get the first value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").first())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.first())

    def last(self) -> Expr:
        """
        Get the last value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").last())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.last())

    def list(self) -> Expr:
        """
        Aggregate to list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [4, 5, 6],
        ...     }
        ... )
        >>> df.select(pl.all().list())
        shape: (1, 2)
        ┌───────────┬───────────┐
        │ a         ┆ b         │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [1, 2, 3] ┆ [4, 5, 6] │
        └───────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.list())

    def over(self, expr: str | Expr | List[Expr | str]) -> Expr:
        """
        Apply window function over a subgroup.

        This is similar to a groupby + aggregation + self join.
        Or similar to `window functions in Postgres
        <https://www.postgresql.org/docs/current/tutorial-window.html>`_.

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

    def is_unique(self) -> Expr:
        """
        Get mask of unique values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").is_unique())
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        ├╌╌╌╌╌╌╌┤
        │ false │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        └───────┘

        """
        return wrap_expr(self._pyexpr.is_unique())

    def is_first(self) -> Expr:
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "num": [1, 2, 3, 1, 5],
        ...     }
        ... )
        >>> df.with_column(pl.col("num").is_first().alias("is_first"))
        shape: (5, 2)
        ┌─────┬──────────┐
        │ num ┆ is_first │
        │ --- ┆ ---      │
        │ i64 ┆ bool     │
        ╞═════╪══════════╡
        │ 1   ┆ true     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ true     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ true     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ false    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ true     │
        └─────┴──────────┘

        """
        return wrap_expr(self._pyexpr.is_first())

    def is_duplicated(self) -> Expr:
        """
        Get mask of duplicated values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").is_duplicated())
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        ├╌╌╌╌╌╌╌┤
        │ false │
        └───────┘

        """
        return wrap_expr(self._pyexpr.is_duplicated())

    def quantile(self, quantile: float, interpolation: str = "nearest") -> Expr:
        """
        Get quantile value.


        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options:
            ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
        >>> df.select(pl.col("a").quantile(0.3))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="higher"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 2.0 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="lower"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="midpoint"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.5 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="linear"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.5 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.quantile(quantile, interpolation))

    def filter(self, predicate: Expr) -> Expr:
        """
        Filter a single column.
        Mostly useful in in aggregation context. If you want to filter on a DataFrame
        level, use `LazyFrame.filter`.

        Parameters
        ----------
        predicate
            Boolean expression.

        """
        return wrap_expr(self._pyexpr.filter(predicate._pyexpr))

    def where(self, predicate: Expr) -> Expr:
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
        f: Callable[[pli.Series], pli.Series | Any],
        return_dtype: type[DataType] | None = None,
        agg_list: bool = False,
    ) -> Expr:
        """
        Apply a custom python function to a Series or sequence of Series.

        The output of this custom function must be a Series.
        If you want to apply a custom function elementwise over single values, see
        :func:`apply`. A use case for ``map`` is when you want to transform an
        expression with a third-party library.

        Read more in `the book
        <https://pola-rs.github.io/polars-book/user-guide/dsl/custom_functions.html>`_.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        agg_list
            Aggregate list

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "sine": [0.0, 1.0, 0.0, -1.0],
        ...         "cosine": [1.0, 0.0, -1.0, 0.0],
        ...     }
        ... )
        >>> (df.select(pl.all().map(lambda x: x.to_numpy().argmax())))
        shape: (1, 2)
        ┌──────┬────────┐
        │ sine ┆ cosine │
        │ ---  ┆ ---    │
        │ i64  ┆ i64    │
        ╞══════╪════════╡
        │ 1    ┆ 0      │
        └──────┴────────┘

        """
        if return_dtype is not None:
            return_dtype = py_type_to_dtype(return_dtype)
        return wrap_expr(self._pyexpr.map(f, return_dtype, agg_list))

    def apply(
        self,
        f: Callable[[pli.Series], pli.Series] | Callable[[Any], Any],
        return_dtype: type[DataType] | None = None,
    ) -> Expr:
        """
        Apply a custom function in a GroupBy or Projection context.

        Depending on the context it has the following behavior:

        * Selection
            Expects `f` to be of type Callable[[Any], Any].
            Applies a python function over each individual value in the column.
        * GroupBy
            Expects `f` to be of type Callable[[Series], Series].
            Applies a python function over each group.

        Implementing logic using the ``.apply`` method is generally slower and more
        memory intensive than implementing the same logic using the expression API
        because:

        - with .apply the logic is implemented in Python but with an expression the
          logic is implemented in Rust
        - with ``.apply`` the DataFrame is materialized in memory
        - expressions can be parallelised
        - expressions can be optimised

        If possible, use the expression API for best performance.

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
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["a", "b", "c", "c"],
        ...     }
        ... )

        In a selection context, the function is applied by row.

        >>> (
        ...     df.with_column(
        ...         pl.col("a").apply(lambda x: x * 2).alias("a_times_2"),
        ...     )
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────────┐
        │ a   ┆ b   ┆ a_times_2 │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ str ┆ i64       │
        ╞═════╪═════╪═══════════╡
        │ 1   ┆ a   ┆ 2         │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ b   ┆ 4         │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ c   ┆ 6         │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ c   ┆ 2         │
        └─────┴─────┴───────────┘

        It is better to implement this with an expression:

        >>> (
        ...     df.with_column(
        ...         (pl.col("a") * 2).alias("a_times_2"),
        ...     )
        ... )  # doctest: +IGNORE_RESULT

        In a GroupBy context the function is applied by group:

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
        │ c   ┆ 4   │
        └─────┴─────┘

        It is better to implement this with an expression:

        >>> (
        ...     df.groupby("b", maintain_order=True).agg(
        ...         pl.col("a").sum(),
        ...     )
        ... )  # doctest: +IGNORE_RESULT

        """
        # input x: Series of type list containing the group values
        def wrap_f(x: pli.Series) -> pli.Series:  # pragma: no cover
            return x.apply(f, return_dtype=return_dtype)

        return self.map(wrap_f, agg_list=True)

    def flatten(self) -> Expr:
        """
        Alias for :func:`explode`.

        Explode a list or utf8 Series. This means that every item is expanded to a new
        row.

        Returns
        -------
        Exploded Series of same dtype

        Examples
        --------
        The following example turns each character into a separate row:

        >>> df = pl.DataFrame({"foo": ["hello", "world"]})
        >>> df.select(pl.col("foo").flatten())
        shape: (10, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ h   │
        ├╌╌╌╌╌┤
        │ e   │
        ├╌╌╌╌╌┤
        │ l   │
        ├╌╌╌╌╌┤
        │ l   │
        ├╌╌╌╌╌┤
        │ ... │
        ├╌╌╌╌╌┤
        │ o   │
        ├╌╌╌╌╌┤
        │ r   │
        ├╌╌╌╌╌┤
        │ l   │
        ├╌╌╌╌╌┤
        │ d   │
        └─────┘

        This example turns each word into a separate row:

        >>> df = pl.DataFrame({"foo": ["hello world"]})
        >>> df.select(pl.col("foo").str.split(by=" ").flatten())
        shape: (2, 1)
        ┌───────┐
        │ foo   │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ hello │
        ├╌╌╌╌╌╌╌┤
        │ world │
        └───────┘

        """
        return wrap_expr(self._pyexpr.explode())

    def explode(self) -> Expr:
        """
        Explode a list or utf8 Series.

        This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype

        Examples
        --------
        >>> df = pl.DataFrame({"b": [[1, 2, 3], [4, 5, 6]]})
        >>> df.select(pl.col("b").explode())
        shape: (6, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 3   │
        ├╌╌╌╌╌┤
        │ 4   │
        ├╌╌╌╌╌┤
        │ 5   │
        ├╌╌╌╌╌┤
        │ 6   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.explode())

    def take_every(self, n: int) -> Expr:
        """
        Take every nth value in the Series and return as a new Series.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        >>> df.select(pl.col("foo").take_every(3))
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 4   │
        ├╌╌╌╌╌┤
        │ 7   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.take_every(n))

    def head(self, n: int | Expr | None = None) -> Expr:
        """Take the first n values."""
        if isinstance(n, Expr):
            return self.slice(0, n)
        return wrap_expr(self._pyexpr.head(n))

    def tail(self, n: int | None = None) -> Expr:
        """Take the last n values."""
        return wrap_expr(self._pyexpr.tail(n))

    def pow(self, exponent: int | float | pli.Series | Expr) -> Expr:
        """
        Raise expression to the power of exponent.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").pow(3))
        shape: (4, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ 1.0  │
        ├╌╌╌╌╌╌┤
        │ 8.0  │
        ├╌╌╌╌╌╌┤
        │ 27.0 │
        ├╌╌╌╌╌╌┤
        │ 64.0 │
        └──────┘

        """
        exponent = expr_to_lit_or_expr(exponent)
        return wrap_expr(self._pyexpr.pow(exponent._pyexpr))

    def is_in(self, other: Expr | List[Any] | str) -> Expr:
        """
        Check if elements of this Series are in the right Series, or List values of the
        right Series.

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

    def repeat_by(self, by: Expr | str) -> Expr:
        """
        Repeat the elements in this Series `n` times by dictated by the number given by
        `by`. The elements are expanded into a `List`

        Parameters
        ----------
        by
            Numeric column that determines how often the values will be repeated.
            The column will be coerced to UInt32. Give this dtype to make the coercion a
            no-op.

        Returns
        -------
        Series of type List

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "n": [1, 2, 3],
        ...     }
        ... )
        >>> df.select(pl.col("a").repeat_by("n"))
        shape: (3, 1)
        ┌─────────────────┐
        │ a               │
        │ ---             │
        │ list[str]       │
        ╞═════════════════╡
        │ ["x"]           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["y", "y"]      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["z", "z", "z"] │
        └─────────────────┘

        """
        by = expr_to_lit_or_expr(by, False)
        return wrap_expr(self._pyexpr.repeat_by(by._pyexpr))

    def is_between(
        self,
        start: Expr | datetime | int,
        end: Expr | datetime | int,
        include_bounds: bool | Sequence[bool] = False,
    ) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "num": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> df.with_column(pl.col("num").is_between(2, 4))
        shape: (5, 2)
        ┌─────┬────────────┐
        │ num ┆ is_between │
        │ --- ┆ ---        │
        │ i64 ┆ bool       │
        ╞═════╪════════════╡
        │ 1   ┆ false      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ false      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ true       │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ false      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ false      │
        └─────┴────────────┘

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

    def hash(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> Expr:
        """
        Hash the elements in the selection.

        The hash value is of type `UInt64`.

        Parameters
        ----------
        seed
            Random seed parameter. Defaults to 0.
        seed_1
            Random seed parameter. Defaults to `seed` if not set.
        seed_2
            Random seed parameter. Defaults to `seed` if not set.
        seed_3
            Random seed parameter. Defaults to `seed` if not set.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": ["x", None, "z"],
        ...     }
        ... )
        >>> df.with_column(pl.all().hash(10, 20, 30, 40))
        shape: (3, 2)
        ┌──────────────────────┬──────────────────────┐
        │ a                    ┆ b                    │
        │ ---                  ┆ ---                  │
        │ u64                  ┆ u64                  │
        ╞══════════════════════╪══════════════════════╡
        │ 2461716855791224000  ┆ 16174362112783765148 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 13569566217648818014 ┆ 11638928888656214026 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 11638928888656214026 ┆ 6351727772611549480  │
        └──────────────────────┴──────────────────────┘

        """
        k0 = seed
        k1 = seed_1 if seed_1 is not None else seed
        k2 = seed_2 if seed_2 is not None else seed
        k3 = seed_3 if seed_3 is not None else seed
        return wrap_expr(self._pyexpr.hash(k0, k1, k2, k3))

    def reinterpret(self, signed: bool) -> Expr:
        """
        Reinterpret the underlying bits as a signed/unsigned integer.

        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            If True, reinterpret as `pl.Int64`. Otherwise, reinterpret as `pl.UInt64`.

        """
        return wrap_expr(self._pyexpr.reinterpret(signed))

    def inspect(self, fmt: str = "{}") -> Expr:
        """
        Print the value that this expression evaluates to and pass on the value.

        Examples
        --------
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

        def inspect(s: pli.Series) -> pli.Series:  # pragma: no cover
            print(fmt.format(s))
            return s

        return self.map(inspect, return_dtype=None, agg_list=True)

    def interpolate(self) -> Expr:
        """Linearly interpolate intermediate values."""
        return wrap_expr(self._pyexpr.interpolate())

    def rolling_min(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Apply a rolling min (moving min) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}


        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> (
        ...     df.select(
        ...         [
        ...             pl.col("A").rolling_min(window_size=2),
        ...         ]
        ...     )
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 1.0  │
        ├╌╌╌╌╌╌┤
        │ 2.0  │
        ├╌╌╌╌╌╌┤
        │ 3.0  │
        ├╌╌╌╌╌╌┤
        │ 4.0  │
        ├╌╌╌╌╌╌┤
        │ 5.0  │
        └──────┘

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_min(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_max(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Apply a rolling max (moving max) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> (
        ...     df.select(
        ...         [
        ...             pl.col("A").rolling_max(window_size=2),
        ...         ]
        ...     )
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 2.0  │
        ├╌╌╌╌╌╌┤
        │ 3.0  │
        ├╌╌╌╌╌╌┤
        │ 4.0  │
        ├╌╌╌╌╌╌┤
        │ 5.0  │
        ├╌╌╌╌╌╌┤
        │ 6.0  │
        └──────┘

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_max(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_mean(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Apply a rolling mean (moving mean) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

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
        │ 7.0  │
        ├╌╌╌╌╌╌┤
        │ 4.0  │
        ├╌╌╌╌╌╌┤
        │ 9.0  │
        ├╌╌╌╌╌╌┤
        │ 13.0 │
        └──────┘

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_mean(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_sum(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Apply a rolling sum (moving sum) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> (
        ...     df.select(
        ...         [
        ...             pl.col("A").rolling_sum(window_size=2),
        ...         ]
        ...     )
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ 3.0  │
        ├╌╌╌╌╌╌┤
        │ 5.0  │
        ├╌╌╌╌╌╌┤
        │ 7.0  │
        ├╌╌╌╌╌╌┤
        │ 9.0  │
        ├╌╌╌╌╌╌┤
        │ 11.0 │
        └──────┘

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_sum(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_std(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Compute a rolling standard deviation.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_std(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_var(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_var(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_median(
        self,
        window_size: int | str,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Compute a rolling median.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_median(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_quantile(
        self,
        quantile: float,
        interpolation: str = "nearest",
        window_size: int | str = 2,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: str = "left",
    ) -> Expr:
        """
        Compute a rolling quantile.

        Parameters
        ----------
        quantile
            quantile to compute
        interpolation
            interpolation type, options:
            ['nearest', 'higher', 'lower', 'midpoint', 'linear']
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If the dynamic string language is used, the `by` and `closed` arguments must
            also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed
            Defines if the temporal window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        .. warning::
            The dynamic windows functionality is still experimental and may change
            without it considered being a breaking change

        .. note::
            If you want to compute multiple aggregation statistics over the same dynamic
            window, consider using `groupby_rolling` this method can cache the window
            size computation.

        """
        window_size, min_periods = _prepare_rolling_window_args(
            window_size, min_periods
        )
        return wrap_expr(
            self._pyexpr.rolling_quantile(
                quantile,
                interpolation,
                window_size,
                weights,
                min_periods,
                center,
                by,
                closed,
            )
        )

    def rolling_apply(
        self,
        function: Callable[[pli.Series], Any],
        window_size: int,
        weights: List[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Expr:
        """
        Apply a custom rolling window function.

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
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
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
        ┌──────────┐
        │ A        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ null     │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 4.358899 │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 4.041452 │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 5.567764 │
        └──────────┘

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_expr(
            self._pyexpr.rolling_apply(
                function, window_size, weights, min_periods, center
            )
        )

    def rolling_skew(self, window_size: int, bias: bool = True) -> Expr:
        """
        Compute a rolling skew.

        Parameters
        ----------
        window_size
            Size of the rolling window
        bias
            If False, then the calculations are corrected for statistical bias.

        """
        return wrap_expr(self._pyexpr.rolling_skew(window_size, bias))

    def abs(self) -> Expr:
        """Compute absolute values."""
        return wrap_expr(self._pyexpr.abs())

    def argsort(self, reverse: bool = False) -> Expr:
        """Alias for `arg_sort`."""
        return self.arg_sort(reverse)

    def rank(self, method: str = "average", reverse: bool = False) -> Expr:
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'dense', 'ordinal', 'random'}, optional
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):

            - 'average' : The average of the ranks that would have been assigned to
              all the tied values is assigned to each value.
            - 'min' : The minimum of the ranks that would have been assigned to all
              the tied values is assigned to each value. (This is also referred to
              as "competition" ranking.)
            - 'max' : The maximum of the ranks that would have been assigned to all
              the tied values is assigned to each value.
            - 'dense' : Like 'min', but the rank of the next highest element is
              assigned the rank immediately after those assigned to the tied
              elements.
            - 'ordinal' : All values are given a distinct rank, corresponding to
              the order that the values occur in the Series.
            - 'random' : Like 'ordinal', but the rank for ties is not dependent
              on the order that the values occur in the Series.
        reverse
            Reverse the operation.

        Examples
        --------
        The 'average' method:

        >>> df = pl.DataFrame({"a": [3, 6, 1, 1, 6]})
        >>> df.select(pl.col("a").rank())
        shape: (5, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f32 │
        ╞═════╡
        │ 3.0 │
        ├╌╌╌╌╌┤
        │ 4.5 │
        ├╌╌╌╌╌┤
        │ 1.5 │
        ├╌╌╌╌╌┤
        │ 1.5 │
        ├╌╌╌╌╌┤
        │ 4.5 │
        └─────┘

        The 'ordinal' method:

        >>> df = pl.DataFrame({"a": [3, 6, 1, 1, 6]})
        >>> df.select(pl.col("a").rank("ordinal"))
        shape: (5, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 3   │
        ├╌╌╌╌╌┤
        │ 4   │
        ├╌╌╌╌╌┤
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 5   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.rank(method, reverse))

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> Expr:
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            number of slots to shift
        null_behavior
            {'ignore', 'drop'}

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").diff())
        shape: (3, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        ├╌╌╌╌╌╌┤
        │ -10  │
        ├╌╌╌╌╌╌┤
        │ 20   │
        └──────┘

        """
        return wrap_expr(self._pyexpr.diff(n, null_behavior))

    def pct_change(self, n: int = 1) -> Expr:
        """
        Percentage change (as fraction) between current element and most-recent
        non-null element at least n period(s) before the current element.

        Computes the change from the previous row by default.

        Parameters
        ----------
        n
            periods to shift for forming percent change.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [10, 11, 12, None, 12],
        ...     }
        ... )
        >>> df.with_column(pl.col("a").pct_change().alias("pct_change"))
        shape: (5, 2)
        ┌──────┬────────────┐
        │ a    ┆ pct_change │
        │ ---  ┆ ---        │
        │ i64  ┆ f64        │
        ╞══════╪════════════╡
        │ 10   ┆ null       │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 11   ┆ 0.1        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 12   ┆ 0.090909   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ 0.0        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 12   ┆ 0.0        │
        └──────┴────────────┘

        """
        return wrap_expr(self._pyexpr.pct_change(n))

    def skew(self, bias: bool = True) -> Expr:
        r"""
        Compute the sample skewness of a data set.

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

        .. math::
            G_1 = \frac{k_3}{k_2^{3/2}} = \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}

        """
        return wrap_expr(self._pyexpr.skew(bias))

    def kurtosis(self, fisher: bool = True, bias: bool = True) -> Expr:
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset.

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

    def clip(self, min_val: int | float, max_val: int | float) -> Expr:
        """
        Clip (limit) the values in an array to any value that fits in 64 floating point
        range.

        Only works for the following dtypes: {Int32, Int64, Float32, Float64, UInt32}.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        min_val
            Minimum value.
        max_val
            Maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [-50, 5, None, 50]})
        >>> df.with_column(pl.col("foo").clip(1, 10).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ 1           │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5    ┆ 5           │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ null        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 50   ┆ 10          │
        └──────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.clip(min_val, max_val))

    def lower_bound(self) -> Expr:
        """
        Calculate the lower bound.

        Returns a unit Series with the lowest value possible for the dtype of this
        expression.

        """
        return wrap_expr(self._pyexpr.lower_bound())

    def upper_bound(self) -> Expr:
        """
        Calculate the upper bound.

        Returns a unit Series with the highest value possible for the dtype of this
        expression.

        """
        return wrap_expr(self._pyexpr.upper_bound())

    def sign(self) -> Expr:
        """
        Compute the element-wise indication of the sign.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-9.0, -0.0, 0.0, 4.0, None]})
        >>> df.select(pl.col("a").sign())
        shape: (5, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ -1   │
        ├╌╌╌╌╌╌┤
        │ 0    │
        ├╌╌╌╌╌╌┤
        │ 0    │
        ├╌╌╌╌╌╌┤
        │ 1    │
        ├╌╌╌╌╌╌┤
        │ null │
        └──────┘

        """
        return wrap_expr(self._pyexpr.sign())

    def sin(self) -> Expr:
        """
        Compute the element-wise value for the sine.

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
        return wrap_expr(self._pyexpr.sin())

    def cos(self) -> Expr:
        """
        Compute the element-wise value for the cosine.

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
        │ 1.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.cos())

    def tan(self) -> Expr:
        """
        Compute the element-wise value for the tangent.

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
        return wrap_expr(self._pyexpr.tan())

    def arcsin(self) -> Expr:
        """
        Compute the element-wise value for the inverse sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arcsin())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.570796 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.arcsin())

    def arccos(self) -> Expr:
        """
        Compute the element-wise value for the inverse cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").arccos())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.570796 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.arccos())

    def arctan(self) -> Expr:
        """
        Compute the element-wise value for the inverse tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arctan())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.785398 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.arctan())

    def sinh(self) -> Expr:
        """
        Compute the element-wise value for the hyperbolic sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").sinh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.175201 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.sinh())

    def cosh(self) -> Expr:
        """
        Compute the element-wise value for the hyperbolic cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").cosh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.543081 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.cosh())

    def tanh(self) -> Expr:
        """
        Compute the element-wise value for the hyperbolic tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").tanh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.761594 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.tanh())

    def arcsinh(self) -> Expr:
        """
        Compute the element-wise value for the inverse hyperbolic sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arcsinh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.881374 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.arcsinh())

    def arccosh(self) -> Expr:
        """
        Compute the element-wise value for the inverse hyperbolic cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arccosh())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arccosh())

    def arctanh(self) -> Expr:
        """
        Compute the element-wise value for the inverse hyperbolic tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arctanh())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ inf │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arctanh())

    def reshape(self, dims: tuple[int, ...]) -> Expr:
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

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        >>> df.select(pl.col("foo").reshape((3, 3)))
        shape: (3, 1)
        ┌───────────┐
        │ foo       │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        ├╌╌╌╌╌╌╌╌╌╌╌┤
        │ [4, 5, 6] │
        ├╌╌╌╌╌╌╌╌╌╌╌┤
        │ [7, 8, 9] │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.reshape(dims))

    def shuffle(self, seed: int | None = None) -> Expr:
        """
        Shuffle the contents of this expr.

        Parameters
        ----------
        seed
            Seed initialization. If None given, the `random` module is used to generate
            a random seed.

        """
        if seed is None:
            seed = random.randint(0, 10000)
        return wrap_expr(self._pyexpr.shuffle(seed))

    def sample(
        self,
        fraction: float = 1.0,
        with_replacement: bool = True,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Expr:
        """
        Sample a fraction of the `Series`.

        Parameters
        ----------
        fraction
            Fraction 0.0 <= value <= 1.0
        with_replacement
            Allow values to be sampled more than once.
        seed
            Seed initialization. If None given a random seed is used.
        shuffle
            Shuffle the order of sampled data points.

        """
        return wrap_expr(
            self._pyexpr.sample_frac(fraction, with_replacement, shuffle, seed)
        )

    def ewm_mean(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> Expr:
        r"""
        Exponential moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass,
            :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span,
            :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life,
            :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights
                    :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is Null).

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(self._pyexpr.ewm_mean(alpha, adjust, min_periods))

    def ewm_std(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> Expr:
        r"""
        Exponential moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass,
            :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span,
            :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life,
            :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights
                    :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is Null).

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(self._pyexpr.ewm_std(alpha, adjust, min_periods))

    def ewm_var(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> Expr:
        r"""
        Exponential moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass,
            :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span,
            :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life,
            :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights
                    :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is Null).

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(self._pyexpr.ewm_var(alpha, adjust, min_periods))

    def extend_constant(self, value: int | float | str | bool | None, n: int) -> Expr:
        """
        Extend the Series with given number of values.

        Parameters
        ----------
        value
            The value to extend the Series with. This value may be None to fill with
            nulls.
        n
            The number of values to extend.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.extend_constant(99, n=2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                99
                99
        ]

        """
        return wrap_expr(self._pyexpr.extend_constant(value, n))

    def value_counts(self, multithreaded: bool = False, sort: bool = False) -> Expr:
        """
        Count all unique values and create a struct mapping value to count

        Parameters
        ----------
        multithreaded:
            Better to turn this off in the aggregation context, as it can lead to
            contention.
        sort:
            Ensure the output is sorted from most values to least.

        Returns
        -------
        Dtype Struct

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["a", "b", "b", "c", "c", "c"],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("id").value_counts(sort=True),
        ...     ]
        ... )
        shape: (3, 1)
        ┌───────────┐
        │ id        │
        │ ---       │
        │ struct[2] │
        ╞═══════════╡
        │ {"c",3}   │
        ├╌╌╌╌╌╌╌╌╌╌╌┤
        │ {"b",2}   │
        ├╌╌╌╌╌╌╌╌╌╌╌┤
        │ {"a",1}   │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.value_counts(multithreaded, sort))

    def unique_counts(self) -> Expr:
        """
        Return a count of the unique values in the order of appearance.

        This method differs from `value_counts` in that it does not return the
        values, only the counts and might be faster

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["a", "b", "b", "c", "c", "c"],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("id").unique_counts(),
        ...     ]
        ... )
        shape: (3, 1)
        ┌─────┐
        │ id  │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 3   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.unique_counts())

    def log(self, base: float = math.e) -> Expr:
        """
        Compute the logarithm to a given base

        Parameters
        ----------
        base
            Given base, defaults to `e`

        """
        return wrap_expr(self._pyexpr.log(base))

    def entropy(self, base: float = math.e, normalize: bool = True) -> Expr:
        """
        Compute the entropy as `-sum(pk * log(pk)`.
        where `pk` are discrete probabilities.

        Parameters
        ----------
        base
            Given base, defaults to `e`
        normalize
            Normalize pk if it doesn't sum to 1.

        """
        return wrap_expr(self._pyexpr.entropy(base, normalize))

    def cumulative_eval(
        self, expr: Expr, min_periods: int = 1, parallel: bool = False
    ) -> Expr:
        """
        Run an expression over a sliding window that increases `1` slot every iteration.

        .. warning::
            This can be really slow as it can have `O(n^2)` complexity. Don't use this
            for operations that visit all elements.

        .. warning::
            This API is experimental and may change without it being considered a
            breaking change.

        Parameters
        ----------
        expr
            Expression to evaluate
        min_periods
            Number of valid values there should be in the window before the expression
            is evaluated. valid values = `length - null_count`
        parallel
            Run in parallel. Don't do this in a groupby or another operation that
            already has much parallelization.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
        >>> df.select(
        ...     [
        ...         pl.col("values").cumulative_eval(
        ...             pl.element().first() - pl.element().last() ** 2
        ...         )
        ...     ]
        ... )
        shape: (5, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ f64    │
        ╞════════╡
        │ 0.0    │
        ├╌╌╌╌╌╌╌╌┤
        │ -3.0   │
        ├╌╌╌╌╌╌╌╌┤
        │ -8.0   │
        ├╌╌╌╌╌╌╌╌┤
        │ -15.0  │
        ├╌╌╌╌╌╌╌╌┤
        │ -24.0  │
        └────────┘

        """
        return wrap_expr(
            self._pyexpr.cumulative_eval(expr._pyexpr, min_periods, parallel)
        )

    def set_sorted(self, reverse: bool = False) -> Expr:
        """
        Set this `Series` as `sorted` so that downstream code can use
        fast paths for sorted arrays.

        .. warning::
            This can lead to incorrect results if this `Series` is not sorted!!
            Use with care!

        Parameters
        ----------
        reverse
            If the `Series` order is reversed, e.g. descending.

        """
        return self.map(lambda s: s.set_sorted(reverse))

    # Below are the namespaces defined. Keep these at the end of the definition of Expr,
    # as to not confuse mypy with the type annotation `str` with the namespace "str"

    @property
    def dt(self) -> ExprDateTimeNameSpace:
        """Create an object namespace of all datetime related methods."""
        return ExprDateTimeNameSpace(self)

    @property
    def str(self) -> ExprStringNameSpace:
        """Create an object namespace of all string related methods."""
        return ExprStringNameSpace(self)

    @property
    def arr(self) -> ExprListNameSpace:
        """Create an object namespace of all list related methods."""
        return ExprListNameSpace(self)

    @property
    def cat(self) -> ExprCatNameSpace:
        """Create an object namespace of all categorical related methods."""
        return ExprCatNameSpace(self)

    @property
    def struct(self) -> ExprStructNameSpace:
        """Create an object namespace of all struct related methods."""
        return ExprStructNameSpace(self)


class ExprStructNameSpace:
    """Namespace for struct related expressions."""

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def field(self, name: str) -> Expr:
        """
        Retrieve one of the fields of this `Struct` as a new Series.

        Parameters
        ----------
        name
            Name of the field

        Examples
        --------
        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "int": [1, 2],
        ...             "str": ["a", "b"],
        ...             "bool": [True, None],
        ...             "list": [[1, 2], [3]],
        ...         }
        ...     )
        ...     .to_struct("my_struct")
        ...     .to_frame()
        ... )
        >>> df.select(pl.col("my_struct").struct.field("str"))
        shape: (2, 1)
        ┌─────┐
        │ str │
        │ --- │
        │ str │
        ╞═════╡
        │ a   │
        ├╌╌╌╌╌┤
        │ b   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.struct_field_by_name(name))

    def rename_fields(self, names: List[str]) -> Expr:
        """
        Rename the fields of the struct

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        Examples
        --------
        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "int": [1, 2],
        ...             "str": ["a", "b"],
        ...             "bool": [True, None],
        ...             "list": [[1, 2], [3]],
        ...         }
        ...     )
        ...     .to_struct("my_struct")
        ...     .to_frame()
        ... )
        >>> df = df.with_column(
        ...     pl.col("my_struct").struct.rename_fields(["INT", "STR", "BOOL", "LIST"])
        ... )

        Does NOT work anymore:
        # df.select(pl.col("my_struct").struct.field("int"))
        #               PanicException: int not found ^^^

        >>> df.select(pl.col("my_struct").struct.field("INT"))
        shape: (2, 1)
        ┌─────┐
        │ INT │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.struct_rename_fields(names))


class ExprListNameSpace:
    """Namespace for list related expressions."""

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def lengths(self) -> Expr:
        """
        Get the length of the arrays as UInt32.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2], "bar": [["a", "b"], ["c"]]})
        >>> df.select(pl.col("bar").arr.lengths())
        shape: (2, 1)
        ┌─────┐
        │ bar │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        ├╌╌╌╌╌┤
        │ 1   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arr_lengths())

    def sum(self) -> Expr:
        """Sum all the arrays in the list."""
        return wrap_expr(self._pyexpr.lst_sum())

    def max(self) -> Expr:
        """Compute the max value of the arrays in the list."""
        return wrap_expr(self._pyexpr.lst_max())

    def min(self) -> Expr:
        """Compute the min value of the arrays in the list."""
        return wrap_expr(self._pyexpr.lst_min())

    def mean(self) -> Expr:
        """Compute the mean value of the arrays in the list."""
        return wrap_expr(self._pyexpr.lst_mean())

    def sort(self, reverse: bool = False) -> Expr:
        """
        Sort the arrays in the list

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.sort())
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        ├╌╌╌╌╌╌╌╌╌╌╌┤
        │ [1, 2, 9] │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.lst_sort(reverse))

    def reverse(self) -> Expr:
        """
        Reverse the arrays in the list

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.reverse())
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        ├╌╌╌╌╌╌╌╌╌╌╌┤
        │ [2, 1, 9] │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.lst_reverse())

    def unique(self) -> Expr:
        """Get the unique/distinct values in the list."""
        return wrap_expr(self._pyexpr.lst_unique())

    def concat(
        self, other: List[Expr | str] | Expr | str | pli.Series | List[Any]
    ) -> Expr:
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [["a"], ["x"]],
        ...         "b": [["b", "c"], ["y", "z"]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.concat("b"))
        shape: (2, 1)
        ┌─────────────────┐
        │ a               │
        │ ---             │
        │ list[str]       │
        ╞═════════════════╡
        │ ["a", "b", "c"] │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["x", "y", "z"] │
        └─────────────────┘

        """
        if isinstance(other, list) and (
            not isinstance(other[0], (Expr, str, pli.Series))
        ):
            return self.concat(pli.Series([other]))

        other_list: List[Expr | str | pli.Series]
        if not isinstance(other, list):
            other_list = [other]
        else:
            other_list = copy.copy(other)  # type: ignore[arg-type]

        other_list.insert(0, wrap_expr(self._pyexpr))
        return pli.concat_list(other_list)

    def get(self, index: int) -> Expr:
        """
        Get the value by index in the sublists.
        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sublist

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.get(0))
        shape: (3, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 3    │
        ├╌╌╌╌╌╌┤
        │ null │
        ├╌╌╌╌╌╌┤
        │ 1    │
        └──────┘

        """
        return wrap_expr(self._pyexpr.lst_get(index))

    def first(self) -> Expr:
        """
        Get the first value of the sublists.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.first())
        shape: (3, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 3    │
        ├╌╌╌╌╌╌┤
        │ null │
        ├╌╌╌╌╌╌┤
        │ 1    │
        └──────┘

        """
        return self.get(0)

    def last(self) -> Expr:
        """
        Get the last value of the sublists.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.last())
        shape: (3, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 1    │
        ├╌╌╌╌╌╌┤
        │ null │
        ├╌╌╌╌╌╌┤
        │ 2    │
        └──────┘

        """
        return self.get(-1)

    def contains(self, item: float | str | bool | int | date | datetime | Expr) -> Expr:
        """
        Check if sublists contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Boolean mask

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.contains(1))
        shape: (3, 1)
        ┌───────┐
        │ foo   │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        ├╌╌╌╌╌╌╌┤
        │ false │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        └───────┘

        """
        return wrap_expr(self._pyexpr.arr_contains(expr_to_lit_or_expr(item)._pyexpr))

    def join(self, separator: str) -> Expr:
        """
        Join all string items in a sublist and place a separator between them.
        This errors if inner type of list `!= Utf8`.

        Parameters
        ----------
        separator
            string to separate the items with

        Returns
        -------
        Series of dtype Utf8

        Examples
        --------
        >>> df = pl.DataFrame({"s": [["a", "b", "c"], ["x", "y"]]})
        >>> df.select(pl.col("s").arr.join(" "))
        shape: (2, 1)
        ┌───────┐
        │ s     │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a b c │
        ├╌╌╌╌╌╌╌┤
        │ x y   │
        └───────┘

        """
        return wrap_expr(self._pyexpr.lst_join(separator))

    def arg_min(self) -> Expr:
        """
        Retrieve the index of the minimal value in every sublist

        Returns
        -------
        Series of dtype UInt32/UInt64 (depending on compilation)

        """
        return wrap_expr(self._pyexpr.lst_arg_min())

    def arg_max(self) -> Expr:
        """
        Retrieve the index of the maximum value in every sublist

        Returns
        -------
        Series of dtype UInt32/UInt64 (depending on compilation)

        """
        return wrap_expr(self._pyexpr.lst_arg_max())

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> Expr:
        """
        Calculate the n-th discrete difference of every sublist.

        Parameters
        ----------
        n
            number of slots to shift
        null_behavior
            {'ignore', 'drop'}

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.diff()
        shape: (2,)
        Series: 'a' [list]
        [
            [null, 1, ... 1]
            [null, -8, -1]
        ]

        """
        return wrap_expr(self._pyexpr.lst_diff(n, null_behavior))

    def shift(self, periods: int = 1) -> Expr:
        """
        Shift the values by a given period and fill the parts that will be empty due to
        this operation with nulls.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.shift()
        shape: (2,)
        Series: 'a' [list]
        [
            [null, 1, ... 3]
            [null, 10, 2]
        ]

        """
        return wrap_expr(self._pyexpr.lst_shift(periods))

    def slice(self, offset: int, length: int) -> Expr:
        """
        Slice every sublist

        Parameters
        ----------
        offset
            Take the values from this index offset.
        length
            The length of the slice to take.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.slice(1, 2)
        shape: (2,)
        Series: 'a' [list]
        [
            [2, 3]
            [2, 1]
        ]

        """
        return wrap_expr(self._pyexpr.lst_slice(offset, length))

    def head(self, n: int = 5) -> Expr:
        """
        Slice the head of every sublist

        Parameters
        ----------
        n
            How many values to take in the slice.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.head(2)
        shape: (2,)
        Series: 'a' [list]
        [
            [1, 2]
            [10, 2]
        ]

        """
        return self.slice(0, n)

    def tail(self, n: int = 5) -> Expr:
        """
        Slice the tail of every sublist

        Parameters
        ----------
        n
            How many values to take in the slice.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.tail(2)
        shape: (2,)
        Series: 'a' [list]
        [
            [3, 4]
            [2, 1]
        ]

        """
        return self.slice(-n, n)

    def to_struct(
        self,
        n_field_strategy: str = "first_non_null",
        name_generator: Callable[[int], str] | None = None,
    ) -> Expr:
        """
        Convert the series of type `List` to a series of type `Struct`.

        Parameters
        ----------
        n_field_strategy
            Strategy to determine the number of fields of the struct.
            Any of {'first_non_null', 'max_width'}
        name_generator
            A custom function that can be used to generate the field names.
            Default field names are `field_0, field_1 .. field_n`

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3], [1, 2]]})
        >>> df.select([pl.col("a").arr.to_struct()])
        shape: (2, 1)
        ┌────────────┐
        │ a          │
        │ ---        │
        │ struct[3]  │
        ╞════════════╡
        │ {1,2,3}    │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {1,2,null} │
        └────────────┘
        >>> df.select(
        ...     [
        ...         pl.col("a").arr.to_struct(
        ...             name_generator=lambda idx: f"col_name_{idx}"
        ...         )
        ...     ]
        ... ).to_series().to_list()
        [{'col_name_0': 1, 'col_name_1': 2, 'col_name_2': 3},
        {'col_name_0': 1, 'col_name_1': 2, 'col_name_2': None}]

        """
        return wrap_expr(self._pyexpr.lst_to_struct(n_field_strategy, name_generator))

    def eval(self, expr: Expr, parallel: bool = False) -> Expr:
        """
        Run any polars expression against the lists' elements

        Parameters
        ----------
        expr
            Expression to run. Note that you can select an element with `pl.first()`, or
            `pl.col()`
        parallel
            Run all expression parallel. Don't activate this blindly.
            Parallelism is worth it if there is enough work to do per thread.

            This likely should not be use in the groupby context, because we already
            parallel execution per group

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
        >>> df.with_column(
        ...     pl.concat_list(["a", "b"]).arr.eval(pl.element().rank()).alias("rank")
        ... )
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ a   ┆ b   ┆ rank       │
        │ --- ┆ --- ┆ ---        │
        │ i64 ┆ i64 ┆ list[f32]  │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 4   ┆ [1.0, 2.0] │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 8   ┆ 5   ┆ [2.0, 1.0] │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 2   ┆ [2.0, 1.0] │
        └─────┴─────┴────────────┘

        """
        return wrap_expr(self._pyexpr.lst_eval(expr._pyexpr, parallel))


class ExprStringNameSpace:
    """Namespace for string related expressions."""

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def strptime(
        self,
        datatype: type[Date] | type[Datetime] | type[Time],
        fmt: str | None = None,
        strict: bool = True,
        exact: bool = True,
    ) -> Expr:
        """
        Parse a Utf8 expression to a Date/Datetime/Time type.

        Parameters
        ----------
        datatype
            Date | Datetime | Time.
        fmt
            Format to use, refer to the `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: ``"%y-%m-%d"``.
        strict
            Raise an error if any conversion fails.
        exact
            - If True, require an exact format match.
            - If False, allow the format to match anywhere in the target string.

        Examples
        --------
        Dealing with different formats.

        >>> s = pl.Series(
        ...     "date",
        ...     [
        ...         "2021-04-22",
        ...         "2022-01-04 00:00:00",
        ...         "01/31/22",
        ...         "Sun Jul  8 00:34:60 2001",
        ...     ],
        ... )
        >>> (
        ...     s.to_frame().with_column(
        ...         pl.col("date")
        ...         .str.strptime(pl.Date, "%F", strict=False)
        ...         .fill_null(
        ...             pl.col("date").str.strptime(pl.Date, "%F %T", strict=False)
        ...         )
        ...         .fill_null(pl.col("date").str.strptime(pl.Date, "%D", strict=False))
        ...         .fill_null(pl.col("date").str.strptime(pl.Date, "%c", strict=False))
        ...     )
        ... )
        shape: (4, 1)
        ┌────────────┐
        │ date       │
        │ ---        │
        │ date       │
        ╞════════════╡
        │ 2021-04-22 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2022-01-04 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2022-01-31 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-07-08 │
        └────────────┘

        """
        if not issubclass(datatype, DataType):  # pragma: no cover
            raise ValueError(f"expected: {DataType} got: {datatype}")
        if datatype == Date:
            return wrap_expr(self._pyexpr.str_parse_date(fmt, strict, exact))
        elif datatype == Datetime:
            return wrap_expr(self._pyexpr.str_parse_datetime(fmt, strict, exact))
        elif datatype == Time:
            return wrap_expr(self._pyexpr.str_parse_time(fmt, strict, exact))
        else:  # pragma: no cover
            raise ValueError("dtype should be of type {Date, Datetime, Time}")

    def lengths(self) -> Expr:
        """
        Get the length of the Strings as UInt32.

        Examples
        --------
        >>> df = pl.DataFrame({"s": [None, "bears", "110"]})
        >>> df.select(["s", pl.col("s").str.lengths().alias("len")])
        shape: (3, 2)
        ┌───────┬──────┐
        │ s     ┆ len  │
        │ ---   ┆ ---  │
        │ str   ┆ u32  │
        ╞═══════╪══════╡
        │ null  ┆ null │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ bears ┆ 5    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 110   ┆ 3    │
        └───────┴──────┘

        """
        return wrap_expr(self._pyexpr.str_lengths())

    def concat(self, delimiter: str = "-") -> Expr:
        """
        Vertically concat the values in the Series to a single string value.

        Parameters
        ----------
        delimiter
            The delimiter to insert between consecutive string values.

        Returns
        -------
        Series of dtype Utf8

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, None, 2]})
        >>> df.select(pl.col("foo").str.concat("-"))
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

    def to_uppercase(self) -> Expr:
        """Transform to uppercase variant."""
        return wrap_expr(self._pyexpr.str_to_uppercase())

    def to_lowercase(self) -> Expr:
        """Transform to lowercase variant."""
        return wrap_expr(self._pyexpr.str_to_lowercase())

    def strip(self) -> Expr:
        """Remove leading and trailing whitespace."""
        return wrap_expr(self._pyexpr.str_strip())

    def lstrip(self) -> Expr:
        """Remove leading whitespace."""
        return wrap_expr(self._pyexpr.str_lstrip())

    def rstrip(self) -> Expr:
        """Remove trailing whitespace."""
        return wrap_expr(self._pyexpr.str_rstrip())

    def zfill(self, alignment: int) -> Expr:
        """
        Return a copy of the string left filled with ASCII '0' digits to make a string
        of length width.

        A leading sign prefix ('+'/'-') is handled by inserting the padding after the
        sign character rather than before. The original string is returned if width is
        less than or equal to ``len(s)``.

        Parameters
        ----------
        alignment
            Fill the value up to this length

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "num": [-10, -1, 0, 1, 10, 100, 1000, 10000, 100000, 1000000, None],
        ...     }
        ... )
        >>> df.with_column(pl.col("num").cast(str).str.zfill(5))
        shape: (11, 1)
        ┌─────────┐
        │ num     │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ -0010   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ -0001   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 00000   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 00001   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ ...     │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 10000   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 100000  │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 1000000 │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.str_zfill(alignment))

    def ljust(self, width: int, fillchar: str = " ") -> Expr:
        """
        Return the string left justified in a string of length ``width``.

        Padding is done using the specified ``fillchar``.
        The original string is returned if ``width`` is less than or equal to
        ``len(s)``.

        Parameters
        ----------
        width
            Justify left to this length.
        fillchar
            Fill with this ASCII character.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["cow", "monkey", None, "hippopotamus"]})
        >>> df.select(pl.col("a").str.ljust(8, "*"))
        shape: (4, 1)
        ┌──────────────┐
        │ a            │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ cow*****     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ monkey**     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ hippopotamus │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.str_ljust(width, fillchar))

    def rjust(self, width: int, fillchar: str = " ") -> Expr:
        """
        Return the string right justified in a string of length ``width``.

        Padding is done using the specified ``fillchar``.
        The original string is returned if ``width`` is less than or equal to
        ``len(s)``.

        Parameters
        ----------
        width
            Justify right to this length.
        fillchar
            Fill with this ASCII character.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["cow", "monkey", None, "hippopotamus"]})
        >>> df.select(pl.col("a").str.rjust(8, "*"))
        shape: (4, 1)
        ┌──────────────┐
        │ a            │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ *****cow     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ **monkey     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ hippopotamus │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.str_rjust(width, fillchar))

    def contains(self, pattern: str, literal: bool = False) -> Expr:
        """
        Check if string contains a substring that matches a regex.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        literal
            Treat pattern as a literal string.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["Crab", "cat and dog", "rab$bit", None]})
        >>> df.select(
        ...     [
        ...         pl.col("a"),
        ...         pl.col("a").str.contains("cat|bit").alias("regex"),
        ...         pl.col("a").str.contains("rab$", literal=True).alias("literal"),
        ...     ]
        ... )
        shape: (4, 3)
        ┌─────────────┬───────┬─────────┐
        │ a           ┆ regex ┆ literal │
        │ ---         ┆ ---   ┆ ---     │
        │ str         ┆ bool  ┆ bool    │
        ╞═════════════╪═══════╪═════════╡
        │ Crab        ┆ false ┆ false   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ cat and dog ┆ true  ┆ false   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ rab$bit     ┆ true  ┆ true    │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null  ┆ null    │
        └─────────────┴───────┴─────────┘

        See Also
        --------
        starts_with : Check if string values start with a substring.
        ends_with : Check if string values end with a substring.

        """
        return wrap_expr(self._pyexpr.str_contains(pattern, literal))

    def ends_with(self, sub: str) -> Expr:
        """
        Check if string values end with a substring.

        Parameters
        ----------
        sub
            Suffix substring.

        Examples
        --------
        >>> df = pl.DataFrame({"fruits": ["apple", "mango", None]})
        >>> df.with_column(
        ...     pl.col("fruits").str.ends_with("go").alias("has_suffix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_suffix │
        │ ---    ┆ ---        │
        │ str    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ false      │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ mango  ┆ true       │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null   ┆ null       │
        └────────┴────────────┘

        Using ``ends_with`` as a filter condition:

        >>> df.filter(pl.col("fruits").str.ends_with("go"))
        shape: (1, 1)
        ┌────────┐
        │ fruits │
        │ ---    │
        │ str    │
        ╞════════╡
        │ mango  │
        └────────┘

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        starts_with : Check if string values start with a substring.

        """
        return wrap_expr(self._pyexpr.str_ends_with(sub))

    def starts_with(self, sub: str) -> Expr:
        """
        Check if string values start with a substring.

        Parameters
        ----------
        sub
            Prefix substring.

        Examples
        --------
        >>> df = pl.DataFrame({"fruits": ["apple", "mango", None]})
        >>> df.with_column(
        ...     pl.col("fruits").str.starts_with("app").alias("has_prefix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_prefix │
        │ ---    ┆ ---        │
        │ str    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ true       │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ mango  ┆ false      │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null   ┆ null       │
        └────────┴────────────┘

        Using ``starts_with`` as a filter condition:

        >>> df.filter(pl.col("fruits").str.starts_with("app"))
        shape: (1, 1)
        ┌────────┐
        │ fruits │
        │ ---    │
        │ str    │
        ╞════════╡
        │ apple  │
        └────────┘

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        ends_with : Check if string values end with a substring.

        """
        return wrap_expr(self._pyexpr.str_starts_with(sub))

    def json_path_match(self, json_path: str) -> Expr:
        """
        Extract the first match of json string with provided JSONPath expression.
        Throw errors if encounter invalid json strings.
        All return value will be casted to Utf8 regardless of the original value.

        Documentation on JSONPath standard can be found
        `here <https://goessner.net/articles/JsonPath/>`_.

        Parameters
        ----------
        json_path
            A valid JSON path query string.

        Returns
        -------
        Utf8 array. Contain null if original value is null or the json_path return
        nothing.

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

    def decode(self, encoding: Literal["hex", "base64"], strict: bool = False) -> Expr:
        """
        Decode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            How to handle invalid inputs:

            - ``True``: An error will be thrown if unable to decode a value.
            - ``False``: Unhandled values will be replaced with `None`.

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
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def encode(self, encoding: Literal["hex", "base64"]) -> Expr:
        """
        Encode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

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
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

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

    def extract_all(self, pattern: str) -> Expr:
        r"""
        Extract each successive non-overlapping regex match in an individual string as
        an array.

        Parameters
        ----------
        pattern
            A valid regex pattern

        Returns
        -------
        List[Utf8] array. Contain null if original value is null or regex capture
        nothing.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
        >>> df.select(
        ...     [
        ...         pl.col("foo").str.extract_all(r"(\d+)").alias("extracted_nrs"),
        ...     ]
        ... )
        shape: (2, 1)
        ┌────────────────┐
        │ extracted_nrs  │
        │ ---            │
        │ list[str]      │
        ╞════════════════╡
        │ ["123", "45"]  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["678", "910"] │
        └────────────────┘

        """
        return wrap_expr(self._pyexpr.str_extract_all(pattern))

    def count_match(self, pattern: str) -> Expr:
        r"""
        Count all successive non-overlapping regex matches.

        Parameters
        ----------
        pattern
            A valid regex pattern

        Returns
        -------
        UInt32 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
        >>> df.select(
        ...     [
        ...         pl.col("foo").str.count_match(r"\d").alias("count_digits"),
        ...     ]
        ... )
        shape: (2, 1)
        ┌──────────────┐
        │ count_digits │
        │ ---          │
        │ u32          │
        ╞══════════════╡
        │ 5            │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 6            │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.count_match(pattern))

    def split(self, by: str, inclusive: bool = False) -> Expr:
        """
        Split the string by a substring.

        Parameters
        ----------
        by
            Substring to split by.
        inclusive
            If True, include the split character/string in the results.

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["foo bar", "foo-bar", "foo bar baz"]})
        >>> df.select(pl.col("s").str.split(by=" "))
        shape: (3, 1)
        ┌───────────────────────┐
        │ s                     │
        │ ---                   │
        │ list[str]             │
        ╞═══════════════════════╡
        │ ["foo", "bar"]        │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["foo-bar"]           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["foo", "bar", "baz"] │
        └───────────────────────┘

        Returns
        -------
        List of Utf8 type

        """
        if inclusive:
            return wrap_expr(self._pyexpr.str_split_inclusive(by))
        return wrap_expr(self._pyexpr.str_split(by))

    def split_exact(self, by: str, n: int, inclusive: bool = False) -> Expr:
        """
        Split the string by a substring into a struct of ``n`` fields.

        If it cannot make ``n`` splits, the remaining field elements will be null.

        Parameters
        ----------
        by
            Substring to split by.
        n
            Number of splits to make.
        inclusive
            If True, include the split character/string in the results.

        Examples
        --------
        >>> (
        ...     pl.DataFrame({"x": ["a_1", None, "c", "d_4"]}).select(
        ...         [
        ...             pl.col("x").str.split_exact("_", 1).alias("fields"),
        ...         ]
        ...     )
        ... )
        shape: (4, 1)
        ┌─────────────┐
        │ fields      │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {"a","1"}   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {null,null} │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {"c",null}  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {"d","4"}   │
        └─────────────┘


        Split string values in column x in exactly 2 parts and assign
        each part to a new column.

        >>> pl.DataFrame({"x": ["a_1", None, "c", "d_4"]}).with_columns(
        ...     [
        ...         pl.col("x")
        ...         .str.split_exact("_", 1)
        ...         .struct.rename_fields(["first_part", "second_part"])
        ...         .alias("fields"),
        ...     ]
        ... ).unnest("fields")
        shape: (4, 3)
        ┌──────┬────────────┬─────────────┐
        │ x    ┆ first_part ┆ second_part │
        │ ---  ┆ ---        ┆ ---         │
        │ str  ┆ str        ┆ str         │
        ╞══════╪════════════╪═════════════╡
        │ a_1  ┆ a          ┆ 1           │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ null       ┆ null        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ c    ┆ c          ┆ null        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ d_4  ┆ d          ┆ 4           │
        └──────┴────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        if inclusive:
            return wrap_expr(self._pyexpr.str_split_exact_inclusive(by, n))
        return wrap_expr(self._pyexpr.str_split_exact(by, n))

    def replace(self, pattern: str, value: str, literal: bool = False) -> Expr:
        r"""
        Replace first matching regex/literal substring with a new string value.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        literal
             Treat pattern as a literal string.

        See Also
        --------
        replace_all : Replace all matching regex/literal substrings.

        Examples
        --------
        >>> df = pl.DataFrame({"id": [1, 2], "text": ["123abc", "abc456"]})
        >>> df.with_column(
        ...     pl.col("text").str.replace(r"abc\b", "ABC")
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌─────┬────────┐
        │ id  ┆ text   │
        │ --- ┆ ---    │
        │ i64 ┆ str    │
        ╞═════╪════════╡
        │ 1   ┆ 123ABC │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2   ┆ abc456 │
        └─────┴────────┘

        """
        return wrap_expr(self._pyexpr.str_replace(pattern, value, literal))

    def replace_all(self, pattern: str, value: str, literal: bool = False) -> Expr:
        """
        Replace all matching regex/literal substrings with a new string value.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        literal
             Treat pattern as a literal string.

        See Also
        --------
        replace : Replace first matching regex/literal substring.

        Examples
        --------
        >>> df = pl.DataFrame({"id": [1, 2], "text": ["abcabc", "123a123"]})
        >>> df.with_column(pl.col("text").str.replace_all("a", "-"))
        shape: (2, 2)
        ┌─────┬─────────┐
        │ id  ┆ text    │
        │ --- ┆ ---     │
        │ i64 ┆ str     │
        ╞═════╪═════════╡
        │ 1   ┆ -bc-bc  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ 123-123 │
        └─────┴─────────┘

        """
        return wrap_expr(self._pyexpr.str_replace_all(pattern, value, literal))

    def slice(self, start: int, length: int | None = None) -> Expr:
        """
        Create subslices of the string values of a Utf8 Series.

        Parameters
        ----------
        start
            Starting index of the slice (zero-indexed). Negative indexing
            may be used.
        length
            Optional length of the slice. If None (default), the slice is taken to the
            end of the string.

        Returns
        -------
        Series of Utf8 type

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["pear", None, "papaya", "dragonfruit"]})
        >>> df.with_column(
        ...     pl.col("s").str.slice(-3).alias("s_sliced"),
        ... )
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ s           ┆ s_sliced │
        │ ---         ┆ ---      │
        │ str         ┆ str      │
        ╞═════════════╪══════════╡
        │ pear        ┆ ear      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ papaya      ┆ aya      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ dragonfruit ┆ uit      │
        └─────────────┴──────────┘

        Using the optional `length` parameter

        >>> df.with_column(
        ...     pl.col("s").str.slice(4, length=3).alias("s_sliced"),
        ... )
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ s           ┆ s_sliced │
        │ ---         ┆ ---      │
        │ str         ┆ str      │
        ╞═════════════╪══════════╡
        │ pear        ┆          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ papaya      ┆ ya       │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ dragonfruit ┆ onf      │
        └─────────────┴──────────┘

        """
        return wrap_expr(self._pyexpr.str_slice(start, length))


class ExprDateTimeNameSpace:
    """Namespace for datetime related expressions."""

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def truncate(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> Expr:
        """
        Divide the date/ datetime range into buckets.

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
        Format Date/datetime with a formatting rule.

        See `chrono strftime/strptime
        <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_.

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

    def quarter(self) -> Expr:
        """
        Extract quarter from underlying Date representation.

        Can be performed on Date and Datetime.

        Returns the quarter ranging from 1 to 4.

        Returns
        -------
        Quarter as UInt32

        """
        return wrap_expr(self._pyexpr.quarter())

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
        """Go from Date/Datetime to python DateTime objects."""
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.to_python_datetime(), return_dtype=Object
        )

    def epoch(self, tu: str = "us") -> Expr:
        """
        Get the time passed since the Unix EPOCH in the give time unit

        Parameters
        ----------
        tu
            One of {'ns', 'us', 'ms', 's', 'd'}

        """
        if tu in DTYPE_TEMPORAL_UNITS:
            return self.timestamp(tu)
        if tu == "s":
            return wrap_expr(self._pyexpr.dt_epoch_seconds())
        if tu == "d":
            return wrap_expr(self._pyexpr).cast(Date).cast(Int32)
        else:
            raise ValueError(f"time unit {tu} not understood")

    def epoch_days(self) -> Expr:
        """
        Get the number of days since the unix EPOCH.
        If the date is before the unix EPOCH, the number of days will be negative.

        .. deprecated:: 0.13.9
            Use :func:`epoch` instead.

        Returns
        -------
        Days as Int32

        """
        return wrap_expr(self._pyexpr).cast(Date).cast(Int32)

    def epoch_milliseconds(self) -> Expr:
        """
        Get the number of milliseconds since the unix EPOCH.

        If the date is before the unix EPOCH, the number of milliseconds will be
        negative.

        .. deprecated:: 0.13.9
            Use :func:`epoch` instead.

        Returns
        -------
        Milliseconds as Int64

        """
        return self.timestamp("ms")

    def epoch_seconds(self) -> Expr:
        """
        Get the number of seconds since the unix EPOCH
        If the date is before the unix EPOCH, the number of seconds will be negative.

        .. deprecated:: 0.13.9
            Use :func:`epoch` instead.

        Returns
        -------
        Milliseconds as Int64

        """
        return wrap_expr(self._pyexpr.dt_epoch_seconds())

    def timestamp(self, tu: str = "us") -> Expr:
        """
        Return a timestamp in the given time unit.

        Parameters
        ----------
        tu
            One of {'ns', 'us', 'ms'}

        """
        return wrap_expr(self._pyexpr.timestamp(tu))

    def with_time_unit(self, tu: str) -> Expr:
        """
        Set time unit a Series of dtype Datetime or Duration.

        This does not modify underlying data, and should be used to fix an incorrect
        time unit.

        Parameters
        ----------
        tu
            Time unit for the `Datetime` Series: one of {"ns", "us", "ms"}

        """
        return wrap_expr(self._pyexpr.dt_with_time_unit(tu))

    def cast_time_unit(self, tu: str) -> Expr:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        tu
            Time unit for the `Datetime` Series: any of {"ns", "us", "ms"}

        """
        return wrap_expr(self._pyexpr.dt_cast_time_unit(tu))

    def and_time_unit(self, tu: str, dtype: type[DataType] = Datetime) -> Expr:
        """
        Set time unit a Series of type Datetime. This does not modify underlying data,
        and should be used to fix an incorrect time unit.

        .. deprecated::
            Use :func:`with_time_unit` instead.

        Parameters
        ----------
        tu
            Time unit for the `Datetime` Series: any of {"ns", "us", "ms"}
        dtype
            Output data type.

        """
        return self.with_time_unit(tu)

    def and_time_zone(self, tz: str | None) -> Expr:
        """
        Set time zone for a Series of type Datetime.

        .. deprecated::
            Use :func:`with_time_zone` instead.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.with_time_zone(tz), return_dtype=Datetime
        )

    def with_time_zone(self, tz: str | None) -> Expr:
        """
        Set time zone for a Series of type Datetime.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """
        return wrap_expr(self._pyexpr).map(
            lambda s: s.dt.with_time_zone(tz), return_dtype=Datetime
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

    def minutes(self) -> Expr:
        """
        Extract the minutes from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return wrap_expr(self._pyexpr.duration_minutes())

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

    def offset_by(self, by: str) -> Expr:
        """
        Offset this date by a relative time offset.

         This differs from `pl.col("foo") + timedelta` in that it can
         take months and leap years into account

        Parameters
        ----------
        by
            The offset is dictated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

        Returns
        -------
        Date/Datetime expression

        """
        return wrap_expr(self._pyexpr.dt_offset_by(by))


def expr_to_lit_or_expr(
    expr: (
        Expr
        | bool
        | int
        | float
        | str
        | pli.Series
        | None
        | date
        | datetime
        | Sequence[(int | float | str | None)]
    ),
    str_to_lit: bool = True,
) -> Expr:
    """
    Convert args to expressions.

    Parameters
    ----------
    expr
        Any argument.
    str_to_lit
        If True string argument `"foo"` will be converted to `lit("foo")`,
        If False it will be converted to `col("foo")`

    Returns
    -------
    Expr

    """
    if isinstance(expr, str) and not str_to_lit:
        return pli.col(expr)
    elif (
        isinstance(expr, (int, float, str, pli.Series, datetime, date)) or expr is None
    ):
        return pli.lit(expr)
    elif isinstance(expr, Expr):
        return expr
    elif isinstance(expr, list):
        return pli.lit(pli.Series("", [expr]))
    else:
        raise ValueError(
            f"did not expect value {expr} of type {type(expr)}, maybe disambiguate with"
            " pl.lit or pl.col"
        )


class ExprCatNameSpace:
    """Namespace for categorical related expressions."""

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def set_ordering(self, ordering: str) -> Expr:
        """
        Determine how this categorical series should be sorted.

        Parameters
        ----------
        ordering
            One of:
                - 'physical' -> use the physical representation of the categories to
                    determine the order (default)
                - 'lexical' -. use the string values to determine the ordering

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"cats": ["z", "z", "k", "a", "b"], "vals": [3, 1, 2, 2, 3]}
        ... ).with_columns(
        ...     [
        ...         pl.col("cats").cast(pl.Categorical).cat.set_ordering("lexical"),
        ...     ]
        ... )
        >>> df.sort(["cats", "vals"])
        shape: (5, 2)
        ┌──────┬──────┐
        │ cats ┆ vals │
        │ ---  ┆ ---  │
        │ cat  ┆ i64  │
        ╞══════╪══════╡
        │ a    ┆ 2    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ b    ┆ 3    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ k    ┆ 2    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ z    ┆ 1    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ z    ┆ 3    │
        └──────┴──────┘

        """
        return wrap_expr(self._pyexpr.cat_set_ordering(ordering))


def _prepare_alpha(
    com: float | None = None,
    span: float | None = None,
    half_life: float | None = None,
    alpha: float | None = None,
) -> float:
    if com is not None and alpha is None:
        assert com >= 0.0
        alpha = 1.0 / (1.0 + com)
    if span is not None and alpha is None:
        assert span >= 1.0
        alpha = 2.0 / (span + 1.0)
    if half_life is not None and alpha is None:
        assert half_life > 0.0
        alpha = 1.0 - math.exp(-math.log(2.0) / half_life)
    if alpha is None:
        raise ValueError("at least one of {com, span, half_life, alpha} should be set")
    return alpha


def _prepare_rolling_window_args(
    window_size: int | str,
    min_periods: int | None = None,
) -> tuple[str, int]:
    if isinstance(window_size, int):
        if min_periods is None:
            min_periods = window_size
        window_size = f"{window_size}i"
    if min_periods is None:
        min_periods = 1
    return (window_size, min_periods)
