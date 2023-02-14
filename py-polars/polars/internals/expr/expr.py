from __future__ import annotations

import math
import os
import random
import warnings
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, NoReturn, Sequence, cast

from polars import internals as pli
from polars.datatypes import (
    DataType,
    PolarsDataType,
    Struct,
    UInt32,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np
from polars.internals.expr.binary import ExprBinaryNameSpace
from polars.internals.expr.categorical import ExprCatNameSpace
from polars.internals.expr.datetime import ExprDateTimeNameSpace
from polars.internals.expr.list import ExprListNameSpace
from polars.internals.expr.meta import ExprMetaNameSpace
from polars.internals.expr.string import ExprStringNameSpace
from polars.internals.expr.struct import ExprStructNameSpace
from polars.internals.type_aliases import PythonLiteral
from polars.utils import (
    _timedelta_to_pl_duration,
    deprecate_nonkeyword_arguments,
    sphinx_accessor,
)

try:
    from polars.polars import PyExpr

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if TYPE_CHECKING:
    from polars.internals.type_aliases import (
        ClosedInterval,
        FillNullStrategy,
        InterpolationMethod,
        IntoExpr,
        NullBehavior,
        RankMethod,
        RollingInterpolationMethod,
        SearchSortedSide,
    )
elif os.getenv("BUILDING_SPHINX_DOCS"):
    property = sphinx_accessor


def selection_to_pyexpr_list(
    exprs: IntoExpr | Iterable[IntoExpr],
    structify: bool = False,
) -> list[PyExpr]:
    if exprs is None:
        exprs = []
    elif isinstance(exprs, (str, Expr, pli.Series, pli.WhenThen, pli.WhenThenThen)):
        exprs = [exprs]
    elif not isinstance(exprs, Iterable):
        exprs = [exprs]
    return [
        expr_to_lit_or_expr(e, str_to_lit=False, structify=structify)._pyexpr
        for e in exprs  # type: ignore[union-attr]
    ]


def expr_output_name(expr: pli.Expr) -> str | None:
    try:
        return expr.meta.output_name()
    except Exception:
        return None


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
    if isinstance(expr, str) and not str_to_lit:
        expr = pli.col(expr)
    elif (
        isinstance(expr, (int, float, str, pli.Series, datetime, date, time, timedelta))
        or expr is None
    ):
        expr = pli.lit(expr)
        structify = False
    elif isinstance(expr, list):
        expr = pli.lit(pli.Series("", [expr]))
        structify = False
    elif isinstance(expr, (pli.WhenThen, pli.WhenThenThen)):
        expr = expr.otherwise(None)  # implicitly add the null branch.
    elif not isinstance(expr, Expr):
        raise TypeError(
            f"did not expect value {expr} of type {type(expr)}, maybe disambiguate with"
            " pl.lit or pl.col"
        )

    if structify:
        unaliased_expr = expr.meta.undo_aliases()
        if unaliased_expr.meta.has_multiple_outputs():
            expr_name = expr_output_name(expr)
            expr = cast(Expr, pli.struct(expr if expr_name is None else unaliased_expr))
            name = name or expr_name

    return expr if name is None else expr.alias(name)


def wrap_expr(pyexpr: PyExpr) -> Expr:
    return Expr._from_pyexpr(pyexpr)


class Expr:
    """Expressions that can be used in various contexts."""

    _pyexpr: PyExpr = None
    _accessors: set[str] = {"arr", "cat", "dt", "meta", "str", "bin", "struct"}

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:
        expr = cls.__new__(cls)
        expr._pyexpr = pyexpr
        return expr

    def _to_pyexpr(self, other: Any) -> PyExpr:
        return self._to_expr(other)._pyexpr

    def _to_expr(self, other: Any) -> Expr:
        if isinstance(other, Expr):
            return other
        return pli.lit(other)

    def _repr_html_(self) -> str:
        return self._pyexpr.to_str()

    def __str__(self) -> str:
        return self._pyexpr.to_str()

    def __bool__(self) -> NoReturn:
        raise ValueError(
            "Since Expr are lazy, the truthiness of an Expr is ambiguous. "
            "Hint: use '&' or '|' to logically combine Expr, not 'and'/'or', and "
            "use 'x.is_in([y,z])' instead of 'x in [y,z]' to check membership."
        )

    def __abs__(self) -> Expr:
        return self.abs()

    def __invert__(self) -> Expr:
        return self.is_not()

    def __xor__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._xor(self._to_pyexpr(other)))

    def __rxor__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._xor(self._to_pyexpr(other)))

    def __and__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._and(self._to_pyexpr(other)))

    def __rand__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr._and(self._to_pyexpr(other)))

    def __or__(self, other: Expr) -> Expr:
        return wrap_expr(self._pyexpr._or(self._to_pyexpr(other)))

    def __ror__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other)._or(self._pyexpr))

    def __add__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr + self._to_pyexpr(other))

    def __radd__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other) + self._pyexpr)

    def __sub__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr - self._to_pyexpr(other))

    def __rsub__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other) - self._pyexpr)

    def __mul__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr * self._to_pyexpr(other))

    def __rmul__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other) * self._pyexpr)

    def __truediv__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr / self._to_pyexpr(other))

    def __rtruediv__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other) / self._pyexpr)

    def __floordiv__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr // self._to_pyexpr(other))

    def __rfloordiv__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other) // self._pyexpr)

    def __mod__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr % self._to_pyexpr(other))

    def __rmod__(self, other: Any) -> Expr:
        return wrap_expr(self._to_pyexpr(other) % self._pyexpr)

    def __pow__(self, power: int | float | pli.Series | Expr) -> Expr:
        return self.pow(power)

    def __rpow__(self, base: int | float | Expr) -> Expr:
        return pli.expr_to_lit_or_expr(base) ** self

    def __ge__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr.gt_eq(self._to_expr(other)._pyexpr))

    def __le__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr.lt_eq(self._to_expr(other)._pyexpr))

    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return wrap_expr(self._pyexpr.eq(self._to_expr(other)._pyexpr))

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return wrap_expr(self._pyexpr.neq(self._to_expr(other)._pyexpr))

    def __lt__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr.lt(self._to_expr(other)._pyexpr))

    def __gt__(self, other: Any) -> Expr:
        return wrap_expr(self._pyexpr.gt(self._to_expr(other)._pyexpr))

    def ge(self, other: Any) -> Expr:
        return self.__ge__(other)

    def le(self, other: Any) -> Expr:
        return self.__le__(other)

    def eq(self, other: Any) -> Expr:
        return self.__eq__(other)

    def ne(self, other: Any) -> Expr:
        return self.__ne__(other)

    def lt(self, other: Any) -> Expr:
        return self.__lt__(other)

    def gt(self, other: Any) -> Expr:
        return self.__gt__(other)

    def __neg__(self) -> Expr:
        return pli.lit(0) - self

    def __pos__(self) -> Expr:
        return pli.lit(0) + self

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Expr:
        """Numpy universal functions."""

        num_expr = sum(isinstance(inp, Expr) for inp in inputs)
        if num_expr > 1:
            raise ValueError(
                f"Numpy ufunc can only be used with one expression, {num_expr} given. Use `pl.reduce` to call numpy functions over multiple expressions."
            )

        def function(s: pli.Series) -> pli.Series:  # pragma: no cover
            args = [inp if not isinstance(inp, Expr) else s for inp in inputs]
            return ufunc(*args, **kwargs)

        return self.map(function)

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
        │ x    ┆ 1             │
        │ null ┆ null          │
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
        """
        Compute the square root of the elements.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").sqrt())
        shape: (3, 1)
        ┌──────────┐
        │ values   │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.0      │
        │ 1.414214 │
        │ 2.0      │
        └──────────┘

        """
        return self**0.5

    def log10(self) -> Expr:
        """
        Compute the base 10 logarithm of the input array, element-wise.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").log10())
        shape: (3, 1)
        ┌─────────┐
        │ values  │
        │ ---     │
        │ f64     │
        ╞═════════╡
        │ 0.0     │
        │ 0.30103 │
        │ 0.60206 │
        └─────────┘

        """
        return self.log(10.0)

    def exp(self) -> Expr:
        """
        Compute the exponential, element-wise.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").exp())
        shape: (3, 1)
        ┌──────────┐
        │ values   │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 2.718282 │
        │ 7.389056 │
        │ 54.59815 │
        └──────────┘

        """
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
        │ 2   ┆ b    │
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
        │ 2   ┆ b    │
        │ 3   ┆ null │
        └─────┴──────┘

        """
        return wrap_expr(self._pyexpr.alias(name))

    def exclude(
        self,
        columns: (
            str
            | PolarsDataType
            | Sequence[str]
            | Sequence[PolarsDataType]
            | set[PolarsDataType]
            | frozenset[PolarsDataType]
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
        │ 2   ┆ b    ┆ 2.5  │
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
        │ 2   ┆ 2.5  │
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
        │ 2.5  │
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
        │ b    │
        │ null │
        └──────┘

        """
        if isinstance(columns, str):
            columns = [columns]
            return wrap_expr(self._pyexpr.exclude(columns))
        elif isinstance(columns, (set, frozenset)):
            return wrap_expr(self._pyexpr.exclude_dtype(list(columns)))
        elif not isinstance(columns, Sequence) or isinstance(columns, DataType):
            columns = [columns]
            return wrap_expr(self._pyexpr.exclude_dtype(columns))

        if not all((isinstance(a, str) or is_polars_dtype(a)) for a in columns):
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2],
        ...         "b": [3, 4],
        ...     }
        ... )

        Keep original column name to undo an alias operation.

        >>> df.with_columns([(pl.col("a") * 9).alias("c").keep_name()])
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 9   ┆ 3   │
        │ 18  ┆ 4   │
        └─────┴─────┘

        Prevent
        "DuplicateError: Column with name: 'literal' has more than one occurrences"
        errors.

        >>> df.select([(pl.lit(10) / pl.all()).keep_name()])
        shape: (2, 2)
        ┌──────┬──────────┐
        │ a    ┆ b        │
        │ ---  ┆ ---      │
        │ f64  ┆ f64      │
        ╞══════╪══════════╡
        │ 10.0 ┆ 3.333333 │
        │ 5.0  ┆ 2.5      │
        └──────┴──────────┘

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
        │ 2   ┆ banana ┆ 4   ┆ audi   │
        │ 3   ┆ apple  ┆ 3   ┆ beetle │
        │ 4   ┆ apple  ┆ 2   ┆ beetle │
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
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """  # noqa: W505
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
        │ 2   ┆ banana ┆ 4   ┆ audi   │
        │ 3   ┆ apple  ┆ 3   ┆ beetle │
        │ 4   ┆ apple  ┆ 2   ┆ beetle │
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
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """  # noqa: W505
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
        │ false ┆ b    │
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
        │ true  │
        │ true  │
        └───────┘

        """
        return wrap_expr(self._pyexpr.is_not())

    def is_null(self) -> Expr:
        """
        Returns a boolean Series indicating which values are null.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.all().is_null().suffix("_isnull"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬──────────┬──────────┐
        │ a    ┆ b   ┆ a_isnull ┆ b_isnull │
        │ ---  ┆ --- ┆ ---      ┆ ---      │
        │ i64  ┆ f64 ┆ bool     ┆ bool     │
        ╞══════╪═════╪══════════╪══════════╡
        │ 1    ┆ 1.0 ┆ false    ┆ false    │
        │ 2    ┆ 2.0 ┆ false    ┆ false    │
        │ null ┆ NaN ┆ true     ┆ false    │
        │ 1    ┆ 1.0 ┆ false    ┆ false    │
        │ 5    ┆ 5.0 ┆ false    ┆ false    │
        └──────┴─────┴──────────┴──────────┘

        """
        return wrap_expr(self._pyexpr.is_null())

    def is_not_null(self) -> Expr:
        """
        Returns a boolean Series indicating which values are not null.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.all().is_not_null().suffix("_not_null"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬────────────┬────────────┐
        │ a    ┆ b   ┆ a_not_null ┆ b_not_null │
        │ ---  ┆ --- ┆ ---        ┆ ---        │
        │ i64  ┆ f64 ┆ bool       ┆ bool       │
        ╞══════╪═════╪════════════╪════════════╡
        │ 1    ┆ 1.0 ┆ true       ┆ true       │
        │ 2    ┆ 2.0 ┆ true       ┆ true       │
        │ null ┆ NaN ┆ false      ┆ true       │
        │ 1    ┆ 1.0 ┆ true       ┆ true       │
        │ 5    ┆ 5.0 ┆ true       ┆ true       │
        └──────┴─────┴────────────┴────────────┘

        """
        return wrap_expr(self._pyexpr.is_not_null())

    def is_finite(self) -> Expr:
        """
        Returns a boolean Series indicating which values are finite.

        Returns
        -------
        out
            Series of type Boolean

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2],
        ...         "B": [3.0, float("inf")],
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
        │ true ┆ false │
        └──────┴───────┘

        """
        return wrap_expr(self._pyexpr.is_finite())

    def is_infinite(self) -> Expr:
        """
        Returns a boolean Series indicating which values are infinite.

        Returns
        -------
        out
            Series of type Boolean

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2],
        ...         "B": [3.0, float("inf")],
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
        │ false ┆ true  │
        └───────┴───────┘

        """
        return wrap_expr(self._pyexpr.is_infinite())

    def is_nan(self) -> Expr:
        """
        Returns a boolean Series indicating which values are NaN.

        Notes
        -----
        Floating point ```NaN`` (Not A Number) should not be confused
        with missing data represented as ``Null/None``.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.col(pl.Float64).is_nan().suffix("_isnan"))
        shape: (5, 3)
        ┌──────┬─────┬─────────┐
        │ a    ┆ b   ┆ b_isnan │
        │ ---  ┆ --- ┆ ---     │
        │ i64  ┆ f64 ┆ bool    │
        ╞══════╪═════╪═════════╡
        │ 1    ┆ 1.0 ┆ false   │
        │ 2    ┆ 2.0 ┆ false   │
        │ null ┆ NaN ┆ true    │
        │ 1    ┆ 1.0 ┆ false   │
        │ 5    ┆ 5.0 ┆ false   │
        └──────┴─────┴─────────┘

        """
        return wrap_expr(self._pyexpr.is_nan())

    def is_not_nan(self) -> Expr:
        """
        Returns a boolean Series indicating which values are not NaN.

        Notes
        -----
        Floating point ```NaN`` (Not A Number) should not be confused
        with missing data represented as ``Null/None``.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.col(pl.Float64).is_not_nan().suffix("_is_not_nan"))
        shape: (5, 3)
        ┌──────┬─────┬──────────────┐
        │ a    ┆ b   ┆ b_is_not_nan │
        │ ---  ┆ --- ┆ ---          │
        │ i64  ┆ f64 ┆ bool         │
        ╞══════╪═════╪══════════════╡
        │ 1    ┆ 1.0 ┆ true         │
        │ 2    ┆ 2.0 ┆ true         │
        │ null ┆ NaN ┆ false        │
        │ 1    ┆ 1.0 ┆ true         │
        │ 5    ┆ 5.0 ┆ true         │
        └──────┴─────┴──────────────┘

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
        │ two   ┆ [3, 4, 5] │
        └───────┴───────────┘

        """
        return wrap_expr(self._pyexpr.agg_groups())

    def count(self) -> Expr:
        """
        Count the number of values in this expression.

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
        Count the number of values in this expression.

        Alias for :func:`count`.

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

    def slice(self, offset: int | Expr, length: int | Expr | None = None) -> Expr:
        """
        Get a slice of this expression.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None``, all rows starting at the offset
            will be selected.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10, 11],
        ...         "b": [None, 4, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.all().slice(1, 2))
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 9   ┆ 4   │
        │ 10  ┆ 4   │
        └─────┴─────┘

        """
        if not isinstance(offset, Expr):
            offset = pli.lit(offset)
        if not isinstance(length, Expr):
            length = pli.lit(length)
        return wrap_expr(self._pyexpr.slice(offset._pyexpr, length._pyexpr))

    def append(self, other: Expr, upcast: bool = True) -> Expr:
        """
        Append expressions.

        This is done by adding the chunks of `other` to this `Series`.

        Parameters
        ----------
        other
            Expression to append.
        upcast
            Cast both `Series` to the same supertype.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10],
        ...         "b": [None, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.all().head(1).append(pl.all().tail(1)))
        shape: (2, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ i64  │
        ╞═════╪══════╡
        │ 8   ┆ null │
        │ 10  ┆ 4    │
        └─────┴──────┘

        """
        other = expr_to_lit_or_expr(other)
        return wrap_expr(self._pyexpr.append(other._pyexpr, upcast))

    def rechunk(self) -> Expr:
        """
        Create a single chunk of memory for this Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})

        Create a Series with 3 nulls, append column a then rechunk

        >>> df.select(pl.repeat(None, 3).append(pl.col("a")).rechunk())
        shape: (6, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ null    │
        │ null    │
        │ null    │
        │ 1       │
        │ 1       │
        │ 2       │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.rechunk())

    def drop_nulls(self) -> Expr:
        """
        Drop null values.

        Warnings
        --------
        Note that null values are not floating point NaN values!
        To drop NaN values, use :func:`drop_nans`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10, 11],
        ...         "b": [None, 4.0, 4.0, float("nan")],
        ...     }
        ... )
        >>> df.select(pl.col("b").drop_nulls())
        shape: (3, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        │ 4.0 │
        │ NaN │
        └─────┘

        """
        return wrap_expr(self._pyexpr.drop_nulls())

    def drop_nans(self) -> Expr:
        """
        Drop floating point NaN values.

        Warnings
        --------
        Note that NaN values are not null values!
        To drop null values, use :func:`drop_nulls`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10, 11],
        ...         "b": [None, 4.0, 4.0, float("nan")],
        ...     }
        ... )
        >>> df.select(pl.col("b").drop_nans())
        shape: (3, 1)
        ┌──────┐
        │ b    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 4.0  │
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
        │ 3   ┆ 9         │
        │ 6   ┆ 7         │
        │ 10  ┆ 4         │
        └─────┴───────────┘

        Null values are excluded, but can also be filled by calling ``forward_fill``.

        >>> df = pl.DataFrame({"values": [None, 10, None, 8, 9, None, 16, None]})
        >>> df.with_columns(
        ...     [
        ...         pl.col("values").cumsum().alias("value_cumsum"),
        ...         pl.col("values")
        ...         .cumsum()
        ...         .forward_fill()
        ...         .alias("value_cumsum_all_filled"),
        ...     ]
        ... )
        shape: (8, 3)
        ┌────────┬──────────────┬─────────────────────────┐
        │ values ┆ value_cumsum ┆ value_cumsum_all_filled │
        │ ---    ┆ ---          ┆ ---                     │
        │ i64    ┆ i64          ┆ i64                     │
        ╞════════╪══════════════╪═════════════════════════╡
        │ null   ┆ null         ┆ null                    │
        │ 10     ┆ 10           ┆ 10                      │
        │ null   ┆ null         ┆ 10                      │
        │ 8      ┆ 18           ┆ 18                      │
        │ 9      ┆ 27           ┆ 27                      │
        │ null   ┆ null         ┆ 27                      │
        │ 16     ┆ 43           ┆ 43                      │
        │ null   ┆ null         ┆ 43                      │
        └────────┴──────────────┴─────────────────────────┘

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
        │ 2   ┆ 24        │
        │ 6   ┆ 12        │
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
        │ 1   ┆ 2         │
        │ 1   ┆ 3         │
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
        │ 2   ┆ 4         │
        │ 3   ┆ 4         │
        │ 4   ┆ 4         │
        └─────┴───────────┘

        Null values are excluded, but can also be filled by calling ``forward_fill``.

        >>> df = pl.DataFrame({"values": [None, 10, None, 8, 9, None, 16, None]})
        >>> df.with_columns(
        ...     [
        ...         pl.col("values").cummax().alias("value_cummax"),
        ...         pl.col("values")
        ...         .cummax()
        ...         .forward_fill()
        ...         .alias("value_cummax_all_filled"),
        ...     ]
        ... )
        shape: (8, 3)
        ┌────────┬──────────────┬─────────────────────────┐
        │ values ┆ value_cummax ┆ value_cummax_all_filled │
        │ ---    ┆ ---          ┆ ---                     │
        │ i64    ┆ i64          ┆ i64                     │
        ╞════════╪══════════════╪═════════════════════════╡
        │ null   ┆ null         ┆ null                    │
        │ 10     ┆ 10           ┆ 10                      │
        │ null   ┆ null         ┆ 10                      │
        │ 8      ┆ 10           ┆ 10                      │
        │ 9      ┆ 10           ┆ 10                      │
        │ null   ┆ null         ┆ 10                      │
        │ 16     ┆ 16           ┆ 16                      │
        │ null   ┆ null         ┆ 16                      │
        └────────┴──────────────┴─────────────────────────┘

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
        │ 1   ┆ 2         │
        │ 2   ┆ 1         │
        │ 3   ┆ 0         │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.cumcount(reverse))

    def floor(self) -> Expr:
        """
        Rounds down to the nearest integer value.

        Only works on floating point Series.

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
        │ 0.0 │
        │ 1.0 │
        │ 1.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.floor())

    def ceil(self) -> Expr:
        """
        Rounds up to the nearest integer value.

        Only works on floating point Series.

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
        │ 1.0 │
        │ 1.0 │
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
        │ 0.5 │
        │ 1.0 │
        │ 1.2 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.round(decimals))

    def dot(self, other: Expr | str) -> Expr:
        """
        Compute the dot/inner product between two Expressions.

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
        │ 1   ┆ 2   │
        └─────┴─────┘

        """
        return wrap_expr(self._pyexpr.mode())

    def cast(self, dtype: PolarsDataType | type[Any], strict: bool = True) -> Expr:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["4", "5", "6"],
        ...     }
        ... )
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
        │ 2.0 ┆ 5   │
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
        │ 2     │
        │ 3     │
        │ 4     │
        │ 98    │
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
        │ 2     │
        │ 3     │
        │ 4     │
        │ 98    │
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
        │ one   ┆ [1, 2, 98] │
        └───────┴────────────┘

        """
        return wrap_expr(self._pyexpr.sort_with(reverse, nulls_last))

    def top_k(self, k: int = 5, reverse: bool = False) -> Expr:
        r"""
        Return the `k` largest elements.

        If 'reverse=True` the smallest elements will be given.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.
        reverse
            Return the smallest elements.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("value").top_k().alias("top_k"),
        ...         pl.col("value").top_k(reverse=True).alias("bottom_k"),
        ...     ]
        ... )
        shape: (5, 2)
        ┌───────┬──────────┐
        │ top_k ┆ bottom_k │
        │ ---   ┆ ---      │
        │ i64   ┆ i64      │
        ╞═══════╪══════════╡
        │ 99    ┆ 1        │
        │ 98    ┆ 2        │
        │ 4     ┆ 3        │
        │ 3     ┆ 4        │
        │ 2     ┆ 98       │
        └───────┴──────────┘

        """
        return wrap_expr(self._pyexpr.top_k(k, reverse))

    @deprecate_nonkeyword_arguments()
    def arg_sort(self, reverse: bool = False, nulls_last: bool = False) -> Expr:
        """
        Get the index values that would sort this column.

        Parameters
        ----------
        reverse
            Sort in reverse (descending) order.
        nulls_last
            Place null values last instead of first.

        Returns
        -------
        Expr
            Series of dtype UInt32.

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
        │ 0   │
        │ 2   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arg_sort(reverse, nulls_last))

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

    def search_sorted(
        self, element: Expr | int | float | pli.Series, side: SearchSortedSide = "any"
    ) -> Expr:
        """
        Find indices where elements should be inserted to maintain order.

        .. math:: a[i-1] < v <= a[i]

        Parameters
        ----------
        element
            Expression or scalar value.
        side : {'any', 'left', 'right'}
            If 'any', the index of the first suitable location found is given.
            If 'left', the index of the leftmost suitable location found is given.
            If 'right', return the rightmost suitable location found is given.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "values": [1, 2, 3, 5],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("values").search_sorted(0).alias("zero"),
        ...         pl.col("values").search_sorted(3).alias("three"),
        ...         pl.col("values").search_sorted(6).alias("six"),
        ...     ]
        ... )
        shape: (1, 3)
        ┌──────┬───────┬─────┐
        │ zero ┆ three ┆ six │
        │ ---  ┆ ---   ┆ --- │
        │ u32  ┆ u32   ┆ u32 │
        ╞══════╪═══════╪═════╡
        │ 0    ┆ 2     ┆ 4   │
        └──────┴───────┴─────┘

        """
        element = expr_to_lit_or_expr(element, str_to_lit=False)
        return wrap_expr(self._pyexpr.search_sorted(element._pyexpr, side))

    def sort_by(
        self,
        by: Expr | str | list[Expr | str],
        *,
        reverse: bool | list[bool] = False,
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
        │ one   │
        │ two   │
        │ two   │
        │ one   │
        │ two   │
        └───────┘

        """
        if not isinstance(by, list):
            by = [by]
        if not isinstance(reverse, list):
            reverse = [reverse]
        by = selection_to_pyexpr_list(by)

        return wrap_expr(self._pyexpr.sort_by(by, reverse))

    def take(
        self, indices: int | list[int] | Expr | pli.Series | np.ndarray[Any, Any]
    ) -> Expr:
        """
        Take values by index.

        Parameters
        ----------
        indices
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
        │ two   ┆ 99    │
        └───────┴───────┘

        """
        if isinstance(indices, list) or (
            _check_for_numpy(indices) and isinstance(indices, np.ndarray)
        ):
            indices = cast("np.ndarray[Any, Any]", indices)
            indices_lit = pli.lit(pli.Series("", indices, dtype=UInt32))
        else:
            indices_lit = pli.expr_to_lit_or_expr(indices, str_to_lit=False)
        return pli.wrap_expr(self._pyexpr.take(indices_lit._pyexpr))

    def shift(self, periods: int = 1) -> Expr:
        """
        Shift the values by a given period.

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
        │ 1    │
        │ 2    │
        │ 3    │
        └──────┘

        """
        return wrap_expr(self._pyexpr.shift(periods))

    def shift_and_fill(
        self,
        periods: int,
        fill_value: int | float | bool | str | Expr | list[Any],
    ) -> Expr:
        """
        Shift the values by a given period and fill the resulting null values.

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
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        """
        fill_value = expr_to_lit_or_expr(fill_value, str_to_lit=True)
        return wrap_expr(self._pyexpr.shift_and_fill(periods, fill_value._pyexpr))

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Expr:
        """
        Fill null values using the specified value or strategy.

        To interpolate over null values see interpolate.

        Parameters
        ----------
        value
            Value used to fill null values.
        strategy : {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}
            Strategy used to fill null values.
        limit
            Number of consecutive null values to fill when using the 'forward' or
            'backward' strategy.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [4, None, 6],
        ...     }
        ... )
        >>> df.fill_null(strategy="zero")
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 0   │
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
        │ 2   ┆ 99  │
        │ 99  ┆ 6   │
        └─────┴─────┘
        >>> df.fill_null(strategy="forward")
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 4   │
        │ 2   ┆ 6   │
        └─────┴─────┘

        """
        if value is not None and strategy is not None:
            raise ValueError("cannot specify both 'value' and 'strategy'.")
        elif value is None and strategy is None:
            raise ValueError("must specify either a fill 'value' or 'strategy'")
        elif strategy not in ("forward", "backward") and limit is not None:
            raise ValueError(
                "can only specify 'limit' when strategy is set to"
                " 'backward' or 'forward'"
            )

        if value is not None:
            value = expr_to_lit_or_expr(value, str_to_lit=True)
            return wrap_expr(self._pyexpr.fill_null(value._pyexpr))
        else:
            return wrap_expr(self._pyexpr.fill_null_with_strategy(strategy, limit))

    def fill_nan(self, fill_value: int | float | Expr | None) -> Expr:
        """
        Fill floating point NaN value with a fill value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, None, float("nan")],
        ...         "b": [4.0, float("nan"), 6],
        ...     }
        ... )
        >>> df.fill_nan("zero")
        shape: (3, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ str  ┆ str  │
        ╞══════╪══════╡
        │ 1.0  ┆ 4.0  │
        │ null ┆ zero │
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [4, None, 6],
        ...     }
        ... )
        >>> df.select(pl.all().forward_fill())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 4   │
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [4, None, 6],
        ...     }
        ... )
        >>> df.select(pl.all().backward_fill())
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 1    ┆ 4   │
        │ 2    ┆ 6   │
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
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """  # noqa: W505
        return wrap_expr(self._pyexpr.reverse())

    def std(self, ddof: int = 1) -> Expr:
        """
        Get standard deviation.

        Parameters
        ----------
        ddof
            Degrees of freedom.

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
        return wrap_expr(self._pyexpr.std(ddof))

    def var(self, ddof: int = 1) -> Expr:
        """
        Get variance.

        Parameters
        ----------
        ddof
            Degrees of freedom.

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
        return wrap_expr(self._pyexpr.var(ddof))

    def max(self) -> Expr:
        """
        Get maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, float("nan"), 1]})
        >>> df.select(pl.col("a").max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.max())

    def min(self) -> Expr:
        """
        Get minimum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, float("nan"), 1]})
        >>> df.select(pl.col("a").min())
        shape: (1, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ -1.0 │
        └──────┘

        """
        return wrap_expr(self._pyexpr.min())

    def nan_max(self) -> Expr:
        """
        Get maximum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0, float("nan")]})
        >>> df.select(pl.col("a").nan_max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ NaN │
        └─────┘

        """
        return wrap_expr(self._pyexpr.nan_max())

    def nan_min(self) -> Expr:
        """
        Get minimum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0, float("nan")]})
        >>> df.select(pl.col("a").nan_min())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ NaN │
        └─────┘

        """
        return wrap_expr(self._pyexpr.nan_min())

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
        Compute the product of an expression.

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

    def n_unique(self) -> Expr:
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

    def null_count(self) -> Expr:
        """
        Count null values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 1, None],
        ...         "b": [1, 2, 3],
        ...     }
        ... )
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10],
        ...         "b": [None, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_unique())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        │ 1   │
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

    def over(self, expr: str | Expr | list[Expr | str]) -> Expr:
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
        ...         "groups": ["g1", "g1", "g2"],
        ...         "values": [1, 2, 3],
        ...     }
        ... )
        >>> df.with_columns(pl.col("values").max().over("groups").alias("max_by_group"))
        shape: (3, 3)
        ┌────────┬────────┬──────────────┐
        │ groups ┆ values ┆ max_by_group │
        │ ---    ┆ ---    ┆ ---          │
        │ str    ┆ i64    ┆ i64          │
        ╞════════╪════════╪══════════════╡
        │ g1     ┆ 1      ┆ 2            │
        │ g1     ┆ 2      ┆ 2            │
        │ g2     ┆ 3      ┆ 3            │
        └────────┴────────┴──────────────┘
        >>> df = pl.DataFrame(
        ...     {
        ...         "groups": [1, 1, 2, 2, 1, 2, 3, 3, 1],
        ...         "values": [1, 2, 3, 4, 5, 6, 7, 8, 8],
        ...     }
        ... )
        >>> df.lazy().select(
        ...     pl.col("groups").sum().over("groups"),
        ... ).collect()
        shape: (9, 1)
        ┌────────┐
        │ groups │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 4      │
        │ 4      │
        │ 6      │
        │ 6      │
        │ ...    │
        │ 6      │
        │ 6      │
        │ 6      │
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
        │ false │
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
        >>> df.with_columns(pl.col("num").is_first().alias("is_first"))
        shape: (5, 2)
        ┌─────┬──────────┐
        │ num ┆ is_first │
        │ --- ┆ ---      │
        │ i64 ┆ bool     │
        ╞═════╪══════════╡
        │ 1   ┆ true     │
        │ 2   ┆ true     │
        │ 3   ┆ true     │
        │ 1   ┆ false    │
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
        │ true  │
        │ false │
        └───────┘

        """
        return wrap_expr(self._pyexpr.is_duplicated())

    def quantile(
        self,
        quantile: float | Expr,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> Expr:
        """
        Get quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

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
        quantile = expr_to_lit_or_expr(quantile, str_to_lit=False)
        return wrap_expr(self._pyexpr.quantile(quantile._pyexpr, interpolation))

    def filter(self, predicate: Expr) -> Expr:
        """
        Filter a single column.

        Mostly useful in an aggregation context. If you want to filter on a DataFrame
        level, use `LazyFrame.filter`.

        Parameters
        ----------
        predicate
            Boolean expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group_col": ["g1", "g1", "g2"],
        ...         "b": [1, 2, 3],
        ...     }
        ... )
        >>> df.groupby("group_col").agg(
        ...     [
        ...         pl.col("b").filter(pl.col("b") < 2).sum().alias("lt"),
        ...         pl.col("b").filter(pl.col("b") >= 2).sum().alias("gte"),
        ...     ]
        ... ).sort("group_col")
        shape: (2, 3)
        ┌───────────┬──────┬─────┐
        │ group_col ┆ lt   ┆ gte │
        │ ---       ┆ ---  ┆ --- │
        │ str       ┆ i64  ┆ i64 │
        ╞═══════════╪══════╪═════╡
        │ g1        ┆ 1    ┆ 2   │
        │ g2        ┆ null ┆ 3   │
        └───────────┴──────┴─────┘

        """
        return wrap_expr(self._pyexpr.filter(predicate._pyexpr))

    def where(self, predicate: Expr) -> Expr:
        """
        Filter a single column.

        Alias for :func:`filter`.

        Parameters
        ----------
        predicate
            Boolean expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group_col": ["g1", "g1", "g2"],
        ...         "b": [1, 2, 3],
        ...     }
        ... )
        >>> df.groupby("group_col").agg(
        ...     [
        ...         pl.col("b").where(pl.col("b") < 2).sum().alias("lt"),
        ...         pl.col("b").where(pl.col("b") >= 2).sum().alias("gte"),
        ...     ]
        ... ).sort("group_col")
        shape: (2, 3)
        ┌───────────┬──────┬─────┐
        │ group_col ┆ lt   ┆ gte │
        │ ---       ┆ ---  ┆ --- │
        │ str       ┆ i64  ┆ i64 │
        ╞═══════════╪══════╪═════╡
        │ g1        ┆ 1    ┆ 2   │
        │ g2        ┆ null ┆ 3   │
        └───────────┴──────┴─────┘

        """
        return self.filter(predicate)

    @deprecate_nonkeyword_arguments(allowed_args=["self", "f", "return_dtype"])
    def map(
        self,
        f: Callable[[pli.Series], pli.Series | Any],
        return_dtype: PolarsDataType | None = None,
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
        >>> df.select(pl.all().map(lambda x: x.to_numpy().argmax()))
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

    @deprecate_nonkeyword_arguments(allowed_args=["self", "f", "return_dtype"])
    def apply(
        self,
        f: Callable[[pli.Series], pli.Series] | Callable[[Any], Any],
        return_dtype: PolarsDataType | None = None,
        skip_nulls: bool = True,
        pass_name: bool = False,
    ) -> Expr:
        """
        Apply a custom/user-defined function (UDF) in a GroupBy or Projection context.

        Depending on the context it has the following behavior:

        * Selection
            Expects `f` to be of type Callable[[Any], Any].
            Applies a python function over each individual value in the column.
        * GroupBy
            Expects `f` to be of type Callable[[Series], Series].
            Applies a python function over each group.

        Implementing logic using a Python function is almost always _significantly_
        slower and more memory intensive than implementing the same logic using
        the native expression API because:

        - The native expression engine runs in Rust; UDFs run in Python.
        - Use of Python UDFs forces the DataFrame to be materialized in memory.
        - Polars-native expressions can be parallelised (UDFs cannot).
        - Polars-native expressions can be logically optimised (UDFs cannot).

        Wherever possible you should strongly prefer the native expression API
        to achieve the best performance.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
            If not set, polars will assume that
            the dtype remains unchanged.
        skip_nulls
            Don't apply the function over values
            that contain nulls. This is faster.
        pass_name
            Pass the Series name to the custom function
            This is more expensive.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["a", "b", "c", "c"],
        ...     }
        ... )

        In a selection context, the function is applied by row.

        >>> df.with_columns(
        ...     pl.col("a").apply(lambda x: x * 2).alias("a_times_2"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────────┐
        │ a   ┆ b   ┆ a_times_2 │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ str ┆ i64       │
        ╞═════╪═════╪═══════════╡
        │ 1   ┆ a   ┆ 2         │
        │ 2   ┆ b   ┆ 4         │
        │ 3   ┆ c   ┆ 6         │
        │ 1   ┆ c   ┆ 2         │
        └─────┴─────┴───────────┘

        It is better to implement this with an expression:

        >>> df.with_columns(
        ...     (pl.col("a") * 2).alias("a_times_2"),
        ... )  # doctest: +IGNORE_RESULT

        In a GroupBy context the function is applied by group:

        >>> df.lazy().groupby("b", maintain_order=True).agg(
        ...     pl.col("a").apply(lambda x: x.sum())
        ... ).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        │ b   ┆ 2   │
        │ c   ┆ 4   │
        └─────┴─────┘

        It is better to implement this with an expression:

        >>> df.groupby("b", maintain_order=True).agg(
        ...     pl.col("a").sum(),
        ... )  # doctest: +IGNORE_RESULT

        """
        # input x: Series of type list containing the group values
        if pass_name:

            def wrap_f(x: pli.Series) -> pli.Series:  # pragma: no cover
                def inner(s: pli.Series) -> pli.Series:  # pragma: no cover
                    s.rename(x.name, in_place=True)
                    return f(s)

                return x.apply(inner, return_dtype=return_dtype, skip_nulls=skip_nulls)

            return self.map(wrap_f, agg_list=True, return_dtype=return_dtype)

        else:

            def wrap_f(x: pli.Series) -> pli.Series:  # pragma: no cover
                return x.apply(f, return_dtype=return_dtype, skip_nulls=skip_nulls)

            return self.map(wrap_f, agg_list=True, return_dtype=return_dtype)

    def flatten(self) -> Expr:
        """
        Flatten a list or string column.

        Alias for :func:`polars.internals.expr.list.ExprListNameSpace.explode`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["a", "b", "b"],
        ...         "values": [[1, 2], [2, 3], [4]],
        ...     }
        ... )
        >>> df.groupby("group").agg(pl.col("values").flatten())  # doctest: +SKIP
        shape: (2, 2)
        ┌───────┬───────────┐
        │ group ┆ values    │
        │ ---   ┆ ---       │
        │ str   ┆ list[i64] │
        ╞═══════╪═══════════╡
        │ a     ┆ [1, 2]    │
        │ b     ┆ [2, 3, 4] │
        └───────┴───────────┘

        """
        return wrap_expr(self._pyexpr.explode())

    def explode(self) -> Expr:
        """
        Explode a list or utf8 Series.

        This means that every item is expanded to a new row.

        .. deprecated:: 0.15.16
            `Expr.explode` will be removed in favour of `Expr.arr.explode` and
            `Expr.str.explode`.

        Returns
        -------
        Exploded Series of same dtype

        See Also
        --------
        ExprListNameSpace.explode : Explode a list column
        ExprStringNameSpace.explode : Explode a string column

        """
        warnings.warn(
            "`Series/Expr.explode()` is deprecated in favor of the identical method"
            " under the list and string namespaces. Use `.arr.explode()` or"
            " `.str.explode()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        │ 4   │
        │ 7   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.take_every(n))

    def head(self, n: int = 10) -> Expr:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7]})
        >>> df.head(3)
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.head(n))

    def tail(self, n: int = 10) -> Expr:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7]})
        >>> df.tail(3)
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 5   │
        │ 6   │
        │ 7   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.tail(n))

    def limit(self, n: int = 10) -> Expr:
        """
        Get the first `n` rows.

        Alias for :func:`Expr.head`.

        Parameters
        ----------
        n
            Number of rows to return.

        """
        return self.head(n)

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
        │ 8.0  │
        │ 27.0 │
        │ 64.0 │
        └──────┘

        """
        exponent = expr_to_lit_or_expr(exponent)
        return wrap_expr(self._pyexpr.pow(exponent._pyexpr))

    def is_in(self, other: Expr | Sequence[Any] | str | pli.Series) -> Expr:
        """
        Check if elements of this expression are present in the other Series.

        Parameters
        ----------
        other
            Series or sequence of primitive type.

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
        │ true     │
        │ false    │
        └──────────┘

        """
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = pli.lit(None) if len(other) == 0 else pli.lit(pli.Series(other))
        else:
            other = expr_to_lit_or_expr(other, str_to_lit=False)
        return wrap_expr(self._pyexpr.is_in(other._pyexpr))

    def repeat_by(self, by: Expr | str) -> Expr:
        """
        Repeat the elements in this Series as specified in the given expression.

        The repeated elements are expanded into a `List`.

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
        │ ["y", "y"]      │
        │ ["z", "z", "z"] │
        └─────────────────┘

        """
        by = expr_to_lit_or_expr(by, False)
        return wrap_expr(self._pyexpr.repeat_by(by._pyexpr))

    def is_between(
        self,
        start: Expr | datetime | date | time | int | float | str,
        end: Expr | datetime | date | time | int | float | str,
        closed: ClosedInterval = "both",
    ) -> Expr:
        """
        Check if this expression is between the given start and end values.

        Parameters
        ----------
        start
            Lower bound value (can be an expression or literal).
        end
            Upper bound value (can be an expression or literal).
        closed : {'both', 'left', 'right', 'none'}
            Define which sides of the interval are closed (inclusive).

        Returns
        -------
        Expr that evaluates to a Boolean Series.

        Examples
        --------
        >>> df = pl.DataFrame({"num": [1, 2, 3, 4, 5]})
        >>> df.with_columns(pl.col("num").is_between(2, 4).alias("is_between"))
        shape: (5, 2)
        ┌─────┬────────────┐
        │ num ┆ is_between │
        │ --- ┆ ---        │
        │ i64 ┆ bool       │
        ╞═════╪════════════╡
        │ 1   ┆ false      │
        │ 2   ┆ true       │
        │ 3   ┆ true       │
        │ 4   ┆ true       │
        │ 5   ┆ false      │
        └─────┴────────────┘

        Use the ``closed`` argument to include or exclude the values at the bounds:

        >>> df.with_columns(
        ...     pl.col("num").is_between(2, 4, closed="left").alias("is_between")
        ... )
        shape: (5, 2)
        ┌─────┬────────────┐
        │ num ┆ is_between │
        │ --- ┆ ---        │
        │ i64 ┆ bool       │
        ╞═════╪════════════╡
        │ 1   ┆ false      │
        │ 2   ┆ true       │
        │ 3   ┆ true       │
        │ 4   ┆ false      │
        │ 5   ┆ false      │
        └─────┴────────────┘

        You can also use strings as well as numeric/temporal values (note: ensure that
        string literals are wrapped with ``lit`` so as not to conflate them with
        column names):

        >>> df = pl.DataFrame({"a": ["a", "b", "c", "d", "e"]})
        >>> df.with_columns(
        ...     pl.col("a")
        ...     .is_between(pl.lit("a"), pl.lit("c"), closed="both")
        ...     .alias("is_between")
        ... )
        shape: (5, 2)
        ┌─────┬────────────┐
        │ a   ┆ is_between │
        │ --- ┆ ---        │
        │ str ┆ bool       │
        ╞═════╪════════════╡
        │ a   ┆ true       │
        │ b   ┆ true       │
        │ c   ┆ true       │
        │ d   ┆ false      │
        │ e   ┆ false      │
        └─────┴────────────┘
        """
        start = expr_to_lit_or_expr(start, str_to_lit=False)
        end = expr_to_lit_or_expr(end, str_to_lit=False)

        if closed == "none":
            return (self > start) & (self < end)
        elif closed == "both":
            return (self >= start) & (self <= end)
        elif closed == "right":
            return (self > start) & (self <= end)
        elif closed == "left":
            return (self >= start) & (self < end)
        else:
            raise ValueError(
                "closed must be one of {'left', 'right', 'both', 'none'},"
                f" got {closed!r}"
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
        >>> df.with_columns(pl.all().hash(10, 20, 30, 40))  # doctest: +IGNORE_RESULT
        shape: (3, 2)
        ┌──────────────────────┬──────────────────────┐
        │ a                    ┆ b                    │
        │ ---                  ┆ ---                  │
        │ u64                  ┆ u64                  │
        ╞══════════════════════╪══════════════════════╡
        │ 9774092659964970114  ┆ 13614470193936745724 │
        │ 1101441246220388612  ┆ 11638928888656214026 │
        │ 11638928888656214026 ┆ 13382926553367784577 │
        └──────────────────────┴──────────────────────┘

        """
        k0 = seed
        k1 = seed_1 if seed_1 is not None else seed
        k2 = seed_2 if seed_2 is not None else seed
        k3 = seed_3 if seed_3 is not None else seed
        return wrap_expr(self._pyexpr.hash(k0, k1, k2, k3))

    def reinterpret(self, signed: bool = True) -> Expr:
        """
        Reinterpret the underlying bits as a signed/unsigned integer.

        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            If True, reinterpret as `pl.Int64`. Otherwise, reinterpret as `pl.UInt64`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 1, 2], dtype=pl.UInt64)
        >>> df = pl.DataFrame([s])
        >>> df.select(
        ...     [
        ...         pl.col("a").reinterpret(signed=True).alias("reinterpreted"),
        ...         pl.col("a").alias("original"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌───────────────┬──────────┐
        │ reinterpreted ┆ original │
        │ ---           ┆ ---      │
        │ i64           ┆ u64      │
        ╞═══════════════╪══════════╡
        │ 1             ┆ 1        │
        │ 1             ┆ 1        │
        │ 2             ┆ 2        │
        └───────────────┴──────────┘

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
        │ 2   │
        │ 4   │
        └─────┘

        """

        def inspect(s: pli.Series) -> pli.Series:  # pragma: no cover
            print(fmt.format(s))
            return s

        return self.map(inspect, return_dtype=None, agg_list=True)

    def interpolate(self, method: InterpolationMethod = "linear") -> Expr:
        """
        Fill nulls with linear interpolation over missing values.

        Can also be used to regrid data to a new grid - see examples below.

        Parameters
        ----------
        method : {'linear', 'linear'}
            Interpolation method

        Examples
        --------
        >>> # Fill nulls with linear interpolation
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, None, 3],
        ...         "b": [1.0, float("nan"), 3.0],
        ...     }
        ... )
        >>> df.select(pl.all().interpolate())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 1.0 │
        │ 2   ┆ NaN │
        │ 3   ┆ 3.0 │
        └─────┴─────┘
        >>> df_original_grid = pl.DataFrame(
        ...     {
        ...         "grid_points": [1, 3, 10],
        ...         "values": [2.0, 6.0, 20.0],
        ...     }
        ... )  # Interpolate from this to the new grid
        >>> df_new_grid = pl.DataFrame({"grid_points": range(1, 11)})
        >>> df_new_grid.join(
        ...     df_original_grid, on="grid_points", how="left"
        ... ).with_columns(pl.col("values").interpolate())
        shape: (10, 2)
        ┌─────────────┬────────┐
        │ grid_points ┆ values │
        │ ---         ┆ ---    │
        │ i64         ┆ f64    │
        ╞═════════════╪════════╡
        │ 1           ┆ 2.0    │
        │ 2           ┆ 4.0    │
        │ 3           ┆ 6.0    │
        │ 4           ┆ 8.0    │
        │ ...         ┆ ...    │
        │ 7           ┆ 14.0   │
        │ 8           ┆ 16.0   │
        │ 9           ┆ 18.0   │
        │ 10          ┆ 20.0   │
        └─────────────┴────────┘

        """
        return wrap_expr(self._pyexpr.interpolate(method))

    def rolling_min(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
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
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_min(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 1.0  │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
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
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
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
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal, for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_max(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        │ 5.0  │
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
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
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
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

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
        │ 4.5  │
        │ 7.0  │
        │ 4.0  │
        │ 9.0  │
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
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
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
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_sum(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 3.0  │
        │ 5.0  │
        │ 7.0  │
        │ 9.0  │
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
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
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
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_std(window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────────┐
        │ A        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        │ null     │
        │ 1.0      │
        │ 1.0      │
        │ 1.527525 │
        │ 2.0      │
        └──────────┘

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
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
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
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_var(window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────────┐
        │ A        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        │ null     │
        │ 1.0      │
        │ 1.0      │
        │ 2.333333 │
        │ 4.0      │
        └──────────┘

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
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Expr:
        """
        Compute a rolling median.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_median(window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ null │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        │ 6.0  │
        └──────┘

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
        interpolation: RollingInterpolationMethod = "nearest",
        window_size: int | timedelta | str = 2,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Expr:
        """
        Compute a rolling quantile.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

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

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_quantile(quantile=0.33, window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ null │
        │ 1.0  │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        └──────┘

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
        weights: list[float] | None = None,
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
        │ null     │
        │ 4.358899 │
        │ 4.041452 │
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
            Integer size of the rolling window.
        bias
            If False, the calculations are corrected for statistical bias.

        """
        return wrap_expr(self._pyexpr.rolling_skew(window_size, bias))

    def abs(self) -> Expr:
        """
        Compute absolute values.

        Same as `abs(expr)`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [-1.0, 0.0, 1.0, 2.0],
        ...     }
        ... )
        >>> df.select(pl.col("A").abs())
        shape: (4, 1)
        ┌─────┐
        │ A   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        │ 0.0 │
        │ 1.0 │
        │ 2.0 │
        └─────┘

        """
        return wrap_expr(self._pyexpr.abs())

    def argsort(self, reverse: bool = False, nulls_last: bool = False) -> Expr:
        """
        Get the index values that would sort this column.

        Alias for :func:`Expr.arg_sort`.

        .. deprecated:: 0.16.5
            `Expr.argsort` will be removed in favour of `Expr.arg_sort`.

        Parameters
        ----------
        reverse
            Sort in reverse (descending) order.
        nulls_last
            Place null values last instead of first.

        Returns
        -------
        Expr
            Series of dtype UInt32.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").argsort())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        │ 0   │
        │ 2   │
        └─────┘

        """
        warnings.warn(
            "`Expr.argsort()` is deprecated in favor of `Expr.arg_sort()`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.arg_sort(reverse, nulls_last)

    def rank(self, method: RankMethod = "average", reverse: bool = False) -> Expr:
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'dense', 'ordinal', 'random'}
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
        │ 4.5 │
        │ 1.5 │
        │ 1.5 │
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
        │ 4   │
        │ 1   │
        │ 2   │
        │ 5   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.rank(method, reverse))

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Expr:
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            Number of slots to shift.
        null_behavior : {'ignore', 'drop'}
            How to handle null values.

        Examples
        --------
        >>> df = pl.DataFrame({"int": [20, 10, 30, 25, 35]})
        >>> df.with_columns(change=pl.col("int").diff())
        shape: (5, 2)
        ┌─────┬────────┐
        │ int ┆ change │
        │ --- ┆ ---    │
        │ i64 ┆ i64    │
        ╞═════╪════════╡
        │ 20  ┆ null   │
        │ 10  ┆ -10    │
        │ 30  ┆ 20     │
        │ 25  ┆ -5     │
        │ 35  ┆ 10     │
        └─────┴────────┘

        >>> df.with_columns(change=pl.col("int").diff(n=2))
        shape: (5, 2)
        ┌─────┬────────┐
        │ int ┆ change │
        │ --- ┆ ---    │
        │ i64 ┆ i64    │
        ╞═════╪════════╡
        │ 20  ┆ null   │
        │ 10  ┆ null   │
        │ 30  ┆ 10     │
        │ 25  ┆ 15     │
        │ 35  ┆ 5      │
        └─────┴────────┘

        >>> df.select(pl.col("int").diff(n=2, null_behavior="drop").alias("diff"))
        shape: (3, 1)
        ┌──────┐
        │ diff │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 10   │
        │ 15   │
        │ 5    │
        └──────┘

        """
        return wrap_expr(self._pyexpr.diff(n, null_behavior))

    def pct_change(self, n: int = 1) -> Expr:
        """
        Computes percentage change between values.

        Percentage change (as fraction) between current element and most-recent
        non-null element at least ``n`` period(s) before the current element.

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
        >>> df.with_columns(pl.col("a").pct_change().alias("pct_change"))
        shape: (5, 2)
        ┌──────┬────────────┐
        │ a    ┆ pct_change │
        │ ---  ┆ ---        │
        │ i64  ┆ f64        │
        ╞══════╪════════════╡
        │ 10   ┆ null       │
        │ 11   ┆ 0.1        │
        │ 12   ┆ 0.090909   │
        │ null ┆ 0.0        │
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
            If False, the calculations are corrected for statistical bias.

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

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").skew())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.343622 │
        └──────────┘

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
            If False, the calculations are corrected for statistical bias.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").kurtosis())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ f64       │
        ╞═══════════╡
        │ -1.153061 │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.kurtosis(fisher, bias))

    def clip(self, min_val: int | float, max_val: int | float) -> Expr:
        """
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

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
        >>> df.with_columns(pl.col("foo").clip(1, 10).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ 1           │
        │ 5    ┆ 5           │
        │ null ┆ null        │
        │ 50   ┆ 10          │
        └──────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.clip(min_val, max_val))

    def clip_min(self, min_val: int | float) -> Expr:
        """
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        min_val
            Minimum value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [-50, 5, None, 50]})
        >>> df.with_columns(pl.col("foo").clip_min(0).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ 0           │
        │ 5    ┆ 5           │
        │ null ┆ null        │
        │ 50   ┆ 50          │
        └──────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.clip_min(min_val))

    def clip_max(self, max_val: int | float) -> Expr:
        """
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        max_val
            Maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [-50, 5, None, 50]})
        >>> df.with_columns(pl.col("foo").clip_max(0).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ -50         │
        │ 5    ┆ 0           │
        │ null ┆ null        │
        │ 50   ┆ 0           │
        └──────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.clip_max(max_val))

    def lower_bound(self) -> Expr:
        """
        Calculate the lower bound.

        Returns a unit Series with the lowest value possible for the dtype of this
        expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").lower_bound())
        shape: (1, 1)
        ┌──────────────────────┐
        │ a                    │
        │ ---                  │
        │ i64                  │
        ╞══════════════════════╡
        │ -9223372036854775808 │
        └──────────────────────┘

        """
        return wrap_expr(self._pyexpr.lower_bound())

    def upper_bound(self) -> Expr:
        """
        Calculate the upper bound.

        Returns a unit Series with the highest value possible for the dtype of this
        expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").upper_bound())
        shape: (1, 1)
        ┌─────────────────────┐
        │ a                   │
        │ ---                 │
        │ i64                 │
        ╞═════════════════════╡
        │ 9223372036854775807 │
        └─────────────────────┘

        """
        return wrap_expr(self._pyexpr.upper_bound())

    def sign(self) -> Expr:
        """
        Compute the element-wise indication of the sign.

        The returned values can be -1, 0, or 1:

        * -1 if x  < 0.
        *  0 if x == 0.
        *  1 if x  > 0.

        (null values are preserved as-is).

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
        │ 0    │
        │ 0    │
        │ 1    │
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
        Reshape this Expr to a flat Series or a Series of Lists.

        Parameters
        ----------
        dims
            Tuple of the dimension sizes. If a -1 is used in any of the dimensions, that
            dimension is inferred.

        Returns
        -------
        Expr
            If a single dimension is given, results in a flat Series of shape (len,).
            If a multiple dimensions are given, results in a Series of Lists with shape
            (rows, cols).

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
        │ [4, 5, 6] │
        │ [7, 8, 9] │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.reshape(dims))

    def shuffle(self, seed: int | None = None) -> Expr:
        """
        Shuffle the contents of this expression.

        Parameters
        ----------
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").shuffle(seed=1))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 1   │
        │ 3   │
        └─────┘

        """
        if seed is None:
            seed = random.randint(0, 10000)
        return wrap_expr(self._pyexpr.shuffle(seed))

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Expr:
        """
        Sample from this expression.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `frac`. Defaults to 1 if
            `frac` is None.
        frac
            Fraction of items to return. Cannot be used with `n`.
        with_replacement
            Allow values to be sampled more than once.
        shuffle
            Shuffle the order of sampled data points.
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").sample(frac=1.0, with_replacement=True, seed=1))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 1   │
        │ 1   │
        └─────┘

        """
        if n is not None and frac is not None:
            raise ValueError("cannot specify both `n` and `frac`")

        if seed is None:
            seed = random.randint(0, 10000)

        if frac is not None:
            return wrap_expr(
                self._pyexpr.sample_frac(frac, with_replacement, shuffle, seed)
            )

        if n is None:
            n = 1
        return wrap_expr(self._pyexpr.sample_n(n, with_replacement, shuffle, seed))

    def ewm_mean(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Expr:
        r"""
        Exponentially-weighted moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.


        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").ewm_mean(com=1))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.0      │
        │ 1.666667 │
        │ 2.428571 │
        └──────────┘

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(
            self._pyexpr.ewm_mean(alpha, adjust, min_periods, ignore_nulls)
        )

    def ewm_std(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Expr:
        r"""
        Exponentially-weighted moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").ewm_std(com=1))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.0      │
        │ 0.707107 │
        │ 0.963624 │
        └──────────┘

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(
            self._pyexpr.ewm_std(alpha, adjust, bias, min_periods, ignore_nulls)
        )

    def ewm_var(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Expr:
        r"""
        Exponentially-weighted moving variance.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").ewm_var(com=1))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.0      │
        │ 0.5      │
        │ 0.928571 │
        └──────────┘

        """
        alpha = _prepare_alpha(com, span, half_life, alpha)
        return wrap_expr(
            self._pyexpr.ewm_var(alpha, adjust, bias, min_periods, ignore_nulls)
        )

    def extend_constant(self, value: PythonLiteral | None, n: int) -> Expr:
        """
        Extremely fast method for extending the Series with 'n' copies of a value.

        Parameters
        ----------
        value
            A constant literal value (not an expression) with which to extend the
            expression result Series; can pass None to extend with nulls.
        n
            The number of additional values that will be added.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1, 2, 3]})
        >>> df.select((pl.col("values") - 1).extend_constant(99, n=2))
        shape: (5, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 0      │
        │ 1      │
        │ 2      │
        │ 99     │
        │ 99     │
        └────────┘

        """
        if isinstance(value, Expr):
            raise TypeError(f"'value' must be a supported literal; found {value!r}")

        return wrap_expr(self._pyexpr.extend_constant(value, n))

    def value_counts(self, multithreaded: bool = False, sort: bool = False) -> Expr:
        """
        Count all unique values and create a struct mapping value to count.

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
        │ {"b",2}   │
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
        │ 2   │
        │ 3   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.unique_counts())

    def log(self, base: float = math.e) -> Expr:
        """
        Compute the logarithm to a given base.

        Parameters
        ----------
        base
            Given base, defaults to `e`

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").log(base=2))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.0      │
        │ 1.0      │
        │ 1.584963 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.log(base))

    def entropy(self, base: float = math.e, normalize: bool = True) -> Expr:
        """
        Computes the entropy.

        Uses the formula ``-sum(pk * log(pk)`` where ``pk`` are discrete probabilities.

        Parameters
        ----------
        base
            Given base, defaults to `e`
        normalize
            Normalize pk if it doesn't sum to 1.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").entropy(base=2))
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.459148 │
        └──────────┘
        >>> df.select(pl.col("a").entropy(base=2, normalize=False))
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ f64       │
        ╞═══════════╡
        │ -6.754888 │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.entropy(base, normalize))

    def cumulative_eval(
        self, expr: Expr, min_periods: int = 1, parallel: bool = False
    ) -> Expr:
        """
        Run an expression over a sliding window that increases `1` slot every iteration.

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

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        This can be really slow as it can have `O(n^2)` complexity. Don't use this
        for operations that visit all elements.

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
        │ -3.0   │
        │ -8.0   │
        │ -15.0  │
        │ -24.0  │
        └────────┘

        """
        return wrap_expr(
            self._pyexpr.cumulative_eval(expr._pyexpr, min_periods, parallel)
        )

    def set_sorted(self, reverse: bool = False) -> Expr:
        """
        Flags the expression as 'sorted'.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        reverse
            If the `Series` order is reversed, e.g. descending.

        Warnings
        --------
        This can lead to incorrect results if this `Series` is not sorted!!
        Use with care!

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1, 2, 3]})
        >>> df.select(pl.col("values").set_sorted().max())
        shape: (1, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 3      │
        └────────┘

        """
        return wrap_expr(self._pyexpr.set_sorted_flag(reverse))

    # Keep the `list` and `str` methods below at the end of the definition of Expr,
    # as to not confuse mypy with the type annotation `str` and `list`

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

    def shrink_dtype(self) -> Expr:
        """
        Shrink numeric columns to the minimal required datatype.

        Shrink to the dtype needed to fit the extrema of this [`Series`].
        This can be used to reduce memory pressure.

        Examples
        --------
        >>> pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [1, 2, 2 << 32],
        ...         "c": [-1, 2, 1 << 30],
        ...         "d": [-112, 2, 112],
        ...         "e": [-112, 2, 129],
        ...         "f": ["a", "b", "c"],
        ...         "g": [0.1, 1.32, 0.12],
        ...         "h": [True, None, False],
        ...     }
        ... ).select(pl.all().shrink_dtype())
        shape: (3, 8)
        ┌─────┬────────────┬────────────┬──────┬──────┬─────┬──────┬───────┐
        │ a   ┆ b          ┆ c          ┆ d    ┆ e    ┆ f   ┆ g    ┆ h     │
        │ --- ┆ ---        ┆ ---        ┆ ---  ┆ ---  ┆ --- ┆ ---  ┆ ---   │
        │ i8  ┆ i64        ┆ i32        ┆ i8   ┆ i16  ┆ str ┆ f32  ┆ bool  │
        ╞═════╪════════════╪════════════╪══════╪══════╪═════╪══════╪═══════╡
        │ 1   ┆ 1          ┆ -1         ┆ -112 ┆ -112 ┆ a   ┆ 0.1  ┆ true  │
        │ 2   ┆ 2          ┆ 2          ┆ 2    ┆ 2    ┆ b   ┆ 1.32 ┆ null  │
        │ 3   ┆ 8589934592 ┆ 1073741824 ┆ 112  ┆ 129  ┆ c   ┆ 0.12 ┆ false │
        └─────┴────────────┴────────────┴──────┴──────┴─────┴──────┴───────┘

        """
        return wrap_expr(self._pyexpr.shrink_dtype())

    def map_dict(
        self,
        remapping: dict[Any, Any],
        *,
        default: Any = None,
    ) -> Expr:
        """
        Replace values in column according to remapping dictionary.

        Parameters
        ----------
        remapping
            Mapping dictionary to use for remapping the values.
        default
            Value to use when original value was not found in remapping dictionary.

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Examples
        --------
        >>> country_code_dict = {
        ...     "CA": "Canada",
        ...     "DE": "Germany",
        ...     "FR": "France",
        ...     None: "Not specified",
        ... }
        >>> df = pl.DataFrame(
        ...     {
        ...         "country_code": ["FR", None, "ES", "DE"],
        ...     }
        ... ).with_row_count()
        >>> df
        shape: (4, 2)
        ┌────────┬──────────────┐
        │ row_nr ┆ country_code │
        │ ---    ┆ ---          │
        │ u32    ┆ str          │
        ╞════════╪══════════════╡
        │ 0      ┆ FR           │
        │ 1      ┆ null         │
        │ 2      ┆ ES           │
        │ 3      ┆ DE           │
        └────────┴──────────────┘

        >>> df.with_columns(
        ...     pl.col("country_code").map_dict(country_code_dict).alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ null          │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        Set a default value for values that couldn't be mapped.

        >>> df.with_columns(
        ...     pl.col("country_code")
        ...     .map_dict(country_code_dict, default="unknown")
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ unknown       │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        Keep the original value for values that couldn't be mapped.

        >>> df.with_columns(
        ...     pl.col("country_code")
        ...     .map_dict(country_code_dict, default=pl.col("country_code"))
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ ES            │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        If you need to access different columns to set a default value for values that
        couldn't be mapped, a struct needs to be constructed, with in the first field
        the column that you want to remap and the rest of the fields the other columns
        you use in the default expression.

        >>> df.with_columns(
        ...     pl.struct(pl.col(["country_code", "row_nr"])).map_dict(
        ...         remapping=country_code_dict,
        ...         default=pl.col("row_nr").cast(pl.Utf8),
        ...     )
        ... )
        shape: (4, 2)
        ┌────────┬───────────────┐
        │ row_nr ┆ country_code  │
        │ ---    ┆ ---           │
        │ u32    ┆ str           │
        ╞════════╪═══════════════╡
        │ 0      ┆ France        │
        │ 1      ┆ Not specified │
        │ 2      ┆ 2             │
        │ 3      ┆ Germany       │
        └────────┴───────────────┘

        """

        # Use two functions to save unneeded work.
        # This factors out allocations and branches.
        def inner_with_default(s: pli.Series) -> pli.Series:
            # Convert Series to:
            #   - multicolumn DataFrame, if Series is a Struct.
            #   - one column DataFrame in other cases.
            df = s.to_frame().unnest(s.name) if s.dtype == Struct else s.to_frame()

            column = df.columns[0]
            remap_key_column = f"__POLARS_REMAP_KEY_{column}"
            remap_value_column = f"__POLARS_REMAP_VALUE_{column}"
            is_remapped_column = f"__POLARS_REMAP_IS_REMAPPED_{column}"

            return (
                (
                    df.lazy()
                    .join(
                        pli.DataFrame(
                            [
                                pli.Series(remap_key_column, list(remapping.keys())),
                                pli.Series(
                                    remap_value_column, list(remapping.values())
                                ),
                            ]
                        )
                        .lazy()
                        .with_columns(pli.lit(True).alias(is_remapped_column)),
                        how="left",
                        left_on=column,
                        right_on=remap_key_column,
                    )
                    .select(
                        pli.when(pli.col(is_remapped_column).is_not_null())
                        .then(pli.col(remap_value_column))
                        .otherwise(default)
                        .alias(column)
                    )
                )
                .collect(no_optimization=True)
                .to_series()
            )

        def inner(s: pli.Series) -> pli.Series:
            column = s.name
            remap_key_column = f"__POLARS_REMAP_KEY_{column}"
            remap_value_column = f"__POLARS_REMAP_VALUE_{column}"
            is_remapped_column = f"__POLARS_REMAP_IS_REMAPPED_{column}"

            return (
                (
                    s.to_frame()
                    .lazy()
                    .join(
                        pli.DataFrame(
                            [
                                pli.Series(remap_key_column, list(remapping.keys())),
                                pli.Series(
                                    remap_value_column, list(remapping.values())
                                ),
                            ]
                        )
                        .lazy()
                        .with_columns(pli.lit(True).alias(is_remapped_column)),
                        how="left",
                        left_on=column,
                        right_on=remap_key_column,
                    )
                )
                .collect(no_optimization=True)
                .to_series(1)
            )

        func = inner_with_default if default is not None else inner

        return self.map(func)

    @property
    def arr(self) -> ExprListNameSpace:
        """
        Create an object namespace of all list related methods.

        See the individual method pages for full details

        """
        return ExprListNameSpace(self)

    @property
    def cat(self) -> ExprCatNameSpace:
        """
        Create an object namespace of all categorical related methods.

        See the individual method pages for full details

        Examples
        --------
        >>> df = pl.DataFrame({"values": ["a", "b"]}).select(
        ...     pl.col("values").cast(pl.Categorical)
        ... )
        >>> df.select(pl.col("values").cat.set_ordering(ordering="physical"))
        shape: (2, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ cat    │
        ╞════════╡
        │ a      │
        │ b      │
        └────────┘

        """
        return ExprCatNameSpace(self)

    @property
    def dt(self) -> ExprDateTimeNameSpace:
        """Create an object namespace of all datetime related methods."""
        return ExprDateTimeNameSpace(self)

    @property
    def meta(self) -> ExprMetaNameSpace:
        """
        Create an object namespace of all meta related expression methods.

        This can be used to modify and traverse existing expressions

        """
        return ExprMetaNameSpace(self)

    @property
    def str(self) -> ExprStringNameSpace:
        """
        Create an object namespace of all string related methods.

        See the individual method pages for full details

        Examples
        --------
        >>> df = pl.DataFrame({"letters": ["a", "b"]})
        >>> df.select(pl.col("letters").str.to_uppercase())
        shape: (2, 1)
        ┌─────────┐
        │ letters │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ A       │
        │ B       │
        └─────────┘

        """
        return ExprStringNameSpace(self)

    @property
    def bin(self) -> ExprBinaryNameSpace:
        """
        Create an object namespace of all binary related methods.

        See the individual method pages for full details
        """
        return ExprBinaryNameSpace(self)

    @property
    def struct(self) -> ExprStructNameSpace:
        """
        Create an object namespace of all struct related methods.

        See the individual method pages for full details

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
        │ b   │
        └─────┘

        """
        return ExprStructNameSpace(self)


def _prepare_alpha(
    com: float | int | None = None,
    span: float | int | None = None,
    half_life: float | int | None = None,
    alpha: float | int | None = None,
) -> float:
    """Normalise EWM decay specification in terms of smoothing factor 'alpha'."""
    if sum((param is not None) for param in (com, span, half_life, alpha)) > 1:
        raise ValueError(
            "Parameters 'com', 'span', 'half_life', and 'alpha' are mutually exclusive"
        )
    if com is not None:
        if com < 0.0:
            raise ValueError(f"Require 'com' >= 0 (found {com})")
        alpha = 1.0 / (1.0 + com)

    elif span is not None:
        if span < 1.0:
            raise ValueError(f"Require 'span' >= 1 (found {span})")
        alpha = 2.0 / (span + 1.0)

    elif half_life is not None:
        if half_life <= 0.0:
            raise ValueError(f"Require 'half_life' > 0 (found {half_life})")
        alpha = 1.0 - math.exp(-math.log(2.0) / half_life)

    elif alpha is None:
        raise ValueError("One of 'com', 'span', 'half_life', or 'alpha' must be set")

    elif not (0 < alpha <= 1):
        raise ValueError(f"Require 0 < 'alpha' <= 1 (found {alpha})")

    return alpha


def _prepare_rolling_window_args(
    window_size: int | timedelta | str,
    min_periods: int | None = None,
) -> tuple[str, int]:
    if isinstance(window_size, int):
        if min_periods is None:
            min_periods = window_size
        window_size = f"{window_size}i"
    elif isinstance(window_size, timedelta):
        window_size = _timedelta_to_pl_duration(window_size)
    if min_periods is None:
        min_periods = 1
    return window_size, min_periods
