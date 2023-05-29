from __future__ import annotations

import contextlib
import math
import operator
import os
import random
from datetime import timedelta
from functools import partial, reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    FrozenSet,
    Iterable,
    NoReturn,
    Sequence,
    Set,
    TypeVar,
)

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    Categorical,
    Struct,
    UInt32,
    Utf8,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np
from polars.expr.array import ExprArrayNameSpace
from polars.expr.binary import ExprBinaryNameSpace
from polars.expr.categorical import ExprCatNameSpace
from polars.expr.datetime import ExprDateTimeNameSpace
from polars.expr.list import ExprListNameSpace
from polars.expr.meta import ExprMetaNameSpace
from polars.expr.string import ExprStringNameSpace
from polars.expr.struct import ExprStructNameSpace
from polars.utils._parse_expr_input import (
    parse_as_expression,
    parse_as_list_of_expressions,
)
from polars.utils.convert import _timedelta_to_pl_duration
from polars.utils.decorators import deprecated_alias
from polars.utils.meta import threadpool_size
from polars.utils.various import sphinx_accessor

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import arg_where as py_arg_where
    from polars.polars import reduce as pyreduce

if TYPE_CHECKING:
    import sys

    from polars import DataFrame, LazyFrame, Series
    from polars.polars import PyExpr
    from polars.type_aliases import (
        ApplyStrategy,
        ClosedInterval,
        FillNullStrategy,
        InterpolationMethod,
        IntoExpr,
        NullBehavior,
        PolarsDataType,
        PythonLiteral,
        RankMethod,
        RollingInterpolationMethod,
        SearchSortedSide,
        WindowMappingStrategy,
    )

    if sys.version_info >= (3, 11):
        from typing import Concatenate, ParamSpec, Self
    else:
        from typing_extensions import Concatenate, ParamSpec, Self

    T = TypeVar("T")
    P = ParamSpec("P")

elif os.getenv("BUILDING_SPHINX_DOCS"):
    property = sphinx_accessor


class Expr:
    """Expressions that can be used in various contexts."""

    _pyexpr: PyExpr = None
    _accessors: set[str] = {"arr", "cat", "dt", "list", "meta", "str", "bin", "struct"}

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Self:
        expr = cls.__new__(cls)
        expr._pyexpr = pyexpr
        return expr

    def _to_pyexpr(self, other: Any) -> PyExpr:
        return self._to_expr(other)._pyexpr

    def _to_expr(self, other: Any) -> Expr:
        if isinstance(other, Expr):
            return other
        return F.lit(other)

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

    def __abs__(self) -> Self:
        return self.abs()

    # operators
    def __add__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr + self._to_pyexpr(other))

    def __radd__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other) + self._pyexpr)

    def __and__(self, other: Expr | int) -> Self:
        return self._from_pyexpr(self._pyexpr._and(self._to_pyexpr(other)))

    def __rand__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr._and(self._to_pyexpr(other)))

    def __eq__(self, other: Any) -> Self:  # type: ignore[override]
        return self._from_pyexpr(self._pyexpr.eq(self._to_expr(other)._pyexpr))

    def __floordiv__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr // self._to_pyexpr(other))

    def __rfloordiv__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other) // self._pyexpr)

    def __ge__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr.gt_eq(self._to_expr(other)._pyexpr))

    def __gt__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr.gt(self._to_expr(other)._pyexpr))

    def __invert__(self) -> Self:
        return self.is_not()

    def __le__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr.lt_eq(self._to_expr(other)._pyexpr))

    def __lt__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr.lt(self._to_expr(other)._pyexpr))

    def __mod__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr % self._to_pyexpr(other))

    def __rmod__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other) % self._pyexpr)

    def __mul__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr * self._to_pyexpr(other))

    def __rmul__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other) * self._pyexpr)

    def __ne__(self, other: Any) -> Self:  # type: ignore[override]
        return self._from_pyexpr(self._pyexpr.neq(self._to_expr(other)._pyexpr))

    def __neg__(self) -> Expr:
        return F.lit(0) - self

    def __or__(self, other: Expr | int) -> Self:
        return self._from_pyexpr(self._pyexpr._or(self._to_pyexpr(other)))

    def __ror__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other)._or(self._pyexpr))

    def __pos__(self) -> Expr:
        return F.lit(0) + self

    def __pow__(self, power: int | float | Series | Expr) -> Self:
        return self.pow(power)

    def __rpow__(self, base: int | float | Expr) -> Expr:
        return parse_as_expression(base) ** self

    def __sub__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr - self._to_pyexpr(other))

    def __rsub__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other) - self._pyexpr)

    def __truediv__(self, other: Any) -> Self:
        return self._from_pyexpr(self._pyexpr / self._to_pyexpr(other))

    def __rtruediv__(self, other: Any) -> Self:
        return self._from_pyexpr(self._to_pyexpr(other) / self._pyexpr)

    def __xor__(self, other: Expr) -> Self:
        return self._from_pyexpr(self._pyexpr._xor(self._to_pyexpr(other)))

    def __rxor__(self, other: Expr) -> Self:
        return self._from_pyexpr(self._pyexpr._xor(self._to_pyexpr(other)))

    # state
    def __getstate__(self) -> Any:
        return self._pyexpr.__getstate__()

    def __setstate__(self, state: Any) -> None:
        # init with a dummy
        self._pyexpr = F.lit(0)._pyexpr
        self._pyexpr.__setstate__(state)

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Self:
        """Numpy universal functions."""
        num_expr = sum(isinstance(inp, Expr) for inp in inputs)
        if num_expr > 1:
            if num_expr < len(inputs):
                raise ValueError(
                    "Numpy ufunc with more than one expression can only be used if all non-expression inputs are provided as keyword arguments only"
                )

            exprs = parse_as_list_of_expressions(inputs)
            return self._from_pyexpr(pyreduce(partial(ufunc, **kwargs), exprs))

        def function(s: Series) -> Series:  # pragma: no cover
            args = [inp if not isinstance(inp, Expr) else s for inp in inputs]
            return ufunc(*args, **kwargs)

        return self.map(function)

    def to_physical(self) -> Self:
        """
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - ``List(inner)`` -> ``List(physical of inner)``

        Other data types will be left unchanged.

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
        return self._from_pyexpr(self._pyexpr.to_physical())

    def any(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.any())

    def all(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.all())

    def arg_true(self) -> Self:
        """
        Return indices where expression evaluates `True`.

        .. warning::
            Modifies number of rows returned, so will fail in combination with other
            expressions. Use as only expression in `select` / `with_columns`.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2, 1]})
        >>> df.select((pl.col("a") == 1).arg_true())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        │ 1   │
        │ 3   │
        └─────┘

        See Also
        --------
        Series.arg_true : Return indices where Series is True
        pl.arg_where

        """
        return self._from_pyexpr(py_arg_where(self._pyexpr))

    def sqrt(self) -> Self:
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

    def log10(self) -> Self:
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

    def exp(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.exp())

    def alias(self, name: str) -> Self:
        """
        Rename the output of an expression.

        Parameters
        ----------
        name
            New name.

        See Also
        --------
        map_alias
        prefix
        suffix

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
        ...     pl.col("a").alias("bar"),
        ...     pl.col("b").alias("foo"),
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
        return self._from_pyexpr(self._pyexpr.alias(name))

    def exclude(
        self,
        columns: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_columns: str | PolarsDataType,
    ) -> Self:
        """
        Exclude columns from a multi-column expression.

        Only works after a wildcard or regex column selection.

        Parameters
        ----------
        columns
            The name or datatype of the column(s) to exclude. Accepts regular expression
            input. Regular expressions should start with ``^`` and end with ``$``.
        *more_columns
            Additional names or datatypes of columns to exclude, specified as positional
            arguments.

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
        if more_columns:
            if isinstance(columns, str):
                columns_str = [columns]
                columns_str.extend(more_columns)  # type: ignore[arg-type]
                return self._from_pyexpr(self._pyexpr.exclude(columns_str))
            elif is_polars_dtype(columns):
                dtypes = [columns]
                dtypes.extend(more_columns)
                return self._from_pyexpr(self._pyexpr.exclude_dtype(dtypes))
            else:
                raise TypeError(
                    f"Invalid input for `exclude`. Expected `str` or `DataType`, got {type(columns)!r}"
                )

        if isinstance(columns, str):
            return self._from_pyexpr(self._pyexpr.exclude([columns]))
        elif is_polars_dtype(columns):
            return self._from_pyexpr(self._pyexpr.exclude_dtype([columns]))
        elif isinstance(columns, Iterable):
            columns_list = list(columns)
            if not columns_list:
                return self

            item = columns_list[0]
            if isinstance(item, str):
                return self._from_pyexpr(self._pyexpr.exclude(columns_list))
            elif is_polars_dtype(item):
                return self._from_pyexpr(self._pyexpr.exclude_dtype(columns_list))
            else:
                raise TypeError(
                    "Invalid input for `exclude`. Expected iterable of type `str` or `DataType`,"
                    f" got iterable of type {type(item)!r}"
                )
        else:
            raise TypeError(
                f"Invalid input for `exclude`. Expected `str` or `DataType`, got {type(columns)!r}"
            )

    def keep_name(self) -> Self:
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

        >>> df.select((pl.lit(10) / pl.all()).keep_name())
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
        return self._from_pyexpr(self._pyexpr.keep_name())

    def pipe(
        self,
        function: Callable[Concatenate[Expr, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        r'''
        Offers a structured way to apply a sequence of user-defined functions (UDFs).

        Parameters
        ----------
        function
            Callable; will receive the expression as the first parameter,
            followed by any given args/kwargs.
        *args
            Arguments to pass to the UDF.
        **kwargs
            Keyword arguments to pass to the UDF.

        Examples
        --------
        >>> def extract_number(expr: pl.Expr) -> pl.Expr:
        ...     """Extract the digits from a string."""
        ...     return expr.str.extract(r"\d+", 0).cast(pl.Int64)
        >>>
        >>> def scale_negative_even(expr: pl.Expr, *, n: int = 1) -> pl.Expr:
        ...     """Set even numbers negative, and scale by a user-supplied value."""
        ...     expr = pl.when(expr % 2 == 0).then(-expr).otherwise(expr)
        ...     return expr * n
        >>>
        >>> df = pl.DataFrame({"val": ["a: 1", "b: 2", "c: 3", "d: 4"]})
        >>> df.with_columns(
        ...     udfs=(
        ...         pl.col("val").pipe(extract_number).pipe(scale_negative_even, n=5)
        ...     ),
        ... )
        shape: (4, 2)
        ┌──────┬──────┐
        │ val  ┆ udfs │
        │ ---  ┆ ---  │
        │ str  ┆ i64  │
        ╞══════╪══════╡
        │ a: 1 ┆ 5    │
        │ b: 2 ┆ -10  │
        │ c: 3 ┆ 15   │
        │ d: 4 ┆ -20  │
        └──────┴──────┘

        '''
        return function(self, *args, **kwargs)

    def prefix(self, prefix: str) -> Self:
        """
        Add a prefix to the root column name of the expression.

        Parameters
        ----------
        prefix
            Prefix to add to root column name.

        See Also
        --------
        alias
        map_alias
        suffix

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
        ...     pl.all(),
        ...     pl.all().reverse().prefix("reverse_"),
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
        return self._from_pyexpr(self._pyexpr.prefix(prefix))

    def suffix(self, suffix: str) -> Self:
        """
        Add a suffix to the root column name of the expression.

        Parameters
        ----------
        suffix
            Suffix to add to root column name.

        See Also
        --------
        alias
        map_alias
        prefix

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
        ...     pl.all(),
        ...     pl.all().reverse().suffix("_reverse"),
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
        return self._from_pyexpr(self._pyexpr.suffix(suffix))

    def map_alias(self, function: Callable[[str], str]) -> Self:
        """
        Rename the output of an expression by mapping a function over the root name.

        Parameters
        ----------
        function
            Function that maps root name to new name.

        See Also
        --------
        alias
        prefix
        suffix

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2],
        ...         "B": [3, 4],
        ...     }
        ... )

        >>> df.select(pl.all().reverse().suffix("_reverse")).with_columns(
        ...     pl.all().map_alias(
        ...         # Remove "_reverse" suffix and convert to lower case.
        ...         lambda col_name: col_name.rsplit("_reverse", 1)[0].lower()
        ...     )
        ... )
        shape: (2, 4)
        ┌───────────┬───────────┬─────┬─────┐
        │ A_reverse ┆ B_reverse ┆ a   ┆ b   │
        │ ---       ┆ ---       ┆ --- ┆ --- │
        │ i64       ┆ i64       ┆ i64 ┆ i64 │
        ╞═══════════╪═══════════╪═════╪═════╡
        │ 2         ┆ 4         ┆ 2   ┆ 4   │
        │ 1         ┆ 3         ┆ 1   ┆ 3   │
        └───────────┴───────────┴─────┴─────┘


        """
        return self._from_pyexpr(self._pyexpr.map_alias(function))

    def is_not(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_not())

    def is_null(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_null())

    def is_not_null(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_not_null())

    def is_finite(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_finite())

    def is_infinite(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_infinite())

    def is_nan(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_nan())

    def is_not_nan(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_not_nan())

    def agg_groups(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.agg_groups())

    def count(self) -> Self:
        """
        Count the number of values in this expression.

        .. warning::
            `null` is deemed a value in this context.

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
        return self._from_pyexpr(self._pyexpr.count())

    def len(self) -> Self:
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

    def slice(self, offset: int | Expr, length: int | Expr | None = None) -> Self:
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
            offset = F.lit(offset)
        if not isinstance(length, Expr):
            length = F.lit(length)
        return self._from_pyexpr(self._pyexpr.slice(offset._pyexpr, length._pyexpr))

    def append(self, other: Expr, *, upcast: bool = True) -> Self:
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
        other = parse_as_expression(other)._pyexpr
        return self._from_pyexpr(self._pyexpr.append(other, upcast))

    def rechunk(self) -> Self:
        """
        Create a single chunk of memory for this Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})

        Create a Series with 3 nulls, append column a then rechunk

        >>> df.select(pl.repeat(None, 3).append(pl.col("a")).rechunk())
        shape: (6, 1)
        ┌────────┐
        │ repeat │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ null   │
        │ null   │
        │ null   │
        │ 1      │
        │ 1      │
        │ 2      │
        └────────┘

        """
        return self._from_pyexpr(self._pyexpr.rechunk())

    def drop_nulls(self) -> Self:
        """
        Drop all null values.

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
        return self._from_pyexpr(self._pyexpr.drop_nulls())

    def drop_nans(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.drop_nans())

    def cumsum(self, *, reverse: bool = False) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cumsum(reverse))

    def cumprod(self, *, reverse: bool = False) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cumprod(reverse))

    def cummin(self, *, reverse: bool = False) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cummin(reverse))

    def cummax(self, *, reverse: bool = False) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cummax(reverse))

    def cumcount(self, *, reverse: bool = False) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cumcount(reverse))

    def floor(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.floor())

    def ceil(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.ceil())

    def round(self, decimals: int = 0) -> Self:
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
        return self._from_pyexpr(self._pyexpr.round(decimals))

    def dot(self, other: Expr | str) -> Self:
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
        other = parse_as_expression(other)._pyexpr
        return self._from_pyexpr(self._pyexpr.dot(other))

    def mode(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.mode())

    def cast(self, dtype: PolarsDataType | type[Any], *, strict: bool = True) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cast(dtype, strict))

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """
        Sort this column.

        When used in a projection/selection context, the whole column is sorted.
        When used in a groupby context, the groups are sorted.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, None, 3, 2],
        ...     }
        ... )
        >>> df.select(pl.col("a").sort())
        shape: (4, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        │ 1    │
        │ 2    │
        │ 3    │
        └──────┘
        >>> df.select(pl.col("a").sort(descending=True))
        shape: (4, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        │ 3    │
        │ 2    │
        │ 1    │
        └──────┘
        >>> df.select(pl.col("a").sort(nulls_last=True))
        shape: (4, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 1    │
        │ 2    │
        │ 3    │
        │ null │
        └──────┘

        When sorting in a groupby context, the groups are sorted.

        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["one", "one", "one", "two", "two", "two"],
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
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
        return self._from_pyexpr(self._pyexpr.sort_with(descending, nulls_last))

    def top_k(self, k: int = 5) -> Self:
        r"""
        Return the `k` largest elements.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.

        See Also
        --------
        bottom_k

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
        ...         pl.col("value").bottom_k().alias("bottom_k"),
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
        return self._from_pyexpr(self._pyexpr.top_k(k))

    def bottom_k(self, k: int = 5) -> Self:
        r"""
        Return the `k` smallest elements.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.

        See Also
        --------
        top_k

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
        ...         pl.col("value").bottom_k().alias("bottom_k"),
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
        return self._from_pyexpr(self._pyexpr.bottom_k(k))

    def arg_sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """
        Get the index values that would sort this column.

        Parameters
        ----------
        descending
            Sort in descending (descending) order.
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
        return self._from_pyexpr(self._pyexpr.arg_sort(descending, nulls_last))

    def arg_max(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arg_max())

    def arg_min(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arg_min())

    def search_sorted(
        self, element: Expr | int | float | Series, side: SearchSortedSide = "any"
    ) -> Self:
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
        element = parse_as_expression(element)._pyexpr
        return self._from_pyexpr(self._pyexpr.search_sorted(element, side))

    def sort_by(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        """
        Sort this column by the ordering of other columns.

        When used in a projection/selection context, the whole column is sorted.
        When used in a groupby context, the groups are sorted.

        Parameters
        ----------
        by
            Column(s) to sort by. Accepts expression input. Strings are parsed as column
            names.
        *more_by
            Additional columns to sort by, specified as positional arguments.
        descending
            Sort in descending order. When sorting by multiple columns, can be specified
            per column by passing a sequence of booleans.

        Examples
        --------
        Pass a single column name to sort by that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["a", "a", "b", "b"],
        ...         "value1": [1, 3, 4, 2],
        ...         "value2": [8, 7, 6, 5],
        ...     }
        ... )
        >>> df.select(pl.col("group").sort_by("value1"))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a     │
        │ b     │
        │ a     │
        │ b     │
        └───────┘

        Sorting by expressions is also supported.

        >>> df.select(pl.col("group").sort_by(pl.col("value1") + pl.col("value2")))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ b     │
        │ a     │
        │ a     │
        │ b     │
        └───────┘

        Sort by multiple columns by passing a list of columns.

        >>> df.select(pl.col("group").sort_by(["value1", "value2"], descending=True))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ b     │
        │ a     │
        │ b     │
        │ a     │
        └───────┘

        Or use positional arguments to sort by multiple columns in the same way.

        >>> df.select(pl.col("group").sort_by("value1", "value2"))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a     │
        │ b     │
        │ a     │
        │ b     │
        └───────┘

        When sorting in a groupby context, the groups are sorted.

        >>> df.groupby("group").agg(
        ...     pl.col("value1").sort_by("value2")
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌───────┬───────────┐
        │ group ┆ value1    │
        │ ---   ┆ ---       │
        │ str   ┆ list[i64] │
        ╞═══════╪═══════════╡
        │ a     ┆ [3, 1]    │
        │ b     ┆ [2, 4]    │
        └───────┴───────────┘

        Take a single row from each group where a column attains its minimal value
        within that group.

        >>> df.groupby("group").agg(
        ...     pl.all().sort_by("value2").first()
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 3)
        ┌───────┬────────┬────────┐
        │ group ┆ value1 ┆ value2 |
        │ ---   ┆ ---    ┆ ---    │
        │ str   ┆ i64    ┆ i64    |
        ╞═══════╪════════╪════════╡
        │ a     ┆ 3      ┆ 7      |
        │ b     ┆ 2      ┆ 5      |
        └───────┴────────┴────────┘

        """
        by = parse_as_list_of_expressions(by, *more_by)
        if isinstance(descending, bool):
            descending = [descending]
        elif len(by) != len(descending):
            raise ValueError(
                f"the length of `descending` ({len(descending)}) does not match the length of `by` ({len(by)})"
            )
        return self._from_pyexpr(self._pyexpr.sort_by(by, descending))

    def take(
        self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]
    ) -> Self:
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
            indices_lit = F.lit(pl.Series("", indices, dtype=UInt32))
        else:
            indices_lit = parse_as_expression(indices)  # type: ignore[arg-type]
        return self._from_pyexpr(self._pyexpr.take(indices_lit._pyexpr))

    def shift(self, periods: int = 1) -> Self:
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
        return self._from_pyexpr(self._pyexpr.shift(periods))

    def shift_and_fill(
        self,
        fill_value: IntoExpr,
        *,
        periods: int = 1,
    ) -> Self:
        """
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        fill_value
            Fill None values with the result of this expression.
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").shift_and_fill("a", periods=1))
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
        fill_value = parse_as_expression(fill_value, str_as_lit=True)._pyexpr
        return self._from_pyexpr(self._pyexpr.shift_and_fill(periods, fill_value))

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
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
            value = parse_as_expression(value, str_as_lit=True)._pyexpr
            return self._from_pyexpr(self._pyexpr.fill_null(value))
        else:
            return self._from_pyexpr(
                self._pyexpr.fill_null_with_strategy(strategy, limit)
            )

    def fill_nan(self, value: int | float | Expr | None) -> Self:
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
        fill_value = parse_as_expression(value, str_as_lit=True)._pyexpr
        return self._from_pyexpr(self._pyexpr.fill_nan(fill_value))

    def forward_fill(self, limit: int | None = None) -> Self:
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
        return self._from_pyexpr(self._pyexpr.forward_fill(limit))

    def backward_fill(self, limit: int | None = None) -> Self:
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
        return self._from_pyexpr(self._pyexpr.backward_fill(limit))

    def reverse(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.reverse())

    def std(self, ddof: int = 1) -> Self:
        """
        Get standard deviation.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

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
        return self._from_pyexpr(self._pyexpr.std(ddof))

    def var(self, ddof: int = 1) -> Self:
        """
        Get variance.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

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
        return self._from_pyexpr(self._pyexpr.var(ddof))

    def max(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.max())

    def min(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.min())

    def nan_max(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.nan_max())

    def nan_min(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.nan_min())

    def sum(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.sum())

    def mean(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.mean())

    def median(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.median())

    def product(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.product())

    def n_unique(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.n_unique())

    def approx_unique(self) -> Self:
        """
        Approx count unique values.

        This is done using the HyperLogLog++ algorithm for cardinality estimation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").approx_unique())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
        return self._from_pyexpr(self._pyexpr.approx_unique())

    def null_count(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.null_count())

    def arg_unique(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arg_unique())

    def unique(self, *, maintain_order: bool = False) -> Self:
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
            return self._from_pyexpr(self._pyexpr.unique_stable())
        return self._from_pyexpr(self._pyexpr.unique())

    def first(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.first())

    def last(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.last())

    def over(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        mapping_strategy: WindowMappingStrategy = "group_to_rows",
    ) -> Self:
        """
        Compute expressions over the given groups.

        This expression is similar to performing a groupby aggregation and joining the
        result back into the original dataframe.

        The outcome is similar to how `window functions
        <https://www.postgresql.org/docs/current/tutorial-window.html>`_
        work in PostgreSQL.

        Parameters
        ----------
        expr
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        *more_exprs
            Additional columns to group by, specified as positional arguments.
        mapping_strategy: {'group_to_rows', 'join', 'explode'}
            - group_to_rows
                If the aggregation results in multiple values, assign them back to there
                position in the DataFrame. This can only be done if the group yields
                the same elements before aggregation as after
            - join
                Join the groups as 'List<group_dtype>' to the row positions.
                warning: this can be memory intensive
            - explode
                Don't do any mapping, but simply flatten the group.
                This only makes sense if the input data is sorted.

        Examples
        --------
        Pass the name of a column to compute the expression over that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "a", "b", "b", "b"],
        ...         "b": [1, 2, 3, 5, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.with_columns(pl.col("c").max().over("a").suffix("_max"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_max │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 5     │
        │ b   ┆ 3   ┆ 3   ┆ 3     │
        │ b   ┆ 5   ┆ 2   ┆ 3     │
        │ b   ┆ 3   ┆ 1   ┆ 3     │
        └─────┴─────┴─────┴───────┘

        Expression input is supported.

        >>> df.with_columns(pl.col("c").max().over(pl.col("b") // 2).suffix("_max"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_max │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 4     │
        │ b   ┆ 3   ┆ 3   ┆ 4     │
        │ b   ┆ 5   ┆ 2   ┆ 2     │
        │ b   ┆ 3   ┆ 1   ┆ 4     │
        └─────┴─────┴─────┴───────┘

        Group by multiple columns by passing a list of column names or expressions.

        >>> df.with_columns(pl.col("c").min().over(["a", "b"]).suffix("_min"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_min │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 4     │
        │ b   ┆ 3   ┆ 3   ┆ 1     │
        │ b   ┆ 5   ┆ 2   ┆ 2     │
        │ b   ┆ 3   ┆ 1   ┆ 1     │
        └─────┴─────┴─────┴───────┘

        Or use positional arguments to group by multiple columns in the same way.

        >>> df.with_columns(pl.col("c").min().over("a", pl.col("b") % 2).suffix("_min"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_min │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 4     │
        │ b   ┆ 3   ┆ 3   ┆ 1     │
        │ b   ┆ 5   ┆ 2   ┆ 1     │
        │ b   ┆ 3   ┆ 1   ┆ 1     │
        └─────┴─────┴─────┴───────┘

        """
        exprs = parse_as_list_of_expressions(expr, *more_exprs)
        return self._from_pyexpr(self._pyexpr.over(exprs, mapping_strategy))

    def is_unique(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_unique())

    def is_first(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_first())

    def is_duplicated(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.is_duplicated())

    def quantile(
        self,
        quantile: float | Expr,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> Self:
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
        quantile = parse_as_expression(quantile)._pyexpr
        return self._from_pyexpr(self._pyexpr.quantile(quantile, interpolation))

    def filter(self, predicate: Expr) -> Self:
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
        return self._from_pyexpr(self._pyexpr.filter(predicate._pyexpr))

    def where(self, predicate: Expr) -> Self:
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

    def map(
        self,
        function: Callable[[Series], Series | Any],
        return_dtype: PolarsDataType | None = None,
        *,
        agg_list: bool = False,
    ) -> Self:
        """
        Apply a custom python function to a Series or sequence of Series.

        The output of this custom function must be a Series.
        If you want to apply a custom function elementwise over single values, see
        :func:`apply`. A use case for ``map`` is when you want to transform an
        expression with a third-party library.

        Read more in `the book
        <https://pola-rs.github.io/polars-book/user-guide/expressions/user-defined-functions>`_.

        Parameters
        ----------
        function
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        agg_list
            Aggregate list

        See Also
        --------
        map_dict

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
        return self._from_pyexpr(self._pyexpr.map(function, return_dtype, agg_list))

    def apply(
        self,
        function: Callable[[Series], Series] | Callable[[Any], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        skip_nulls: bool = True,
        pass_name: bool = False,
        strategy: ApplyStrategy = "thread_local",
    ) -> Self:
        """
        Apply a custom/user-defined function (UDF) in a GroupBy or Projection context.

        Depending on the context it has the following behavior:

        * Selection
            Expects `f` to be of type Callable[[Any], Any].
            Applies a python function over each individual value in the column.
        * GroupBy
            Expects `f` to be of type Callable[[Series], Series].
            Applies a python function over each group.

        Notes
        -----
        * Using ``apply`` is strongly discouraged as you will be effectively running
          python "for" loops. This will be very slow. Wherever possible you should
          strongly prefer the native expression API to achieve the best performance.

        * If your function is expensive and you don't want it to be called more than
          once for a given input, consider applying an ``@lru_cache`` decorator to it.
          With suitable data you may achieve order-of-magnitude speedups (or more).

        Parameters
        ----------
        function
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
            If not set, the dtype will be
            ``polars.Unknown``.
        skip_nulls
            Don't apply the function over values
            that contain nulls. This is faster.
        pass_name
            Pass the Series name to the custom function
            This is more expensive.
        strategy : {'thread_local', 'threading'}
            This functionality is in `alpha` stage. This may be removed
            /changed without it being considered a breaking change.

            - 'thread_local': run the python function on a single thread.
            - 'threading': run the python function on separate threads. Use with
                        care as this can slow performance. This might only speed up
                        your code if the amount of work per element is significant
                        and the python function releases the GIL (e.g. via calling
                        a c function)

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

            def wrap_f(x: Series) -> Series:  # pragma: no cover
                def inner(s: Series) -> Series:  # pragma: no cover
                    return function(s.alias(x.name))

                return x.apply(inner, return_dtype=return_dtype, skip_nulls=skip_nulls)

        else:

            def wrap_f(x: Series) -> Series:  # pragma: no cover
                return x.apply(
                    function, return_dtype=return_dtype, skip_nulls=skip_nulls
                )

        if strategy == "thread_local":
            return self.map(wrap_f, agg_list=True, return_dtype=return_dtype)
        elif strategy == "threading":

            def wrap_threading(x: Series) -> Series:
                df = x.to_frame("x")

                n_threads = threadpool_size()
                chunk_size = x.len() // n_threads
                remainder = x.len() % n_threads
                if chunk_size == 0:
                    chunk_sizes = [1 for _ in range(remainder)]
                else:
                    chunk_sizes = [
                        chunk_size + 1 if i < remainder else chunk_size
                        for i in range(n_threads)
                    ]

                def get_lazy_promise(df: DataFrame) -> LazyFrame:
                    return df.lazy().select(
                        F.col("x").map(wrap_f, agg_list=True, return_dtype=return_dtype)
                    )

                # create partitions with LazyFrames
                # these are promises on a computation
                partitions = []
                b = 0
                for step in chunk_sizes:
                    a = b
                    b = b + step
                    partition_df = df[a:b]
                    partitions.append(get_lazy_promise(partition_df))

                out = [df.to_series() for df in F.collect_all(partitions)]
                return F.concat(out, rechunk=False)

            return self.map(wrap_threading, agg_list=True, return_dtype=return_dtype)
        else:
            ValueError(f"Strategy {strategy} is not supported.")

    def flatten(self) -> Self:
        """
        Flatten a list or string column.

        Alias for :func:`polars.expr.list.ExprListNameSpace.explode`.

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
        return self._from_pyexpr(self._pyexpr.explode())

    def explode(self) -> Self:
        """
        Explode a list Series.

        This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype

        See Also
        --------
        ExprListNameSpace.explode : Explode a list column.
        ExprStringNameSpace.explode : Explode a string column.

        """
        return self._from_pyexpr(self._pyexpr.explode())

    def implode(self) -> Self:
        """
        Aggregate values into a list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [4, 5, 6],
        ...     }
        ... )
        >>> df.select(pl.all().implode())
        shape: (1, 2)
        ┌───────────┬───────────┐
        │ a         ┆ b         │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [1, 2, 3] ┆ [4, 5, 6] │
        └───────────┴───────────┘

        """
        return self._from_pyexpr(self._pyexpr.implode())

    def take_every(self, n: int) -> Self:
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
        return self._from_pyexpr(self._pyexpr.take_every(n))

    def head(self, n: int | Expr = 10) -> Self:
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
        return self.slice(0, n)

    def tail(self, n: int | Expr = 10) -> Self:
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
        offset = -parse_as_expression(n)
        return self.slice(offset, n)

    def limit(self, n: int | Expr = 10) -> Self:
        """
        Get the first `n` rows (alias for :func:`Expr.head`).

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7]})
        >>> df.limit(3)
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
        return self.head(n)

    def and_(self, *others: Any) -> Self:
        """
        Method equivalent of logical "and" operator ``expr & other & ...``.

        Parameters
        ----------
        *others
            One or more logical boolean expressions to evaluate/combine.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5, 6, 7, 4, 8],
        ...         "y": [1.5, 2.5, 1.0, 4.0, -5.75],
        ...         "z": [-9, 2, -1, 4, 8],
        ...     }
        ... )
        >>> df.select(
        ...     (pl.col("x") >= pl.col("z"))
        ...     .and_(
        ...         pl.col("y") >= pl.col("z"),
        ...         pl.col("y") == pl.col("y"),
        ...         pl.col("z") <= pl.col("x"),
        ...         pl.col("y") != pl.col("x"),
        ...     )
        ...     .alias("all")
        ... )
        shape: (5, 1)
        ┌───────┐
        │ all   │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ true  │
        │ true  │
        │ false │
        │ false │
        └───────┘

        """
        return reduce(operator.and_, (self,) + others)

    def or_(self, *others: Any) -> Self:
        """
        Method equivalent of logical "or" operator ``expr | other | ...``.

        Parameters
        ----------
        *others
            One or more logical boolean expressions to evaluate/combine.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5, 6, 7, 4, 8],
        ...         "y": [1.5, 2.5, 1.0, 4.0, -5.75],
        ...         "z": [-9, 2, -1, 4, 8],
        ...     }
        ... )
        >>> df.select(
        ...     (pl.col("x") == pl.col("y"))
        ...     .or_(
        ...         pl.col("x") == pl.col("y"),
        ...         pl.col("y") == pl.col("z"),
        ...         pl.col("y").cast(int) == pl.col("z"),
        ...     )
        ...     .alias("any")
        ... )
        shape: (5, 1)
        ┌───────┐
        │ any   │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ true  │
        │ false │
        │ true  │
        │ false │
        └───────┘

        """
        return reduce(operator.or_, (self,) + others)

    def eq(self, other: Any) -> Self:
        """
        Method equivalent of equality operator ``expr == other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 4.0],
        ...         "y": [2.0, 2.0, float("nan"), 4.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").eq(pl.col("y")).alias("x == y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x == y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 1.0 ┆ 2.0 ┆ false  │
        │ 2.0 ┆ 2.0 ┆ true   │
        │ NaN ┆ NaN ┆ false  │
        │ 4.0 ┆ 4.0 ┆ true   │
        └─────┴─────┴────────┘

        """
        return self.__eq__(other)

    def eq_missing(self, other: Any) -> Self:
        """
        Method equivalent of equality operator ``expr == other`` where `None` == None`.

        This differs from default ``eq`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        """
        return self._from_pyexpr(self._pyexpr.eq_missing(self._to_expr(other)._pyexpr))

    def ge(self, other: Any) -> Self:
        """
        Method equivalent of "greater than or equal" operator ``expr >= other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5.0, 4.0, float("nan"), 2.0],
        ...         "y": [5.0, 3.0, float("nan"), 1.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").ge(pl.col("y")).alias("x >= y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x >= y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 5.0 ┆ 5.0 ┆ true   │
        │ 4.0 ┆ 3.0 ┆ true   │
        │ NaN ┆ NaN ┆ false  │
        │ 2.0 ┆ 1.0 ┆ true   │
        └─────┴─────┴────────┘

        """
        return self.__ge__(other)

    def gt(self, other: Any) -> Self:
        """
        Method equivalent of "greater than" operator ``expr > other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5.0, 4.0, float("nan"), 2.0],
        ...         "y": [5.0, 3.0, float("nan"), 1.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").gt(pl.col("y")).alias("x > y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────┐
        │ x   ┆ y   ┆ x > y │
        │ --- ┆ --- ┆ ---   │
        │ f64 ┆ f64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 5.0 ┆ 5.0 ┆ false │
        │ 4.0 ┆ 3.0 ┆ true  │
        │ NaN ┆ NaN ┆ false │
        │ 2.0 ┆ 1.0 ┆ true  │
        └─────┴─────┴───────┘

        """
        return self.__gt__(other)

    def le(self, other: Any) -> Self:
        """
        Method equivalent of "less than or equal" operator ``expr <= other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5.0, 4.0, float("nan"), 0.5],
        ...         "y": [5.0, 3.5, float("nan"), 2.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").le(pl.col("y")).alias("x <= y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x <= y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 5.0 ┆ 5.0 ┆ true   │
        │ 4.0 ┆ 3.5 ┆ false  │
        │ NaN ┆ NaN ┆ false  │
        │ 0.5 ┆ 2.0 ┆ true   │
        └─────┴─────┴────────┘

        """
        return self.__le__(other)

    def lt(self, other: Any) -> Self:
        """
        Method equivalent of "less than" operator ``expr < other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 3.0],
        ...         "y": [2.0, 2.0, float("nan"), 4.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").lt(pl.col("y")).alias("x < y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────┐
        │ x   ┆ y   ┆ x < y │
        │ --- ┆ --- ┆ ---   │
        │ f64 ┆ f64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 1.0 ┆ 2.0 ┆ true  │
        │ 2.0 ┆ 2.0 ┆ false │
        │ NaN ┆ NaN ┆ false │
        │ 3.0 ┆ 4.0 ┆ true  │
        └─────┴─────┴───────┘

        """
        return self.__lt__(other)

    def ne(self, other: Any) -> Self:
        """
        Method equivalent of inequality operator ``expr != other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 4.0],
        ...         "y": [2.0, 2.0, float("nan"), 4.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").ne(pl.col("y")).alias("x != y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x != y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 1.0 ┆ 2.0 ┆ true   │
        │ 2.0 ┆ 2.0 ┆ false  │
        │ NaN ┆ NaN ┆ true   │
        │ 4.0 ┆ 4.0 ┆ false  │
        └─────┴─────┴────────┘

        """
        return self.__ne__(other)

    def ne_missing(self, other: Any) -> Self:
        """
        Method equivalent of equality operator ``expr != other`` where `None` == None`.

        This differs from default ``ne`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        """
        return self._from_pyexpr(self._pyexpr.neq_missing(self._to_expr(other)._pyexpr))

    def add(self, other: Any) -> Self:
        """
        Method equivalent of addition operator ``expr + other``.

        Parameters
        ----------
        other
            numeric or string value; accepts expression input.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        >>> df.with_columns(
        ...     pl.col("x").add(2).alias("x+int"),
        ...     pl.col("x").add(pl.col("x").cumprod()).alias("x+expr"),
        ... )
        shape: (5, 3)
        ┌─────┬───────┬────────┐
        │ x   ┆ x+int ┆ x+expr │
        │ --- ┆ ---   ┆ ---    │
        │ i64 ┆ i64   ┆ i64    │
        ╞═════╪═══════╪════════╡
        │ 1   ┆ 3     ┆ 2      │
        │ 2   ┆ 4     ┆ 4      │
        │ 3   ┆ 5     ┆ 9      │
        │ 4   ┆ 6     ┆ 28     │
        │ 5   ┆ 7     ┆ 125    │
        └─────┴───────┴────────┘

        >>> df = pl.DataFrame(
        ...     {"x": ["a", "d", "g"], "y": ["b", "e", "h"], "z": ["c", "f", "i"]}
        ... )
        >>> df.with_columns(pl.col("x").add(pl.col("y")).add(pl.col("z")).alias("xyz"))
        shape: (3, 4)
        ┌─────┬─────┬─────┬─────┐
        │ x   ┆ y   ┆ z   ┆ xyz │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ str ┆ str ┆ str │
        ╞═════╪═════╪═════╪═════╡
        │ a   ┆ b   ┆ c   ┆ abc │
        │ d   ┆ e   ┆ f   ┆ def │
        │ g   ┆ h   ┆ i   ┆ ghi │
        └─────┴─────┴─────┴─────┘

        """
        return self.__add__(other)

    def floordiv(self, other: Any) -> Self:
        """
        Method equivalent of integer division operator ``expr // other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        >>> df.with_columns(
        ...     pl.col("x").truediv(2).alias("x/2"),
        ...     pl.col("x").floordiv(2).alias("x//2"),
        ... )
        shape: (5, 3)
        ┌─────┬─────┬──────┐
        │ x   ┆ x/2 ┆ x//2 │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ i64  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 0.5 ┆ 0    │
        │ 2   ┆ 1.0 ┆ 1    │
        │ 3   ┆ 1.5 ┆ 1    │
        │ 4   ┆ 2.0 ┆ 2    │
        │ 5   ┆ 2.5 ┆ 2    │
        └─────┴─────┴──────┘

        See Also
        --------
        truediv

        """
        return self.__floordiv__(other)

    def mod(self, other: Any) -> Self:
        """
        Method equivalent of modulus operator ``expr % other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
        >>> df.with_columns(pl.col("x").mod(2).alias("x%2"))
        shape: (5, 2)
        ┌─────┬─────┐
        │ x   ┆ x%2 │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 0   ┆ 0   │
        │ 1   ┆ 1   │
        │ 2   ┆ 0   │
        │ 3   ┆ 1   │
        │ 4   ┆ 0   │
        └─────┴─────┘

        """
        return self.__mod__(other)

    def mul(self, other: Any) -> Self:
        """
        Method equivalent of multiplication operator ``expr * other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 4, 8, 16]})
        >>> df.with_columns(
        ...     pl.col("x").mul(2).alias("x*2"),
        ...     pl.col("x").mul(pl.col("x").log(2)).alias("x * xlog2"),
        ... )
        shape: (5, 3)
        ┌─────┬─────┬───────────┐
        │ x   ┆ x*2 ┆ x * xlog2 │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ i64 ┆ f64       │
        ╞═════╪═════╪═══════════╡
        │ 1   ┆ 2   ┆ 0.0       │
        │ 2   ┆ 4   ┆ 2.0       │
        │ 4   ┆ 8   ┆ 8.0       │
        │ 8   ┆ 16  ┆ 24.0      │
        │ 16  ┆ 32  ┆ 64.0      │
        └─────┴─────┴───────────┘

        """
        return self.__mul__(other)

    def sub(self, other: Any) -> Self:
        """
        Method equivalent of subtraction operator ``expr - other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
        >>> df.with_columns(
        ...     pl.col("x").sub(2).alias("x-2"),
        ...     pl.col("x").sub(pl.col("x").cumsum()).alias("x-expr"),
        ... )
        shape: (5, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ x-2 ┆ x-expr │
        │ --- ┆ --- ┆ ---    │
        │ i64 ┆ i64 ┆ i64    │
        ╞═════╪═════╪════════╡
        │ 0   ┆ -2  ┆ 0      │
        │ 1   ┆ -1  ┆ 0      │
        │ 2   ┆ 0   ┆ -1     │
        │ 3   ┆ 1   ┆ -3     │
        │ 4   ┆ 2   ┆ -6     │
        └─────┴─────┴────────┘

        """
        return self.__sub__(other)

    def truediv(self, other: Any) -> Self:
        """
        Method equivalent of float division operator ``expr / other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Notes
        -----
        Zero-division behaviour follows IEEE-754:

        0/0: Invalid operation - mathematically undefined, returns NaN.
        n/0: On finite operands gives an exact infinite result, eg: ±infinity.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"x": [-2, -1, 0, 1, 2], "y": [0.5, 0.0, 0.0, -4.0, -0.5]}
        ... )
        >>> df.with_columns(
        ...     pl.col("x").truediv(2).alias("x/2"),
        ...     pl.col("x").truediv(pl.col("y")).alias("x/y"),
        ... )
        shape: (5, 4)
        ┌─────┬──────┬──────┬───────┐
        │ x   ┆ y    ┆ x/2  ┆ x/y   │
        │ --- ┆ ---  ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ f64  ┆ f64   │
        ╞═════╪══════╪══════╪═══════╡
        │ -2  ┆ 0.5  ┆ -1.0 ┆ -4.0  │
        │ -1  ┆ 0.0  ┆ -0.5 ┆ -inf  │
        │ 0   ┆ 0.0  ┆ 0.0  ┆ NaN   │
        │ 1   ┆ -4.0 ┆ 0.5  ┆ -0.25 │
        │ 2   ┆ -0.5 ┆ 1.0  ┆ -4.0  │
        └─────┴──────┴──────┴───────┘

        See Also
        --------
        floordiv

        """
        return self.__truediv__(other)

    def pow(self, exponent: int | float | Series | Expr) -> Self:
        """
        Method equivalent of exponentiation operator ``expr ** exponent``.

        Parameters
        ----------
        exponent
            Numeric literal or expression exponent value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 4, 8]})
        >>> df.with_columns(
        ...     pl.col("x").pow(3).alias("cube"),
        ...     pl.col("x").pow(pl.col("x").log(2)).alias("x ** xlog2"),
        ... )
        shape: (4, 3)
        ┌─────┬───────┬────────────┐
        │ x   ┆ cube  ┆ x ** xlog2 │
        │ --- ┆ ---   ┆ ---        │
        │ i64 ┆ f64   ┆ f64        │
        ╞═════╪═══════╪════════════╡
        │ 1   ┆ 1.0   ┆ 1.0        │
        │ 2   ┆ 8.0   ┆ 2.0        │
        │ 4   ┆ 64.0  ┆ 16.0       │
        │ 8   ┆ 512.0 ┆ 512.0      │
        └─────┴───────┴────────────┘

        """
        exponent = parse_as_expression(exponent)._pyexpr
        return self._from_pyexpr(self._pyexpr.pow(exponent))

    def xor(self, other: Any) -> Self:
        """
        Method equivalent of logical exclusive-or operator ``expr ^ other``.

        Parameters
        ----------
        other
            Integer or boolean value; accepts expression input.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"x": [True, False, True, False], "y": [True, True, False, False]}
        ... )
        >>> df.with_columns(pl.col("x").xor(pl.col("y")).alias("x ^ y"))
        shape: (4, 3)
        ┌───────┬───────┬───────┐
        │ x     ┆ y     ┆ x ^ y │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ false │
        │ false ┆ true  ┆ true  │
        │ true  ┆ false ┆ true  │
        │ false ┆ false ┆ false │
        └───────┴───────┴───────┘

        >>> def binary_string(n: int) -> str:
        ...     return bin(n)[2:].zfill(8)
        >>>
        >>> df = pl.DataFrame(
        ...     data={"x": [10, 8, 250, 66], "y": [1, 2, 3, 4]},
        ...     schema={"x": pl.UInt8, "y": pl.UInt8},
        ... )
        >>> df.with_columns(
        ...     pl.col("x").apply(binary_string).alias("bin_x"),
        ...     pl.col("y").apply(binary_string).alias("bin_y"),
        ...     pl.col("x").xor(pl.col("y")).alias("xor_xy"),
        ...     pl.col("x").xor(pl.col("y")).apply(binary_string).alias("bin_xor_xy"),
        ... )
        shape: (4, 6)
        ┌─────┬─────┬──────────┬──────────┬────────┬────────────┐
        │ x   ┆ y   ┆ bin_x    ┆ bin_y    ┆ xor_xy ┆ bin_xor_xy │
        │ --- ┆ --- ┆ ---      ┆ ---      ┆ ---    ┆ ---        │
        │ u8  ┆ u8  ┆ str      ┆ str      ┆ u8     ┆ str        │
        ╞═════╪═════╪══════════╪══════════╪════════╪════════════╡
        │ 10  ┆ 1   ┆ 00001010 ┆ 00000001 ┆ 11     ┆ 00001011   │
        │ 8   ┆ 2   ┆ 00001000 ┆ 00000010 ┆ 10     ┆ 00001010   │
        │ 250 ┆ 3   ┆ 11111010 ┆ 00000011 ┆ 249    ┆ 11111001   │
        │ 66  ┆ 4   ┆ 01000010 ┆ 00000100 ┆ 70     ┆ 01000110   │
        └─────┴─────┴──────────┴──────────┴────────┴────────────┘

        """
        return self.__xor__(other)

    def is_in(self, other: Expr | Collection[Any] | Series) -> Self:
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
        if isinstance(other, Collection) and not isinstance(other, str):
            if isinstance(other, (Set, FrozenSet)):
                other = sorted(other)
            other = F.lit(None) if len(other) == 0 else F.lit(pl.Series(other))
        else:
            other = parse_as_expression(other)
        return self._from_pyexpr(self._pyexpr.is_in(other._pyexpr))

    def repeat_by(self, by: pl.Series | Expr | str | int) -> Self:
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
        by = parse_as_expression(by)._pyexpr
        return self._from_pyexpr(self._pyexpr.repeat_by(by))

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        """
        Check if this expression is between the given start and end values.

        Parameters
        ----------
        lower_bound
            Lower bound value. Accepts expression input. Strings are parsed as column
            names, other non-expression inputs are parsed as literals.
        upper_bound
            Upper bound value. Accepts expression input. Strings are parsed as column
            names, other non-expression inputs are parsed as literals.
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
        lower_bound = parse_as_expression(lower_bound)
        upper_bound = parse_as_expression(upper_bound)

        if closed == "none":
            return (self > lower_bound) & (self < upper_bound)
        elif closed == "both":
            return (self >= lower_bound) & (self <= upper_bound)
        elif closed == "right":
            return (self > lower_bound) & (self <= upper_bound)
        elif closed == "left":
            return (self >= lower_bound) & (self < upper_bound)
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
    ) -> Self:
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
        return self._from_pyexpr(self._pyexpr.hash(k0, k1, k2, k3))

    def reinterpret(self, *, signed: bool = True) -> Self:
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
        return self._from_pyexpr(self._pyexpr.reinterpret(signed))

    def inspect(self, fmt: str = "{}") -> Self:
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

        def inspect(s: Series) -> Series:  # pragma: no cover
            print(fmt.format(s))
            return s

        return self.map(inspect, return_dtype=None, agg_list=True)

    def interpolate(self, method: InterpolationMethod = "linear") -> Self:
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
        │ …           ┆ …      │
        │ 7           ┆ 14.0   │
        │ 8           ┆ 16.0   │
        │ 9           ┆ 18.0   │
        │ 10          ┆ 20.0   │
        └─────────────┴────────┘

        """
        return self._from_pyexpr(self._pyexpr.interpolate(method))

    def rolling_min(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
            self._pyexpr.rolling_min(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_max(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal, for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
            self._pyexpr.rolling_max(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_mean(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
            self._pyexpr.rolling_mean(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_sum(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
            self._pyexpr.rolling_sum(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_std(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
            self._pyexpr.rolling_std(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_var(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
            self._pyexpr.rolling_var(
                window_size, weights, min_periods, center, by, closed
            )
        )

    def rolling_median(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
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
        *,
        center: bool = False,
        by: str | None = None,
        closed: ClosedInterval = "left",
    ) -> Self:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
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
        return self._from_pyexpr(
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
        function: Callable[[Series], Any],
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
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
        return self._from_pyexpr(
            self._pyexpr.rolling_apply(
                function, window_size, weights, min_periods, center
            )
        )

    def rolling_skew(self, window_size: int, *, bias: bool = True) -> Self:
        """
        Compute a rolling skew.

        Parameters
        ----------
        window_size
            Integer size of the rolling window.
        bias
            If False, the calculations are corrected for statistical bias.

        """
        return self._from_pyexpr(self._pyexpr.rolling_skew(window_size, bias))

    def abs(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.abs())

    def rank(
        self,
        method: RankMethod = "average",
        *,
        descending: bool = False,
        seed: int | None = None,
    ) -> Self:
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
        descending
            Rank in descending order.
        seed
            If `method="random"`, use this as seed.

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

        Use 'rank' with 'over' to rank within groups:

        >>> df = pl.DataFrame({"a": [1, 1, 2, 2, 2], "b": [6, 7, 5, 14, 11]})
        >>> df.with_columns(pl.col("b").rank().over("a").alias("rank"))
        shape: (5, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ rank │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ f32  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ 1.0  │
        │ 1   ┆ 7   ┆ 2.0  │
        │ 2   ┆ 5   ┆ 1.0  │
        │ 2   ┆ 14  ┆ 3.0  │
        │ 2   ┆ 11  ┆ 2.0  │
        └─────┴─────┴──────┘

        """
        return self._from_pyexpr(self._pyexpr.rank(method, descending, seed))

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Self:
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
        return self._from_pyexpr(self._pyexpr.diff(n, null_behavior))

    def pct_change(self, n: int = 1) -> Self:
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
        return self._from_pyexpr(self._pyexpr.pct_change(n))

    def skew(self, *, bias: bool = True) -> Self:
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
        return self._from_pyexpr(self._pyexpr.skew(bias))

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
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
        return self._from_pyexpr(self._pyexpr.kurtosis(fisher, bias))

    def clip(self, lower_bound: int | float, upper_bound: int | float) -> Self:
        """
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Lower bound.
        upper_bound
            Upper bound.

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
        return self._from_pyexpr(self._pyexpr.clip(lower_bound, upper_bound))

    def clip_min(self, lower_bound: int | float) -> Self:
        """
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Lower bound.

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
        return self._from_pyexpr(self._pyexpr.clip_min(lower_bound))

    def clip_max(self, upper_bound: int | float) -> Self:
        """
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        upper_bound
            Upper bound.

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
        return self._from_pyexpr(self._pyexpr.clip_max(upper_bound))

    def lower_bound(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.lower_bound())

    def upper_bound(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.upper_bound())

    def sign(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.sign())

    def sin(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.sin())

    def cos(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cos())

    def tan(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.tan())

    def arcsin(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arcsin())

    def arccos(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arccos())

    def arctan(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arctan())

    def sinh(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.sinh())

    def cosh(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.cosh())

    def tanh(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.tanh())

    def arcsinh(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arcsinh())

    def arccosh(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arccosh())

    def arctanh(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.arctanh())

    def reshape(self, dimensions: tuple[int, ...]) -> Self:
        """
        Reshape this Expr to a flat Series or a Series of Lists.

        Parameters
        ----------
        dimensions
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

        See Also
        --------
        ExprListNameSpace.explode : Explode a list column.

        """
        return self._from_pyexpr(self._pyexpr.reshape(dimensions))

    def shuffle(self, seed: int | None = None) -> Self:
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
        return self._from_pyexpr(self._pyexpr.shuffle(seed))

    @deprecated_alias(frac="fraction")
    def sample(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Self:
        """
        Sample from this expression.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `fraction`. Defaults to 1 if
            `fraction` is None.
        fraction
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
        >>> df.select(pl.col("a").sample(fraction=1.0, with_replacement=True, seed=1))
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
        if n is not None and fraction is not None:
            raise ValueError("cannot specify both `n` and `fraction`")

        if seed is None:
            seed = random.randint(0, 10000)

        if fraction is not None:
            return self._from_pyexpr(
                self._pyexpr.sample_frac(fraction, with_replacement, shuffle, seed)
            )

        if n is None:
            n = 1
        return self._from_pyexpr(
            self._pyexpr.sample_n(n, with_replacement, shuffle, seed)
        )

    def ewm_mean(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Self:
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
        return self._from_pyexpr(
            self._pyexpr.ewm_mean(alpha, adjust, min_periods, ignore_nulls)
        )

    def ewm_std(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Self:
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
        return self._from_pyexpr(
            self._pyexpr.ewm_std(alpha, adjust, bias, min_periods, ignore_nulls)
        )

    def ewm_var(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Self:
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
        return self._from_pyexpr(
            self._pyexpr.ewm_var(alpha, adjust, bias, min_periods, ignore_nulls)
        )

    def extend_constant(self, value: PythonLiteral | None, n: int) -> Self:
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

        return self._from_pyexpr(self._pyexpr.extend_constant(value, n))

    def value_counts(self, *, multithreaded: bool = False, sort: bool = False) -> Self:
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
        return self._from_pyexpr(self._pyexpr.value_counts(multithreaded, sort))

    def unique_counts(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.unique_counts())

    def log(self, base: float = math.e) -> Self:
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
        return self._from_pyexpr(self._pyexpr.log(base))

    def log1p(self) -> Self:
        """
        Compute the natural logarithm of each element plus one.

        This computes `log(1 + x)` but is more numerically stable for `x` close to zero.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").log1p())
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.693147 │
        │ 1.098612 │
        │ 1.386294 │
        └──────────┘

        """
        return self._from_pyexpr(self._pyexpr.log1p())

    def entropy(self, base: float = math.e, *, normalize: bool = True) -> Self:
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
        return self._from_pyexpr(self._pyexpr.entropy(base, normalize))

    def cumulative_eval(
        self, expr: Expr, min_periods: int = 1, *, parallel: bool = False
    ) -> Self:
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
        return self._from_pyexpr(
            self._pyexpr.cumulative_eval(expr._pyexpr, min_periods, parallel)
        )

    def set_sorted(self, *, descending: bool = False) -> Self:
        """
        Flags the expression as 'sorted'.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        descending
            Whether the `Series` order is descending.

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
        return self._from_pyexpr(self._pyexpr.set_sorted_flag(descending))

    def shrink_dtype(self) -> Self:
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
        return self._from_pyexpr(self._pyexpr.shrink_dtype())

    def cache(self) -> Self:
        """
        Cache this expression so that it only is executed once per context.

        This can actually hurt performance and can have a lot of contention.
        It is advised not to use it until actually benchmarked on your problem.

        """
        return self._from_pyexpr(self._pyexpr.cache())

    def map_dict(
        self,
        remapping: dict[Any, Any],
        *,
        default: Any = None,
        return_dtype: PolarsDataType | None = None,
    ) -> Self:
        """
        Replace values in column according to remapping dictionary.

        Needs a global string cache for lazily evaluated queries on columns of
        type ``pl.Categorical``.

        Parameters
        ----------
        remapping
            Dictionary containing the before/after values to map.
        default
            Value to use when the remapping dict does not contain the lookup value.
            Use ``pl.first()``, to keep the original value.
        return_dtype
            Set return dtype to override automatic return dtype determination.

        See Also
        --------
        map

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

        Set a default value for values that cannot be mapped...

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

        ...or keep the original value, by making use of ``pl.first()``:

        >>> df.with_columns(
        ...     pl.col("country_code")
        ...     .map_dict(country_code_dict, default=pl.first())
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

        ...or keep the original value, by explicitly referring to the column:

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

        If you need to access different columns to set a default value, a struct needs
        to be constructed; in the first field is the column that you want to remap and
        the rest of the fields are the other columns used in the default expression.

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

        Override return dtype:

        >>> df.with_columns(
        ...     pl.col("row_nr")
        ...     .map_dict({1: 7, 3: 4}, default=3, return_dtype=pl.UInt8)
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬──────────┐
        │ row_nr ┆ country_code ┆ remapped │
        │ ---    ┆ ---          ┆ ---      │
        │ u32    ┆ str          ┆ u8       │
        ╞════════╪══════════════╪══════════╡
        │ 0      ┆ FR           ┆ 3        │
        │ 1      ┆ null         ┆ 7        │
        │ 2      ┆ ES           ┆ 3        │
        │ 3      ┆ DE           ┆ 4        │
        └────────┴──────────────┴──────────┘

        """

        def _remap_key_or_value_series(
            name: str,
            values: Iterable[Any],
            dtype: PolarsDataType | None,
            dtype_if_empty: PolarsDataType | None,
            dtype_keys: PolarsDataType | None,
            is_keys: bool,
        ) -> Series:
            """
            Convert remapping keys or remapping values to `Series` with `dtype`.

            Try to convert the remapping keys or remapping values to `Series` with
            the specified dtype and check that none of the values are accidentally
            lost (replaced by nulls) during the conversion.

            Parameters
            ----------
            name
                Name of the keys or values series.
            values
                Values for the series: `remapping.keys()` or `remapping.values()`.
            dtype
                User specified dtype. If None,
            dtype_if_empty
                If no dtype is specified and values contains None, an empty list,
                or a list with only None values, set the Polars dtype of the Series
                data.
            dtype_keys
                If user set dtype is None, try to see if Series for remapping.values()
                can be converted to same dtype as the remapping.keys() Series dtype.
            is_keys
                If values contains keys or values from remapping dict.

            """
            try:
                if dtype is None:
                    # If no dtype was set, which should only happen when:
                    #     values = remapping.values()
                    # create a Series from those values and infer the dtype.
                    s = pl.Series(
                        name,
                        values,
                        dtype=None,
                        dtype_if_empty=dtype_if_empty,
                        strict=True,
                    )

                    if dtype_keys is not None:
                        if s.dtype == dtype_keys:
                            # Values Series has same dtype as keys Series.
                            dtype = s.dtype
                        elif (
                            (s.dtype in INTEGER_DTYPES and dtype_keys in INTEGER_DTYPES)
                            or (s.dtype in FLOAT_DTYPES and dtype_keys in FLOAT_DTYPES)
                            or (s.dtype == Utf8 and dtype_keys == Categorical)
                        ):
                            # Values Series and keys Series are of similar dtypes,
                            # that we can assume that the user wants the values Series
                            # of the same dtype as the key Series.
                            dtype = dtype_keys
                            s = pl.Series(
                                name,
                                values,
                                dtype=dtype_keys,
                                dtype_if_empty=dtype_if_empty,
                                strict=True,
                            )
                            if dtype != s.dtype:
                                raise ValueError(
                                    f"Remapping values for map_dict could not be converted to {dtype}: found {s.dtype}"
                                )
                else:
                    # dtype was set, which should always be the case when:
                    #     values = remapping.keys()
                    # and in cases where the user set the output dtype when:
                    #     values = remapping.values()
                    s = pl.Series(
                        name,
                        values,
                        dtype=dtype,
                        dtype_if_empty=dtype_if_empty,
                        strict=True,
                    )
                    if dtype != s.dtype:
                        raise ValueError(
                            f"Remapping {'keys' if is_keys else 'values'} for map_dict could not be converted to {dtype}: found {s.dtype}"
                        )

            except OverflowError as exc:
                if is_keys:
                    raise ValueError(
                        f"Remapping keys for map_dict could not be converted to {dtype}: {str(exc)}"
                    ) from exc
                else:
                    raise ValueError(
                        f"Choose a more suitable output dtype for map_dict as remapping value could not be converted to {dtype}: {str(exc)}"
                    ) from exc

            if is_keys:
                # values = remapping.keys()
                if s.null_count() == 0:  # noqa: SIM114
                    pass
                elif s.null_count() == 1 and None in remapping:
                    pass
                else:
                    raise ValueError(
                        f"Remapping keys for map_dict could not be converted to {dtype} without losing values in the conversion."
                    )
            else:
                # values = remapping.values()
                if s.null_count() == 0:  # noqa: SIM114
                    pass
                elif s.len() - s.null_count() == len(list(filter(None, values))):
                    pass
                else:
                    raise ValueError(
                        f"Remapping values for map_dict could not be converted to {dtype} without losing values in the conversion."
                    )

            return s

        # Use two functions to save unneeded work.
        # This factors out allocations and branches.
        def inner_with_default(s: Series) -> Series:
            # Convert Series to:
            #   - multicolumn DataFrame, if Series is a Struct.
            #   - one column DataFrame in other cases.
            df = s.to_frame().unnest(s.name) if s.dtype == Struct else s.to_frame()

            # For struct we always apply mapping to the first column.
            column = df.columns[0]
            input_dtype = df.dtypes[0]
            remap_key_column = f"__POLARS_REMAP_KEY_{column}"
            remap_value_column = f"__POLARS_REMAP_VALUE_{column}"
            is_remapped_column = f"__POLARS_REMAP_IS_REMAPPED_{column}"

            # Set output dtype:
            #  - to dtype, if specified.
            #  - to same dtype as expression specified as default value.
            #  - to None, if dtype was not specified and default was not an expression.
            return_dtype_ = (
                df.lazy().select(default).dtypes[0]
                if return_dtype is None and isinstance(default, Expr)
                else return_dtype
            )

            remap_key_s = _remap_key_or_value_series(
                name=remap_key_column,
                values=remapping.keys(),
                dtype=input_dtype,
                dtype_if_empty=input_dtype,
                dtype_keys=input_dtype,
                is_keys=True,
            )

            if return_dtype_:
                # Create remap value Series with specified output dtype.
                remap_value_s = pl.Series(
                    remap_value_column,
                    remapping.values(),
                    dtype=return_dtype_,
                    dtype_if_empty=input_dtype,
                )
            else:
                # Create remap value Series with same output dtype as remap key Series,
                # if possible (if both are integers, both are floats or remap value
                # Series is pl.Utf8 and remap key Series is pl.Categorical).
                remap_value_s = _remap_key_or_value_series(
                    name=remap_value_column,
                    values=remapping.values(),
                    dtype=None,
                    dtype_if_empty=input_dtype,
                    dtype_keys=input_dtype,
                    is_keys=False,
                )

            return (
                (
                    df.lazy()
                    .join(
                        pl.DataFrame(
                            [
                                remap_key_s,
                                remap_value_s,
                            ]
                        )
                        .lazy()
                        .with_columns(F.lit(True).alias(is_remapped_column)),
                        how="left",
                        left_on=column,
                        right_on=remap_key_column,
                    )
                    .select(
                        F.when(F.col(is_remapped_column).is_not_null())
                        .then(F.col(remap_value_column))
                        .otherwise(default)
                        .alias(column)
                    )
                )
                .collect(no_optimization=True)
                .to_series()
            )

        def inner(s: Series) -> Series:
            column = s.name
            input_dtype = s.dtype
            remap_key_column = f"__POLARS_REMAP_KEY_{column}"
            remap_value_column = f"__POLARS_REMAP_VALUE_{column}"
            is_remapped_column = f"__POLARS_REMAP_IS_REMAPPED_{column}"

            remap_key_s = _remap_key_or_value_series(
                name=remap_key_column,
                values=list(remapping.keys()),
                dtype=input_dtype,
                dtype_if_empty=input_dtype,
                dtype_keys=input_dtype,
                is_keys=True,
            )

            if return_dtype:
                # Create remap value Series with specified output dtype.
                remap_value_s = pl.Series(
                    remap_value_column,
                    remapping.values(),
                    dtype=return_dtype,
                    dtype_if_empty=input_dtype,
                )
            else:
                # Create remap value Series with same output dtype as remap key Series,
                # if possible (if both are integers, both are floats or remap value
                # Series is pl.Utf8 and remap key Series is pl.Categorical).
                remap_value_s = _remap_key_or_value_series(
                    name=remap_value_column,
                    values=remapping.values(),
                    dtype=None,
                    dtype_if_empty=input_dtype,
                    dtype_keys=input_dtype,
                    is_keys=False,
                )

            return (
                (
                    s.to_frame()
                    .lazy()
                    .join(
                        pl.DataFrame(
                            [
                                remap_key_s,
                                remap_value_s,
                            ]
                        )
                        .lazy()
                        .with_columns(F.lit(True).alias(is_remapped_column)),
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
    def bin(self) -> ExprBinaryNameSpace:
        """
        Create an object namespace of all binary related methods.

        See the individual method pages for full details
        """
        return ExprBinaryNameSpace(self)

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

    # Keep the `list` and `str` properties below at the end of the definition of Expr,
    # as to not confuse mypy with the type annotation `str` and `list`

    @property
    def list(self) -> ExprListNameSpace:
        """
        Create an object namespace of all list related methods.

        See the individual method pages for full details

        """
        return ExprListNameSpace(self)

    @property
    def arr(self) -> ExprArrayNameSpace:
        """
        Create an object namespace of all array related methods.

        See the individual method pages for full details

        """
        return ExprArrayNameSpace(self)

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
        if window_size < 1:
            raise ValueError("'window_size' should be positive")

        if min_periods is None:
            min_periods = window_size
        window_size = f"{window_size}i"
    elif isinstance(window_size, timedelta):
        window_size = _timedelta_to_pl_duration(window_size)
    if min_periods is None:
        min_periods = 1
    return window_size, min_periods
