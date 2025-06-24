from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars._reexport as pl
from polars._utils.various import BUILDING_SPHINX_DOCS, sphinx_accessor
from polars.datatype_expr.array import DataTypeExprArrNameSpace
from polars.datatype_expr.enum import DataTypeExprEnumNameSpace
from polars.datatype_expr.int import DataTypeExprIntNameSpace
from polars.datatype_expr.list import DataTypeExprListNameSpace
from polars.datatype_expr.struct import DataTypeExprStructNameSpace

if TYPE_CHECKING:
    import contextlib
    from typing import ClassVar

    from polars import DataType
    from polars._typing import PolarsDataType, SchemaDict

    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyDataTypeExpr
elif BUILDING_SPHINX_DOCS:
    import sys

    # note: we assign this way to work around an autocomplete issue in ipython/jedi
    # (ref: https://github.com/davidhalter/jedi/issues/2057)
    current_module = sys.modules[__name__]
    current_module.property = sphinx_accessor


class DataTypeExpr:
    """
    A lazily instantiated :class:`DataType` that can be used in an :class:`Expr`.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    A expression made to represent a :class:`DataType` that can be used to
    reference the datatype of an expression of column in a lazy context.

    Examples
    --------
    >>> lf = pl.LazyFrame({"a": [1, 2, 3]})
    >>> lf.with_columns(
    ...     pl.col.a.map_batches(lambda x: x * 2, return_dtype=pl.dtype_of("a"))
    ... ).collect()
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 2   │
    │ 4   │
    │ 6   │
    └─────┘
    """

    # NOTE: This `= None` is needed to generate the docs with sphinx_accessor.
    _pydatatype_expr: PyDataTypeExpr = None
    _accessors: ClassVar[set[str]] = {
        "arr",
        "enum",
        "list",
        "int",
        "struct",
    }

    def __eq__(self, value: PolarsDataType | DataTypeExpr) -> pl.Expr:  # type: ignore[override]
        cmp_with: DataTypeExpr
        if isinstance(value, pl.DataType):
            cmp_with = value.to_dtype_expr()
        elif isinstance(value, pl.DataTypeClass):
            cmp_with = value.to_dtype_expr()
        elif isinstance(value, DataTypeExpr):
            cmp_with = value
        else:
            msg = f"cannot compare {self!r} to {value!r}"
            raise TypeError(msg) from None

        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.equals(cmp_with._pydatatype_expr)
        )

    def __ne__(self, value: PolarsDataType | DataTypeExpr) -> pl.Expr:  # type: ignore[override]
        return (self == value).not_()

    @classmethod
    def _from_pydatatype_expr(cls, pydatatype_expr: PyDataTypeExpr) -> DataTypeExpr:
        slf = cls()
        slf._pydatatype_expr = pydatatype_expr
        return slf

    def inner_dtype(self) -> DataTypeExpr:
        """Get the inner DataType of a List or Array."""
        return DataTypeExpr._from_pydatatype_expr(self._pydatatype_expr.inner_dtype())

    def to_string(self) -> pl.Expr:
        """
        Get a formatted version of the output DataType.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["X", "Y", "Z"],
        ...         "c": [1.3, 3.7, 4.2],
        ...     }
        ... )
        >>> df.select(
        ...     a=pl.dtype_of("a").to_string(),
        ...     b=pl.dtype_of("b").to_string(),
        ...     c=pl.dtype_of("c").to_string(),
        ... ).transpose(include_header=True, column_names=["dtype"])
        shape: (3, 2)
        ┌────────┬───────┐
        │ column ┆ dtype │
        │ ---    ┆ ---   │
        │ str    ┆ str   │
        ╞════════╪═══════╡
        │ a      ┆ i64   │
        │ b      ┆ str   │
        │ c      ┆ f64   │
        └────────┴───────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.to_string())

    def element_bitsize(self) -> pl.Expr:
        """
        Get the amount of bits used by every element of the output DataType.

        This raises an error for DataTypes that do not have a static amount of bits per
        element. This only includes the value and not the validity.
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.element_bitsize())

    def is_numeric(self) -> pl.Expr:
        """
        Get whether the output DataType is numeric.

        A numeric is any integer, float or decimal.
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_numeric())

    def is_integer(self) -> pl.Expr:
        """
        Get whether the output DataType is an integer.

        An integer is any signed or unsigned integer.
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_integer())

    def is_float(self) -> pl.Expr:
        """Get whether the output DataType is a float."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_float())

    def is_decimal(self) -> pl.Expr:
        """Get whether the output DataType is a decimal."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_decimal())

    def is_categorical(self) -> pl.Expr:
        """Get whether the output DataType is a categorical."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_categorical())

    def is_enum(self) -> pl.Expr:
        """Get whether the output DataType is an enum."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_enum())

    def is_nested(self) -> pl.Expr:
        """
        Get whether the output DataType is a nested datatype.

        A nested datatype is any list, array or struct.
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_nested())

    def is_list(self) -> pl.Expr:
        """Get whether the output DataType is a list."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_list())

    def is_array(self) -> pl.Expr:
        """Get whether the output DataType is an array."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_array())

    def is_struct(self) -> pl.Expr:
        """Get whether the output DataType is a struct."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_struct())

    def is_temporal(self) -> pl.Expr:
        """
        Get whether the output DataType is a temporal.

        A temporal is any date, datetime, duration or time.
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_temporal())

    def is_datetime(self) -> pl.Expr:
        """Get whether the output DataType is a datetime."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_datetime())

    def is_duration(self) -> pl.Expr:
        """Get whether the output DataType is a duration."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_duration())

    def is_object(self) -> pl.Expr:
        """Get whether the output DataType is an object."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.is_object())

    def wrap_in_list(self) -> DataTypeExpr:
        """
        Get the DataType wrapped in a list.

        Examples
        --------
        >>> pl.Int32.to_dtype_expr().wrap_in_list().collect_dtype({})
        List(Int32)

        """
        return DataTypeExpr._from_pydatatype_expr(self._pydatatype_expr.wrap_in_list())

    def wrap_in_array(self, *, width: int) -> DataTypeExpr:
        """
        Get the DataType wrapped in an array.

        Examples
        --------
        >>> pl.Int32.to_dtype_expr().wrap_in_array(width=5).collect_dtype({})
        Array(Int32, shape=(5,))
        """
        return DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.wrap_in_array(width)
        )

    @property
    def int(self) -> DataTypeExprIntNameSpace:
        """Create an object namespace of all integer related methods."""
        return DataTypeExprIntNameSpace(self)

    @property
    def enum(self) -> DataTypeExprEnumNameSpace:
        """Create an object namespace of all enum related methods."""
        return DataTypeExprEnumNameSpace(self)

    @property
    def list(self) -> DataTypeExprListNameSpace:
        """Create an object namespace of all list related methods."""
        return DataTypeExprListNameSpace(self)

    @property
    def arr(self) -> DataTypeExprArrNameSpace:
        """Create an object namespace of all array related methods."""
        return DataTypeExprArrNameSpace(self)

    @property
    def struct(self) -> DataTypeExprStructNameSpace:
        """Create an object namespace of all struct related methods."""
        return DataTypeExprStructNameSpace(self)

    def collect_dtype(
        self, context: SchemaDict | pl.Schema | pl.DataFrame | pl.LazyFrame
    ) -> DataType:
        """
        Materialize the :class:`DataTypeExpr` in a specific context.

        This is a useful function when debugging datatype expressions.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...     }
        ... )
        >>> pl.dtype_of("a").collect_dtype(lf)
        Int64
        >>> pl.dtype_of("a").collect_dtype({"a": pl.String})
        String
        """
        schema: pl.Schema
        if isinstance(context, pl.Schema):
            schema = context
        elif isinstance(context, Mapping):
            schema = pl.Schema(context)
        elif isinstance(context, pl.DataFrame):
            schema = context.schema
        elif isinstance(context, pl.LazyFrame):
            schema = context.collect_schema()
        else:
            msg = f"DataTypeExpr.collect_dtype did not expect {context!r}"
            raise TypeError(msg)

        return self._pydatatype_expr.collect_dtype(schema)
