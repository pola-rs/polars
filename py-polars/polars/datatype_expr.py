from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars._reexport as pl
from polars._typing import PolarsDataType
from polars._utils.various import qualified_type_name

if TYPE_CHECKING:
    import contextlib

    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyDataTypeExpr
    from polars import DataType
    from polars._typing import SchemaDict


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

    _pydatatype_expr: PyDataTypeExpr

    def __eq__(self, value: PolarsDataType | DataTypeExpr) -> pl.Expr:
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

    def __ne__(self, value: PolarsDataType | DataTypeExpr) -> pl.Expr:
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
        """Get a formatted version of the output DataType."""
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
        """Get whether the output DataType is a enum."""
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
        """Get whether the output DataType is an struct."""
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
        """Get the DataType wrapped in a list."""
        return DataTypeExpr._from_pydatatype_expr(self._pydatatype_expr.wrap_in_list())

    def wrap_in_array(self, *, width: int) -> DataTypeExpr:
        """Get the DataType wrapped in an array."""
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
        """Materialize the :class:`DataTypeExpr` in a specific context."""
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


class DataTypeExprIntNameSpace:
    """Namespace for integers datatype expressions."""

    _accessor = "int"

    def __init__(self, expr: DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def to_unsigned(self) -> pl.DataTypeExpr:
        """Get the unsigned integer version of the same bitsize."""
        return DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.int_to_unsigned()
        )

    def to_signed(self) -> pl.DataTypeExpr:
        """Get the signed integer version of the same bitsize."""
        return DataTypeExpr._from_pydatatype_expr(self._pydatatype_expr.int_to_signed())

    def is_unsigned(self) -> pl.Expr:
        """Get whether the given integer is unsigned."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.int_is_unsigned())

    def is_signed(self) -> pl.Expr:
        """Get whether the given integer is signed."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.int_is_signed())


class DataTypeExprEnumNameSpace:
    """Namespace for enum datatype expressions."""

    _accessor = "enum"

    def __init__(self, expr: DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def num_categories(self) -> pl.Expr:
        """Get the number of enum categories."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.enum_num_categories())

    def categories(self) -> pl.Expr:
        """Get the enum categories as a list."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.enum_categories())

    def get_category(self, index: int, *, raise_on_oob: bool = True) -> pl.Expr:
        """Get the enum category at a specific index."""
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.enum_get_category(index, raise_on_oob)
        )

    def index_of_category(
        self, category: str, *, raise_on_missing: bool = True
    ) -> pl.Expr:
        """Get the index of a specific enum category."""
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.enum_index_of_category(category, raise_on_missing)
        )


class DataTypeExprListNameSpace:
    """Namespace for list datatype expressions."""

    _accessor = "list"

    def __init__(self, expr: DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def inner_dtype(self) -> DataTypeExpr:
        """Get the inner DataType of list."""
        return DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.list_inner_dtype()
        )


class DataTypeExprArrNameSpace:
    """Namespace for arr datatype expressions."""

    _accessor = "arr"

    def __init__(self, expr: DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def inner_dtype(self) -> DataTypeExpr:
        """Get the inner DataType of array."""
        return DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.arr_inner_dtype()
        )

    def has_width(self, width: int) -> pl.Expr:
        """Get whether an array has a specific width."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.arr_has_width(width))

    def width(self) -> pl.Expr:
        """
        Get the array width.

        Examples
        --------
        >>> pl.select(pl.Array(pl.Int8, (1, 2, 3)).to_dtype_expr().arr.width())
        shape: (1, 1)
        ┌───────────┐
        │ literal   │
        │ ---       │
        │ int       │
        ╞═══════════╡
        │ 1         │
        └───────────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.arr_width())

    def dimensions(self) -> pl.Expr:
        """
        Get the dimensions of sequentially nested arrays.

        Examples
        --------
        >>> pl.select(pl.Array(pl.Int8, (1, 2, 3)).to_dtype_expr().arr.dimensions())
        shape: (1, 1)
        ┌───────────┐
        │ literal   │
        │ ---       │
        │ list[u32] │
        ╞═══════════╡
        │ [1, 2, 3] │
        └───────────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.arr_dimensions())


class DataTypeExprStructNameSpace:
    """Namespace for struct datatype expressions."""

    _accessor = "struct"

    def __init__(self, expr: DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def __getitem__(self, item: str | int) -> DataTypeExpr:
        if isinstance(item, str):
            return self.field_dtype(item)
        elif isinstance(item, int):
            return DataTypeExpr._from_pydatatype_expr(
                self._pydatatype_expr.struct_field_dtype_by_index(item)
            )
        else:
            msg = f"expected type 'int | str', got {qualified_type_name(item)!r} ({item!r})"
            raise TypeError(msg)

    def field_dtype(self, field_name: str) -> DataTypeExpr:
        """Get the DataType of field with a specific field name."""
        return DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.struct_field_dtype_by_name(field_name)
        )

    def num_fields(self) -> pl.Expr:
        """Get the number of fields in a struct."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.struct_num_fields())

    def field_names(self) -> pl.Expr:
        """Get the field names in a struct as a list."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.struct_field_names())

    def field_name(self, index: int, *, raise_on_oob: bool = True) -> pl.Expr:
        """Get the n-th field name."""
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.struct_field_name(index, raise_on_oob)
        )

    def field_index(self, field_name: str, *, raise_on_missing: bool = True) -> pl.Expr:
        """Get the index of a field."""
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.struct_field_index(field_name, raise_on_missing)
        )
