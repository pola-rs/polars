from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import (
    Array,
    Categorical,
    Datetime,
    Decimal,
    Duration,
    Enum,
    List,
    Struct,
    Time,
)

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

    from polars.datatypes import DataType, DataTypeClass


def is_struct(datatype: DataType | DataTypeClass) -> TypeGuard[Struct]:
    """Check if a DataType is a Struct with TypeGuard.

    Examples
    --------
    Non-working example
    >>> df = pl.DataFrame({"a": {"b": [1]}})
    >>> a_datatype = df.schema["a"]
    >>> if a_datatype == pl.Struct:
    ...     fields = a_datatype.fields
    The above example will work at runtime but will result in a type error because
    `fields` doesn't exist in the DataType class, only on Struct.

    Working example
    >>> if is_struct(a_datatype):
    ...     fields = a_datatype.fields
    The above works as expected because `is_struct` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Struct


def is_array(datatype: DataType | DataTypeClass) -> TypeGuard[Array]:
    """Check if a DataType is an Array with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series([[1]], dtype=pl.Array(pl.Int8, 1))
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Array:
    ...     size = a_datatype.size
    The above example will work at runtime but will result in a type error because
    `size` doesn't exist in the DataType class, only on Array.

    Working example
    >>> if is_array(a_datatype):
    ...     size = a_datatype.size
    The above works as expected because `is_array` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Array


def is_list(datatype: DataType | DataTypeClass) -> TypeGuard[List]:
    """Check if a DataType is a List with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series([[1, 2, 3]])
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.List:
    ...     inner = a_datatype.inner
    The above example will work at runtime but will result in a type error because
    `inner` doesn't exist in the DataType class, only on List.

    Working example
    >>> if is_list(a_datatype):
    ...     inner = a_datatype.inner_dtype
    The above works as expected because `is_list` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == List


def is_decimal(datatype: DataType | DataTypeClass) -> TypeGuard[Decimal]:
    """Check if a DataType is a Decimal with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series([1.23], dtype=pl.Decimal(10, 2))
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Decimal:
    ...     prec = a_datatype.precision
    The above example will work at runtime but will result in a type error because
    `precision` doesn't exist in the DataType class, only on Decimal.

    Working example
    >>> if is_decimal(a_datatype):
    ...     prec = a_datatype.precision
    The above works as expected because `is_decimal` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Decimal


def is_time(datatype: DataType | DataTypeClass) -> TypeGuard[Time]:
    """Check if a DataType is a Time with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series(["12:34:56"], dtype=pl.Time)
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Time:
    ...     u = a_datatype.max()
    The above example will work at runtime but will result in a type error because
    `max` doesn't exist in the DataType class, it does on Time.

    Working example
    >>> if is_time(a_datatype):
    ...     u = a_datatype.max()
    The above works as expected because `is_time` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Time


def is_datetime(datatype: DataType | DataTypeClass) -> TypeGuard[Datetime]:
    """Check if a DataType is a Datetime with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series([datetime(2025, 1, 1, 10, 30)])
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Datetime:
    ...     tz = a_datatype.time_zone
    The above example will work at runtime but will result in a type error because
    `time_zone` doesn't exist in the DataType class, only on Datetime.

    Working example
    >>> if is_datetime(a_datatype):
    ...     tz = a_datatype.time_zone
    The above works as expected because `is_datetime` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Datetime


def is_duration(datatype: DataType | DataTypeClass) -> TypeGuard[Duration]:
    """Check if a DataType is a Duration with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series([123456], dtype=pl.Duration("ns"))
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Duration:
    ...     u = a_datatype.time_unit
    The above example will work at runtime but will result in a type error because
    `time_unit` doesn't exist in the DataType class, only on Duration.

    Working example
    >>> if is_duration(a_datatype):
    ...     u = a_datatype.time_unit
    The above works as expected because `is_duration` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Duration


def is_categorical(datatype: DataType | DataTypeClass) -> TypeGuard[Categorical]:
    """Check if a DataType is Categorical with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series(["a", "b"], dtype=pl.Categorical)
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Categorical:
    ...     ordering = a_datatype.ordering
    The above example will work at runtime but will result in a type error because
    `ordering` doesn't exist in the DataType class, only on Categorical.

    Working example
    >>> if is_categorical(a_datatype):
    ...     ordering = a_datatype.ordering
    The above works as expected because `is_categorical` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Categorical


def is_enum(datatype: DataType | DataTypeClass) -> TypeGuard[Enum]:
    """Check if a DataType is an Enum with TypeGuard.

    Examples
    --------
    Non-working example
    >>> s = pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))
    >>> a_datatype = s.dtype
    >>> if a_datatype == pl.Enum:
    ...     categories = a_datatype.categories
    The above example will work at runtime but will result in a type error because
    `categories` doesn't exist in the DataType class, only on Enum.

    Working example
    >>> if is_enum(a_datatype):
    ...     categories = a_datatype.categories
    The above works as expected because `is_enum` is a TypeGuard that
    informs the type checker what the type should be when it resolves True.
    """
    return datatype == Enum


__all__ = [
    "is_struct",
    "is_array",
    "is_list",
    "is_decimal",
    "is_time",
    "is_datetime",
    "is_duration",
    "is_categorical",
    "is_enum",
]
