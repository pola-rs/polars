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
    """TypeGuard a DataType into a Struct."""
    return datatype == Struct


def is_array(datatype: DataType | DataTypeClass) -> TypeGuard[Array]:
    """TypeGuard a DataType into a Array."""
    return datatype == Array


def is_list(datatype: DataType | DataTypeClass) -> TypeGuard[List]:
    """TypeGuard a DataType into a List."""
    return datatype == List


def is_decimal(datatype: DataType | DataTypeClass) -> TypeGuard[Decimal]:
    """TypeGuard a DataType into a Decimal."""
    return datatype == Decimal


def is_time(datatype: DataType | DataTypeClass) -> TypeGuard[Time]:
    """TypeGuard a DataType into a Time."""
    return datatype == Time


def is_datetime(datatype: DataType | DataTypeClass) -> TypeGuard[Datetime]:
    """TypeGuard a DataType into a Datetime."""
    return datatype == Datetime


def is_duration(datatype: DataType | DataTypeClass) -> TypeGuard[Duration]:
    """TypeGuard a DataType into a Datetime."""
    return datatype == Duration


def is_categorical(datatype: DataType | DataTypeClass) -> TypeGuard[Categorical]:
    """TypeGuard a DataType into a Categorical."""
    return datatype == Categorical


def is_enum(datatype: DataType | DataTypeClass) -> TypeGuard[Enum]:
    """TypeGuard a DataType into a Enum."""
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
