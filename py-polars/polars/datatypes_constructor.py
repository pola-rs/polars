from typing import Any, Callable, Sequence, Type

import numpy as np

from polars.datatypes import (
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Object,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)

try:
    from polars.polars import PySeries

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


if not _DOCUMENTING:
    _POLARS_TYPE_TO_CONSTRUCTOR = {
        Float32: PySeries.new_opt_f32,
        Float64: PySeries.new_opt_f64,
        Int8: PySeries.new_opt_i8,
        Int16: PySeries.new_opt_i16,
        Int32: PySeries.new_opt_i32,
        Int64: PySeries.new_opt_i64,
        UInt8: PySeries.new_opt_u8,
        UInt16: PySeries.new_opt_u16,
        UInt32: PySeries.new_opt_u32,
        UInt64: PySeries.new_opt_u64,
        Date: PySeries.new_opt_i32,
        Datetime: PySeries.new_opt_i64,
        Duration: PySeries.new_opt_i64,
        Time: PySeries.new_opt_i64,
        Boolean: PySeries.new_opt_bool,
        Utf8: PySeries.new_str,
        Object: PySeries.new_object,
        Categorical: PySeries.new_str,
    }


def polars_type_to_constructor(
    dtype: Type[DataType],
) -> Callable[[str, Sequence[Any], bool], "PySeries"]:
    """
    Get the right PySeries constructor for the given Polars dtype.
    """
    try:
        return _POLARS_TYPE_TO_CONSTRUCTOR[dtype]
    except KeyError:  # pragma: no cover
        raise ValueError(f"Cannot construct PySeries for type {dtype}.")


if not _DOCUMENTING:
    _NUMPY_TYPE_TO_CONSTRUCTOR = {
        np.float32: PySeries.new_f32,
        np.float64: PySeries.new_f64,
        np.int8: PySeries.new_i8,
        np.int16: PySeries.new_i16,
        np.int32: PySeries.new_i32,
        np.int64: PySeries.new_i64,
        np.uint8: PySeries.new_u8,
        np.uint16: PySeries.new_u16,
        np.uint32: PySeries.new_u32,
        np.uint64: PySeries.new_u64,
        np.str_: PySeries.new_str,
        np.bool_: PySeries.new_bool,
    }


def numpy_type_to_constructor(dtype: Type[np.dtype]) -> Callable[..., "PySeries"]:
    """
    Get the right PySeries constructor for the given Polars dtype.
    """
    try:
        return _NUMPY_TYPE_TO_CONSTRUCTOR[dtype]
    except KeyError:
        return PySeries.new_object


if not _DOCUMENTING:
    _PY_TYPE_TO_CONSTRUCTOR = {
        float: PySeries.new_opt_f64,
        int: PySeries.new_opt_i64,
        str: PySeries.new_str,
        bool: PySeries.new_opt_bool,
    }


def py_type_to_constructor(dtype: Type[Any]) -> Callable[..., "PySeries"]:
    """
    Get the right PySeries constructor for the given Python dtype.
    """
    try:
        return _PY_TYPE_TO_CONSTRUCTOR[dtype]
    except KeyError:
        return PySeries.new_object
