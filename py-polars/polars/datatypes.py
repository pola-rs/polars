import ctypes
import typing as tp
from typing import Any, Callable, Dict, Sequence, Type

import numpy as np
import pyarrow as pa
from _ctypes import _SimpleCData

try:
    from polars.polars import PySeries

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

__all__ = [
    "DataType",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Boolean",
    "Utf8",
    "List",
    "Date32",
    "Date64",
    "Object",
    "Categorical",
    "DTYPES",
    "DTYPE_TO_FFINAME",
    "dtype_to_primitive",
    "dtype_to_ctype",
    "pytype_to_polars_type",
]


class DataType:
    pass


class Int8(DataType):
    pass


class Int16(DataType):
    pass


class Int32(DataType):
    pass


class Int64(DataType):
    pass


class UInt8(DataType):
    pass


class UInt16(DataType):
    pass


class UInt32(DataType):
    pass


class UInt64(DataType):
    pass


class Float32(DataType):
    pass


class Float64(DataType):
    pass


class Boolean(DataType):
    pass


class Utf8(DataType):
    pass


class List(DataType):
    pass


class Date32(DataType):
    pass


class Date64(DataType):
    pass


class Time32Millisecond(DataType):
    pass


class Time32Second(DataType):
    pass


class Time64Nanosecond(DataType):
    pass


class Time64Microsecond(DataType):
    pass


class DurationNanosecond(DataType):
    pass


class DurationMicrosecond(DataType):
    pass


class DurationMillisecond(DataType):
    pass


class DurationSecond(DataType):
    pass


class TimestampNanosecond(DataType):
    pass


class TimestampMicrosecond(DataType):
    pass


class TimestampMillisecond(DataType):
    pass


class TimestampSecond(DataType):
    pass


class Object(DataType):
    pass


class Categorical(DataType):
    pass


# Don't change the order of these!
DTYPES: tp.List[Type[DataType]] = [
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Boolean,
    Utf8,
    List,
    Date32,
    Date64,
    Time64Nanosecond,
    DurationNanosecond,
    DurationMillisecond,
    Object,
    Categorical,
]
DTYPE_TO_FFINAME: Dict[Type[DataType], str] = {
    Int8: "i8",
    Int16: "i16",
    Int32: "i32",
    Int64: "i64",
    UInt8: "u8",
    UInt16: "u16",
    UInt32: "u32",
    UInt64: "u64",
    Float32: "f32",
    Float64: "f64",
    Boolean: "bool",
    Utf8: "str",
    List: "list",
    Date32: "date32",
    Date64: "date64",
    Time64Nanosecond: "time64_nanosecond",
    DurationNanosecond: "duration_nanosecond",
    DurationMillisecond: "duration_millisecond",
    Object: "object",
    Categorical: "categorical",
}


def dtype_to_primitive(dtype: Type[DataType]) -> Type[DataType]:
    #  TODO: add more
    if dtype == Date32:
        return Int32
    if dtype == Date64:
        return Int64
    ffi_name = DTYPE_TO_FFINAME[dtype]
    if "duration" in ffi_name:
        return Int64
    return dtype


def dtype_to_ctype(dtype: Type[DataType]) -> Type[_SimpleCData]:  # noqa: F821
    ptr_type: Type[_SimpleCData]
    if dtype == UInt8:
        ptr_type = ctypes.c_uint8
    elif dtype == UInt16:
        ptr_type = ctypes.c_uint16
    elif dtype == UInt32:
        ptr_type = ctypes.c_uint
    elif dtype == UInt64:
        ptr_type = ctypes.c_ulong
    elif dtype == Int8:
        ptr_type = ctypes.c_int8
    elif dtype == Int16:
        ptr_type = ctypes.c_int16
    elif dtype == Int32:
        ptr_type = ctypes.c_int
    elif dtype == Int64:
        ptr_type = ctypes.c_long
    elif dtype == Float32:
        ptr_type = ctypes.c_float
    elif dtype == Float64:
        ptr_type = ctypes.c_double
    elif dtype == Date32:
        ptr_type = ctypes.c_int
    elif dtype == Date64:
        ptr_type = ctypes.c_long
    else:
        raise NotImplementedError
    return ptr_type


def pytype_to_polars_type(data_type: Type[Any]) -> Type[DataType]:
    polars_type: Type[DataType]
    if data_type == int:
        polars_type = Int64
    elif data_type == str:
        polars_type = Utf8
    elif data_type == float:
        polars_type = Float64
    else:
        polars_type = data_type
    return polars_type


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
        Date32: PySeries.new_opt_i32,
        Date64: PySeries.new_opt_i32,
        Boolean: PySeries.new_opt_bool,
        Utf8: PySeries.new_str,
        Object: PySeries.new_object,
    }


def polars_type_to_constructor(
    dtype: Type[DataType],
) -> Callable[[str, Sequence[Any]], "PySeries"]:
    """
    Get the right PySeries constructor for the given Polars dtype.
    """
    try:
        return _POLARS_TYPE_TO_CONSTRUCTOR[dtype]
    except KeyError:
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


if not _DOCUMENTING:
    _PY_TYPE_TO_ARROW_TYPE = {
        float: pa.float64(),
        int: pa.int64(),
        str: pa.large_utf8(),
        bool: pa.bool_(),
    }


def py_type_to_arrow_type(dtype: Type[Any]) -> pa.lib.DataType:
    """
    Convert a Python dtype to an Arrow dtype.
    """
    try:
        return _PY_TYPE_TO_ARROW_TYPE[dtype]
    except KeyError:
        raise ValueError(f"Cannot parse dtype {dtype} into arrow dtype.")
