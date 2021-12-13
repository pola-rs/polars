import ctypes
from typing import Any, Dict, Type

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

from _ctypes import _SimpleCData


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


class Date(DataType):
    pass


class Datetime(DataType):
    pass


class Time(DataType):
    pass


class Object(DataType):
    pass


class Categorical(DataType):
    pass


_DTYPE_TO_FFINAME: Dict[Type[DataType], str] = {
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
    Date: "date",
    Datetime: "datetime",
    Time: "time",
    Object: "object",
    Categorical: "categorical",
}

_DTYPE_TO_CTYPE = {
    UInt8: ctypes.c_uint8,
    UInt16: ctypes.c_uint16,
    UInt32: ctypes.c_uint32,
    UInt64: ctypes.c_uint64,
    Int8: ctypes.c_int8,
    Int16: ctypes.c_int16,
    Int32: ctypes.c_int32,
    Date: ctypes.c_int32,
    Int64: ctypes.c_int64,
    Float32: ctypes.c_float,
    Float64: ctypes.c_double,
    Datetime: ctypes.c_int64,
    Time: ctypes.c_int64,
}


_PY_TYPE_TO_DTYPE = {
    float: Float64,
    int: Int64,
    str: Utf8,
    bool: Boolean,
}


_DTYPE_TO_PY_TYPE = {
    Float64: float,
    Float32: float,
    Int64: int,
    Int32: int,
    Int16: int,
    Int8: int,
    Utf8: str,
    UInt8: int,
    UInt16: int,
    UInt32: int,
    UInt64: int,
    Boolean: bool,
}

if _PYARROW_AVAILABLE:
    _PY_TYPE_TO_ARROW_TYPE = {
        float: pa.float64(),
        int: pa.int64(),
        str: pa.large_utf8(),
        bool: pa.bool_(),
    }


def dtype_to_ctype(dtype: Type[DataType]) -> Type[_SimpleCData]:
    try:
        return _DTYPE_TO_CTYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError


def dtype_to_ffiname(dtype: Type[DataType]) -> str:
    try:
        return _DTYPE_TO_FFINAME[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError


def dtype_to_py_type(dtype: Type[DataType]) -> Type:
    try:
        return _DTYPE_TO_PY_TYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError


def py_type_to_dtype(data_type: Type[Any]) -> Type[DataType]:
    # when the passed in is already a Polars datatype, return that
    if issubclass(data_type, DataType):
        return data_type

    try:
        return _PY_TYPE_TO_DTYPE[data_type]
    except KeyError:  # pragma: no cover
        raise NotImplementedError


def py_type_to_arrow_type(dtype: Type[Any]) -> "pa.lib.DataType":
    """
    Convert a Python dtype to an Arrow dtype.
    """
    try:
        return _PY_TYPE_TO_ARROW_TYPE[dtype]
    except KeyError:  # pragma: no cover
        raise ValueError(f"Cannot parse dtype {dtype} into Arrow dtype.")


def maybe_cast(el: Type[DataType], dtype: Type) -> Type[DataType]:
    # cast el if it doesn't match
    py_type = dtype_to_py_type(dtype)
    if not isinstance(el, py_type):
        el = py_type(el)
    return el
