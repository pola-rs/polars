import ctypes
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence, Type

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

from _ctypes import _SimpleCData  # type: ignore


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


class Null(DataType):
    pass


class List(DataType):
    def __init__(self, inner: Type[DataType]):
        self.inner = py_type_to_dtype(inner)

    def __eq__(self, other: Type[DataType]) -> bool:  # type: ignore
        # The comparison allows comparing objects to classes
        # and specific inner types to none specific.
        # if one of the arguments is not specific about its inner type
        # we infer it as being equal.
        # List[i64] == List[i64] == True
        # List[i64] == List == True
        # List[i64] == List[None] == True
        # List[i64] == List[f32] == False

        # allow comparing object instances to class
        if type(other) is type and issubclass(other, List):
            return True
        if isinstance(other, List):
            if self.inner is None or other.inner is None:
                return True
            else:
                return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash(List)


class Date(DataType):
    pass


class Datetime(DataType):
    pass


class Duration(DataType):
    pass


class Time(DataType):
    pass


class Object(DataType):
    pass


class Categorical(DataType):
    pass


class Struct(DataType):
    def __init__(self, inner_types: Sequence[Type[DataType]]):
        self.inner_types = [py_type_to_dtype(dt) for dt in inner_types]

    def __eq__(self, other: Type[DataType]) -> bool:  # type: ignore
        # The comparison allows comparing objects to classes
        # and specific inner types to none specific.
        # if one of the arguments is not specific about its inner type
        # we infer it as being equal.
        # See the list type for more info
        if type(other) is type and issubclass(other, Struct):
            return True
        if isinstance(other, Struct):
            if self.inner_types is None or other.inner_types is None:
                return True
            else:
                return self.inner_types == other.inner_types
        else:
            return False

    def __hash__(self) -> int:
        return hash(Struct)


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
    Duration: "duration",
    Time: "time",
    Object: "object",
    Categorical: "categorical",
    Struct: "struct",
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
    Duration: ctypes.c_int64,
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
    if (
        type(data_type) is type
        and issubclass(data_type, DataType)
        or isinstance(data_type, DataType)
    ):
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


def maybe_cast(
    el: Type[DataType], dtype: Type, time_unit: Optional[str] = None
) -> Type[DataType]:
    # cast el if it doesn't match
    from polars.utils import _datetime_to_pl_timestamp, _timedelta_to_pl_timedelta

    if isinstance(el, datetime):
        return _datetime_to_pl_timestamp(el, time_unit)
    elif isinstance(el, timedelta):
        return _timedelta_to_pl_timedelta(el, time_unit)
    py_type = dtype_to_py_type(dtype)
    if not isinstance(el, py_type):
        el = py_type(el)
    return el
