import ctypes
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence, Type

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

from _ctypes import _SimpleCData  # type: ignore

try:
    from polars.polars import dtype_str_repr

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


class DataType:
    """Base class for all Polars data types"""

    @classmethod
    def string_repr(cls) -> str:
        return dtype_str_repr(cls)

    def __repr__(self) -> str:
        return dtype_str_repr(self)


class Int8(DataType):
    """8-bit signed integer type"""

    pass


class Int16(DataType):
    """16-bit signed integer type"""

    pass


class Int32(DataType):
    """32-bit signed integer type"""

    pass


class Int64(DataType):
    """64-bit signed integer type"""

    pass


class UInt8(DataType):
    """8-bit unsigned integer type"""

    pass


class UInt16(DataType):
    """16-bit unsigned integer type"""

    pass


class UInt32(DataType):
    """32-bit unsigned integer type"""

    pass


class UInt64(DataType):
    """64-bit unsigned integer type"""

    pass


class Float32(DataType):
    """32-bit floating point type"""

    pass


class Float64(DataType):
    """64-bit floating point type"""

    pass


class Boolean(DataType):
    """Boolean type"""

    pass


class Utf8(DataType):
    """UTF-8 encoded string type"""

    pass


class Null(DataType):
    """Type representing Null / None values"""

    pass


class List(DataType):
    def __init__(self, inner: Type[DataType]):
        """
        Nested list/array type

        Parameters
        ----------
        inner
            The `DataType` of values within the list
        """
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
    """Calendar date type"""

    pass


class Datetime(DataType):
    """Calendar date and time type"""

    def __init__(self, time_unit: str = "us", time_zone: Optional[str] = None):
        """
        Calendar date and time type

        Parameters
        ----------
        time_unit
            Any of {'ns', 'us', 'ms'}
        time_zone
            Timezone string as defined in pytz
        """
        self.tu = time_unit
        self.tz = time_zone

    def __eq__(self, other: Type[DataType]) -> bool:  # type: ignore
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, Datetime):
            return True
        if isinstance(other, Datetime):
            return self.tu == other.tu and self.tz == other.tz
        else:
            return False

    def __hash__(self) -> int:
        return hash(Datetime)


class Duration(DataType):
    """Time duration/delta type"""

    def __init__(self, time_unit: str = "us"):
        """
        Time duration/delta type

        Parameters
        ----------
        time_unit
            Any of {'ns', 'us', 'ms'}
        """
        self.tu = time_unit

    def __eq__(self, other: Type[DataType]) -> bool:  # type: ignore
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, Duration):
            return True
        if isinstance(other, Duration):
            return self.tu == other.tu
        else:
            return False

    def __hash__(self) -> int:
        return hash(Duration)


class Time(DataType):
    """Time of day type"""

    pass


class Object(DataType):
    """Type for wrapping arbitrary Python objects"""

    pass


class Categorical(DataType):
    """A categorical encoding of a set of strings"""

    pass


class Field:
    def __init__(self, name: str, dtype: Type[DataType]):
        """
        Definition of a single field within a `Struct` DataType

        Parameters
        ----------
        name
            The name of the field within its parent `Struct`
        dtype
            The `DataType` of the field's values
        """
        self.name = name
        self.dtype = py_type_to_dtype(dtype)

    def __eq__(self, other: "Field") -> bool:  # type: ignore
        return (self.name == other.name) & (self.dtype == other.dtype)

    def __repr__(self) -> str:
        if isinstance(self.dtype, type):
            dtype_str = self.dtype.string_repr()
        else:
            dtype_str = repr(self.dtype)
        return f'Field("{self.name}": {dtype_str})'


class Struct(DataType):
    def __init__(self, fields: Sequence[Field]):
        """
        Struct composite type

        Parameters
        ----------
        fields
            The sequence of fields that make up the struct
        """
        self.fields = fields

    def __eq__(self, other: Type[DataType]) -> bool:  # type: ignore
        # The comparison allows comparing objects to classes
        # and specific inner types to none specific.
        # if one of the arguments is not specific about its inner type
        # we infer it as being equal.
        # See the list type for more info
        if type(other) is type and issubclass(other, Struct):
            return True
        if isinstance(other, Struct):
            if self.fields is None or other.fields is None:
                return True
            else:
                return self.fields == other.fields
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


#: Mapping of `~polars.DataFrame` / `~polars.LazyFrame` column names to their `DataType`
Schema = Dict[str, Type[DataType]]
