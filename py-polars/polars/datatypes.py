from __future__ import annotations

import ctypes
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    ForwardRef,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

from _ctypes import _SimpleCData  # type: ignore[import]

try:
    from polars.polars import dtype_str_repr
    from polars.polars import get_idx_type as _get_idx_type

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

UnionType: type
OptionType = type(Optional[type])

if sys.version_info >= (3, 10):
    from types import UnionType
else:
    # infer equivalent class
    UnionType = type(Union[int, float])

if sys.version_info >= (3, 8):
    from typing import get_args
else:

    # pass-through (only impact is that under 3.7 we'll end-up doing
    # standard inference for dataclass fields with an option/union)
    def get_args(tp: Any) -> Any:
        return tp


if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


def get_idx_type() -> PolarsDataType:
    """
    Get the datatype used for polars Indexing.

    This is UInt32 in regulars polars and UInt64 in polars_u64_idx

    """
    return _get_idx_type()


def _custom_reconstruct(cls: type[Any], base: type[Any], state: Any) -> PolarsDataType:
    if state:
        obj = base.__new__(cls, state)
        if base.__init__ != object.__init__:
            base.__init__(obj, state)
    else:
        obj = object.__new__(cls)
    return obj


class DataType:
    """Base class for all Polars data types."""

    def __new__(cls, *args: Any, **kwargs: Any) -> PolarsDataType:  # type: ignore[misc]
        # this formulation allows for equivalent use of "pl.Type" and "pl.Type()", while
        # still respecting types that take initialisation params (eg: Duration/Datetime)
        if args or kwargs:
            return super().__new__(cls)
        return cls

    def __reduce__(self) -> Any:
        return (_custom_reconstruct, (type(self), object, None), self.__dict__)

    @classmethod
    def string_repr(cls) -> str:
        return dtype_str_repr(cls)

    def __repr__(self) -> str:
        return dtype_str_repr(self)


# note: defined this way as some types can have instances that
# act as specialisations (eg: "List" and "List[Int32]")
PolarsDataType = Union[Type[DataType], DataType]

ColumnsType = Union[
    Sequence[str],
    Mapping[str, PolarsDataType],
    Sequence[Tuple[str, Optional[PolarsDataType]]],
]
NoneType = type(None)


class Int8(DataType):
    """8-bit signed integer type."""


class Int16(DataType):
    """16-bit signed integer type."""


class Int32(DataType):
    """32-bit signed integer type."""


class Int64(DataType):
    """64-bit signed integer type."""


class UInt8(DataType):
    """8-bit unsigned integer type."""


class UInt16(DataType):
    """16-bit unsigned integer type."""


class UInt32(DataType):
    """32-bit unsigned integer type."""


class UInt64(DataType):
    """64-bit unsigned integer type."""


class Float32(DataType):
    """32-bit floating point type."""


class Float64(DataType):
    """64-bit floating point type."""


class Boolean(DataType):
    """Boolean type."""


class Utf8(DataType):
    """UTF-8 encoded string type."""


class Binary(DataType):
    """Binary type."""


class Null(DataType):
    """Type representing Null / None values."""


class Unknown(DataType):
    """Type representing Datatype values that could not be determined statically."""


class List(DataType):
    inner: PolarsDataType | None = None

    def __init__(self, inner: PolarsDataType):
        """
        Nested list/array type.

        Parameters
        ----------
        inner
            The `DataType` of values within the list

        """
        self.inner = py_type_to_dtype(inner)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
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
        return hash((List, self.inner))


class Date(DataType):
    """Calendar date type."""


class Datetime(DataType):
    """Calendar date and time type."""

    tu: TimeUnit | None = None
    tz: str | None = None

    def __init__(self, time_unit: TimeUnit = "us", time_zone: str | None = None):
        """
        Calendar date and time type.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms'}
            Time unit.
        time_zone
            Timezone string as defined in zoneinfo (run
            ``import zoneinfo; zoneinfo.available_timezones()`` for a full list).

        """
        self.tu = time_unit or "us"
        self.tz = time_zone

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, Datetime):
            return True
        elif isinstance(other, Datetime):
            return self.tu == other.tu and self.tz == other.tz
        else:
            return False

    def __hash__(self) -> int:
        return hash((Datetime, self.tu))


class Duration(DataType):
    """Time duration/delta type."""

    tu: TimeUnit | None = None

    def __init__(self, time_unit: TimeUnit = "us"):
        """
        Time duration/delta type.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms'}
            Time unit.

        """
        self.tu = time_unit

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, Duration):
            return True
        elif isinstance(other, Duration):
            return self.tu == other.tu
        else:
            return False

    def __hash__(self) -> int:
        return hash((Duration, self.tu))


class Time(DataType):
    """Time of day type."""


class Object(DataType):
    """Type for wrapping arbitrary Python objects."""


class Categorical(DataType):
    """A categorical encoding of a set of strings."""


class Field:
    def __init__(self, name: str, dtype: PolarsDataType):
        """
        Definition of a single field within a `Struct` DataType.

        Parameters
        ----------
        name
            The name of the field within its parent `Struct`
        dtype
            The `DataType` of the field's values

        """
        self.name = name
        self.dtype = py_type_to_dtype(dtype)

    def __eq__(self, other: Field) -> bool:  # type: ignore[override]
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
        Struct composite type.

        Parameters
        ----------
        fields
            The sequence of fields that make up the struct

        """
        self.fields = fields

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # The comparison allows comparing objects to classes, and specific
        # inner types to those without (eg: inner=None). if one of the
        # arguments is not specific about its inner type we infer it
        # as being equal. (See the List type for more info).
        if isclass(other) and issubclass(other, Struct):
            return True
        elif isinstance(other, Struct):
            return any((f is None) for f in (self.fields, other.fields)) or (
                self.fields == other.fields
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(Struct)


TemporalDataType = Union[Type[Datetime], Datetime, Type[Date], Date, Type[Time], Time]

DTYPE_TEMPORAL_UNITS: frozenset[TimeUnit] = frozenset(["ns", "us", "ms"])


_DTYPE_TO_FFINAME: dict[PolarsDataType, str] = {
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
    Binary: "binary",
}


_DTYPE_TO_CTYPE: dict[PolarsDataType, Any] = {
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


_PY_TYPE_TO_DTYPE: dict[type, PolarsDataType] = {
    float: Float64,
    int: Int64,
    str: Utf8,
    bool: Boolean,
    date: Date,
    datetime: Datetime("us"),
    timedelta: Duration("us"),
    time: Time,
    list: List,
    tuple: List,
    Decimal: Float64,
    bytes: Binary,
    object: Object,
}

_PY_STR_TO_DTYPE: dict[str, PolarsDataType] = {
    str(tp.__name__): dtype for tp, dtype in _PY_TYPE_TO_DTYPE.items()
}

_DTYPE_TO_PY_TYPE: dict[PolarsDataType, type] = {
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
    Duration: timedelta,
    Datetime: datetime,
    Date: date,
    Time: time,
    Binary: bytes,
    List: list,
}

# Map Numpy char codes to polars dtypes.
#
# Windows behaves differently from other platforms as C long is
# only 32-bit on Windows, while it is 64-bit on other platforms.
# See: https://numpy.org/doc/stable/reference/arrays.scalars.html
_NUMPY_CHAR_CODE_TO_DTYPE = {
    "b": Int8,
    "h": Int16,
    "i": Int32,
    ("q" if sys.platform == "win32" else "l"): Int64,
    "B": UInt8,
    "H": UInt16,
    "I": UInt32,
    ("Q" if sys.platform == "win32" else "L"): UInt64,
    "f": Float32,
    "d": Float64,
    "?": Boolean,
}

if _PYARROW_AVAILABLE:
    _PY_TYPE_TO_ARROW_TYPE: dict[type, pa.lib.DataType] = {
        float: pa.float64(),
        int: pa.int64(),
        str: pa.large_utf8(),
        bool: pa.bool_(),
        date: pa.date32(),
        time: pa.time64("us"),
        datetime: pa.timestamp("us"),
        timedelta: pa.duration("us"),
    }

    _DTYPE_TO_ARROW_TYPE = {
        Int8: pa.int8(),
        Int16: pa.int16(),
        Int32: pa.int32(),
        Int64: pa.int64(),
        UInt8: pa.uint8(),
        UInt16: pa.uint16(),
        UInt32: pa.uint32(),
        UInt64: pa.uint64(),
        Float32: pa.float32(),
        Float64: pa.float64(),
        Boolean: pa.bool_(),
        Utf8: pa.large_utf8(),
        Date: pa.date32(),
        Datetime: pa.timestamp("us"),
        Datetime("ms"): pa.timestamp("ms"),
        Datetime("us"): pa.timestamp("us"),
        Datetime("ns"): pa.timestamp("ns"),
        Duration: pa.duration("us"),
        Duration("ms"): pa.duration("ms"),
        Duration("us"): pa.duration("us"),
        Duration("ns"): pa.duration("ns"),
        Time: pa.time64("us"),
        # Time("ms"): pa.time32("ms"),
        # Time("us"): pa.time64("us"),
        # Time("ns"): pa.time64("ns"),
    }


def _base_type(dtype: PolarsDataType) -> type[DataType]:
    """Ensure return of the DataType base dtype/class."""
    if isinstance(dtype, DataType):
        return type(dtype)
    return dtype


def dtype_to_ctype(dtype: PolarsDataType) -> type[_SimpleCData]:
    """Convert a Polars dtype to a ctype."""
    try:
        dtype = _base_type(dtype)
        return _DTYPE_TO_CTYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to C-type not implemented."
        ) from None


def dtype_to_ffiname(dtype: PolarsDataType) -> str:
    """Return FFI function name associated with the given Polars dtype."""
    try:
        dtype = _base_type(dtype)
        return _DTYPE_TO_FFINAME[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to FFI not implemented."
        ) from None


def dtype_to_py_type(dtype: PolarsDataType) -> type:
    """Convert a Polars dtype to a Python dtype."""
    try:
        dtype = _base_type(dtype)
        return _DTYPE_TO_PY_TYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to Python type not implemented."
        ) from None


def is_polars_dtype(data_type: Any) -> bool:
    """Indicate whether the given input is a Polars dtype, or dtype specialisation."""
    return isinstance(data_type, DataType) or (
        type(data_type) is type and issubclass(data_type, DataType)
    )


def py_type_to_dtype(data_type: Any, raise_unmatched: bool = True) -> PolarsDataType:
    """Convert a Python dtype (or type annotation) to a Polars dtype."""
    if isinstance(data_type, ForwardRef):
        annotation = data_type.__forward_arg__
        data_type = (
            _PY_STR_TO_DTYPE.get(
                re.sub(r"(^None \|)|(\| None$)", "", annotation).strip(), data_type
            )
            if isinstance(annotation, str)  # type: ignore[redundant-expr]
            else annotation
        )

    if is_polars_dtype(data_type):
        return data_type

    elif isinstance(data_type, (OptionType, UnionType)):
        # not exhaustive; handles the common "type | None" case, but
        # should probably pick appropriate supertype when n_types > 1?
        possible_types = [tp for tp in get_args(data_type) if tp is not NoneType]
        if len(possible_types) == 1:
            data_type = possible_types[0]
    try:
        return _PY_TYPE_TO_DTYPE[data_type]
    except KeyError:  # pragma: no cover
        if not raise_unmatched:
            return None  # type: ignore[return-value]
        raise NotImplementedError(
            f"Conversion of Python data type {data_type} to Polars data type not"
            " implemented."
        ) from None


def py_type_to_arrow_type(dtype: type[Any]) -> pa.lib.DataType:
    """Convert a Python dtype to an Arrow dtype."""
    try:
        return _PY_TYPE_TO_ARROW_TYPE[dtype]
    except KeyError:  # pragma: no cover
        raise ValueError(
            f"Cannot parse Python data type {dtype} into Arrow data type."
        ) from None


def dtype_to_arrow_type(dtype: PolarsDataType) -> pa.lib.DataType:
    """Convert a Polars dtype to an Arrow dtype."""
    try:
        # special handling for mapping to tz-aware timestamp type.
        # (don't want to include every possible tz string in the lookup)
        tz = None
        if dtype == Datetime:
            dtype, tz = Datetime(dtype.tu), dtype.tz  # type: ignore[union-attr]

        arrow_type = _DTYPE_TO_ARROW_TYPE[dtype]
        if tz:
            arrow_type = pa.timestamp(dtype.tu or "us", tz)  # type: ignore[union-attr]
        return arrow_type
    except KeyError:  # pragma: no cover
        raise ValueError(
            f"Cannot parse data type {dtype} into Arrow data type."
        ) from None


def supported_numpy_char_code(dtype: str) -> bool:
    return dtype in _NUMPY_CHAR_CODE_TO_DTYPE


def numpy_char_code_to_dtype(dtype: str) -> PolarsDataType:
    """Convert a numpy character dtype to a Polars dtype."""
    try:
        return _NUMPY_CHAR_CODE_TO_DTYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Cannot parse numpy data type {dtype} into Polars data type."
        ) from None


def maybe_cast(
    el: PolarsDataType, dtype: type, time_unit: TimeUnit | None = None
) -> PolarsDataType:
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
Schema = Dict[str, PolarsDataType]
