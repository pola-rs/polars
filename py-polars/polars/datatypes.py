from __future__ import annotations

import ctypes
import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ForwardRef,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing import List as ListType

from polars.dependencies import numpy as np
from polars.dependencies import pyarrow as pa

try:
    from polars.polars import dtype_str_repr
    from polars.polars import get_idx_type as _get_idx_type

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True


if sys.version_info >= (3, 8):
    from typing import Literal, get_args
else:
    from typing_extensions import Literal

    # pass-through (only impact is that under 3.7 we'll end-up doing
    # standard inference for dataclass fields with an option/union)
    def get_args(tp: Any) -> Any:
        return tp


OptionType = type(Optional[type])
if sys.version_info >= (3, 10):
    from types import NoneType, UnionType
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

    # infer equivalent class
    NoneType = type(None)
    UnionType = type(Union[int, float])


if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


# number of rows to scan by default when inferring datatypes
N_INFER_DEFAULT = 100


# note: defined this way as some types can have instances that
# act as specialisations (eg: "List" and "List[Int32]")
PolarsDataType: TypeAlias = Union["DataTypeClass", "DataType"]
PolarsTemporalType: TypeAlias = Union[Type["TemporalType"], "TemporalType"]

PythonDataType: TypeAlias = Union[
    Type[int],
    Type[float],
    Type[bool],
    Type[str],
    Type[date],
    Type[time],
    Type[datetime],
    Type[timedelta],
    Type[ListType[Any]],
    Type[Tuple[Any, ...]],
    Type[bytes],
    Type[Decimal],
]

SchemaDefinition: TypeAlias = Union[
    Sequence[str],
    Mapping[str, Union[PolarsDataType, PythonDataType]],
    Sequence[Union[str, Tuple[str, Union[PolarsDataType, PythonDataType, None]]]],
]
SchemaDict: TypeAlias = Mapping[str, PolarsDataType]

DTYPE_TEMPORAL_UNITS: frozenset[TimeUnit] = frozenset(["ns", "us", "ms"])


def get_idx_type() -> PolarsDataType:
    """
    Get the datatype used for Polars indexing.

    Returns
    -------
    UInt32 in regular Polars, UInt64 in bigidx Polars.

    """
    return _get_idx_type()


def _custom_reconstruct(
    cls: type[Any], base: type[Any], state: Any
) -> PolarsDataType | type:
    """Helper function for unpickling DataType objects."""
    if state:
        obj = base.__new__(cls, state)
        if base.__init__ != object.__init__:
            base.__init__(obj, state)
    else:
        obj = object.__new__(cls)
    return obj


class DataTypeClass(type):
    """Metaclass for nicely printing DataType classes."""

    def __repr__(cls) -> str:
        return cls.__name__

    def _string_repr(cls) -> str:
        return dtype_str_repr(cls)


class DataType(metaclass=DataTypeClass):
    """Base class for all Polars data types."""

    def __new__(cls, *args: Any, **kwargs: Any) -> PolarsDataType:  # type: ignore[misc]
        # this formulation allows for equivalent use of "pl.Type" and "pl.Type()", while
        # still respecting types that take initialisation params (eg: Duration/Datetime)
        if args or kwargs:
            return super().__new__(cls)
        return cls

    def __reduce__(self) -> Any:
        return (_custom_reconstruct, (type(self), object, None), self.__dict__)

    def _string_repr(self) -> str:
        return dtype_str_repr(self)


class NumericType(DataType):
    """Base class for numeric data types."""


class IntegralType(NumericType):
    """Base class for integral data types."""


class FractionalType(NumericType):
    """Base class for fractional data types."""


class TemporalType(DataType):
    """Base class for temporal data types."""


class NestedType(DataType):
    """Base class for nested data types."""


class Int8(IntegralType):
    """8-bit signed integer type."""


class Int16(IntegralType):
    """16-bit signed integer type."""


class Int32(IntegralType):
    """32-bit signed integer type."""


class Int64(IntegralType):
    """64-bit signed integer type."""


class UInt8(IntegralType):
    """8-bit unsigned integer type."""


class UInt16(IntegralType):
    """16-bit unsigned integer type."""


class UInt32(IntegralType):
    """32-bit unsigned integer type."""


class UInt64(IntegralType):
    """64-bit unsigned integer type."""


class Float32(FractionalType):
    """32-bit floating point type."""


class Float64(FractionalType):
    """64-bit floating point type."""


class Boolean(DataType):
    """Boolean type."""


class Utf8(DataType):
    """UTF-8 encoded string type."""


class List(NestedType):
    inner: PolarsDataType | None = None

    def __init__(self, inner: PolarsDataType | PythonDataType):
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
        if type(other) is DataTypeClass and issubclass(other, List):
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

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r})"


class Date(TemporalType):
    """Calendar date type."""


class Datetime(TemporalType):
    """Calendar date and time type."""

    tu: TimeUnit | None = None
    tz: str | None = None

    def __init__(self, time_unit: TimeUnit | None = "us", time_zone: str | None = None):
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
        if type(other) is DataTypeClass and issubclass(other, Datetime):
            return True
        elif isinstance(other, Datetime):
            return self.tu == other.tu and self.tz == other.tz
        else:
            return False

    def __hash__(self) -> int:
        return hash((Datetime, self.tu))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(tu={self.tu!r}, tz={self.tz!r})"


class Duration(TemporalType):
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
        if type(other) is DataTypeClass and issubclass(other, Duration):
            return True
        elif isinstance(other, Duration):
            return self.tu == other.tu
        else:
            return False

    def __hash__(self) -> int:
        return hash((Duration, self.tu))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(tu={self.tu!r})"


class Time(TemporalType):
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
        class_name = self.__class__.__name__
        return f"{class_name}({self.name!r}, {self.dtype})"


class Struct(NestedType):
    def __init__(self, fields: Sequence[Field] | SchemaDict):
        """
        Struct composite type.

        Parameters
        ----------
        fields
            The sequence of fields that make up the struct

        """
        if isinstance(fields, Mapping):
            self.fields = [Field(name, dtype) for name, dtype in fields.items()]
        else:
            self.fields = list(fields)

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

    def __iter__(self) -> Iterator[tuple[str, PolarsDataType]]:
        for fld in self.fields or []:
            yield fld.name, fld.dtype

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.fields})"

    def to_schema(self) -> SchemaDict | None:
        """Return Struct dtype as a schema dict."""
        return dict(self)


class Binary(DataType):
    """Binary type."""


class Null(DataType):
    """Type representing Null / None values."""


class Unknown(DataType):
    """Type representing Datatype values that could not be determined statically."""


TEMPORAL_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        Datetime("ms"),
        Datetime("us"),
        Datetime("ns"),
        Date,
        Time,
        Duration("ms"),
        Duration("us"),
        Duration("ns"),
    ]
)
DATETIME_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        # TODO: ideally need a mechanism to wildcard timezones here too
        Datetime("ms"),
        Datetime("us"),
        Datetime("ns"),
    ]
)
DURATION_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        Duration("ms"),
        Duration("us"),
        Duration("ns"),
    ]
)
INTEGER_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
    ]
)
FLOAT_DTYPES: frozenset[PolarsDataType] = frozenset([Float32, Float64])
NUMERIC_DTYPES: frozenset[PolarsDataType] = FLOAT_DTYPES | INTEGER_DTYPES


T = TypeVar("T")


def cache(func: Callable[..., T]) -> T:
    # need this to satisfy mypy issue with "@property/@cache combination"
    # See: https://github.com/python/mypy/issues/5858
    return functools.lru_cache()(func)  # type: ignore[return-value]


class _DataTypeMappings:
    @property
    @cache
    def DTYPE_TO_FFINAME(self) -> dict[PolarsDataType, str]:
        return {
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

    @property
    @cache
    def DTYPE_TO_CTYPE(self) -> dict[PolarsDataType, Any]:
        return {
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

    @property
    @cache
    def PY_TYPE_TO_DTYPE(self) -> dict[PythonDataType | type[object], PolarsDataType]:
        return {
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

    @property
    @cache
    def PY_STR_TO_DTYPE(self) -> SchemaDict:
        return {str(tp.__name__): dtype for tp, dtype in self.PY_TYPE_TO_DTYPE.items()}

    @property
    @cache
    def DTYPE_TO_PY_TYPE(self) -> dict[PolarsDataType, PythonDataType]:
        return {
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
            # don't really know what type could be used in python
            Null: None,  # type: ignore[dict-item]
        }

    @property
    @cache
    def NUMPY_KIND_AND_ITEMSIZE_TO_DTYPE(self) -> dict[tuple[str, int], PolarsDataType]:
        return {
            # (np.dtype().kind, np.dtype().itemsize)
            ("i", 1): Int8,
            ("i", 2): Int16,
            ("i", 4): Int32,
            ("i", 8): Int64,
            ("u", 1): UInt8,
            ("u", 2): UInt16,
            ("u", 4): UInt32,
            ("u", 8): UInt64,
            ("f", 4): Float32,
            ("f", 8): Float64,
        }

    @property
    @cache
    def PY_TYPE_TO_ARROW_TYPE(self) -> dict[PythonDataType, pa.lib.DataType]:
        return {
            float: pa.float64(),
            int: pa.int64(),
            str: pa.large_utf8(),
            bool: pa.bool_(),
            date: pa.date32(),
            time: pa.time64("us"),
            datetime: pa.timestamp("us"),
            timedelta: pa.duration("us"),
        }

    @property
    @cache
    def DTYPE_TO_ARROW_TYPE(self) -> dict[PolarsDataType, pa.lib.DataType]:
        return {
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
        }


# initialise once (poor man's singleton :)
DataTypeMappings = _DataTypeMappings()


def _base_type(dtype: PolarsDataType) -> DataTypeClass:
    """Ensure return of the DataType base dtype/class."""
    if isinstance(dtype, DataType):
        return type(dtype)
    return dtype


def dtype_to_ctype(dtype: PolarsDataType) -> Any:
    """Convert a Polars dtype to a ctype."""
    try:
        dtype = _base_type(dtype)
        return DataTypeMappings.DTYPE_TO_CTYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to C-type not implemented."
        ) from None


def dtype_to_ffiname(dtype: PolarsDataType) -> str:
    """Return FFI function name associated with the given Polars dtype."""
    try:
        dtype = _base_type(dtype)
        return DataTypeMappings.DTYPE_TO_FFINAME[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to FFI not implemented."
        ) from None


def dtype_to_py_type(dtype: PolarsDataType) -> PythonDataType:
    """Convert a Polars dtype to a Python dtype."""
    try:
        dtype = _base_type(dtype)
        return DataTypeMappings.DTYPE_TO_PY_TYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to Python type not implemented."
        ) from None


def is_polars_dtype(data_type: Any, include_unknown: bool = False) -> bool:
    """Indicate whether the given input is a Polars dtype, or dtype specialisation."""
    try:
        if data_type == Unknown:
            # does not represent a realisable dtype, so ignore by default
            return include_unknown
        else:
            return isinstance(data_type, (DataType, DataTypeClass))
    except ValueError:
        return False


@overload
def py_type_to_dtype(
    data_type: Any, raise_unmatched: Literal[True] = True
) -> PolarsDataType:
    ...


@overload
def py_type_to_dtype(
    data_type: Any, raise_unmatched: Literal[False]
) -> PolarsDataType | None:
    ...


def py_type_to_dtype(
    data_type: Any, raise_unmatched: bool = True
) -> PolarsDataType | None:
    """Convert a Python dtype (or type annotation) to a Polars dtype."""
    if isinstance(data_type, ForwardRef):
        annotation = data_type.__forward_arg__
        data_type = (
            DataTypeMappings.PY_STR_TO_DTYPE.get(
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
        return DataTypeMappings.PY_TYPE_TO_DTYPE[data_type]
    except (KeyError, TypeError):  # pragma: no cover
        if not raise_unmatched:
            return None
        raise ValueError(
            f"Cannot infer dtype from '{data_type}' (type: {type(data_type).__name__})"
        ) from None


def py_type_to_arrow_type(dtype: PythonDataType) -> pa.lib.DataType:
    """Convert a Python dtype to an Arrow dtype."""
    try:
        return DataTypeMappings.PY_TYPE_TO_ARROW_TYPE[dtype]
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

        arrow_type = DataTypeMappings.DTYPE_TO_ARROW_TYPE[dtype]
        if tz:
            arrow_type = pa.timestamp(dtype.tu or "us", tz)  # type: ignore[union-attr]
        return arrow_type
    except KeyError:  # pragma: no cover
        raise ValueError(
            f"Cannot parse data type {dtype} into Arrow data type."
        ) from None


def supported_numpy_char_code(dtype_char: str) -> bool:
    dtype = np.dtype(dtype_char)
    return (
        dtype.kind,
        dtype.itemsize,
    ) in DataTypeMappings.NUMPY_KIND_AND_ITEMSIZE_TO_DTYPE


def numpy_char_code_to_dtype(dtype_char: str) -> PolarsDataType:
    """Convert a numpy character dtype to a Polars dtype."""
    dtype = np.dtype(dtype_char)

    try:
        return DataTypeMappings.NUMPY_KIND_AND_ITEMSIZE_TO_DTYPE[
            (dtype.kind, dtype.itemsize)
        ]
    except KeyError:  # pragma: no cover
        raise ValueError(
            f"Cannot parse numpy data type {dtype} into Polars data type."
        ) from None


def maybe_cast(
    el: Any, dtype: PolarsDataType, time_unit: TimeUnit | None = None
) -> Any:
    """Try casting a value to a value that is valid for the given Polars dtype."""
    # cast el if it doesn't match
    from polars.utils import _datetime_to_pl_timestamp, _timedelta_to_pl_timedelta

    if isinstance(el, datetime):
        return _datetime_to_pl_timestamp(el, time_unit)
    elif isinstance(el, timedelta):
        return _timedelta_to_pl_timedelta(el, time_unit)

    py_type = dtype_to_py_type(dtype)
    if not isinstance(el, py_type):
        try:
            el = py_type(el)  # type: ignore[call-arg]
        except Exception:
            raise ValueError(
                f"Cannot convert Python type {type(el)} to {dtype}"
            ) from None
    return el
