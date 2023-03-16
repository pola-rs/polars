from __future__ import annotations

import ctypes
import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ForwardRef,
    Optional,
    TypeVar,
    Union,
    overload,
)

from polars.datatypes import (
    Binary,
    Boolean,
    Categorical,
    DataType,
    DataTypeClass,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    Object,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Unknown,
    Utf8,
)
from polars.dependencies import numpy as np
from polars.dependencies import pyarrow as pa

if sys.version_info >= (3, 8):
    from typing import get_args
else:
    # pass-through (only impact is that under 3.7 we'll end-up doing
    # standard inference for dataclass fields with an option/union)
    def get_args(tp: Any) -> Any:
        return tp


OptionType = type(Optional[type])
if sys.version_info >= (3, 10):
    from types import NoneType, UnionType
else:
    # infer equivalent class
    NoneType = type(None)
    UnionType = type(Union[int, float])

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType, PythonDataType, SchemaDict, TimeUnit

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


T = TypeVar("T")


def cache(function: Callable[..., T]) -> T:
    # need this to satisfy mypy issue with "@property/@cache combination"
    # See: https://github.com/python/mypy/issues/5858
    return functools.lru_cache()(function)  # type: ignore[return-value]


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
            PyDecimal: Decimal,
            bytes: Binary,
            object: Object,
            None.__class__: Null,
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
            Decimal: PyDecimal,
            Boolean: bool,
            Duration: timedelta,
            Datetime: datetime,
            Date: date,
            Time: time,
            Binary: bytes,
            List: list,
            Null: None.__class__,
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
            None.__class__: pa.null(),
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
            Null: pa.null(),
        }


# Initialize once (poor man's singleton :)
DataTypeMappings = _DataTypeMappings()


def dtype_to_ctype(dtype: PolarsDataType) -> Any:
    """Convert a Polars dtype to a ctype."""
    try:
        dtype = dtype.base_type()
        return DataTypeMappings.DTYPE_TO_CTYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to C-type not implemented."
        ) from None


def dtype_to_ffiname(dtype: PolarsDataType) -> str:
    """Return FFI function name associated with the given Polars dtype."""
    try:
        dtype = dtype.base_type()
        return DataTypeMappings.DTYPE_TO_FFINAME[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to FFI not implemented."
        ) from None


def dtype_to_py_type(dtype: PolarsDataType) -> PythonDataType:
    """Convert a Polars dtype to a Python dtype."""
    try:
        dtype = dtype.base_type()
        return DataTypeMappings.DTYPE_TO_PY_TYPE[dtype]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Conversion of polars data type {dtype} to Python type not implemented."
        ) from None


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
    """Check if the input can be mapped to a Polars dtype."""
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
    from polars.utils.convert import (
        _datetime_to_pl_timestamp,
        _timedelta_to_pl_timedelta,
    )

    if isinstance(el, datetime):
        return _datetime_to_pl_timestamp(el, time_unit)
    elif isinstance(el, timedelta):
        return _timedelta_to_pl_timedelta(el, time_unit)

    py_type = dtype_to_py_type(dtype)
    if not isinstance(el, py_type):
        try:
            el = py_type(el)  # type: ignore[call-arg, misc]
        except Exception:
            raise ValueError(
                f"Cannot convert Python type {type(el)} to {dtype}"
            ) from None
    return el
