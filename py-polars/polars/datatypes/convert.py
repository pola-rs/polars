from __future__ import annotations

import contextlib
import ctypes
import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from inspect import isclass
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

import polars.datatypes as pldt
from polars.dependencies import numpy as np
from polars.dependencies import pyarrow as pa

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import dtype_str_repr as _dtype_str_repr

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
    from polars.datatypes import DataType
    from polars.type_aliases import PolarsDataType, PythonDataType, TimeUnit

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


T = TypeVar("T")


def cache(function: Callable[..., T]) -> T:
    # need this to satisfy mypy issue with "@property/@cache combination"
    # See: https://github.com/python/mypy/issues/5858
    return functools.lru_cache()(function)  # type: ignore[return-value]


def _normalize_polars_dtype(data_type: PolarsDataType) -> DataType:
    if isclass(data_type) and issubclass(data_type, pldt.DataType):
        return data_type()
    else:
        return data_type  # type: ignore[return-value]


PY_STR_TO_DTYPE: dict[str, PolarsDataType] = {
    "float": pldt.Float64,
    "int": pldt.Int64,
    "str": pldt.Utf8,
    "bool": pldt.Boolean,
    "date": pldt.Date,
    "datetime": pldt.Datetime("us"),
    "timedelta": pldt.Duration("us"),
    "time": pldt.Time,
    "list": pldt.List,
    "tuple": pldt.List,
    "Decimal": pldt.Decimal,
    "bytes": pldt.Binary,
    "object": pldt.Object,
    "NoneType": pldt.Null,
}


@functools.lru_cache(16)
def map_py_type_to_dtype(python_dtype: PythonDataType | type[object]) -> PolarsDataType:
    if python_dtype is float:
        return pldt.Float64
    if python_dtype is int:
        return pldt.Int64
    if python_dtype is str:
        return pldt.Utf8
    if python_dtype is bool:
        return pldt.Boolean
    if issubclass(python_dtype, datetime):
        # `datetime` is a subclass of `date`,
        # so need to check `datetime` first
        return pldt.Datetime("us")
    if issubclass(python_dtype, date):
        return pldt.Date
    if python_dtype is timedelta:
        return pldt.Duration("us")
    if python_dtype is time:
        return pldt.Time
    if python_dtype is list:
        return pldt.List
    if python_dtype is tuple:
        return pldt.List
    if python_dtype is PyDecimal:
        return pldt.Decimal
    if python_dtype is bytes:
        return pldt.Binary
    if python_dtype is object:
        return pldt.Object
    if python_dtype is None.__class__:
        return pldt.Null
    raise TypeError("Invalid type")


def is_polars_dtype(data_type: Any, include_unknown: bool = False) -> bool:
    """Indicate whether the given input is a Polars dtype, or dtype specialisation."""
    try:
        if data_type == pldt.Unknown:
            # does not represent a realisable dtype, so ignore by default
            return include_unknown
        else:
            return isinstance(data_type, pldt.DataType) or (
                isclass(data_type) and issubclass(data_type, pldt.DataType)
            )
    except ValueError:
        return False


class _DataTypeMappings:
    @property
    @cache
    def DTYPE_TO_FFINAME(self) -> dict[PolarsDataType, str]:
        return {
            pldt.Int8: "i8",
            pldt.Int16: "i16",
            pldt.Int32: "i32",
            pldt.Int64: "i64",
            pldt.UInt8: "u8",
            pldt.UInt16: "u16",
            pldt.UInt32: "u32",
            pldt.UInt64: "u64",
            pldt.Float32: "f32",
            pldt.Float64: "f64",
            pldt.Boolean: "bool",
            pldt.Utf8: "str",
            pldt.List: "list",
            pldt.Date: "date",
            pldt.Datetime: "datetime",
            pldt.Duration: "duration",
            pldt.Time: "time",
            pldt.Object: "object",
            pldt.Categorical: "categorical",
            pldt.Struct: "struct",
            pldt.Binary: "binary",
        }

    @property
    @cache
    def DTYPE_TO_CTYPE(self) -> dict[PolarsDataType, Any]:
        return {
            pldt.UInt8: ctypes.c_uint8,
            pldt.UInt16: ctypes.c_uint16,
            pldt.UInt32: ctypes.c_uint32,
            pldt.UInt64: ctypes.c_uint64,
            pldt.Int8: ctypes.c_int8,
            pldt.Int16: ctypes.c_int16,
            pldt.Int32: ctypes.c_int32,
            pldt.Date: ctypes.c_int32,
            pldt.Int64: ctypes.c_int64,
            pldt.Float32: ctypes.c_float,
            pldt.Float64: ctypes.c_double,
            pldt.Datetime: ctypes.c_int64,
            pldt.Duration: ctypes.c_int64,
            pldt.Time: ctypes.c_int64,
        }

    @property
    @cache
    def DTYPE_TO_PY_TYPE(self) -> dict[PolarsDataType, PythonDataType]:
        return {
            pldt.Float64: float,
            pldt.Float32: float,
            pldt.Int64: int,
            pldt.Int32: int,
            pldt.Int16: int,
            pldt.Int8: int,
            pldt.Utf8: str,
            pldt.UInt8: int,
            pldt.UInt16: int,
            pldt.UInt32: int,
            pldt.UInt64: int,
            pldt.Decimal: PyDecimal,
            pldt.Boolean: bool,
            pldt.Duration: timedelta,
            pldt.Datetime: datetime,
            pldt.Date: date,
            pldt.Time: time,
            pldt.Binary: bytes,
            pldt.List: list,
            pldt.Null: None.__class__,
        }

    @property
    @cache
    def NUMPY_KIND_AND_ITEMSIZE_TO_DTYPE(self) -> dict[tuple[str, int], PolarsDataType]:
        return {
            # (np.dtype().kind, np.dtype().itemsize)
            ("i", 1): pldt.Int8,
            ("i", 2): pldt.Int16,
            ("i", 4): pldt.Int32,
            ("i", 8): pldt.Int64,
            ("u", 1): pldt.UInt8,
            ("u", 2): pldt.UInt16,
            ("u", 4): pldt.UInt32,
            ("u", 8): pldt.UInt64,
            ("f", 4): pldt.Float32,
            ("f", 8): pldt.Float64,
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
            pldt.Int8(): pa.int8(),
            pldt.Int16(): pa.int16(),
            pldt.Int32(): pa.int32(),
            pldt.Int64(): pa.int64(),
            pldt.UInt8(): pa.uint8(),
            pldt.UInt16(): pa.uint16(),
            pldt.UInt32(): pa.uint32(),
            pldt.UInt64(): pa.uint64(),
            pldt.Float32(): pa.float32(),
            pldt.Float64(): pa.float64(),
            pldt.Boolean(): pa.bool_(),
            pldt.Utf8(): pa.large_utf8(),
            pldt.Date(): pa.date32(),
            pldt.Datetime("ms"): pa.timestamp("ms"),
            pldt.Datetime("us"): pa.timestamp("us"),
            pldt.Datetime("ns"): pa.timestamp("ns"),
            pldt.Duration("ms"): pa.duration("ms"),
            pldt.Duration("us"): pa.duration("us"),
            pldt.Duration("ns"): pa.duration("ns"),
            pldt.Time(): pa.time64("us"),
            pldt.Null(): pa.null(),
        }

    @property
    @cache
    def REPR_TO_DTYPE(self) -> dict[str, PolarsDataType]:
        def _dtype_str_repr_safe(o: Any) -> PolarsDataType | None:
            try:
                return _dtype_str_repr(o.base_type()).split("[")[0]
            except ValueError:
                return None

        return {
            _dtype_str_repr_safe(obj): obj  # type: ignore[misc]
            for obj in pldt.POLARS_DTYPES
            if _dtype_str_repr_safe(obj) is not None
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
def py_type_to_dtype(data_type: Any, raise_unmatched: Literal[True] = True) -> DataType:
    ...


@overload
def py_type_to_dtype(
    data_type: Any, raise_unmatched: Literal[False]
) -> DataType | None:
    ...


def py_type_to_dtype(data_type: Any, raise_unmatched: bool = True) -> DataType | None:
    """Convert a Python dtype (or type annotation) to a Polars dtype."""
    if isinstance(data_type, ForwardRef):
        annotation = data_type.__forward_arg__
        data_type = (
            PY_STR_TO_DTYPE.get(
                re.sub(r"(^None \|)|(\| None$)", "", annotation).strip(), data_type
            )
            if isinstance(annotation, str)  # type: ignore[redundant-expr]
            else annotation
        )

    if is_polars_dtype(data_type):
        return _normalize_polars_dtype(data_type)

    elif isinstance(data_type, (OptionType, UnionType)):
        # not exhaustive; handles the common "type | None" case, but
        # should probably pick appropriate supertype when n_types > 1?
        possible_types = [tp for tp in get_args(data_type) if tp is not NoneType]
        if len(possible_types) == 1:
            data_type = possible_types[0]

    elif isinstance(data_type, str):
        data_type = DataTypeMappings.REPR_TO_DTYPE.get(data_type, data_type)
        if is_polars_dtype(data_type):
            return _normalize_polars_dtype(data_type)
    try:
        dtype = map_py_type_to_dtype(data_type)
        return _normalize_polars_dtype(dtype)
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
    dtype = _normalize_polars_dtype(dtype)
    try:
        # special handling for mapping to tz-aware timestamp type.
        # (don't want to include every possible tz string in the lookup)
        if isinstance(dtype, pldt.Datetime):
            dtype, time_zone = pldt.Datetime(dtype.time_unit), dtype.time_zone
            arrow_type = DataTypeMappings.DTYPE_TO_ARROW_TYPE[dtype]
            if time_zone is not None:
                arrow_type = pa.timestamp(dtype.time_unit, time_zone)
        else:
            arrow_type = DataTypeMappings.DTYPE_TO_ARROW_TYPE[dtype]
        return arrow_type
    except KeyError:  # pragma: no cover
        raise ValueError(
            f"Cannot parse data type {dtype} into Arrow data type."
        ) from None


def dtype_short_repr_to_dtype(dtype_string: str | None) -> PolarsDataType | None:
    """Map a PolarsDataType short repr (eg: 'i64', 'list[str]') back into a dtype."""
    if dtype_string is None:
        return None
    m = re.match(r"^(\w+)(?:\[(.+)\])?$", dtype_string)
    if m is None:
        return None

    dtype_base, subtype = m.groups()
    dtype = DataTypeMappings.REPR_TO_DTYPE.get(dtype_base)
    if dtype and subtype:
        # TODO: better-handle nested types (such as List,Struct)
        subtype = (s.strip("""'" """) for s in subtype.replace("μs", "us").split(","))
        try:
            return dtype(*subtype)  # type: ignore[operator]
        except ValueError:
            pass

    return _normalize_polars_dtype(dtype)  # type: ignore[arg-type]


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
