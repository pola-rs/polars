from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from polars.datatypes import (
    Boolean,
    Categorical,
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
    PolarsDataType,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    _base_type,
)

try:
    from polars.polars import PySeries

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if TYPE_CHECKING:
    import numpy as np


if not _DOCUMENTING:
    _POLARS_TYPE_TO_CONSTRUCTOR: dict[
        PolarsDataType, Callable[[str, Sequence[Any], bool], PySeries]
    ] = {
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
    dtype: PolarsDataType,
) -> Callable[[str, Sequence[Any], bool], PySeries]:
    """Get the right PySeries constructor for the given Polars dtype."""
    try:
        dtype = _base_type(dtype)
        return _POLARS_TYPE_TO_CONSTRUCTOR[dtype]
    except KeyError:  # pragma: no cover
        raise ValueError(f"Cannot construct PySeries for type {dtype}.") from None


_NUMPY_TYPE_TO_CONSTRUCTOR = None


def _set_numpy_to_constructor() -> None:
    import numpy as np

    global _NUMPY_TYPE_TO_CONSTRUCTOR
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
        np.datetime64: PySeries.new_i64,
    }


def numpy_type_to_constructor(dtype: type[np.dtype[Any]]) -> Callable[..., PySeries]:
    """Get the right PySeries constructor for the given Polars dtype."""
    if _NUMPY_TYPE_TO_CONSTRUCTOR is None:
        _set_numpy_to_constructor()
    try:
        return _NUMPY_TYPE_TO_CONSTRUCTOR[dtype]  # type:ignore[index]
    except KeyError:
        return PySeries.new_object
    except NameError:  # pragma: no cover
        raise ImportError(
            f"'numpy' is required to convert numpy dtype {dtype}."
        ) from None


if not _DOCUMENTING:
    _PY_TYPE_TO_CONSTRUCTOR = {
        float: PySeries.new_opt_f64,
        int: PySeries.new_opt_i64,
        str: PySeries.new_str,
        bool: PySeries.new_opt_bool,
    }


def py_type_to_constructor(dtype: type[Any]) -> Callable[..., PySeries]:
    """Get the right PySeries constructor for the given Python dtype."""
    try:
        return _PY_TYPE_TO_CONSTRUCTOR[dtype]
    except KeyError:
        return PySeries.new_object
