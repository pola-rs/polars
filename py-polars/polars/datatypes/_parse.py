from __future__ import annotations

import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from typing import TYPE_CHECKING, Any, ForwardRef, NoReturn, Union, get_args

from polars.datatypes.classes import (
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float64,
    Int64,
    List,
    Null,
    Object,
    String,
    Time,
)
from polars.datatypes.convert import is_polars_dtype

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType, PythonDataType, SchemaDict


if sys.version_info >= (3, 10):
    from types import NoneType, UnionType
else:
    # Define equivalent for older Python versions
    NoneType = type(None)
    UnionType = type(int | str)
UnionTypeOld = type(Union[int, str])


def parse_into_dtype(input: Any) -> PolarsDataType:
    """Parse an input into a Polars data type."""
    if is_polars_dtype(input):
        return input

    elif isinstance(input, ForwardRef):
        return _parse_forward_ref_into_dtype(input)

    elif isinstance(input, (UnionType, UnionTypeOld)):
        return _parse_union_type_into_dtype(input)

    else:
        return parse_py_type_into_dtype(input)


def try_parse_into_dtype(input: Any) -> PolarsDataType | None:
    """Try parsing an input into a Polars data type, returning None on failure."""
    try:
        return parse_into_dtype(input)
    except TypeError:
        return None


@functools.lru_cache(16)
def parse_py_type_into_dtype(input: PythonDataType | type[object]) -> PolarsDataType:
    """Convert Python data type to Polars data type."""
    if input is int:
        return Int64()
    elif input is float:
        return Float64()
    elif input is str:
        return String()
    elif input is bool:
        return Boolean()
    elif input is date:
        return Date()
    elif input is datetime:
        return Datetime("us")
    elif input is timedelta:
        return Duration
    elif input is time:
        return Time()
    elif input is PyDecimal:
        return Decimal
    elif input is bytes:
        return Binary()
    elif input is object:
        return Object()
    elif input is NoneType:
        return Null()
    elif input is list or input is tuple:
        return List

    elif hasattr(input, "__origin__") and hasattr(input, "__args__"):
        base_type = input.__origin__
        if base_type is not None:
            dtype = parse_py_type_into_dtype(base_type)
            nested = input.__args__
            if len(nested) == 1:
                nested = nested[0]
            return (
                dtype if nested is None else dtype(parse_py_type_into_dtype(nested))  # type: ignore[operator]
            )

    else:
        _raise_on_invalid_dtype(input)


PY_TYPE_STR_TO_DTYPE: SchemaDict = {
    "int": Int64(),
    "float": Float64(),
    "bool": Boolean(),
    "str": String(),
    "bytes": Binary(),
    "date": Date(),
    "time": Time(),
    "datetime": Datetime("us"),
    "timedelta": Duration("us"),
    "object": Object(),
    "NoneType": Null(),
    "Decimal": Decimal,
    "list": List,
    "tuple": List,
}


def _parse_forward_ref_into_dtype(input: ForwardRef) -> PolarsDataType:
    """Parse a ForwardRef into a Polars data type."""
    annotation = input.__forward_arg__

    # Strip "optional" designation - Polars data types are always nullable
    formatted = re.sub(r"(^None \|)|(\| None$)", "", annotation).strip()

    try:
        return PY_TYPE_STR_TO_DTYPE[formatted]
    except KeyError:
        _raise_on_invalid_dtype(input)


def _parse_union_type_into_dtype(input: UnionType | UnionTypeOld) -> PolarsDataType:
    """
    Parse a union of types into a Polars data type.

    Unions of multiple non-null types (e.g. `int | float`) are not supported.

    Parameters
    ----------
    input
        A union type, e.g. `str | None` (new syntax) or `Union[str, None]` (old syntax).
    """
    # Strip "optional" designation - Polars data types are always nullable
    inner_types = [tp for tp in get_args(input) if tp is not NoneType]

    if len(inner_types) > 1:
        _raise_on_invalid_dtype(input)

    input = inner_types[0]
    return parse_into_dtype(input)


def _raise_on_invalid_dtype(input: Any) -> NoReturn:
    """Raise an informative error if the input could not be parsed."""
    msg = f"cannot parse input of type {type(input).__name__!r} into Polars data type: {input!r}"
    raise TypeError(msg) from None
