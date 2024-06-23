from __future__ import annotations

import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from typing import TYPE_CHECKING, Any, ForwardRef, Optional, Union, get_args

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


OptionType = type(Optional[type])
if sys.version_info >= (3, 10):
    from types import NoneType, UnionType
else:
    # infer equivalent class
    NoneType = type(None)
    UnionType = type(Union[int, float])

PY_STR_TO_DTYPE: SchemaDict = {
    "float": Float64,
    "int": Int64,
    "str": String,
    "bool": Boolean,
    "date": Date,
    "datetime": Datetime("us"),
    "timedelta": Duration("us"),
    "time": Time,
    "list": List,
    "tuple": List,
    "Decimal": Decimal,
    "bytes": Binary,
    "object": Object,
    "NoneType": Null,
}


@functools.lru_cache(16)
def parse_py_type_into_dtype(
    python_dtype: PythonDataType | type[object],
) -> PolarsDataType:
    """Convert Python data type to Polars data type."""
    if python_dtype is int:
        return Int64()
    elif python_dtype is float:
        return Float64()
    elif python_dtype is str:
        return String()
    elif python_dtype is bool:
        return Boolean()
    elif issubclass(python_dtype, datetime):
        # `datetime` is a subclass of `date`,
        # so need to check `datetime` first
        return Datetime("us")
    elif issubclass(python_dtype, date):
        return Date()
    elif python_dtype is timedelta:
        return Duration
    elif python_dtype is time:
        return Time()
    elif python_dtype is PyDecimal:
        return Decimal
    elif python_dtype is bytes:
        return Binary()
    elif python_dtype is object:
        return Object()
    elif python_dtype is None.__class__:
        return Null()
    elif python_dtype is list or python_dtype is tuple:
        return List

    elif hasattr(python_dtype, "__origin__") and hasattr(python_dtype, "__args__"):
        base_type = python_dtype.__origin__
        if base_type is not None:
            dtype = parse_py_type_into_dtype(base_type)
            nested = python_dtype.__args__
            if len(nested) == 1:
                nested = nested[0]
            return (
                dtype if nested is None else dtype(parse_py_type_into_dtype(nested))  # type: ignore[operator]
            )

    else:
        msg = f"unrecognized Python type: {python_dtype!r}"
        raise TypeError(msg)


def parse_into_dtype(input: Any) -> PolarsDataType:
    """Parse an input into a Polars data type."""
    if isinstance(input, ForwardRef):
        annotation = input.__forward_arg__
        input = (
            PY_STR_TO_DTYPE.get(
                re.sub(r"(^None \|)|(\| None$)", "", annotation).strip(), input
            )
            if isinstance(annotation, str)  # type: ignore[redundant-expr]
            else annotation
        )
    elif type(input).__name__ == "InitVar":
        input = input.type

    if is_polars_dtype(input):
        return input

    elif isinstance(input, (OptionType, UnionType)):
        # not exhaustive; handles the common "type | None" case, but
        # should probably pick appropriate supertype when n_types > 1?
        possible_types = [tp for tp in get_args(input) if tp is not NoneType]
        if len(possible_types) == 1:
            input = possible_types[0]

    try:
        return parse_py_type_into_dtype(input)
    except (KeyError, TypeError):  # pragma: no cover
        msg = f"cannot parse input of type {type(input).__name__!r} into Polars data type: {input!r}"
        raise TypeError(msg) from None


def try_parse_into_dtype(input: Any) -> PolarsDataType | None:
    """Try parsing an input into a Polars data type, returning None on failure."""
    try:
        return parse_into_dtype(input)
    except TypeError:
        return None
