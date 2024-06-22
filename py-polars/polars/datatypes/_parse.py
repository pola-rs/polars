from __future__ import annotations

import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Literal,
    Optional,
    Union,
    get_args,
    overload,
)

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
def _map_py_type_to_dtype(
    python_dtype: PythonDataType | type[object],
) -> PolarsDataType:
    """Convert Python data type to Polars data type."""
    if python_dtype is float:
        return Float64
    if python_dtype is int:
        return Int64
    if python_dtype is str:
        return String
    if python_dtype is bool:
        return Boolean
    if issubclass(python_dtype, datetime):
        # `datetime` is a subclass of `date`,
        # so need to check `datetime` first
        return Datetime("us")
    if issubclass(python_dtype, date):
        return Date
    if python_dtype is timedelta:
        return Duration
    if python_dtype is time:
        return Time
    if python_dtype is list:
        return List
    if python_dtype is tuple:
        return List
    if python_dtype is PyDecimal:
        return Decimal
    if python_dtype is bytes:
        return Binary
    if python_dtype is object:
        return Object
    if python_dtype is None.__class__:
        return Null

    # cover generic typing aliases, such as 'list[str]'
    if hasattr(python_dtype, "__origin__") and hasattr(python_dtype, "__args__"):
        base_type = python_dtype.__origin__
        if base_type is not None:
            dtype = _map_py_type_to_dtype(base_type)
            nested = python_dtype.__args__
            if len(nested) == 1:
                nested = nested[0]
            return (
                dtype if nested is None else dtype(_map_py_type_to_dtype(nested))  # type: ignore[operator]
            )

    msg = f"unrecognised Python type: {python_dtype!r}"
    raise TypeError(msg)


@overload
def parse_into_dtype(
    data_type: Any, *, raise_unmatched: Literal[True] = ...
) -> PolarsDataType: ...


@overload
def parse_into_dtype(
    data_type: Any, *, raise_unmatched: Literal[False]
) -> PolarsDataType | None: ...


def parse_into_dtype(
    data_type: Any, *, raise_unmatched: bool = True
) -> PolarsDataType | None:
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
    elif type(data_type).__name__ == "InitVar":
        data_type = data_type.type

    if is_polars_dtype(data_type):
        return data_type

    elif isinstance(data_type, (OptionType, UnionType)):
        # not exhaustive; handles the common "type | None" case, but
        # should probably pick appropriate supertype when n_types > 1?
        possible_types = [tp for tp in get_args(data_type) if tp is not NoneType]
        if len(possible_types) == 1:
            data_type = possible_types[0]

    try:
        return _map_py_type_to_dtype(data_type)
    except (KeyError, TypeError):  # pragma: no cover
        if raise_unmatched:
            msg = f"cannot infer dtype from {data_type!r} (type: {type(data_type).__name__!r})"
            raise ValueError(msg) from None
        return None
