"""Strategies for generating various forms of data."""

from __future__ import annotations

import decimal
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Sequence

import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument

from polars._utils.constants import (
    EPOCH,
    I8_MAX,
    I8_MIN,
    I16_MAX,
    I16_MIN,
    I32_MAX,
    I32_MIN,
    I64_MAX,
    I64_MIN,
    U8_MAX,
    U16_MAX,
    U32_MAX,
    U64_MAX,
)
from polars.datatypes import (
    Array,
    Binary,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    String,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from polars.testing.parametric.strategies._utils import flexhash
from polars.testing.parametric.strategies.dtype import (
    _DEFAULT_ARRAY_WIDTH_LIMIT,
    _DEFAULT_ENUM_CATEGORIES_LIMIT,
)

if TYPE_CHECKING:
    from datetime import date, datetime, time

    from hypothesis.strategies import SearchStrategy

    from polars.datatypes import DataType, DataTypeClass
    from polars.type_aliases import PolarsDataType, TimeUnit

_DEFAULT_LIST_LEN_LIMIT = 3
_DEFAULT_N_CATEGORIES = 10

_INTEGER_STRATEGIES: dict[bool, dict[int, SearchStrategy[int]]] = {
    True: {
        8: st.integers(I8_MIN, I8_MAX),
        16: st.integers(I16_MIN, I16_MAX),
        32: st.integers(I32_MIN, I32_MAX),
        64: st.integers(I64_MIN, I64_MAX),
    },
    False: {
        8: st.integers(0, U8_MAX),
        16: st.integers(0, U16_MAX),
        32: st.integers(0, U32_MAX),
        64: st.integers(0, U64_MAX),
    },
}


def integers(
    bit_width: Literal[8, 16, 32, 64] = 64, *, signed: bool = True
) -> SearchStrategy[int]:
    """Create a strategy for generating integers."""
    return _INTEGER_STRATEGIES[signed][bit_width]


def floats(
    bit_width: Literal[32, 64] = 64, *, allow_infinity: bool = True
) -> SearchStrategy[float]:
    """Create a strategy for generating integers."""
    return st.floats(width=bit_width, allow_infinity=allow_infinity)


def booleans() -> SearchStrategy[bool]:
    """Create a strategy for generating booleans."""
    return st.booleans()


def strings() -> SearchStrategy[str]:
    """Create a strategy for generating string values."""
    alphabet = st.characters(max_codepoint=1000, exclude_categories=["Cs", "Cc"])
    return st.text(alphabet=alphabet, max_size=8)


def binary() -> SearchStrategy[bytes]:
    """Create a strategy for generating bytes."""
    return st.binary()


def categories(n_categories: int = _DEFAULT_N_CATEGORIES) -> SearchStrategy[str]:
    """
    Create a strategy for generating category strings.

    Parameters
    ----------
    n_categories
        The number of categories.
    """
    categories = [f"c{i}" for i in range(n_categories)]
    return st.sampled_from(categories)


def times() -> SearchStrategy[time]:
    """Create a strategy for generating `time` objects."""
    return st.times()


def dates() -> SearchStrategy[date]:
    """Create a strategy for generating `date` objects."""
    return st.dates()


def datetimes(time_unit: TimeUnit = "us") -> SearchStrategy[datetime]:
    """
    Create a strategy for generating `datetime` objects in the time unit's range.

    Parameters
    ----------
    time_unit
        Time unit for which the datetime objects are valid.
    """
    if time_unit in ("us", "ms"):
        # datetime.min/max fall within the range
        return st.datetimes()
    elif time_unit == "ns":
        return st.datetimes(
            min_value=EPOCH + timedelta(microseconds=I64_MIN // 1000),
            max_value=EPOCH + timedelta(microseconds=I64_MAX // 1000),
        )
    else:
        msg = f"invalid time unit: {time_unit}"
        raise InvalidArgument(msg)


def durations(time_unit: TimeUnit = "us") -> SearchStrategy[timedelta]:
    """
    Create a strategy for generating `timedelta` objects in the time unit's range.

    Parameters
    ----------
    time_unit
        Time unit for which the timedelta objects are valid.
    """
    if time_unit == "us":
        return st.timedeltas(
            min_value=timedelta(microseconds=I64_MIN),
            max_value=timedelta(microseconds=I64_MAX),
        )
    elif time_unit == "ns":
        return st.timedeltas(
            min_value=timedelta(microseconds=I64_MIN // 1000),
            max_value=timedelta(microseconds=I64_MAX // 1000),
        )
    elif time_unit == "ms":
        # TODO: Enable full range of millisecond durations
        # timedelta.min/max fall within the range
        # return st.timedeltas()
        return st.timedeltas(
            min_value=timedelta(microseconds=I64_MIN),
            max_value=timedelta(microseconds=I64_MAX),
        )
    else:
        msg = f"invalid time unit: {time_unit}"
        raise InvalidArgument(msg)


def decimals(
    precision: int | None = 38, scale: int = 0
) -> SearchStrategy[decimal.Decimal]:
    """
    Create a strategy for generating `Decimal` objects.

    Parameters
    ----------
    precision
        Maximum number of digits in each number.
        If set to `None`, the precision is set to 38 (the maximum supported by Polars).
    scale
        Number of digits to the right of the decimal point in each number.
    """
    if precision is None:
        precision = 38

    c = decimal.Context(prec=precision)
    exclusive_limit = c.create_decimal(f"1E+{precision - scale}")
    max_value = c.next_minus(exclusive_limit)
    min_value = c.copy_negate(max_value)

    return st.decimals(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        places=scale,
    )


def lists(
    inner_dtype: DataType,
    *,
    select_from: Sequence[Any] | None = None,
    min_len: int = 0,
    max_len: int | None = None,
    unique: bool = False,
    **kwargs: Any,
) -> SearchStrategy[list[Any]]:
    """
    Create a strategy for generating lists of the given data type.

    Parameters
    ----------
    inner_dtype
        Data type of the list elements. If the data type is not fully instantiated,
        defaults will be used, e.g. `Datetime` will become `Datetime('us')`.
    select_from
        The values to use for the innermost lists. If set to `None` (default),
        the default strategy associated with the innermost data type is used.
    min_len
        The minimum length of the generated lists.
    max_len
        The maximum length of the generated lists. If set to `None` (default), the
        maximum is set based on `min_size`: `3` if `min_len` is zero,
        otherwise `2 * min_len`.
    unique
        Ensure that the generated lists contain unique values.
    **kwargs
        Additional arguments that are passed to nested data generation strategies.

    Examples
    --------
    ...
    """
    if max_len is None:
        max_len = _DEFAULT_LIST_LEN_LIMIT if min_len == 0 else min_len * 2

    if select_from is not None and not inner_dtype.is_nested():
        inner_strategy = st.sampled_from(select_from)
    else:
        inner_strategy = data(
            inner_dtype,
            select_from=select_from,
            min_size=min_len,
            max_size=max_len,
            unique=unique,
            **kwargs,
        )

    return st.lists(
        elements=inner_strategy,
        min_size=min_len,
        max_size=max_len,
        unique_by=(flexhash if unique else None),
    )


def nulls() -> SearchStrategy[None]:
    """Create a strategy for generating null values."""
    return st.none()


# Strategies that are not customizable through parameters
_STATIC_STRATEGIES: dict[DataTypeClass, SearchStrategy[Any]] = {
    Boolean: booleans(),
    Int8: integers(8, signed=True),
    Int16: integers(16, signed=True),
    Int32: integers(32, signed=True),
    Int64: integers(64, signed=True),
    UInt8: integers(8, signed=False),
    UInt16: integers(16, signed=False),
    UInt32: integers(32, signed=False),
    UInt64: integers(64, signed=False),
    Time: times(),
    Date: dates(),
    String: strings(),
    Binary: binary(),
    Null: nulls(),
}


def data(
    dtype: PolarsDataType, *, allow_null: bool = False, **kwargs: Any
) -> SearchStrategy[Any]:
    """
    Create a strategy for generating data for the given data type.

    Parameters
    ----------
    dtype
        A Polars data type. If the data type is not fully instantiated, defaults will
        be used, e.g. `Datetime` will become `Datetime('us')`.
    allow_null
        Allow nulls as possible values.
    **kwargs
        Additional parameters for the strategy associated with the given `dtype`.
    """
    if (strategy := _STATIC_STRATEGIES.get(dtype.base_type())) is not None:
        strategy = strategy
    elif dtype == Float32:
        strategy = floats(32, allow_infinity=kwargs.pop("allow_infinity", True))
    elif dtype == Float64:
        strategy = floats(64, allow_infinity=kwargs.pop("allow_infinity", True))
    elif dtype == Datetime:
        # TODO: Handle time zones
        strategy = datetimes(time_unit=getattr(dtype, "time_unit", None) or "us")
    elif dtype == Duration:
        strategy = durations(time_unit=getattr(dtype, "time_unit", None) or "us")
    elif dtype == Categorical:
        strategy = categories(
            n_categories=kwargs.pop("n_categories", _DEFAULT_N_CATEGORIES)
        )
    elif dtype == Enum:
        if isinstance(dtype, Enum):
            if (cats := dtype.categories).is_empty():
                strategy = nulls()
            else:
                strategy = st.sampled_from(cats.to_list())
        else:
            strategy = categories(
                n_categories=kwargs.pop("n_categories", _DEFAULT_ENUM_CATEGORIES_LIMIT)
            )
    elif dtype == Decimal:
        strategy = decimals(
            getattr(dtype, "precision", None), getattr(dtype, "scale", 0)
        )
    elif dtype == List:
        inner = getattr(dtype, "inner", None) or Null()
        strategy = lists(inner, allow_null=allow_null, **kwargs)
    elif dtype == Array:
        inner = getattr(dtype, "inner", None) or Null()
        width = getattr(dtype, "width", _DEFAULT_ARRAY_WIDTH_LIMIT)
        strategy = lists(
            inner,
            min_len=width,
            max_len=width,
            allow_null=allow_null,
            **kwargs,
        )
    else:
        msg = f"unsupported data type: {dtype}"
        raise InvalidArgument(msg)

    if allow_null:
        strategy = nulls() | strategy

    return strategy
