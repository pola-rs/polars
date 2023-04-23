from __future__ import annotations

import os
from datetime import datetime, timedelta
from random import choice
from string import ascii_letters, ascii_uppercase, digits, punctuation
from typing import TYPE_CHECKING, Any, Sequence

from hypothesis.strategies import (
    booleans,
    characters,
    composite,
    dates,
    datetimes,
    decimals,
    floats,
    from_type,
    integers,
    lists,
    sampled_from,
    text,
    timedeltas,
    times,
)

from polars.datatypes import (
    Boolean,
    Categorical,
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
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)

if TYPE_CHECKING:
    from decimal import Decimal as PyDecimal

    from hypothesis.strategies import DrawFn, SearchStrategy

    from polars.type_aliases import PolarsDataType


def between(draw: DrawFn, type_: type, min_: Any, max_: Any) -> Any:
    """Draw a value in a given range from a type-inferred strategy."""
    strategy_init = from_type(type_).function  # type: ignore[attr-defined]
    return draw(strategy_init(min_, max_))


# scalar dtype strategies are largely straightforward, mapping directly
# onto the associated hypothesis strategy, with dtype-defined limits
strategy_bool = booleans()
strategy_f32 = floats(width=32)
strategy_f64 = floats(width=64)
strategy_i8 = integers(min_value=-(2**7), max_value=(2**7) - 1)
strategy_i16 = integers(min_value=-(2**15), max_value=(2**15) - 1)
strategy_i32 = integers(min_value=-(2**31), max_value=(2**31) - 1)
strategy_i64 = integers(min_value=-(2**63), max_value=(2**63) - 1)
strategy_u8 = integers(min_value=0, max_value=(2**8) - 1)
strategy_u16 = integers(min_value=0, max_value=(2**16) - 1)
strategy_u32 = integers(min_value=0, max_value=(2**32) - 1)
strategy_u64 = integers(min_value=0, max_value=(2**64) - 1)

strategy_ascii = text(max_size=8, alphabet=ascii_letters + digits + punctuation)
strategy_categorical = text(max_size=2, alphabet=ascii_uppercase)
strategy_utf8 = text(
    alphabet=characters(max_codepoint=1000, blacklist_categories=("Cs", "Cc")),
    max_size=8,
)
strategy_datetime_ns = datetimes(
    min_value=datetime(1677, 9, 22, 0, 12, 43, 145225),
    max_value=datetime(2262, 4, 11, 23, 47, 16, 854775),
)
strategy_datetime_us = strategy_datetime_ms = datetimes(
    min_value=datetime(1, 1, 1),
    max_value=datetime(9999, 12, 31, 23, 59, 59, 999000),
)
strategy_time = times()
strategy_date = dates()
strategy_duration = timedeltas(
    min_value=timedelta(microseconds=-(2**46)),
    max_value=timedelta(microseconds=(2**46) - 1),
)


@composite
def strategy_decimal(draw: DrawFn) -> PyDecimal:
    places = draw(integers(min_value=0, max_value=18))
    return draw(
        # TODO: once fixed, re-enable decimal nan/inf values.
        decimals(
            allow_nan=False,
            allow_infinity=False,
            min_value=-(2**66),
            max_value=(2**66) - 1,
            places=places,
        )
    )


scalar_strategies: dict[PolarsDataType, SearchStrategy[Any]] = {
    Boolean: strategy_bool,
    Float32: strategy_f32,
    Float64: strategy_f64,
    Int8: strategy_i8,
    Int16: strategy_i16,
    Int32: strategy_i32,
    Int64: strategy_i64,
    UInt8: strategy_u8,
    UInt16: strategy_u16,
    UInt32: strategy_u32,
    UInt64: strategy_u64,
    Time: strategy_time,
    Date: strategy_date,
    Datetime("ns"): strategy_datetime_ns,
    Datetime("us"): strategy_datetime_us,
    Datetime("ms"): strategy_datetime_ms,
    Datetime: strategy_datetime_us,
    Duration("ns"): strategy_duration,
    Duration("us"): strategy_duration,
    Duration("ms"): strategy_duration,
    Duration: strategy_duration,
    Categorical: strategy_categorical,
    Utf8: strategy_utf8,
}

# note: decimal support is in early development and requires opt-in
if os.environ.get("POLARS_ACTIVATE_DECIMAL") == "1":
    scalar_strategies[Decimal] = strategy_decimal()

_strategy_dtypes = list(scalar_strategies) + [List]


def _hash(elem: Any) -> int:
    """Hashing that also handles lists/dicts (for 'unique' check)."""
    if isinstance(elem, list):
        return hash(tuple(_hash(e) for e in elem))
    elif isinstance(elem, dict):
        return hash((_hash(k), _hash(v)) for k, v in elem.items())
    return hash(elem)


def create_list_strategy(
    inner_dtype: PolarsDataType | None,
    select_from: Sequence[Any] | None = None,
    size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    unique: bool = False,
) -> Any:
    """
    Hypothesis strategy for producing polars List data.

    Parameters
    ----------
    inner_dtype : PolarsDataType
        type of the inner list elements (can also be another List).
    select_from : list, optional
        randomly select the innermost values from this list (otherwise
        the default strategy associated with the innermost dtype is used).
    size : int, optional
        if set, generated lists will be of exactly this size (and
        ignore the min_size/max_size params).
    min_size : int, optional
        set the minimum size of the generated lists (default: 0 if unset).
    max_size : int, optional
        set the maximum size of the generated lists (default: 3 if
        min_size is unset or zero, otherwise 2x min_size).
    unique : bool, optional
        ensure that the generated lists contain unique values.

    Examples
    --------
    Create a strategy that generates a list of i32 values:

    >>> lst = create_list_strategy(inner_dtype=pl.Int32)
    >>> lst.example()  # doctest: +SKIP
    [-11330, 24030, 116]

    Create a strategy that generates lists of lists of specific strings:

    >>> lst = create_list_strategy(
    ...     inner_dtype=pl.List(pl.Utf8),
    ...     select_from=["xx", "yy", "zz"],
    ... )
    >>> lst.example()  # doctest: +SKIP
    [['yy', 'xx'], [], ['zz']]

    Create a UInt8 dtype strategy as a hypothesis composite that generates
    pairs of small int values where the first is always <= the second:

    >>> from hypothesis.strategies import composite
    >>>
    >>> @composite
    ... def uint8_pairs(draw, uints=create_list_strategy(pl.UInt8, size=2)):
    ...     pairs = list(zip(draw(uints), draw(uints)))
    ...     return [sorted(ints) for ints in pairs]
    ...
    >>> uint8_pairs().example()  # doctest: +SKIP
    [(12, 22), (15, 131)]
    >>> uint8_pairs().example()  # doctest: +SKIP
    [(59, 176), (149, 149)]

    """
    if select_from and inner_dtype is None:
        raise ValueError("If specifying 'select_from', must also specify 'inner_dtype'")

    if inner_dtype is None:
        inner_dtype = choice(_strategy_dtypes)
    if size:
        min_size = max_size = size
    else:
        min_size = min_size or 0
        if max_size is None:
            max_size = 3 if not min_size else (min_size * 2)

    if inner_dtype == List:
        st = create_list_strategy(
            inner_dtype=inner_dtype.inner,  # type: ignore[union-attr]
            select_from=select_from,
            min_size=min_size,
            max_size=max_size,
        )
        if inner_dtype.inner is None:  # type: ignore[union-attr]
            inner_dtype = st._dtype
    else:
        st = (
            sampled_from(list(select_from))
            if select_from
            else scalar_strategies[inner_dtype]
        )

    ls = lists(
        elements=st,
        min_size=min_size,
        max_size=max_size,
        unique_by=(_hash if unique else None),
    )
    ls._dtype = List(inner_dtype)  # type: ignore[attr-defined, arg-type]
    return ls


# TODO: strategy for Struct dtype.
# def create_struct_strategy(
