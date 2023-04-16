from __future__ import annotations

from datetime import datetime, timedelta
from string import ascii_letters, ascii_uppercase, digits, punctuation
from typing import TYPE_CHECKING, Any

from hypothesis.strategies import (
    booleans,
    characters,
    dates,
    datetimes,
    floats,
    from_type,
    integers,
    text,
    timedeltas,
    times,
)

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
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn

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
    max_size=8,
    alphabet=characters(max_codepoint=1000, blacklist_categories=("Cs", "Cc")),
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

scalar_strategies: dict[PolarsDataType, Any] = {
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
