from __future__ import annotations

import contextlib
import enum
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._utils.convert import (
    date_to_int,
    datetime_to_int,
    time_to_int,
    timedelta_to_int,
)
from polars._utils.wrap import wrap_expr
from polars.datatypes import Date, Datetime, Duration, Enum, Time
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from polars import Expr
    from polars._typing import PolarsDataType, TimeUnit


def lit(
    value: Any, dtype: PolarsDataType | None = None, *, allow_object: bool = False
) -> Expr:
    """
    Return an expression representing a literal value.

    Parameters
    ----------
    value
        Value that should be used as a `literal`.
    dtype
        The data type of the resulting expression.
        If set to `None` (default), the data type is inferred from the `value` input.
    allow_object
        If type is unknown use an 'object' type.
        By default, we will raise a `ValueException`
        if the type is unknown.

    Notes
    -----
    Expected datatypes:

    - `pl.lit([])` -> empty  Series Float32
    - `pl.lit([1, 2, 3])` -> Series Int64
    - `pl.lit([[]])`-> empty  Series List<Null>
    - `pl.lit([[1, 2, 3]])` -> Series List<i64>
    - `pl.lit(None)` -> Series Null

    Examples
    --------
    Literal scalar values:

    >>> pl.lit(1)  # doctest: +IGNORE_RESULT
    >>> pl.lit(5.5)  # doctest: +IGNORE_RESULT
    >>> pl.lit(None)  # doctest: +IGNORE_RESULT
    >>> pl.lit("foo_bar")  # doctest: +IGNORE_RESULT
    >>> pl.lit(date(2021, 1, 20))  # doctest: +IGNORE_RESULT
    >>> pl.lit(datetime(2023, 3, 31, 10, 30, 45))  # doctest: +IGNORE_RESULT

    Literal list/Series data (1D):

    >>> pl.lit([1, 2, 3])  # doctest: +SKIP
    >>> pl.lit(pl.Series("x", [1, 2, 3]))  # doctest: +IGNORE_RESULT

    Literal list/Series data (2D):

    >>> pl.lit([[1, 2], [3, 4]])  # doctest: +SKIP
    >>> pl.lit(pl.Series("y", [[1, 2], [3, 4]]))  # doctest: +IGNORE_RESULT
    """
    time_unit: TimeUnit

    if isinstance(value, datetime):
        if dtype is not None and (tu := getattr(dtype, "time_unit", "us")) is not None:
            time_unit = tu  # type: ignore[assignment]
        else:
            time_unit = "us"

        time_zone: str | None = getattr(dtype, "time_zone", None)
        if (tzinfo := value.tzinfo) is not None:
            tzinfo_str = str(tzinfo)
            if time_zone is not None and time_zone != tzinfo_str:
                msg = f"time zone of dtype ({time_zone!r}) differs from time zone of value ({tzinfo!r})"
                raise TypeError(msg)
            time_zone = tzinfo_str

        dt_utc = value.replace(tzinfo=timezone.utc)
        dt_int = datetime_to_int(dt_utc, time_unit)
        expr = lit(dt_int).cast(Datetime(time_unit))
        if time_zone is not None:
            expr = expr.dt.replace_time_zone(
                time_zone, ambiguous="earliest" if value.fold == 0 else "latest"
            )
        return expr

    elif isinstance(value, timedelta):
        if dtype is not None and (tu := getattr(dtype, "time_unit", "us")) is not None:
            time_unit = tu  # type: ignore[assignment]
        else:
            time_unit = "us"

        td_int = timedelta_to_int(value, time_unit)
        return lit(td_int).cast(Duration(time_unit))

    elif isinstance(value, time):
        time_int = time_to_int(value)
        return lit(time_int).cast(Time)

    elif isinstance(value, date):
        date_int = date_to_int(value)
        return lit(date_int).cast(Date)

    elif isinstance(value, pl.Series):
        value = value._s
        return wrap_expr(plr.lit(value, allow_object))

    elif _check_for_numpy(value) and isinstance(value, np.ndarray):
        return lit(pl.Series("literal", value, dtype=dtype))

    elif isinstance(value, (list, tuple)):
        return lit(pl.Series("literal", [value], dtype=dtype))

    elif isinstance(value, enum.Enum):
        lit_value = value.value
        if dtype is None and isinstance(value, str):
            dtype = Enum(m.value for m in type(value))
        return lit(lit_value, dtype=dtype)

    if dtype:
        return wrap_expr(plr.lit(value, allow_object)).cast(dtype)

    try:
        # numpy literals like np.float32(0) have item/dtype
        item = value.item()

        # numpy item() is py-native datetime/timedelta when units < 'ns'
        if isinstance(item, (datetime, timedelta)):
            return lit(item)

        # handle 'ns' units
        if isinstance(item, int) and hasattr(value, "dtype"):
            dtype_name = value.dtype.name
            if dtype_name.startswith("datetime64["):
                time_unit = dtype_name[len("datetime64[") : -1]
                return lit(item).cast(Datetime(time_unit))
            if dtype_name.startswith("timedelta64["):
                time_unit = dtype_name[len("timedelta64[") : -1]
                return lit(item).cast(Duration(time_unit))

    except AttributeError:
        item = value

    return wrap_expr(plr.lit(item, allow_object))
