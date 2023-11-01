from __future__ import annotations

import contextlib
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars.datatypes import Date, Datetime, Duration, Time
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np
from polars.utils._wrap import wrap_expr
from polars.utils.convert import (
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
)
from polars.utils.deprecation import issue_deprecation_warning

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import PolarsDataType, TimeUnit


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
        Optionally define a dtype.
    allow_object
        If type is unknown use an 'object' type.
        By default, we will raise a `ValueException`
        if the type is unknown.

    Notes
    -----
    Expected datatypes

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
        time_unit = "us" if dtype is None else getattr(dtype, "time_unit", "us")
        time_zone = (
            value.tzinfo
            if getattr(dtype, "time_zone", None) is None
            else getattr(dtype, "time_zone", None)
        )
        if (
            value.tzinfo is not None
            and getattr(dtype, "time_zone", None) is not None
            and dtype.time_zone != str(value.tzinfo)  # type: ignore[union-attr]
        ):
            raise TypeError(
                f"time zone of dtype ({dtype.time_zone!r}) differs from time zone of value ({value.tzinfo!r})"  # type: ignore[union-attr]
            )
        e = lit(
            _datetime_to_pl_timestamp(value.replace(tzinfo=timezone.utc), time_unit)
        ).cast(Datetime(time_unit))
        if time_zone is not None:
            return e.dt.replace_time_zone(
                str(time_zone), ambiguous="earliest" if value.fold == 0 else "latest"
            )
        else:
            return e

    elif isinstance(value, timedelta):
        time_unit = "us" if dtype is None else getattr(dtype, "time_unit", "us")
        return lit(_timedelta_to_pl_timedelta(value, time_unit)).cast(
            Duration(time_unit)
        )

    elif isinstance(value, time):
        return lit(_time_to_pl_time(value)).cast(Time)

    elif isinstance(value, date):
        return lit(datetime(value.year, value.month, value.day)).cast(Date)

    elif isinstance(value, pl.Series):
        name = value.name
        value = value._s
        e = wrap_expr(plr.lit(value, allow_object))
        if name == "":
            return e
        return e.alias(name)

    elif _check_for_numpy(value) and isinstance(value, np.ndarray):
        return lit(pl.Series("", value))

    elif isinstance(value, (list, tuple)):
        issue_deprecation_warning(
            "Behavior for `lit` will change for sequence inputs."
            " The result will change to be a literal of type List."
            " To retain the old behavior, pass a Series instead, e.g. `Series(sequence)`.",
            version="0.18.14",
        )
        return lit(pl.Series("", value))

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
