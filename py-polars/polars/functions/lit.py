from __future__ import annotations

import contextlib
import enum
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._utils.wrap import wrap_expr
from polars.datatypes import Date, Datetime, Duration, Enum
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from polars import Expr
    from polars._typing import PolarsDataType, TimeUnit


def _np_int_and_dtype_datetime(
    value: np.datetime64, dtype: PolarsDataType | None
) -> tuple[int, PolarsDataType]:
    """Determine numpy scalar conversion from datetime64."""
    if dtype == Date:
        np_unit = "D"
        int_type = np.int32
    elif dtype == Datetime:
        np_unit = dtype.time_unit or "us"  # type: ignore[union-attr]
        int_type = np.int64  # type: ignore[assignment]
    else:
        # output dtype not provided; infer from numpy dtype and cast value to proper int
        unit, _ = np.datetime_data(value)
        if unit in ("Y", "M", "W", "D"):
            np_unit = "D"
            dtype = Date
            int_type = np.int32
        elif unit in ("h", "m", "s", "ms"):
            np_unit = "ms"
            int_type = np.int64  # type: ignore[assignment]
            dtype = Datetime("ms")
        elif unit in ("ns", "ps", "fs", "as"):
            np_unit = "ns"
            int_type = np.int64  # type: ignore[assignment]
            dtype = Datetime("ns")
        else:
            np_unit = "us"
            int_type = np.int64  # type: ignore[assignment]
            dtype = Datetime("us")
    return (
        # cast to np time unit, then to np.int, then to int
        int(value.astype(f"datetime64[{np_unit}]").astype(int_type)),
        dtype,
    )


def _np_int_and_dtype_timedelta(
    value: np.timedelta64, dtype: PolarsDataType | None
) -> tuple[int, PolarsDataType]:
    """Determine numpy scalar conversion from timedelta64."""
    if dtype is None:
        # output dtype not provided; infer from numpy dtype and cast value to proper int
        unit, _ = np.datetime_data(value)
        if unit in ("Y", "M", "W", "D", "h", "m", "s", "ms"):
            np_unit = "ms"
            int_type = np.int64
            dtype = Duration("ms")
        elif unit in ("ns", "ps", "fs", "as"):
            np_unit = "ns"
            int_type = np.int64
            dtype = Duration("ns")
        else:
            np_unit = "us"
            int_type = np.int64
            dtype = Duration("us")
    else:
        np_unit = dtype.time_unit or "us"  # type: ignore[union-attr]
        int_type = np.int64
    return (
        # cast to np time unit, then to np.int, then to int
        int(value.astype(f"timedelta64[{np_unit}]").astype(int_type)),
        dtype,
    )


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

    if _check_for_numpy(value):
        if isinstance(value, np.datetime64):
            # Units < ns must be converted to ns. Units > ms are converted to ms.
            value, dtype = _np_int_and_dtype_datetime(value, dtype)
            return wrap_expr(plr.lit(value, allow_object=False)).cast(dtype)
        elif isinstance(value, np.timedelta64):
            value, dtype = _np_int_and_dtype_timedelta(value, dtype)
            return wrap_expr(plr.lit(value, allow_object=False)).cast(dtype)
        elif isinstance(value, np.ndarray):
            return lit(pl.Series("literal", value, dtype=dtype))

    if isinstance(value, datetime):
        if dtype == Date:
            return wrap_expr(plr.lit(value.date(), allow_object=False))

        # parse time unit
        if dtype is not None and (tu := getattr(dtype, "time_unit", "us")) is not None:
            time_unit = tu  # type: ignore[assignment]
        else:
            time_unit = "us"

        # parse time zone
        dtype_tz = getattr(dtype, "time_zone", None)
        value_tz = value.tzinfo
        if value_tz is None:
            tz = dtype_tz
        else:
            if dtype_tz is None:
                # value has time zone, but dtype does not: keep value time zone
                tz = str(value_tz)
            elif str(value_tz) == dtype_tz:
                # dtype and value both have same time zone
                tz = str(value_tz)
            else:
                # value has time zone that differs from dtype time zone
                msg = (
                    f"time zone of dtype ({dtype_tz!r}) differs from time zone of "
                    f"value ({value_tz!r})"
                )
                raise TypeError(msg)

        dt_utc = value.replace(tzinfo=timezone.utc)
        expr = wrap_expr(plr.lit(dt_utc, allow_object=False)).cast(Datetime(time_unit))
        if tz is not None:
            expr = expr.dt.replace_time_zone(
                tz, ambiguous="earliest" if value.fold == 0 else "latest"
            )
        return expr

    elif isinstance(value, timedelta):
        expr = wrap_expr(plr.lit(value, allow_object=False))
        if dtype is not None and (tu := getattr(dtype, "time_unit", None)) is not None:
            expr = expr.cast(Duration(tu))
        return expr

    elif isinstance(value, time):
        return wrap_expr(plr.lit(value, allow_object=False))

    elif isinstance(value, date):
        if dtype == Datetime:
            time_unit = getattr(dtype, "time_unit", "us") or "us"
            dt_utc = datetime(value.year, value.month, value.day)
            expr = wrap_expr(plr.lit(dt_utc, allow_object=False)).cast(
                Datetime(time_unit)
            )
            if (time_zone := getattr(dtype, "time_zone", None)) is not None:
                expr = expr.dt.replace_time_zone(str(time_zone))
            return expr
        else:
            return wrap_expr(plr.lit(value, allow_object=False))

    elif isinstance(value, pl.Series):
        value = value._s
        return wrap_expr(plr.lit(value, allow_object))

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
