from __future__ import annotations

import contextlib
import warnings
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, overload

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import Date, Int64
from polars.expr.datetime import TIME_ZONE_DEPRECATION_MESSAGE
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr, wrap_s
from polars.utils.convert import (
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_duration,
)
from polars.utils.decorators import deprecated_alias
from polars.utils.various import find_stacklevel

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    import sys
    from datetime import date

    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, PolarsDataType, TimeUnit

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


@overload
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = ...,
    *,
    dtype: PolarsDataType | None = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = ...,
    *,
    dtype: PolarsDataType | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = ...,
    *,
    dtype: PolarsDataType | None = ...,
    eager: bool,
) -> Expr | Series:
    ...


@deprecated_alias(low="start", high="end")
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = 1,
    *,
    dtype: PolarsDataType | None = None,
    eager: bool = False,
) -> Expr | Series:
    """
    Create a range expression (or Series).

    This can be used in a `select`, `with_column` etc. Be sure that the resulting
    range size is equal to the length of the DataFrame you are collecting.

    Parameters
    ----------
    start
        Lower bound of range.
    end
        Upper bound of range.
    step
        Step size of the range.
    dtype
        Apply an explicit integer dtype to the resulting expression (default is Int64).
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.

    Examples
    --------
    >>> pl.arange(0, 3, eager=True)
    shape: (3,)
    Series: 'arange' [i64]
    [
            0
            1
            2
    ]

    >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df.select(pl.arange(pl.col("a"), pl.col("b")))
    shape: (2, 1)
    ┌───────────┐
    │ arange    │
    │ ---       │
    │ list[i64] │
    ╞═══════════╡
    │ [1, 2]    │
    │ [2, 3]    │
    └───────────┘

    """
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    range_expr = wrap_expr(plr.arange(start, end, step))

    if dtype is not None and dtype != Int64:
        range_expr = range_expr.cast(dtype)
    if not eager:
        return range_expr
    else:
        return pl.DataFrame().select(range_expr.alias("arange")).to_series()


@overload
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[False] = ...,
    name: str | None = ...,
) -> Expr:
    ...


@overload
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[True],
    name: str | None = ...,
) -> Series:
    ...


@overload
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool,
    name: str | None = ...,
) -> Series | Expr:
    ...


@deprecated_alias(low="start", high="end")
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
    name: str | None = None,
) -> Series | Expr:
    """
    Create a range of type `Datetime` (or `Date`).

    Parameters
    ----------
    start
        Lower bound of the date range, given as a date, datetime, Expr, or column name.
    end
        Upper bound of the date range, given as a date, datetime, Expr, or column name.
    interval
        Interval of the range periods; can be a python timedelta object like
        ``timedelta(days=10)`` or a polars duration string, such as ``3d12h4m25s``
        (representing 3 days, 12 hours, 4 minutes, and 25 seconds). Append
        ``_saturating`` to the interval string to restrict resulting invalid dates to
        valid ranges.

        It is common to attempt to create a month-end date series by using the "1mo"
        offset string with a start date at the end of the month. This will not produce
        the desired results. See Note #2 below for further information.
    closed : {'both', 'left', 'right', 'none'}
        Define whether the temporal window interval is closed or not.
    time_unit : {None, 'ns', 'us', 'ms'}
        Set the time unit.
    time_zone:
        Optional timezone
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    name
        Name of the output column.

        .. deprecated:: 0.18.0
            This argument is deprecated. Use the ``alias`` method instead.


    Notes
    -----
    1) If both ``start`` and ``end`` are passed as date types (not datetime), and the
    interval granularity is no finer than 1d, the returned range is also of
    type date. All other permutations return a datetime Series.

    2) Because different months of the year have differing numbers of days, the offset
    strings "1mo" and "1y" are not well-defined units of time, and vary according to
    their starting point. For example, February 1st offset by one month returns a time
    28 days later (in a non-leap year), whereas May 1st offset by one month returns a
    time 31 days later. In general, an offset of one month selects the same day in the
    following month. However, this is not always intended: when one begins Febrary 28th
    and offsets by 1 month, does the user intend to target March 28th (the next month
    but same day), or March 31st (the end of the month)?

    Polars uses the first approach: February 28th offset by 1 month is March 28th. When
    a date-series is generated, each date is offset as of the prior date, meaning that
    if one began January 31st, 2023, and offset by ``1mo_saturating`` until May 31st,
    the following dates would be generated:

    ``2023-01-31``, ``2023-02-28``, ``2023-03-28``, ``2023-04-28``, ``2023-05-28``.

    This is almost never the intended result. Instead, it is recommended to begin with
    the first day of the month and use the ``.dt.month_end()`` conversion routine, as
    in:

    >>> from datetime import date
    >>> pl.date_range(
    ...     date(2023, 1, 1), date(2023, 5, 1), "1mo", eager=True
    ... ).dt.month_end()
    shape: (5,)
    Series: 'date' [date]
    [
            2023-01-31
            2023-02-28
            2023-03-31
            2023-04-30
            2023-05-31
    ]

    Returns
    -------
    A Series of type `Datetime` or `Date`.

    Examples
    --------
    Using polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)
    shape: (3,)
    Series: 'date' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]

    Using `timedelta` object to specify the interval:

    >>> from datetime import datetime, timedelta
    >>> pl.date_range(
    ...     datetime(1985, 1, 1),
    ...     datetime(1985, 1, 10),
    ...     timedelta(days=1, hours=12),
    ...     time_unit="ms",
    ...     eager=True,
    ... )
    shape: (7,)
    Series: 'date' [datetime[ms]]
    [
        1985-01-01 00:00:00
        1985-01-02 12:00:00
        1985-01-04 00:00:00
        1985-01-05 12:00:00
        1985-01-07 00:00:00
        1985-01-08 12:00:00
        1985-01-10 00:00:00
    ]

    Specify a time zone

    >>> pl.date_range(
    ...     datetime(2022, 1, 1),
    ...     datetime(2022, 3, 1),
    ...     "1mo",
    ...     time_zone="America/New_York",
    ...     eager=True,
    ... )
    shape: (3,)
    Series: 'date' [datetime[μs, America/New_York]]
    [
        2022-01-01 00:00:00 EST
        2022-02-01 00:00:00 EST
        2022-03-01 00:00:00 EST
    ]

    Combine with ``month_end`` to get the last day of the month:

    >>> (
    ...     pl.date_range(
    ...         datetime(2022, 1, 1), datetime(2022, 3, 1), "1mo", eager=True
    ...     ).dt.month_end()
    ... )
    shape: (3,)
    Series: 'date' [datetime[μs]]
    [
        2022-01-31 00:00:00
        2022-02-28 00:00:00
        2022-03-31 00:00:00
    ]

    """
    if name is not None:
        warnings.warn(
            "the `name` argument is deprecated. Use the `alias` method instead.",
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    from polars.dependencies import zoneinfo

    if time_zone is not None and time_zone not in zoneinfo.available_timezones():
        warnings.warn(
            TIME_ZONE_DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "")

    if (
        not eager
        or isinstance(start, (str, pl.Expr))
        or isinstance(end, (str, pl.Expr))
    ):
        start = parse_as_expression(start)
        end = parse_as_expression(end)
        expr = wrap_expr(plr.date_range_lazy(start, end, interval, closed, time_zone))
        if name is not None:
            expr = expr.alias(name)
        return expr

    start, start_is_date = _ensure_datetime(start)
    end, end_is_date = _ensure_datetime(end)

    if start.tzinfo is not None or time_zone is not None:
        if start.tzinfo != end.tzinfo:
            raise ValueError(
                "Cannot mix different timezone aware datetimes."
                f" Got: '{start.tzinfo}' and '{end.tzinfo}'."
            )

        if time_zone is not None and start.tzinfo is not None:
            if str(start.tzinfo) != time_zone:
                raise ValueError(
                    "Given time_zone is different from that of timezone aware datetimes."
                    f" Given: '{time_zone}', got: '{start.tzinfo}'."
                )
        if time_zone is None and start.tzinfo is not None:
            time_zone = str(start.tzinfo)

    time_unit_: TimeUnit
    if time_unit is not None:
        time_unit_ = time_unit
    elif "ns" in interval:
        time_unit_ = "ns"
    else:
        time_unit_ = "us"

    start_pl = _datetime_to_pl_timestamp(start, time_unit_)
    end_pl = _datetime_to_pl_timestamp(end, time_unit_)
    dt_range = wrap_s(
        plr.date_range_eager(start_pl, end_pl, interval, closed, time_unit_, time_zone)
    )
    if (
        start_is_date
        and end_is_date
        and not _interval_granularity(interval).endswith(("h", "m", "s"))
    ):
        dt_range = dt_range.cast(Date)

    if name is not None:
        dt_range = dt_range.alias(name)
    return dt_range


def _ensure_datetime(value: date | datetime) -> tuple[datetime, bool]:
    is_date_type = False
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
        is_date_type = True
    return value, is_date_type


def _interval_granularity(interval: str) -> str:
    return interval[-2:].lstrip("0123456789")


@overload
def time_range(
    start: time | Expr | str | None = ...,
    end: time | Expr | str | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
    name: str | None = ...,
) -> Expr:
    ...


@overload
def time_range(
    start: time | Expr | str | None = ...,
    end: time | Expr | str | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
    name: str | None = ...,
) -> Series:
    ...


@overload
def time_range(
    start: time | Expr | str | None = ...,
    end: time | Expr | str | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
    name: str | None = ...,
) -> Series | Expr:
    ...


def time_range(
    start: time | Expr | str | None = None,
    end: time | Expr | str | None = None,
    interval: str | timedelta = "1h",
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
    name: str | None = None,
) -> Series | Expr:
    """
    Create a range of type `Time`.

    Parameters
    ----------
    start
        Lower bound of the time range, given as a time, Expr, or column name.
        If omitted, will default to ``time(0,0,0,0)``.
    end
        Upper bound of the time range, given as a time, Expr, or column name.
        If omitted, will default to ``time(23,59,59,999999)``.
    interval
        Interval of the range periods; can be a python timedelta object like
        ``timedelta(minutes=10)`` or a polars duration string, such as ``1h30m25s``
        (representing 1 hour, 30 minutes, and 25 seconds).
    closed : {'both', 'left', 'right', 'none'}
        Define whether the temporal window interval is closed or not.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    name
        Name of the output column.

        .. deprecated:: 0.18.0
            This argument is deprecated. Use the ``alias`` method instead.

    Returns
    -------
    A Series of type `Time`.

    Examples
    --------
    Create a Series that starts at 14:00, with intervals of 3 hours and 15 mins:

    >>> from datetime import time
    >>> pl.time_range(
    ...     start=time(14, 0),
    ...     interval=timedelta(hours=3, minutes=15),
    ...     eager=True,
    ... )
    shape: (4,)
    Series: 'time' [time]
    [
        14:00:00
        17:15:00
        20:30:00
        23:45:00
    ]

    Generate a DataFrame with two columns made of eager ``time_range`` Series,
    and create a third column using ``time_range`` in expression context:

    >>> lf = pl.LazyFrame(
    ...     {
    ...         "start": pl.time_range(interval="6h", eager=True),
    ...         "stop": pl.time_range(start=time(2, 59), interval="5h59m", eager=True),
    ...     }
    ... ).with_columns(
    ...     intervals=pl.time_range("start", "stop", interval="1h29m", eager=False)
    ... )
    >>> lf.collect()
    shape: (4, 3)
    ┌──────────┬──────────┬────────────────────────────────┐
    │ start    ┆ stop     ┆ intervals                      │
    │ ---      ┆ ---      ┆ ---                            │
    │ time     ┆ time     ┆ list[time]                     │
    ╞══════════╪══════════╪════════════════════════════════╡
    │ 00:00:00 ┆ 02:59:00 ┆ [00:00:00, 01:29:00, 02:58:00] │
    │ 06:00:00 ┆ 08:58:00 ┆ [06:00:00, 07:29:00, 08:58:00] │
    │ 12:00:00 ┆ 14:57:00 ┆ [12:00:00, 13:29:00]           │
    │ 18:00:00 ┆ 20:56:00 ┆ [18:00:00, 19:29:00]           │
    └──────────┴──────────┴────────────────────────────────┘

    """
    if name is not None:
        warnings.warn(
            "the `name` argument is deprecated. Use the `alias` method instead.",
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "").lower()

    for unit in ("y", "mo", "w", "d"):
        if unit in interval:
            raise ValueError(f"invalid interval unit for time_range: found {unit!r}")

    default_start = time(0, 0, 0)
    default_end = time(23, 59, 59, 999999)
    if (
        not eager
        or isinstance(start, (str, pl.Expr))
        or isinstance(end, (str, pl.Expr))
    ):
        start_expr = (
            F.lit(default_start)._pyexpr
            if start is None
            else parse_as_expression(start)
        )

        end_expr = (
            F.lit(default_end)._pyexpr if end is None else parse_as_expression(end)
        )

        tm_expr = wrap_expr(plr.time_range_lazy(start_expr, end_expr, interval, closed))
        if name is not None:
            tm_expr = tm_expr.alias(name)
        return tm_expr
    else:
        tm_srs = wrap_s(
            plr.time_range_eager(
                _time_to_pl_time(default_start if start is None else start),
                _time_to_pl_time(default_end if end is None else end),
                interval,
                closed,
            )
        )
        if name is not None:
            tm_srs = tm_srs.alias(name)
        return tm_srs
