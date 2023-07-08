from __future__ import annotations

import contextlib
import warnings
from datetime import time, timedelta
from typing import TYPE_CHECKING, overload

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import Int64
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr, wrap_s
from polars.utils.convert import (
    _time_to_pl_time,
    _timedelta_to_pl_duration,
)
from polars.utils.decorators import deprecated_alias
from polars.utils.various import find_stacklevel

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from datetime import date, datetime
    from typing import Literal

    from polars import Expr, Series
    from polars.type_aliases import (
        ClosedInterval,
        IntoExpr,
        PolarsDataType,
        PolarsIntegerType,
        TimeUnit,
    )


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
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsDataType | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def arange(
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsDataType | None = ...,
    eager: bool,
) -> Expr | Series:
    ...


@deprecated_alias(low="start", high="end")
def arange(
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = 1,
    *,
    dtype: PolarsDataType | None = None,
    eager: bool = False,
) -> Expr | Series:
    """
    Generate a range of integers.

    .. deprecated:: 0.18.5
        ``arange`` has been replaced by two new functions: ``int_range`` for generating
        a single range, and ``int_ranges`` for generating a list column with multiple
        ranges. ``arange`` will remain available as an alias for `int_range`, which
        means it will lose the functionality to generate multiple ranges.

    Parameters
    ----------
    start
        Lower bound of the range (inclusive).
    end
        Upper bound of the range (exclusive).
    step
        Step size of the range.
    dtype
        Data type of the resulting column. Defaults to ``Int64``.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.

    See Also
    --------
    int_range : Generate a range of integers.
    int_ranges : Generate a range of integers for each row of the input columns.

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

    """
    # This check is not water-proof, but we cannot check for literal expressions here
    if not (isinstance(start, int) and isinstance(end, int)):
        warnings.warn(
            " `arange` has been replaced by two new functions:"
            " `int_range` for generating a single range,"
            " and `int_ranges` for generating a list column with multiple ranges."
            " `arange` will remain available as an alias for `int_range`, which means its behaviour will change."
            " To silence this warning, use either of the new functions.",
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.arange(start, end, step))

    if dtype is not None and dtype != Int64:
        result = result.cast(dtype)
    if eager:
        return F.select(result).to_series()

    return result


@overload
def int_range(
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def int_range(
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def int_range(
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: bool,
) -> Expr | Series:
    ...


def int_range(
    start: int | IntoExpr,
    end: int | IntoExpr,
    step: int = 1,
    *,
    dtype: PolarsIntegerType = Int64,
    eager: bool = False,
) -> Expr | Series:
    """
    Generate a range of integers.

    Parameters
    ----------
    start
        Lower bound of the range (inclusive).
    end
        Upper bound of the range (exclusive).
    step
        Step size of the range.
    dtype
        Data type of the range. Defaults to ``Int64``.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.

    Returns
    -------
    Column of data type ``Int64``.

    See Also
    --------
    int_ranges : Generate a range of integers for each row of the input columns.

    Examples
    --------
    >>> pl.int_range(0, 3, eager=True)
    shape: (3,)
    Series: 'int' [i64]
    [
            0
            1
            2
    ]

    """
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.int_range(start, end, step, dtype))

    if eager:
        return F.select(result).to_series()

    return result


@overload
def int_ranges(
    start: IntoExpr,
    end: IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def int_ranges(
    start: IntoExpr,
    end: IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def int_ranges(
    start: IntoExpr,
    end: IntoExpr,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: bool,
) -> Expr | Series:
    ...


def int_ranges(
    start: IntoExpr,
    end: IntoExpr,
    step: int = 1,
    *,
    dtype: PolarsIntegerType = Int64,
    eager: bool = False,
) -> Expr | Series:
    """
    Generate a range of integers for each row of the input columns.

    Parameters
    ----------
    start
        Lower bound of the range (inclusive).
    end
        Upper bound of the range (exclusive).
    step
        Step size of the range.
    dtype
        Integer data type of the ranges. Defaults to ``Int64``.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.

    Returns
    -------
    Column of data type ``List(dtype)``.

    See Also
    --------
    int_range : Generate a single range of integers.

    Examples
    --------
    >>> df = pl.DataFrame({"start": [1, -1], "end": [3, 2]})
    >>> df.with_columns(pl.int_ranges("start", "end"))
    shape: (2, 3)
    ┌───────┬─────┬────────────┐
    │ start ┆ end ┆ int_range  │
    │ ---   ┆ --- ┆ ---        │
    │ i64   ┆ i64 ┆ list[i64]  │
    ╞═══════╪═════╪════════════╡
    │ 1     ┆ 3   ┆ [1, 2]     │
    │ -1    ┆ 2   ┆ [-1, 0, 1] │
    └───────┴─────┴────────────┘

    """
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.int_ranges(start, end, step, dtype))

    if eager:
        return F.select(result).to_series()

    return result


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
        Set the time unit. Only takes effect if output is of ``Datetime`` type.
    time_zone:
        Optional timezone. Only takes effect if output is of ``Datetime`` type.
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

    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "")

    time_unit_: TimeUnit | None
    if time_unit is not None:
        time_unit_ = time_unit
    elif "ns" in interval:
        time_unit_ = "ns"
    else:
        time_unit_ = None

    start_pl = parse_as_expression(start)
    end_pl = parse_as_expression(end)
    dt_range = wrap_expr(
        plr.date_range_lazy(start_pl, end_pl, interval, closed, time_unit_, time_zone)
    )
    if name is not None:
        dt_range = dt_range.alias(name)

    if (
        not eager
        or isinstance(start_pl, (str, pl.Expr))
        or isinstance(end_pl, (str, pl.Expr))
    ):
        return dt_range
    res = F.select(dt_range).to_series().explode().set_sorted()
    return res


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
