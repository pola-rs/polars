from __future__ import annotations

import contextlib
from datetime import time, timedelta
from typing import TYPE_CHECKING, overload

from polars import functions as F
from polars.datatypes import Int64
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.convert import (
    _timedelta_to_pl_duration,
)
from polars.utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)

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


@deprecate_renamed_parameter("low", "start", version="0.18.0")
@deprecate_renamed_parameter("high", "end", version="0.18.0")
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
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

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
        issue_deprecation_warning(
            " `arange` has been replaced by two new functions:"
            " `int_range` for generating a single range,"
            " and `int_ranges` for generating a list column with multiple ranges."
            " `arange` will remain available as an alias for `int_range`, which means its behaviour will change."
            " To silence this warning, use either of the new functions.",
            version="0.18.5",
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
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Int64`.

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
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

    Returns
    -------
    Expr or Series
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
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
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
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
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
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool,
    name: str | None = ...,
) -> Series | Expr:
    ...


@deprecate_renamed_parameter("low", "start", version="0.18.0")
@deprecate_renamed_parameter("high", "end", version="0.18.0")
def date_range(
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
    interval: str | timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
    name: str | None = None,
) -> Series | Expr:
    """
    Generate a date range.

    Parameters
    ----------
    start
        Lower bound of the date range.
    end
        Upper bound of the date range.
    interval
        Interval of the range periods, specified as a Python ``timedelta`` object
        or a Polars duration string like ``1h30m25s``.

        Append ``_saturating`` to the interval string to restrict resulting invalid
        dates to valid ranges.

        It is common to attempt to create a month-end date series by using the "1mo"
        offset string with a start date at the end of the month. This will not produce
        the desired results. See Note #2 below for further information.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    time_unit : {None, 'ns', 'us', 'ms'}
        Time unit of the resulting ``Datetime`` data type.
        Only takes effect if the output column is of type ``Datetime``.
    time_zone
        Time zone of the resulting ``Datetime`` data type.
        Only takes effect if the output column is of type ``Datetime``.
    eager
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.
    name
        Name of the output column.

        .. deprecated:: 0.18.0
            This argument is deprecated. Use the ``alias`` method instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Date` or :class:`Datetime`.

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

    Examples
    --------
    Using Polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)
    shape: (3,)
    Series: 'date' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]

    Using ``timedelta`` object to specify the interval:

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

    Specifying a time zone:

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

    >>> pl.date_range(
    ...     datetime(2022, 1, 1), datetime(2022, 3, 1), "1mo", eager=True
    ... ).dt.month_end()
    shape: (3,)
    Series: 'date' [datetime[μs]]
    [
        2022-01-31 00:00:00
        2022-02-28 00:00:00
        2022-03-31 00:00:00
    ]

    """
    if name is not None:
        issue_deprecation_warning(
            "the `name` argument is deprecated. Use the `alias` method instead.",
            version="0.18.0",
        )

    interval = _parse_interval_argument(interval)
    if time_unit is None and "ns" in interval:
        time_unit = "ns"

    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(
        plr.date_range(start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone)
    )

    if name is not None:
        result = result.alias(name)

    if eager:
        s = F.select(result).to_series()
        if s.len() == 1:
            s = s.explode().set_sorted()
        else:
            _warn_for_deprecation_date_range()
        return s

    _warn_for_deprecation_date_range()

    return result


@overload
def date_ranges(
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def date_ranges(
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def date_ranges(
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool,
) -> Series | Expr:
    ...


def date_ranges(
    start: date | datetime | IntoExpr,
    end: date | datetime | IntoExpr,
    interval: str | timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
) -> Series | Expr:
    """
    Create a column of date ranges.

    Parameters
    ----------
    start
        Lower bound of the date range.
    end
        Upper bound of the date range.
    interval
        Interval of the range periods, specified as a Python ``timedelta`` object
        or a Polars duration string like ``1h30m25s``.

        Append ``_saturating`` to the interval string to restrict resulting invalid
        dates to valid ranges.

        It is common to attempt to create a month-end date series by using the "1mo"
        offset string with a start date at the end of the month. This will not produce
        the desired results. See Note #2 below for further information.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    time_unit : {None, 'ns', 'us', 'ms'}
        Time unit of the resulting ``Datetime`` data type.
        Only takes effect if the output column is of type ``Datetime``.
    time_zone
        Time zone of the resulting ``Datetime`` data type.
        Only takes effect if the output column is of type ``Datetime``.
    eager
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type ``List(Date)`` or ``List(Datetime)``.

    Examples
    --------
    >>> from datetime import date
    >>> df = pl.DataFrame(
    ...     {
    ...         "start": [date(2022, 1, 1), date(2022, 1, 2)],
    ...         "end": date(2022, 1, 3),
    ...     }
    ... )
    >>> df.with_columns(pl.date_ranges("start", "end"))
    shape: (2, 3)
    ┌────────────┬────────────┬───────────────────────────────────┐
    │ start      ┆ end        ┆ date_range                        │
    │ ---        ┆ ---        ┆ ---                               │
    │ date       ┆ date       ┆ list[date]                        │
    ╞════════════╪════════════╪═══════════════════════════════════╡
    │ 2022-01-01 ┆ 2022-01-03 ┆ [2022-01-01, 2022-01-02, 2022-01… │
    │ 2022-01-02 ┆ 2022-01-03 ┆ [2022-01-02, 2022-01-03]          │
    └────────────┴────────────┴───────────────────────────────────┘

    """
    interval = _parse_interval_argument(interval)
    if time_unit is None and "ns" in interval:
        time_unit = "ns"

    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)

    result = wrap_expr(
        plr.date_ranges(
            start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone
        )
    )

    if eager:
        return F.select(result).to_series()

    return result


@overload
def time_range(
    start: time | IntoExpr | None = ...,
    end: time | IntoExpr | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
    name: str | None = ...,
) -> Expr:
    ...


@overload
def time_range(
    start: time | IntoExpr | None = ...,
    end: time | IntoExpr | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
    name: str | None = ...,
) -> Series:
    ...


@overload
def time_range(
    start: time | IntoExpr | None = ...,
    end: time | IntoExpr | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
    name: str | None = ...,
) -> Series | Expr:
    ...


def time_range(
    start: time | IntoExpr | None = None,
    end: time | IntoExpr | None = None,
    interval: str | timedelta = "1h",
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
    name: str | None = None,
) -> Series | Expr:
    """
    Generate a time range.

    Parameters
    ----------
    start
        Lower bound of the time range.
        If omitted, defaults to ``time(0,0,0,0)``.
    end
        Upper bound of the time range.
        If omitted, defaults to ``time(23,59,59,999999)``.
    interval
        Interval of the range periods, specified as a Python ``timedelta`` object
        or a Polars duration string like ``1h30m25s``.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    eager
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.
    name
        Name of the output column.

        .. deprecated:: 0.18.0
            This argument is deprecated. Use the ``alias`` method instead.

    Returns
    -------
    Expr or Series
        Column of data type `:class:Time`.

    See Also
    --------
    time_ranges : Create a column of time ranges.

    Examples
    --------
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

    """
    if name is not None:
        issue_deprecation_warning(
            "the `name` argument is deprecated. Use the `alias` method instead.",
            version="0.18.0",
        )

    interval = _parse_interval_argument(interval)
    for unit in ("y", "mo", "w", "d"):
        if unit in interval:
            raise ValueError(f"invalid interval unit for time_range: found {unit!r}")

    if start is None:
        start = time(0, 0, 0)
    if end is None:
        end = time(23, 59, 59, 999999)

    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)

    result = wrap_expr(plr.time_range(start_pyexpr, end_pyexpr, interval, closed))

    if name is not None:
        result = result.alias(name)

    if eager:
        s = F.select(result).to_series()
        if s.len() == 1:
            s = s.explode().set_sorted()
        else:
            _warn_for_deprecation_time_range()
        return s

    _warn_for_deprecation_time_range()

    return result


@overload
def time_ranges(
    start: time | IntoExpr | None = ...,
    end: time | IntoExpr | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def time_ranges(
    start: time | IntoExpr | None = ...,
    end: time | IntoExpr | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def time_ranges(
    start: time | IntoExpr | None = ...,
    end: time | IntoExpr | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
) -> Series | Expr:
    ...


def time_ranges(
    start: time | IntoExpr | None = None,
    end: time | IntoExpr | None = None,
    interval: str | timedelta = "1h",
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
) -> Series | Expr:
    """
    Create a column of time ranges.

    Parameters
    ----------
    start
        Lower bound of the time range.
        If omitted, defaults to ``time(0, 0, 0, 0)``.
    end
        Upper bound of the time range.
        If omitted, defaults to ``time(23, 59, 59, 999999)``.
    interval
        Interval of the range periods, specified as a Python ``timedelta`` object
        or a Polars duration string like ``1h30m25s``.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    eager
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type ``List(Time)``.

    See Also
    --------
    time_range : Generate a single time range.

    Examples
    --------
    >>> from datetime import time
    >>> df = pl.DataFrame(
    ...     {
    ...         "start": [time(9, 0), time(10, 0)],
    ...         "end": time(11, 0),
    ...     }
    ... )
    >>> df.with_columns(pl.time_ranges("start", "end"))
    shape: (2, 3)
    ┌──────────┬──────────┬────────────────────────────────┐
    │ start    ┆ end      ┆ time_range                     │
    │ ---      ┆ ---      ┆ ---                            │
    │ time     ┆ time     ┆ list[time]                     │
    ╞══════════╪══════════╪════════════════════════════════╡
    │ 09:00:00 ┆ 11:00:00 ┆ [09:00:00, 10:00:00, 11:00:00] │
    │ 10:00:00 ┆ 11:00:00 ┆ [10:00:00, 11:00:00]           │
    └──────────┴──────────┴────────────────────────────────┘

    """
    interval = _parse_interval_argument(interval)
    for unit in ("y", "mo", "w", "d"):
        if unit in interval:
            raise ValueError(f"invalid interval unit for time_range: found {unit!r}")

    if start is None:
        start = time(0, 0, 0)
    if end is None:
        end = time(23, 59, 59, 999999)

    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)

    result = wrap_expr(plr.time_ranges(start_pyexpr, end_pyexpr, interval, closed))

    if eager:
        return F.select(result).to_series()

    return result


def _parse_interval_argument(interval: str | timedelta) -> str:
    """Parse the interval argument as a Polars duration string."""
    if isinstance(interval, timedelta):
        return _timedelta_to_pl_duration(interval)

    if " " in interval:
        interval = interval.replace(" ", "")
    return interval.lower()


def _warn_for_deprecation_date_range() -> None:
    issue_deprecation_warning(
        "behavior of `date_range` will change in a future version."
        " The result will be a single range of type Date or Datetime instead of List."
        " Use the new `date_ranges` function to retain the old functionality and silence this warning.",
        version="0.18.9",
    )


def _warn_for_deprecation_time_range() -> None:
    issue_deprecation_warning(
        "behavior of `time_range` will change in a future version."
        " The result will be a single range of type Time instead of List."
        " Use the new `date_ranges` function to retain the old functionality and silence this warning.",
        version="0.18.9",
    )
