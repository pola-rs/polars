from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, overload

from polars import functions as F
from polars.functions.range._utils import parse_interval_argument
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from datetime import date, datetime, timedelta
    from typing import Literal

    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, IntoExpr, TimeUnit


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

    interval = parse_interval_argument(interval)
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
        return F.select(result).to_series()

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
    interval = parse_interval_argument(interval)
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
