from __future__ import annotations

import contextlib
from datetime import datetime
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
    from datetime import date, timedelta
    from typing import Literal

    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, IntoExprColumn, TimeUnit


@overload
def date_range(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
        or using the Polars duration string language (see "Notes" section below).

        To create a month-end date series, combine with :meth:`Expr.dt.month_end` (see
        "Examples" section below).
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

       .. deprecated:: 0.19.3
           In a future version of Polars, `date_range` will always return a `Date`.
           Please use :func:`datetime_range` if you want a `Datetime` instead.

    2) `interval` is created according to the following string language:

       - 1ns   (1 nanosecond)
       - 1us   (1 microsecond)
       - 1ms   (1 millisecond)
       - 1s    (1 second)
       - 1m    (1 minute)
       - 1h    (1 hour)
       - 1d    (1 calendar day)
       - 1w    (1 calendar week)
       - 1mo   (1 calendar month)
       - 1q    (1 calendar quarter)
       - 1y    (1 calendar year)

       Or combine them:
       "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

       Suffix with `"_saturating"` to indicate that dates too large for
       their month should saturate at the largest date (e.g. 2022-02-29 -> 2022-02-28)
       instead of erroring.

       By "calendar day", we mean the corresponding time on the next day (which may
       not be 24 hours, due to daylight savings). Similarly for "calendar week",
       "calendar month", "calendar quarter", and "calendar year".

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

    >>> from datetime import timedelta
    >>> pl.date_range(
    ...     date(1985, 1, 1),
    ...     date(1985, 1, 10),
    ...     timedelta(days=2),
    ...     eager=True,
    ... )
    shape: (5,)
    Series: 'date' [date]
    [
        1985-01-01
        1985-01-03
        1985-01-05
        1985-01-07
        1985-01-09
    ]

    Combine with :meth:`Expr.dt.month_end` to get the last day of the month:

    >>> pl.date_range(
    ...     date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True
    ... ).dt.month_end()
    shape: (3,)
    Series: 'date' [date]
    [
        2022-01-31
        2022-02-28
        2022-03-31
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

    _warn_for_deprecated_date_range_use(start, end, interval, time_unit, time_zone)

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
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool,
) -> Series | Expr:
    ...


def date_ranges(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
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
        or using the Polars duration string language (see "Notes" section below).
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

    Notes
    -----
    `interval` is created according to the following string language:

    - 1ns   (1 nanosecond)
    - 1us   (1 microsecond)
    - 1ms   (1 millisecond)
    - 1s    (1 second)
    - 1m    (1 minute)
    - 1h    (1 hour)
    - 1d    (1 calendar day)
    - 1w    (1 calendar week)
    - 1mo   (1 calendar month)
    - 1q    (1 calendar quarter)
    - 1y    (1 calendar year)

    Or combine them:
    "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

    Suffix with `"_saturating"` to indicate that dates too large for
    their month should saturate at the largest date (e.g. 2022-02-29 -> 2022-02-28)
    instead of erroring.

    By "calendar day", we mean the corresponding time on the next day (which may
    not be 24 hours, due to daylight savings). Similarly for "calendar week",
    "calendar month", "calendar quarter", and "calendar year".

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

    _warn_for_deprecated_date_range_use(start, end, interval, time_unit, time_zone)

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


def _warn_for_deprecated_date_range_use(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str,
    time_unit: TimeUnit | None,
    time_zone: str | None,
) -> None:
    # This check is not foolproof, but should catch most cases
    if (
        isinstance(start, datetime)
        or isinstance(end, datetime)
        or "s" in interval
        or "h" in interval
        or ("m" in interval and "mo" not in interval)
        or time_unit is not None
        or time_zone is not None
    ):
        issue_deprecation_warning(
            "Creating Datetime ranges using `date_range(s)` is deprecated."
            " Use `datetime_range(s)` instead.",
            version="0.19.3",
        )
