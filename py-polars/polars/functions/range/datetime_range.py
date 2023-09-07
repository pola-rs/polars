from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, overload

from polars import functions as F
from polars.functions.range._utils import parse_interval_argument
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from typing import Literal

    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, IntoExpr, TimeUnit


@overload
def datetime_range(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def datetime_range(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def datetime_range(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool,
) -> Series | Expr:
    ...


def datetime_range(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
) -> Series | Expr:
    """
    Generate a datetime range.

    Parameters
    ----------
    start
        Lower bound of the datetime range.
    end
        Upper bound of the datetime range.
    interval
        Interval of the range periods, specified as a Python ``timedelta`` object
        or a Polars duration string like ``1h30m25s``.

        Append ``_saturating`` to the interval string to restrict resulting invalid
        dates to valid ranges.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    time_unit : {None, 'ns', 'us', 'ms'}
        Time unit of the resulting ``Datetime`` data type.
    time_zone
        Time zone of the resulting ``Datetime`` data type.
    eager
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Datetime`.

    Examples
    --------
    Using Polars duration string to specify the interval:

    >>> from datetime import datetime
    >>> pl.datetime_range(datetime(2022, 1, 1), datetime(2022, 3, 1), "1mo", eager=True)
    shape: (3,)
    Series: 'datetime' [datetime[μs]]
    [
        2022-01-01 00:00:00
        2022-02-01 00:00:00
        2022-03-01 00:00:00
    ]

    Using ``timedelta`` object to specify the interval:

    >>> from datetime import date, timedelta
    >>> pl.datetime_range(
    ...     date(1985, 1, 1),
    ...     date(1985, 1, 10),
    ...     timedelta(days=1, hours=12),
    ...     time_unit="ms",
    ...     eager=True,
    ... )
    shape: (7,)
    Series: 'datetime' [datetime[ms]]
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

    >>> pl.datetime_range(
    ...     datetime(2022, 1, 1),
    ...     datetime(2022, 3, 1),
    ...     "1mo",
    ...     time_zone="America/New_York",
    ...     eager=True,
    ... )
    shape: (3,)
    Series: 'datetime' [datetime[μs, America/New_York]]
    [
        2022-01-01 00:00:00 EST
        2022-02-01 00:00:00 EST
        2022-03-01 00:00:00 EST
    ]

    """
    interval = parse_interval_argument(interval)
    if time_unit is None and "ns" in interval:
        time_unit = "ns"

    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(
        plr.datetime_range(
            start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone
        )
    )

    if eager:
        return F.select(result).to_series()

    return result


@overload
def datetime_ranges(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def datetime_ranges(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def datetime_ranges(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool,
) -> Series | Expr:
    ...


def datetime_ranges(
    start: datetime | IntoExpr,
    end: datetime | IntoExpr,
    interval: str | timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
) -> Series | Expr:
    """
    Create a column of datetime ranges.

    Parameters
    ----------
    start
        Lower bound of the datetime range.
    end
        Upper bound of the datetime range.
    interval
        Interval of the range periods, specified as a Python ``timedelta`` object
        or a Polars duration string like ``1h30m25s``.

        Append ``_saturating`` to the interval string to restrict resulting invalid
        dates to valid ranges.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    time_unit : {None, 'ns', 'us', 'ms'}
        Time unit of the resulting ``Datetime`` data type.
    time_zone
        Time zone of the resulting ``Datetime`` data type.
    eager
        Evaluate immediately and return a ``Series``.
        If set to ``False`` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type ``List(Datetime)``.

    """
    interval = parse_interval_argument(interval)
    if time_unit is None and "ns" in interval:
        time_unit = "ns"

    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)

    result = wrap_expr(
        plr.datetime_ranges(
            start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone
        )
    )

    if eager:
        return F.select(result).to_series()

    return result
