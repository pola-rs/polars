from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, overload

from polars import functions as F
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr
from polars.functions.range._utils import parse_interval_argument

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars._plr as plr

if TYPE_CHECKING:
    from datetime import date, datetime, timedelta
    from typing import Literal

    from polars import Expr, Series
    from polars._typing import ClosedInterval, IntoExprColumn, TimeUnit


@overload
def datetime_range(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[False] = ...,
) -> Expr: ...


@overload
def datetime_range(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def datetime_range(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool = False,
) -> Series | Expr: ...


def datetime_range(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
) -> Series | Expr:
    """
    Generate a datetime range.

    A datetime range may be defined by any three of 'start', 'end', 'interval', and
    'num_samples'. If only two of `start`, `end`, and `num_samples` are provided, the
    interval defaults to 1 day.

    Parameters
    ----------
    start
        Lower bound of the datetime range.
    end
        Upper bound of the datetime range.
    interval
        Interval of the range periods, specified as a Python `timedelta` object
        or using the Polars duration string language (see "Notes" section below).
    num_samples
        Number of periods in the date range. This corresponds to the number of points in
        the output array, and is thus one more than the number of intervals.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    time_unit : {None, 'ns', 'us', 'ms'}
        Time unit of the resulting `Datetime` data type.
    time_zone
        Time zone of the resulting `Datetime` data type.
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Datetime`.

    See Also
    --------
    datetime_ranges
    date_range

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

    By "calendar day", we mean the corresponding time on the next day (which may
    not be 24 hours, due to daylight savings). Similarly for "calendar week",
    "calendar month", "calendar quarter", and "calendar year".

    Examples
    --------
    Using Polars duration string to specify the interval:

    >>> from datetime import datetime
    >>> pl.datetime_range(
    ...     datetime(2022, 1, 1), datetime(2022, 3, 1), "1mo", eager=True
    ... ).alias("datetime")
    shape: (3,)
    Series: 'datetime' [datetime[μs]]
    [
        2022-01-01 00:00:00
        2022-02-01 00:00:00
        2022-03-01 00:00:00
    ]

    Using `timedelta` object to specify the interval:

    >>> from datetime import date, timedelta
    >>> pl.datetime_range(
    ...     date(1985, 1, 1),
    ...     date(1985, 1, 10),
    ...     timedelta(days=1, hours=12),
    ...     time_unit="ms",
    ...     eager=True,
    ... ).alias("datetime")
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
    ... ).alias("datetime")
    shape: (3,)
    Series: 'datetime' [datetime[μs, America/New_York]]
    [
        2022-01-01 00:00:00 EST
        2022-02-01 00:00:00 EST
        2022-03-01 00:00:00 EST
    ]

    Using 'num_samples' to specify the number of periods:

    >>> pl.datetime_range(
    ...     start=date(1985, 1, 1),
    ...     end=date(1985, 1, 10),
    ...     num_samples=5,
    ...     eager=True,
    ... ).alias("date")
    shape: (5,)
    Series: 'date' [datetime[μs]]
    [
            1985-01-01 00:00:00
            1985-01-03 06:00:00
            1985-01-05 12:00:00
            1985-01-07 18:00:00
            1985-01-10 00:00:00
    ]
    >>> pl.datetime_range(
    ...     start=date(1985, 1, 1),
    ...     interval="3d",
    ...     num_samples=4,
    ...     eager=True,
    ... ).alias("date")
    shape: (4,)
    Series: 'date' [datetime[μs]]
    [
            1985-01-01 00:00:00
            1985-01-04 00:00:00
            1985-01-07 00:00:00
            1985-01-10 00:00:00
    ]

    Omit `eager=True` if you want to use `datetime_range` as an expression:

    >>> df = pl.DataFrame(
    ...     {
    ...         "date": [
    ...             date(2024, 1, 1),
    ...             date(2024, 1, 2),
    ...             date(2024, 1, 1),
    ...             date(2024, 1, 3),
    ...         ],
    ...         "key": ["one", "one", "two", "two"],
    ...     }
    ... )
    >>> result = (
    ...     df.group_by("key")
    ...     .agg(pl.datetime_range(pl.col("date").min(), pl.col("date").max()))
    ...     .sort("key")
    ... )
    >>> with pl.Config(fmt_str_lengths=70):
    ...     print(result)
    shape: (2, 2)
    ┌─────┬─────────────────────────────────────────────────────────────────┐
    │ key ┆ date                                                            │
    │ --- ┆ ---                                                             │
    │ str ┆ list[datetime[μs]]                                              │
    ╞═════╪═════════════════════════════════════════════════════════════════╡
    │ one ┆ [2024-01-01 00:00:00, 2024-01-02 00:00:00]                      │
    │ two ┆ [2024-01-01 00:00:00, 2024-01-02 00:00:00, 2024-01-03 00:00:00] │
    └─────┴─────────────────────────────────────────────────────────────────┘
    """
    if interval is None and (num_samples is None or start is None or end is None):
        interval = "1d"
    interval = None if interval is None else parse_interval_argument(interval)
    if time_unit is None and interval is not None and "ns" in interval:
        time_unit = "ns"
    start_expr = None if start is None else parse_into_expression(start)
    end_expr = None if end is None else parse_into_expression(end)
    samples_expr = None if num_samples is None else parse_into_expression(num_samples)

    result = wrap_expr(
        plr.datetime_range(
            start_expr,
            end_expr,
            interval,
            samples_expr,
            closed,
            time_unit,
            time_zone,
        )
    )

    if eager:
        return F.select(result).to_series()

    return result


@overload
def datetime_ranges(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[False] = ...,
) -> Expr: ...


@overload
def datetime_ranges(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def datetime_ranges(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
    eager: bool = False,
) -> Series | Expr: ...


def datetime_ranges(
    start: datetime | date | IntoExprColumn | None = None,
    end: datetime | date | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    num_samples: int | IntoExprColumn | None = None,
    *,
    closed: ClosedInterval = "both",
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
    eager: bool = False,
) -> Series | Expr:
    """
    Create a column of datetime ranges.

    Each datetime range may be defined by any three of 'start', 'end', 'interval', and
    'num_samples'. If only two of `start`, `end`, and `num_samples` are provided, the
    interval defaults to 1 day.

    Parameters
    ----------
    start
        Lower bound of the datetime range.
    end
        Upper bound of the datetime range.
    interval
        Interval of the range periods, specified as a Python `timedelta` object
        or using the Polars duration string language (see "Notes" section below).
    num_samples
        Number of periods in the date range. This corresponds to the number of points in
        the output array, and is thus one more than the number of intervals.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    time_unit : {None, 'ns', 'us', 'ms'}
        Time unit of the resulting `Datetime` data type.
    time_zone
        Time zone of the resulting `Datetime` data type.
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

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

    By "calendar day", we mean the corresponding time on the next day (which may
    not be 24 hours, due to daylight savings). Similarly for "calendar week",
    "calendar month", "calendar quarter", and "calendar year".

    Returns
    -------
    Expr or Series
        Column of data type `List(Datetime)`.

    See Also
    --------
    datetime_range
    date_ranges

    Examples
    --------
    >>> from datetime import datetime
    >>> df = pl.DataFrame(
    ...     {
    ...         "start": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
    ...         "end": datetime(2022, 1, 3),
    ...     }
    ... )
    >>> with pl.Config(fmt_str_lengths=100):
    ...     df.select(datetime_range=pl.datetime_ranges("start", "end"))
    shape: (2, 1)
    ┌─────────────────────────────────────────────────────────────────┐
    │ datetime_range                                                  │
    │ ---                                                             │
    │ list[datetime[μs]]                                              │
    ╞═════════════════════════════════════════════════════════════════╡
    │ [2022-01-01 00:00:00, 2022-01-02 00:00:00, 2022-01-03 00:00:00] │
    │ [2022-01-02 00:00:00, 2022-01-03 00:00:00]                      │
    └─────────────────────────────────────────────────────────────────┘
    """
    if interval is None and (num_samples is None or start is None or end is None):
        interval = "1d"
    interval = None if interval is None else parse_interval_argument(interval)
    if time_unit is None and interval is not None and "ns" in interval:
        time_unit = "ns"
    start_expr = None if start is None else parse_into_expression(start)
    end_expr = None if end is None else parse_into_expression(end)
    samples_expr = None if num_samples is None else parse_into_expression(num_samples)

    result = wrap_expr(
        plr.datetime_ranges(
            start_expr,
            end_expr,
            interval,
            samples_expr,
            closed,
            time_unit,
            time_zone,
        )
    )

    if eager:
        return F.select(result).to_series()

    return result
