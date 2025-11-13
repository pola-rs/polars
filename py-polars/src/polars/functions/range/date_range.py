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
    from polars._typing import ClosedInterval, IntoExprColumn


@overload
def date_range(
    start: date | datetime | IntoExprColumn | None = ...,
    end: date | datetime | IntoExprColumn | None = ...,
    interval: str | timedelta | None = ...,
    *,
    num_samples: int | None = ...,
    closed: ClosedInterval = ...,
    eager: Literal[False],
) -> Expr: ...


@overload
def date_range(
    start: date | datetime | IntoExprColumn | None = ...,
    end: date | datetime | IntoExprColumn | None = ...,
    interval: str | timedelta | None = ...,
    *,
    num_samples: int | None = ...,
    closed: ClosedInterval = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def date_range(
    start: date | datetime | IntoExprColumn | None = ...,
    end: date | datetime | IntoExprColumn | None = ...,
    interval: str | timedelta | None = ...,
    *,
    num_samples: int | None = ...,
    closed: ClosedInterval = ...,
    eager: bool = ...,
) -> Series | Expr: ...


def date_range(
    start: date | datetime | IntoExprColumn | None = None,
    end: date | datetime | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    *,
    num_samples: int | None = None,
    closed: ClosedInterval = "both",
    eager: bool = False,
) -> Series | Expr:
    """
    Generate a date range.

    Parameters
    ----------
    start
        First value of the date range.
    end
        Last value of the date range.
    interval
        Interval of the range periods, specified as a Python `timedelta` object
        or using the Polars duration string language (see "Notes" section below).
        Must consist of full days.
    num_samples
        Number of samples in the output.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Date`.

    See Also
    --------
    date_ranges
    datetime_range
    datetime_ranges

    Notes
    -----
    * Date ranges are defined by a `start`, `end`, an `interval`, and a corresponding
      number of samples (`num_samples`). Any date/datetime range can be defined from
      three of these attributes. The `closed` parameter affects whether the endpoints
      are included in the result.

    * Most ranges are defined in a straight-forward manner, with `start` prior to `end`,
      a positive interval which creates an ascending date range, and a resulting number
      of samples. In the following example, we can create a range from January 1st
      through June 6th, separated by one month and one day. We set the interval to be
      right-closed, which means the left-most point is not included:

      .. image:: /_static/figures/date_range/date_range.svg

      We can generate this same range using different combinations of parameters:

      >>> start = pl.date(2025, 1, 1)
      >>> end = pl.date(2025, 6, 6)
      >>> pl.date_range(
      ...     start=start, end=end, interval="1mo1d", closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, end=end, num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, interval="1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     end=end, interval="1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP

      *Descending* date ranges have `end` prior to `start` and have a negative interval.
      For example, the following range uses the same limits as the prior
      ascending example (reversed). Again, the `closed` parameter still determines which
      of these end-points are included in the result--in this case, the left-most value
      of June 6th is not included:

      .. image:: /_static/figures/date_range/date_range_descending.svg

      And once again, we can generate this range using multiple combinations of
      parameters:

      >>> start = pl.date(2025, 6, 6)
      >>> end = pl.date(2025, 1, 1)
      >>> pl.date_range(
      ...     start=start, end=end, interval="-1mo1d", closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, end=end, num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, interval="-1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     end=end, interval="-1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP

    * `interval` is created according to the following string language:

      - 1d    (1 calendar day)
      - 1w    (1 calendar week)
      - 1mo   (1 calendar month)
      - 1q    (1 calendar quarter)
      - 1y    (1 calendar year)

      Or combine them:
      "1w2d" # 1 week, 2 days

      By "calendar day", we mean the corresponding time on the next day (which may
      not be 24 hours, due to daylight savings). Similarly for "calendar week",
      "calendar month", "calendar quarter", and "calendar year".

    Examples
    --------
    Using Polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(
    ...     start=date(2022, 1, 1), end=date(2022, 3, 1), interval="1mo", eager=True
    ... ).alias("date")
    shape: (3,)
    Series: 'date' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]

    Using `timedelta` object to specify the interval:

    >>> from datetime import timedelta
    >>> pl.date_range(
    ...     start=date(1985, 1, 1),
    ...     end=date(1985, 1, 10),
    ...     interval=timedelta(days=2),
    ...     eager=True,
    ... ).alias("date")
    shape: (5,)
    Series: 'date' [date]
    [
        1985-01-01
        1985-01-03
        1985-01-05
        1985-01-07
        1985-01-09
    ]

    Using 'num_samples' to specify the number of periods:

    >>> pl.date_range(
    ...     start=date(1985, 1, 1),
    ...     end=date(1985, 1, 10),
    ...     num_samples=5,
    ...     eager=True,
    ... ).alias("date")
    shape: (5,)
    Series: 'date' [date]
    [
            1985-01-01
            1985-01-03
            1985-01-05
            1985-01-07
            1985-01-10
    ]
    >>> pl.date_range(
    ...     start=date(1985, 1, 1),
    ...     interval="3d",
    ...     num_samples=4,
    ...     eager=True,
    ... ).alias("date")
    shape: (4,)
    Series: 'date' [date]
    [
            1985-01-01
            1985-01-04
            1985-01-07
            1985-01-10
    ]

    Omit `eager=True` if you want to use `date_range` as an expression:

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
    ...     .agg(pl.date_range(pl.col("date").min(), pl.col("date").max()))
    ...     .sort("key")
    ... )
    >>> with pl.Config(fmt_str_lengths=50):
    ...     print(result)
    shape: (2, 2)
    ┌─────┬──────────────────────────────────────┐
    │ key ┆ date                                 │
    │ --- ┆ ---                                  │
    │ str ┆ list[date]                           │
    ╞═════╪══════════════════════════════════════╡
    │ one ┆ [2024-01-01, 2024-01-02]             │
    │ two ┆ [2024-01-01, 2024-01-02, 2024-01-03] │
    └─────┴──────────────────────────────────────┘
    """
    if interval is None and (num_samples is None or start is None or end is None):
        interval = "1d"
    interval = None if interval is None else parse_interval_argument(interval)
    start_expr = None if start is None else parse_into_expression(start)
    end_expr = None if end is None else parse_into_expression(end)
    samples_expr = None if num_samples is None else parse_into_expression(num_samples)

    result = wrap_expr(
        plr.date_range(start_expr, end_expr, interval, samples_expr, closed)
    )

    if eager:
        return F.select(result).to_series()

    return result


@overload
def date_ranges(
    start: date | datetime | IntoExprColumn | None = ...,
    end: date | datetime | IntoExprColumn | None = ...,
    interval: str | timedelta = ...,
    *,
    num_samples: int | IntoExprColumn | None = ...,
    closed: ClosedInterval = ...,
    eager: Literal[False],
) -> Expr: ...


@overload
def date_ranges(
    start: date | datetime | IntoExprColumn | None = ...,
    end: date | datetime | IntoExprColumn | None = ...,
    interval: str | timedelta | None = ...,
    *,
    num_samples: int | IntoExprColumn | None = ...,
    closed: ClosedInterval = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def date_ranges(
    start: date | datetime | IntoExprColumn | None = ...,
    end: date | datetime | IntoExprColumn | None = ...,
    interval: str | timedelta | None = ...,
    *,
    num_samples: int | IntoExprColumn | None = ...,
    closed: ClosedInterval = ...,
    eager: bool = ...,
) -> Series | Expr: ...


def date_ranges(
    start: date | datetime | IntoExprColumn | None = None,
    end: date | datetime | IntoExprColumn | None = None,
    interval: str | timedelta | None = None,
    *,
    num_samples: int | IntoExprColumn | None = None,
    closed: ClosedInterval = "both",
    eager: bool = False,
) -> Series | Expr:
    """
    Create a column of date ranges.

    Each date range may be defined by any three of 'start', 'end', 'interval', and
    'num_samples'. If only two of `start`, `end`, and `num_samples` are provided, the
    interval defaults to 1 day.

    Parameters
    ----------
    start
        First value of the date range.
    end
        Last value of the date range.
    interval
        Interval of the range periods, specified as a Python `timedelta` object
        or using the Polars duration string language (see "Notes" section below).
        Must consist of full days.
    num_samples
        Number of samples in the output.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type `List(Date)`.

    See Also
    --------
    date_range
    datetime_range
    datetime_ranges

    Notes
    -----
    * Date ranges are defined by a `start`, `end`, an `interval`, and a corresponding
      number of samples (`num_samples`). Any date/datetime range can be defined from
      three of these attributes. The `closed` parameter affects whether the endpoints
      are included in the result.

    * Most ranges are defined in a straight-forward manner, with `start` prior to `end`,
      a positive interval which creates an ascending date range, and a resulting number
      of samples. In the following example, we can create a range from January 1st
      through June 6th, separated by one month and one day. We set the interval to be
      right-closed, which means the left-most point is not included:

      .. image:: /_static/figures/date_range/date_range.svg

      We can generate this same range using different combinations of parameters:

      >>> start = pl.date(2025, 1, 1)
      >>> end = pl.date(2025, 6, 6)
      >>> pl.date_range(
      ...     start=start, end=end, interval="1mo1d", closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, end=end, num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, interval="1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     end=end, interval="1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP

      *Descending* date ranges have `end` prior to `start` and have a negative interval.
      For example, the following range uses the same limits as the prior
      ascending example (reversed). Again, the `closed` parameter still determines which
      of these end-points are included in the result--in this case, the left-most value
      of June 6th is not included:

      .. image:: /_static/figures/date_range/date_range_descending.svg

      And once again, we can generate this range using multiple combinations of
      parameters:

      >>> start = pl.date(2025, 6, 6)
      >>> end = pl.date(2025, 1, 1)
      >>> pl.date_range(
      ...     start=start, end=end, interval="-1mo1d", closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, end=end, num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     start=start, interval="-1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP
      >>> pl.date_range(
      ...     end=end, interval="-1mo1d", num_samples=5, closed="right"
      ... )  # doctest: +SKIP

    * `interval` is created according to the following string language:

      - 1d    (1 calendar day)
      - 1w    (1 calendar week)
      - 1mo   (1 calendar month)
      - 1q    (1 calendar quarter)
      - 1y    (1 calendar year)

      Or combine them:
      "1w2d" # 1 week, 2 days

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
    >>> with pl.Config(fmt_str_lengths=50):
    ...     df.with_columns(date_range=pl.date_ranges("start", "end"))
    shape: (2, 3)
    ┌────────────┬────────────┬──────────────────────────────────────┐
    │ start      ┆ end        ┆ date_range                           │
    │ ---        ┆ ---        ┆ ---                                  │
    │ date       ┆ date       ┆ list[date]                           │
    ╞════════════╪════════════╪══════════════════════════════════════╡
    │ 2022-01-01 ┆ 2022-01-03 ┆ [2022-01-01, 2022-01-02, 2022-01-03] │
    │ 2022-01-02 ┆ 2022-01-03 ┆ [2022-01-02, 2022-01-03]             │
    └────────────┴────────────┴──────────────────────────────────────┘
    """
    if interval is None and (num_samples is None or start is None or end is None):
        interval = "1d"
    interval = None if interval is None else parse_interval_argument(interval)
    start_expr = None if start is None else parse_into_expression(start)
    end_expr = None if end is None else parse_into_expression(end)
    samples_expr = None if num_samples is None else parse_into_expression(num_samples)

    result = wrap_expr(
        plr.date_ranges(start_expr, end_expr, interval, samples_expr, closed)
    )

    if eager:
        return F.select(result).to_series()

    return result
