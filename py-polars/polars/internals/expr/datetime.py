from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import DTYPE_TEMPORAL_UNITS, Date, Int32
from polars.utils import _timedelta_to_pl_duration

if TYPE_CHECKING:
    from polars.internals.type_aliases import EpochTimeUnit, TimeUnit


class ExprDateTimeNameSpace:
    """Namespace for datetime related expressions."""

    _accessor = "dt"

    def __init__(self, expr: pli.Expr):
        self._pyexpr = expr._pyexpr

    def truncate(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> pli.Expr:
        """
        Divide the date/datetime range into buckets.

        Each date/datetime is mapped to the start of its bucket.

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

        Notes
        -----
        The `every` and `offset` argument are created with the
        the following small string formatting language:

        1ns  # 1 nanosecond
        1us  # 1 microsecond
        1ms  # 1 millisecond
        1s   # 1 second
        1m   # 1 minute
        1h   # 1 hour
        1d   # 1 day
        1w   # 1 week
        1mo  # 1 calendar month
        1y   # 1 calendar year

        eg: 3d12h4m25s  # 3 days, 12 hours, 4 minutes, and 25 seconds

        Returns
        -------
        Date/Datetime series

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df = pl.date_range(
        ...     start, stop, timedelta(minutes=225), name="dates"
        ... ).to_frame()
        >>> df
        shape: (7, 1)
        ┌─────────────────────┐
        │ dates               │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 03:45:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 07:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 11:15:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 15:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 18:45:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 22:30:00 │
        └─────────────────────┘
        >>> df.select(pl.col("dates").dt.truncate("1h"))
        shape: (7, 1)
        ┌─────────────────────┐
        │ dates               │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 03:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 07:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 11:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 15:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 18:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 22:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("dates").dt.truncate("1h")).frame_equal(
        ...     df.select(pl.col("dates").dt.truncate(timedelta(hours=1)))
        ... )
        True

        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 1)
        >>> df = pl.date_range(start, stop, "10m", name="dates").to_frame()
        >>> df.select(["dates", pl.col("dates").dt.truncate("30m").alias("truncate")])
        shape: (7, 2)
        ┌─────────────────────┬─────────────────────┐
        │ dates               ┆ truncate            │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:10:00 ┆ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:20:00 ┆ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:30:00 ┆ 2001-01-01 00:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:40:00 ┆ 2001-01-01 00:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:50:00 ┆ 2001-01-01 00:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 01:00:00 ┆ 2001-01-01 01:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        if offset is None:
            offset = "0ns"

        return pli.wrap_expr(
            self._pyexpr.dt_truncate(
                _timedelta_to_pl_duration(every),
                _timedelta_to_pl_duration(offset),
            )
        )

    def round(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> pli.Expr:
        """
        Divide the date/datetime range into buckets.

        Each date/datetime in the first half of the interval
        is mapped to the start of its bucket.
        Each date/datetime in the seconod half of the interval
        is mapped to the end of its bucket.

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

        Notes
        -----
        The `every` and `offset` argument are created with the
        the following small string formatting language:

        1ns  # 1 nanosecond
        1us  # 1 microsecond
        1ms  # 1 millisecond
        1s   # 1 second
        1m   # 1 minute
        1h   # 1 hour
        1d   # 1 day
        1w   # 1 week
        1mo  # 1 calendar month
        1y   # 1 calendar year

        eg: 3d12h4m25s  # 3 days, 12 hours, 4 minutes, and 25 seconds

        Returns
        -------
        Date/Datetime series

        Warnings
        --------
        This functionality is currently experimental and may
        change without it being considered a breaking change.

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df = pl.date_range(
        ...     start, stop, timedelta(minutes=225), name="dates"
        ... ).to_frame()
        >>> df
        shape: (7, 1)
        ┌─────────────────────┐
        │ dates               │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 03:45:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 07:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 11:15:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 15:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 18:45:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 22:30:00 │
        └─────────────────────┘
        >>> df.select(pl.col("dates").dt.round("1h"))
        shape: (7, 1)
        ┌─────────────────────┐
        │ dates               │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 04:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 08:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 11:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 15:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 19:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 23:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("dates").dt.round("1h")).frame_equal(
        ...     df.select(pl.col("dates").dt.round(timedelta(hours=1)))
        ... )
        True

        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 1)
        >>> df = pl.date_range(start, stop, "10m", name="dates").to_frame()
        >>> df.select(["dates", pl.col("dates").dt.round("30m").alias("round")])
        shape: (7, 2)
        ┌─────────────────────┬─────────────────────┐
        │ dates               ┆ round               │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:10:00 ┆ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:20:00 ┆ 2001-01-01 00:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:30:00 ┆ 2001-01-01 00:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:40:00 ┆ 2001-01-01 00:30:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:50:00 ┆ 2001-01-01 01:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 01:00:00 ┆ 2001-01-01 01:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        if offset is None:
            offset = "0ns"

        return pli.wrap_expr(
            self._pyexpr.dt_round(
                _timedelta_to_pl_duration(every),
                _timedelta_to_pl_duration(offset),
            )
        )

    def strftime(self, fmt: str) -> pli.Expr:
        """
        Format Date/datetime with a formatting rule.

        See `chrono strftime/strptime
        <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_.

        """
        return pli.wrap_expr(self._pyexpr.strftime(fmt))

    def year(self) -> pli.Expr:
        """
        Extract year from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2002, 7, 1)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=180))})
        >>> df
        shape: (4, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-06-30 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-12-27 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2002-06-25 00:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.year())
        shape: (4, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ i32  │
        ╞══════╡
        │ 2001 │
        ├╌╌╌╌╌╌┤
        │ 2001 │
        ├╌╌╌╌╌╌┤
        │ 2001 │
        ├╌╌╌╌╌╌┤
        │ 2002 │
        └──────┘

        """
        return pli.wrap_expr(self._pyexpr.year())

    def iso_year(self) -> pli.Expr:
        """
        Extract ISO year from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the year number in the ISO standard.
        This may not correspond with the calendar year.

        Returns
        -------
        ISO Year as Int32

        """
        return pli.wrap_expr(self._pyexpr.iso_year())

    def quarter(self) -> pli.Expr:
        """
        Extract quarter from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the quarter ranging from 1 to 4.

        Returns
        -------
        Quarter as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2002, 6, 1)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=180))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-06-30 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-12-27 00:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.quarter())
        shape: (3, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 1    │
        ├╌╌╌╌╌╌┤
        │ 2    │
        ├╌╌╌╌╌╌┤
        │ 4    │
        └──────┘

        """
        return pli.wrap_expr(self._pyexpr.quarter())

    def month(self) -> pli.Expr:
        """
        Extract month from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 4, 1)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=31))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-02-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-03-04 00:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.month())
        shape: (3, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 1    │
        ├╌╌╌╌╌╌┤
        │ 2    │
        ├╌╌╌╌╌╌┤
        │ 3    │
        └──────┘

        """
        return pli.wrap_expr(self._pyexpr.month())

    def week(self) -> pli.Expr:
        """
        Extract the week from the underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 4, 1)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=31))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-02-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-03-04 00:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.week())
        shape: (3, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 1    │
        ├╌╌╌╌╌╌┤
        │ 5    │
        ├╌╌╌╌╌╌┤
        │ 9    │
        └──────┘

        """
        return pli.wrap_expr(self._pyexpr.week())

    def weekday(self) -> pli.Expr:
        """
        Extract the week day from the underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the ISO weekday number where monday = 1 and sunday = 7

        Returns
        -------
        Week day as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 9)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=3))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-04 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-07 00:00:00 │
        └─────────────────────┘
        >>> df.select(
        ...     [
        ...         pl.col("date").dt.weekday().alias("weekday"),
        ...         pl.col("date").dt.day().alias("day_of_month"),
        ...         pl.col("date").dt.ordinal_day().alias("day_of_year"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────┬──────────────┬─────────────┐
        │ weekday ┆ day_of_month ┆ day_of_year │
        │ ---     ┆ ---          ┆ ---         │
        │ u32     ┆ u32          ┆ u32         │
        ╞═════════╪══════════════╪═════════════╡
        │ 1       ┆ 1            ┆ 1           │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4       ┆ 4            ┆ 4           │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 7       ┆ 7            ┆ 7           │
        └─────────┴──────────────┴─────────────┘

        """
        return pli.wrap_expr(self._pyexpr.weekday())

    def day(self) -> pli.Expr:
        """
        Extract day from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 9)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=3))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-04 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-07 00:00:00 │
        └─────────────────────┘
        >>> df.select(
        ...     [
        ...         pl.col("date").dt.weekday().alias("weekday"),
        ...         pl.col("date").dt.day().alias("day_of_month"),
        ...         pl.col("date").dt.ordinal_day().alias("day_of_year"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────┬──────────────┬─────────────┐
        │ weekday ┆ day_of_month ┆ day_of_year │
        │ ---     ┆ ---          ┆ ---         │
        │ u32     ┆ u32          ┆ u32         │
        ╞═════════╪══════════════╪═════════════╡
        │ 1       ┆ 1            ┆ 1           │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4       ┆ 4            ┆ 4           │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 7       ┆ 7            ┆ 7           │
        └─────────┴──────────────┴─────────────┘

        """
        return pli.wrap_expr(self._pyexpr.day())

    def ordinal_day(self) -> pli.Expr:
        """
        Extract ordinal day from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 9)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=3))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-04 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-07 00:00:00 │
        └─────────────────────┘
        >>> df.select(
        ...     [
        ...         pl.col("date").dt.weekday().alias("weekday"),
        ...         pl.col("date").dt.day().alias("day_of_month"),
        ...         pl.col("date").dt.ordinal_day().alias("day_of_year"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────┬──────────────┬─────────────┐
        │ weekday ┆ day_of_month ┆ day_of_year │
        │ ---     ┆ ---          ┆ ---         │
        │ u32     ┆ u32          ┆ u32         │
        ╞═════════╪══════════════╪═════════════╡
        │ 1       ┆ 1            ┆ 1           │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4       ┆ 4            ┆ 4           │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 7       ┆ 7            ┆ 7           │
        └─────────┴──────────────┴─────────────┘

        """
        return pli.wrap_expr(self._pyexpr.ordinal_day())

    def hour(self) -> pli.Expr:
        """
        Extract hour from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(hours=12))})
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 12:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-02 00:00:00 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.hour())
        shape: (3, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 0    │
        ├╌╌╌╌╌╌┤
        │ 12   │
        ├╌╌╌╌╌╌┤
        │ 0    │
        └──────┘

        """
        return pli.wrap_expr(self._pyexpr.hour())

    def minute(self) -> pli.Expr:
        """
        Extract minutes from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 4, 0)
        >>> df = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, timedelta(minutes=2))}
        ... )
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:02:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:04:00 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.minute())
        shape: (3, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 0    │
        ├╌╌╌╌╌╌┤
        │ 2    │
        ├╌╌╌╌╌╌┤
        │ 4    │
        └──────┘

        """
        return pli.wrap_expr(self._pyexpr.minute())

    def second(self, fractional: bool = False) -> pli.Expr:
        """
        Extract seconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the integer second number from 0 to 59, or a floating
        point number from 0 < 60 if ``fractional=True`` that includes
        any milli/micro/nanosecond component.

        Returns
        -------
        Second as UInt32 (or Float64)

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> df = pl.DataFrame(
        ...     data={
        ...         "date": pl.date_range(
        ...             low=datetime(2001, 1, 1, 0, 0, 0, 456789),
        ...             high=datetime(2001, 1, 1, 0, 0, 6),
        ...             interval=timedelta(seconds=2, microseconds=654321),
        ...         )
        ...     }
        ... )
        >>> df
        shape: (3, 1)
        ┌────────────────────────────┐
        │ date                       │
        │ ---                        │
        │ datetime[μs]               │
        ╞════════════════════════════╡
        │ 2001-01-01 00:00:00.456789 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:00:03.111110 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:00:05.765431 │
        └────────────────────────────┘
        >>> df.select(pl.col("date").dt.second().alias("secs"))
        shape: (3, 1)
        ┌──────┐
        │ secs │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 0    │
        ├╌╌╌╌╌╌┤
        │ 3    │
        ├╌╌╌╌╌╌┤
        │ 5    │
        └──────┘

        >>> df.select(pl.col("date").dt.second(fractional=True).alias("secs"))
        shape: (3, 1)
        ┌──────────┐
        │ secs     │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.456789 │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 3.11111  │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 5.765431 │
        └──────────┘

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 0, 4)
        >>> df = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, timedelta(seconds=2))}
        ... )
        >>> df
        shape: (3, 1)
        ┌─────────────────────┐
        │ date                │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:00:02 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-01 00:00:04 │
        └─────────────────────┘
        >>> df.select(pl.col("date").dt.second())
        shape: (3, 1)
        ┌──────┐
        │ date │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 0    │
        ├╌╌╌╌╌╌┤
        │ 2    │
        ├╌╌╌╌╌╌┤
        │ 4    │
        └──────┘

        """
        sec = pli.wrap_expr(self._pyexpr.second())
        return (
            sec + (pli.wrap_expr(self._pyexpr.nanosecond()) / pli.lit(1_000_000_000.0))
            if fractional
            else sec
        )

    def millisecond(self) -> pli.Expr:
        """
        Extract milliseconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Milliseconds as UInt32

        """
        return pli.wrap_expr(self._pyexpr.millisecond())

    def microsecond(self) -> pli.Expr:
        """
        Extract microseconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Microseconds as UInt32

        """
        return pli.wrap_expr(self._pyexpr.microsecond())

    def nanosecond(self) -> pli.Expr:
        """
        Extract nanoseconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Nanoseconds as UInt32

        """
        return pli.wrap_expr(self._pyexpr.nanosecond())

    def epoch(self, tu: EpochTimeUnit = "us") -> pli.Expr:
        """
        Get the time passed since the Unix EPOCH in the give time unit.

        Parameters
        ----------
        tu : {'us', 'ns', 'ms', 's', 'd'}
            Time unit.

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=1))})
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.epoch().alias("epoch_ns"),
        ...         pl.col("date").dt.epoch(tu="s").alias("epoch_s"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────────────────┬─────────────────┬───────────┐
        │ date                ┆ epoch_ns        ┆ epoch_s   │
        │ ---                 ┆ ---             ┆ ---       │
        │ datetime[μs]        ┆ i64             ┆ i64       │
        ╞═════════════════════╪═════════════════╪═══════════╡
        │ 2001-01-01 00:00:00 ┆ 978307200000000 ┆ 978307200 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-02 00:00:00 ┆ 978393600000000 ┆ 978393600 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-03 00:00:00 ┆ 978480000000000 ┆ 978480000 │
        └─────────────────────┴─────────────────┴───────────┘

        """
        if tu in DTYPE_TEMPORAL_UNITS:
            return self.timestamp(tu)  # type: ignore[arg-type]
        elif tu == "s":
            return pli.wrap_expr(self._pyexpr.dt_epoch_seconds())
        elif tu == "d":
            return pli.wrap_expr(self._pyexpr).cast(Date).cast(Int32)
        else:
            raise ValueError(
                f"tu must be one of {{'ns', 'us', 'ms', 's', 'd'}}, got {tu}"
            )

    def timestamp(self, tu: TimeUnit = "us") -> pli.Expr:
        """
        Return a timestamp in the given time unit.

        Parameters
        ----------
        tu : {'us', 'ns', 'ms'}
            Time unit.

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=1))})
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.timestamp().alias("timestamp_ns"),
        ...         pl.col("date").dt.timestamp(tu="ms").alias("timestamp_ms"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────────────────┬─────────────────┬──────────────┐
        │ date                ┆ timestamp_ns    ┆ timestamp_ms │
        │ ---                 ┆ ---             ┆ ---          │
        │ datetime[μs]        ┆ i64             ┆ i64          │
        ╞═════════════════════╪═════════════════╪══════════════╡
        │ 2001-01-01 00:00:00 ┆ 978307200000000 ┆ 978307200000 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-02 00:00:00 ┆ 978393600000000 ┆ 978393600000 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-03 00:00:00 ┆ 978480000000000 ┆ 978480000000 │
        └─────────────────────┴─────────────────┴──────────────┘

        """
        return pli.wrap_expr(self._pyexpr.timestamp(tu))

    def with_time_unit(self, tu: TimeUnit) -> pli.Expr:
        """
        Set time unit of a Series of dtype Datetime or Duration.

        This does not modify underlying data, and should be used to fix an incorrect
        time unit.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2001, 1, 1), datetime(2001, 1, 3), "1d", time_unit="ns"
        ...         )
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.with_time_unit(tu="us").alias("tu_us"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────────────────────┬───────────────────────┐
        │ date                ┆ tu_us                 │
        │ ---                 ┆ ---                   │
        │ datetime[ns]        ┆ datetime[μs]          │
        ╞═════════════════════╪═══════════════════════╡
        │ 2001-01-01 00:00:00 ┆ +32971-04-28 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-02 00:00:00 ┆ +32974-01-22 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-03 00:00:00 ┆ +32976-10-18 00:00:00 │
        └─────────────────────┴───────────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.dt_with_time_unit(tu))

    def cast_time_unit(self, tu: TimeUnit) -> pli.Expr:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2001, 1, 1), datetime(2001, 1, 3), "1d"
        ...         )
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.cast_time_unit(tu="ms").alias("tu_ms"),
        ...         pl.col("date").dt.cast_time_unit(tu="ns").alias("tu_ns"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────────────────┬─────────────────────┬─────────────────────┐
        │ date                ┆ tu_ms               ┆ tu_ns               │
        │ ---                 ┆ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[ms]        ┆ datetime[ns]        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-02 00:00:00 ┆ 2001-01-02 00:00:00 ┆ 2001-01-02 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-01-03 00:00:00 ┆ 2001-01-03 00:00:00 ┆ 2001-01-03 00:00:00 │
        └─────────────────────┴─────────────────────┴─────────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.dt_cast_time_unit(tu))

    def with_time_zone(self, tz: str | None) -> pli.Expr:
        """
        Set time zone for a Series of type Datetime.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date")
        ...         .dt.with_time_zone(tz="Europe/London")
        ...         .alias("London"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────────────────────┬─────────────────────────────┐
        │ date                ┆ London                      │
        │ ---                 ┆ ---                         │
        │ datetime[μs]        ┆ datetime[μs, Europe/London] │
        ╞═════════════════════╪═════════════════════════════╡
        │ 2020-03-01 00:00:00 ┆ 2020-03-01 00:00:00 GMT     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-04-01 00:00:00 ┆ 2020-04-01 01:00:00 BST     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-05-01 00:00:00 ┆ 2020-05-01 01:00:00 BST     │
        └─────────────────────┴─────────────────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.dt_with_time_zone(tz))

    def cast_time_zone(self, tz: str) -> pli.Expr:
        """
        Cast time zone for a Series of type Datetime.

        Different from ``with_time_zone``, this will also modify
        the underlying timestamp,

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """
        return pli.wrap_expr(self._pyexpr.dt_cast_time_zone(tz))

    def tz_localize(self, tz: str) -> pli.Expr:
        """
        Localize tz-naive Datetime Series to tz-aware Datetime Series.

        This method takes a naive Datetime Series and makes this time zone aware.
        It does not move the time to another time zone.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """
        return pli.wrap_expr(self._pyexpr.dt_tz_localize(tz))

    def days(self) -> pli.Expr:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.days().alias("days_diff"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────────────────────┬───────────┐
        │ date                ┆ days_diff │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ i64       │
        ╞═════════════════════╪═══════════╡
        │ 2020-03-01 00:00:00 ┆ null      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-04-01 00:00:00 ┆ 31        │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-05-01 00:00:00 ┆ 30        │
        └─────────────────────┴───────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_days())

    def hours(self) -> pli.Expr:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 4), "1d"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.hours().alias("hours_diff"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────────────────────┬────────────┐
        │ date                ┆ hours_diff │
        │ ---                 ┆ ---        │
        │ datetime[μs]        ┆ i64        │
        ╞═════════════════════╪════════════╡
        │ 2020-01-01 00:00:00 ┆ null       │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-02 00:00:00 ┆ 24         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-03 00:00:00 ┆ 24         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-04 00:00:00 ┆ 24         │
        └─────────────────────┴────────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_hours())

    def minutes(self) -> pli.Expr:
        """
        Extract the minutes from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 4), "1d"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.minutes().alias("minutes_diff"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────────────────────┬──────────────┐
        │ date                ┆ minutes_diff │
        │ ---                 ┆ ---          │
        │ datetime[μs]        ┆ i64          │
        ╞═════════════════════╪══════════════╡
        │ 2020-01-01 00:00:00 ┆ null         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-02 00:00:00 ┆ 1440         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-03 00:00:00 ┆ 1440         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-04 00:00:00 ┆ 1440         │
        └─────────────────────┴──────────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_minutes())

    def seconds(self) -> pli.Expr:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 4, 0), "1m"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.seconds().alias("seconds_diff"),
        ...     ]
        ... )
        shape: (5, 2)
        ┌─────────────────────┬──────────────┐
        │ date                ┆ seconds_diff │
        │ ---                 ┆ ---          │
        │ datetime[μs]        ┆ i64          │
        ╞═════════════════════╪══════════════╡
        │ 2020-01-01 00:00:00 ┆ null         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:01:00 ┆ 60           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:02:00 ┆ 60           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:03:00 ┆ 60           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:04:00 ┆ 60           │
        └─────────────────────┴──────────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_seconds())

    def milliseconds(self) -> pli.Expr:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1, 0), "1ms"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.milliseconds().alias("milliseconds_diff"),
        ...     ]
        ... )
        shape: (1001, 2)
        ┌─────────────────────────┬───────────────────┐
        │ date                    ┆ milliseconds_diff │
        │ ---                     ┆ ---               │
        │ datetime[μs]            ┆ i64               │
        ╞═════════════════════════╪═══════════════════╡
        │ 2020-01-01 00:00:00     ┆ null              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.001 ┆ 1                 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.002 ┆ 1                 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.003 ┆ 1                 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ...                     ┆ ...               │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.997 ┆ 1                 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.998 ┆ 1                 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.999 ┆ 1                 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:01     ┆ 1                 │
        └─────────────────────────┴───────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_milliseconds())

    def microseconds(self) -> pli.Expr:
        """
        Extract the microseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1, 0), "1ms"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.microseconds().alias("microseconds_diff"),
        ...     ]
        ... )
        shape: (1001, 2)
        ┌─────────────────────────┬───────────────────┐
        │ date                    ┆ microseconds_diff │
        │ ---                     ┆ ---               │
        │ datetime[μs]            ┆ i64               │
        ╞═════════════════════════╪═══════════════════╡
        │ 2020-01-01 00:00:00     ┆ null              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.001 ┆ 1000              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.002 ┆ 1000              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.003 ┆ 1000              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ...                     ┆ ...               │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.997 ┆ 1000              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.998 ┆ 1000              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.999 ┆ 1000              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:01     ┆ 1000              │
        └─────────────────────────┴───────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_microseconds())

    def nanoseconds(self) -> pli.Expr:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1, 0), "1ms"
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.nanoseconds().alias("nanoseconds_diff"),
        ...     ]
        ... )
        shape: (1001, 2)
        ┌─────────────────────────┬──────────────────┐
        │ date                    ┆ nanoseconds_diff │
        │ ---                     ┆ ---              │
        │ datetime[μs]            ┆ i64              │
        ╞═════════════════════════╪══════════════════╡
        │ 2020-01-01 00:00:00     ┆ null             │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.001 ┆ 1000000          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.002 ┆ 1000000          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.003 ┆ 1000000          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ...                     ┆ ...              │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.997 ┆ 1000000          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.998 ┆ 1000000          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:00.999 ┆ 1000000          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2020-01-01 00:00:01     ┆ 1000000          │
        └─────────────────────────┴──────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.duration_nanoseconds())

    def offset_by(self, by: str) -> pli.Expr:
        """
        Offset this date by a relative time offset.

        This differs from ``pl.col("foo") + timedelta`` in that it can
        take months and leap years into account. Note that only a single minus
        sign is allowed in the ``by`` string, as the first character.

        Parameters
        ----------
        by
            The offset is dictated by the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

        Returns
        -------
        Date/Datetime expression

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "dates": pl.date_range(
        ...             datetime(2000, 1, 1), datetime(2005, 1, 1), "1y"
        ...         )
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("dates").dt.offset_by("1y").alias("date_plus_1y"),
        ...         pl.col("dates").dt.offset_by("-1y2mo").alias("date_min"),
        ...     ]
        ... )
        shape: (6, 2)
        ┌─────────────────────┬─────────────────────┐
        │ date_plus_1y        ┆ date_min            │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 1998-11-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2002-01-01 00:00:00 ┆ 1999-11-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2003-01-01 00:00:00 ┆ 2000-11-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2004-01-01 00:00:00 ┆ 2001-11-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2005-01-01 00:00:00 ┆ 2002-11-01 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2006-01-01 00:00:00 ┆ 2003-11-01 00:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.dt_offset_by(by))
