from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import DTYPE_TEMPORAL_UNITS, Date, Int32
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.convert import _timedelta_to_pl_duration
from polars.utils.deprecation import (
    deprecate_renamed_function,
    deprecate_saturating,
    issue_deprecation_warning,
    rename_use_earliest_to_ambiguous,
)

if TYPE_CHECKING:
    from datetime import timedelta

    from polars import Expr
    from polars.type_aliases import Ambiguous, EpochTimeUnit, TimeUnit


class ExprDateTimeNameSpace:
    """Namespace for datetime related expressions."""

    _accessor = "dt"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def truncate(
        self,
        every: str | timedelta | Expr,
        offset: str | timedelta | None = None,
        *,
        use_earliest: bool | None = None,
        ambiguous: Ambiguous | Expr | None = None,
    ) -> Expr:
        """
        Divide the date/datetime range into buckets.

        Each date/datetime is mapped to the start of its bucket using the corresponding
        local datetime. Note that weekly buckets start on Monday.
        Ambiguous results are localised using the DST offset of the original timestamp -
        for example, truncating `'2022-11-06 01:30:00 CST'` by `'1h'` results in
        `'2022-11-06 01:00:00 CST'`, whereas truncating `'2022-11-06 01:30:00 CDT'` by
        `'1h'` results in `'2022-11-06 01:00:00 CDT'`.

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window
        use_earliest
            Determine how to deal with ambiguous datetimes:

            - `None` (default): raise
            - `True`: use the earliest datetime
            - `False`: use the latest datetime

            .. deprecated:: 0.19.0
                Use `ambiguous` instead
        ambiguous
            Determine how to deal with ambiguous datetimes:

            - `'raise'` (default): raise
            - `'earliest'`: use the earliest datetime
            - `'latest'`: use the latest datetime

            .. deprecated:: 0.19.3
                This is now auto-inferred, you can safely remove this argument.

        Notes
        -----
        The `every` and `offset` argument are created with the
        the following string language:

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

        These strings can be combined:

        - 3d12h4m25s # 3 days, 12 hours, 4 minutes, and 25 seconds


        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        Returns
        -------
        Expr
            Expression of data type :class:`Date` or :class:`Datetime`.

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> df = pl.datetime_range(
        ...     datetime(2001, 1, 1),
        ...     datetime(2001, 1, 2),
        ...     timedelta(minutes=225),
        ...     eager=True,
        ... ).to_frame()
        >>> df
        shape: (7, 1)
        ┌─────────────────────┐
        │ datetime            │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        │ 2001-01-01 03:45:00 │
        │ 2001-01-01 07:30:00 │
        │ 2001-01-01 11:15:00 │
        │ 2001-01-01 15:00:00 │
        │ 2001-01-01 18:45:00 │
        │ 2001-01-01 22:30:00 │
        └─────────────────────┘
        >>> df.select(pl.col("datetime").dt.truncate("1h"))
        shape: (7, 1)
        ┌─────────────────────┐
        │ datetime            │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2001-01-01 00:00:00 │
        │ 2001-01-01 03:00:00 │
        │ 2001-01-01 07:00:00 │
        │ 2001-01-01 11:00:00 │
        │ 2001-01-01 15:00:00 │
        │ 2001-01-01 18:00:00 │
        │ 2001-01-01 22:00:00 │
        └─────────────────────┘
        >>> truncate_str = df.select(pl.col("datetime").dt.truncate("1h"))
        >>> truncate_td = df.select(pl.col("datetime").dt.truncate(timedelta(hours=1)))
        >>> truncate_str.equals(truncate_td)
        True

        >>> df = pl.datetime_range(
        ...     datetime(2001, 1, 1), datetime(2001, 1, 1, 1), "10m", eager=True
        ... ).to_frame()
        >>> df.select(
        ...     "datetime", pl.col("datetime").dt.truncate("30m").alias("truncate")
        ... )
        shape: (7, 2)
        ┌─────────────────────┬─────────────────────┐
        │ datetime            ┆ truncate            │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-01 00:10:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-01 00:20:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-01 00:30:00 ┆ 2001-01-01 00:30:00 │
        │ 2001-01-01 00:40:00 ┆ 2001-01-01 00:30:00 │
        │ 2001-01-01 00:50:00 ┆ 2001-01-01 00:30:00 │
        │ 2001-01-01 01:00:00 ┆ 2001-01-01 01:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        every = deprecate_saturating(every)
        offset = deprecate_saturating(offset)
        if not isinstance(every, pl.Expr):
            every = _timedelta_to_pl_duration(every)

        if use_earliest is not None:
            issue_deprecation_warning(
                "`use_earliest` is deprecated. It is now auto-inferred, you can safely remove this argument.",
                version="0.19.13",
            )
        if ambiguous is not None:
            issue_deprecation_warning(
                "`ambiguous` is deprecated. It is now auto-inferred, you can safely remove this argument.",
                version="0.19.13",
            )
        every = parse_as_expression(every, str_as_lit=True)

        if offset is None:
            offset = "0ns"

        return wrap_expr(
            self._pyexpr.dt_truncate(
                every,
                _timedelta_to_pl_duration(offset),
            )
        )

    def round(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
        *,
        ambiguous: Ambiguous | Expr | None = None,
    ) -> Expr:
        """
        Divide the date/datetime range into buckets.

        Each date/datetime in the first half of the interval
        is mapped to the start of its bucket.
        Each date/datetime in the second half of the interval
        is mapped to the end of its bucket.
        Ambiguous results are localised using the DST offset of the original timestamp -
        for example, rounding `'2022-11-06 01:20:00 CST'` by `'1h'` results in
        `'2022-11-06 01:00:00 CST'`, whereas rounding `'2022-11-06 01:20:00 CDT'` by
        `'1h'` results in `'2022-11-06 01:00:00 CDT'`.

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window
        ambiguous
            Determine how to deal with ambiguous datetimes:

            - `'raise'` (default): raise
            - `'earliest'`: use the earliest datetime
            - `'latest'`: use the latest datetime

            .. deprecated: 0.19.3
                This is now auto-inferred, you can safely remove this argument.

        Notes
        -----
        The `every` and `offset` argument are created with the
        the following small string formatting language:

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

        eg: 3d12h4m25s  # 3 days, 12 hours, 4 minutes, and 25 seconds


        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        Returns
        -------
        Expr
            Expression of data type :class:`Date` or :class:`Datetime`.

        Warnings
        --------
        This functionality is currently experimental and may
        change without it being considered a breaking change.

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> df = pl.datetime_range(
        ...     datetime(2001, 1, 1),
        ...     datetime(2001, 1, 2),
        ...     timedelta(minutes=225),
        ...     eager=True,
        ... ).to_frame()
        >>> df.with_columns(pl.col("datetime").dt.round("1h").alias("round"))
        shape: (7, 2)
        ┌─────────────────────┬─────────────────────┐
        │ datetime            ┆ round               │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-01 03:45:00 ┆ 2001-01-01 04:00:00 │
        │ 2001-01-01 07:30:00 ┆ 2001-01-01 08:00:00 │
        │ 2001-01-01 11:15:00 ┆ 2001-01-01 11:00:00 │
        │ 2001-01-01 15:00:00 ┆ 2001-01-01 15:00:00 │
        │ 2001-01-01 18:45:00 ┆ 2001-01-01 19:00:00 │
        │ 2001-01-01 22:30:00 ┆ 2001-01-01 23:00:00 │
        └─────────────────────┴─────────────────────┘

        >>> df = pl.datetime_range(
        ...     datetime(2001, 1, 1), datetime(2001, 1, 1, 1), "10m", eager=True
        ... ).to_frame()
        >>> df.with_columns(pl.col("datetime").dt.round("30m").alias("round"))
        shape: (7, 2)
        ┌─────────────────────┬─────────────────────┐
        │ datetime            ┆ round               │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-01 00:10:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-01 00:20:00 ┆ 2001-01-01 00:30:00 │
        │ 2001-01-01 00:30:00 ┆ 2001-01-01 00:30:00 │
        │ 2001-01-01 00:40:00 ┆ 2001-01-01 00:30:00 │
        │ 2001-01-01 00:50:00 ┆ 2001-01-01 01:00:00 │
        │ 2001-01-01 01:00:00 ┆ 2001-01-01 01:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        every = deprecate_saturating(every)
        offset = deprecate_saturating(offset)
        if offset is None:
            offset = "0ns"

        if ambiguous is not None:
            issue_deprecation_warning(
                "`ambiguous` is deprecated. It is now auto-inferred, you can safely remove this argument.",
                version="0.19.13",
            )

        return wrap_expr(
            self._pyexpr.dt_round(
                _timedelta_to_pl_duration(every),
                _timedelta_to_pl_duration(offset),
            )
        )

    def combine(self, time: dt.time | Expr, time_unit: TimeUnit = "us") -> Expr:
        """
        Create a naive Datetime from an existing Date/Datetime expression and a Time.

        If the underlying expression is a Datetime then its time component is replaced,
        and if it is a Date then a new Datetime is created by combining the two values.

        Parameters
        ----------
        time
            A python time literal or polars expression/column that resolves to a time.
        time_unit : {'ns', 'us', 'ms'}
            Unit of time.

        Examples
        --------
        >>> from datetime import datetime, date, time
        >>> df = pl.DataFrame(
        ...     {
        ...         "dtm": [
        ...             datetime(2022, 12, 31, 10, 30, 45),
        ...             datetime(2023, 7, 5, 23, 59, 59),
        ...         ],
        ...         "dt": [date(2022, 10, 10), date(2022, 7, 5)],
        ...         "tm": [time(1, 2, 3, 456000), time(7, 8, 9, 101000)],
        ...     }
        ... )
        >>> df
        shape: (2, 3)
        ┌─────────────────────┬────────────┬──────────────┐
        │ dtm                 ┆ dt         ┆ tm           │
        │ ---                 ┆ ---        ┆ ---          │
        │ datetime[μs]        ┆ date       ┆ time         │
        ╞═════════════════════╪════════════╪══════════════╡
        │ 2022-12-31 10:30:45 ┆ 2022-10-10 ┆ 01:02:03.456 │
        │ 2023-07-05 23:59:59 ┆ 2022-07-05 ┆ 07:08:09.101 │
        └─────────────────────┴────────────┴──────────────┘
        >>> df.select(
        ...     [
        ...         pl.col("dtm").dt.combine(pl.col("tm")).alias("d1"),
        ...         pl.col("dt").dt.combine(pl.col("tm")).alias("d2"),
        ...         pl.col("dt").dt.combine(time(4, 5, 6)).alias("d3"),
        ...     ]
        ... )
        shape: (2, 3)
        ┌─────────────────────────┬─────────────────────────┬─────────────────────┐
        │ d1                      ┆ d2                      ┆ d3                  │
        │ ---                     ┆ ---                     ┆ ---                 │
        │ datetime[μs]            ┆ datetime[μs]            ┆ datetime[μs]        │
        ╞═════════════════════════╪═════════════════════════╪═════════════════════╡
        │ 2022-12-31 01:02:03.456 ┆ 2022-10-10 01:02:03.456 ┆ 2022-10-10 04:05:06 │
        │ 2023-07-05 07:08:09.101 ┆ 2022-07-05 07:08:09.101 ┆ 2022-07-05 04:05:06 │
        └─────────────────────────┴─────────────────────────┴─────────────────────┘
        """
        if not isinstance(time, (dt.time, pl.Expr)):
            raise TypeError(
                f"expected 'time' to be a Python time or Polars expression, found {type(time).__name__!r}"
            )
        time = parse_as_expression(time)
        return wrap_expr(self._pyexpr.dt_combine(time, time_unit))

    def to_string(self, format: str) -> Expr:
        """
        Convert a Date/Time/Datetime column into a Utf8 column with the given format.

        Similar to `cast(pl.Utf8)`, but this method allows you to customize the
        formatting of the resulting string.

        Parameters
        ----------
        format
            Format to use, refer to the `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: `"%y-%m-%d"`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "datetime": [
        ...             datetime(2020, 3, 1),
        ...             datetime(2020, 4, 1),
        ...             datetime(2020, 5, 1),
        ...         ]
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("datetime")
        ...     .dt.to_string("%Y/%m/%d %H:%M:%S")
        ...     .alias("datetime_string")
        ... )
        shape: (3, 2)
        ┌─────────────────────┬─────────────────────┐
        │ datetime            ┆ datetime_string     │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ str                 │
        ╞═════════════════════╪═════════════════════╡
        │ 2020-03-01 00:00:00 ┆ 2020/03/01 00:00:00 │
        │ 2020-04-01 00:00:00 ┆ 2020/04/01 00:00:00 │
        │ 2020-05-01 00:00:00 ┆ 2020/05/01 00:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        return wrap_expr(self._pyexpr.dt_to_string(format))

    def strftime(self, format: str) -> Expr:
        """
        Convert a Date/Time/Datetime column into a Utf8 column with the given format.

        Similar to `cast(pl.Utf8)`, but this method allows you to customize the
        formatting of the resulting string.

        Alias for :func:`to_string`.

        Parameters
        ----------
        format
            Format to use, refer to the `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: `"%y-%m-%d"`.

        See Also
        --------
        to_string : The identical expression for which `strftime` is an alias.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "datetime": [
        ...             datetime(2020, 3, 1),
        ...             datetime(2020, 4, 1),
        ...             datetime(2020, 5, 1),
        ...         ]
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("datetime")
        ...     .dt.strftime("%Y/%m/%d %H:%M:%S")
        ...     .alias("datetime_string")
        ... )
        shape: (3, 2)
        ┌─────────────────────┬─────────────────────┐
        │ datetime            ┆ datetime_string     │
        │ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ str                 │
        ╞═════════════════════╪═════════════════════╡
        │ 2020-03-01 00:00:00 ┆ 2020/03/01 00:00:00 │
        │ 2020-04-01 00:00:00 ┆ 2020/04/01 00:00:00 │
        │ 2020-05-01 00:00:00 ┆ 2020/05/01 00:00:00 │
        └─────────────────────┴─────────────────────┘

        """
        return self.to_string(format)

    def year(self) -> Expr:
        """
        Extract year from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the year number in the calendar date.

        Returns
        -------
        Expr
            Expression of data type :class:`Int32`.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {"date": [date(1977, 1, 1), date(1978, 1, 1), date(1979, 1, 1)]}
        ... )
        >>> df.select(
        ...     "date",
        ...     pl.col("date").dt.year().alias("calendar_year"),
        ...     pl.col("date").dt.iso_year().alias("iso_year"),
        ... )
        shape: (3, 3)
        ┌────────────┬───────────────┬──────────┐
        │ date       ┆ calendar_year ┆ iso_year │
        │ ---        ┆ ---           ┆ ---      │
        │ date       ┆ i32           ┆ i32      │
        ╞════════════╪═══════════════╪══════════╡
        │ 1977-01-01 ┆ 1977          ┆ 1976     │
        │ 1978-01-01 ┆ 1978          ┆ 1977     │
        │ 1979-01-01 ┆ 1979          ┆ 1979     │
        └────────────┴───────────────┴──────────┘

        """
        return wrap_expr(self._pyexpr.dt_year())

    def is_leap_year(self) -> Expr:
        """
        Determine whether the year of the underlying date is a leap year.

        Applies to Date and Datetime columns.

        Returns
        -------
        Expr
            Expression of data type :class:`Boolean`.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {"date": [date(2000, 1, 1), date(2001, 1, 1), date(2002, 1, 1)]}
        ... )
        >>> df.select(pl.col("date").dt.is_leap_year())
        shape: (3, 1)
        ┌───────┐
        │ date  │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ false │
        │ false │
        └───────┘

        """
        return wrap_expr(self._pyexpr.dt_is_leap_year())

    def iso_year(self) -> Expr:
        """
        Extract ISO year from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the year number in the ISO standard.
        This may not correspond with the calendar year.

        Returns
        -------
        Expr
            Expression of data type :class:`Int32`.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {"date": [date(1977, 1, 1), date(1978, 1, 1), date(1979, 1, 1)]}
        ... )
        >>> df.select(
        ...     "date",
        ...     pl.col("date").dt.year().alias("calendar_year"),
        ...     pl.col("date").dt.iso_year().alias("iso_year"),
        ... )
        shape: (3, 3)
        ┌────────────┬───────────────┬──────────┐
        │ date       ┆ calendar_year ┆ iso_year │
        │ ---        ┆ ---           ┆ ---      │
        │ date       ┆ i32           ┆ i32      │
        ╞════════════╪═══════════════╪══════════╡
        │ 1977-01-01 ┆ 1977          ┆ 1976     │
        │ 1978-01-01 ┆ 1978          ┆ 1977     │
        │ 1979-01-01 ┆ 1979          ┆ 1979     │
        └────────────┴───────────────┴──────────┘

        """
        return wrap_expr(self._pyexpr.dt_iso_year())

    def quarter(self) -> Expr:
        """
        Extract quarter from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the quarter ranging from 1 to 4.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {"date": [date(2001, 1, 1), date(2001, 6, 30), date(2001, 12, 27)]}
        ... )
        >>> df.with_columns(pl.col("date").dt.quarter().alias("quarter"))
        shape: (3, 2)
        ┌────────────┬─────────┐
        │ date       ┆ quarter │
        │ ---        ┆ ---     │
        │ date       ┆ u32     │
        ╞════════════╪═════════╡
        │ 2001-01-01 ┆ 1       │
        │ 2001-06-30 ┆ 2       │
        │ 2001-12-27 ┆ 4       │
        └────────────┴─────────┘

        """
        return wrap_expr(self._pyexpr.dt_quarter())

    def month(self) -> Expr:
        """
        Extract month from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {"date": [date(2001, 1, 1), date(2001, 6, 30), date(2001, 12, 27)]}
        ... )
        >>> df.with_columns(pl.col("date").dt.month().alias("month"))
        shape: (3, 2)
        ┌────────────┬───────┐
        │ date       ┆ month │
        │ ---        ┆ ---   │
        │ date       ┆ u32   │
        ╞════════════╪═══════╡
        │ 2001-01-01 ┆ 1     │
        │ 2001-06-30 ┆ 6     │
        │ 2001-12-27 ┆ 12    │
        └────────────┴───────┘

        """
        return wrap_expr(self._pyexpr.dt_month())

    def week(self) -> Expr:
        """
        Extract the week from the underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {"date": [date(2001, 1, 1), date(2001, 6, 30), date(2001, 12, 27)]}
        ... )
        >>> df.with_columns(pl.col("date").dt.week().alias("week"))
        shape: (3, 2)
        ┌────────────┬──────┐
        │ date       ┆ week │
        │ ---        ┆ ---  │
        │ date       ┆ u32  │
        ╞════════════╪══════╡
        │ 2001-01-01 ┆ 1    │
        │ 2001-06-30 ┆ 26   │
        │ 2001-12-27 ┆ 52   │
        └────────────┴──────┘

        """
        return wrap_expr(self._pyexpr.dt_week())

    def weekday(self) -> Expr:
        """
        Extract the week day from the underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the ISO weekday number where monday = 1 and sunday = 7

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        See Also
        --------
        day
        ordinal_day

        Examples
        --------
        >>> from datetime import timedelta, date
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             date(2001, 12, 22), date(2001, 12, 25), eager=True
        ...         )
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("date").dt.weekday().alias("weekday"),
        ...     pl.col("date").dt.day().alias("day_of_month"),
        ...     pl.col("date").dt.ordinal_day().alias("day_of_year"),
        ... )
        shape: (4, 4)
        ┌────────────┬─────────┬──────────────┬─────────────┐
        │ date       ┆ weekday ┆ day_of_month ┆ day_of_year │
        │ ---        ┆ ---     ┆ ---          ┆ ---         │
        │ date       ┆ u32     ┆ u32          ┆ u32         │
        ╞════════════╪═════════╪══════════════╪═════════════╡
        │ 2001-12-22 ┆ 6       ┆ 22           ┆ 356         │
        │ 2001-12-23 ┆ 7       ┆ 23           ┆ 357         │
        │ 2001-12-24 ┆ 1       ┆ 24           ┆ 358         │
        │ 2001-12-25 ┆ 2       ┆ 25           ┆ 359         │
        └────────────┴─────────┴──────────────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.dt_weekday())

    def day(self) -> Expr:
        """
        Extract day from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        See Also
        --------
        weekday
        ordinal_day

        Examples
        --------
        >>> from datetime import timedelta, date
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             date(2001, 12, 22), date(2001, 12, 25), eager=True
        ...         )
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("date").dt.weekday().alias("weekday"),
        ...     pl.col("date").dt.day().alias("day_of_month"),
        ...     pl.col("date").dt.ordinal_day().alias("day_of_year"),
        ... )
        shape: (4, 4)
        ┌────────────┬─────────┬──────────────┬─────────────┐
        │ date       ┆ weekday ┆ day_of_month ┆ day_of_year │
        │ ---        ┆ ---     ┆ ---          ┆ ---         │
        │ date       ┆ u32     ┆ u32          ┆ u32         │
        ╞════════════╪═════════╪══════════════╪═════════════╡
        │ 2001-12-22 ┆ 6       ┆ 22           ┆ 356         │
        │ 2001-12-23 ┆ 7       ┆ 23           ┆ 357         │
        │ 2001-12-24 ┆ 1       ┆ 24           ┆ 358         │
        │ 2001-12-25 ┆ 2       ┆ 25           ┆ 359         │
        └────────────┴─────────┴──────────────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.dt_day())

    def ordinal_day(self) -> Expr:
        """
        Extract ordinal day from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        See Also
        --------
        weekday
        day

        Examples
        --------
        >>> from datetime import timedelta, date
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             date(2001, 12, 22), date(2001, 12, 25), eager=True
        ...         )
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("date").dt.weekday().alias("weekday"),
        ...     pl.col("date").dt.day().alias("day_of_month"),
        ...     pl.col("date").dt.ordinal_day().alias("day_of_year"),
        ... )
        shape: (4, 4)
        ┌────────────┬─────────┬──────────────┬─────────────┐
        │ date       ┆ weekday ┆ day_of_month ┆ day_of_year │
        │ ---        ┆ ---     ┆ ---          ┆ ---         │
        │ date       ┆ u32     ┆ u32          ┆ u32         │
        ╞════════════╪═════════╪══════════════╪═════════════╡
        │ 2001-12-22 ┆ 6       ┆ 22           ┆ 356         │
        │ 2001-12-23 ┆ 7       ┆ 23           ┆ 357         │
        │ 2001-12-24 ┆ 1       ┆ 24           ┆ 358         │
        │ 2001-12-25 ┆ 2       ┆ 25           ┆ 359         │
        └────────────┴─────────┴──────────────┴─────────────┘

        """
        return wrap_expr(self._pyexpr.dt_ordinal_day())

    def time(self) -> Expr:
        """
        Extract time.

        Applies to Datetime columns only; fails on Date.

        Returns
        -------
        Expr
            Expression of data type :class:`Time`.

        """
        return wrap_expr(self._pyexpr.dt_time())

    def date(self) -> Expr:
        """
        Extract date from date(time).

        Applies to Date and Datetime columns.

        Returns
        -------
        Expr
            Expression of data type :class:`Date`.

        """
        return wrap_expr(self._pyexpr.dt_date())

    def datetime(self) -> Expr:
        """
        Return datetime.

        Applies to Datetime columns.

        Returns
        -------
        Expr
            Expression of data type :class:`Datetime`.

        """
        return wrap_expr(self._pyexpr.dt_datetime())

    def hour(self) -> Expr:
        """
        Extract hour from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "datetime": [
        ...             datetime(2001, 1, 1, 0, 0, 0),
        ...             datetime(2010, 1, 1, 15, 30, 45),
        ...             datetime(2022, 12, 31, 23, 59, 59),
        ...         ]
        ...     }
        ... )
        >>> df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        shape: (3, 2)
        ┌─────────────────────┬──────┐
        │ datetime            ┆ hour │
        │ ---                 ┆ ---  │
        │ datetime[μs]        ┆ u32  │
        ╞═════════════════════╪══════╡
        │ 2001-01-01 00:00:00 ┆ 0    │
        │ 2010-01-01 15:30:45 ┆ 15   │
        │ 2022-12-31 23:59:59 ┆ 23   │
        └─────────────────────┴──────┘

        """
        return wrap_expr(self._pyexpr.dt_hour())

    def minute(self) -> Expr:
        """
        Extract minutes from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "datetime": [
        ...             datetime(2001, 1, 1, 0, 0, 0),
        ...             datetime(2010, 1, 1, 15, 30, 45),
        ...             datetime(2022, 12, 31, 23, 59, 59),
        ...         ]
        ...     }
        ... )
        >>> df.with_columns(pl.col("datetime").dt.minute().alias("minute"))
        shape: (3, 2)
        ┌─────────────────────┬────────┐
        │ datetime            ┆ minute │
        │ ---                 ┆ ---    │
        │ datetime[μs]        ┆ u32    │
        ╞═════════════════════╪════════╡
        │ 2001-01-01 00:00:00 ┆ 0      │
        │ 2010-01-01 15:30:45 ┆ 30     │
        │ 2022-12-31 23:59:59 ┆ 59     │
        └─────────────────────┴────────┘

        """
        return wrap_expr(self._pyexpr.dt_minute())

    def second(self, *, fractional: bool = False) -> Expr:
        """
        Extract seconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the integer second number from 0 to 59, or a floating
        point number from 0 < 60 if `fractional=True` that includes
        any milli/micro/nanosecond component.

        Parameters
        ----------
        fractional
            Whether to include the fractional component of the second.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32` or :class:`Float64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "datetime": [
        ...             datetime(2000, 1, 1, 0, 0, 0, 456789),
        ...             datetime(2000, 1, 1, 0, 0, 3, 111110),
        ...             datetime(2000, 1, 1, 0, 0, 5, 765431),
        ...         ]
        ...     }
        ... )
        >>> df.with_columns(pl.col("datetime").dt.second().alias("second"))
        shape: (3, 2)
        ┌────────────────────────────┬────────┐
        │ datetime                   ┆ second │
        │ ---                        ┆ ---    │
        │ datetime[μs]               ┆ u32    │
        ╞════════════════════════════╪════════╡
        │ 2000-01-01 00:00:00.456789 ┆ 0      │
        │ 2000-01-01 00:00:03.111110 ┆ 3      │
        │ 2000-01-01 00:00:05.765431 ┆ 5      │
        └────────────────────────────┴────────┘
        >>> df.with_columns(
        ...     pl.col("datetime").dt.second(fractional=True).alias("second")
        ... )
        shape: (3, 2)
        ┌────────────────────────────┬──────────┐
        │ datetime                   ┆ second   │
        │ ---                        ┆ ---      │
        │ datetime[μs]               ┆ f64      │
        ╞════════════════════════════╪══════════╡
        │ 2000-01-01 00:00:00.456789 ┆ 0.456789 │
        │ 2000-01-01 00:00:03.111110 ┆ 3.11111  │
        │ 2000-01-01 00:00:05.765431 ┆ 5.765431 │
        └────────────────────────────┴──────────┘

        """
        sec = wrap_expr(self._pyexpr.dt_second())
        return (
            sec + (wrap_expr(self._pyexpr.dt_nanosecond()) / F.lit(1_000_000_000.0))
            if fractional
            else sec
        )

    def millisecond(self) -> Expr:
        """
        Extract milliseconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        """
        return wrap_expr(self._pyexpr.dt_millisecond())

    def microsecond(self) -> Expr:
        """
        Extract microseconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1),
        ...             datetime(2020, 1, 1, 0, 0, 1, 0),
        ...             "1ms",
        ...             eager=True,
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.microsecond().alias("microseconds"),
        ...     ]
        ... )
        shape: (1_001, 2)
        ┌─────────────────────────┬──────────────┐
        │ date                    ┆ microseconds │
        │ ---                     ┆ ---          │
        │ datetime[μs]            ┆ u32          │
        ╞═════════════════════════╪══════════════╡
        │ 2020-01-01 00:00:00     ┆ 0            │
        │ 2020-01-01 00:00:00.001 ┆ 1000         │
        │ 2020-01-01 00:00:00.002 ┆ 2000         │
        │ 2020-01-01 00:00:00.003 ┆ 3000         │
        │ …                       ┆ …            │
        │ 2020-01-01 00:00:00.997 ┆ 997000       │
        │ 2020-01-01 00:00:00.998 ┆ 998000       │
        │ 2020-01-01 00:00:00.999 ┆ 999000       │
        │ 2020-01-01 00:00:01     ┆ 0            │
        └─────────────────────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.dt_microsecond())

    def nanosecond(self) -> Expr:
        """
        Extract nanoseconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        """
        return wrap_expr(self._pyexpr.dt_nanosecond())

    def epoch(self, time_unit: EpochTimeUnit = "us") -> Expr:
        """
        Get the time passed since the Unix EPOCH in the give time unit.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms', 's', 'd'}
            Time unit.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.date_range(
        ...     date(2001, 1, 1), date(2001, 1, 3), eager=True
        ... ).to_frame()
        >>> df.with_columns(
        ...     pl.col("date").dt.epoch().alias("epoch_ns"),
        ...     pl.col("date").dt.epoch(time_unit="s").alias("epoch_s"),
        ... )
        shape: (3, 3)
        ┌────────────┬─────────────────┬───────────┐
        │ date       ┆ epoch_ns        ┆ epoch_s   │
        │ ---        ┆ ---             ┆ ---       │
        │ date       ┆ i64             ┆ i64       │
        ╞════════════╪═════════════════╪═══════════╡
        │ 2001-01-01 ┆ 978307200000000 ┆ 978307200 │
        │ 2001-01-02 ┆ 978393600000000 ┆ 978393600 │
        │ 2001-01-03 ┆ 978480000000000 ┆ 978480000 │
        └────────────┴─────────────────┴───────────┘

        """
        if time_unit in DTYPE_TEMPORAL_UNITS:
            return self.timestamp(time_unit)  # type: ignore[arg-type]
        elif time_unit == "s":
            return wrap_expr(self._pyexpr.dt_epoch_seconds())
        elif time_unit == "d":
            return wrap_expr(self._pyexpr).cast(Date).cast(Int32)
        else:
            raise ValueError(
                f"`time_unit` must be one of {{'ns', 'us', 'ms', 's', 'd'}}, got {time_unit!r}"
            )

    def timestamp(self, time_unit: TimeUnit = "us") -> Expr:
        """
        Return a timestamp in the given time unit.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            Time unit.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.date_range(
        ...     date(2001, 1, 1), date(2001, 1, 3), eager=True
        ... ).to_frame()
        >>> df.with_columns(
        ...     pl.col("date").dt.timestamp().alias("timestamp_ns"),
        ...     pl.col("date").dt.timestamp("ms").alias("timestamp_ms"),
        ... )
        shape: (3, 3)
        ┌────────────┬─────────────────┬──────────────┐
        │ date       ┆ timestamp_ns    ┆ timestamp_ms │
        │ ---        ┆ ---             ┆ ---          │
        │ date       ┆ i64             ┆ i64          │
        ╞════════════╪═════════════════╪══════════════╡
        │ 2001-01-01 ┆ 978307200000000 ┆ 978307200000 │
        │ 2001-01-02 ┆ 978393600000000 ┆ 978393600000 │
        │ 2001-01-03 ┆ 978480000000000 ┆ 978480000000 │
        └────────────┴─────────────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.dt_timestamp(time_unit))

    def with_time_unit(self, time_unit: TimeUnit) -> Expr:
        """
        Set time unit of an expression of dtype Datetime or Duration.

        This does not modify underlying data, and should be used to fix an incorrect
        time unit.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            Unit of time for the `Datetime` expression.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2001, 1, 1),
        ...             datetime(2001, 1, 3),
        ...             "1d",
        ...             time_unit="ns",
        ...             eager=True,
        ...         )
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.with_time_unit("us").alias("time_unit_us"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────────────────────┬───────────────────────┐
        │ date                ┆ time_unit_us          │
        │ ---                 ┆ ---                   │
        │ datetime[ns]        ┆ datetime[μs]          │
        ╞═════════════════════╪═══════════════════════╡
        │ 2001-01-01 00:00:00 ┆ +32971-04-28 00:00:00 │
        │ 2001-01-02 00:00:00 ┆ +32974-01-22 00:00:00 │
        │ 2001-01-03 00:00:00 ┆ +32976-10-18 00:00:00 │
        └─────────────────────┴───────────────────────┘

        """
        return wrap_expr(self._pyexpr.dt_with_time_unit(time_unit))

    def cast_time_unit(self, time_unit: TimeUnit) -> Expr:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            Time unit for the `Datetime` expression.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2001, 1, 1), datetime(2001, 1, 3), "1d", eager=True
        ...         )
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").dt.cast_time_unit("ms").alias("time_unit_ms"),
        ...         pl.col("date").dt.cast_time_unit("ns").alias("time_unit_ns"),
        ...     ]
        ... )
        shape: (3, 3)
        ┌─────────────────────┬─────────────────────┬─────────────────────┐
        │ date                ┆ time_unit_ms        ┆ time_unit_ns        │
        │ ---                 ┆ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[ms]        ┆ datetime[ns]        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╡
        │ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 ┆ 2001-01-01 00:00:00 │
        │ 2001-01-02 00:00:00 ┆ 2001-01-02 00:00:00 ┆ 2001-01-02 00:00:00 │
        │ 2001-01-03 00:00:00 ┆ 2001-01-03 00:00:00 ┆ 2001-01-03 00:00:00 │
        └─────────────────────┴─────────────────────┴─────────────────────┘

        """
        return wrap_expr(self._pyexpr.dt_cast_time_unit(time_unit))

    def convert_time_zone(self, time_zone: str) -> Expr:
        """
        Convert to given time zone for an expression of type Datetime.

        Parameters
        ----------
        time_zone
            Time zone for the `Datetime` expression.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 3, 1),
        ...             datetime(2020, 5, 1),
        ...             "1mo",
        ...             time_zone="UTC",
        ...             eager=True,
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date")
        ...         .dt.convert_time_zone(time_zone="Europe/London")
        ...         .alias("London"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────────────────────────┬─────────────────────────────┐
        │ date                    ┆ London                      │
        │ ---                     ┆ ---                         │
        │ datetime[μs, UTC]       ┆ datetime[μs, Europe/London] │
        ╞═════════════════════════╪═════════════════════════════╡
        │ 2020-03-01 00:00:00 UTC ┆ 2020-03-01 00:00:00 GMT     │
        │ 2020-04-01 00:00:00 UTC ┆ 2020-04-01 01:00:00 BST     │
        │ 2020-05-01 00:00:00 UTC ┆ 2020-05-01 01:00:00 BST     │
        └─────────────────────────┴─────────────────────────────┘
        """
        return wrap_expr(self._pyexpr.dt_convert_time_zone(time_zone))

    def replace_time_zone(
        self,
        time_zone: str | None,
        *,
        use_earliest: bool | None = None,
        ambiguous: Ambiguous | Expr = "raise",
    ) -> Expr:
        """
        Replace time zone for an expression of type Datetime.

        Different from `convert_time_zone`, this will also modify
        the underlying timestamp and will ignore the original time zone.

        Parameters
        ----------
        time_zone
            Time zone for the `Datetime` expression. Pass `None` to unset time zone.
        use_earliest
            Determine how to deal with ambiguous datetimes:

            - `None` (default): raise
            - `True`: use the earliest datetime
            - `False`: use the latest datetime

            .. deprecated:: 0.19.0
                Use `ambiguous` instead
        ambiguous
            Determine how to deal with ambiguous datetimes:

            - `'raise'` (default): raise
            - `'earliest'`: use the earliest datetime
            - `'latest'`: use the latest datetime

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "london_timezone": pl.datetime_range(
        ...             datetime(2020, 3, 1),
        ...             datetime(2020, 7, 1),
        ...             "1mo",
        ...             time_zone="UTC",
        ...             eager=True,
        ...         ).dt.convert_time_zone(time_zone="Europe/London"),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("london_timezone"),
        ...         pl.col("london_timezone")
        ...         .dt.replace_time_zone(time_zone="Europe/Amsterdam")
        ...         .alias("London_to_Amsterdam"),
        ...     ]
        ... )
        shape: (5, 2)
        ┌─────────────────────────────┬────────────────────────────────┐
        │ london_timezone             ┆ London_to_Amsterdam            │
        │ ---                         ┆ ---                            │
        │ datetime[μs, Europe/London] ┆ datetime[μs, Europe/Amsterdam] │
        ╞═════════════════════════════╪════════════════════════════════╡
        │ 2020-03-01 00:00:00 GMT     ┆ 2020-03-01 00:00:00 CET        │
        │ 2020-04-01 01:00:00 BST     ┆ 2020-04-01 01:00:00 CEST       │
        │ 2020-05-01 01:00:00 BST     ┆ 2020-05-01 01:00:00 CEST       │
        │ 2020-06-01 01:00:00 BST     ┆ 2020-06-01 01:00:00 CEST       │
        │ 2020-07-01 01:00:00 BST     ┆ 2020-07-01 01:00:00 CEST       │
        └─────────────────────────────┴────────────────────────────────┘

        You can use `use_earliest` to deal with ambiguous datetimes:

        >>> dates = [
        ...     "2018-10-28 01:30",
        ...     "2018-10-28 02:00",
        ...     "2018-10-28 02:30",
        ...     "2018-10-28 02:00",
        ... ]
        >>> df = pl.DataFrame(
        ...     {
        ...         "ts": pl.Series(dates).str.strptime(pl.Datetime),
        ...         "ambiguous": ["earliest", "earliest", "latest", "latest"],
        ...     }
        ... )
        >>> df.with_columns(
        ...     ts_localized=pl.col("ts").dt.replace_time_zone(
        ...         "Europe/Brussels", ambiguous=pl.col("ambiguous")
        ...     )
        ... )
        shape: (4, 3)
        ┌─────────────────────┬───────────┬───────────────────────────────┐
        │ ts                  ┆ ambiguous ┆ ts_localized                  │
        │ ---                 ┆ ---       ┆ ---                           │
        │ datetime[μs]        ┆ str       ┆ datetime[μs, Europe/Brussels] │
        ╞═════════════════════╪═══════════╪═══════════════════════════════╡
        │ 2018-10-28 01:30:00 ┆ earliest  ┆ 2018-10-28 01:30:00 CEST      │
        │ 2018-10-28 02:00:00 ┆ earliest  ┆ 2018-10-28 02:00:00 CEST      │
        │ 2018-10-28 02:30:00 ┆ latest    ┆ 2018-10-28 02:30:00 CET       │
        │ 2018-10-28 02:00:00 ┆ latest    ┆ 2018-10-28 02:00:00 CET       │
        └─────────────────────┴───────────┴───────────────────────────────┘

        """
        ambiguous = rename_use_earliest_to_ambiguous(use_earliest, ambiguous)
        if not isinstance(ambiguous, pl.Expr):
            ambiguous = F.lit(ambiguous)
        return wrap_expr(
            self._pyexpr.dt_replace_time_zone(time_zone, ambiguous._pyexpr)
        )

    def total_days(self) -> Expr:
        """
        Extract the total days from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo", eager=True
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.total_days().alias("days_diff"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────────────────────┬───────────┐
        │ date                ┆ days_diff │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ i64       │
        ╞═════════════════════╪═══════════╡
        │ 2020-03-01 00:00:00 ┆ null      │
        │ 2020-04-01 00:00:00 ┆ 31        │
        │ 2020-05-01 00:00:00 ┆ 30        │
        └─────────────────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_days())

    def total_hours(self) -> Expr:
        """
        Extract the total hours from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 4), "1d", eager=True
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.total_hours().alias("hours_diff"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────────────────────┬────────────┐
        │ date                ┆ hours_diff │
        │ ---                 ┆ ---        │
        │ datetime[μs]        ┆ i64        │
        ╞═════════════════════╪════════════╡
        │ 2020-01-01 00:00:00 ┆ null       │
        │ 2020-01-02 00:00:00 ┆ 24         │
        │ 2020-01-03 00:00:00 ┆ 24         │
        │ 2020-01-04 00:00:00 ┆ 24         │
        └─────────────────────┴────────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_hours())

    def total_minutes(self) -> Expr:
        """
        Extract the total minutes from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1), datetime(2020, 1, 4), "1d", eager=True
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("date"),
        ...         pl.col("date").diff().dt.total_minutes().alias("minutes_diff"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────────────────────┬──────────────┐
        │ date                ┆ minutes_diff │
        │ ---                 ┆ ---          │
        │ datetime[μs]        ┆ i64          │
        ╞═════════════════════╪══════════════╡
        │ 2020-01-01 00:00:00 ┆ null         │
        │ 2020-01-02 00:00:00 ┆ 1440         │
        │ 2020-01-03 00:00:00 ┆ 1440         │
        │ 2020-01-04 00:00:00 ┆ 1440         │
        └─────────────────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_minutes())

    def total_seconds(self) -> Expr:
        """
        Extract the total seconds from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1),
        ...             datetime(2020, 1, 1, 0, 4, 0),
        ...             "1m",
        ...             eager=True,
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("date"),
        ...     pl.col("date").diff().dt.total_seconds().alias("seconds_diff"),
        ... )
        shape: (5, 2)
        ┌─────────────────────┬──────────────┐
        │ date                ┆ seconds_diff │
        │ ---                 ┆ ---          │
        │ datetime[μs]        ┆ i64          │
        ╞═════════════════════╪══════════════╡
        │ 2020-01-01 00:00:00 ┆ null         │
        │ 2020-01-01 00:01:00 ┆ 60           │
        │ 2020-01-01 00:02:00 ┆ 60           │
        │ 2020-01-01 00:03:00 ┆ 60           │
        │ 2020-01-01 00:04:00 ┆ 60           │
        └─────────────────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_seconds())

    def total_milliseconds(self) -> Expr:
        """
        Extract the total milliseconds from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1),
        ...             datetime(2020, 1, 1, 0, 0, 1, 0),
        ...             "1ms",
        ...             eager=True,
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("date"),
        ...     milliseconds_diff=pl.col("date").diff().dt.total_milliseconds(),
        ... )
        shape: (1_001, 2)
        ┌─────────────────────────┬───────────────────┐
        │ date                    ┆ milliseconds_diff │
        │ ---                     ┆ ---               │
        │ datetime[μs]            ┆ i64               │
        ╞═════════════════════════╪═══════════════════╡
        │ 2020-01-01 00:00:00     ┆ null              │
        │ 2020-01-01 00:00:00.001 ┆ 1                 │
        │ 2020-01-01 00:00:00.002 ┆ 1                 │
        │ 2020-01-01 00:00:00.003 ┆ 1                 │
        │ …                       ┆ …                 │
        │ 2020-01-01 00:00:00.997 ┆ 1                 │
        │ 2020-01-01 00:00:00.998 ┆ 1                 │
        │ 2020-01-01 00:00:00.999 ┆ 1                 │
        │ 2020-01-01 00:00:01     ┆ 1                 │
        └─────────────────────────┴───────────────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_milliseconds())

    def total_microseconds(self) -> Expr:
        """
        Extract the total microseconds from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1),
        ...             datetime(2020, 1, 1, 0, 0, 1, 0),
        ...             "1ms",
        ...             eager=True,
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("date"),
        ...     microseconds_diff=pl.col("date").diff().dt.total_microseconds(),
        ... )
        shape: (1_001, 2)
        ┌─────────────────────────┬───────────────────┐
        │ date                    ┆ microseconds_diff │
        │ ---                     ┆ ---               │
        │ datetime[μs]            ┆ i64               │
        ╞═════════════════════════╪═══════════════════╡
        │ 2020-01-01 00:00:00     ┆ null              │
        │ 2020-01-01 00:00:00.001 ┆ 1000              │
        │ 2020-01-01 00:00:00.002 ┆ 1000              │
        │ 2020-01-01 00:00:00.003 ┆ 1000              │
        │ …                       ┆ …                 │
        │ 2020-01-01 00:00:00.997 ┆ 1000              │
        │ 2020-01-01 00:00:00.998 ┆ 1000              │
        │ 2020-01-01 00:00:00.999 ┆ 1000              │
        │ 2020-01-01 00:00:01     ┆ 1000              │
        └─────────────────────────┴───────────────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_microseconds())

    def total_nanoseconds(self) -> Expr:
        """
        Extract the total nanoseconds from a Duration type.

        Returns
        -------
        Expr
            Expression of data type :class:`Int64`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "date": pl.datetime_range(
        ...             datetime(2020, 1, 1),
        ...             datetime(2020, 1, 1, 0, 0, 1, 0),
        ...             "1ms",
        ...             eager=True,
        ...         ),
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("date"),
        ...     nanoseconds_diff=pl.col("date").diff().dt.total_nanoseconds(),
        ... )
        shape: (1_001, 2)
        ┌─────────────────────────┬──────────────────┐
        │ date                    ┆ nanoseconds_diff │
        │ ---                     ┆ ---              │
        │ datetime[μs]            ┆ i64              │
        ╞═════════════════════════╪══════════════════╡
        │ 2020-01-01 00:00:00     ┆ null             │
        │ 2020-01-01 00:00:00.001 ┆ 1000000          │
        │ 2020-01-01 00:00:00.002 ┆ 1000000          │
        │ 2020-01-01 00:00:00.003 ┆ 1000000          │
        │ …                       ┆ …                │
        │ 2020-01-01 00:00:00.997 ┆ 1000000          │
        │ 2020-01-01 00:00:00.998 ┆ 1000000          │
        │ 2020-01-01 00:00:00.999 ┆ 1000000          │
        │ 2020-01-01 00:00:01     ┆ 1000000          │
        └─────────────────────────┴──────────────────┘

        """
        return wrap_expr(self._pyexpr.dt_total_nanoseconds())

    def offset_by(self, by: str | Expr) -> Expr:
        """
        Offset this date by a relative time offset.

        This differs from `pl.col("foo") + timedelta` in that it can
        take months and leap years into account. Note that only a single minus
        sign is allowed in the `by` string, as the first character.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            By "calendar day", we mean the corresponding time on the next day (which may
            not be 24 hours, due to daylight savings). Similarly for "calendar week",
            "calendar month", "calendar quarter", and "calendar year".

        Returns
        -------
        Expr
            Expression of data type :class:`Date` or :class:`Datetime`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "dates": pl.datetime_range(
        ...             datetime(2000, 1, 1), datetime(2005, 1, 1), "1y", eager=True
        ...         ),
        ...         "offset": ["1d", "2d", "-1d", "1mo", None, "1y"],
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
        │ 2002-01-01 00:00:00 ┆ 1999-11-01 00:00:00 │
        │ 2003-01-01 00:00:00 ┆ 2000-11-01 00:00:00 │
        │ 2004-01-01 00:00:00 ┆ 2001-11-01 00:00:00 │
        │ 2005-01-01 00:00:00 ┆ 2002-11-01 00:00:00 │
        │ 2006-01-01 00:00:00 ┆ 2003-11-01 00:00:00 │
        └─────────────────────┴─────────────────────┘

        You can also pass the relative offset as an expression:

        >>> df.with_columns(new_dates=pl.col("dates").dt.offset_by(pl.col("offset")))
        shape: (6, 3)
        ┌─────────────────────┬────────┬─────────────────────┐
        │ dates               ┆ offset ┆ new_dates           │
        │ ---                 ┆ ---    ┆ ---                 │
        │ datetime[μs]        ┆ str    ┆ datetime[μs]        │
        ╞═════════════════════╪════════╪═════════════════════╡
        │ 2000-01-01 00:00:00 ┆ 1d     ┆ 2000-01-02 00:00:00 │
        │ 2001-01-01 00:00:00 ┆ 2d     ┆ 2001-01-03 00:00:00 │
        │ 2002-01-01 00:00:00 ┆ -1d    ┆ 2001-12-31 00:00:00 │
        │ 2003-01-01 00:00:00 ┆ 1mo    ┆ 2003-02-01 00:00:00 │
        │ 2004-01-01 00:00:00 ┆ null   ┆ null                │
        │ 2005-01-01 00:00:00 ┆ 1y     ┆ 2006-01-01 00:00:00 │
        └─────────────────────┴────────┴─────────────────────┘
        """
        by = deprecate_saturating(by)
        by = parse_as_expression(by, str_as_lit=True)
        return wrap_expr(self._pyexpr.dt_offset_by(by))

    def month_start(self) -> Expr:
        """
        Roll backward to the first day of the month.

        Returns
        -------
        Expr
            Expression of data type :class:`Date` or :class:`Datetime`.

        Notes
        -----
        If you're coming from pandas, you can think of this as a vectorised version
        of `pandas.tseries.offsets.MonthBegin().rollback(datetime)`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "dates": pl.datetime_range(
        ...             datetime(2000, 1, 15, 2),
        ...             datetime(2000, 12, 15, 2),
        ...             "1mo",
        ...             eager=True,
        ...         )
        ...     }
        ... )
        >>> df.select(pl.col("dates").dt.month_start())
        shape: (12, 1)
        ┌─────────────────────┐
        │ dates               │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2000-01-01 02:00:00 │
        │ 2000-02-01 02:00:00 │
        │ 2000-03-01 02:00:00 │
        │ 2000-04-01 02:00:00 │
        │ …                   │
        │ 2000-09-01 02:00:00 │
        │ 2000-10-01 02:00:00 │
        │ 2000-11-01 02:00:00 │
        │ 2000-12-01 02:00:00 │
        └─────────────────────┘
        """
        return wrap_expr(self._pyexpr.dt_month_start())

    def month_end(self) -> Expr:
        """
        Roll forward to the last day of the month.

        Returns
        -------
        Expr
            Expression of data type :class:`Date` or :class:`Datetime`.

        Notes
        -----
        If you're coming from pandas, you can think of this as a vectorised version
        of `pandas.tseries.offsets.MonthEnd().rollforward(datetime)`.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "dates": pl.datetime_range(
        ...             datetime(2000, 1, 1, 2),
        ...             datetime(2000, 12, 1, 2),
        ...             "1mo",
        ...             eager=True,
        ...         )
        ...     }
        ... )
        >>> df.select(pl.col("dates").dt.month_end())
        shape: (12, 1)
        ┌─────────────────────┐
        │ dates               │
        │ ---                 │
        │ datetime[μs]        │
        ╞═════════════════════╡
        │ 2000-01-31 02:00:00 │
        │ 2000-02-29 02:00:00 │
        │ 2000-03-31 02:00:00 │
        │ 2000-04-30 02:00:00 │
        │ …                   │
        │ 2000-09-30 02:00:00 │
        │ 2000-10-31 02:00:00 │
        │ 2000-11-30 02:00:00 │
        │ 2000-12-31 02:00:00 │
        └─────────────────────┘
        """
        return wrap_expr(self._pyexpr.dt_month_end())

    def base_utc_offset(self) -> Expr:
        """
        Base offset from UTC.

        This is usually constant for all datetimes in a given time zone, but
        may vary in the rare case that a country switches time zone, like
        Samoa (Apia) did at the end of 2011.

        Returns
        -------
        Expr
            Expression of data type :class:`Duration`.

        See Also
        --------
        Expr.dt.dst_offset : Daylight savings offset from UTC.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "ts": [datetime(2011, 12, 29), datetime(2012, 1, 1)],
        ...     }
        ... )
        >>> df = df.with_columns(pl.col("ts").dt.replace_time_zone("Pacific/Apia"))
        >>> df.with_columns(pl.col("ts").dt.base_utc_offset().alias("base_utc_offset"))
        shape: (2, 2)
        ┌────────────────────────────┬─────────────────┐
        │ ts                         ┆ base_utc_offset │
        │ ---                        ┆ ---             │
        │ datetime[μs, Pacific/Apia] ┆ duration[ms]    │
        ╞════════════════════════════╪═════════════════╡
        │ 2011-12-29 00:00:00 -10    ┆ -11h            │
        │ 2012-01-01 00:00:00 +14    ┆ 13h             │
        └────────────────────────────┴─────────────────┘
        """
        return wrap_expr(self._pyexpr.dt_base_utc_offset())

    def dst_offset(self) -> Expr:
        """
        Additional offset currently in effect (typically due to daylight saving time).

        Returns
        -------
        Expr
            Expression of data type :class:`Duration`.

        See Also
        --------
        Expr.dt.base_utc_offset : Base offset from UTC.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "ts": [datetime(2020, 10, 25), datetime(2020, 10, 26)],
        ...     }
        ... )
        >>> df = df.with_columns(pl.col("ts").dt.replace_time_zone("Europe/London"))
        >>> df.with_columns(pl.col("ts").dt.dst_offset().alias("dst_offset"))
        shape: (2, 2)
        ┌─────────────────────────────┬──────────────┐
        │ ts                          ┆ dst_offset   │
        │ ---                         ┆ ---          │
        │ datetime[μs, Europe/London] ┆ duration[ms] │
        ╞═════════════════════════════╪══════════════╡
        │ 2020-10-25 00:00:00 BST     ┆ 1h           │
        │ 2020-10-26 00:00:00 GMT     ┆ 0ms          │
        └─────────────────────────────┴──────────────┘
        """
        return wrap_expr(self._pyexpr.dt_dst_offset())

    @deprecate_renamed_function("total_days", version="0.19.13")
    def days(self) -> Expr:
        """
        Extract the total days from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_days` instead.

        """
        return self.total_days()

    @deprecate_renamed_function("total_hours", version="0.19.13")
    def hours(self) -> Expr:
        """
        Extract the total hours from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_hours` instead.

        """
        return self.total_hours()

    @deprecate_renamed_function("total_minutes", version="0.19.13")
    def minutes(self) -> Expr:
        """
        Extract the total minutes from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_minutes` instead.

        """
        return self.total_minutes()

    @deprecate_renamed_function("total_seconds", version="0.19.13")
    def seconds(self) -> Expr:
        """
        Extract the total seconds from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_seconds` instead.

        """
        return self.total_seconds()

    @deprecate_renamed_function("total_milliseconds", version="0.19.13")
    def milliseconds(self) -> Expr:
        """
        Extract the total milliseconds from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_milliseconds` instead.

        """
        return self.total_milliseconds()

    @deprecate_renamed_function("total_microseconds", version="0.19.13")
    def microseconds(self) -> Expr:
        """
        Extract the total microseconds from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_microseconds` instead.

        """
        return self.total_microseconds()

    @deprecate_renamed_function("total_nanoseconds", version="0.19.13")
    def nanoseconds(self) -> Expr:
        """
        Extract the total nanoseconds from a Duration type.

        .. deprecated:: 0.19.13
            Use :meth:`total_nanoseconds` instead.

        """
        return self.total_nanoseconds()
