from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import DTYPE_TEMPORAL_UNITS, Date, Datetime, Int32
from polars.utils import _timedelta_to_pl_duration

if TYPE_CHECKING:
    from polars.internals.type_aliases import EpochTimeUnit, TimeUnit


class ExprDateTimeNameSpace:
    """Namespace for datetime related expressions."""

    def __init__(self, expr: pli.Expr):
        self._pyexpr = expr._pyexpr

    def truncate(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> pli.Expr:
        """
        Divide the date/ datetime range into buckets.

        The `every` and `offset` arguments are created with
        the following string language:

        1ns # 1 nanosecond
        1us # 1 microsecond
        1ms # 1 millisecond
        1s  # 1 second
        1m  # 1 minute
        1h  # 1 hour
        1d  # 1 day
        1w  # 1 week
        1mo # 1 calendar month
        1y  # 1 calendar year

        3d12h4m25s # 3 days, 12 hours, 4 minutes, and 25 seconds

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

        Returns
        -------
        Date/Datetime series

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> s = pl.date_range(start, stop, timedelta(minutes=30), name="dates")
        >>> s
        shape: (49,)
        Series: 'dates' [datetime[ns]]
        [
            2001-01-01 00:00:00
            2001-01-01 00:30:00
            2001-01-01 01:00:00
            2001-01-01 01:30:00
            2001-01-01 02:00:00
            2001-01-01 02:30:00
            2001-01-01 03:00:00
            2001-01-01 03:30:00
            2001-01-01 04:00:00
            2001-01-01 04:30:00
            2001-01-01 05:00:00
            2001-01-01 05:30:00
            ...
            2001-01-01 18:30:00
            2001-01-01 19:00:00
            2001-01-01 19:30:00
            2001-01-01 20:00:00
            2001-01-01 20:30:00
            2001-01-01 21:00:00
            2001-01-01 21:30:00
            2001-01-01 22:00:00
            2001-01-01 22:30:00
            2001-01-01 23:00:00
            2001-01-01 23:30:00
            2001-01-02 00:00:00
        ]
        >>> s.dt.truncate("1h")
        shape: (49,)
        Series: 'dates' [datetime[ns]]
        [
            2001-01-01 00:00:00
            2001-01-01 00:00:00
            2001-01-01 01:00:00
            2001-01-01 01:00:00
            2001-01-01 02:00:00
            2001-01-01 02:00:00
            2001-01-01 03:00:00
            2001-01-01 03:00:00
            2001-01-01 04:00:00
            2001-01-01 04:00:00
            2001-01-01 05:00:00
            2001-01-01 05:00:00
            ...
            2001-01-01 18:00:00
            2001-01-01 19:00:00
            2001-01-01 19:00:00
            2001-01-01 20:00:00
            2001-01-01 20:00:00
            2001-01-01 21:00:00
            2001-01-01 21:00:00
            2001-01-01 22:00:00
            2001-01-01 22:00:00
            2001-01-01 23:00:00
            2001-01-01 23:00:00
            2001-01-02 00:00:00
        ]
        >>> assert s.dt.truncate("1h") == s.dt.truncate(timedelta(hours=1))

        """
        if offset is None:
            offset = "0ns"
        if isinstance(every, timedelta):
            every = _timedelta_to_pl_duration(every)
        if isinstance(offset, timedelta):
            offset = _timedelta_to_pl_duration(offset)
        return pli.wrap_expr(self._pyexpr.date_truncate(every, offset))

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

        Can be performed on Date and Datetime.

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32

        """
        return pli.wrap_expr(self._pyexpr.year())

    def quarter(self) -> pli.Expr:
        """
        Extract quarter from underlying Date representation.

        Can be performed on Date and Datetime.

        Returns the quarter ranging from 1 to 4.

        Returns
        -------
        Quarter as UInt32

        """
        return pli.wrap_expr(self._pyexpr.quarter())

    def month(self) -> pli.Expr:
        """
        Extract month from underlying Date representation.

        Can be performed on Date and Datetime.

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32

        """
        return pli.wrap_expr(self._pyexpr.month())

    def week(self) -> pli.Expr:
        """
        Extract the week from the underlying Date representation.

        Can be performed on Date and Datetime

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32

        """
        return pli.wrap_expr(self._pyexpr.week())

    def weekday(self) -> pli.Expr:
        """
        Extract the week day from the underlying Date representation.

        Can be performed on Date and Datetime.

        Returns the weekday number where monday = 0 and sunday = 6

        Returns
        -------
        Week day as UInt32

        """
        return pli.wrap_expr(self._pyexpr.weekday())

    def day(self) -> pli.Expr:
        """
        Extract day from underlying Date representation.

        Can be performed on Date and Datetime.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32

        """
        return pli.wrap_expr(self._pyexpr.day())

    def ordinal_day(self) -> pli.Expr:
        """
        Extract ordinal day from underlying Date representation.

        Can be performed on Date and Datetime.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32

        """
        return pli.wrap_expr(self._pyexpr.ordinal_day())

    def hour(self) -> pli.Expr:
        """
        Extract hour from underlying DateTime representation.

        Can be performed on Datetime.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32

        """
        return pli.wrap_expr(self._pyexpr.hour())

    def minute(self) -> pli.Expr:
        """
        Extract minutes from underlying DateTime representation.

        Can be performed on Datetime.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32

        """
        return pli.wrap_expr(self._pyexpr.minute())

    def second(self) -> pli.Expr:
        """
        Extract seconds from underlying DateTime representation.

        Can be performed on Datetime.

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32

        """
        return pli.wrap_expr(self._pyexpr.second())

    def nanosecond(self) -> pli.Expr:
        """
        Extract seconds from underlying DateTime representation.

        Can be performed on Datetime.

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32

        """
        return pli.wrap_expr(self._pyexpr.nanosecond())

    def epoch(self, tu: EpochTimeUnit = "us") -> pli.Expr:
        """
        Get the time passed since the Unix EPOCH in the give time unit

        Parameters
        ----------
        tu : {'us', 'ns', 'ms', 's', 'd'}
            Time unit.

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

        """
        return pli.wrap_expr(self._pyexpr.dt_with_time_unit(tu))

    def cast_time_unit(self, tu: TimeUnit) -> pli.Expr:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        """
        return pli.wrap_expr(self._pyexpr.dt_cast_time_unit(tu))

    def with_time_zone(self, tz: str | None) -> pli.Expr:
        """
        Set time zone for a Series of type Datetime.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """
        return pli.wrap_expr(self._pyexpr).map(
            lambda s: s.dt.with_time_zone(tz), return_dtype=Datetime
        )

    def days(self) -> pli.Expr:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.wrap_expr(self._pyexpr.duration_days())

    def hours(self) -> pli.Expr:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.wrap_expr(self._pyexpr.duration_hours())

    def minutes(self) -> pli.Expr:
        """
        Extract the minutes from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.wrap_expr(self._pyexpr.duration_minutes())

    def seconds(self) -> pli.Expr:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.wrap_expr(self._pyexpr.duration_seconds())

    def milliseconds(self) -> pli.Expr:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.wrap_expr(self._pyexpr.duration_milliseconds())

    def nanoseconds(self) -> pli.Expr:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

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

        """
        return pli.wrap_expr(self._pyexpr.dt_offset_by(by))
