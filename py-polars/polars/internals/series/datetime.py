from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import DTYPE_TEMPORAL_UNITS, Date, Int32
from polars.utils import _to_python_datetime

if TYPE_CHECKING:
    from polars.internals.type_aliases import EpochTimeUnit, TimeUnit


class DateTimeNameSpace:
    """Series.dt namespace."""

    def __init__(self, series: pli.Series):
        self._s = series._s

    def truncate(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> pli.Series:
        """
        .. warning::
            This API is experimental and may change without it being considered a
            breaking change.

        Divide the date/ datetime range into buckets.

        The `every` and `offset` argument are created with the
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
        return pli.select(
            pli.lit(pli.wrap_s(self._s)).dt.truncate(every, offset)
        ).to_series()

    def __getitem__(self, item: int) -> date | datetime:
        s = pli.wrap_s(self._s)
        return s[item]

    def strftime(self, fmt: str) -> pli.Series:
        """
        Format Date/datetime with a formatting rule:

        See `chrono strftime/strptime
        <https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html>`_.

        Returns
        -------
        Utf8 Series

        """
        return pli.wrap_s(self._s.strftime(fmt))

    def year(self) -> pli.Series:
        """
        Extract the year from the underlying date representation.
        Can be performed on Date and Datetime.

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32

        """
        return pli.wrap_s(self._s.year())

    def quarter(self) -> pli.Series:
        """
        Extract quarter from underlying Date representation.
        Can be performed on Date and Datetime.

        Returns the quarter ranging from 1 to 4.

        Returns
        -------
        Quarter as UInt32

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.quarter()).to_series()

    def month(self) -> pli.Series:
        """
        Extract the month from the underlying date representation.
        Can be performed on Date and Datetime

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32

        """
        return pli.wrap_s(self._s.month())

    def week(self) -> pli.Series:
        """
        Extract the week from the underlying date representation.
        Can be performed on Date and Datetime

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32

        """
        return pli.wrap_s(self._s.week())

    def weekday(self) -> pli.Series:
        """
        Extract the week day from the underlying date representation.
        Can be performed on Date and Datetime.

        Returns the weekday number where monday = 0 and sunday = 6

        Returns
        -------
        Week day as UInt32

        """
        return pli.wrap_s(self._s.weekday())

    def day(self) -> pli.Series:
        """
        Extract the day from the underlying date representation.
        Can be performed on Date and Datetime.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32

        """
        return pli.wrap_s(self._s.day())

    def ordinal_day(self) -> pli.Series:
        """
        Extract ordinal day from underlying date representation.
        Can be performed on Date and Datetime.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32

        """
        return pli.wrap_s(self._s.ordinal_day())

    def hour(self) -> pli.Series:
        """
        Extract the hour from the underlying DateTime representation.
        Can be performed on Datetime.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32

        """
        return pli.wrap_s(self._s.hour())

    def minute(self) -> pli.Series:
        """
        Extract the minutes from the underlying DateTime representation.
        Can be performed on Datetime.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32

        """
        return pli.wrap_s(self._s.minute())

    def second(self) -> pli.Series:
        """
        Extract the seconds the from underlying DateTime representation.
        Can be performed on Datetime.

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32

        """
        return pli.wrap_s(self._s.second())

    def nanosecond(self) -> pli.Series:
        """
        Extract the nanoseconds from the underlying DateTime representation.
        Can be performed on Datetime.

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32

        """
        return pli.wrap_s(self._s.nanosecond())

    def timestamp(self, tu: TimeUnit = "us") -> pli.Series:
        """
        Return a timestamp in the given time unit.

        Parameters
        ----------
        tu : {'us', 'ns', 'ms'}
            Time unit.

        """
        return pli.wrap_s(self._s.timestamp(tu))

    def min(self) -> date | datetime | timedelta:
        """Return minimum as python DateTime."""
        # we can ignore types because we are certain we get a logical type
        return pli.wrap_s(self._s).min()  # type: ignore[return-value]

    def max(self) -> date | datetime | timedelta:
        """Return maximum as python DateTime."""
        return pli.wrap_s(self._s).max()  # type: ignore[return-value]

    def median(self) -> date | datetime | timedelta:
        """Return median as python DateTime."""
        s = pli.wrap_s(self._s)
        out = int(s.median())
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def mean(self) -> date | datetime:
        """Return mean as python DateTime."""
        s = pli.wrap_s(self._s)
        out = int(s.mean())
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def epoch(self, tu: EpochTimeUnit = "us") -> pli.Series:
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
            return pli.wrap_s(self._s.dt_epoch_seconds())
        elif tu == "d":
            return pli.wrap_s(self._s).cast(Date).cast(Int32)
        else:
            raise ValueError(
                f"tu must be one of {{'ns', 'us', 'ms', 's', 'd'}}, got {tu}"
            )

    def with_time_unit(self, tu: TimeUnit) -> pli.Series:
        """
        Set time unit a Series of dtype Datetime or Duration. This does not modify
        underlying data, and should be used to fix an incorrect time unit.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        """
        return pli.select(
            pli.lit(pli.wrap_s(self._s)).dt.with_time_unit(tu)
        ).to_series()

    def cast_time_unit(self, tu: TimeUnit) -> pli.Series:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        """
        return pli.select(
            pli.lit(pli.wrap_s(self._s)).dt.cast_time_unit(tu)
        ).to_series()

    def with_time_zone(self, tz: str | None) -> pli.Series:
        """
        Set time zone a Series of type Datetime.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """
        return pli.wrap_s(self._s.with_time_zone(tz))

    def days(self) -> pli.Series:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.days()).to_series()

    def hours(self) -> pli.Series:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.hours()).to_series()

    def minutes(self) -> pli.Series:
        """
        Extract the minutes from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.minutes()).to_series()

    def seconds(self) -> pli.Series:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.seconds()).to_series()

    def milliseconds(self) -> pli.Series:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.milliseconds()).to_series()

    def nanoseconds(self) -> pli.Series:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        """
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.nanoseconds()).to_series()

    def offset_by(self, by: str) -> pli.Series:
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
        return pli.select(pli.lit(pli.wrap_s(self._s)).dt.offset_by(by)).to_series()
