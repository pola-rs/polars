from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import Date
from polars.series.utils import expr_dispatch
from polars.utils._wrap import wrap_s
from polars.utils.convert import _to_python_date, _to_python_datetime
from polars.utils.decorators import deprecated_alias

if TYPE_CHECKING:
    import datetime as dt

    from polars import Expr, Series
    from polars.polars import PySeries
    from polars.type_aliases import EpochTimeUnit, TimeUnit


@expr_dispatch
class DateTimeNameSpace:
    """Series.dt namespace."""

    _accessor = "dt"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def __getitem__(self, item: int) -> dt.date | dt.datetime:
        s = wrap_s(self._s)
        return s[item]

    def min(self) -> dt.date | dt.datetime | dt.timedelta | None:
        """
        Return minimum as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2001, 1, 1), datetime(2001, 1, 3), "1d", eager=True
        ... )
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.min()
        datetime.datetime(2001, 1, 1, 0, 0)

        """
        return wrap_s(self._s).min()  # type: ignore[return-value]

    def max(self) -> dt.date | dt.datetime | dt.timedelta | None:
        """
        Return maximum as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2001, 1, 1), datetime(2001, 1, 3), "1d", eager=True
        ... )
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.max()
        datetime.datetime(2001, 1, 3, 0, 0)

        """
        return wrap_s(self._s).max()  # type: ignore[return-value]

    def median(self) -> dt.date | dt.datetime | dt.timedelta | None:
        """
        Return median as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2001, 1, 1), datetime(2001, 1, 3), "1d", eager=True
        ... )
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.median()
        datetime.datetime(2001, 1, 2, 0, 0)

        """
        s = wrap_s(self._s)
        out = s.median()
        if out is not None:
            if s.dtype == Date:
                return _to_python_date(int(out))
            else:
                return _to_python_datetime(int(out), s._s.time_unit())
        return None

    def mean(self) -> dt.date | dt.datetime | None:
        """
        Return mean as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2001, 1, 1), datetime(2001, 1, 3), "1d", eager=True
        ... )
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.mean()
        datetime.datetime(2001, 1, 2, 0, 0)

        """
        s = wrap_s(self._s)
        out = s.mean()
        if out is not None:
            if s.dtype == Date:
                return _to_python_date(int(out))
            else:
                return _to_python_datetime(int(out), s._s.time_unit())
        return None

    def to_string(self, format: str) -> Series:
        """
        Convert a Date/Time/Datetime column into a Utf8 column with the given format.

        Similar to ``cast(pl.Utf8)``, but this method allows you to customize the
        formatting of the resulting string.

        Parameters
        ----------
        format
            Format to use, refer to the `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: ``"%y-%m-%d"``.

        Examples
        --------
        >>> from datetime import datetime
        >>> s = pl.Series(
        ...     "datetime",
        ...     [datetime(2020, 3, 1), datetime(2020, 4, 1), datetime(2020, 5, 1)],
        ... )
        >>> s.dt.to_string("%Y/%m/%d")
        shape: (3,)
        Series: 'datetime' [str]
        [
                "2020/03/01"
                "2020/04/01"
                "2020/05/01"
        ]

        """

    @deprecated_alias(fmt="format")
    def strftime(self, format: str) -> Series:
        """
        Convert a Date/Time/Datetime column into a Utf8 column with the given format.

        Similar to ``cast(pl.Utf8)``, but this method allows you to customize the
        formatting of the resulting string.

        Alias for :func:`to_string`.

        Parameters
        ----------
        format
            Format to use, refer to the `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: ``"%y-%m-%d"``.

        Examples
        --------
        >>> from datetime import datetime
        >>> s = pl.Series(
        ...     "datetime",
        ...     [datetime(2020, 3, 1), datetime(2020, 4, 1), datetime(2020, 5, 1)],
        ... )
        >>> s.dt.strftime("%Y/%m/%d")
        shape: (3,)
        Series: 'datetime' [str]
        [
                "2020/03/01"
                "2020/04/01"
                "2020/05/01"
        ]

        See Also
        --------
        to_string : The identical Series method for which ``strftime`` is an alias.

        """
        return self.to_string(format)

    def year(self) -> Series:
        """
        Extract the year from the underlying date representation.

        Applies to Date and Datetime columns.

        Returns the year number in the calendar date.

        Returns
        -------
        Year part as Int32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2002, 1, 1)
        >>> date = pl.date_range(start, stop, interval="1y", eager=True)
        >>> date
        shape: (2,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2002-01-01 00:00:00
        ]
        >>> date.dt.year()
        shape: (2,)
        Series: 'date' [i32]
        [
                2001
                2002
        ]

        """

    def is_leap_year(self) -> Series:
        """
        Determine whether the year of the underlying date representation is a leap year.

        Applies to Date and Datetime columns.

        Returns
        -------
        Leap year info as Boolean

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2000, 1, 1)
        >>> stop = datetime(2002, 1, 1)
        >>> date = pl.date_range(start, stop, interval="1y", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2000-01-01 00:00:00
                2001-01-01 00:00:00
                2002-01-01 00:00:00
        ]
        >>> date.dt.is_leap_year()
        shape: (3,)
        Series: 'date' [bool]
        [
                true
                false
                false
        ]

        """

    def iso_year(self) -> Series:
        """
        Extract ISO year from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the year number according to the ISO standard.
        This may not correspond with the calendar year.

        Returns
        -------
        ISO year as Int32

        Examples
        --------
        >>> from datetime import datetime
        >>> dt = datetime(2022, 1, 1, 7, 8, 40)
        >>> pl.Series([dt]).dt.iso_year()
        shape: (1,)
        Series: '' [i32]
        [
                2021
        ]

        """

    def quarter(self) -> Series:
        """
        Extract quarter from underlying Date representation.

        Applies to Date and Datetime columns.

        Returns the quarter ranging from 1 to 4.

        Returns
        -------
        Quarter as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 4, 1)
        >>> date = pl.date_range(start, stop, interval="1mo", eager=True)
        >>> date
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
                2001-04-01 00:00:00
        ]
        >>> date.dt.quarter()
        shape: (4,)
        Series: 'date' [u32]
        [
                1
                1
                1
                2
        ]

        """

    def month(self) -> Series:
        """
        Extract the month from the underlying date representation.

        Applies to Date and Datetime columns.

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 4, 1)
        >>> date = pl.date_range(start, stop, interval="1mo", eager=True)
        >>> date
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
                2001-04-01 00:00:00
        ]
        >>> date.dt.month()
        shape: (4,)
        Series: 'date' [u32]
        [
                1
                2
                3
                4
        ]

        """

    def week(self) -> Series:
        """
        Extract the week from the underlying date representation.

        Applies to Date and Datetime columns.

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 4, 1)
        >>> date = pl.date_range(start, stop, interval="1mo", eager=True)
        >>> date
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
                2001-04-01 00:00:00
        ]
        >>> date.dt.week()
        shape: (4,)
        Series: 'date' [u32]
        [
                1
                5
                9
                13
        ]

        """

    def weekday(self) -> Series:
        """
        Extract the week day from the underlying date representation.

        Applies to Date and Datetime columns.

        Returns the ISO weekday number where monday = 1 and sunday = 7

        Returns
        -------
        Weekday as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 7)
        >>> date = pl.date_range(start, stop, interval="1d", eager=True)
        >>> date
        shape: (7,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
                2001-01-04 00:00:00
                2001-01-05 00:00:00
                2001-01-06 00:00:00
                2001-01-07 00:00:00
        ]
        >>> date.dt.weekday()
        shape: (7,)
        Series: 'date' [u32]
        [
                1
                2
                3
                4
                5
                6
                7
        ]

        """

    def day(self) -> Series:
        """
        Extract the day from the underlying date representation.

        Applies to Date and Datetime columns.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 9)
        >>> date = pl.date_range(start, stop, interval="2d", eager=True)
        >>> date
        shape: (5,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-03 00:00:00
                2001-01-05 00:00:00
                2001-01-07 00:00:00
                2001-01-09 00:00:00
        ]
        >>> date.dt.day()
        shape: (5,)
        Series: 'date' [u32]
        [
                1
                3
                5
                7
                9
        ]

        """

    def ordinal_day(self) -> Series:
        """
        Extract ordinal day from underlying date representation.

        Applies to Date and Datetime columns.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Ordinal day as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 3, 1)
        >>> date = pl.date_range(start, stop, interval="1mo", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
        ]
        >>> date.dt.ordinal_day()
        shape: (3,)
        Series: 'date' [u32]
        [
                1
                32
                60
        ]

        """

    def time(self) -> Series:
        """
        Extract (local) time.

        Applies to Date/Datetime/Time columns.

        Returns
        -------
        Time Series

        Examples
        --------
        >>> from datetime import datetime
        >>> ser = pl.Series([datetime(2021, 1, 2, 5)]).dt.replace_time_zone(
        ...     "Asia/Kathmandu"
        ... )
        >>> ser
        shape: (1,)
        Series: '' [datetime[μs, Asia/Kathmandu]]
        [
                2021-01-02 05:00:00 +0545
        ]
        >>> ser.dt.time()
        shape: (1,)
        Series: '' [time]
        [
                05:00:00
        ]
        """

    def date(self) -> Series:
        """
        Extract (local) date.

        Applies to Date/Datetime columns.

        Returns
        -------
        Date Series

        Examples
        --------
        >>> from datetime import datetime
        >>> ser = pl.Series([datetime(2021, 1, 2, 5)]).dt.replace_time_zone(
        ...     "Asia/Kathmandu"
        ... )
        >>> ser
        shape: (1,)
        Series: '' [datetime[μs, Asia/Kathmandu]]
        [
                2021-01-02 05:00:00 +0545
        ]
        >>> ser.dt.date()
        shape: (1,)
        Series: '' [date]
        [
                2021-01-02
        ]
        """

    def datetime(self) -> Series:
        """
        Extract (local) datetime.

        Applies to Datetime columns.

        Returns
        -------
        Datetime Series

        Examples
        --------
        >>> from datetime import datetime
        >>> ser = pl.Series([datetime(2021, 1, 2, 5)]).dt.replace_time_zone(
        ...     "Asia/Kathmandu"
        ... )
        >>> ser
        shape: (1,)
        Series: '' [datetime[μs, Asia/Kathmandu]]
        [
                2021-01-02 05:00:00 +0545
        ]
        >>> ser.dt.datetime()
        shape: (1,)
        Series: '' [datetime[μs]]
        [
                2021-01-02 05:00:00
        ]
        """

    def hour(self) -> Series:
        """
        Extract the hour from the underlying DateTime representation.

        Applies to Datetime columns.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 3)
        >>> date = pl.date_range(start, stop, interval="1h", eager=True)
        >>> date
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 01:00:00
                2001-01-01 02:00:00
                2001-01-01 03:00:00
        ]
        >>> date.dt.hour()
        shape: (4,)
        Series: 'date' [u32]
        [
                0
                1
                2
                3
        ]

        """

    def minute(self) -> Series:
        """
        Extract the minutes from the underlying DateTime representation.

        Applies to Datetime columns.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 4, 0)
        >>> date = pl.date_range(start, stop, interval="2m", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:02:00
                2001-01-01 00:04:00
        ]
        >>> date.dt.minute()
        shape: (3,)
        Series: 'date' [u32]
        [
                0
                2
                4
        ]

        """

    def second(self, *, fractional: bool = False) -> Series:
        """
        Extract seconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the integer second number from 0 to 59, or a floating
        point number from 0 < 60 if ``fractional=True`` that includes
        any milli/micro/nanosecond component.

        Parameters
        ----------
        fractional
            Whether to include the fractional component of the second.

        Returns
        -------
        Second part as UInt32 (or Float64)

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 0, 4)
        >>> date = pl.date_range(start, stop, interval="500ms", eager=True)
        >>> date
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:00:00.500
                2001-01-01 00:00:01
                2001-01-01 00:00:01.500
                2001-01-01 00:00:02
                2001-01-01 00:00:02.500
                2001-01-01 00:00:03
                2001-01-01 00:00:03.500
                2001-01-01 00:00:04
        ]
        >>> date.dt.second()
        shape: (9,)
        Series: 'date' [u32]
        [
                0
                0
                1
                1
                2
                2
                3
                3
                4
        ]
        >>> date.dt.second(fractional=True)
        shape: (9,)
        Series: 'date' [f64]
        [
                0.0
                0.5
                1.0
                1.5
                2.0
                2.5
                3.0
                3.5
                4.0
        ]

        """

    def millisecond(self) -> Series:
        """
        Extract the milliseconds from the underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Millisecond part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 0, 4)
        >>> date = pl.date_range(start, stop, interval="500ms", eager=True)
        >>> date
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:00:00.500
                2001-01-01 00:00:01
                2001-01-01 00:00:01.500
                2001-01-01 00:00:02
                2001-01-01 00:00:02.500
                2001-01-01 00:00:03
                2001-01-01 00:00:03.500
                2001-01-01 00:00:04
        ]
        >>> date.dt.millisecond()
        shape: (9,)
        Series: 'date' [u32]
        [
                0
                500
                0
                500
                0
                500
                0
                500
                0
        ]

        """

    def microsecond(self) -> Series:
        """
        Extract the microseconds from the underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Microsecond part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 0, 4)
        >>> date = pl.date_range(start, stop, interval="500ms", eager=True)
        >>> date
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:00:00.500
                2001-01-01 00:00:01
                2001-01-01 00:00:01.500
                2001-01-01 00:00:02
                2001-01-01 00:00:02.500
                2001-01-01 00:00:03
                2001-01-01 00:00:03.500
                2001-01-01 00:00:04
        ]
        >>> date.dt.microsecond()
        shape: (9,)
        Series: 'date' [u32]
        [
                0
                500000
                0
                500000
                0
                500000
                0
                500000
                0
        ]

        """

    def nanosecond(self) -> Series:
        """
        Extract the nanoseconds from the underlying DateTime representation.

        Applies to Datetime columns.

        Returns
        -------
        Nanosecond part as UInt32

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 0, 4)
        >>> date = pl.date_range(start, stop, interval="500ms", eager=True)
        >>> date
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:00:00.500
                2001-01-01 00:00:01
                2001-01-01 00:00:01.500
                2001-01-01 00:00:02
                2001-01-01 00:00:02.500
                2001-01-01 00:00:03
                2001-01-01 00:00:03.500
                2001-01-01 00:00:04
        ]
        >>> date.dt.nanosecond()
        shape: (9,)
        Series: 'date' [u32]
        [
                0
                500000000
                0
                500000000
                0
                500000000
                0
                500000000
                0
        ]

        """

    def timestamp(self, time_unit: TimeUnit = "us") -> Series:
        """
        Return a timestamp in the given time unit.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms'}
            Time unit.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, interval="1d", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.timestamp().alias("timestamp_us")
        shape: (3,)
        Series: 'timestamp_us' [i64]
        [
                978307200000000
                978393600000000
                978480000000000
        ]
        >>> date.dt.timestamp("ns").alias("timestamp_ns")
        shape: (3,)
        Series: 'timestamp_ns' [i64]
        [
                978307200000000000
                978393600000000000
                978480000000000000
        ]

        """

    def epoch(self, time_unit: EpochTimeUnit = "us") -> Series:
        """
        Get the time passed since the Unix EPOCH in the give time unit.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms', 's', 'd'}
            Unit of time.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, interval="1d", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.epoch().alias("epoch_ns")
        shape: (3,)
        Series: 'epoch_ns' [i64]
        [
                978307200000000
                978393600000000
                978480000000000
        ]
        >>> date.dt.epoch(time_unit="s").alias("epoch_s")
        shape: (3,)
        Series: 'epoch_s' [i64]
        [
                978307200
                978393600
                978480000
        ]

        """

    def with_time_unit(self, time_unit: TimeUnit) -> Series:
        """
        Set time unit a Series of dtype Datetime or Duration.

        This does not modify underlying data, and should be used to fix an incorrect
        time unit.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            Unit of time for the ``Datetime`` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, "1d", time_unit="ns", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[ns]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.with_time_unit("us").alias("time_unit_us")
        shape: (3,)
        Series: 'time_unit_us' [datetime[μs]]
        [
                +32971-04-28 00:00:00
                +32974-01-22 00:00:00
                +32976-10-18 00:00:00
        ]

        """

    def cast_time_unit(self, time_unit: TimeUnit) -> Series:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            Unit of time for the ``Datetime`` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, "1d", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.cast_time_unit("ms").alias("time_unit_ms")
        shape: (3,)
        Series: 'time_unit_ms' [datetime[ms]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.cast_time_unit("ns").alias("time_unit_ns")
        shape: (3,)
        Series: 'time_unit_ns' [datetime[ns]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]

        """

    def convert_time_zone(self, time_zone: str) -> Series:
        """
        Convert to given time zone for a Series of type Datetime.

        Parameters
        ----------
        time_zone
            Time zone for the `Datetime` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2020, 3, 1)
        >>> stop = datetime(2020, 5, 1)
        >>> date = pl.date_range(start, stop, "1mo", time_zone="UTC", eager=True)
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs, UTC]]
        [
                2020-03-01 00:00:00 UTC
                2020-04-01 00:00:00 UTC
                2020-05-01 00:00:00 UTC
        ]
        >>> date = date.dt.convert_time_zone("Europe/London").alias("London")
        >>> date
        shape: (3,)
        Series: 'London' [datetime[μs, Europe/London]]
        [
            2020-03-01 00:00:00 GMT
            2020-04-01 01:00:00 BST
            2020-05-01 01:00:00 BST
        ]
        """

    def replace_time_zone(
        self, time_zone: str | None, *, use_earliest: bool | None = None
    ) -> Series:
        """
        Replace time zone for a Series of type Datetime.

        Different from ``convert_time_zone``, this will also modify
        the underlying timestamp and will ignore the original time zone.

        Parameters
        ----------
        time_zone
            Time zone for the `Datetime` Series. Pass `None` to unset time zone.
        use_earliest
            If localizing an ambiguous datetime (say, due to daylight saving time),
            determine whether to localize to the earliest datetime or not.
            If None (the default), then ambiguous datetimes will raise.

        Examples
        --------
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "london_timezone": pl.date_range(
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
        ...     "2018-10-28 02:30",
        ... ]
        >>> df = pl.DataFrame(
        ...     {
        ...         "ts": pl.Series(dates).str.strptime(pl.Datetime),
        ...         "DST": [True, True, True, False, False],
        ...     }
        ... )
        >>> df.with_columns(
        ...     ts_localized=pl.when(pl.col("DST"))
        ...     .then(
        ...         pl.col("ts").dt.replace_time_zone(
        ...             "Europe/Brussels", use_earliest=True
        ...         )
        ...     )
        ...     .otherwise(
        ...         pl.col("ts").dt.replace_time_zone(
        ...             "Europe/Brussels", use_earliest=False
        ...         )
        ...     )
        ... )
        shape: (5, 3)
        ┌─────────────────────┬───────┬───────────────────────────────┐
        │ ts                  ┆ DST   ┆ ts_localized                  │
        │ ---                 ┆ ---   ┆ ---                           │
        │ datetime[μs]        ┆ bool  ┆ datetime[μs, Europe/Brussels] │
        ╞═════════════════════╪═══════╪═══════════════════════════════╡
        │ 2018-10-28 01:30:00 ┆ true  ┆ 2018-10-28 01:30:00 CEST      │
        │ 2018-10-28 02:00:00 ┆ true  ┆ 2018-10-28 02:00:00 CEST      │
        │ 2018-10-28 02:30:00 ┆ true  ┆ 2018-10-28 02:30:00 CEST      │
        │ 2018-10-28 02:00:00 ┆ false ┆ 2018-10-28 02:00:00 CET       │
        │ 2018-10-28 02:30:00 ┆ false ┆ 2018-10-28 02:30:00 CET       │
        └─────────────────────┴───────┴───────────────────────────────┘

        """

    def days(self) -> Series:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo", eager=True
        ... )
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2020-03-01 00:00:00
                2020-04-01 00:00:00
                2020-05-01 00:00:00
        ]
        >>> date.diff().dt.days()
        shape: (3,)
        Series: 'date' [i64]
        [
                null
                31
                30
        ]

        """

    def hours(self) -> Series:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 4), "1d", eager=True
        ... )
        >>> date
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-02 00:00:00
                2020-01-03 00:00:00
                2020-01-04 00:00:00
        ]
        >>> date.diff().dt.hours()
        shape: (4,)
        Series: 'date' [i64]
        [
                null
                24
                24
                24
        ]

        """

    def minutes(self) -> Series:
        """
        Extract the minutes from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 4), "1d", eager=True
        ... )
        >>> date
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-02 00:00:00
                2020-01-03 00:00:00
                2020-01-04 00:00:00
        ]
        >>> date.diff().dt.minutes()
        shape: (4,)
        Series: 'date' [i64]
        [
                null
                1440
                1440
                1440
        ]

        """

    def seconds(self) -> Series:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 4, 0), "1m", eager=True
        ... )
        >>> date
        shape: (5,)
        Series: 'date' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:01:00
                2020-01-01 00:02:00
                2020-01-01 00:03:00
                2020-01-01 00:04:00
        ]
        >>> date.diff().dt.seconds()
        shape: (5,)
        Series: 'date' [i64]
        [
                null
                60
                60
                60
                60
        ]

        """

    def milliseconds(self) -> Series:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1),
        ...     datetime(2020, 1, 1, 0, 0, 1, 0),
        ...     "1ms",
        ...     eager=True,
        ... )[:3]
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:00:00.001
                2020-01-01 00:00:00.002
        ]
        >>> date.diff().dt.milliseconds()
        shape: (3,)
        Series: 'date' [i64]
        [
                null
                1
                1
        ]

        """

    def microseconds(self) -> Series:
        """
        Extract the microseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1),
        ...     datetime(2020, 1, 1, 0, 0, 1, 0),
        ...     "1ms",
        ...     eager=True,
        ... )[:3]
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:00:00.001
                2020-01-01 00:00:00.002
        ]
        >>> date.diff().dt.microseconds()
        shape: (3,)
        Series: 'date' [i64]
        [
                null
                1000
                1000
        ]

        """

    def nanoseconds(self) -> Series:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1),
        ...     datetime(2020, 1, 1, 0, 0, 1, 0),
        ...     "1ms",
        ...     eager=True,
        ... )[:3]
        >>> date
        shape: (3,)
        Series: 'date' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:00:00.001
                2020-01-01 00:00:00.002
        ]
        >>> date.diff().dt.nanoseconds()
        shape: (3,)
        Series: 'date' [i64]
        [
                null
                1000000
                1000000
        ]

        """

    def offset_by(self, by: str) -> Series:
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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

        Returns
        -------
        Date/Datetime expression

        Examples
        --------
        >>> from datetime import datetime
        >>> dates = pl.date_range(
        ...     datetime(2000, 1, 1), datetime(2005, 1, 1), "1y", eager=True
        ... )
        >>> dates
        shape: (6,)
        Series: 'date' [datetime[μs]]
        [
                2000-01-01 00:00:00
                2001-01-01 00:00:00
                2002-01-01 00:00:00
                2003-01-01 00:00:00
                2004-01-01 00:00:00
                2005-01-01 00:00:00
        ]
        >>> dates.dt.offset_by("1y").alias("date_plus_1y")
        shape: (6,)
        Series: 'date_plus_1y' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2002-01-01 00:00:00
                2003-01-01 00:00:00
                2004-01-01 00:00:00
                2005-01-01 00:00:00
                2006-01-01 00:00:00
        ]
        >>> dates.dt.offset_by("-1y2mo").alias("date_minus_1y_2mon")
        shape: (6,)
        Series: 'date_minus_1y_2mon' [datetime[μs]]
        [
                1998-11-01 00:00:00
                1999-11-01 00:00:00
                2000-11-01 00:00:00
                2001-11-01 00:00:00
                2002-11-01 00:00:00
                2003-11-01 00:00:00
        ]

        To get to the end of each month, combine with `truncate`:

        >>> dates.dt.truncate("1mo").dt.offset_by("1mo").dt.offset_by("-1d")
        shape: (6,)
        Series: 'date' [datetime[μs]]
        [
                2000-01-31 00:00:00
                2001-01-31 00:00:00
                2002-01-31 00:00:00
                2003-01-31 00:00:00
                2004-01-31 00:00:00
                2005-01-31 00:00:00
        ]
        """

    def truncate(
        self,
        every: str | dt.timedelta,
        offset: str | dt.timedelta | None = None,
    ) -> Series:
        """
        Divide the date/ datetime range into buckets.

        Each date/datetime is mapped to the start of its bucket.

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

        Notes
        -----
        The ``every`` and ``offset`` argument are created with the
        the following string language:

        - 1ns # 1 nanosecond
        - 1us # 1 microsecond
        - 1ms # 1 millisecond
        - 1s  # 1 second
        - 1m  # 1 minute
        - 1h  # 1 hour
        - 1d  # 1 day
        - 1w  # 1 calendar week
        - 1mo # 1 calendar month
        - 1y  # 1 calendar year

        These strings can be combined:

        - 3d12h4m25s # 3 days, 12 hours, 4 minutes, and 25 seconds

        Suffix with `"_saturating"` to indicate that dates too large for
        their month should saturate at the largest date (e.g. 2022-02-29 -> 2022-02-28)
        instead of erroring.

        Returns
        -------
        Date/Datetime series

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> s = pl.date_range(start, stop, timedelta(minutes=165), eager=True)
        >>> s
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
            2001-01-01 00:00:00
            2001-01-01 02:45:00
            2001-01-01 05:30:00
            2001-01-01 08:15:00
            2001-01-01 11:00:00
            2001-01-01 13:45:00
            2001-01-01 16:30:00
            2001-01-01 19:15:00
            2001-01-01 22:00:00
        ]
        >>> s.dt.truncate("1h")
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
            2001-01-01 00:00:00
            2001-01-01 02:00:00
            2001-01-01 05:00:00
            2001-01-01 08:00:00
            2001-01-01 11:00:00
            2001-01-01 13:00:00
            2001-01-01 16:00:00
            2001-01-01 19:00:00
            2001-01-01 22:00:00
        ]
        >>> s.dt.truncate("1h").series_equal(s.dt.truncate(timedelta(hours=1)))
        True

        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 1)
        >>> s = pl.date_range(start, stop, "10m", eager=True)
        >>> s
        shape: (7,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:10:00
                2001-01-01 00:20:00
                2001-01-01 00:30:00
                2001-01-01 00:40:00
                2001-01-01 00:50:00
                2001-01-01 01:00:00
        ]
        >>> s.dt.truncate("30m")
        shape: (7,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:00:00
                2001-01-01 00:00:00
                2001-01-01 00:30:00
                2001-01-01 00:30:00
                2001-01-01 00:30:00
                2001-01-01 01:00:00
        ]

        """

    def round(
        self,
        every: str | dt.timedelta,
        offset: str | dt.timedelta | None = None,
    ) -> Series:
        """
        Divide the date/ datetime range into buckets.

        Each date/datetime in the first half of the interval
        is mapped to the start of its bucket.
        Each date/datetime in the second half of the interval
        is mapped to the end of its bucket.

        The `every` and `offset` argument are created with the
        the following string language:

        1ns # 1 nanosecond
        1us # 1 microsecond
        1ms # 1 millisecond
        1s  # 1 second
        1m  # 1 minute
        1h  # 1 hour
        1d  # 1 day
        1w  # 1 calendar week
        1mo # 1 calendar month
        1y  # 1 calendar year

        3d12h4m25s # 3 days, 12 hours, 4 minutes, and 25 seconds

        Suffix with `"_saturating"` to indicate that dates too large for
        their month should saturate at the largest date (e.g. 2022-02-29 -> 2022-02-28)
        instead of erroring.

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

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
        >>> s = pl.date_range(start, stop, timedelta(minutes=165), eager=True)
        >>> s
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
            2001-01-01 00:00:00
            2001-01-01 02:45:00
            2001-01-01 05:30:00
            2001-01-01 08:15:00
            2001-01-01 11:00:00
            2001-01-01 13:45:00
            2001-01-01 16:30:00
            2001-01-01 19:15:00
            2001-01-01 22:00:00
        ]
        >>> s.dt.round("1h")
        shape: (9,)
        Series: 'date' [datetime[μs]]
        [
            2001-01-01 00:00:00
            2001-01-01 03:00:00
            2001-01-01 06:00:00
            2001-01-01 08:00:00
            2001-01-01 11:00:00
            2001-01-01 14:00:00
            2001-01-01 17:00:00
            2001-01-01 19:00:00
            2001-01-01 22:00:00
        ]
        >>> s.dt.round("1h").series_equal(s.dt.round(timedelta(hours=1)))
        True

        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 1)
        >>> s = pl.date_range(start, stop, "10m", eager=True)
        >>> s.dt.round("30m")
        shape: (7,)
        Series: 'date' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:00:00
                2001-01-01 00:30:00
                2001-01-01 00:30:00
                2001-01-01 00:30:00
                2001-01-01 01:00:00
                2001-01-01 01:00:00
        ]

        """

    def combine(self, time: dt.time | Series, time_unit: TimeUnit = "us") -> Expr:
        """
        Create a naive Datetime from an existing Date/Datetime expression and a Time.

        If the underlying expression is a Datetime then its time component is replaced,
        and if it is a Date then a new Datetime is created by combining the two values.

        Parameters
        ----------
        time
            A python time literal or Series of the same length as this Series.
        time_unit : {'ns', 'us', 'ms'}
            Unit of time.

        Examples
        --------
        >>> from datetime import datetime, time
        >>> s = pl.Series(
        ...     "dtm",
        ...     [datetime(2022, 12, 31, 10, 30, 45), datetime(2023, 7, 5, 23, 59, 59)],
        ... )
        >>> s.dt.combine(time(1, 2, 3, 456000))
        shape: (2,)
        Series: 'dtm' [datetime[μs]]
        [
            2022-12-31 01:02:03.456
            2023-07-05 01:02:03.456
        ]

        """

    def month_start(self) -> Series:
        """
        Roll backward to the first day of the month.

        Returns
        -------
        Date/Datetime expression

        Notes
        -----
        If you're coming from pandas, you can think of this as a vectorised version
        of ``pandas.tseries.offsets.MonthBegin().rollback(datetime)``.

        Examples
        --------
        >>> from datetime import datetime
        >>> s = pl.date_range(
        ...     datetime(2000, 1, 2, 2), datetime(2000, 4, 2, 2), "1mo", eager=True
        ... )
        >>> s.dt.month_start()
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2000-01-01 02:00:00
                2000-02-01 02:00:00
                2000-03-01 02:00:00
                2000-04-01 02:00:00
        ]
        """

    def month_end(self) -> Series:
        """
        Roll forward to the last day of the month.

        Returns
        -------
        Date/Datetime expression

        Notes
        -----
        If you're coming from pandas, you can think of this as a vectorised version
        of ``pandas.tseries.offsets.MonthEnd().rollforward(datetime)``.

        Examples
        --------
        >>> from datetime import datetime
        >>> s = pl.date_range(
        ...     datetime(2000, 1, 2, 2), datetime(2000, 4, 2, 2), "1mo", eager=True
        ... )
        >>> s.dt.month_end()
        shape: (4,)
        Series: 'date' [datetime[μs]]
        [
                2000-01-31 02:00:00
                2000-02-29 02:00:00
                2000-03-31 02:00:00
                2000-04-30 02:00:00
        ]
        """
