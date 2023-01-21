from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import expr_dispatch
from polars.utils import _to_python_datetime

if TYPE_CHECKING:
    from polars.internals.type_aliases import EpochTimeUnit, TimeUnit
    from polars.polars import PySeries


@expr_dispatch
class DateTimeNameSpace:
    """Series.dt namespace."""

    _accessor = "dt"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    def __getitem__(self, item: int) -> date | datetime:
        s = pli.wrap_s(self._s)
        return s[item]

    def min(self) -> date | datetime | timedelta:
        """
        Return minimum as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2001, 1, 1), datetime(2001, 1, 3), "1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.min()
        datetime.datetime(2001, 1, 1, 0, 0)

        """
        # we can ignore types because we are certain we get a logical type
        return pli.wrap_s(self._s).min()  # type: ignore[return-value]

    def max(self) -> date | datetime | timedelta:
        """
        Return maximum as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2001, 1, 1), datetime(2001, 1, 3), "1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.max()
        datetime.datetime(2001, 1, 3, 0, 0)

        """
        return pli.wrap_s(self._s).max()  # type: ignore[return-value]

    def median(self) -> date | datetime | timedelta:
        """
        Return median as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2001, 1, 1), datetime(2001, 1, 3), "1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.median()
        datetime.datetime(2001, 1, 2, 0, 0)

        """
        s = pli.wrap_s(self._s)
        out = int(s.median())
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def mean(self) -> date | datetime:
        """
        Return mean as python DateTime.

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2001, 1, 1), datetime(2001, 1, 3), "1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.mean()
        datetime.datetime(2001, 1, 2, 0, 0)

        """
        s = pli.wrap_s(self._s)
        out = int(s.mean())
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def strftime(self, fmt: str) -> pli.Series:
        """
        Format Date/datetime with a formatting rule.

        See `chrono strftime/strptime
        <https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html>`_.

        Returns
        -------
        Utf8 Series

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 4)
        >>> date = pl.date_range(start, stop, interval="1d")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
                2001-01-04 00:00:00
        ]
        >>> date.dt.strftime(fmt="%Y-%m-%d")
        shape: (4,)
        Series: '' [str]
        [
                "2001-01-01"
                "2001-01-02"
                "2001-01-03"
                "2001-01-04"
        ]

        """

    def year(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1y")
        >>> date
        shape: (2,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2002-01-01 00:00:00
        ]
        >>> date.dt.year()
        shape: (2,)
        Series: '' [i32]
        [
                2001
                2002
        ]

        """

    def iso_year(self) -> pli.Series:
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

    def quarter(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1mo")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
                2001-04-01 00:00:00
        ]
        >>> date.dt.quarter()
        shape: (4,)
        Series: '' [u32]
        [
                1
                1
                1
                2
        ]

        """

    def month(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1mo")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
                2001-04-01 00:00:00
        ]
        >>> date.dt.month()
        shape: (4,)
        Series: '' [u32]
        [
                1
                2
                3
                4
        ]

        """

    def week(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1mo")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
                2001-04-01 00:00:00
        ]
        >>> date.dt.week()
        shape: (4,)
        Series: '' [u32]
        [
                1
                5
                9
                13
        ]

        """

    def weekday(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1d")
        >>> date
        shape: (7,)
        Series: '' [datetime[μs]]
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
        Series: '' [u32]
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

    def day(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="2d")
        >>> date
        shape: (5,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-03 00:00:00
                2001-01-05 00:00:00
                2001-01-07 00:00:00
                2001-01-09 00:00:00
        ]
        >>> date.dt.day()
        shape: (5,)
        Series: '' [u32]
        [
                1
                3
                5
                7
                9
        ]

        """

    def ordinal_day(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1mo")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-02-01 00:00:00
                2001-03-01 00:00:00
        ]
        >>> date.dt.ordinal_day()
        shape: (3,)
        Series: '' [u32]
        [
                1
                32
                60
        ]

        """

    def hour(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="1h")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 01:00:00
                2001-01-01 02:00:00
                2001-01-01 03:00:00
        ]
        >>> date.dt.hour()
        shape: (4,)
        Series: '' [u32]
        [
                0
                1
                2
                3
        ]

        """

    def minute(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="2m")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-01 00:02:00
                2001-01-01 00:04:00
        ]
        >>> date.dt.minute()
        shape: (3,)
        Series: '' [u32]
        [
                0
                2
                4
        ]

        """

    def second(self, fractional: bool = False) -> pli.Series:
        """
        Extract seconds from underlying DateTime representation.

        Applies to Datetime columns.

        Returns the integer second number from 0 to 59, or a floating
        point number from 0 < 60 if ``fractional=True`` that includes
        any milli/micro/nanosecond component.

        Returns
        -------
        Second part as UInt32 (or Float64)

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 1, 0, 0, 4)
        >>> date = pl.date_range(start, stop, interval="500ms")
        >>> date
        shape: (9,)
        Series: '' [datetime[μs]]
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
        Series: '' [u32]
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
        Series: '' [f64]
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

    def millisecond(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="500ms")
        >>> date
        shape: (9,)
        Series: '' [datetime[μs]]
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
        Series: '' [u32]
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

    def microsecond(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="500ms")
        >>> date
        shape: (9,)
        Series: '' [datetime[μs]]
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
        Series: '' [u32]
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

    def nanosecond(self) -> pli.Series:
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
        >>> date = pl.date_range(start, stop, interval="500ms")
        >>> date
        shape: (9,)
        Series: '' [datetime[μs]]
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
        Series: '' [u32]
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

    def timestamp(self, tu: TimeUnit = "us") -> pli.Series:
        """
        Return a timestamp in the given time unit.

        Parameters
        ----------
        tu : {'us', 'ns', 'ms'}
            Time unit.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, interval="1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
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
        >>> date.dt.timestamp(tu="ns").alias("timestamp_ns")
        shape: (3,)
        Series: 'timestamp_ns' [i64]
        [
                978307200000000000
                978393600000000000
                978480000000000000
        ]

        """

    def epoch(self, tu: EpochTimeUnit = "us") -> pli.Series:
        """
        Get the time passed since the Unix EPOCH in the give time unit.

        Parameters
        ----------
        tu : {'us', 'ns', 'ms', 's', 'd'}
            Time unit.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, interval="1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
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
        >>> date.dt.epoch(tu="s").alias("epoch_s")
        shape: (3,)
        Series: 'epoch_s' [i64]
        [
                978307200
                978393600
                978480000
        ]

        """

    def with_time_unit(self, tu: TimeUnit) -> pli.Series:
        """
        Set time unit a Series of dtype Datetime or Duration.

        This does not modify underlying data, and should be used to fix an incorrect
        time unit.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, "1d", time_unit="ns")
        >>> date
        shape: (3,)
        Series: '' [datetime[ns]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.with_time_unit(tu="us").alias("tu_us")
        shape: (3,)
        Series: 'tu_us' [datetime[μs]]
        [
                +32971-04-28 00:00:00
                +32974-01-22 00:00:00
                +32976-10-18 00:00:00
        ]

        """

    def cast_time_unit(self, tu: TimeUnit) -> pli.Series:
        """
        Cast the underlying data to another time unit. This may lose precision.

        Parameters
        ----------
        tu : {'ns', 'us', 'ms'}
            Time unit for the ``Datetime`` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 3)
        >>> date = pl.date_range(start, stop, "1d")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.cast_time_unit(tu="ms").alias("tu_ms")
        shape: (3,)
        Series: 'tu_ms' [datetime[ms]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]
        >>> date.dt.cast_time_unit(tu="ns").alias("tu_ns")
        shape: (3,)
        Series: 'tu_ns' [datetime[ns]]
        [
                2001-01-01 00:00:00
                2001-01-02 00:00:00
                2001-01-03 00:00:00
        ]

        """

    def with_time_zone(self, tz: str | None) -> pli.Series:
        """
        Set time zone a Series of type Datetime.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2020, 3, 1)
        >>> stop = datetime(2020, 5, 1)
        >>> date = pl.date_range(start, stop, "1mo")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2020-03-01 00:00:00
                2020-04-01 00:00:00
                2020-05-01 00:00:00
        ]
        >>> date = date.dt.with_time_zone(tz="Europe/London").alias("London")
        >>> date
        shape: (3,)
        Series: 'London' [datetime[μs, Europe/London]]
        [
            2020-03-01 00:00:00 GMT
            2020-04-01 01:00:00 BST
            2020-05-01 01:00:00 BST
        ]

        """

    def cast_time_zone(self, tz: str) -> pli.Series:
        """
        Cast time zone for a Series of type Datetime.

        Different from ``with_time_zone``, this will also modify
        the underlying timestamp.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        Examples
        --------
        >>> from datetime import datetime
        >>> start = datetime(2020, 3, 1)
        >>> stop = datetime(2020, 5, 1)
        >>> date = pl.date_range(start, stop, "1mo")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2020-03-01 00:00:00
                2020-04-01 00:00:00
                2020-05-01 00:00:00
        ]
        >>> date.dt.epoch(tu="s")
        shape: (3,)
        Series: '' [i64]
        [
                1583020800
                1585699200
                1588291200
        ]
        >>> date = date.dt.with_time_zone(tz="Europe/London").alias("London")
        >>> date
        shape: (3,)
        Series: 'London' [datetime[μs, Europe/London]]
        [
            2020-03-01 00:00:00 GMT
            2020-04-01 01:00:00 BST
            2020-05-01 01:00:00 BST
        ]
        >>> # Timestamps have not changed after with_time_zone
        >>> date.dt.epoch(tu="s")
        shape: (3,)
        Series: 'London' [i64]
        [
                1583020800
                1585699200
                1588291200
        ]
        >>> date = date.dt.cast_time_zone(tz="America/New_York").alias("NYC")
        >>> date
        shape: (3,)
        Series: 'NYC' [datetime[μs, America/New_York]]
        [
            2020-03-01 00:00:00 EST
            2020-04-01 01:00:00 EDT
            2020-05-01 01:00:00 EDT
        ]
        >>> # Timestamps have changed after cast_time_zone
        >>> date.dt.epoch(tu="s")
        shape: (3,)
        Series: 'NYC' [i64]
        [
            1583038800
            1585717200
            1588309200
        ]

        """

    def tz_localize(self, tz: str) -> pli.Series:
        """
        Localize tz-naive Datetime Series to tz-aware Datetime Series.

        This method takes a naive Datetime Series and makes this time zone aware.
        It does not move the time to another time zone.

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series.

        """

    def days(self) -> pli.Series:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo")
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2020-03-01 00:00:00
                2020-04-01 00:00:00
                2020-05-01 00:00:00
        ]
        >>> date.diff().dt.days()
        shape: (3,)
        Series: '' [i64]
        [
                null
                31
                30
        ]

        """

    def hours(self) -> pli.Series:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2020, 1, 1), datetime(2020, 1, 4), "1d")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-02 00:00:00
                2020-01-03 00:00:00
                2020-01-04 00:00:00
        ]
        >>> date.diff().dt.hours()
        shape: (4,)
        Series: '' [i64]
        [
                null
                24
                24
                24
        ]

        """

    def minutes(self) -> pli.Series:
        """
        Extract the minutes from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(datetime(2020, 1, 1), datetime(2020, 1, 4), "1d")
        >>> date
        shape: (4,)
        Series: '' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-02 00:00:00
                2020-01-03 00:00:00
                2020-01-04 00:00:00
        ]
        >>> date.diff().dt.minutes()
        shape: (4,)
        Series: '' [i64]
        [
                null
                1440
                1440
                1440
        ]

        """

    def seconds(self) -> pli.Series:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 4, 0), "1m"
        ... )
        >>> date
        shape: (5,)
        Series: '' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:01:00
                2020-01-01 00:02:00
                2020-01-01 00:03:00
                2020-01-01 00:04:00
        ]
        >>> date.diff().dt.seconds()
        shape: (5,)
        Series: '' [i64]
        [
                null
                60
                60
                60
                60
        ]

        """

    def milliseconds(self) -> pli.Series:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1, 0), "1ms"
        ... )[:3]
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:00:00.001
                2020-01-01 00:00:00.002
        ]
        >>> date.diff().dt.milliseconds()
        shape: (3,)
        Series: '' [i64]
        [
                null
                1
                1
        ]

        """

    def microseconds(self) -> pli.Series:
        """
        Extract the microseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1, 0), "1ms"
        ... )[:3]
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:00:00.001
                2020-01-01 00:00:00.002
        ]
        >>> date.diff().dt.microseconds()
        shape: (3,)
        Series: '' [i64]
        [
                null
                1000
                1000
        ]

        """

    def nanoseconds(self) -> pli.Series:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64

        Examples
        --------
        >>> from datetime import datetime
        >>> date = pl.date_range(
        ...     datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1, 0), "1ms"
        ... )[:3]
        >>> date
        shape: (3,)
        Series: '' [datetime[μs]]
        [
                2020-01-01 00:00:00
                2020-01-01 00:00:00.001
                2020-01-01 00:00:00.002
        ]
        >>> date.diff().dt.nanoseconds()
        shape: (3,)
        Series: '' [i64]
        [
                null
                1000000
                1000000
        ]

        """

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

        Examples
        --------
        >>> from datetime import datetime
        >>> dates = pl.date_range(datetime(2000, 1, 1), datetime(2005, 1, 1), "1y")
        >>> dates
        shape: (6,)
        Series: '' [datetime[μs]]
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

        """

    def truncate(
        self,
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> pli.Series:
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

        Returns
        -------
        Date/Datetime series

        Examples
        --------
        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> s = pl.date_range(start, stop, timedelta(minutes=165), name="dates")
        >>> s
        shape: (9,)
        Series: 'dates' [datetime[μs]]
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
        Series: 'dates' [datetime[μs]]
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
        >>> s = pl.date_range(start, stop, "10m", name="dates")
        >>> s
        shape: (7,)
        Series: 'dates' [datetime[μs]]
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
        Series: 'dates' [datetime[μs]]
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
        every: str | timedelta,
        offset: str | timedelta | None = None,
    ) -> pli.Series:
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
        >>> s = pl.date_range(start, stop, timedelta(minutes=165), name="dates")
        >>> s
        shape: (9,)
        Series: 'dates' [datetime[μs]]
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
        Series: 'dates' [datetime[μs]]
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
        >>> s = pl.date_range(start, stop, "10m", name="dates")
        >>> s.dt.round("30m")
        shape: (7,)
        Series: 'dates' [datetime[μs]]
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

    def combine(self, tm: time | pli.Series, tu: TimeUnit = "us") -> pli.Expr:
        """
        Create a naive Datetime from an existing Date/Datetime expression and a Time.

        If the underlying expression is a Datetime then its time component is replaced,
        and if it is a Date then a new Datetime is created by combining the two values.

        Parameters
        ----------
        tm
            A python time literal or Series of the same length as this Series.
        tu : {'ns', 'us', 'ms'}
            Time unit.

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
