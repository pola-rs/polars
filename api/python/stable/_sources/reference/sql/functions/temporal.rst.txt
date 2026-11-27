Temporal
========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description

   * - :ref:`CURRENT_DATE <current_date>`
     - Returns the local date at the start of the statement.
   * - :ref:`CURRENT_TIME <current_time>`
     - Returns the local time at the start of the statement.
   * - :ref:`CURRENT_TIMESTAMP <current_timestamp>`
     - Returns the local date and time at the start of the statement.
   * - :ref:`DATE_DIFF <date_diff>`
     - Counts the boundaries of a part, such as 'day' or 'month', crossed from a start to an end value.
   * - :ref:`DATE_PART <date_part>`
     - Extracts a part of a date (or datetime) such as 'year', 'month', etc.
   * - :ref:`DATE_TRUNC <date_trunc>`
     - Truncates a date (or datetime) to the given part, such as 'hour' or 'month'.
   * - :ref:`EXTRACT <extract>`
     - Offers the same functionality as `DATE_PART` with slightly different syntax.
   * - :ref:`MAKE_DATE <make_date>`
     - Builds a date from integer year, month and day parts.
   * - :ref:`STRFTIME <strftime>`
     - Formats a temporal value (Datetime, Date, or Time) as a string.
   * - :ref:`TO_CHAR <to_char>`
     - Formats a temporal value (Datetime, Date, or Time) as a string using a PostgreSQL template.


.. _current_date:

CURRENT_DATE
------------
Returns the local date at the start of the statement, as a Date. All calls in a statement,
including its subqueries, return the same value. A lazy query reads the clock when its SQL is
translated, which can be before it is collected (for example, when its schema is read). A
table that is itself a lazy SQL query reads the clock separately.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"dt": [date(2020, 1, 1), date(2999, 1, 1)]})
    df.sql("SELECT dt FROM self WHERE dt <= CURRENT_DATE - INTERVAL '1 year'")
    # shape: (1, 1)
    # ┌────────────┐
    # │ dt         │
    # │ ---        │
    # │ date       │
    # ╞════════════╡
    # │ 2020-01-01 │
    # └────────────┘

.. _current_time:

CURRENT_TIME
------------
Returns the local time at the start of the statement, as a Time. `LOCALTIME` is an alias.

**Example:**

.. code-block:: python

    df.sql("SELECT CURRENT_TIME AS now FROM self")

.. _current_timestamp:

CURRENT_TIMESTAMP
-----------------
Returns the local date and time at the start of the statement, as a Datetime with
microsecond precision and no time zone. `LOCALTIMESTAMP` and `NOW()` are aliases.

**Example:**

.. code-block:: python

    df.sql("SELECT CURRENT_TIMESTAMP AS ts, NOW() AS ts_now FROM self")

.. _date_diff:

DATE_DIFF
---------
Counts the boundaries of a part crossed from a start to an end value, as an Int64 that is
negative if the end is before the start. From 2024-01-31 to 2024-02-01 is 1 month, while from
2024-01-01 to 2024-01-31 is 0 months. `DATEDIFF` is an alias.

Supported parts are 'year', 'quarter', 'month', 'week' (starts on Monday), 'day', 'hour',
'minute', 'second', 'millisecond', 'microsecond' and 'nanosecond' (plural forms are also
accepted). The values can be dates, timestamps or times; times only support parts shorter
than a day. A string literal is read as a value of the type of the other argument.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "start_date": [date(2024, 1, 31), date(2024, 12, 31)],
            "end_ts": [datetime(2024, 2, 1, 10, 30), datetime(2025, 1, 1)],
        }
    )
    df.sql("""
      SELECT
        DATE_DIFF('day', start_date, end_ts) AS days,
        DATE_DIFF('month', start_date, end_ts) AS months,
        DATE_DIFF('year', start_date, end_ts) AS years,
        DATE_DIFF('hour', start_date, end_ts) AS hours
      FROM self
    """)
    # shape: (2, 4)
    # ┌──────┬────────┬───────┬───────┐
    # │ days ┆ months ┆ years ┆ hours │
    # │ ---  ┆ ---    ┆ ---   ┆ ---   │
    # │ i64  ┆ i64    ┆ i64   ┆ i64   │
    # ╞══════╪════════╪═══════╪═══════╡
    # │ 1    ┆ 1      ┆ 0     ┆ 34    │
    # │ 1    ┆ 1      ┆ 1     ┆ 24    │
    # └──────┴────────┴───────┴───────┘

.. _date_part:

DATE_PART
---------
Extracts a part of a date (or datetime) such as 'year', 'month', etc.

**Supported parts/fields:**

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Part
     - Description
     - DataType
   * - 'millennium', 'millennia'
     - Millennium number
     - Int32
   * - 'century', 'centuries', 'c'
     - Century number
     - Int32
   * - 'decade', 'decades'
     - Decade number (year / 10)
     - Int32
   * - 'isoyear'
     - ISO year number
     - Int32
   * - 'year', 'years', 'y'
     - Calendar year
     - Int32
   * - 'quarter', 'quarters'
     - Quarter of the year (1–4)
     - Int8
   * - 'month', 'months', 'mon', 'mons'
     - Month of the year (1–12)
     - Int8
   * - 'dayofyear', 'doy'
     - Day of the year (1–366)
     - Int16
   * - 'dayofweek', 'dow', 'weekday'
     - Day of the week (0:Sunday – 6:Saturday)
     - Int8
   * - 'isoweek', 'week'
     - ISO week number (1–53)
     - Int8
   * - 'isodow'
     - ISO day of the week (1:Monday – 7:Sunday)
     - Int8
   * - 'day', 'days', 'dayofmonth', 'd'
     - Day of the month (1–31)
     - Int8
   * - 'hour', 'hours', 'h'
     - Hour of the day (0–23)
     - Int8
   * - 'minute', 'minutes', 'mins', 'min', 'm'
     - Minute of the hour (0–59)
     - Int8
   * - 'second', 'seconds', 'sec', 'secs', 's'
     - Second of the minute (0–59)
     - Int8
   * - 'millisecond', 'milliseconds', 'ms'
     - Sub-minute seconds and milliseconds (0–59999.)
     - Float64
   * - 'microsecond', 'microseconds', 'us'
     - Sub-minute seconds and microseconds (0–59999999.)
     - Float64
   * - 'nanosecond', 'nanoseconds', 'ns'
     - Sub-minute seconds and nanoseconds (0–59999999999.)
     - Float64
   * - 'timezone'
     - UTC offset of the timezone, in seconds ()
     - Int64
   * - 'time'
     - Time component
     - Time
   * - 'epoch'
     - Seconds since Unix epoch (1970-01-01)
     - Float64

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
          "dt": [
              datetime(1969, 12, 31, 4, 30, 45, 123456),
              datetime(2026, 7, 12, 10, 23, 59, 999999),
              datetime(2077, 2, 10, 18, 10, 15, 654321),
          ]
      }
    )
    df.sql("""
      SELECT
        dt,
        DATE_PART('year', dt) AS year,
        DATE_PART('month', dt) AS month,
        DATE_PART('day', dt) AS day,
        DATE_PART('ms', dt) AS secs_ms,
      FROM self
    """)
    # shape: (3, 5)
    # ┌────────────────────────────┬──────┬───────┬─────┬───────────┐
    # │ dt                         ┆ year ┆ month ┆ day ┆ secs_ms   │
    # │ ---                        ┆ ---  ┆ ---   ┆ --- ┆ ---       │
    # │ datetime[μs]               ┆ i32  ┆ i8    ┆ i8  ┆ f64       │
    # ╞════════════════════════════╪══════╪═══════╪═════╪═══════════╡
    # │ 1969-12-31 04:30:45.123456 ┆ 1969 ┆ 12    ┆ 31  ┆ 45123.456 │
    # │ 2026-07-12 10:23:59.999999 ┆ 2026 ┆ 7     ┆ 12  ┆ 59999.999 │
    # │ 2077-02-10 18:10:15.654321 ┆ 2077 ┆ 2     ┆ 10  ┆ 15654.321 │
    # └────────────────────────────┴──────┴───────┴─────┴───────────┘

.. _date_trunc:

DATE_TRUNC
----------
Truncates a date (or datetime) to the given part. Supported parts are 'microsecond',
'millisecond', 'second', 'minute', 'hour', 'day', 'week' (starts on Monday), 'month',
'quarter' and 'year' (plural forms are also accepted). A date is returned as a timestamp.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {"dt": [datetime(2024, 5, 17, 13, 45, 30), datetime(2024, 11, 3, 8, 5, 0)]}
    )
    df.sql("""
      SELECT dt, DATE_TRUNC('hour', dt) AS hour, DATE_TRUNC('month', dt) AS month FROM self
    """)
    # shape: (2, 3)
    # ┌─────────────────────┬─────────────────────┬─────────────────────┐
    # │ dt                  ┆ hour                ┆ month               │
    # │ ---                 ┆ ---                 ┆ ---                 │
    # │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        │
    # ╞═════════════════════╪═════════════════════╪═════════════════════╡
    # │ 2024-05-17 13:45:30 ┆ 2024-05-17 13:00:00 ┆ 2024-05-01 00:00:00 │
    # │ 2024-11-03 08:05:00 ┆ 2024-11-03 08:00:00 ┆ 2024-11-01 00:00:00 │
    # └─────────────────────┴─────────────────────┴─────────────────────┘

.. _extract:

EXTRACT
-------
Extracts a part of a date (or datetime) such as 'year', 'month', etc.


**Supported parts/fields:**

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Part
     - Description
     - DataType
   * - 'millennium', 'millennia'
     - Millennium number
     - Int32
   * - 'century', 'centuries', 'c'
     - Century number
     - Int32
   * - 'decade', 'decades'
     - Decade number (year / 10)
     - Int32
   * - 'isoyear'
     - ISO year number
     - Int32
   * - 'year', 'years', 'y'
     - Calendar year
     - Int32
   * - 'quarter', 'quarters'
     - Quarter of the year (1–4)
     - Int8
   * - 'month', 'months', 'mon', 'mons'
     - Month of the year (1–12)
     - Int8
   * - 'dayofyear', 'doy'
     - Day of the year (1–366)
     - Int16
   * - 'dayofweek', 'dow', 'weekday'
     - Day of the week (0:Sunday – 6:Saturday)
     - Int8
   * - 'isoweek', 'week'
     - ISO week number (1–53)
     - Int8
   * - 'isodow'
     - ISO day of the week (1:Monday – 7:Sunday)
     - Int8
   * - 'day', 'days', 'dayofmonth', 'd'
     - Day of the month (1–31)
     - Int8
   * - 'hour', 'hours', 'h'
     - Hour of the day (0–23)
     - Int8
   * - 'minute', 'minutes', 'mins', 'min', 'm'
     - Minute of the hour (0–59)
     - Int8
   * - 'second', 'seconds', 'sec', 'secs', 's'
     - Second of the minute (0–59)
     - Int8
   * - 'millisecond', 'milliseconds', 'ms'
     - Sub-minute seconds and milliseconds (0–59999.)
     - Float64
   * - 'microsecond', 'microseconds', 'us'
     - Sub-minute seconds and microseconds (0–59999999.)
     - Float64
   * - 'nanosecond', 'nanoseconds', 'ns'
     - Sub-minute seconds and nanoseconds (0–59999999999.)
     - Float64
   * - 'timezone'
     - UTC offset of the timezone, in seconds ()
     - Int64
   * - 'time'
     - Time component
     - Time
   * - 'epoch'
     - Seconds since Unix epoch (1970-01-01)
     - Float64

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "dt": [
          date(1969, 12, 31),
          date(2026, 8, 22),
          date(2077, 2, 10),
        ],
      }
    )
    df.sql("""
      SELECT
        dt,
        EXTRACT(decade FROM dt) AS decade,
        EXTRACT(year FROM dt) AS year,
        EXTRACT(quarter FROM dt) AS quarter,
      FROM self
    """)
    # shape: (3, 4)
    # ┌────────────┬────────┬──────┬─────────┐
    # │ dt         ┆ decade ┆ year ┆ quarter │
    # │ ---        ┆ ---    ┆ ---  ┆ ---     │
    # │ date       ┆ i32    ┆ i32  ┆ i8      │
    # ╞════════════╪════════╪══════╪═════════╡
    # │ 1969-12-31 ┆ 196    ┆ 1969 ┆ 4       │
    # │ 2026-08-22 ┆ 202    ┆ 2026 ┆ 3       │
    # │ 2077-02-10 ┆ 207    ┆ 2077 ┆ 1       │
    # └────────────┴────────┴──────┴─────────┘

.. _make_date:

MAKE_DATE
---------
Builds a date from year, month and day parts, like :func:`polars.date`. Raises an error if the
parts do not form a valid date. Years before 1 are counted as in ISO 8601: year 0 is 1 BC.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"y": [2024, 2025], "m": [2, 12], "d": [29, 31]})
    df.sql("""
      SELECT y, m, d, MAKE_DATE(y, m, d) AS dt FROM self
    """)
    # shape: (2, 4)
    # ┌──────┬─────┬─────┬────────────┐
    # │ y    ┆ m   ┆ d   ┆ dt         │
    # │ ---  ┆ --- ┆ --- ┆ ---        │
    # │ i64  ┆ i64 ┆ i64 ┆ date       │
    # ╞══════╪═════╪═════╪════════════╡
    # │ 2024 ┆ 2   ┆ 29  ┆ 2024-02-29 │
    # │ 2025 ┆ 12  ┆ 31  ┆ 2025-12-31 │
    # └──────┴─────┴─────┴────────────┘

.. _strftime:

STRFTIME
--------
Formats a temporal value (Datetime, Date, or Time) as a string using a `chrono strftime <https://docs.rs/chrono/latest/chrono/format/strftime/>`_-compatible formatting string.

.. code-block:: python

    df = pl.DataFrame(
      {
        "dt": [date(1978, 7, 5), None, date(2020, 4, 10)],
        "tm": [time(10, 10, 10), time(22, 33, 55), None],
      }
    )
    df.sql("""
      SELECT
        STRFTIME(dt, '%B %d, %Y') AS s_dt,
        STRFTIME(tm, '%H.%M.%S') AS s_tm,
      FROM self
    """)
    # shape: (3, 2)
    # ┌────────────────┬──────────┐
    # │ s_dt           ┆ s_tm     │
    # │ ---            ┆ ---      │
    # │ str            ┆ str      │
    # ╞════════════════╪══════════╡
    # │ July 05, 1978  ┆ 10.10.10 │
    # │ null           ┆ 22.33.55 │
    # │ April 10, 2020 ┆ null     │
    # └────────────────┴──────────┘

.. _to_char:

TO_CHAR
-------
Formats a temporal value (Datetime, Date, or Time) as a string using a PostgreSQL template,
such as 'YYYY-MM-DD HH24:MI:SS'. The supported patterns are `HH`, `HH12`, `HH24`, `MI`, `SS`,
`MS`, `US`, `FF3`, `FF6`, `AM`/`PM`, `YYYY`, `YY`, `IYYY`, `IY`, `MONTH`, `MON`, `MM`, `DAY`,
`DY`, `DDD`, `DD`, `ID` and `IW`, each optionally prefixed with `FM` to drop padding. Text in
double quotes is copied as is. A template that contains `%` is read as a strftime format, as in
`STRFTIME`.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {"dt": [datetime(1978, 7, 5, 10, 30), datetime(2020, 4, 10, 22, 5)]}
    )
    df.sql("""
      SELECT
        TO_CHAR(dt, 'FMDay, DD Mon YYYY') AS s_dt,
        TO_CHAR(dt, 'HH12:MI AM') AS s_tm,
      FROM self
    """)
    # shape: (2, 2)
    # ┌────────────────────────┬──────────┐
    # │ s_dt                   ┆ s_tm     │
    # │ ---                    ┆ ---      │
    # │ str                    ┆ str      │
    # ╞════════════════════════╪══════════╡
    # │ Wednesday, 05 Jul 1978 ┆ 10:30 AM │
    # │ Friday, 10 Apr 2020    ┆ 10:05 PM │
    # └────────────────────────┴──────────┘
