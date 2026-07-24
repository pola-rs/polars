Temporal
========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description

   * - :ref:`DATE_PART <date_part>`
     - Extracts a part of a date (or datetime) such as 'year', 'month', etc.
   * - :ref:`EXTRACT <extract>`
     - Offers the same functionality as `DATE_PART` with slightly different syntax.
   * - :ref:`STRFTIME <strftime>`
     - Formats a temporal value (Datetime, Date, or Time) as a string.


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
   * - 'day', 'days', 'dayofmonth', d'
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
   * - 'day', 'days', 'dayofmonth', d'
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
