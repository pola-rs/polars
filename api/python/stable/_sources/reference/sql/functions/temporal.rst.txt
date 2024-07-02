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
    - "millennium" | "millennia"
    - "century" | "centuries"
    - "decade" | "decades"
    - "isoyear"
    - "year" | "years" | "y"
    - "quarter" | "quarters"
    - "month" | "months" | "mon" | "mons"
    - "dayofyear" | "doy"
    - "dayofweek" | "dow"
    - "isoweek" | "week"
    - "isodow"
    - "day" | "days" | "d"
    - "hour" | "hours" | "h"
    - "minute" | "minutes" | "mins" | "min" | "m"
    - "second" | "seconds" | "sec" | "secs" | "s"
    - "millisecond" | "milliseconds" | "ms"
    - "microsecond" | "microseconds" | "us"
    - "nanosecond" | "nanoseconds" | "ns"
    - "timezone"
    - "time"
    - "epoch"

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
          "dt": [
              date(1969, 12, 31),
              date(2026, 8, 22),
              date(2077, 2, 10),
          ]
      }
    )
    df.sql("""
      SELECT
        dt,
        DATE_PART('year', dt) AS year,
        DATE_PART('month', dt) AS month,
        DATE_PART('day', dt) AS day
      FROM self
    """)
    # shape: (3, 4)
    # ┌────────────┬──────┬───────┬─────┐
    # │ dt         ┆ year ┆ month ┆ day │
    # │ ---        ┆ ---  ┆ ---   ┆ --- │
    # │ date       ┆ i32  ┆ i8    ┆ i8  │
    # ╞════════════╪══════╪═══════╪═════╡
    # │ 1969-12-31 ┆ 1969 ┆ 12    ┆ 31  │
    # │ 2026-08-22 ┆ 2026 ┆ 8     ┆ 22  │
    # │ 2077-02-10 ┆ 2077 ┆ 2     ┆ 10  │
    # └────────────┴──────┴───────┴─────┘

.. _extract:

EXTRACT
-------
Extracts a part of a date (or datetime) such as 'year', 'month', etc.

**Supported parts/fields:**
    - "millennium" | "millennia"
    - "century" | "centuries"
    - "decade" | "decades"
    - "isoyear"
    - "year" | "years" | "y"
    - "quarter" | "quarters"
    - "month" | "months" | "mon" | "mons"
    - "dayofyear" | "doy"
    - "dayofweek" | "dow"
    - "isoweek" | "week"
    - "isodow"
    - "day" | "days" | "d"
    - "hour" | "hours" | "h"
    - "minute" | "minutes" | "mins" | "min" | "m"
    - "second" | "seconds" | "sec" | "secs" | "s"
    - "millisecond" | "milliseconds" | "ms"
    - "microsecond" | "microseconds" | "us"
    - "nanosecond" | "nanoseconds" | "ns"
    - "timezone"
    - "time"
    - "epoch"


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
