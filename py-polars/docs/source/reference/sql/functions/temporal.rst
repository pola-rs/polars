Temporal
========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`DATE <date>`
     - Converts a formatted string date to an actual Date type.
   * - :ref:`DATE_PART <date_part>`
     - Extracts a part of a date (or datetime) such as 'year', 'month', etc.
   * - :ref:`EXTRACT <extract>`
     - Offers the same functionality as `DATE_PART` with slightly different syntax.

.. _date:

DATE
----
Converts a formatted string date to an actual Date type; ISO-8601 format is assumed
unless a strftime-compatible formatting string is provided as the second parameter.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"str_date": ["1969.10.30", "2024.07.05", "2077.02.28"]})
    df.sql("""
      SELECT str_date, DATE(str_date, '%Y.%m.%d') AS date FROM self
    """)
    # shape: (3, 2)
    # ┌────────────┬────────────┐
    # │ str_date   ┆ date       │
    # │ ---        ┆ ---        │
    # │ str        ┆ date       │
    # ╞════════════╪════════════╡
    # │ 1969.10.30 ┆ 1969-10-30 │
    # │ 2024.07.05 ┆ 2024-07-05 │
    # │ 2077.02.28 ┆ 2077-02-28 │
    # └────────────┴────────────┘

.. _date_part:

DATE_PART
---------
Extracts a part of a date (or datetime) such as 'year', 'month', etc.

**Supported parts/fields:**
    - "day"
    - "dayofweek" | "dow"
    - "dayofyear" | "doy"
    - "decade"
    - "epoch"
    - "hour"
    - "isodow"
    - "isoweek" | "week"
    - "isoyear"
    - "microsecond(s)"
    - "millisecond(s)"
    - "nanosecond(s)"
    - "minute"
    - "month"
    - "quarter"
    - "second"
    - "time"
    - "year"

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
    - "day"
    - "dayofweek" | "dow"
    - "dayofyear" | "doy"
    - "decade"
    - "epoch"
    - "hour"
    - "isodow"
    - "isoweek" | "week"
    - "isoyear"
    - "microsecond(s)"
    - "millisecond(s)"
    - "nanosecond(s)"
    - "minute"
    - "month"
    - "quarter"
    - "second"
    - "time"
    - "year"


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
