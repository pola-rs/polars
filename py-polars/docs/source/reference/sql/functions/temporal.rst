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
  
    >>> df = pl.DataFrame(
        {
              "dt": [
                  datetime(2025, 1, 10,),
                  datetime(2026, 12, 30),
                  datetime(2027, 1, 1),
              ]
        }
    )
    >>> df.sql("SELECT DATE('2021-03', '%Y-%m') FROM self")
    shape: (1, 1)
    ┌────────────┐
    │ literal    │
    │ ---        │
    │ date       │
    ╞════════════╡
    │ 2021-03-01 │
    └────────────┘

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
  
    >>> df = pl.DataFrame(
      {
          "dt": [
              datetime(2025, 1, 10),
              datetime(2026, 12, 30),
              datetime(2027, 1, 1),
          ]
      }
    )
    >>> df.sql("SELECT EXTRACT(year FROM dt) AS year, EXTRACT(month FROM dt) AS month FROM self")
    shape: (3, 2)
    ┌──────┬───────┐
    │ year ┆ month │
    │ ---  ┆ ---   │
    │ i32  ┆ i8    │
    ╞══════╪═══════╡
    │ 2025 ┆ 1     │
    │ 2026 ┆ 12    │
    │ 2027 ┆ 1     │
    └──────┴───────┘

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
  
  >>> df = pl.DataFrame(
      {
        "dt": [
            datetime(2025, 1, 10),
            datetime(2026, 12, 30),
            datetime(2027, 1, 1),
        ]
      }
    )
    >>> df.sql("SELECT EXTRACT(year FROM dt) AS year FROM self")
    shape: (3, 1)
    ┌──────┐
    │ year │
    │ ---  │
    │ i32  │
    ╞══════╡
    │ 2025 │
    │ 2026 │
    │ 2027 │
    └──────┘
