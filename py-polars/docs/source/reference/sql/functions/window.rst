Window
======

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`OVER <over>`
     - Define a window (a set of rows) within which a function is applied.


.. _over:

OVER
----
Used to define a window (a set of rows) within which a function is applied.

**Notes:**
As a DataFrame engine Polars defaults to `ROWS` framing semantics for window
functions when an explicit window specification is omitted; specifically,
`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. This differs from the
default `RANGE` framing semantics typically used by database engines.

**Example:**
.. code-block:: python

    df = pl.DataFrame(
      {
        "idx": [0, 1, 2, 3, 4, 5, 6],
        "label": ["aaa", "aaa", "bbb", "bbb", "aaa", "ccc", "aaa"],
        "value": [10, 20, 30, 40, 50, -5, 0],
      }
    )
    df.sql("""
      SELECT
        *,
        FIRST_VALUE(value) OVER (PARTITION BY label ORDER BY idx) AS first_val,
        LAST_VALUE(value) OVER (PARTITION BY label ORDER BY idx) AS last_val,
        SUM(value) OVER (PARTITION BY label ORDER BY idx) AS running_total_by_label
      FROM self
      ORDER BY label, idx
    """)
    # shape: (7, 6)
    # ┌─────┬───────┬───────┬───────────┬──────────┬────────────────────────┐
    # │ idx ┆ label ┆ value ┆ first_val ┆ last_val ┆ running_total_by_label │
    # │ --- ┆ ---   ┆ ---   ┆ ---       ┆ ---      ┆ ---                    │
    # │ i64 ┆ str   ┆ i64   ┆ i64       ┆ i64      ┆ i64                    │
    # ╞═════╪═══════╪═══════╪═══════════╪══════════╪════════════════════════╡
    # │ 0   ┆ aaa   ┆ 10    ┆ 10        ┆ 10       ┆ 10                     │
    # │ 1   ┆ aaa   ┆ 20    ┆ 10        ┆ 20       ┆ 30                     │
    # │ 4   ┆ aaa   ┆ 50    ┆ 10        ┆ 50       ┆ 80                     │
    # │ 6   ┆ aaa   ┆ 0     ┆ 10        ┆ 0        ┆ 80                     │
    # │ 2   ┆ bbb   ┆ 30    ┆ 30        ┆ 30       ┆ 30                     │
    # │ 3   ┆ bbb   ┆ 40    ┆ 30        ┆ 40       ┆ 70                     │
    # │ 5   ┆ ccc   ┆ -5    ┆ -5        ┆ -5       ┆ -5                     │
    # └─────┴───────┴───────┴───────────┴──────────┴────────────────────────┘
