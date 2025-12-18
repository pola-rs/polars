Window
======

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`DENSE_RANK <dense_rank>`
     - Returns the rank of each row within a window partition, without gaps for ties.
   * - :ref:`FIRST_VALUE <first_value>`
     - Returns the first value in an ordered set of values with respect to the window declared in `OVER`.
   * - :ref:`LAST_VALUE <last_value>`
     - Returns the last value in an ordered set of values with respect to the window declared in `OVER`.
   * - :ref:`OVER <over>`
     - Define a window (a set of rows) within which a function is applied.
   * - :ref:`RANK <rank>`
     - Returns the rank of each row within a window partition, with gaps for ties.
   * - :ref:`ROW_NUMBER <row_number>`
     - Returns the sequential row number within a window partition, starting from 1.


.. note::

    As a DataFrame engine Polars defaults to `ROWS` framing semantics for window functions when an explicit
    window specification is omitted; specifically, `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. This
    differs from the default `RANGE` framing semantics typically used by database engines.


.. _dense_rank:

DENSE_RANK
----------
Returns the rank of each row within a window partition, without gaps for ties. Rows with
equal values receive the same rank, and the next rank number is consecutive (no gaps).

**Requirements:**

- Must be used with an ``OVER`` clause.
- That clause must have ``ORDER BY`` in the window specification.

**Example:**

.. code-block:: python

    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "category": ["A", "A", "A", "B", "B", "B"],
        "score": [85, 90, 90, 75, 80, 80]
    })
    df.sql("""
      SELECT
        id,
        category,
        score,
        RANK() OVER (PARTITION BY category ORDER BY score DESC) AS rank,
        DENSE_RANK() OVER (PARTITION BY category ORDER BY score DESC) AS dense_rank
      FROM self
      ORDER BY category, score DESC
    """)
    # shape: (6, 5)
    # ┌─────┬──────────┬───────┬──────┬────────────┐
    # │ id  ┆ category ┆ score ┆ rank ┆ dense_rank │
    # │ --- ┆ ---      ┆ ---   ┆ ---  ┆ ---        │
    # │ i64 ┆ str      ┆ i64   ┆ u32  ┆ u32        │
    # ╞═════╪══════════╪═══════╪══════╪════════════╡
    # │ 2   ┆ A        ┆ 90    ┆ 1    ┆ 1          │
    # │ 3   ┆ A        ┆ 90    ┆ 1    ┆ 1          │
    # │ 1   ┆ A        ┆ 85    ┆ 3    ┆ 2          │
    # │ 5   ┆ B        ┆ 80    ┆ 1    ┆ 1          │
    # │ 6   ┆ B        ┆ 80    ┆ 1    ┆ 1          │
    # │ 4   ┆ B        ┆ 75    ┆ 3    ┆ 2          │
    # └─────┴──────────┴───────┴──────┴────────────┘


.. _first_value:

FIRST_VALUE
-----------
Returns the first value in an ordered set of values with respect to the window declared in `OVER`.


.. _last_value:

LAST_VALUE
----------
Returns the last value in an ordered set of values with respect to the window declared in `OVER`.


.. _rank:

RANK
----
Returns the rank of each row within a window partition, with gaps for ties. Rows with equal values
receive the same rank, and the next rank skips numbers (creating gaps).

**Requirements:**

- Must be used with an ``OVER`` clause.
- That clause must have ``ORDER BY`` in the window specification.

**Example:**

.. code-block:: python

    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "category": ["A", "A", "A", "B", "B", "B"],
        "score": [85, 90, 90, 75, 80, 80]
    })
    df.sql("""
      SELECT
        id,
        category,
        score,
        DENSE_RANK() OVER (PARTITION BY category ORDER BY score DESC) AS dense_rank,
        RANK() OVER (PARTITION BY category ORDER BY score DESC) AS rank
      FROM self
      ORDER BY category, score DESC
    """)
    # shape: (6, 5)
    # ┌─────┬──────────┬───────┬────────────┬──────┐
    # │ id  ┆ category ┆ score ┆ dense_rank ┆ rank │
    # │ --- ┆ ---      ┆ ---   ┆ ---        ┆ ---  │
    # │ i64 ┆ str      ┆ i64   ┆ u32        ┆ u32  │
    # ╞═════╪══════════╪═══════╪════════════╪══════╡
    # │ 2   ┆ A        ┆ 90    ┆ 1          ┆ 1    │
    # │ 3   ┆ A        ┆ 90    ┆ 1          ┆ 1    │
    # │ 1   ┆ A        ┆ 85    ┆ 2          ┆ 3    │2)
    # │ 5   ┆ B        ┆ 80    ┆ 1          ┆ 1    │
    # │ 6   ┆ B        ┆ 80    ┆ 1          ┆ 1    │
    # │ 4   ┆ B        ┆ 75    ┆ 2          ┆ 3    │2)
    # └─────┴──────────┴───────┴────────────┴──────┘


.. _row_number:

ROW_NUMBER
----------
Returns the sequential row number, optionally within a window partition, starting from 1. Unlike
``RANK`` and ``DENSE_RANK``, ``ROW_NUMBER`` always returns unique numbers even when values are tied.

**Example:**

.. code-block:: python

    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "category": ["A", "A", "A", "B", "B", "B"],
        "value": [100, 200, 200, 150, 300, 150]
    })
    df.sql("""
      SELECT
        ROW_NUMBER() AS x,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY id) AS y,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY id DESC) AS z,
        category,
        value
      FROM self
      ORDER BY category, id
    """)
    # shape: (6, 5)
    # ┌─────┬─────┬─────┬──────────┬───────┐
    # │ x   ┆ y   ┆ z   ┆ category ┆ value │
    # │ --- ┆ --- ┆ --- ┆ ---      ┆ ---   │
    # │ u32 ┆ u32 ┆ u32 ┆ str      ┆ i64   │
    # ╞═════╪═════╪═════╪══════════╪═══════╡
    # │ 1   ┆ 1   ┆ 3   ┆ A        ┆ 100   │
    # │ 2   ┆ 2   ┆ 2   ┆ A        ┆ 200   │
    # │ 3   ┆ 3   ┆ 1   ┆ A        ┆ 200   │
    # │ 4   ┆ 1   ┆ 3   ┆ B        ┆ 150   │
    # │ 5   ┆ 2   ┆ 2   ┆ B        ┆ 300   │
    # │ 6   ┆ 3   ┆ 1   ┆ B        ┆ 150   │
    # └─────┴─────┴─────┴──────────┴───────┘


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
        FIRST_VALUE(value) OVER (
          PARTITION BY label ORDER BY idx
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS first_val,
        LAST_VALUE(value) OVER (
          PARTITION BY label ORDER BY idx
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_val,
        SUM(value) OVER (
          PARTITION BY label ORDER BY idx
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_total_by_label
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
