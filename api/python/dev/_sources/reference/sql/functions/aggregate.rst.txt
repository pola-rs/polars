Aggregate
=========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`AVG <avg>`
     - Returns the average (mean) of all the elements in the grouping.
   * - :ref:`COUNT <count>`
     - Returns the amount of elements in the grouping.
   * - :ref:`FIRST <first>`
     - Returns the first element of the grouping.
   * - :ref:`LAST <last>`
     - Returns the last element of the grouping.
   * - :ref:`MAX <max>`
     - Returns the greatest (maximum) of all the elements in the grouping.
   * - :ref:`MEDIAN <median>`
     - Returns the median element from the grouping.
   * - :ref:`MIN <min>`
     - Returns the smallest (minimum) of all the elements in the grouping.
   * - :ref:`STDDEV <stddev>`
     - Returns the standard deviation of all the elements in the grouping.
   * - :ref:`SUM <sum>`
     - Returns the sum of all the elements in the grouping.
   * - :ref:`VARIANCE <variance>`
     - Returns the variance of all the elements in the grouping.

.. _avg:

AVG
---
Returns the average (mean) of all the elements in the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"bar": [20, 10, 30, 40]})
    df.sql("""
      SELECT AVG(bar) AS bar_avg FROM self
    """)
    # shape: (1, 1)
    # ┌─────────┐
    # │ bar_avg │
    # │ ---     │
    # │ f64     │
    # ╞═════════╡
    # │ 25.0    │
    # └─────────┘

.. _count:

COUNT
-----
Returns the amount of elements in the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    df.sql("""
      SELECT
        COUNT(bar) AS n_bar,
        COUNT(DISTINCT foo) AS n_foo_unique
      FROM self
    """)

    # shape: (1, 2)
    # ┌───────┬──────────────┐
    # │ n_bar ┆ n_foo_unique │
    # │ ---   ┆ ---          │
    # │ u32   ┆ u32          │
    # ╞═══════╪══════════════╡
    # │ 4     ┆ 3            │
    # └───────┴──────────────┘

.. _first:

FIRST
-----
Returns the first element of the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["b", "a", "b", "c"]})
    df.sql("""
      SELECT FIRST(foo) AS ff FROM self
    """)
    # shape: (1, 1)
    # ┌─────┐
    # │ ff  │
    # │ --- │
    # │ str │
    # ╞═════╡
    # │ b   │
    # └─────┘

.. _last:

LAST
----
Returns the last element of the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": ["b", "a", "b", "c"]})
    df.sql("""
      SELECT LAST(foo) AS lf FROM self
    """)
    # shape: (1, 1)
    # ┌─────┐
    # │ lf  │
    # │ --- │
    # │ str │
    # ╞═════╡
    # │ c   │
    # └─────┘

.. _max:

MAX
---
Returns the greatest (maximum) of all the elements in the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"bar": [20, 10, 30, 40]})
    df.sql("""
      SELECT MAX(bar) AS bar_max FROM self
    """)
    # shape: (1, 1)
    # ┌─────────┐
    # │ bar_max │
    # │ ---     │
    # │ i64     │
    # ╞═════════╡
    # │ 40      │
    # └─────────┘

.. _median:

MEDIAN
------
Returns the median element from the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"bar": [20, 10, 30, 40]})
    df.sql("""
      SELECT MEDIAN(bar) AS bar_median FROM self
    """)
    # shape: (1, 1)
    # ┌────────────┐
    # │ bar_median │
    # │ ---        │
    # │ f64        │
    # ╞════════════╡
    # │ 25.0       │
    # └────────────┘

.. _min:

MIN
---
Returns the smallest (minimum) of all the elements in the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"bar": [20, 10, 30, 40]})
    df.sql("""
      SELECT MIN(bar) AS bar_min FROM self
    """)
    # shape: (1, 1)
    # ┌─────────┐
    # │ bar_min │
    # │ ---     │
    # │ i64     │
    # ╞═════════╡
    # │ 10      │
    # └─────────┘

.. _stddev:

STDDEV
------
Returns the sample standard deviation of all the elements in the grouping.

.. admonition:: Aliases

   `STDEV`, `STDEV_SAMP`, `STDDEV_SAMP`

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "foo": [10, 20, 8],
            "bar": [10, 7, 18],
        }
    )
    df.sql("""
      SELECT STDDEV(foo) AS foo_std, STDDEV(bar) AS bar_std FROM self
    """)
    # shape: (1, 2)
    # ┌──────────┬──────────┐
    # │ foo_std  ┆ bar_std  │
    # │ ---      ┆ ---      │
    # │ f64      ┆ f64      │
    # ╞══════════╪══════════╡
    # │ 6.429101 ┆ 5.686241 │
    # └──────────┴──────────┘

.. _sum:

SUM
---
Returns the sum of all the elements in the grouping.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    df.sql("""
      SELECT SUM(foo) AS foo_sum, SUM(bar) AS bar_sum FROM self
    """)
    # shape: (1, 2)
    # ┌─────────┬─────────┐
    # │ foo_sum ┆ bar_sum │
    # │ ---     ┆ ---     │
    # │ i64     ┆ i64     │
    # ╞═════════╪═════════╡
    # │ 6       ┆ 21      │
    # └─────────┴─────────┘

.. _variance:

VARIANCE
--------
Returns the variance of all the elements in the grouping.

.. admonition:: Aliases

   `VAR`, `VAR_SAMP`

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "foo": [10, 20, 8],
            "bar": [10, 7, 18],
        }
    )
    df.sql("""
      SELECT VARIANCE(foo) AS foo_var, VARIANCE(bar) AS bar_var FROM self
    """)
    # shape: (1, 2)
    # ┌───────────┬───────────┐
    # │ foo_var   ┆ bar_var   │
    # │ ---       ┆ ---       │
    # │ f64       ┆ f64       │
    # ╞═══════════╪═══════════╡
    # │ 41.333333 ┆ 32.333333 │
    # └───────────┴───────────┘
