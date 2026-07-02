Aggregate
=========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`AVG <avg>`
     - Returns the average (mean) of all the elements in the grouping.
   * - :ref:`CORR <corr>`
     - Returns the Pearson correlation coefficient between two columns.
   * - :ref:`COUNT <count>`
     - Returns the amount of elements in the grouping.
   * - :ref:`COVAR <covar>`
     - Returns the covariance between two columns.
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
   * - :ref:`QUANTILE_CONT <quantile_cont>`
     - Returns the continuous quantile element from the grouping (interpolated value between two closest values).
   * - :ref:`QUANTILE_DISC <quantile_disc>`
     - Divides the [0, 1] interval into equal-length subintervals, each corresponding to a value, and returns the
       value associated with the subinterval where the quantile value falls.
   * - :ref:`STDDEV <stddev>`
     - Returns the standard deviation of all the elements in the grouping.
   * - :ref:`STRING_AGG <string_agg>`
     - Concatenates the input string values into a single string, separated by a delimiter.
   * - :ref:`SUM <sum>`
     - Returns the sum of all the elements in the grouping.
   * - :ref:`VARIANCE <variance>`
     - Returns the variance of all the elements in the grouping.


.. _filter:

Filtering aggregates
--------------------
Any aggregate function call can be qualified with a ``FILTER (WHERE …)`` clause
that restricts it to the rows where the predicate is true. The clause is attached
to an individual aggregate call and is independent of the query's ``WHERE`` clause,
so multiple aggregates in the same ``SELECT`` can see different row sets.

.. note::

   ``FILTER`` cannot be combined with ``OVER`` on the same aggregate call.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B"],
            "value":    [10, 20, 30, 40, 50, 60],
        }
    )
    df.sql("""
      SELECT
        category,
        SUM(value) AS total,
        SUM(value) FILTER (WHERE value > 25) AS total_high,
        SUM(value) FILTER (WHERE value <= 25) AS total_low,
        COUNT(*) FILTER (WHERE value >= 40) AS n_top
      FROM self
      GROUP BY category
      ORDER BY category
    """)
    # shape: (2, 5)
    # ┌──────────┬───────┬────────────┬───────────┬───────┐
    # │ category ┆ total ┆ total_high ┆ total_low ┆ n_top │
    # │ ---      ┆ ---   ┆ ---        ┆ ---       ┆ ---   │
    # │ str      ┆ i64   ┆ i64        ┆ i64       ┆ u32   │
    # ╞══════════╪═══════╪════════════╪═══════════╪═══════╡
    # │ A        ┆ 90    ┆ 80         ┆ 10        ┆ 1     │
    # │ B        ┆ 120   ┆ 100        ┆ 20        ┆ 2     │
    # └──────────┴───────┴────────────┴───────────┴───────┘


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

.. _corr:

CORR
----
Returns the Pearson correlation coefficient between two columns.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [1, 2, 3, 4, 5], "bar": [2, 4, 7, 5, 9]})
    df.sql("""
      SELECT CORR(foo, bar) AS corr FROM self
    """)
    # shape: (1, 1)
    # ┌──────────┐
    # │ corr     │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 0.877809 │
    # └──────────┘


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

.. _covar:

COVAR
-----
Returns the covariance between two columns.

.. admonition:: Aliases
    
   `COVAR_SAMP`

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [1, 2, 3, 4, 5], "bar": [2, 4, 7, 5, 9]})
    df.sql("""
      SELECT COVAR(foo, bar) AS covar FROM self
    """)

    # shape: (1, 1)
    # ┌───────┐
    # │ covar │
    # │ ---   │
    # │ f64   │
    # ╞═══════╡
    # │ 3.75  │
    # └───────┘


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


.. _quantile_cont:

QUANTILE_CONT
-------------
Returns the continuous quantile element from the grouping (interpolated value between two closest values).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [5, 20, 10, 30, 70, 40, 10, 90]})
    df.sql("""
      SELECT
        QUANTILE_CONT(foo, 0.25) AS foo_q25,
        QUANTILE_CONT(foo, 0.50) AS foo_q50,
        QUANTILE_CONT(foo, 0.75) AS foo_q75,
      FROM self
    """)
    # shape: (1, 3)
    # ┌─────────┬─────────┬─────────┐
    # │ foo_q25 ┆ foo_q50 ┆ foo_q75 │
    # │ ---     ┆ ---     ┆ ---     │
    # │ f64     ┆ f64     ┆ f64     │
    # ╞═════════╪═════════╪═════════╡
    # │ 10.0    ┆ 25.0    ┆ 47.5    │
    # └─────────┴─────────┴─────────┘


.. _quantile_disc:

QUANTILE_DISC
-------------
Divides the [0, 1] interval into equal-length subintervals, each corresponding to a value, and
returns the value associated with the subinterval where the quantile value falls.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [5, 20, 10, 30, 70, 40, 10, 90]})
    df.sql("""
      SELECT
        QUANTILE_DISC(foo, 0.25) AS foo_q25,
        QUANTILE_DISC(foo, 0.50) AS foo_q50,
        QUANTILE_DISC(foo, 0.75) AS foo_q75,
      FROM self
    """)
    # shape: (1, 3)
    # ┌─────────┬─────────┬─────────┐
    # │ foo_q25 ┆ foo_q50 ┆ foo_q75 │
    # │ ---     ┆ ---     ┆ ---     │
    # │ f64     ┆ f64     ┆ f64     │
    # ╞═════════╪═════════╪═════════╡
    # │ 10.0    ┆ 20.0    ┆ 40.0    │
    # └─────────┴─────────┴─────────┘


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

.. _string_agg:

STRING_AGG
----------
Concatenates the input string values into a single string, separated by the given
delimiter. Supports ``DISTINCT`` and in-argument ``ORDER BY`` and ``LIMIT``
clauses to control which values are concatenated and in what order; the
separator is optional, defaulting to ``","``.

.. admonition:: Aliases

   `GROUP_CONCAT`, `LISTAGG`

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B"],
            "label": ["x1", "y1", "x2", "y2", "x3", "y3"],
            "value": [10, 20, 30, 40, 50, 60],
        }
    )
    df.sql("""
      SELECT
        category,
        STRING_AGG(label LIMIT 2) AS two_labels,
        STRING_AGG(label, ':' ORDER BY value DESC) AS labels_desc,
        STRING_AGG(label, ',' ORDER BY value ASC) FILTER(WHERE label !~ '1$') AS labels_omit_1,
      FROM self
      GROUP BY category
      ORDER BY category
    """)
    # shape: (2, 4)
    # ┌──────────┬────────────┬─────────────┬───────────────┐
    # │ category ┆ two_labels ┆ labels_desc ┆ labels_omit_1 │
    # │ ---      ┆ ---        ┆ ---         ┆ ---           │
    # │ str      ┆ str        ┆ str         ┆ str           │
    # ╞══════════╪════════════╪═════════════╪═══════════════╡
    # │ A        ┆ x1,x2      ┆ x3:x2:x1    ┆ x2,x3         │
    # │ B        ┆ y1,y2      ┆ y3:y2:y1    ┆ y2,y3         │
    # └──────────┴────────────┴─────────────┴───────────────┘

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
