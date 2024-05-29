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

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT AVG(bar) FROM self")
    shape: (1, 1)
    ┌──────┐
    │ bar  │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ 25.0 │
    └──────┘

.. _count:

COUNT
-----
Returns the amount of elements in the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT COUNT(bar) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 4   │
    └─────┘

    >>> df.sql("SELECT COUNT(DISTINCT foo) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 3   │
    └─────┘

.. _first:

FIRST
-----
Returns the first element of the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT FIRST(foo) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ b   │
    └─────┘

.. _last:

LAST
----
Returns the last element of the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT LAST(foo) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ c   │
    └─────┘

.. _max:

MAX
---
Returns the greatest (maximum) of all the elements in the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT MAX(bar) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 40  │
    └─────┘

.. _median:

MEDIAN
------
Returns the median element from the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"], 
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT MEDIAN(bar) FROM self")
    shape: (1, 1)
    ┌──────┐
    │ bar  │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ 25.0 │
    └──────┘

.. _min:

MIN
---
Returns the smallest (minimum) of all the elements in the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": ["b", "a", "b", "c"],
        "bar": [20, 10, 30, 40]
      }
    )
    >>> df.sql("SELECT MIN(bar) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 10  │
    └─────┘

.. _stddev:

STDDEV
------
Returns the standard deviation of all the elements in the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    >>> df.sql("SELECT STDDEV(foo), STDDEV(bar) FROM self")
    shape: (1, 2)
    ┌─────┬─────┐
    │ bar ┆ foo │
    │ --- ┆ --- │
    │ f64 ┆ f64 │
    ╞═════╪═════╡
    │ 1.0 ┆ 1.0 │
    └─────┴─────┘

.. _sum:

SUM
---
Returns the sum of all the elements in the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    >>> df.sql("SELECT SUM(foo), SUM(bar) FROM self")
    shape: (1, 2)
    ┌─────┬─────┐
    │ bar ┆ foo │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 21  ┆ 6   │
    └─────┴─────┘

.. _variance:

VARIANCE
--------
Returns the variance of all the elements in the grouping.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    >>> df.sql("SELECT VARIANCE(foo) FROM self")
    shape: (1, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 1.0 │
    └─────┘
