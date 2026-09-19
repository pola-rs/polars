Conditional
===========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`COALESCE <coalesce>`
     - Returns the first non-null value in the provided values/columns.
   * - :ref:`GREATEST <greatest>`
     - Returns the greatest value in the list of expressions.
   * - :ref:`IF <if>`
     - Returns expr1 if the boolean condition provided as the first parameter evaluates to true, and expr2 otherwise.
   * - :ref:`IFNULL <ifnull>`
     - If an expression value is NULL, return an alternative value.
   * - :ref:`LEAST <least>`
     - Returns the smallest value in the list of expressions.
   * - :ref:`NULLIF <nullif>`
     - Returns NULL if two expressions are equal, otherwise returns the first.

.. _coalesce:

COALESCE
--------
Returns the first non-null value in the provided values/columns.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [1, None, 3, None],
        "bar": [1, 2, None, 4],
      }
    )
    df.sql("""
      SELECT foo, bar, COALESCE(foo, bar) AS baz FROM self
    """)
    # shape: (4, 3)
    # ┌──────┬──────┬─────┐
    # │ foo  ┆ bar  ┆ baz │
    # │ ---  ┆ ---  ┆ --- │
    # │ i64  ┆ i64  ┆ i64 │
    # ╞══════╪══════╪═════╡
    # │ 1    ┆ 1    ┆ 1   │
    # │ null ┆ 2    ┆ 2   │
    # │ 3    ┆ null ┆ 3   │
    # │ null ┆ 4    ┆ 4   │
    # └──────┴──────┴─────┘

.. _greatest:

GREATEST
--------
Returns the greatest value in the list of expressions.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [100, 200, 300, 400], 
        "bar": [20, 10, 30, 40]
      }
    )
    df.sql("""
      SELECT GREATEST(foo, bar) AS baz FROM self
    """)
    # shape: (4, 1)
    # ┌─────┐
    # │ baz │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 100 │
    # │ 200 │
    # │ 300 │
    # │ 400 │
    # └─────┘

.. _if:

IF
--
Returns expr1 if the boolean condition provided as the first parameter evaluates to true, and expr2 otherwise.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [100, 200, 300, 400],
        "bar": [10, 20, 30, 40]
      }
    )
    df.sql("""
      SELECT IF(foo < 250, 111, 999) AS baz FROM self
    """)
    # shape: (4, 1)
    # ┌─────┐
    # │ baz │
    # │ --- │
    # │ i32 │
    # ╞═════╡
    # │ 111 │
    # │ 111 │
    # │ 999 │
    # │ 999 │
    # └─────┘

.. _ifnull:

IFNULL
------
If an expression value is NULL, return an alternative value.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["a", None, None, "d"],
        "bar": [1, 2, 3, 4],
      }
    )
    df.sql("""
      SELECT IFNULL(foo, 'n/a') AS baz FROM self
    """)
    # shape: (4, 1)
    # ┌─────┐
    # │ baz │
    # │ --- │
    # │ str │
    # ╞═════╡
    # │ a   │
    # │ n/a │
    # │ n/a │
    # │ d   │
    # └─────┘

.. _least:

LEAST
-----
Returns the smallest value in the list of expressions.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [100, 200, 300, 400],
        "bar": [20, 10, 30, 40]
      }
    )
    df.sql("""
      SELECT LEAST(foo, bar) AS baz FROM self
    """)
    # shape: (4, 1)
    # ┌─────┐
    # │ baz │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 20  │
    # │ 10  │
    # │ 30  │
    # │ 40  │
    # └─────┘

.. _nullif:

NULLIF
------
Returns NULL if two expressions are equal, otherwise returns the first.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [100, 200, 300, 400],
        "bar": [20, 10, 30, 40]
      }
    )
    df.sql("""
      SELECT NULLIF(foo, bar) AS baz FROM self
    """)
    # shape: (4, 1)
    # ┌─────┐
    # │ baz │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 100 │
    # │ 200 │
    # │ 300 │
    # │ 400 │
    # └─────┘
