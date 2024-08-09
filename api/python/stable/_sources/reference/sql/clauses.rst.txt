SQL Clauses
===========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`SELECT <select>`
     - Retrieves specific column data from one or more tables.
   * - :ref:`DISTINCT <distinct>`
     - Returns unique values from a query.
   * - :ref:`FROM <from>`
     - Specify the table(s) from which to retrieve or delete data.
   * - :ref:`JOIN <join>`
     - Combine rows from two or more tables based on a related column.
   * - :ref:`WHERE <where>`
     - Filter rows returned from the query based on the given conditions.
   * - :ref:`GROUP BY <group_by>`
     - Aggregate row values based based on one or more key columns.
   * - :ref:`HAVING <having>`
     - Filter groups in a `GROUP BY` based on the given conditions.
   * - :ref:`ORDER BY <order_by>`
     - Sort the query result based on one or more specified columns.
   * - :ref:`LIMIT <limit>`
     - Specify the number of rows returned.
   * - :ref:`OFFSET <offset>`
     - Skip a specified number of rows.


.. _select:

SELECT
------
Select the columns to be returned by the query.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [1, 2, 3],
        "b": ["zz", "yy", "xx"],
      }
    )
    df.sql("""
      SELECT a, b FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 1   ┆ zz  │
    # │ 2   ┆ yy  │
    # │ 3   ┆ xx  │
    # └─────┴─────┘

.. _distinct:

DISTINCT
--------
Returns unique values from a query.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [1, 2, 2, 1],
        "b": ["xx", "yy", "yy", "xx"],
      }
    )
    df.sql("""
      SELECT DISTINCT * FROM self
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 1   ┆ xx  │
    # │ 2   ┆ yy  │
    # └─────┴─────┘

.. _from:

FROM
----
Specifies the table(s) from which to retrieve or delete data.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [1, 2, 3],
        "b": ["zz", "yy", "xx"],
      }
    )
    df.sql("""
      SELECT * FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 1   ┆ zz  │
    # │ 2   ┆ yy  │
    # │ 3   ┆ xx  │
    # └─────┴─────┘

.. _join:

JOIN
----
Combines rows from two or more tables based on a related column. 

**Join Types**

* `CROSS JOIN`
* `[NATURAL] FULL JOIN`
* `[NATURAL] INNER JOIN`
* `[NATURAL] LEFT JOIN`
* `[LEFT | RIGHT] ANTI JOIN`
* `[LEFT | RIGHT] SEMI JOIN`

**Example:**

.. code-block:: python

    df1 = pl.DataFrame(
      {
        "foo": [1, 2, 3],
        "ham": ["a", "b", "c"],
      }
    )
    df2 = pl.DataFrame(
      {
        "apple": ["x", "y", "z"],
        "ham": ["a", "b", "d"],
      }
    )
    pl.sql("""
      SELECT foo, apple, COALESCE(df1.ham, df2.ham) AS ham
      FROM df1 FULL JOIN df2
      USING (ham)
    """).collect()
    # shape: (4, 3)
    # ┌──────┬───────┬─────┐
    # │ foo  ┆ apple ┆ ham │
    # │ ---  ┆ ---   ┆ --- │
    # │ i64  ┆ str   ┆ str │
    # ╞══════╪═══════╪═════╡
    # │ 1    ┆ x     ┆ a   │
    # │ 2    ┆ y     ┆ b   │
    # │ null ┆ z     ┆ d   │
    # │ 3    ┆ null  ┆ c   │
    # └──────┴───────┴─────┘

    pl.sql("""
      SELECT COLUMNS('^\w+$')
      FROM df1 NATURAL INNER JOIN df2
    """).collect()
    # shape: (2, 3)
    # ┌─────┬───────┬─────┐
    # │ foo ┆ apple ┆ ham │
    # │ --- ┆ ---   ┆ --- │
    # │ i64 ┆ str   ┆ str │
    # ╞═════╪═══════╪═════╡
    # │ 1   ┆ x     ┆ a   │
    # │ 2   ┆ y     ┆ b   │
    # └─────┴───────┴─────┘

.. _where:

WHERE
-----

Filter rows returned from the query based on the given conditions.

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [30, 40, 50],
        "ham": ["a", "b", "c"],
      }
    )
    df.sql("""
      SELECT * FROM self WHERE foo > 42
    """)
    # shape: (1, 2)
    # ┌─────┬─────┐
    # │ foo ┆ ham │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 50  ┆ c   │
    # └─────┴─────┘

.. _group_by:

GROUP BY
--------
Group rows that have the same values in specified columns into summary rows.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
          "foo": ["a", "b", "b"],
          "bar": [10, 20, 30],
        }
      )
    df.sql("""
      SELECT foo, SUM(bar) FROM self GROUP BY foo
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ b   ┆ 50  │
    # │ a   ┆ 10  │
    # └─────┴─────┘

.. _having:

HAVING
------
Filter groups in a `GROUP BY` based on the given conditions.

.. code-block:: python

    df = pl.DataFrame(
          {
          "foo": ["a", "b", "b", "c"],
          "bar": [10, 20, 30, 40],
        }
      )
    df.sql("""
      SELECT foo, SUM(bar) FROM self GROUP BY foo HAVING bar >= 40
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ c   ┆ 40  │
    # │ b   ┆ 50  │
    # └─────┴─────┘

.. _order_by:

ORDER BY
--------
Sort the query result based on one or more specified columns.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["b", "a", "c", "b"],
        "bar": [20, 10, 40, 30],
      }
    )
    df.sql("""
      SELECT foo, bar FROM self ORDER BY bar DESC
    """)
    # shape: (4, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ c   ┆ 40  │
    # │ b   ┆ 30  │
    # │ b   ┆ 20  │
    # │ a   ┆ 10  │
    # └─────┴─────┘

.. _limit:

LIMIT
-----
Limit the number of rows returned by the query.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["b", "a", "c", "b"],
        "bar": [20, 10, 40, 30],
      }
    )
    df.sql("""
      SELECT foo, bar FROM self LIMIT 2
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ b   ┆ 20  │
    # │ a   ┆ 10  │
    # └─────┴─────┘

.. _offset:

OFFSET
------
Skip a number of rows before starting to return rows from the query.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["b", "a", "c", "b"],
        "bar": [20, 10, 40, 30],
      }
    )
    df.sql("""
      SELECT foo, bar FROM self LIMIT 2 OFFSET 2
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ c   ┆ 40  │
    # │ b   ┆ 30  │
    # └─────┴─────┘
