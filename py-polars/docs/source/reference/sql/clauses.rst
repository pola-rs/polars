Query Clauses
=============

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`SELECT <select>`
     - Retrieves specific column data from one or more tables.
   * - :ref:`DISTINCT <distinct>`
     - Returns unique values from a query.
   * - :ref:`DISTINCT ON <distinct_on>`
     - Returns the first row for each unique combination of the specified columns.
   * - :ref:`FROM <from>`
     - Specify the table(s) from which to retrieve or delete data. Can also be used as the leading clause.
   * - :ref:`JOIN <join>`
     - Combine rows from two or more tables based on a related column.
   * - :ref:`WHERE <where>`
     - Filter rows returned from the query based on the given conditions.
   * - :ref:`GROUP BY <group_by>`
     - Aggregate row values based based on one or more key columns.
   * - :ref:`GROUP BY ALL <group_by_all>`
     - Automatically group by all non-aggregate columns in the projection.
   * - :ref:`HAVING <having>`
     - Filter groups in a `GROUP BY` based on the given conditions.
   * - :ref:`WINDOW <window>`
     - Define named window specifications for window functions.
   * - :ref:`QUALIFY <qualify>`
     - Filter rows in a query based on window function results.
   * - :ref:`ORDER BY <order_by>`
     - Sort the query result based on one or more specified columns.
   * - :ref:`ORDER BY ALL <order_by_all>`
     - Sort the query result by all selected columns.
   * - :ref:`OFFSET <offset>`
     - Skip a specified number of rows.
   * - :ref:`LIMIT <limit>`
     - Specify the number of rows returned.
   * - :ref:`FETCH <fetch>`
     - Limit the number of rows returned (alternative to LIMIT).


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

.. note::

   Use of bare ``FROM tbl`` is also supported, as shorthand for ``SELECT * FROM tbl``;
   see the :ref:`FROM <from>` clause for more detail.

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

.. _distinct_on:

DISTINCT ON
-----------
Returns the first row for each unique combination of the specified columns. When used
with ``ORDER BY``, this keeps the first row per group according to the given ordering.

.. note::

   ``DISTINCT ON`` only supports column names (not arbitrary expressions).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "category": ["A", "A", "A", "B", "B", "B"],
        "value": [30, 10, 20, 50, 40, 60],
        "label": ["x", "y", "z", "p", "q", "r"],
      }
    )
    df.sql("""
      SELECT DISTINCT ON (category)
        category,
        value,
        label
      FROM self
      ORDER BY category, value DESC
    """)
    # shape: (2, 3)
    # ┌──────────┬───────┬───────┐
    # │ category ┆ value ┆ label │
    # │ ---      ┆ ---   ┆ ---   │
    # │ str      ┆ i64   ┆ str   │
    # ╞══════════╪═══════╪═══════╡
    # │ A        ┆ 30    ┆ x     │
    # │ B        ┆ 60    ┆ r     │
    # └──────────┴───────┴───────┘

.. _from:

FROM
----
Specifies the table(s) from which to retrieve or delete data.

In addition to the usual ``SELECT ... FROM tbl`` syntax, the ``FROM`` clause can
also be used as the leading clause in a query, supporting the following variations:

* ``FROM tbl`` - equivalent to ``SELECT * FROM tbl``.
* ``FROM tbl SELECT ...`` - a reordered ``SELECT`` with explicit projections.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [1, 2, 3],
        "b": ["zz", "yy", "xx"],
      }
    )
    for query in (
      "SELECT * FROM self",
      "FROM self SELECT *",
      "FROM self",
    ):
      df.sql(query)
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

Using ``FROM`` as the leading clause, with ``SELECT``:

.. code-block:: python

    df.sql("""
      FROM self SELECT b, a
    """)
    # shape: (3, 2)
    # ┌─────┬─────┐
    # │ b   ┆ a   │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ zz  ┆ 1   │
    # │ yy  ┆ 2   │
    # │ xx  ┆ 3   │
    # └─────┴─────┘

.. _join:

JOIN
----
Combines rows from two or more tables based on a related column.

**Join Types**

* `CROSS JOIN`
* `[NATURAL] FULL [OUTER] JOIN`
* `[NATURAL] INNER [OUTER] JOIN`
* `[NATURAL] LEFT [OUTER] JOIN`
* `[NATURAL] RIGHT [OUTER] JOIN`
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
      SELECT * FROM df1 NATURAL INNER JOIN df2
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

.. _group_by_all:

GROUP BY ALL
------------
Automatically groups by all columns in the ``SELECT`` projection that are not wrapped in
an aggregate function, a window expression, or a literal value. This is a convenience
shorthand that avoids having to manually repeat column names in the ``GROUP BY`` clause.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "category": ["A", "A", "B", "B"],
        "sub": ["x", "y", "x", "y"],
        "value": [10, 20, 30, 40],
      }
    )
    df.sql("""
      SELECT category, sub, SUM(value) AS total
      FROM self
      GROUP BY ALL
      ORDER BY category, sub
    """)
    # shape: (4, 3)
    # ┌──────────┬─────┬───────┐
    # │ category ┆ sub ┆ total │
    # │ ---      ┆ --- ┆ ---   │
    # │ str      ┆ str ┆ i64   │
    # ╞══════════╪═════╪═══════╡
    # │ A        ┆ x   ┆ 10    │
    # │ A        ┆ y   ┆ 20    │
    # │ B        ┆ x   ┆ 30    │
    # │ B        ┆ y   ┆ 40    │
    # └──────────┴─────┴───────┘

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

.. _window:

WINDOW
------
Define named window specifications that can be referenced by window functions.

**Example:**

One window, multiple expressions:

.. code-block:: python

    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "category": ["A", "A", "A", "B", "B", "B", "C"],
        "value": [20, 10, 30, 15, 50, 30, 35],
    })
    df.sql("""
      SELECT
        category,
        value,
        SUM(value) OVER w AS "w:sum",
        MIN(value) OVER w AS "w:min",
        AVG(value) OVER w AS "w:avg",
      FROM self
      WINDOW w AS (PARTITION BY category ORDER BY value)
      ORDER BY category, value
    """)
    # shape: (7, 5)
    # ┌──────────┬───────┬───────┬───────┬───────────┐
    # │ category ┆ value ┆ w:sum ┆ w:min ┆ w:avg     │
    # │ ---      ┆ ---   ┆ ---   ┆ ---   ┆ ---       │
    # │ str      ┆ i64   ┆ i64   ┆ i64   ┆ f64       │
    # ╞══════════╪═══════╪═══════╪═══════╪═══════════╡
    # │ A        ┆ 10    ┆ 10    ┆ 10    ┆ 20.0      │
    # │ A        ┆ 20    ┆ 30    ┆ 10    ┆ 20.0      │
    # │ A        ┆ 30    ┆ 60    ┆ 10    ┆ 20.0      │
    # │ B        ┆ 15    ┆ 15    ┆ 15    ┆ 31.666667 │
    # │ B        ┆ 30    ┆ 45    ┆ 15    ┆ 31.666667 │
    # │ B        ┆ 50    ┆ 95    ┆ 15    ┆ 31.666667 │
    # │ C        ┆ 35    ┆ 35    ┆ 35    ┆ 35.0      │
    # └──────────┴───────┴───────┴───────┴───────────┘

Multiple windows, multiple expressions:

.. code-block:: python

    df.sql("""
      SELECT
        category,
        value,
        AVG(value) OVER w1 AS category_avg,
        SUM(value) OVER w2 AS running_value,
        COUNT(*) OVER w3 AS total_count
      FROM self
      WINDOW
        w1 AS (PARTITION BY category),
        w2 AS (ORDER BY value),
        w3 AS ()
      ORDER BY category, value
    """)
    # shape: (7, 5)
    # ┌──────────┬───────┬──────────────┬───────────────┬─────────────┐
    # │ category ┆ value ┆ category_avg ┆ running_value ┆ total_count │
    # │ ---      ┆ ---   ┆ ---          ┆ ---           ┆ ---         │
    # │ str      ┆ i64   ┆ f64          ┆ i64           ┆ u32         │
    # ╞══════════╪═══════╪══════════════╪═══════════════╪═════════════╡
    # │ A        ┆ 10    ┆ 20.0         ┆ 10            ┆ 7           │
    # │ A        ┆ 20    ┆ 20.0         ┆ 45            ┆ 7           │
    # │ A        ┆ 30    ┆ 20.0         ┆ 75            ┆ 7           │
    # │ B        ┆ 15    ┆ 31.666667    ┆ 25            ┆ 7           │
    # │ B        ┆ 30    ┆ 31.666667    ┆ 105           ┆ 7           │
    # │ B        ┆ 50    ┆ 31.666667    ┆ 190           ┆ 7           │
    # │ C        ┆ 35    ┆ 35.0         ┆ 140           ┆ 7           │
    # └──────────┴───────┴──────────────┴───────────────┴─────────────┘

.. _qualify:

QUALIFY
-------
Filter rows in a query based on window function results.

**Example:**

Constrain the result to the top (largest) two values per category:

.. code-block:: python

    df = pl.DataFrame({
        "id": [100, 200, 300, 400, 500, 600, 700, 800],
        "category": ["A", "A", "A", "B", "B", "B", "B", "A"],
        "value": [20, 15, 30, 25, 15, 50, 35, 45],
    })
    df.sql("""
      SELECT
        id,
        category,
        value
      FROM self
      WINDOW w AS (PARTITION BY category ORDER BY value DESC)
      QUALIFY ROW_NUMBER() OVER w <= 2
      ORDER BY category, value DESC
    """)
    # shape: (4, 3)
    # ┌─────┬──────────┬───────┐
    # │ id  ┆ category ┆ value │
    # │ --- ┆ ---      ┆ ---   │
    # │ i64 ┆ str      ┆ i64   │
    # ╞═════╪══════════╪═══════╡
    # │ 800 ┆ A        ┆ 45    │
    # │ 300 ┆ A        ┆ 30    │
    # │ 600 ┆ B        ┆ 50    │
    # │ 700 ┆ B        ┆ 35    │
    # └─────┴──────────┴───────┘

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

.. _order_by_all:

ORDER BY ALL
------------
Sort the query result by all selected columns. This is a convenience shorthand that
avoids repeating column names. The ``ASC``/``DESC`` and ``NULLS FIRST``/``NULLS LAST``
modifiers apply to every column.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": ["x", "y", "x", "y"],
        "b": [30, 10, 20, 40],
      }
    )
    df.sql("""
      SELECT a, b FROM self ORDER BY ALL
    """)
    # shape: (4, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ x   ┆ 20  │
    # │ x   ┆ 30  │
    # │ y   ┆ 10  │
    # │ y   ┆ 40  │
    # └─────┴─────┘

    df.sql("""
      SELECT a, b FROM self ORDER BY ALL DESC
    """)
    # shape: (4, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ y   ┆ 40  │
    # │ y   ┆ 10  │
    # │ x   ┆ 30  │
    # │ x   ┆ 20  │
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

.. _fetch:

FETCH
-----
Limit the number of rows returned by the query; this is the ANSI SQL standard
alternative to the ``LIMIT`` clause, and can be combined with ``OFFSET``. The
`WITH TIES` and `PERCENT` modifiers are not currently supported.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": ["b", "a", "c", "b"],
        "bar": [20, 10, 40, 30],
      }
    )
    df.sql("""
      SELECT foo, bar
      FROM self
      ORDER BY bar
      OFFSET 1 FETCH NEXT 2 ROWS ONLY
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ b   ┆ 20  │
    # │ b   ┆ 30  │
    # └─────┴─────┘
