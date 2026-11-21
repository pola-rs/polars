Set Operations
==============

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`EXCEPT <except>`
     - Combine the result sets of two SELECT statements, returning only the rows
       that appear in the first result set but not in the second.
   * - :ref:`INTERSECT <intersect>`
     - Combine the result sets of two SELECT statements, returning only the rows
       that appear in both result sets.
   * - :ref:`UNION <union>`
     - Combine the distinct result sets of two or more SELECT statements.
       The final result set will have no duplicate rows.
   * - :ref:`UNION ALL <union_all>`
     - Combine the complete result sets of two or more SELECT statements.
       The final result set will be composed of all rows from each query.
   * - :ref:`UNION [ALL] BY NAME <union_by_name>`
     - Combine the result sets of two or more SELECT statements, aligning columns
       by name instead of by ordinal position; if `ALL` is omitted the final result
       will have no duplicate rows. This also combines columns from both datasets.


.. _except:

EXCEPT
------
Combine the result sets of two SELECT statements, returning only the rows
that appear in the first result set but not in the second.

**Example:**

.. code-block:: python

    lf1 = pl.LazyFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    })
    lf2 = pl.LazyFrame({
        "id": [2, 3, 4],
        "age": [30, 25, 45],
        "name": ["Bob", "Charlie", "David"],
    })
    pl.sql("""
        SELECT id, name FROM lf1
        EXCEPT
        SELECT id, name FROM lf2
    """).sort(by="id").collect()
    # shape: (1, 2)
    # ┌─────┬───────┐
    # │ id  ┆ name  │
    # │ --- ┆ ---   │
    # │ i64 ┆ str   │
    # ╞═════╪═══════╡
    # │ 1   ┆ Alice │
    # └─────┴───────┘

.. _intersect:

INTERSECT
---------
Combine the result sets of two SELECT statements, returning only the rows
that appear in both result sets.

**Example:**

.. code-block:: python

    pl.sql("""
        SELECT id, name FROM lf1
        INTERSECT
        SELECT id, name FROM lf2
    """).sort(by="id").collect()
    # shape: (2, 2)
    # ┌─────┬─────────┐
    # │ id  ┆ name    │
    # │ --- ┆ ---     │
    # │ i64 ┆ str     │
    # ╞═════╪═════════╡
    # │ 2   ┆ Bob     │
    # │ 3   ┆ Charlie │
    # └─────┴─────────┘

.. _union:

UNION
-----
Combine the distinct result sets of two or more SELECT statements.
The final result set will have no duplicate rows.

**Example:**

.. code-block:: python

    pl.sql("""
        SELECT id, name FROM lf1
        UNION
        SELECT id, name FROM lf2
    """).sort(by="id").collect()
    # shape: (4, 2)
    # ┌─────┬─────────┐
    # │ id  ┆ name    │
    # │ --- ┆ ---     │
    # │ i64 ┆ str     │
    # ╞═════╪═════════╡
    # │ 1   ┆ Alice   │
    # │ 2   ┆ Bob     │
    # │ 3   ┆ Charlie │
    # │ 4   ┆ David   │
    # └─────┴─────────┘

.. _union_all:

UNION ALL
---------
Combine the complete result sets of two or more SELECT statements.
The final result set will be composed of all rows from each query.

**Example:**

.. code-block:: python

    pl.sql("""
        SELECT id, name FROM lf1
        UNION ALL
        SELECT id, name FROM lf2
    """).sort(by="id").collect()
    # shape: (6, 2)
    # ┌─────┬─────────┐
    # │ id  ┆ name    │
    # │ --- ┆ ---     │
    # │ i64 ┆ str     │
    # ╞═════╪═════════╡
    # │ 1   ┆ Alice   │
    # │ 2   ┆ Bob     │
    # │ 2   ┆ Bob     │
    # │ 3   ┆ Charlie │
    # │ 3   ┆ Charlie │
    # │ 4   ┆ David   │
    # └─────┴─────────┘

.. _union_by_name:

UNION BY NAME
-------------
Combine the result sets of two or more SELECT statements, aligning columns
by name instead of by ordinal position; if `ALL` is omitted the final result
will have no duplicate rows. This also combines columns from both datasets.

**Example:**

.. code-block:: python

    pl.sql("""
        SELECT * FROM lf1
        UNION BY NAME
        SELECT * FROM lf2
    """).sort(by="id").collect()
    # shape: (6, 3)
    # ┌─────┬─────────┬──────┐
    # │ id  ┆ name    ┆ age  │
    # │ --- ┆ ---     ┆ ---  │
    # │ i64 ┆ str     ┆ i64  │
    # ╞═════╪═════════╪══════╡
    # │ 1   ┆ Alice   ┆ null │
    # │ 2   ┆ Bob     ┆ null │
    # │ 2   ┆ Bob     ┆ 30   │
    # │ 3   ┆ Charlie ┆ 25   │
    # │ 3   ┆ Charlie ┆ null │
    # │ 4   ┆ David   ┆ 45   │
    # └─────┴─────────┴──────┘
