Set Operations
==============

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`UNION <union>`
     - Combine the distinct result sets of two or more SELECT statements.
       The final result set will have no duplicate rows.
   * - :ref:`UNION ALL <union_all>`
     - Combine the complete result sets of two or more SELECT statements.
       The final result set will be composed of all rows from each query.
   * - :ref:`UNION [ALL] BY NAME <union_by_name>`
     - Combine the result sets of two or more SELECT statements by column name
       instead of by position; if `ALL` is omitted the final result will have
       no duplicate rows.


.. _union:

UNION
-----
Combine the distinct result sets of two or more SELECT statements.
The final result set will have no duplicate rows.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
          {
              "foo": [1, 2, 3],
              "ham": ["a", "a", "c"],
          }
      )
    >>> other_df = pl.DataFrame(
          {
              "apple": ["x", "y", "z"],
              "ham": ["a", "b", "b"],
          }
      )
    >>> df.sql("""
        SELECT ham FROM df
        UNION
        SELECT ham FROM other_df
        """
      )
    shape: (3, 1)
    ┌─────┐
    │ ham │
    │ --- │
    │ str │
    ╞═════╡
    │ c   │
    │ b   │
    │ a   │
    └─────┘

.. _union_all:

UNION ALL
---------
Combine the complete result sets of two or more SELECT statements.
The final result set will be composed of all rows from each query.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
          {
              "foo": [1, 2, 3],
              "ham": ["a", "b", "c"],
          }
      )
    >>> other_df = pl.DataFrame(
          {
              "apple": ["x", "y", "z"],
              "ham": ["a", "b", "d"],
          }
      )
    >>> df.sql("""
        SELECT ham FROM df
        UNION ALL
        SELECT ham FROM other_df
        """
      )
    shape: (6, 1)
    ┌─────┐
    │ ham │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    │ a   │
    │ b   │
    │ d   │
    └─────┘

.. _union_by_name:

UNION BY NAME
-------------
Combine the result sets of two or more SELECT statements by column name
instead of by position; if `ALL` is omitted the final result will have
no duplicate rows.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
          {
              "foo": [1, 2, 3],
              "ham": ["a", "a", "c"],
          }
      )
    >>> other_df = pl.DataFrame(
          {
              "apple": ["x", "y", "z"],
              "ham": ["a", "b", "c"],
          }
      )
    >>> df.sql("""
        SELECT ham FROM df
        UNION BY NAME
        SELECT ham FROM other_df
        """
      )
    shape: (6, 2)
    ┌──────┬──────┐
    │ foo  ┆ ham  │
    │ ---  ┆ ---  │
    │ i64  ┆ str  │
    ╞══════╪══════╡
    │ null ┆ c    │
    │ 2    ┆ null │
    │ 1    ┆ null │
    │ 3    ┆ null │
    │ null ┆ b    │
    │ null ┆ a    │
    └──────┴──────┘
