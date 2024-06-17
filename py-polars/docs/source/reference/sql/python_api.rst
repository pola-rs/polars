==========
Python API
==========

.. currentmodule:: polars

Introduction
------------

There are four primary entry points to the Polars SQL interface, each operating at a
different level of granularity. There is the :class:`~polars.sql.SQLContext` object,
a top-level :func:`polars.sql` function that operates on the global context,
frame-level :meth:`DataFrame.sql` and :meth:`LazyFrame.sql` methods, and the
:func:`polars.sql_expr` function that creates native expressions from SQL.


Querying
--------

SQL queries can be issued against compatible data structures in the current globals,
against specific frames, or incorporated into expressions.


.. _global_sql:

Global SQL
~~~~~~~~~~

Both :class:`~polars.sql.SQLContext` and the :func:`polars.sql` function can be used
to execute SQL queries mediated by the Polars execution engine against Polars
:ref:`DataFrame <dataframe>`, :ref:`LazyFrame <lazyframe>`, and :ref:`Series <series>`
data, as well as `Pandas <https://pandas.pydata.org/>`_ DataFrame and Series, and
`PyArrow <https://arrow.apache.org/docs/python/>`_ Table and RecordBatch objects.
Non-Polars objects are implicitly converted to DataFrame when used in a SQL query; for
PyArrow, and Pandas data that uses PyArrow dtypes, this conversion can often be
zero-copy if the underlying data maps cleanly to a natively-supported dtype.

**Example:**

.. code-block:: python

    import polars as pl
    import pandas as pd

    polars_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    pandas_df = pd.DataFrame({"a": [3, 4, 5, 6], "b": [6, 7, 8, 9]})
    polars_series = (polars_df["a"] * 2).rename("c")
    pyarrow_table = polars_df.to_arrow()

    pl.sql(
        """
        SELECT a, b, SUM(c) AS c_total FROM (
          SELECT * FROM polars_df                  -- polars frame
            UNION ALL SELECT * FROM pandas_df      -- pandas frame
            UNION ALL SELECT * FROM pyarrow_table  -- pyarrow table
        ) all_data
        INNER JOIN polars_series
          ON polars_series.c = all_data.b          -- polars series
        GROUP BY "a", "b"
        ORDER BY "a", "b"
        """
    ).collect()

    # shape: (3, 3)
    # ┌─────┬─────┬─────────┐
    # │ a   ┆ b   ┆ c_total │
    # │ --- ┆ --- ┆ ---     │
    # │ i64 ┆ i64 ┆ i64     │
    # ╞═════╪═════╪═════════╡
    # │ 1   ┆ 4   ┆ 8       │
    # │ 3   ┆ 6   ┆ 18      │
    # │ 5   ┆ 8   ┆ 8       │
    # └─────┴─────┴─────────┘

.. topic:: Documentation

  * :meth:`polars.sql`

.. seealso::

  :ref:`SQLContext <sql_context>`


.. _frame_sql:

Frame SQL
~~~~~~~~~

Executes SQL directly against the specific underlying eager/lazy frame, referencing
it as "self"; returns a new frame representing the query result.

**Example:**

.. code-block:: python

    import polars as pl

    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    df.sql("""
      SELECT a::uint4, (b * b) AS bb
      FROM self WHERE a != 2
    """)

    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ a   ┆ bb  │
    # │ --- ┆ --- │
    # │ u32 ┆ i64 │
    # ╞═════╪═════╡
    # │ 1   ┆ 16  │
    # │ 3   ┆ 36  │
    # └─────┴─────┘

.. topic:: Documentation

  * :meth:`DataFrame.sql`
  * :meth:`LazyFrame.sql`


.. _expression_sql:

Expression SQL
~~~~~~~~~~~~~~

The :func:`polars.sql_expr` function can be used to create native Polars expressions
from SQL fragments.

**Example:**

.. code-block:: python

    import polars as pl

    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    df.with_columns(
        pl.sql_expr("(a * a) + (b::float / 2) AS expr1"),
        pl.sql_expr("CONCAT_WS(':',a,b) AS expr2")
    )

    # shape: (3, 4)
    # ┌─────┬─────┬───────┬───────┐
    # │ a   ┆ b   ┆ expr1 ┆ expr2 │
    # │ --- ┆ --- ┆ ---   ┆ ---   │
    # │ i64 ┆ i64 ┆ f64   ┆ str   │
    # ╞═════╪═════╪═══════╪═══════╡
    # │ 1   ┆ 4   ┆ 3.0   ┆ 1:4   │
    # │ 2   ┆ 5   ┆ 6.5   ┆ 2:5   │
    # │ 3   ┆ 6   ┆ 12.0  ┆ 3:6   │
    # └─────┴─────┴───────┴───────┘

.. topic:: Documentation

  * :meth:`polars.sql_expr`


.. _sql_context:

SQLContext
~~~~~~~~~~

Polars provides a dedicated class for querying frame data that offers additional
control over table registration and management of state, and can also be used as
a context manager. This is the :class:`SQLContext` object, and it provides all of
the core functionality used by the other SQL functions.


.. py:class:: SQLContext
    :canonical: polars.sql.SQLContext

    Run SQL queries against DataFrame/LazyFrame data.

    .. automethod:: __init__

    **Note:** can also be used as a context manager.

    .. automethod:: __enter__
    .. automethod:: __exit__

Methods
^^^^^^^

.. autosummary::
   :toctree: api/

    SQLContext.execute
    SQLContext.execute_global
    SQLContext.register
    SQLContext.register_globals
    SQLContext.register_many
    SQLContext.tables
    SQLContext.unregister


**Example:**

.. code-block:: python

    import polars as pl

    df1 = pl.DataFrame({"id": [1, 2, 3], "value": [0.1, 0.2, 0.3]})
    df2 = pl.DataFrame({"id": [3, 2, 1], "value": [25.6, 53.4, 12.7]})

    with pl.SQLContext(df_a=df1, df_b=df2, eager=True) as ctx:
        df = ctx.execute("""
          SELECT
            a.id,
            a.value AS value_a,
            b.value AS value_b
          FROM df_a AS a INNER JOIN df_b AS b USING (id)
          ORDER BY id
        """)

        # shape: (3, 3)
        # ┌─────┬─────────┬─────────┐
        # │ id  ┆ value_a ┆ value_b │
        # │ --- ┆ ---     ┆ ---     │
        # │ i64 ┆ f64     ┆ f64     │
        # ╞═════╪═════════╪═════════╡
        # │ 1   ┆ 0.1     ┆ 25.6    │
        # │ 2   ┆ 0.2     ┆ 53.4    │
        # │ 3   ┆ 0.3     ┆ 12.7    │
        # └─────┴─────────┴─────────┘

.. seealso::

  :ref:`pl.sql <global_sql>`
