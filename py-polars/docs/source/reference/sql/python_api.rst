==========
Python API
==========

.. currentmodule:: polars

There are three primary entry points to the Polars SQL interface, each operating at a
different level of granularity. There is the :class:`~polars.sql.SQLContext` object,
a top-level :func:`polars.sql` function that operates on the global context, and
frame-level :meth:`DataFrame.sql` and :meth:`LazyFrame.sql` methods.

Global SQL
----------

Both :class:`~polars.sql.SQLContext` and the :func:`polars.sql` function can be used
to execute SQL queries mediated by the Polars execution engine against Polars
:ref:`DataFrame <dataframe>`, :ref:`LazyFrame <lazyframe>`, and :ref:`Series <series>`
data, as well as `Pandas <https://pandas.pydata.org/>`_ DataFrame and Series, and
`PyArrow <https://arrow.apache.org/docs/python/>`_ Table and RecordBatch objects.
Non-Polars objects are implicitly converted to DataFrame when used in a SQL query; for
PyArrow, and Pandas data that uses PyArrow dtypes, this conversion can be zero-copy if
the underlying data maps to a type supported by Polars.

**Example:**

.. code-block:: python

    import polars as pl
    import pandas as pd

    polars_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    pandas_df = pd.DataFrame({"a": [3, 4, 5, 6], "b": [6, 7, 8, 9]})
    pyarrow_table = polars_df.to_arrow()
    polars_series = (polars_df["a"] * 2).rename("c")

    pl.sql(
        """
        SELECT a, b, SUM(c) AS c_total FROM (
          SELECT * FROM polars_df                  -- polars frame
            UNION ALL SELECT * FROM pandas_df      -- pandas frame
            UNION ALL SELECT * FROM pyarrow_table  -- pyarrow table
        ) all_data
        INNER JOIN polars_series
          ON polars_series.c == all_data.b         -- join on series
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


Frame-level SQL
---------------

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


SQLContext
----------

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
~~~~~~~

.. autosummary::
   :toctree: api/

    SQLContext.execute
    SQLContext.execute_global
    SQLContext.register
    SQLContext.register_globals
    SQLContext.register_many
    SQLContext.tables
    SQLContext.unregister
