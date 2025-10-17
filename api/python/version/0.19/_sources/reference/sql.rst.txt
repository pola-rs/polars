===
SQL
===
.. currentmodule:: polars

.. py:class:: SQLContext
    :canonical: polars.sql.SQLContext

    Run SQL queries against DataFrame/LazyFrame data.

    .. automethod:: __init__

    Note: can be used as a context manager.

    .. automethod:: __enter__
    .. automethod:: __exit__

Methods
-------

.. autosummary::
   :toctree: api/

    SQLContext.execute
    SQLContext.register
    SQLContext.register_globals
    SQLContext.register_many
    SQLContext.unregister
    SQLContext.tables
