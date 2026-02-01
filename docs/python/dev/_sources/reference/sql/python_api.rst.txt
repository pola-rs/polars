==========
Python API
==========

SQLContext
----------
.. currentmodule:: polars

Polars provides a SQL interface to query frame data; this is available through
the :class:`SQLContext` object, detailed below, as well as the DataFrame
:meth:`~polars.DataFrame.sql` and LazyFrame :meth:`~polars.LazyFrame.sql`
methods (which make use of SQLContext internally).

.. py:class:: SQLContext
    :canonical: polars.sql.SQLContext

    Run SQL queries against DataFrame/LazyFrame data.

    .. automethod:: __init__

    **Note:** can be used as a context manager.

    .. automethod:: __enter__
    .. automethod:: __exit__

Methods
~~~~~~~

.. autosummary::
   :toctree: api/

    SQLContext.execute
    SQLContext.register
    SQLContext.register_globals
    SQLContext.register_many
    SQLContext.unregister
    SQLContext.tables
