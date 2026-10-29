Table Operations
================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`CREATE SCHEMA <create_schema>`
     - Create a namespace in an attached catalog.
   * - :ref:`CREATE TABLE <create_table>`
     - Create a new table and its columns from a SQL query executed against an existing table.
   * - :ref:`DELETE FROM <delete_from_table>`
     - Remove specific rows of data from a table using an (optional) constraint.
   * - :ref:`DROP SCHEMA <drop_schema>`
     - Drop an empty namespace of an attached catalog.
   * - :ref:`DROP TABLES <drop_tables>`
     - Deletes the specified table, unregistering it.
   * - :ref:`EXPLAIN <explain>`
     - Returns the Polars execution plan for a given SQL query.
   * - :ref:`SHOW TABLES <show_tables>`
     - Returns a list of all tables registered in the given context.
   * - :ref:`GENERATE_SERIES <generate_series_table_func>`
     - Create a table with one column of values from start to stop, stop included.
   * - :ref:`RANGE <range_table_func>`
     - Create a table with one column of values from start to stop, stop not included.
   * - :ref:`UNNEST <unnest_table_func>`
     - Unnest one or more arrays as columns in a new table object.
   * - :ref:`TRUNCATE <truncate>`
     - Remove all data from a table without actually deleting it.


.. _create_schema:

CREATE SCHEMA
-------------
Create a namespace in a catalog attached with :meth:`SQLContext.register_catalog`. A name
that starts with an attached catalog is a namespace of that catalog; any other name is one
of the default catalog, not under its ``default_namespace``. An existing namespace raises an
error unless ``IF NOT EXISTS`` is given, which leaves it as it is. ``WITH (...)`` sets the
namespace's properties, such as ``'location'``, where its tables go by default; the catalog
decides which properties it keeps and uses.

**Example:**

.. code-block:: sql

    CREATE SCHEMA lake.sales
    CREATE SCHEMA IF NOT EXISTS lake.sales WITH ('location' = 's3://bucket/sales')

.. _create_table:

CREATE TABLE
------------
Create a new table from a sequence of column definitions,
a SQL query executed against an existing table, or create
an empty table from the schema of an existing table.

**Example:**

.. code-block:: sql

    CREATE TABLE tbl(colx VARCHAR, coly DATE, colz ARRAY<DOUBLE>, cols STRUCT<a INT, b VARCHAR>)

.. code-block:: sql

    CREATE TABLE new_table AS
    SELECT * FROM existing_table WHERE value > 42

.. code-block:: sql

    CREATE TABLE new_table LIKE existing_table

**Temporary tables:** ``CREATE TEMP TABLE`` (or ``TEMPORARY``) always registers the
table in the context, also when a default catalog is attached. Unlike ``CREATE TABLE``,
it raises an error if the table already exists, unless ``IF NOT EXISTS`` (leave it as it
is) or ``OR REPLACE`` (replace it) is given.

.. code-block:: sql

    CREATE OR REPLACE TEMP TABLE staging AS
    SELECT * FROM existing_table WHERE value > 42

**Catalog tables:** with an attached Iceberg catalog (see
:meth:`SQLContext.register_catalog`), a name that resolves to the catalog, as in a query,
creates the table in the catalog.

.. code-block:: sql

    CREATE TABLE lake.sales.orders (
        id BIGINT NOT NULL, region VARCHAR, ordered_at TIMESTAMP, amount DECIMAL(10, 2)
    )
    WITH ('format-version' = 2)

.. code-block:: sql

    CREATE TABLE IF NOT EXISTS lake.sales.big_orders
    LOCATION 's3://bucket/big_orders'
    PARTITION BY (region, day(ordered_at), bucket(16, id))
    AS SELECT * FROM lake.sales.orders WHERE amount > 1000

- ``NOT NULL`` makes a column a required field; other constraints are not supported.
- ``PARTITION BY (...)`` sets the partition spec: a column (identity), ``year``,
  ``month``, ``day`` or ``hour`` of a column, ``bucket(N, column)`` or
  ``truncate(W, column)``. ``PARTITIONED BY (column, ...)`` takes column names only.
- ``LOCATION`` (or the ``'location'`` option) sets the table location; the other
  ``WITH`` options become table properties.
- ``AS SELECT`` commits the rows together with the create, so a failed query creates
  no table. A catalog that cannot stage a create gets an empty table first, dropped
  again if filling it fails, unless another writer changed it. The columns are never required. The rows are written as by
  :meth:`LazyFrame.sink_iceberg`: write properties such as
  ``write.parquet.compression-codec`` are stored, but not applied to them.
- ``IF NOT EXISTS`` leaves an existing table as it is; its query is checked, but not run.
- ``LIKE`` and ``OR REPLACE`` are not supported for catalog tables.

.. _delete_from_table:

DELETE
------
Remove specific rows from a table using an (optional) constraint.
Omitting the constraint deletes *all* rows, equivalent to `TRUNCATE`.

**Example:**

.. code-block:: sql

    DELETE FROM some_table WHERE value < 0

.. _drop_schema:

DROP SCHEMA
-----------
Drop a namespace of an attached catalog, with the name resolved as in
:ref:`CREATE SCHEMA <create_schema>`. Only an empty namespace is dropped: the namespace is
checked for tables and child namespaces before the drop, and raises an error if it has any
(``CASCADE`` is not supported). Dropping a namespace
that does not exist raises an error unless ``IF EXISTS`` is given.

**Example:**

.. code-block:: sql

    DROP SCHEMA lake.sales_old
    DROP SCHEMA IF EXISTS lake.scratch

.. _drop_tables:

DROP TABLES
-----------
Deletes the specified table, unregistering it. Dropping a table that does not
exist raises an error unless the optional ``IF EXISTS`` clause is given, in
which case the statement is a no-op.

**Example:**

.. code-block:: sql

    DROP TABLE actual_table
    DROP TABLE IF EXISTS possible_table

**Catalog tables:** a name that resolves to an attached catalog, and to no registered
table, drops the table from the catalog, permanently. ``PURGE`` also deletes its data
files.

.. code-block:: sql

    DROP TABLE lake.sales.orders_old PURGE

.. _explain:

EXPLAIN
-------
Returns the Polars execution plan for a given SQL query.

**Example:**

.. code-block:: sql

    EXPLAIN SELECT * FROM some_table

.. _show_tables:

SHOW TABLES
-----------
Returns a list of all tables registered in the given context.

**Example:**

.. code-block:: sql

    SHOW TABLES

.. _generate_series_table_func:

GENERATE_SERIES
---------------
Create a table with one column of values from start to stop, stop included. Takes the
same arguments as the :ref:`GENERATE_SERIES <generate_series>` function; the arguments
must be constants. The column is named ``generate_series`` unless renamed by an alias.

**Example:**

.. code-block:: sql

    SELECT * FROM
      GENERATE_SERIES(DATE '2026-01-01', DATE '2026-12-31', INTERVAL '1 day') AS cal (d)

.. _range_table_func:

RANGE
-----
Like :ref:`GENERATE_SERIES <generate_series_table_func>`, but stop is not included. The
column is named ``range`` unless renamed by an alias.

**Example:**

.. code-block:: sql

    SELECT * FROM RANGE(0, 10, 2)

.. _unnest_table_func:

UNNEST
------
Unnest one or more arrays as columns in a new table object.

**Example:**

.. code-block:: sql

    SELECT * FROM
      UNNEST(
        [1, 2, 3, 4],
        ['ww','xx','yy','zz'],
        [23.0, 24.5, 28.0, 27.5]
      ) AS tbl (x,y,z)

.. _truncate:

TRUNCATE
--------
Remove all data from a table *without* deleting the table itself. Truncating a
table that does not exist raises an error unless the optional ``IF EXISTS``
clause is given, in which case the statement is a no-op.

**Example:**

.. code-block:: sql

    TRUNCATE TABLE actual_table
    TRUNCATE TABLE IF EXISTS possible_table
