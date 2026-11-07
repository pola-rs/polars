Table Operations
================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`ALTER TABLE <alter_table>`
     - Change the columns or properties of a table in an attached catalog.
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
   * - :ref:`INSERT <insert>`
     - Append rows to a table in an attached catalog, or replace its rows.
   * - :ref:`MERGE <merge>`
     - Update, delete and insert rows of a table in an attached catalog from a source.
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
   * - :ref:`UPDATE <update>`
     - Change rows of a table in an attached catalog.


.. _alter_table:

ALTER TABLE
-----------
Change a table of a catalog attached with :meth:`SQLContext.register_catalog`, with the
name resolved as in a query. Each statement makes one change, in one commit:

* ``ADD [COLUMN] [IF NOT EXISTS] <name> <type>`` adds a nullable column (``NOT NULL`` is not
  supported, as the existing rows have no value for it);
* ``DROP [COLUMN] [IF EXISTS] <name>`` drops a column;
* ``RENAME COLUMN <name> TO <new_name>`` renames a column, keeping its data;
* ``ALTER COLUMN <name> [SET DATA] TYPE <type>`` changes a column's type, as the catalog
  allows (such as ``INT`` to ``BIGINT``, ``FLOAT`` to ``DOUBLE``, or a wider ``DECIMAL``);
* ``ALTER COLUMN <name> DROP NOT NULL`` makes a column nullable;
* ``SET TBLPROPERTIES ('key' = value, ...)`` sets table properties.

Column names are case-sensitive and name top-level columns. ``ALTER TABLE IF EXISTS`` does
nothing if the table does not exist.

**Example:**

.. code-block:: sql

    ALTER TABLE lake.sales.orders ADD COLUMN discount DOUBLE
    ALTER TABLE lake.sales.orders RENAME COLUMN qty TO quantity
    ALTER TABLE lake.sales.orders ALTER COLUMN quantity SET DATA TYPE BIGINT
    ALTER TABLE lake.sales.orders SET TBLPROPERTIES ('owner' = 'finance')

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

On a table of an attached catalog, it deletes the matching rows in one commit. It returns
the statement, not the remaining rows. Rows where the condition is NULL are kept. The
condition compares columns with constants (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``,
``IN``, ``BETWEEN``, ``IS [NOT] NULL``, ``LIKE 'prefix%'``), combined with ``AND``, ``OR``
and ``NOT``. On floating-point columns, ``>``, ``>=`` and ``<>`` are not supported. String
constants may only contain letters, digits, spaces, ``-`` and ``_`` for now. Decimal
columns cannot be compared with constants yet. Data
files whose rows all match are dropped, and the others are rewritten (copy-on-write), also
on a merge-on-read table.

If another writer commits first, ``DELETE`` retries when that writer's rows cannot match the
condition. Otherwise it fails, unless the table's ``write.delete.isolation-level`` is
``snapshot``. The other writer's matching rows are then kept.

**Example:**

.. code-block:: sql

    DELETE FROM some_table WHERE value < 0
    DELETE FROM lake.sales.orders WHERE order_date < DATE '2020-01-01'

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
files. Some catalogs drop tables only with ``PURGE``.

.. code-block:: sql

    DROP TABLE lake.sales.orders_old PURGE

.. _explain:

EXPLAIN
-------
Returns the Polars execution plan for a given SQL query.

**Example:**

.. code-block:: sql

    EXPLAIN SELECT * FROM some_table

.. _insert:

INSERT
------
Write rows to a table of a catalog attached with :meth:`SQLContext.register_catalog`, in
one commit. ``INSERT INTO`` appends the rows of a ``VALUES`` list or a query.
``INSERT OVERWRITE`` replaces all rows of the table, in every partition. A ``WHERE`` in its
query selects the new rows, not the rows replaced.

Values fill the columns in order, or the listed columns. Other columns get NULL. A
``NOT NULL`` column needs a value. Numbers convert to any numeric type. A fraction going
into an integer column rounds: a decimal half away from zero, a float half to even.
``CAST`` truncates instead. Dates and timestamps convert among themselves. Other
conversions need a ``CAST``. A value that does not fit raises an error.

If another writer commits first, ``INSERT INTO`` retries. ``INSERT OVERWRITE`` fails if
that writer added rows, unless the table's ``write.delete.isolation-level`` is
``snapshot``.

**Example:**

.. code-block:: sql

    INSERT INTO lake.sales.orders VALUES (1, 10), (2, 20)
    INSERT INTO lake.sales.orders (quantity, id) SELECT qty, order_id FROM staged
    INSERT OVERWRITE lake.sales.orders SELECT * FROM staged

.. _merge:

MERGE
-----
Update, delete and insert rows of a table of a catalog attached with
:meth:`SQLContext.register_catalog` from a source, in one commit. ``ON`` pairs target rows
with source rows. Each target row with a matching source row takes the first
``WHEN MATCHED`` clause whose condition holds: ``UPDATE SET``, ``DELETE`` or
``DO NOTHING``. Each source row that matches no target row takes the first
``WHEN NOT MATCHED`` clause whose condition holds: ``INSERT`` or ``DO NOTHING``. Those
clauses see the source columns only. A row with no applicable clause is left as it is. It
returns the statement.

The source can be a catalog table, the target itself, a registered frame, a subquery
(with an alias), ``VALUES`` with column names or a table function. It is read once, so
every clause sees the same source rows. The source columns the statement uses are held in
memory while it runs.

``ON`` must contain a key: ``target_column = source_expression``, alone or joined with
``AND``. Other conditions of ``ON`` and the ``WHEN`` conditions may use both sides. A
NULL key matches nothing. A target row may be updated or deleted by one source row at
most, or the statement fails.

Values convert to the column types as in :ref:`INSERT <insert>`, and only the rows taking
an action must fit. ``INSERT`` without a column list takes a value for every column.
``NOT MATCHED BY SOURCE``, ``UPDATE SET *``, ``INSERT *``, ``RETURNING``, ``DEFAULT``,
subqueries in conditions or values, ``WITH`` before ``MERGE`` and ``ON`` conditions without
a key are not supported yet. Column names are case-sensitive. Registered tables cannot be
merged into.

The data files holding a target row to update or delete are read in full and replaced,
and the inserted rows are added, in one snapshot (copy-on-write). The source's keys select
the files that are read. A merge that changes nothing commits nothing. Iceberg v3 tables,
and files whose delete files cannot be read with them (such as equality deletes), are not
supported yet.

If another writer commits first, ``MERGE`` retries when that writer's rows cannot match
the source's keys. Otherwise it fails, unless the table's ``write.update.isolation-level``
is ``snapshot``. It also fails if that writer deleted or replaced rows of the files being
rewritten, or changed the table's schema or partition spec.

**Example:**

.. code-block:: sql

    MERGE INTO lake.sales.orders AS t
    USING updates AS s
    ON t.id = s.id
    WHEN MATCHED AND s.cancelled THEN DELETE
    WHEN MATCHED THEN UPDATE SET status = s.status
    WHEN NOT MATCHED THEN INSERT (id, status) VALUES (s.id, s.status)

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

On a catalog table, it commits once and writes no data files. Like ``INSERT OVERWRITE``,
it fails if another writer added rows.

**Example:**

.. code-block:: sql

    TRUNCATE TABLE actual_table
    TRUNCATE TABLE IF EXISTS possible_table

.. _update:

UPDATE
------
Change rows of a table of a catalog attached with :meth:`SQLContext.register_catalog`, in
one commit. ``SET`` gives each column a new value, computed from the row as it was before
the update. ``SET (a, b) = (x, y)`` sets several columns at once. Without ``WHERE``, every
row is updated. Rows where the condition is NULL are not updated. The condition has the
same limits as in :ref:`DELETE <delete_from_table>`. It returns the statement.

Values convert to the column types as in :ref:`INSERT <insert>`, and only the values of
updated rows must fit. A value comes from its own row, so aggregate, window and
set-returning functions are not supported. Subqueries, ``FROM``, ``RETURNING`` and
``DEFAULT`` are not supported either. Column names are case-sensitive. Registered tables
cannot be updated.

The data files holding a matching row are read in full and replaced in one snapshot
(copy-on-write). An update that matches no row commits nothing. Iceberg v3 tables, and
files whose delete files cannot be read with them (such as equality deletes), are not
supported yet.

If another writer commits first, ``UPDATE`` retries when that writer's rows cannot match the
condition. Otherwise it fails, unless the table's ``write.update.isolation-level`` is
``snapshot``. The other writer's matching rows then keep their values. It also fails if that
writer deleted or replaced rows of the files being rewritten, or changed the table's schema
or partition spec.

**Example:**

.. code-block:: sql

    UPDATE lake.sales.orders SET status = 'shipped' WHERE id = 42
    UPDATE lake.sales.orders SET (qty, amount) = (qty + 1, amount * 2) WHERE region = 'EU'
