Table Operations
================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`ALTER TABLE <alter_table>`
     - Change the columns, properties or partitioning of a table in an attached catalog.
   * - :ref:`CALL <call>`
     - Run a maintenance procedure on a table in an attached catalog.
   * - :ref:`COPY <copy>`
     - Write the result of a query, or a table, to a file.
   * - :ref:`CREATE MACRO <create_macro>`
     - Create a named expression or query with parameters.
   * - :ref:`CREATE SCHEMA <create_schema>`
     - Create a namespace in an attached catalog.
   * - :ref:`CREATE TABLE <create_table>`
     - Create a new table and its columns from a SQL query executed against an existing table.
   * - :ref:`CREATE VIEW <create_view>`
     - Create a named query that is run each time it is read.
   * - :ref:`DELETE FROM <delete_from_table>`
     - Remove specific rows of data from a table using an (optional) constraint.
   * - :ref:`DESCRIBE <describe>`
     - Returns the column names and types of a table, view, file or query.
   * - :ref:`DROP MACRO <drop_macro>`
     - Deletes the specified macro.
   * - :ref:`DROP SCHEMA <drop_schema>`
     - Drop an empty namespace of an attached catalog.
   * - :ref:`DROP TABLES <drop_tables>`
     - Deletes the specified table, unregistering it.
   * - :ref:`DROP VIEW <drop_view>`
     - Deletes the specified view.
   * - :ref:`EXPLAIN <explain>`
     - Returns the Polars execution plan for a given SQL query.
   * - :ref:`INSERT <insert>`
     - Append rows to a table, or replace its rows.
   * - :ref:`MERGE <merge>`
     - Update, delete and insert rows of a table in an attached catalog from a source.
   * - :ref:`SHOW TABLES <show_tables>`
     - Returns a list of all tables and views registered in the given context.
   * - :ref:`INFORMATION_SCHEMA.COLUMNS <information_schema_columns>`
     - A table with a row per column of each table and view registered in the given context.
   * - :ref:`GENERATE_SERIES <generate_series_table_func>`
     - Create a table with one column of values from start to stop, stop included.
   * - :ref:`RANGE <range_table_func>`
     - Create a table with one column of values from start to stop, stop not included.
   * - :ref:`UNNEST <unnest_table_func>`
     - Unnest one or more arrays as columns in a new table object.
   * - :ref:`TRUNCATE <truncate>`
     - Remove all data from a table without actually deleting it.
   * - :ref:`UPDATE <update>`
     - Change rows of a table.


.. _alter_table:

ALTER TABLE
-----------
Change a table of a catalog attached with :meth:`SQLContext.register_catalog`, with the
name resolved as in a query. Each statement makes one change, in one commit:

* ``ADD [COLUMN] [IF NOT EXISTS] <name> <type>`` adds a nullable column (``NOT NULL`` is not
  supported, as the existing rows have no value for it);
* ``DROP [COLUMN] [IF EXISTS] <name>`` drops a column (not one that a partition spec of
  the table partitions by, also an older one, nor one that the table's sort order sorts by);
* ``RENAME COLUMN <name> TO <new_name>`` renames a column, keeping its data;
* ``ALTER COLUMN <name> [SET DATA] TYPE <type>`` changes a column's type, as the catalog
  allows (such as ``INT`` to ``BIGINT``, ``FLOAT`` to ``DOUBLE``, or a wider ``DECIMAL``);
* ``ALTER COLUMN <name> DROP NOT NULL`` makes a column nullable;
* ``SET TBLPROPERTIES ('key' = value, ...)`` sets table properties;
* ``SET PARTITIONED BY (<field>, ...)`` sets how new data is partitioned, with the fields of
  ``CREATE TABLE ... PARTITION BY``, in the order written;
* ``RESET PARTITIONED BY`` stops partitioning new data;
* ``SET SORTED BY (<field> [ASC | DESC] [NULLS FIRST | LAST], ...)`` sets how new data is
  sorted, with fields as in ``PARTITION BY``, in the order written (``truncate`` also takes
  decimals);
* ``RESET SORTED BY`` stops sorting new data;
* ``CREATE TAG [IF NOT EXISTS] <name> [AS OF VERSION <snapshot_id>] [RETAIN <n> <unit>]``
  names a snapshot, by default the current one;
* ``CREATE BRANCH [IF NOT EXISTS] <name> [AS OF VERSION <snapshot_id>] [RETAIN <n> <unit>]
  [WITH SNAPSHOT RETENTION <retention>]`` starts a branch at a snapshot, by default the
  current one;
* ``[CREATE OR] REPLACE TAG | BRANCH <name> ...``, with the options of ``CREATE``, points a
  tag or branch at a snapshot, by default also the current one, and can change its
  retention;
* ``DROP TAG | BRANCH [IF EXISTS] <name>`` drops a tag or branch.

Changing the partitioning only changes the table's metadata. Existing data files keep the
partitioning they were written with, and later writes use the new one. A field that the
table had before keeps its name, and an older spec with the same fields is used again. A
statement adds at most one time transform (``year``, ``month``, ``day`` or ``hour``) per
column. Time fields the table has can stay.
Iceberg v1 tables are not supported: upgrade them to v2 first.

Changing the sort order also only changes the table's metadata. Later SQL writes (also
``DELETE``, ``UPDATE`` and ``MERGE``, which rewrite data files) and ``sink_iceberg`` sort
their data by the new order. A field without ``NULLS FIRST`` or ``NULLS LAST`` takes the null order of
``ORDER BY``: nulls last for ``ASC``, first for ``DESC``. Engines differ in this default, so
state the null order in statements that other engines also run. An older sort order with the
same fields is used again.

A tag or branch is read with ``FOR VERSION AS OF '<name>'`` (see :ref:`time travel
<time_travel>`), and the ``$refs`` table lists them with their snapshots. Writes go to the
``main`` branch: writing to another branch is not supported yet. Names are identifiers and
keep their case. A name that starts with a digit or has characters other than letters,
digits and ``_`` needs double quotes. ``CREATE`` fails if the name is taken by a tag or
branch, unless with ``IF NOT EXISTS``. ``REPLACE`` and ``DROP`` fail if the name is a ref of
the other kind. ``REPLACE`` keeps the retention the statement does not set. To change only
the retention, give the ref's own snapshot with ``AS OF VERSION``. The ``main`` branch
cannot be replaced or dropped, and no tag can take its name.
``RETAIN`` sets how long the ref is kept, in ``DAYS``, ``HOURS`` or ``MINUTES``.
``WITH SNAPSHOT RETENTION`` sets how many snapshots a branch keeps (``<n> SNAPSHOTS``), how
long it keeps them (``<n> <unit>``), or both, in that order. Both only store the setting:
snapshots are expired by the engine that does it. A table without snapshots has none to
tag or branch.

Column names are case-sensitive and name top-level columns. ``SET PARTITIONED BY`` and
``SET SORTED BY`` also take fields of struct columns (``s.f``), as
``CREATE TABLE ... PARTITION BY`` does.
``ALTER TABLE IF EXISTS`` does nothing if the table does not exist.

**Example:**

.. code-block:: sql

    ALTER TABLE lake.sales.orders ADD COLUMN discount DOUBLE
    ALTER TABLE lake.sales.orders RENAME COLUMN qty TO quantity
    ALTER TABLE lake.sales.orders ALTER COLUMN quantity SET DATA TYPE BIGINT
    ALTER TABLE lake.sales.orders SET TBLPROPERTIES ('owner' = 'finance')
    ALTER TABLE lake.sales.orders SET PARTITIONED BY (month(ordered_at), bucket(16, id))
    ALTER TABLE lake.sales.orders RESET PARTITIONED BY
    ALTER TABLE lake.sales.orders SET SORTED BY (customer_id, ordered_at DESC NULLS LAST)
    ALTER TABLE lake.sales.orders RESET SORTED BY
    ALTER TABLE lake.sales.orders CREATE TAG "end-of-quarter" RETAIN 365 DAYS
    ALTER TABLE lake.sales.orders CREATE BRANCH audit WITH SNAPSHOT RETENTION 3 SNAPSHOTS
    ALTER TABLE lake.sales.orders DROP TAG IF EXISTS "end-of-quarter"

.. _call:

CALL
----
Run a maintenance procedure on a table of a catalog attached with
:meth:`SQLContext.register_catalog`. A ``CALL`` returns a result row.

.. code-block:: sql

    CALL [<catalog>.]system.<procedure>(<arguments>)

* Without ``<catalog>``, the default catalog is used.
* Arguments are given in order, or by name as ``name => value``.
* Argument values are literals: strings, integers, booleans, ``NULL``, ``map('k', 'v')``
  or ``array('a', 'b')``.
* The ``table`` argument is a string with the table's name in that catalog: ``'ns.t'``, or
  ``'t'`` in the catalog's default namespace. It may start with the catalog's name.

**rewrite_data_files**

Compacts the data files of a table in one commit, as a ``replace`` snapshot. Small files
are combined, and large ones split, into files of about the target size. Sizes are
estimated from the files read: files that another writer compressed differently can take
a second ``CALL`` to reach the target size. The rows stay the
same: rows that delete files removed are not written again. The new files are written
under the table's current partition spec and sorted by its sort order. Delete files stay.
Files are selected as Iceberg's ``binpack`` strategy does: per partition, a group of files is
rewritten when it has enough small or large files, or files with many deleted rows.

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Argument
     - Description
   * - ``table``
     - The table. Required.
   * - ``strategy``
     - ``'binpack'``. Default: ``'binpack'``. ``'sort'`` is not supported yet.
   * - ``sort_order``
     - Not supported yet.
   * - ``options``
     - A ``map('key', 'value', ...)`` of these options:

       * ``rewrite-all``: ``'true'`` rewrites every selected file. Default: ``'false'``.
       * ``target-file-size-bytes``: the size of the new files. Default: the table's
         ``write.target-file-size-bytes``, else 512 MB.
       * ``min-file-size-bytes``, ``max-file-size-bytes``: files outside this range are
         rewritten. Default: 75% and 180% of the target size.
       * ``min-input-files``: the number of such files in a group that is enough for a
         rewrite. Default: 5.
       * ``max-file-group-size-bytes``, ``max-file-group-input-files``: the largest group
         rewritten together. Default: 100 GB, no limit.
       * ``delete-file-threshold``: files with at least this many delete files are
         rewritten. Default: no limit.
       * ``delete-ratio-threshold``: files with at least this share of deleted rows are
         rewritten. Default: 0.3.
   * - ``where``
     - A condition as a string. Only the files that may hold a matching row are
       considered, and they are rewritten whole. ``where`` is a reserved word: give it in
       position, or as ``"where" => '...'``.

It returns ``rewritten_data_files_count``, ``added_data_files_count``,
``rewritten_bytes_count``, ``failed_data_files_count`` and ``removed_delete_files_count``.
The last two are always 0. If no group needs a rewrite, nothing is committed and the counts
are 0. Concurrent appends do not conflict with a compaction. A concurrent change to a
rewritten file, such as a delete, makes it fail. Iceberg v3 tables, tables with equality
delete files, and position delete files that are not of one data file are not supported
yet.

**Example:**

.. code-block:: sql

    -- Compact every data file of table `orders` in namespace `sales` of catalog `lake`,
    -- partition by partition, into files of about 128 MB.
    CALL lake.system.rewrite_data_files(
        table => 'sales.orders',
        options => map('rewrite-all', 'true', 'target-file-size-bytes', '134217728')
    )

    -- Only the files that may hold rows of 2024, by position: the strategy, no sort
    -- order, no options, then the condition.
    CALL lake.system.rewrite_data_files(
        'sales.orders', 'binpack', NULL, NULL, 'ordered_at >= ''2024-01-01'''
    )

**rewrite_manifests**

Rewrites the data manifests of a table in one commit, as a ``replace`` snapshot. The rows
and data files stay the same. The new manifests are sorted by partition and about
``commit.manifest.target-size-bytes`` in size (default 8 MB), so queries skip more of them.
Delete manifests are not changed.

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Argument
     - Description
   * - ``table``
     - The table. Required.
   * - ``use_caching``
     - Accepted. Has no effect.
   * - ``spec_id``
     - The partition spec whose manifests are rewritten. Default: the current spec.
   * - ``sort_by``
     - Not supported yet.

It returns ``rewritten_manifests_count`` and ``added_manifests_count``. If there is nothing
to rewrite (no data manifests, or one small enough), nothing is committed and both counts
are 0. Iceberg v3 tables are not supported yet.

**Example:**

.. code-block:: sql

    -- Table `orders` in namespace `sales` of catalog `lake`: rewrite the data manifests of
    -- its current partition spec into fewer manifests, sorted by partition. Rows and data
    -- files stay the same.
    CALL lake.system.rewrite_manifests(table => 'sales.orders')

    -- The same table, given by position. Only the manifests of partition spec 0 are
    -- rewritten, e.g. those written before the table's partitioning changed. Other specs'
    -- manifests stay as they are.
    CALL lake.system.rewrite_manifests('sales.orders', spec_id => 0)

.. _copy:

COPY
----
Write the result of a query, or a table, to a file. :meth:`SQLContext.execute` returns a
``LazyFrame`` that writes the file once it is collected; collecting it gives an empty frame.

The format is the one that ``FORMAT`` names, or else the one of the file extension: ``csv``;
``parquet``; ``ipc``, ``arrow`` or ``feather``; ``ndjson`` or ``jsonl``. A CSV file takes the
options ``HEADER`` (written unless set to ``false``), ``DELIMITER``, ``NULL`` and ``QUOTE``.
The path can be a cloud URL, and a leading ``~`` is the home directory.

The directory of the file must exist, and an existing file is replaced. If the query fails
while it runs, a partial file can be left behind, also in the place of a file that existed.

Only ``COPY ... TO '<file>'`` is supported. To load a file, read it in a query, as in
``CREATE TABLE t AS SELECT * FROM 'data.csv'``. Writing files can be turned off with
``SQLContext(allow_file_writes=False)``.

**Example:**

.. code-block:: sql

    COPY (SELECT region, SUM(amount) AS total FROM sales GROUP BY region) TO 'totals.parquet'
    COPY sales (region, amount) TO 'sales.csv'
    COPY sales TO 'sales.txt' (FORMAT csv, DELIMITER '|', HEADER false)

.. _create_macro:

CREATE MACRO
------------
Create a macro: a named expression with parameters (a scalar macro), or a named query with
parameters (a table macro, with ``AS TABLE``). A call is replaced by the body of the macro,
with the parameters replaced by the arguments. ``CREATE FUNCTION`` is the same statement.

- A parameter can have a default (``b := 5``); a call can give arguments by name (``b := 1``).
- A macro can have several definitions, for different numbers of arguments.
- A scalar macro is called like a function; a table macro is called in ``FROM``.
- Outside subqueries, the body of a scalar macro can only use its parameters.
- An argument that reads columns cannot be used in a subquery of the macro body.
- ``OR REPLACE`` replaces an existing macro. ``TEMP``/``TEMPORARY`` is accepted; every
  macro lives in its context. A view uses the macros as they are when it is read.

**Example:**

.. code-block:: sql

    CREATE MACRO add_tax(amount, rate := 0.21) AS amount * (1 + rate)
    CREATE MACRO clamp(a) AS greatest(a, 0),
        (a, hi) AS least(greatest(a, 0), hi)
    CREATE MACRO orders_since(d) AS TABLE SELECT * FROM orders WHERE order_date >= d

    SELECT add_tax(amount), add_tax(amount, rate := 0.09) FROM orders
    SELECT * FROM orders_since('2024-01-01')

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
  ``truncate(W, column)``. A column can also be a field of a struct column: ``s.f`` or
  ``"s"."f"`` is field ``f`` of struct ``s``, never a table-qualified column, and
  ``"s.f"`` is a column named ``s.f``. ``DELETE`` and ``UPDATE`` conditions can use such a
  field only when its struct column is ``NOT NULL``, so not one level deeper, and not in a
  table created with ``AS SELECT``. ``PARTITIONED BY (column, ...)`` takes top-level column
  names only.
- ``LOCATION`` (or the ``'location'`` option) sets the table location; the other
  ``WITH`` options become table properties.
- ``AS SELECT`` commits the rows together with the create, so a failed query creates
  no table. A catalog that cannot stage a create gets an empty table first, dropped
  again if filling it fails, unless another writer changed it. The columns are never required. The rows are written as by
  :meth:`LazyFrame.sink_iceberg`: write properties such as
  ``write.parquet.compression-codec`` are stored, but not applied to them.
- ``IF NOT EXISTS`` leaves an existing table as it is; its query is checked, but not run.
- ``LIKE`` and ``OR REPLACE`` are not supported for catalog tables.

.. _create_view:

CREATE VIEW
-----------
Create a view: a named query that is planned again each time a statement reads it, so it
reads the tables as they are then. Use ``CREATE TABLE ... AS`` to keep the result of a
query as it is now.

- ``OR REPLACE`` replaces an existing view; ``IF NOT EXISTS`` leaves it as it is.
- A list of column names after the view name renames the columns of the query.
- ``TEMP``/``TEMPORARY`` is accepted; every view lives in its context.
- Tables and views share names, a view cannot read itself, and a view cannot read the
  tables of an attached catalog.

**Example:**

.. code-block:: sql

    CREATE VIEW recent_orders AS SELECT * FROM orders WHERE order_date >= '2024-01-01'
    CREATE OR REPLACE VIEW order_totals (customer, total) AS
      SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id

.. _delete_from_table:

DELETE
------
Remove specific rows from a table using an (optional) constraint.
Omitting the constraint deletes *all* rows, equivalent to `TRUNCATE`.

On a registered table, it returns the remaining rows, and later statements in the same
context see the table without the deleted rows. Source files and the original frame are
not changed.

The table keeps the condition, not its result. Each read of the table runs the condition
again on the table's source. A subquery of the condition keeps its tables as registered at
the time of the ``DELETE``.

- A condition that names a missing column fails at the ``DELETE`` and leaves the table
  unchanged. Other errors in the condition, such as a string compared with an integer or a
  cast of a string that is not a number, show when the table is read, on each read until
  the table is registered again.
- Collecting the result of a ``DELETE`` computes the remaining rows. Leave the result lazy
  when only the change is needed.
- Each ``DELETE`` adds its condition to the table. After many, register the collected rows
  again: ``ctx.register("t", ctx.execute("SELECT * FROM t", eager=True))``.

On a table of an attached catalog, it deletes the matching rows in one commit, as one
snapshot. It returns the statement, not the remaining rows. Rows where the condition is NULL
are kept.

The condition compares columns with constants (``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``,
``IN``, ``BETWEEN``, ``IS [NOT] NULL``, ``LIKE 'prefix%'``), combined with ``AND``, ``OR``
and ``NOT``. On floating-point columns, ``>``, ``>=`` and ``<>`` are not supported. String
constants may only contain letters, digits, spaces, ``-`` and ``_`` for now. Decimal
columns cannot be compared with constants yet.

Data files whose rows all match are dropped. If other files match only partly, the files
that may match are read, and the partly matching ones are rewritten without the deleted
rows (copy-on-write), also on a merge-on-read table. Not supported yet:

- Iceberg v3 tables.
- Reading a data file that is not Parquet.
- Reading a data file whose delete files cannot be applied, such as equality deletes or
  position deletes that cover several data files.

If another writer commits first:

- ``DELETE`` retries when that writer's rows cannot match the condition.
- Otherwise it fails, unless the table's ``write.delete.isolation-level`` is ``snapshot``.
  The other writer's matching rows are then kept.
- A ``DELETE`` that reads files fails at any isolation level if that writer deleted or
  replaced rows of those files, or changed the table's schema or partition spec.

**Example:**

.. code-block:: sql

    DELETE FROM some_table WHERE value < 0
    DELETE FROM lake.sales.orders WHERE order_date < DATE '2020-01-01'

.. _describe:

DESCRIBE
--------
Returns the column names and data types of a table, a view, a file or the result of a query,
as the columns ``column_name`` and ``column_type``. ``DESC`` is the same, and ``DESCRIBE TABLE t``
is ``DESCRIBE t``. A query is not run.
A file is given by its path in single quotes, as in :ref:`FROM <from>`; only what its columns
are known from is read, such as the metadata of a Parquet file or the first rows of a CSV file.

A type is given by the SQL name that :ref:`CAST <cast>` reads back as that type, with a nested
type in full, such as ``STRUCT<lat DOUBLE PRECISION, lon DOUBLE PRECISION>``. A type that has
no SQL name, such as a fixed-size array, is given by its Polars name in lower case, also inside
a nested type; such a name cannot be used in ``CAST``.

**Example:**

.. code-block:: sql

    DESCRIBE some_table
    DESCRIBE 'data/events.parquet'
    DESCRIBE SELECT a, b * 2 AS c FROM some_table

.. _drop_macro:

DROP MACRO
----------
Deletes the specified scalar macro, or table macro with ``DROP MACRO TABLE``. ``DROP FUNCTION``
deletes a scalar macro. Dropping a macro that does not exist raises an error unless the
optional ``IF EXISTS`` clause is given.

**Example:**

.. code-block:: sql

    DROP MACRO add_tax
    DROP MACRO TABLE IF EXISTS orders_since

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

.. _drop_view:

DROP VIEW
---------
Deletes the specified views. Dropping a view that does not exist raises an error unless
the optional ``IF EXISTS`` clause is given.

**Example:**

.. code-block:: sql

    DROP VIEW recent_orders
    DROP VIEW IF EXISTS order_totals

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
Write rows to a table. ``INSERT INTO`` appends the rows of a ``VALUES`` list or a query.
``INSERT OVERWRITE`` replaces all rows of the table, in every partition. A ``WHERE`` in its
query selects the new rows, not the rows replaced.

Values fill the columns in order, or the listed columns. Other columns get NULL. A
``NOT NULL`` column of a catalog table needs a value. Numbers convert to any numeric type.
A fraction going into an integer column rounds: a decimal half away from zero, a float half
to even. ``CAST`` truncates instead. Dates and timestamps convert among themselves. Other
conversions need a ``CAST``. A value that does not fit raises an error.

On a registered table, it returns the table with the new rows, and later statements in the
same context see them, as after :ref:`DELETE <delete_from_table>`. Source files and the
original frame are not changed. The rows of a ``VALUES`` list without subqueries are computed
at the ``INSERT``. The rows of a query are not: the table keeps the query, and each read of
the table runs it again, on the tables as they were at the time of the ``INSERT``. A value of
the query that does not fit shows when the table is read, on each read until the table is
registered again.

On a table of a catalog attached with :meth:`SQLContext.register_catalog`, it writes the
rows in one commit. If another writer commits first, ``INSERT INTO`` retries.
``INSERT OVERWRITE`` fails if that writer added rows, unless the table's
``write.delete.isolation-level`` is ``snapshot``.

**Example:**

.. code-block:: sql

    INSERT INTO orders VALUES (1, 10), (2, 20)
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
Returns a list of all tables and views registered in the given context.

**Example:**

.. code-block:: sql

    SHOW TABLES

.. _information_schema_columns:

INFORMATION_SCHEMA.COLUMNS
--------------------------
A table with a row per column of each table and view registered in the given context, in the
order of their names. Its columns are ``table_name``, ``column_name``, ``ordinal_position``
(starting at 1), ``data_type`` (the Polars data type, as in :ref:`DESCRIBE <describe>`) and
``is_nullable`` (always ``'YES'``). Read in a view, it does not list the views that read it.

**Example:**

.. code-block:: sql

    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'some_table'

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
Change rows of a table. ``SET`` gives each column a new value, computed from the row as it
was before the update. ``SET (a, b) = (x, y)`` sets several columns at once. Without
``WHERE``, every row is updated. Rows where the condition is NULL are not updated.

Values convert to the column types as in :ref:`INSERT <insert>`, and only the values of
updated rows must fit. A value comes from its own row, so aggregate, window and
set-returning functions are not supported. Subqueries, ``FROM``, ``RETURNING`` and
``DEFAULT`` are not supported either. Column names are case-sensitive.

On a registered table, it returns the table with the new values, and later statements in
the same context see them, as after :ref:`DELETE <delete_from_table>`. Source files and the
original frame are not changed. As with ``DELETE``, the table keeps the change, not its
result: a value that does not fit shows when the table is read, on each read until the table
is registered again.

On a table of a catalog attached with :meth:`SQLContext.register_catalog`, it updates the
rows in one commit, and returns the statement. The condition has the same limits as in
:ref:`DELETE <delete_from_table>`. The data files holding a matching row are read in full
and replaced in one snapshot (copy-on-write). An update that matches no row commits
nothing. Iceberg v3 tables, and files whose delete files cannot be read with them (such as
equality deletes), are not supported yet.

If another writer commits first, ``UPDATE`` retries when that writer's rows cannot match the
condition. Otherwise it fails, unless the table's ``write.update.isolation-level`` is
``snapshot``. The other writer's matching rows then keep their values. It also fails if that
writer deleted or replaced rows of the files being rewritten, or changed the table's schema
or partition spec.

**Example:**

.. code-block:: sql

    UPDATE orders SET status = 'shipped' WHERE id = 42
    UPDATE lake.sales.orders SET (qty, amount) = (qty + 1, amount * 2) WHERE region = 'EU'
