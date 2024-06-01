Table Operations
================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`CREATE TABLE <create_table>`
     - Create a new table and its columns from a SQL query against an existing table.
   * - :ref:`DROP TABLES <drop_tables>`
     - Delete a specified table and related data.
   * - :ref:`EXPLAIN <explain>`
     - Returns logical plan of the query.
   * - :ref:`SHOW TABLES <show_tables>`
     - Returns a list of all tables in the context.
   * - :ref:`UNNEST <unnest_table_func>`
     - Unnest one or more arrays as columns in a new table.
   * - :ref:`TRUNCATE <truncate>`
     - Remove rows from table without deleting the table from context.

.. _create_table:

CREATE TABLE
------------
Create a new table and its columns from a SQL query against an existing table.

**Example:**

.. code-block:: sql

    CREATE TABLE new_table AS SELECT * FROM df WHERE value > 42

.. _drop_tables:

DROP TABLES
-----------
Delete a specified table and related data.

**Example:**

.. code-block:: sql

    DROP TABLE old_table

.. _explain:

EXPLAIN
-------
Returns Logical Plan of the query.

**Example:**

.. code-block:: sql

    EXPLAIN SELECT * FROM df

.. _show_tables:

SHOW TABLES
-----------
Display the list of tables in the context.

**Example:**

.. code-block:: sql

    SHOW TABLES

.. _unnest_table_func:

UNNEST
------
Unnest one or more arrays as columns in a new table.

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
Removes all rows from the specified table, but keeps the table.

**Example:**

.. code-block:: sql

    TRUNCATE TABLE df
