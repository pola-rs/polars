Table Operations
================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`CREATE TABLE <create_table>`
     - Create a new table and its columns from a SQL query executed against an existing table.
   * - :ref:`DELETE FROM <delete_from_table>`
     - Remove specific rows of data from a table using an (optional) constraint.
   * - :ref:`DROP TABLES <drop_tables>`
     - Deletes the specified table, unregistering it.
   * - :ref:`EXPLAIN <explain>`
     - Returns the Polars execution plan for a given SQL query.
   * - :ref:`SHOW TABLES <show_tables>`
     - Returns a list of all tables registered in the given context.
   * - :ref:`UNNEST <unnest_table_func>`
     - Unnest one or more arrays as columns in a new table object.
   * - :ref:`TRUNCATE <truncate>`
     - Remove all data from a table without actually deleting it.


.. _create_table:

CREATE TABLE
------------
Create a new table and its columns from a SQL query executed against an existing table.

**Example:**

.. code-block:: sql

    CREATE TABLE new_table AS
    SELECT * FROM existing_table WHERE value > 42

.. _delete_from_table:

DELETE
------
Remove specific rows from a table using an (optional) constraint.
Omitting the constraint deletes all rows, equivalent to TRUNCATE.

**Example:**

.. code-block:: sql

    DELETE FROM some_table WHERE value < 0

.. _drop_tables:

DROP TABLES
-----------
Deletes the specified table, unregistering it.

**Example:**

.. code-block:: sql

    DROP TABLE old_table

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
Remove all data from a table without actually deleting it.

**Example:**

.. code-block:: sql

    TRUNCATE TABLE some_table
