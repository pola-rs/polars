Table Operations
================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`CREATE TABLES <create_tables>`
     - Define a new table and its columns in the context.
   * - :ref:`DROP TABLES <drop_tables>`
     - Delete a specified table and related data from the context.
   * - :ref:`EXPLAIN <explain>`
     - Returns logical plan of the query.
   * - :ref:`SHOW TABLES <show_tables>`
     - Returns a list of all tables in the context.
   * - :ref:`TRUNCATE <truncate>`
     - Remove rows from table without deleting the table from context.

.. _create_tables:

CREATE TABLES
-------------
Create a new table and its columns in the context.

**Example:**

.. code-block:: sql

    CREATE TABLE new_table
    AS
    SELECT * FROM df WHERE value > 42

.. _drop_tables:

DROP TABLES
-----------
Delete a specified table and related data from the context.

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

.. _truncate:

TRUNCATE
--------
Removes all rows from the specified table, but keeps the table.

**Example:**

.. code-block:: sql

    TRUNCATE TABLE df
