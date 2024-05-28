SQL Operations
=====================

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
   * - :ref:`FROM <from>`
     - Specify the table(s) from which to retrieve or delete data.
   * - :ref:`JOIN <join>`
     - Combine rows from two or more tables based on a related column.
   * - :ref:`GROUP BY <group_by>`
     - Group rows that have the same values in specified columns into summary rows.
   * - :ref:`LIMIT <limit>`
     - Control the number of rows returned.
   * - :ref:`OFFSET <offset>`
     - Skip a specified number of rows.
   * - :ref:`ORDER BY <order_by>`
     - Sort the query result based on one or more specified columns.
   * - :ref:`SELECT <select>`
     - Retrieves specific data from one or more tables.
   * - :ref:`SHOW TABLES <show_tables>`
     - Returns a list of all tables in the context.
   * - :ref:`SUBQUERIES <subqueries>`
     - Retrieves data from nested queries for the main query.
   * - :ref:`TRUNCATE <truncate>`
     - Remove rows from table without deleting the table from context.
   * - :ref:`UNION <union>`
     - Combine the result sets of two or more SELECT statements into a single result set.

.. _create_tables:

CREATE TABLES
-----------------
Create a new table and its columns in the context.

**Example:**

.. code-block:: sql

	CREATE TABLE new_table
    AS
    SELECT * FROM df WHERE value > 42

.. _drop_tables:

DROP TABLES
-----------------
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

.. _from:

FROM
---------
Specifies the table(s) from which to retrieve or delete data.

**Example:**

.. code-block:: sql

	SELECT * FROM df

.. _join:

JOIN
----------
Combines rows from two or more tables based on a related column. 

**Join Types**

* `CROSS JOIN`
* `FULL JOIN`
* `INNER JOIN`
* `LEFT ANTI JOIN`
* `LEFT JOIN`
* `LEFT SEMI JOIN`
* `RIGHT ANTI JOIN`
* `RIGHT SEMI JOIN`

**Example:**

.. code-block:: sql

	SELECT product_id FROM df_product LEFT JOIN df_categories USING (product_id)

.. _group_by:

GROUP BY
---------
Group rows that have the same values in specified columns into summary rows.

**Example:**

.. code-block:: sql

	SELECT column_1, SUM(column_2) FROM df GROUP BY column_1

.. _limit:

LIMIT
---------
Limit the number of rows returned by the query.

**Example:**

.. code-block:: sql

    SELECT column_1, column_2 FROM df LIMIT 10

.. _offset:

OFFSET
-------
Skip a number of rows before starting to return rows from the query.

**Example:**

.. code-block:: sql

    SELECT column_1, column_2 FROM df LIMIT 10 OFFSET 5

.. _order_by:

ORDER BY
---------
Sort the query result based on one or more specified columns.

**Example:**

.. code-block:: sql

	SELECT * FROM df ORDER BY column_1 ASC, column_2 DESC

.. _select:

SELECT
-------
Select the columns to be returned by the query.

**Example:**

.. code-block:: sql

    SELECT column_1, column_2 FROM df;

.. _show_tables:

SHOW TABLES
------------
Display the list of tables in the context.

**Example:**

.. code-block:: sql

    SHOW TABLES

.. _subqueries:

SUBQUERIES
------------
Retrieves data from nested queries for the main query. Registers the nested query as table in the context.

**Example:**

.. code-block:: sql

    SELECT column_1 FROM (SELECT column_1 FROM df WHERE column_2 > 42) AS alias

.. _truncate:

TRUNCATE
--------
Removes all rows from the specified table, but keeps the table.

**Example:**

.. code-block:: sql

    TRUNCATE TABLE df

.. _union:

UNION
---------
Combine the result sets of two or more SELECT statements into a single result set.

**Example:**

.. code-block:: sql

	SELECT name, city FROM df.customers
	UNION
	SELECT name, city FROM df.suppliers