SQL Clauses
===========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`SELECT <select>`
     - Retrieves specific column data from one or more tables.
   * - :ref:`FROM <from>`
     - Specify the table(s) from which to retrieve or delete data.
   * - :ref:`JOIN <join>`
     - Combine rows from two or more tables based on a related column.
   * - :ref:`WHERE <where>`
     - Filter rows returned from the query based on specific condition(s).
   * - :ref:`GROUP BY <group_by>`
     - Aggregate row values based based on one or more key columns.
   * - :ref:`HAVING <having>`
     - Filter groups in a `GROUP BY` based on specific condition(s).
   * - :ref:`ORDER BY <order_by>`
     - Sort the query result based on one or more specified columns.
   * - :ref:`LIMIT <limit>`
     - Specify the number of rows returned, and .
   * - :ref:`OFFSET <offset>` |
     - Skip a specified number of rows.


.. _select:

SELECT
------
Select the columns to be returned by the query.

**Example:**

.. code-block:: sql

    SELECT column_1, column_2 FROM df;
.. _from:

FROM
----
Specifies the table(s) from which to retrieve or delete data.

**Example:**

.. code-block:: sql

    SELECT * FROM df

.. _join:

JOIN
----
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

.. _where:

WHERE
-----

Filter rows returned from the query based on specific condition(s).

.. _group_by:

GROUP BY
--------
Group rows that have the same values in specified columns into summary rows.

**Example:**

.. code-block:: sql

    SELECT column_1, SUM(column_2) FROM df GROUP BY column_1

.. _having:

HAVING
------
Filter groups in a `GROUP BY` based on specific condition(s).


.. _order_by:

ORDER BY
--------
Sort the query result based on one or more specified columns.

**Example:**

.. code-block:: sql

    SELECT * FROM df ORDER BY column_1 ASC, column_2 DESC

.. _limit:

LIMIT
-----
Limit the number of rows returned by the query.

**Example:**

.. code-block:: sql

    SELECT column_1, column_2 FROM df LIMIT 10

.. _offset:

OFFSET
------
Skip a number of rows before starting to return rows from the query.

**Example:**

.. code-block:: sql

    SELECT column_1, column_2 FROM df LIMIT 10 OFFSET 5
