Set Operations
==============

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`UNION <union>`
     - Combine the distinct result sets of two or more SELECT statements.
       The final result set will have no duplicate rows.
   * - :ref:`UNION ALL <union_all>`
     - Combine the complete result sets of two or more SELECT statements.
       The final result set will be composed of all rows from each query.
   * - :ref:`UNION [ALL] BY NAME <union_by_name>`
     - Combine the result sets of two or more SELECT statements by column name
       instead of by position; if `ALL` is omitted the final result will have
       no duplicate rows.


.. _union:

UNION
-----
Combine the result sets of two or more SELECT statements into a single result set.

**Example:**

.. code-block:: sql

    SELECT name, city FROM df.customers
    UNION
    SELECT name, city FROM df.suppliers

.. _union_all:

UNION ALL
---------
Combine the result sets of two or more SELECT statements into a single result set.

**Example:**

.. code-block:: sql

    SELECT name, city FROM df.customers
    UNION ALL
    SELECT name, city FROM df.suppliers

.. _union_by_name:

UNION BY NAME
-------------
Combine the result sets of two or more SELECT statements into a single result set.

**Example:**

.. code-block:: sql

    SELECT name, city FROM df.customers
    UNION BY NAME
    SELECT city, name FROM df.suppliers
