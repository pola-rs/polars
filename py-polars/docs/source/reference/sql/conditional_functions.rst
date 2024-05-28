Conditional functions
==========================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`COALESCE <coalesce>`
     - Returns the first non-null value in the provided values/columns.
   * - :ref:`GREATEST <greatest>`
     - Returns the greatest value in the list of expressions.
   * - :ref:`IF <if>`
     - Returns expr1 if the boolean condition provided as the first parameter evaluates to true, and expr2 otherwise.
   * - :ref:`IFNULL <ifnull>`
     - If an expression value is NULL, return an alternative value.
   * - :ref:`LEAST <least>`
     - Returns the smallest value in the list of expressions.
   * - :ref:`NULLIF <nullif>`
     - Returns NULL if two expressions are equal, otherwise returns the first.

.. _coalesce:

COALESCE
-----------
Returns the first non-null value in the provided values/columns.

**Example:**

.. code-block:: sql

    SELECT COALESCE(column_1, ...) FROM df;

.. _greatest:

GREATEST
-----------
Returns the greatest value in the list of expressions.

**Example:**

.. code-block:: sql

    SELECT GREATEST(column_1, column_2, ...) FROM df;

.. _if:

IF
-----------
Returns expr1 if the boolean condition provided as the first parameter evaluates to true, and expr2 otherwise.

**Example:**

.. code-block:: sql

    SELECT IF(column < 0, expr1, expr2) FROM df;

.. _ifnull:

IFNULL
-----------
If an expression value is NULL, return an alternative value.

**Example:**

.. code-block:: sql

    SELECT IFNULL(string_col, 'n/a') FROM df;

.. _least:

LEAST
-----------
Returns the smallest value in the list of expressions.

**Example:**

.. code-block:: sql

    SELECT LEAST(column_1, column_2, ...) FROM df;

.. _nullif:

NULLIF
-----------
Returns NULL if two expressions are equal, otherwise returns the first.

**Example:**

.. code-block:: sql

    SELECT NULLIF(column_1, column_2) FROM df;
