Conditional functions
==========================

.. list-table::

   * - :ref:`Coalesce <coalesce>`
     - Returns the first non-null value in the provided values/columns.
   * - :ref:`Greatest <greatest>`
     - Returns the greatest value in the list of expressions.
   * - :ref:`If <if>`
     - Returns expr1 if the boolean condition provided as the first parameter evaluates to true, and expr2 otherwise.
   * - :ref:`IfNull <ifnull>`
     - If an expression value is NULL, return an alternative value.
   * - :ref:`Least <least>`
     - Returns the smallest value in the list of expressions.
   * - :ref:`NullIf <nullif>`
     - Returns NULL if two expressions are equal, otherwise returns the first.

.. _coalesce:

Coalesce
-----------
Returns the first non-null value in the provided values/columns.

**Example:**

.. code-block:: sql

    SELECT COALESCE(column_1, ...) FROM df;

.. _greatest:

Greatest
-----------
Returns the greatest value in the list of expressions.

**Example:**

.. code-block:: sql

    SELECT GREATEST(column_1, column_2, ...) FROM df;

.. _if:

If
-----------
Returns expr1 if the boolean condition provided as the first parameter evaluates to true, and expr2 otherwise.

**Example:**

.. code-block:: sql

    SELECT IF(column < 0, expr1, expr2) FROM df;

.. _ifnull:

IfNull
-----------
If an expression value is NULL, return an alternative value.

**Example:**

.. code-block:: sql

    SELECT IFNULL(string_col, 'n/a') FROM df;

.. _least:

Least
-----------
Returns the smallest value in the list of expressions.

**Example:**

.. code-block:: sql

    SELECT LEAST(column_1, column_2, ...) FROM df;

.. _nullif:

NullIf
-----------
Returns NULL if two expressions are equal, otherwise returns the first.

**Example:**

.. code-block:: sql

    SELECT NULLIF(column_1, column_2) FROM df;