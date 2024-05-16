Array functions
=====================

.. list-table::

   * - :ref:`ArrayLength <array_length>`
     - Returns the length of the array.
   * - :ref:`ArrayMin <array_min>`
     - Returns the minimum value in an array; equivalent to `array_min`.
   * - :ref:`ArrayMax <array_max>`
     - Returns the maximum value in an array; equivalent to `array_max`.
   * - :ref:`ArraySum <array_sum>`
     - Returns the sum of all values in an array.
   * - :ref:`ArrayMean <array_mean>`
     - Returns the mean of all values in an array.
   * - :ref:`ArrayReverse <array_reverse>`
     - Returns the array with the elements in reverse order.
   * - :ref:`ArrayUnique <array_unique>`
     - Returns the array with the unique elements.
   * - :ref:`Explode <explode>`
     - Unnests/explodes an array column into multiple rows.
   * - :ref:`ArrayToString <array_to_string>`
     - Takes all elements of the array and joins them into one string.
   * - :ref:`ArrayGet <array_get>`
     - Returns the value at the given index in the array.
   * - :ref:`ArrayContains <array_contains>`
     - Returns true if the array contains the value.

.. _array_length:

ArrayLength
-----------
Returns the length of the array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_LENGTH(column_1) FROM df;

.. _array_min:

ArrayMin
-----------
Returns the minimum value in an array; equivalent to `array_min`.

**Example:**

.. code-block:: sql

    SELECT ARRAY_LOWER(column_1) FROM df;

.. _array_max:

ArrayMax
-----------
Returns the maximum value in an array; equivalent to `array_max`.

**Example:**

.. code-block:: sql

    SELECT ARRAY_UPPER(column_1) FROM df;

.. _array_sum:

ArraySum
-----------
Returns the sum of all values in an array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_SUM(column_1) FROM df;

.. _array_mean:

ArrayMean
-----------
Returns the mean of all values in an array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_MEAN(column_1) FROM df;

.. _array_reverse:

ArrayReverse
---------------
Returns the array with the elements in reverse order.

**Example:**

.. code-block:: sql

    SELECT ARRAY_REVERSE(column_1) FROM df;

.. _array_unique:

ArrayUnique
-------------
Returns the array with the unique elements.

**Example:**

.. code-block:: sql

    SELECT ARRAY_UNIQUE(column_1) FROM df;

.. _explode:

Explode
-----------
Unnests/explodes an array column into multiple rows.

**Example:**

.. code-block:: sql

    SELECT UNNEST(column_1) FROM df;

.. _array_to_string:

ArrayToString
--------------
Takes all elements of the array and joins them into one string.

**Example:**

.. code-block:: sql

    SELECT ARRAY_TO_STRING(column_1, ',') FROM df;
    SELECT ARRAY_TO_STRING(column_1, ',', 'n/a') FROM df;

.. _array_get:

ArrayGet
-----------
Returns the value at the given index in the array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_GET(column_1, 1) FROM df;

.. _array_contains:

ArrayContains
---------------
Returns true if the array contains the value.

**Example:**

.. code-block:: sql

    SELECT ARRAY_CONTAINS(column_1, 'foo') FROM df;
