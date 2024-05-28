Array Functions
=====================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`ARRAY_CONTAINS <array_contains>`
     - Returns true if the array contains the value.
   * - :ref:`ARRAY_GET <array_get>`
     - Returns the value at the given index in the array.
   * - :ref:`ARRAY_LENGTH <array_length>`
     - Returns the length of the array.
   * - :ref:`ARRAY_MAX <array_max>`
     - Returns the maximum value in an array; equivalent to array_max.
   * - :ref:`ARRAY_MEAN <array_mean>`
     - Returns the mean of all values in an array.
   * - :ref:`ARRAY_MIN <array_min>`
     - Returns the minimum value in an array; equivalent to array_min.
   * - :ref:`ARRAY_REVERSE <array_reverse>`
     - Returns the array with the elements in reverse order.
   * - :ref:`ARRAY_SUM <array_sum>`
     - Returns the sum of all values in an array.
   * - :ref:`ARRAY_TO_STRING <array_to_string>`
     - Takes all elements of the array and joins them into one string.
   * - :ref:`ARRAY_UNIQUE <array_unique>`
     - Returns the array with the unique elements.
   * - :ref:`EXPLODE <explode>`
     - Unnests/explodes an array column into multiple rows.

.. _array_contains:

ARRAY_CONTAINS
---------------
Returns true if the array contains the value.

**Example:**

.. code-block:: sql

    SELECT ARRAY_CONTAINS(column_1, 'foo') FROM df;

.. _array_get:

ARRAY_GET
-----------
Returns the value at the given index in the array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_GET(column_1, 1) FROM df;

.. _array_length:

ARRAY_LENGTH
-------------
Returns the length of the array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_LENGTH(column_1) FROM df;

.. _array_max:

ARRAY_MAX
------------
Returns the maximum value in an array; equivalent to `array_max`.

**Example:**

.. code-block:: sql

    SELECT ARRAY_MAX(column_1) FROM df;

.. _array_mean:

ARRAY_MEAN
-----------
Returns the mean of all values in an array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_MEAN(column_1) FROM df;

.. _array_min:

ARRAY_MIN
------------
Returns the minimum value in an array; equivalent to `array_min`.

**Example:**

.. code-block:: sql

    SELECT ARRAY_MIN(column_1) FROM df;

.. _array_reverse:

ARRAY_REVERSE
---------------
Returns the array with the elements in reverse order.

**Example:**

.. code-block:: sql

    SELECT ARRAY_REVERSE(column_1) FROM df;

.. _array_sum:

ARRAY_SUM
-----------
Returns the sum of all values in an array.

**Example:**

.. code-block:: sql

    SELECT ARRAY_SUM(column_1) FROM df;

.. _array_to_string:

ARRAY_TO_STRING
-----------------
Takes all elements of the array and joins them into one string.

**Example:**

.. code-block:: sql

    SELECT ARRAY_TO_STRING(column_1, ',') FROM df;
    SELECT ARRAY_TO_STRING(column_1, ',', 'n/a') FROM df;

.. _array_unique:

ARRAY_UNIQUE
-------------
Returns the array with the unique elements.

**Example:**

.. code-block:: sql

    SELECT ARRAY_UNIQUE(column_1) FROM df;

.. _explode:

EXPLODE
-----------
Unnests/explodes an array column into multiple rows.

**Example:**

.. code-block:: sql

    SELECT UNNEST(column_1) FROM df;


