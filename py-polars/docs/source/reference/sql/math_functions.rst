Math functions
===================

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`ABS <abs>`
     - Returns the absolute value of the input column.
   * - :ref:`CEIL <ceil>`
     - Returns the nearest integer closest from zero.
   * - :ref:`EXP <exp>`
     - Computes the exponential of the given value.
   * - :ref:`FLOOR <floor_function>`
     - Returns the nearest integer away from zero.
   * - :ref:`PI <pi>`
     - Returns a (very good) approximation of 𝜋. 
   * - :ref:`LN <ln>`
     - Computes the natural logarithm of the given value.
   * - :ref:`LOG2 <log2>`
     - Computes the logarithm of the given value in base 2.
   * - :ref:`LOG10 <log10>`
     - Computes the logarithm of the given value in base 10.
   * - :ref:`LOG <log>`
     - Computes the `base` logarithm of the given value.
   * - :ref:`LOG1P <log1p>`
     - Computes the natural logarithm of "given value plus one".
   * - :ref:`POW <pow>`
     - Returns the value to the power of the given exponent.
   * - :ref:`MOD <mod>`
     - Returns the remainder of a numeric expression divided by another numeric expression.
   * - :ref:`SQRT <sqrt>`
     - Returns the square root (√) of a number.
   * - :ref:`CBRT <cbrt>`
     - Returns the cube root (∛) of a number.
   * - :ref:`ROUND <round>`
     - Round a number to `x` decimals (default: 0) away from zero.
   * - :ref:`SIGN <sign>`
     - Returns the sign of the argument as -1, 0, or +1.
   
.. _abs:

ABS
-----------
Returns the absolute value of the input column.

**Example:**

.. code-block:: sql

    SELECT ABS(column_1) FROM df;

.. _ceil:

CEIL 
--------------
Returns the nearest integer closest from zero.

**Example:**

.. code-block:: sql

    SELECT CEIL(column_1) FROM df;

.. _exp:

EXP 
------------
Computes the exponential of the given value.

**Example:**

.. code-block:: sql

    SELECT EXP(column_1) FROM df;

.. _floor_function:

FLOOR 
--------------
Returns the nearest integer away from zero.

**Example:**

.. code-block:: sql

    SELECT FLOOR(column_1) FROM df;

.. _pi:

PI 
-----------
Returns a (very good) approximation of 𝜋.

**Example:**

.. code-block:: sql

    SELECT PI() FROM df;

.. _ln:

LN
-----------
Computes the natural logarithm of the given value.

**Example:**

.. code-block:: sql

    SELECT LN(column_1) FROM df;

.. _log2:

LOG2 
-------------
Computes the logarithm of the given value in base 2.

**Example:**

.. code-block:: sql

    SELECT LOG2(column_1) FROM df;

.. _log10:

LOG10
--------------
Computes the logarithm of the given value in base 10.

**Example:**

.. code-block:: sql

    SELECT LOG10(column_1) FROM df;

.. _log:

LOG
------------
Computes the `base` logarithm of the given value.

**Example:**

.. code-block:: sql

    SELECT LOG(column_1, 10) FROM df;

.. _log1p:

LOG1P
--------------
Computes the natural logarithm of "given value plus one".

**Example:**

.. code-block:: sql

    SELECT LOG1P(column_1) FROM df;

.. _pow:

POW
-----------
Returns the value to the power of the given exponent.

**Example:**

.. code-block:: sql

    SELECT POW(column_1, 2) FROM df;

.. _mod:

MOD
-----------
Returns the remainder of a numeric expression divided by another numeric expression.

**Example:**

.. code-block:: sql

    SELECT MOD(column_1, 2) FROM df;

.. _sqrt:

SQRT
-----------
Returns the square root (√) of a number.

**Example:**

.. code-block:: sql

    SELECT SQRT(column_1) FROM df;

.. _cbrt:

CBRT
-----------
Returns the cube root (∛) of a number.

**Example:**

.. code-block:: sql

    SELECT CBRT(column_1) FROM df;

.. _round:

ROUND
-----------
Round a number to `x` decimals (default: 0) away from zero.

**Example:**

.. code-block:: sql

    SELECT ROUND(column_1, 3) FROM df;

.. _sign:

SIGN
-----------
Returns the sign of the argument as -1, 0, or +1.

**Example:**

.. code-block:: sql

    SELECT SIGN(column_1) FROM df;
