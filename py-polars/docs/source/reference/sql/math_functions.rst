Math functions
===================

.. list-table::

   * - :ref:`Abs <abs>`
     - Returns the absolute value of the input column.
   * - :ref:`Ceil <ceil>`
     - Returns the nearest integer closest from zero.
   * - :ref:`Exp <exp>`
     - Computes the exponential of the given value.
   * - :ref:`Floor <floor_function>`
     - Returns the nearest integer away from zero.
   * - :ref:`Pi <pi>`
     - Returns a (very good) approximation of 𝜋.
   * - :ref:`Ln <ln>`
     - Computes the natural logarithm of the given value.
   * - :ref:`Log2 <log2>`
     - Computes the logarithm of the given value in base 2.
   * - :ref:`Log10 <log10>`
     - Computes the logarithm of the given value in base 10.
   * - :ref:`Log <log>`
     - Computes the `base` logarithm of the given value.
   * - :ref:`Log1p <log1p>`
     - Computes the natural logarithm of "given value plus one".
   * - :ref:`Pow <pow>`
     - Returns the value to the power of the given exponent.
   * - :ref:`Mod <mod>`
     - Returns the remainder of a numeric expression divided by another numeric expression.
   * - :ref:`Sqrt <sqrt>`
     - Returns the square root (√) of a number.
   * - :ref:`Cbrt <cbrt>`
     - Returns the cube root (∛) of a number.
   * - :ref:`Round <round>`
     - Round a number to `x` decimals (default: 0) away from zero.
   * - :ref:`Sign <sign>`
     - Returns the sign of the argument as -1, 0, or +1.
   
.. _abs:

Abs
-----------
Returns the absolute value of the input column.

**Example:**

.. code-block:: sql

    SELECT ABS(column_1) FROM df;

.. _ceil:

Ceil 
--------------
Returns the nearest integer closest from zero.

**Example:**

.. code-block:: sql

    SELECT CEIL(column_1) FROM df;

.. _exp:

Exp 
------------
Computes the exponential of the given value.

**Example:**

.. code-block:: sql

    SELECT EXP(column_1) FROM df;

.. _floor_function:

Floor 
--------------
Returns the nearest integer away from zero.

**Example:**

.. code-block:: sql

    SELECT FLOOR(column_1) FROM df;

.. _pi:

Pi 
-----------
Returns a (very good) approximation of 𝜋.

**Example:**

.. code-block:: sql

    SELECT PI() FROM df;

.. _ln:

Ln 
-----------
Computes the natural logarithm of the given value.

**Example:**

.. code-block:: sql

    SELECT LN(column_1) FROM df;

.. _log2:

Log2 
-------------
Computes the logarithm of the given value in base 2.

**Example:**

.. code-block:: sql

    SELECT LOG2(column_1) FROM df;

.. _log10:

Log10 Function
--------------
Computes the logarithm of the given value in base 10.

**Example:**

.. code-block:: sql

    SELECT LOG10(column_1) FROM df;

.. _log:

Log 
------------
Computes the `base` logarithm of the given value.

**Example:**

.. code-block:: sql

    SELECT LOG(column_1, 10) FROM df;

.. _log1p:

Log1p 
--------------
Computes the natural logarithm of "given value plus one".

**Example:**

.. code-block:: sql

    SELECT LOG1P(column_1) FROM df;

.. _pow:

Pow
-----------
Returns the value to the power of the given exponent.

**Example:**

.. code-block:: sql

    SELECT POW(column_1, 2) FROM df;

.. _mod:

Mod
-----------
Returns the remainder of a numeric expression divided by another numeric expression.

**Example:**

.. code-block:: sql

    SELECT MOD(column_1, 2) FROM df;

.. _sqrt:

Sqrt
-----------
Returns the square root (√) of a number.

**Example:**

.. code-block:: sql

    SELECT SQRT(column_1) FROM df;

.. _cbrt:

Cbrt
-----------
Returns the cube root (∛) of a number.

**Example:**

.. code-block:: sql

    SELECT CBRT(column_1) FROM df;

.. _round:

Round
-----------
Round a number to `x` decimals (default: 0) away from zero.

**Example:**

.. code-block:: sql

    SELECT ROUND(column_1, 3) FROM df;

.. _sign:

Sign
-----------
Returns the sign of the argument as -1, 0, or +1.

**Example:**

.. code-block:: sql

    SELECT SIGN(column_1) FROM df;
