Trig functions
===================

.. list-table::


   * - :ref:`Cos <cos>`
     - Compute the cosine sine of the input column (in radians).
   * - :ref:`Cot <cot>`
     - Compute the cotangent of the input column (in radians).
   * - :ref:`Sin <sin>`
     - Compute the sine of the input column (in radians).
   * - :ref:`Tan <tan>`
     - Compute the tangent of the input column (in radians).
   * - :ref:`CosD <cosd>`
     - Compute the cosine sine of the input column (in degrees).
   * - :ref:`CotD <cotd>`
     - Compute cotangent of the input column (in degrees).
   * - :ref:`SinD <sind>`
     - Compute the sine of the input column (in degrees).
   * - :ref:`TanD <tand>`
     - Compute the tangent of the input column (in degrees).
   * - :ref:`Acos <acos>`
     - Compute inverse cosine of the input column (in radians).
   * - :ref:`Asin <asin>`
     - Compute inverse sine of the input column (in radians).
   * - :ref:`Atan <atan>`
     - Compute inverse tangent of the input column (in radians).
   * - :ref:`Atan2 <atan2>`
     - Compute the inverse tangent of column_2/column_1 (in radians).
   * - :ref:`AcosD <acosd>`
     - Compute inverse cosine of the input column (in degrees).
   * - :ref:`AsinD <asind>`
     - Compute inverse sine of the input column (in degrees).
   * - :ref:`AtanD <atand>`
     - Compute inverse tangent of the input column (in degrees).
   * - :ref:`Atan2D <atan2d>`
     - Compute the inverse tangent of column_2/column_1 (in degrees).
   * - :ref:`Degrees <degrees>`
     - Convert between radians and degrees.
   * - :ref:`Radians <radians>`
     - Convert between degrees and radians.

.. _cos:

Cos
-----------
Compute the cosine of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT COS(column_1) FROM df;

.. _cot:

Cot
-----------
Compute the cotangent of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT COT(column_1) FROM df;

.. _sin:

Sin
-----------
Compute the sine of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT SIN(column_1) FROM df;

.. _tan:

Tan
-----------
Compute the tangent of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT TAN(column_1) FROM df;

.. _cosd:

CosD
-----------
Compute the cosine of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT COSD(column_1) FROM df;

.. _cotd:

CotD
-----------
Compute cotangent of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT COTD(column_1) FROM df;

.. _sind:

SinD
-----------
Compute the sine of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT SIND(column_1) FROM df;

.. _tand:

TanD
-----------
Compute the tangent of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT TAND(column_1) FROM df;

.. _acos:

Acos
-----------
Compute inverse cosine of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT ACOS(column_1) FROM df;

.. _asin:

Asin
-----------
Compute inverse sine of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT ASIN(column_1) FROM df;

.. _atan:

Atan
-----------
Compute inverse tangent of the input column (in radians).

**Example:**

.. code-block:: sql

    SELECT ATAN(column_1) FROM df;

.. _atan2:

Atan2
-----------
Compute the inverse tangent of column_2/column_1 (in radians).

**Example:**

.. code-block:: sql

    SELECT ATAN2(column_1, column_2) FROM df;

.. _acosd:

AcosD
-----------
Compute inverse cosine of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT ACOSD(column_1) FROM df;

.. _asind:

AsinD
-----------
Compute inverse sine of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT ASIND(column_1) FROM df;

.. _atand:

AtanD
-----------
Compute inverse tangent of the input column (in degrees).

**Example:**

.. code-block:: sql

    SELECT ATAND(column_1) FROM df;

.. _atan2d:

Atan2D
-----------
Compute the inverse tangent of column_2/column_1 (in degrees).

**Example:**

.. code-block:: sql

    SELECT ATAN2D(column_1, column_2) FROM df;

.. _degrees:

Degrees
-----------
Convert between radians and degrees.

**Example:**

.. code-block:: sql

    SELECT DEGREES(column_1) FROM df;

.. _radians:

Radians
-----------
Convert between degrees and radians.

**Example:**

.. code-block:: sql

    SELECT RADIANS(column_1) FROM df;
