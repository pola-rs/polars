Trigonometry
============

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - :ref:`ACOS <acos>`
     - Compute inverse cosine of the input column (in radians).
   * - :ref:`ACOSD <acosd>`
     - Compute inverse cosine of the input column (in degrees).
   * - :ref:`ASIN <asin>`
     - Compute inverse sine of the input column (in radians).
   * - :ref:`ASIND <asind>`
     - Compute inverse sine of the input column (in degrees).
   * - :ref:`ATAN <atan>`
     - Compute inverse tangent of the input column (in radians).
   * - :ref:`ATAND <atand>`
     - Compute inverse tangent of the input column (in degrees).
   * - :ref:`ATAN2 <atan2>`
     - Compute the inverse tangent of column_2/column_1 (in radians).
   * - :ref:`ATAN2D <atan2d>`
     - Compute the inverse tangent of column_2/column_1 (in degrees).
   * - :ref:`COT <cot>`
     - Compute the cotangent of the input column (in radians).
   * - :ref:`COTD <cotd>`
     - Compute cotangent of the input column (in degrees).
   * - :ref:`COS <cos>`
     - Compute the cosine of the input column (in radians).
   * - :ref:`COSD <cosd>`
     - Compute the cosine of the input column (in degrees).
   * - :ref:`DEGREES <degrees>`
     - Convert between radians and degrees.
   * - :ref:`RADIANS <radians>`
     - Convert between degrees and radians.
   * - :ref:`SIN <sin>`
     - Compute the sine of the input column (in radians).
   * - :ref:`SIND <sind>`
     - Compute the sine of the input column (in degrees).
   * - :ref:`TAN <tan>`
     - Compute the tangent of the input column (in radians).
   * - :ref:`TAND <tand>`
     - Compute the tangent of the input column (in degrees).

.. _acos:

ACOS
----
Compute inverse cosine of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
      }
    )

    df.sql("SELECT ACOS(a) AS ACOS FROM self")

    # shape: (4, 1)
    # ┌──────────┐
    # │ ACOS     │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 3.141593 │
    # │ 2.094395 │
    # │ 1.047198 │
    # │ 0.0      │
    # └──────────┘

.. _acosd:

ACOSD
-----
Compute inverse cosine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
      }
    )

    df.sql("SELECT ACOSD(a) AS ACOSD FROM self")
    # shape: (4, 1)
    # ┌───────┐
    # │ ACOSD │
    # │ ---   │
    # │ f64   │
    # ╞═══════╡
    # │ 180.0 │
    # │ 120.0 │
    # │ 60.0  │
    # │ 0.0   │
    # └───────┘

.. _asin:

ASIN
----
Compute inverse sine of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
      }
    )

    df.sql("SELECT ASIN(a) AS ASIN FROM self")
    # shape: (4, 1)
    # ┌───────────┐
    # │ ASIN      │
    # │ ---       │
    # │ f64       │
    # ╞═══════════╡
    # │ -1.570796 │
    # │ -0.523599 │
    # │ 0.523599  │
    # │ 1.570796  │
    # └───────────┘

.. _asind:

ASIND
-----
Compute inverse sine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
      }
    )

    df.sql("SELECT ASIND(a) AS ASIND FROM self")
    # shape: (4, 1)
    # ┌───────┐
    # │ ASIND │
    # │ ---   │
    # │ f64   │
    # ╞═══════╡
    # │ -90.0 │
    # │ -30.0 │
    # │ 30.0  │
    # │ 90.0  │
    # └───────┘

.. _atan:

ATAN
----
Compute inverse tangent of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
      }
    )

    df.sql("SELECT ATAN(a) AS ATAN FROM self")
    # shape: (4, 1)
    # ┌───────────┐
    # │ ATAN      │
    # │ ---       │
    # │ f64       │
    # ╞═══════════╡
    # │ -0.785398 │
    # │ -0.463648 │
    # │ 0.463648  │
    # │ 0.785398  │
    # └───────────┘

.. _atand:

ATAND
-----
Compute inverse tangent of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
      }
    )

    df.sql("SELECT ATAND(a) AS ATAND FROM self")
    # shape: (4, 1)
    # ┌────────────┐
    # │ ATAND      │
    # │ ---        │
    # │ f64        │
    # ╞════════════╡
    # │ -45.0      │
    # │ -26.565051 │
    # │ 26.565051  │
    # │ 45.0       │
    # └────────────┘

.. _atan2:

ATAN2
-----
Compute the inverse tangent of column_2/column_1 (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
        "b": [10, 20, 30, 40],
      }
    )

    df.sql("SELECT ATAN2(a, b) AS ATAN2 FROM self")
    # shape: (4, 1)
    # ┌───────────┐
    # │ ATAN2     │
    # │ ---       │
    # │ f64       │
    # ╞═══════════╡
    # │ -0.099669 │
    # │ -0.024995 │
    # │ 0.016665  │
    # │ 0.024995  │
    # └───────────┘

.. _atan2d:

ATAN2D
------
Compute the inverse tangent of column_2/column_1 (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "a": [-1.0, -0.5, 0.5, 1],
        "b": [10, 20, 30, 40],
      }
    )

    df.sql("SELECT ATAN2D(a, b) AS ATAN2D FROM self")
    # shape: (4, 1)
    # ┌───────────┐
    # │ ATAN2D    │
    # │ ---       │
    # │ f64       │
    # ╞═══════════╡
    # │ -5.710593 │
    # │ -1.432096 │
    # │ 0.954841  │
    # │ 1.432096  │
    # └───────────┘

.. _cot:

COT
---
Compute the cotangent of the input column (in radians).

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame(
      {
        "angle": [0.0, math.pi/2, math.pi, 3*math.pi/2],
      }
    )

    df.sql("SELECT COT(angle) AS COT FROM self")
    # shape: (4, 1)
    # ┌────────────┐
    # │ COT        │
    # │ ---        │
    # │ f64        │
    # ╞════════════╡
    # │ inf        │
    # │ 6.1232e-17 │
    # │ -8.1656e15 │
    # │ 1.8370e-16 │
    # └────────────┘

.. _cotd:

COTD
----
Compute cotangent of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "angle": [0, 90, 180, 270]
      }
    )

    df.sql("SELECT COTD(angle) AS COTD FROM self")
    # shape: (4, 1)
    # ┌────────────┐
    # │ COTD       │
    # │ ---        │
    # │ f64        │
    # ╞════════════╡
    # │ inf        │
    # │ 6.1232e-17 │
    # │ -8.1656e15 │
    # | 1.8370e-16 │
    # └────────────┘

.. _cos:

COS
---
Compute the cosine of the input column (in radians).

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame(
      {
        "angle": [0.0, math.pi/2, math.pi, 3*math.pi/2],
      }
    )

    df.sql("SELECT COS(angle) AS COS FROM self")
    # shape: (4, 1)
    # ┌─────────────┐
    # │ COS         │
    # │ ---         │
    # | f64         │
    # ╞═════════════╡
    # │ 1.0         │
    # │ 6.1232e-17  │
    # │ -1.0        │
    # │ -1.8370e-16 │
    # └─────────────┘

.. _cosd:

COSD
----
Compute the cosine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "angle": [0, 90, 180, 270]
      }
    )

    df.sql("SELECT COSD(angle) AS COSD FROM self")
    # shape: (4, 1)
    # ┌─────────────┐
    # │ COSD        │
    # │ ---         │
    # | f64         │
    # ╞═════════════╡
    # │ 1.0         │
    # │ 6.1232e-17  │
    # │ -1.0        │
    # │ -1.8370e-16 │
    # └─────────────┘

.. _degrees:

DEGREES
-------
Convert between radians and degrees.

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame(
      {
        "angle": [0.0, math.pi/2, math.pi, 3*math.pi/2],
      }
    )

    df.sql("SELECT DEGREES(angle) AS DEGREES FROM self")
    # shape: (4, 1)
    # ┌─────────┐
    # │ DEGREES │
    # │ ---     │
    # │ f64     │
    # ╞═════════╡
    # │ 0.0     │
    # │ 90.0    │
    # │ 180.0   │
    # │ 270.0   │
    # └─────────┘

.. _radians:

RADIANS
-------
Convert between degrees and radians.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "angle": [0, 90, 180, 270]
      }
    )

    df.sql("SELECT RADIANS(angle_degrees) FROM self")
    # shape: (4, 1)
    # ┌───────────────┐
    # │ angle_degrees │
    # │ ---           │
    # │ f64           │
    # ╞═══════════════╡
    # │ 0.0           │
    # │ 1.570796      │
    # │ 3.141593      │
    # │ 4.712389      │
    # └───────────────┘

.. _sin:

SIN
---
Compute the sine of the input column (in radians).

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame(
      {
        "angle": [0.0, math.pi/2, math.pi, 3*math.pi/2],
      }
    )

    df.sql("SELECT SIN(angle) AS SIN FROM self")
    # shape: (4, 1)
    # ┌────────────┐
    # │ SIN        │
    # │ ---        │
    # │ f64        │
    # ╞════════════╡
    # │ 0.0        │
    # │ 1.0        │
    # │ 1.2246e-16 │
    # │ -1.0       │
    # └────────────┘

.. _sind:

SIND
----
Compute the sine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "angle": [0, 90, 180, 270]
      }
    )

    df.sql("SELECT SIND(angle) FROM self")
    # shape: (4, 1)
    # ┌────────────┐
    # │ SIND       │
    # │ ---        │
    # | f64        │
    # ╞════════════╡
    # │ 0.0        │
    # │ 1.0        │
    # │ 1.2246e-16 │
    # │ -1.0       │
    # └────────────┘

.. _tan:

TAN
---
Compute the tangent of the input column (in radians).

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame(
      {
        "angle": [0.0, math.pi/2, math.pi, 3*math.pi/2],
      }
    )

    df.sql("SELECT TAN(angle) AS TAN FROM self")
    # shape: (4, 1)
    # ┌─────────────┐
    # │ TAN         │
    # │ ---         │
    # │ f64         │
    # ╞═════════════╡
    # │ 0.0         │
    # │ 1.6331e16   │
    # │ -1.2246e-16 │
    # │ 5.4437e15   │
    # └─────────────┘


.. _tand:

TAND
----
Compute the tangent of the input column (in degrees).

**Example:**

.. code-block:: python


    df = pl.DataFrame(
      {
        "angle": [0, 90, 180, 270]
      }
    )

    df.sql("SELECT TAND(angle) AS TAND FROM self")
    # shape: (4, 1)
    # ┌─────────────┐
    # │ TAND        │
    # │ ---         │
    # │ f64         │
    # ╞═════════════╡
    # │ 0.0         │
    # │ 1.6331e16   │
    # │ -1.2246e-16 │
    # │ 5.4437e15   │
    # └─────────────┘
