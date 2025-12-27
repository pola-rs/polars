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
     - Compute the inverse tangent of column_1/column_2 (in radians).
   * - :ref:`ATAN2D <atan2d>`
     - Compute the inverse tangent of column_1/column_2 (in degrees).
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

    df = pl.DataFrame({"rads": [-1.0, -0.5, 0.5, 1.0]})
    df.sql("SELECT rads, ACOS(rads) AS acos FROM self")
    # shape: (4, 2)
    # ┌──────┬──────────┐
    # │ rads ┆ acos     │
    # │ ---  ┆ ---      │
    # │ f64  ┆ f64      │
    # ╞══════╪══════════╡
    # │ -1.0 ┆ 3.141593 │
    # │ -0.5 ┆ 2.094395 │
    # │ 0.5  ┆ 1.047198 │
    # │ 1.0  ┆ 0.0      │
    # └──────┴──────────┘

.. _acosd:

ACOSD
-----
Compute inverse cosine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [-0.5, 0.0, 0.5]})
    df.sql("SELECT degs, ACOSD(degs) AS acosd FROM self")
    # shape: (3, 2)
    # ┌──────┬───────┐
    # │ degs ┆ acosd │
    # │ ---  ┆ ---   │
    # │ f64  ┆ f64   │
    # ╞══════╪═══════╡
    # │ -0.5 ┆ 120.0 │
    # │ 0.0  ┆ 90.0  │
    # │ 0.5  ┆ 60.0  │
    # └──────┴───────┘

.. _asin:

ASIN
----
Compute inverse sine of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"rads": [-1.0, -0.75, -0.0, 0.5]})
    df.sql("SELECT rads, ASIN(rads) AS asin FROM self")
    # shape: (4, 2)
    # ┌───────┬───────────┐
    # │ rads  ┆ asin      │
    # │ ---   ┆ ---       │
    # │ f64   ┆ f64       │
    # ╞═══════╪═══════════╡
    # │ -1.0  ┆ -1.570796 │
    # │ -0.75 ┆ -0.848062 │
    # │ -0.0  ┆ -0.0      │
    # │ 0.5   ┆ 0.523599  │
    # └───────┴───────────┘

.. _asind:

ASIND
-----
Compute inverse sine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [-0.5, 0.0, 0.5]})
    df.sql("SELECT degs, ASIND(degs) AS asind FROM self")
    # shape: (3, 2)
    # ┌──────┬───────┐
    # │ degs ┆ asind │
    # │ ---  ┆ ---   │
    # │ f64  ┆ f64   │
    # ╞══════╪═══════╡
    # │ -0.5 ┆ -30.0 │
    # │ 0.0  ┆ 0.0   │
    # │ 0.5  ┆ 30.0  │
    # └──────┴───────┘

.. _atan:

ATAN
----
Compute inverse tangent of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"rads": [-1.0, 0.0, 1.0, 2.0]})
    df.sql("SELECT rads, ATAN(rads) AS atan FROM self")
    # shape: (4, 2)
    # ┌──────┬───────────┐
    # │ rads ┆ atan      │
    # │ ---  ┆ ---       │
    # │ f64  ┆ f64       │
    # ╞══════╪═══════════╡
    # │ -1.0 ┆ -0.785398 │
    # │ 0.0  ┆ 0.0       │
    # │ 1.0  ┆ 0.785398  │
    # │ 2.0  ┆ 1.107149  │
    # └──────┴───────────┘

.. _atand:

ATAND
-----
Compute inverse tangent of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [-1.0, 0.0, 1.0]})
    df.sql("SELECT degs, ATAND(degs) AS atand FROM self")
    # shape: (3, 2)
    # ┌──────┬───────┐
    # │ degs ┆ atand │
    # │ ---  ┆ ---   │
    # │ f64  ┆ f64   │
    # ╞══════╪═══════╡
    # │ -1.0 ┆ -45.0 │
    # │ 0.0  ┆ 0.0   │
    # │ 1.0  ┆ 45.0  │
    # └──────┴───────┘

.. _atan2:

ATAN2
-----
Compute the inverse tangent of column_1/column_2 (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "a": [-2.0, -1.0, 1.0, 2.0],
            "b": [1.5, 1.0, 0.5, 0.0],
        }
    )
    df.sql("SELECT a, b, ATAN2(a, b) AS atan2_ab FROM self")
    # shape: (4, 3)
    # ┌──────┬─────┬───────────┐
    # │ a    ┆ b   ┆ atan2_ab  │
    # │ ---  ┆ --- ┆ ---       │
    # │ f64  ┆ f64 ┆ f64       │
    # ╞══════╪═════╪═══════════╡
    # │ -2.0 ┆ 1.5 ┆ -0.927295 │
    # │ -1.0 ┆ 1.0 ┆ -0.785398 │
    # │ 1.0  ┆ 0.5 ┆ 1.107149  │
    # │ 2.0  ┆ 0.0 ┆ 1.570796  │
    # └──────┴─────┴───────────┘

.. _atan2d:

ATAN2D
------
Compute the inverse tangent of column_1/column_2 (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "a": [-1.0, 0.0, 1.0, 1.0],
            "b": [1.0, 1.0, 0.0, -1.0],
        }
    )
    df.sql("SELECT a, b, ATAN2D(a, b) AS atan2d_ab FROM self")
    # shape: (4, 3)
    # ┌──────┬──────┬───────────┐
    # │ a    ┆ b    ┆ atan2d_ab │
    # │ ---  ┆ ---  ┆ ---       │
    # │ f64  ┆ f64  ┆ f64       │
    # ╞══════╪══════╪═══════════╡
    # │ -1   ┆  1.0 ┆   135.0   │
    # │ 0.0  ┆  1.0 ┆    90.0   │
    # │ 1.0  ┆  0.0 ┆     0.0   │
    # │ 1.0  ┆ -1.0 ┆   -45.0   │
    # └──────┴──────┴───────────┘

.. _cot:

COT
---
Compute the cotangent of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"rads": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    df.sql("SELECT rads, COT(rads) AS cot FROM self")
    # shape: (5, 2)
    # ┌──────┬───────────┐
    # │ rads ┆ cot       │
    # │ ---  ┆ ---       │
    # │ f64  ┆ f64       │
    # ╞══════╪═══════════╡
    # │ -2.0 ┆ 0.457658  │
    # │ -1.0 ┆ -0.642093 │
    # │ 0.0  ┆ inf       │
    # │ 1.0  ┆ 0.642093  │
    # │ 2.0  ┆ -0.457658 │
    # └──────┴───────────┘

.. _cotd:

COTD
----
Compute cotangent of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [0, 90, 180, 270, 360]})
    df.sql("SELECT degs, COTD(degs) AS cotd FROM self")
    # shape: (5, 2)
    # ┌──────┬────────────┐
    # │ degs ┆ cotd       │
    # │ ---  ┆ ---        │
    # │ f64  ┆ f64        │
    # ╞══════╪════════════╡
    # │ -2.0 ┆ -28.636253 │
    # │ -1.0 ┆ -57.289962 │
    # │ 0.0  ┆ inf        │
    # │ 1.0  ┆ 57.289962  │
    # │ 2.0  ┆ 28.636253  │
    # └──────┴────────────┘

.. _cos:

COS
---
Compute the cosine of the input column (in radians).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"rads": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    df.sql("SELECT rads, COS(rads) AS cos FROM self")
    # shape: (5, 2)
    # ┌──────┬───────────┐
    # │ rads ┆ cos       │
    # │ ---  ┆ ---       │
    # │ f64  ┆ f64       │
    # ╞══════╪═══════════╡
    # │ -2.0 ┆ -0.416147 │
    # │ -1.0 ┆ 0.540302  │
    # │ 0.0  ┆ 1.0       │
    # │ 1.0  ┆ 0.540302  │
    # │ 2.0  ┆ -0.416147 │
    # └──────┴───────────┘

.. _cosd:

COSD
----
Compute the cosine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [0, 45, 180, 225]})
    df.sql("SELECT degs, COSD(degs) AS cosd FROM self")
    # shape: (4, 2)
    # ┌──────┬───────────┐
    # │ degs ┆ cosd      │
    # │ ---  ┆ ---       │
    # │ i64  ┆ f64       │
    # ╞══════╪═══════════╡
    # │ 0    ┆ 1.0       │
    # │ 45   ┆ 0.707107  │
    # │ 180  ┆ -1.0      │
    # │ 225  ┆ -0.707107 │
    # └──────┴───────────┘

.. _degrees:

DEGREES
-------
Convert between radians and degrees.

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame({"rads": [0.0, math.pi/2, math.pi, 3*math.pi/2]})
    df.sql("SELECT rads, DEGREES(rads) AS degs FROM self")
    # shape: (4, 2)
    # ┌──────────┬───────┐
    # │ rads     ┆ degs  │
    # │ ---      ┆ ---   │
    # │ f64      ┆ f64   │
    # ╞══════════╪═══════╡
    # │ 0.0      ┆ 0.0   │
    # │ 1.570796 ┆ 90.0  │
    # │ 3.141593 ┆ 180.0 │
    # │ 4.712389 ┆ 270.0 │
    # └──────────┴───────┘

.. _radians:

RADIANS
-------
Convert between degrees and radians.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [0, 90, 180, 270]})
    df.sql("SELECT degs, RADIANS(degs) AS rads FROM self")
    # shape: (4, 2)
    # ┌──────┬──────────┐
    # │ degs ┆ rads     │
    # │ ---  ┆ ---      │
    # │ i64  ┆ f64      │
    # ╞══════╪══════════╡
    # │ 0    ┆ 0.0      │
    # │ 90   ┆ 1.570796 │
    # │ 180  ┆ 3.141593 │
    # │ 270  ┆ 4.712389 │
    # └──────┴──────────┘

.. _sin:

SIN
---
Compute the sine of the input column (in radians).

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame({"rads": [0.0, 1/4 * math.pi, 1/2 * math.pi, 3/4 * math.pi]})
    df.sql("SELECT rads, SIN(rads) AS sin FROM self")
    # shape: (4, 2)
    # ┌──────────┬──────────┐
    # │ rads     ┆ sin      │
    # │ ---      ┆ ---      │
    # │ f64      ┆ f64      │
    # ╞══════════╪══════════╡
    # │ 0.0      ┆ 0.0      │
    # │ 0.785398 ┆ 0.707107 │
    # │ 1.570796 ┆ 1.0      │
    # │ 2.356194 ┆ 0.707107 │
    # └──────────┴──────────┘

.. _sind:

SIND
----
Compute the sine of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [0, 90, 225, 270]})
    df.sql("SELECT degs, SIND(degs) AS sind FROM self")
    # shape: (4, 2)
    # ┌──────┬───────────┐
    # │ degs ┆ sind      │
    # │ ---  ┆ ---       │
    # │ i64  ┆ f64       │
    # ╞══════╪═══════════╡
    # │ 0    ┆ 0.0       │
    # │ 90   ┆ 1.0       │
    # │ 225  ┆ -0.707107 │
    # │ 270  ┆ -1.0      │
    # └──────┴───────────┘

.. _tan:

TAN
---
Compute the tangent of the input column (in radians).

**Example:**

.. code-block:: python

    import math

    df = pl.DataFrame({"rads": [0.0, 1/4 * math.pi, 3/4 * math.pi]})
    df.sql("SELECT rads, TAN(rads) AS tan FROM self")
    # shape: (4, 2)
    # ┌──────────┬───────────┐
    # │ rads     ┆ tan       │
    # │ ---      ┆ ---       │
    # │ f64      ┆ f64       │
    # ╞══════════╪═══════════╡
    # │ 0.0      ┆ 0.0       │
    # │ 0.785398 ┆ 1.0       │
    # │ 1.570796 ┆ 1.6331e16 │
    # │ 2.356194 ┆ -1.0      │
    # └──────────┴───────────┘


.. _tand:

TAND
----
Compute the tangent of the input column (in degrees).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"degs": [0, 45, 135, 225]})
    df.sql("SELECT degs, TAND(degs) AS tand FROM self")
    # shape: (4, 2)
    # ┌──────┬──────┐
    # │ degs ┆ tand │
    # │ ---  ┆ ---  │
    # │ i64  ┆ f64  │
    # ╞══════╪══════╡
    # │ 0    ┆ 0.0  │
    # │ 45   ┆ 1.0  │
    # │ 135  ┆ -1.0 │
    # │ 225  ┆ 1.0  │
    # └──────┴──────┘
