Math
====

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`ABS <abs>`
     - Returns the absolute value of the input column.
   * - :ref:`CBRT <cbrt>`
     - Returns the cube root (∛) of a number.
   * - :ref:`CEIL <ceil>`
     - Returns the nearest integer closest from zero.
   * - :ref:`EXP <exp>`
     - Computes the exponential of the given value.
   * - :ref:`FLOOR <floor_function>`
     - Returns the nearest integer away from zero.
   * - :ref:`LN <ln>`
     - Computes the natural logarithm of the given value.
   * - :ref:`LOG <log>`
     - Computes the `base` logarithm of the given value.
   * - :ref:`LOG2 <log2>`
     - Computes the logarithm of the given value in base 2.
   * - :ref:`LOG10 <log10>`
     - Computes the logarithm of the given value in base 10.
   * - :ref:`LOG1P <log1p>`
     - Computes the natural logarithm of "given value plus one".
   * - :ref:`MOD <mod>`
     - Returns the remainder of a numeric expression divided by another numeric expression.
   * - :ref:`PI <pi>`
     - Returns a (very good) approximation of 𝜋.
   * - :ref:`POW <pow>`
     - Returns the value to the power of the given exponent.
   * - :ref:`ROUND <round>`
     - Round a number to `x` decimals (default: 0) away from zero.
   * - :ref:`SIGN <sign>`
     - Returns the sign of the argument as -1, 0, or +1.
   * - :ref:`SQRT <sqrt>`
     - Returns the square root (√) of a number.

.. _abs:

ABS
---
Returns the absolute value of the input column.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [-1.0, 0.0, 1.0, -2.0]})
    df.sql("""
      SELECT ABS(a) FROM self
    """)
    # shape: (4, 1)
    # ┌─────┐
    # │ a   │
    # │ --- │
    # │ f64 │
    # ╞═════╡
    # │ 1.0 │
    # │ 0.0 │
    # │ 1.0 │
    # │ 2.0 │
    # └─────┘

.. _cbrt:

CBRT
----
Returns the cube root (∛) of a number.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1.0, 2.0, 4.0]})
    df.sql("""
      SELECT CBRT(a) FROM self
    """)
    # shape: (3, 1)
    # ┌──────────┐
    # │ a        │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 1.0      │
    # │ 1.259921 │
    # │ 1.587401 │
    # └──────────┘

.. _ceil:

CEIL 
----
Returns the nearest integer closest from zero.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [0.1, 2.8, 4.30]})
    df.sql("""
      SELECT CEIL(a) FROM self
    """)
    # shape: (3, 1)
    # ┌─────┐
    # │ a   │
    # │ --- │
    # │ f64 │
    # ╞═════╡
    # │ 1.0 │
    # │ 3.0 │
    # │ 5.0 │
    # └─────┘

.. _exp:

EXP 
---
Computes the exponential of the given value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT EXP(a) FROM self
    """)
    # shape: (3, 1)
    # ┌──────────┐
    # │ a        │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 2.718282 │
    # │ 7.389056 │
    # │ 54.59815 │
    # └──────────┘

.. _floor_function:

FLOOR 
-----
Returns the nearest integer away from zero.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [0.1, 2.8, 4.30]})
    df.sql("""
      SELECT FLOOR(a) FROM self
    """)
    # shape: (3, 1)
    # ┌─────┐
    # │ a   │
    # │ --- │
    # │ f64 │
    # ╞═════╡
    # │ 0.0 │
    # │ 2.0 │
    # │ 4.0 │
    # └─────┘

.. _ln:

LN
--
Computes the natural logarithm of the given value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT LN(a) FROM self
    """)
    # shape: (3, 1)
    # ┌──────────┐
    # │ a        │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 0.0      │
    # │ 0.693147 │
    # │ 1.386294 │
    # └──────────┘

.. _log:

LOG
---
Computes the `base` logarithm of the given value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT LOG(a, 10) FROM self
    """)
    # shape: (3, 1)
    # ┌─────────┐
    # │ a       │
    # │ ---     │
    # │ f64     │
    # ╞═════════╡
    # │ 0.0     │
    # │ 0.30103 │
    # │ 0.60206 │
    # └─────────┘

.. _log2:

LOG2 
----
Computes the logarithm of the given value in base 2.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT LOG2(a) FROM self
    """)
    # shape: (3, 1)
    # ┌─────┐
    # │ a   │
    # │ --- │
    # │ f64 │
    # ╞═════╡
    # │ 0.0 │
    # │ 1.0 │
    # │ 2.0 │
    # └─────┘

.. _log10:

LOG10
-----
Computes the logarithm of the given value in base 10.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT LOG10(a) FROM self
    """)
    # shape: (3, 1)
    # ┌─────────┐
    # │ a       │
    # │ ---     │
    # │ f64     │
    # ╞═════════╡
    # │ 0.0     │
    # │ 0.30103 │
    # │ 0.60206 │
    # └─────────┘

.. _log1p:

LOG1P
-----
Computes the natural logarithm of "given value plus one".

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT LOG1P(a) FROM self
    """)
    # shape: (3, 1)
    # ┌──────────┐
    # │ a        │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 0.693147 │
    # │ 1.098612 │
    # │ 1.609438 │
    # └──────────┘

.. _mod:

MOD
---
Returns the remainder of a numeric expression divided by another numeric expression.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
    df.sql("""
      SELECT MOD(x, 2) FROM self
    """)
    # shape: (5, 1)
    # ┌─────┐
    # │ x   │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 0   │
    # │ 1   │
    # │ 0   │
    # │ 1   │
    # │ 0   │
    # └─────┘

.. _pi:

PI
--
Returns a (good) approximation of 𝜋.

**Example:**

.. code-block:: python

    df.sql("""
      SELECT PI() AS pi FROM self
    """)
    # shape: (1, 1)
    # ┌──────────┐
    # │ pi       │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 3.141593 │
    # └──────────┘

.. _pow:

POW
---
Returns the value to the power of the given exponent.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
    df.sql("""
      SELECT POW(x, 2) FROM self
    """)
    # shape: (5, 1)
    # ┌─────┐
    # │ x   │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 0   │
    # │ 1   │
    # │ 4   │
    # │ 9   │
    # │ 16  │
    # └─────┘

.. _round:

ROUND
-----
Round a number to `x` decimals (default: 0) away from zero.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0.4, 1.8, 2.2, 3.6, 4.1]})
    df.sql("""
      SELECT ROUND(x) FROM self
    """)
    # shape: (5, 1)
    # ┌─────┐
    # │ x   │
    # │ --- │
    # │ f64 │
    # ╞═════╡
    # │ 0.0 │
    # │ 2.0 │
    # │ 2.0 │
    # │ 4.0 │
    # │ 4.0 │
    # └─────┘

.. _sign:

SIGN
----
Returns the sign of the argument as -1, 0, or +1.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0.4, -1, 0, -2, 4]})
    df.sql("""
      SELECT SIGN(x) FROM self
    """)
    # shape: (5, 1)
    # ┌─────┐
    # │ x   │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 1   │
    # │ -1  │
    # │ 0   │
    # │ -1  │
    # │ 1   │
    # └─────┘

.. _sqrt:

SQRT
----
Returns the square root (√) of a number.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [2, 4, 49, 64]})
    df.sql("""
      SELECT SQRT(x) FROM self
    """)
    # shape: (4, 1)
    # ┌──────────┐
    # │ x        │
    # │ ---      │
    # │ f64      │
    # ╞══════════╡
    # │ 1.414214 │
    # │ 2.0      │
    # │ 7.0      │
    # │ 8.0      │
    # └──────────┘
