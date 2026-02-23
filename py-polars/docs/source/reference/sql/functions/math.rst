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
   * - :ref:`DIV <div>`
     - Returns the integer quotient of the division.
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
     - Round a number to `n` decimals (default: 0) away from zero.
   * - :ref:`SIGN <sign>`
     - Returns the sign of the argument as -1, 0, or +1.
   * - :ref:`SQRT <sqrt>`
     - Returns the square root (√) of a number.
   * - :ref:`TRUNC <trunc>`
     - Truncate a number toward zero to `n` decimals (default: 0).

.. _abs:

ABS
---
Returns the absolute value of the input column.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [-1.0, 0.0, 1.0, -2.0]})
    df.sql("""
      SELECT a, ABS(a) AS abs_a FROM self
    """)
    # shape: (4, 2)
    # ┌──────┬───────┐
    # │ a    ┆ abs_a │
    # │ ---  ┆ ---   │
    # │ f64  ┆ f64   │
    # ╞══════╪═══════╡
    # │ -1.0 ┆ 1.0   │
    # │ 0.0  ┆ 0.0   │
    # │ 1.0  ┆ 1.0   │
    # │ -2.0 ┆ 2.0   │
    # └──────┴───────┘

.. _cbrt:

CBRT
----
Returns the cube root (∛) of a number.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1.0, 2.0, 4.0]})
    df.sql("""
      SELECT a, CBRT(a) AS cbrt_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬──────────┐
    # │ a   ┆ cbrt_a   │
    # │ --- ┆ ---      │
    # │ f64 ┆ f64      │
    # ╞═════╪══════════╡
    # │ 1.0 ┆ 1.0      │
    # │ 2.0 ┆ 1.259921 │
    # │ 4.0 ┆ 1.587401 │
    # └─────┴──────────┘

.. _ceil:

CEIL
----
Returns the nearest integer closest from zero.

.. admonition:: Aliases

   `CEILING`

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [0.1, 2.8, 4.30]})
    df.sql("""
      SELECT a, CEIL(a) AS ceil_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬────────┐
    # │ a   ┆ ceil_a │
    # │ --- ┆ ---    │
    # │ f64 ┆ f64    │
    # ╞═════╪════════╡
    # │ 0.1 ┆ 1.0    │
    # │ 2.8 ┆ 3.0    │
    # │ 4.3 ┆ 5.0    │
    # └─────┴────────┘

.. _div:

DIV
---
Returns the integer quotient of the division.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [-10.0, 6.5, 25.0]})
    df.sql("""
      SELECT a, DIV(a, 2) AS a_div_2, DIV(a, 5) AS a_div_5 FROM self
    """)
    # shape: (3, 3)
    # ┌───────┬─────────┬─────────┐
    # │ a     ┆ a_div_2 ┆ a_div_5 │
    # │ ---   ┆ ---     ┆ ---     │
    # │ f64   ┆ i64     ┆ i64     │
    # ╞═══════╪═════════╪═════════╡
    # │ -10.0 ┆ -5      ┆ -2      │
    # │ 6.5   ┆ 3       ┆ 1       │
    # │ 25.0  ┆ 12      ┆ 5       │
    # └───────┴─────────┴─────────┘

.. _exp:

EXP
---
Computes the exponential of the given value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT a, EXP(a) AS exp_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬──────────┐
    # │ a   ┆ exp_a    │
    # │ --- ┆ ---      │
    # │ i64 ┆ f64      │
    # ╞═════╪══════════╡
    # │ 1   ┆ 2.718282 │
    # │ 2   ┆ 7.389056 │
    # │ 4   ┆ 54.59815 │
    # └─────┴──────────┘

.. _floor_function:

FLOOR
-----
Returns the nearest integer away from zero.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [0.1, 2.8, 4.30]})
    df.sql("""
      SELECT a, FLOOR(a) AS floor_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬─────────┐
    # │ a   ┆ floor_a │
    # │ --- ┆ ---     │
    # │ f64 ┆ f64     │
    # ╞═════╪═════════╡
    # │ 0.1 ┆ 0.0     │
    # │ 2.8 ┆ 2.0     │
    # │ 4.3 ┆ 4.0     │
    # └─────┴─────────┘

.. _ln:

LN
--
Computes the natural logarithm of the given value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT a, LN(a) AS ln_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬──────────┐
    # │ a   ┆ ln_a     │
    # │ --- ┆ ---      │
    # │ i64 ┆ f64      │
    # ╞═════╪══════════╡
    # │ 1   ┆ 0.0      │
    # │ 2   ┆ 0.693147 │
    # │ 4   ┆ 1.386294 │
    # └─────┴──────────┘

.. _log:

LOG
---
Computes the `base` logarithm of the given value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT a, LOG(a, 16) AS log16_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬─────────┐
    # │ a   ┆ log16_a │
    # │ --- ┆ ---     │
    # │ i64 ┆ f64     │
    # ╞═════╪═════════╡
    # │ 1   ┆ 0.0     │
    # │ 2   ┆ 0.25    │
    # │ 4   ┆ 0.5     │
    # └─────┴─────────┘

.. _log2:

LOG2
----
Computes the logarithm of the given value in base 2.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT a, LOG2(a) AS a_log2 FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬────────┐
    # │ a   ┆ a_log2 │
    # │ --- ┆ ---    │
    # │ i64 ┆ f64    │
    # ╞═════╪════════╡
    # │ 1   ┆ 0.0    │
    # │ 2   ┆ 1.0    │
    # │ 4   ┆ 2.0    │
    # └─────┴────────┘

.. _log10:

LOG10
-----
Computes the logarithm of the given value in base 10.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT a, LOG10(a) AS log10_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬─────────┐
    # │ a   ┆ log10_a │
    # │ --- ┆ ---     │
    # │ i64 ┆ f64     │
    # ╞═════╪═════════╡
    # │ 1   ┆ 0.0     │
    # │ 2   ┆ 0.30103 │
    # │ 4   ┆ 0.60206 │
    # └─────┴─────────┘

.. _log1p:

LOG1P
-----
Computes the natural logarithm of "given value plus one".

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 2, 4]})
    df.sql("""
      SELECT a, LOG1P(a) AS log1p_a FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬──────────┐
    # │ a   ┆ log1p_a  │
    # │ --- ┆ ---      │
    # │ i64 ┆ f64      │
    # ╞═════╪══════════╡
    # │ 1   ┆ 0.693147 │
    # │ 2   ┆ 1.098612 │
    # │ 4   ┆ 1.609438 │
    # └─────┴──────────┘

.. _mod:

MOD
---
Returns the remainder of a numeric expression divided by another numeric expression.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
    df.sql("""
      SELECT x, MOD(x, 2) AS a_mod_2 FROM self
    """)
    # shape: (5, 2)
    # ┌─────┬─────────┐
    # │ x   ┆ a_mod_2 │
    # │ --- ┆ ---     │
    # │ i64 ┆ i64     │
    # ╞═════╪═════════╡
    # │ 0   ┆ 0       │
    # │ 1   ┆ 1       │
    # │ 2   ┆ 0       │
    # │ 3   ┆ 1       │
    # │ 4   ┆ 0       │
    # └─────┴─────────┘

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

.. admonition:: Aliases

   `POWER`

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0, 1, 2, 4]})
    df.sql("""
      SELECT x, POW(x, 8) AS x_pow_8 FROM self
    """)
    # shape: (4, 2)
    # ┌─────┬─────────┐
    # │ x   ┆ x_pow_8 │
    # │ --- ┆ ---     │
    # │ i64 ┆ i64     │
    # ╞═════╪═════════╡
    # │ 0   ┆ 0       │
    # │ 1   ┆ 1       │
    # │ 2   ┆ 256     │
    # │ 4   ┆ 65536   │
    # └─────┴─────────┘

.. _round:

ROUND
-----
Round a number to `n` decimals (default: 0) away from zero.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [-0.45, -1.81, 2.25, 3.99]})
    df.sql("""
      SELECT x, ROUND(x) AS x_round, ROUND(x, 1) AS x_round_1 FROM self
    """)
    # shape: (4, 3)
    # ┌───────┬─────────┬───────────┐
    # │ x     ┆ x_round ┆ x_round_1 │
    # │ ---   ┆ ---     ┆ ---       │
    # │ f64   ┆ f64     ┆ f64       │
    # ╞═══════╪═════════╪═══════════╡
    # │ -0.45 ┆ -0.0    ┆ -0.5      │
    # │ -1.81 ┆ -2.0    ┆ -1.8      │
    # │ 2.25  ┆ 2.0     ┆ 2.3       │
    # │ 3.99  ┆ 4.0     ┆ 4.0       │
    # └───────┴─────────┴───────────┘

.. _sign:

SIGN
----
Returns the sign of the argument as -1, 0, or +1.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [0.4, -1, 0, -2, 4]})
    df.sql("""
      SELECT x, SIGN(x) AS sign_x FROM self
    """)
    # shape: (5, 2)
    # ┌──────┬────────┐
    # │ x    ┆ sign_x │
    # │ ---  ┆ ---    │
    # │ f64  ┆ i64    │
    # ╞══════╪════════╡
    # │ 0.4  ┆ 1      │
    # │ -1.0 ┆ -1     │
    # │ 0.0  ┆ 0      │
    # │ -2.0 ┆ -1     │
    # │ 4.0  ┆ 1      │
    # └──────┴────────┘

.. _sqrt:

SQRT
----
Returns the square root (√) of a number.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [2, 16, 4096, 65536]})
    df.sql("""
      SELECT x, SQRT(x) AS sqrt_x FROM self
    """)
    # shape: (4, 2)
    # ┌───────┬──────────┐
    # │ x     ┆ sqrt_x   │
    # │ ---   ┆ ---      │
    # │ i64   ┆ f64      │
    # ╞═══════╪══════════╡
    # │ 2     ┆ 1.414214 │
    # │ 16    ┆ 4.0      │
    # │ 4096  ┆ 64.0     │
    # │ 65536 ┆ 256.0    │
    # └───────┴──────────┘

.. _trunc:

TRUNC
-----
Truncate a number toward zero to `n` decimals (default: 0).

.. admonition:: Aliases

   `TRUNCATE`

**Example:**

.. code-block:: python

    df = pl.DataFrame({"x": [1.5671, -2.999, 3.4569, -4.3218]})
    df.sql("""
      SELECT x, TRUNC(x) AS x_trunc, TRUNC(x, 2) AS x_trunc_1 FROM self
    """)
    # shape: (4, 3)
    # ┌─────────┬─────────┬───────────┐
    # │ x       ┆ x_trunc ┆ x_trunc_1 │
    # │ ---     ┆ ---     ┆ ---       │
    # │ f64     ┆ f64     ┆ f64       │
    # ╞═════════╪═════════╪═══════════╡
    # │ 1.5671  ┆ 1.0     ┆ 1.56      │
    # │ -2.999  ┆ -2.0    ┆ -2.99     │
    # │ 3.4569  ┆ 3.0     ┆ 3.45      │
    # │ -4.3218 ┆ -4.0    ┆ -4.32     │
    # └─────────┴─────────┴───────────┘
