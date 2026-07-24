Array
=====

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
   * - :ref:`ARRAY_LOWER <array_lower>`
     - Returns the lower bound (min value) in an array.
   * - :ref:`ARRAY_MEAN <array_mean>`
     - Returns the mean of all values in an array.
   * - :ref:`ARRAY_REVERSE <array_reverse>`
     - Returns the array with the elements in reverse order.
   * - :ref:`ARRAY_SUM <array_sum>`
     - Returns the sum of all values in an array.
   * - :ref:`ARRAY_TO_STRING <array_to_string>`
     - Takes all elements of the array and joins them into one string.
   * - :ref:`ARRAY_UNIQUE <array_unique>`
     - Returns the array with the unique elements.
   * - :ref:`ARRAY_UPPER <array_upper>`
     - Returns the upper bound (max value) in an array.
   * - :ref:`UNNEST <unnest>`
     - Unnests (explodes) an array column into multiple rows.

.. _array_contains:

ARRAY_CONTAINS
--------------
Returns true if the array contains the value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [[1, 2], [4, 3]]})
    df.sql("""
      SELECT ARRAY_CONTAINS(foo, 2) FROM self
    """)
    # shape: (2, 1)
    # ┌───────┐
    # │ foo   │
    # │ ---   │
    # │ bool  │
    # ╞═══════╡
    # │ true  │
    # │ false │
    # └───────┘

.. _array_get:

ARRAY_GET
---------
Returns the value at the given index in the array.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    df.sql("""
      SELECT ARRAY_GET(foo, 1), ARRAY_GET(bar, 2) FROM self
    """)
    # shape: (2, 2)
    # ┌─────┬──────┐
    # │ foo ┆ bar  │
    # │ --- ┆ ---  │
    # │ i64 ┆ i64  │
    # ╞═════╪══════╡
    # │ 2   ┆ null │
    # │ 3   ┆ 10   │
    # └─────┴──────┘

.. _array_length:

ARRAY_LENGTH
------------
Returns the length of the array.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [[1, 2], [4, 3, 2]]})
    df.sql("""
      SELECT ARRAY_LENGTH(foo) FROM self
    """)
    # shape: (2, 1)
    # ┌─────┐
    # │ foo │
    # │ --- │
    # │ u32 │
    # ╞═════╡
    # │ 2   │
    # │ 3   │
    # └─────┘

.. _array_lower:

ARRAY_LOWER
-----------
Returns the lower bound (min value) in an array.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, -2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    df.sql("""
      SELECT ARRAY_LOWER(foo), ARRAY_LOWER(bar) FROM self
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 1   ┆ 6   │
    # │ -2  ┆ 8   │
    # └─────┴─────┘

.. _array_mean:

ARRAY_MEAN
----------
Returns the mean of all values in an array.

**Example:**

.. code-block:: python
    
    df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, -1]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    df.sql("""
      SELECT
        ARRAY_MEAN(foo) AS foo_mean,
        ARRAY_MEAN(bar) AS bar_mean
      FROM self
    """)

    # shape: (2, 2)
    # ┌──────────┬──────────┐
    # │ foo_mean ┆ bar_mean │
    # │ ---      ┆ ---      │
    # │ f64      ┆ f64      │
    # ╞══════════╪══════════╡
    # │ 1.5      ┆ 6.5      │
    # │ 2.0      ┆ 9.0      │
    # └──────────┴──────────┘

.. _array_reverse:

ARRAY_REVERSE
-------------
Returns the array with the elements in reverse order.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    df.sql("""
      SELECT
        ARRAY_REVERSE(foo) AS oof,
        ARRAY_REVERSE(bar) AS rab
      FROM self
    """)

    # shape: (2, 2)
    # ┌───────────┬────────────┐
    # │ oof       ┆ rab        │
    # │ ---       ┆ ---        │
    # │ list[i64] ┆ list[i64]  │
    # ╞═══════════╪════════════╡
    # │ [2, 1]    ┆ [7, 6]     │
    # │ [2, 3, 4] ┆ [10, 9, 8] │
    # └───────────┴────────────┘

.. _array_sum:

ARRAY_SUM
---------
Returns the sum of all values in an array.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [[1, -2], [-4, 3, -2]],
        "bar": [[-6, 7], [8, -9, 10]]
      }
    )
    df.sql("""
      SELECT
        ARRAY_SUM(foo) AS foo_sum,
        ARRAY_SUM(bar) AS bar_sum
      FROM self
    """)

    # shape: (2, 2)
    # ┌─────────┬─────────┐
    # │ foo_sum ┆ bar_sum │
    # │ ---     ┆ ---     │
    # │ i64     ┆ i64     │
    # ╞═════════╪═════════╡
    # │ -1      ┆ 1       │
    # │ -3      ┆ 9       │
    # └─────────┴─────────┘

.. _array_to_string:

ARRAY_TO_STRING
---------------
Takes all elements of the array and joins them into one string.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [["a", "b"], ["c", "d", "e"]]})
    df.sql("""
      SELECT ARRAY_TO_STRING(foo,',') AS foo_str FROM self
    """)
    # shape: (2, 1)
    # ┌─────────┐
    # │ foo_str │
    # │ ---     │
    # │ str     │
    # ╞═════════╡
    # │ a,b     │
    # │ c,d,e   │
    # └─────────┘

.. _array_unique:

ARRAY_UNIQUE
------------
Returns the array with the unique elements.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [["a", "b"], ["b", "b", "e"]]})
    df.sql("""
      SELECT ARRAY_UNIQUE(foo) AS foo_unique FROM self
    """)
    # shape: (2, 1)
    # ┌────────────┐
    # │ foo_unique │
    # │ ---        │
    # │ list[str]  │
    # ╞════════════╡
    # │ ["a", "b"] │
    # │ ["e", "b"] │
    # └────────────┘

.. _array_upper:

ARRAY_UPPER
-----------
Returns the upper bound (max value) in an array.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, -2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    df.sql("""
      SELECT
        ARRAY_UPPER(foo) AS foo_max,
        ARRAY_UPPER(bar) AS bar_max
      FROM self
    """)

    # shape: (2, 2)
    # ┌─────────┬─────────┐
    # │ foo_max ┆ bar_max │
    # │ ---     ┆ ---     │
    # │ i64     ┆ i64     │
    # ╞═════════╪═════════╡
    # │ 2       ┆ 7       │
    # │ 4       ┆ 10      │
    # └─────────┴─────────┘

.. _unnest:

UNNEST
------
Unnest/explode an array column into multiple rows.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [["a", "b"], ["c", "d", "e"]],
        "bar": [[6, 7, 8], [9, 10]]
      }
    )
    df.sql("""
      SELECT
        UNNEST(foo) AS f,
        UNNEST(bar) AS b
      FROM self
    """)

    # shape: (5, 2)
    # ┌─────┬─────┐
    # │ f   ┆ b   │
    # │ --- ┆ --- │
    # │ str ┆ i64 │
    # ╞═════╪═════╡
    # │ a   ┆ 6   │
    # │ b   ┆ 7   │
    # │ c   ┆ 8   │
    # │ d   ┆ 9   │
    # │ e   ┆ 10  │
    # └─────┴─────┘
