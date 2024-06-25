Array
=====

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`ARRAY_AGG <array_agg>`
     - Aggregate a column/expression values as an array.
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


.. _array_agg:

ARRAY_AGG
---------
Aggregate a column/expression as an array (equivalent to `implode`).

Supports optional inline `ORDER BY` and `LIMIT` clauses.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    df.sql("""
      SELECT
        ARRAY_AGG(foo ORDER BY foo DESC) AS arr_foo,
        ARRAY_AGG(bar LIMIT 2) AS arr_bar
      FROM self
    """)
    # shape: (1, 2)
    # ┌───────────┬───────────┐
    # │ arr_foo   ┆ arr_bar   │
    # │ ---       ┆ ---       │
    # │ list[i64] ┆ list[i64] │
    # ╞═══════════╪═══════════╡
    # │ [3, 2, 1] ┆ [4, 5]    │
    # └───────────┴───────────┘

.. _array_contains:

ARRAY_CONTAINS
--------------
Returns true if the array contains the value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [[1, 2], [4, 3]]})
    df.sql("""
      SELECT foo, ARRAY_CONTAINS(foo, 2) AS has_two FROM self
    """)
    # shape: (2, 2)
    # ┌───────────┬─────────┐
    # │ foo       ┆ has_two │
    # │ ---       ┆ ---     │
    # │ list[i64] ┆ bool    │
    # ╞═══════════╪═════════╡
    # │ [1, 2]    ┆ true    │
    # │ [4, 3]    ┆ false   │
    # └───────────┴─────────┘

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
      SELECT
        foo, bar,
        ARRAY_GET(foo, 1) AS foo_at_1,
        ARRAY_GET(bar, 3) AS bar_at_2
      FROM self
    """)
    # shape: (2, 4)
    # ┌───────────┬────────────┬──────────┬──────────┐
    # │ foo       ┆ bar        ┆ foo_at_1 ┆ bar_at_2 │
    # │ ---       ┆ ---        ┆ ---      ┆ ---      │
    # │ list[i64] ┆ list[i64]  ┆ i64      ┆ i64      │
    # ╞═══════════╪════════════╪══════════╪══════════╡
    # │ [1, 2]    ┆ [6, 7]     ┆ 1        ┆ null     │
    # │ [4, 3, 2] ┆ [8, 9, 10] ┆ 4        ┆ 10       │
    # └───────────┴────────────┴──────────┴──────────┘

.. _array_length:

ARRAY_LENGTH
------------
Returns the length of the array.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [[1, 2], [4, 3, 2]]})
    df.sql("""
      SELECT foo, ARRAY_LENGTH(foo) AS n_elems FROM self
    """)
    # shape: (2, 2)
    # ┌───────────┬─────────┐
    # │ foo       ┆ n_elems │
    # │ ---       ┆ ---     │
    # │ list[i64] ┆ u32     │
    # ╞═══════════╪═════════╡
    # │ [1, 2]    ┆ 2       │
    # │ [4, 3, 2] ┆ 3       │
    # └───────────┴─────────┘

.. _array_lower:

ARRAY_LOWER
-----------
Returns the lower bound (min value) in an array.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [[1, 2], [4, -2, 8]]})
    df.sql("""
      SELECT foo, ARRAY_LOWER(foo) AS min_elem FROM self
    """)
    # shape: (2, 2)
    # ┌────────────┬──────────┐
    # │ foo        ┆ min_elem │
    # │ ---        ┆ ---      │
    # │ list[i64]  ┆ i64      │
    # ╞════════════╪══════════╡
    # │ [1, 2]     ┆ 1        │
    # │ [4, -2, 8] ┆ -2       │
    # └────────────┴──────────┘

.. _array_mean:

ARRAY_MEAN
----------
Returns the mean of all values in an array.

**Example:**

.. code-block:: python
    
    df = pl.DataFrame({"foo": [[1, 2], [4, 3, -1]]})
    df.sql("""
      SELECT foo, ARRAY_MEAN(foo) AS foo_mean FROM self
    """)
    # shape: (2, 2)
    # ┌────────────┬──────────┐
    # │ foo        ┆ foo_mean │
    # │ ---        ┆ ---      │
    # │ list[i64]  ┆ f64      │
    # ╞════════════╪══════════╡
    # │ [1, 2]     ┆ 1.5      │
    # │ [4, 3, -1] ┆ 2.0      │
    # └────────────┴──────────┘

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
        foo,
        ARRAY_REVERSE(foo) AS oof,
        ARRAY_REVERSE(bar) AS rab
      FROM self
    """)
    # shape: (2, 3)
    # ┌───────────┬───────────┬────────────┐
    # │ foo       ┆ oof       ┆ rab        │
    # │ ---       ┆ ---       ┆ ---        │
    # │ list[i64] ┆ list[i64] ┆ list[i64]  │
    # ╞═══════════╪═══════════╪════════════╡
    # │ [1, 2]    ┆ [2, 1]    ┆ [7, 6]     │
    # │ [4, 3, 2] ┆ [2, 3, 4] ┆ [10, 9, 8] │
    # └───────────┴───────────┴────────────┘

.. _array_sum:

ARRAY_SUM
---------
Returns the sum of all values in an array.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [[1, -2], [10, 3, -2]]})
    df.sql("""
      SELECT
        foo,
        ARRAY_SUM(foo) AS foo_sum
      FROM self
    """)
    # shape: (2, 2)
    # ┌─────────────┬─────────┐
    # │ foo         ┆ foo_sum │
    # │ ---         ┆ ---     │
    # │ list[i64]   ┆ i64     │
    # ╞═════════════╪═════════╡
    # │ [1, -2]     ┆ -1      │
    # │ [10, 3, -2] ┆ 11      │
    # └─────────────┴─────────┘

.. _array_to_string:

ARRAY_TO_STRING
---------------
Takes all elements of the array and joins them into one string.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {
            "foo": [["a", "b"], ["c", "d", "e"]],
            "bar": [[8, None, 8], [3, 2, 1, 0]],
        }
    )
    df.sql("""
      SELECT
        ARRAY_TO_STRING(foo,':') AS s_foo,
        ARRAY_TO_STRING(bar,':') AS s_bar
      FROM self
    """)
    # shape: (2, 2)
    # ┌───────┬─────────┐
    # │ s_foo ┆ s_bar   │
    # │ ---   ┆ ---     │
    # │ str   ┆ str     │
    # ╞═══════╪═════════╡
    # │ a:b   ┆ 8:8     │
    # │ c:d:e ┆ 3:2:1:0 │
    # └───────┴─────────┘

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

    df = pl.DataFrame({"foo": [[5, 0], [4, 8, -2]]})
    df.sql("""
      SELECT foo, ARRAY_UPPER(foo) AS max_elem FROM self
    """)
    # shape: (2, 2)
    # ┌────────────┬──────────┐
    # │ foo        ┆ max_elem │
    # │ ---        ┆ ---      │
    # │ list[i64]  ┆ i64      │
    # ╞════════════╪══════════╡
    # │ [5, 0]     ┆ 5        │
    # │ [4, 8, -2] ┆ 8        │
    # └────────────┴──────────┘

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
