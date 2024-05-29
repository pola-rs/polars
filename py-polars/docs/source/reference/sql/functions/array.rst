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
   * - :ref:`UNNEST <unnest>`
     - Unnests/explodes an array column into multiple rows.

.. _array_contains:

ARRAY_CONTAINS
--------------
Returns true if the array contains the value.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3]],
        "bar": [[6, 7], [8, 9]]
      }
    )
    >>> df.sql("SELECT ARRAY_CONTAINS(foo, 2) FROM self")
    shape: (2, 1)
    ┌───────┐
    │ foo   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    └───────┘

.. _array_get:

ARRAY_GET
---------
Returns the value at the given index in the array.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_GET(foo, 1) FROM self")
    shape: (2, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 2   │
    │ 3   │
    └─────┘

.. _array_length:

ARRAY_LENGTH
------------
Returns the length of the array.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_LENGTH(foo) FROM self")
    shape: (2, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 2   │
    │ 3   │
    └─────┘

.. _array_max:

ARRAY_MAX
---------
Returns the maximum value in an array; equivalent to `array_max`.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_UPPER(bar) FROM self")
    shape: (2, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 7   │
    │ 10  │
    └─────┘

.. _array_mean:

ARRAY_MEAN
----------
Returns the mean of all values in an array.

**Example:**

.. code-block:: python
    
    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_MEAN(foo), ARRAY_MEAN(bar) FROM self")
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ f64 ┆ f64 │
    ╞═════╪═════╡
    │ 1.5 ┆ 6.5 │
    │ 3.0 ┆ 9.0 │
    └─────┴─────┘

.. _array_min:

ARRAY_MIN
---------
Returns the minimum value in an array; equivalent to `array_min`.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_LENGTH(foo) FROM self")
    shape: (2, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 6   │
    │ 8   │
    └─────┘

.. _array_reverse:

ARRAY_REVERSE
-------------
Returns the array with the elements in reverse order.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_REVERSE(bar) FROM self")
    shape: (2, 1)
    ┌────────────┐
    │ bar        │
    │ ---        │
    │ list[i64]  │
    ╞════════════╡
    │ [7, 6]     │
    │ [10, 9, 8] │
    └────────────┘

.. _array_sum:

ARRAY_SUM
---------
Returns the sum of all values in an array.

**Example:**

.. code-block:: python

    >>> df = pl.DataFrame(
      {
        "foo": [[1, 2], [4, 3, 2]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_SUM(bar) FROM self")
    shape: (2, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 13  │
    │ 27  │
    └─────┘

.. _array_to_string:

ARRAY_TO_STRING
---------------
Takes all elements of the array and joins them into one string.

**Example:**

.. code-block:: python

   >>> df = pl.DataFrame(
      {
        "foo": [["a", "b"], ["c", "d", "e"]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_TO_STRING(foo, ',') FROM self")
    shape: (2, 1)
    ┌───────┐
    │ foo   │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ a,b   │
    │ c,d,e │
    └───────┘

.. _array_unique:

ARRAY_UNIQUE
------------
Returns the array with the unique elements.

**Example:**

.. code-block:: python

   >>> df = pl.DataFrame(
      {
        "foo": [["a", "b"], ["b", "b", "e"]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT ARRAY_UNIQUE(foo) FROM self")
    shape: (2, 1)
    ┌────────────┐
    │ foo        │
    │ ---        │
    │ list[str]  │
    ╞════════════╡
    │ ["b", "a"] │
    │ ["b", "e"] │
    └────────────┘

.. _unnest:

UNNEST
------
Unnests/explodes an array column into multiple rows.

**Example:**

.. code-block:: python

   >>> df = pl.DataFrame(
      {
        "foo": [["a", "b"], ["b", "b", "e"]],
        "bar": [[6, 7], [8, 9, 10]]
      }
    )
    >>> df.sql("SELECT UNNEST(foo) FROM self")
    shape: (5, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ b   │
    │ b   │
    │ e   │
    └─────┘


