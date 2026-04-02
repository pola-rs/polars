Operators
=========

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Category
     - Description
   * - :ref:`Logical <op_logical>`
     - Boolean logic operators (AND, OR, NOT).
   * - :ref:`Comparison <op_comparison>`
     - Value comparison and membership tests (=, >, <, IS, BETWEEN, IN, ALL, ANY).
   * - :ref:`Numeric <op_numeric>`
     - Arithmetic operators (+, -, \*, /, //, %, unary).
   * - :ref:`Bitwise <op_bitwise>`
     - Bitwise integer operators (&, \|, XOR).
   * - :ref:`String <op_string>`
     - String matching and concatenation (LIKE, ILIKE, ||, ^@).
   * - :ref:`Regex <op_regex>`
     - Regular expression matching (~, ~\*, REGEXP, RLIKE).
   * - :ref:`Indexing <op_indexing>`
     - Struct field and array element access (->, ->>, #>, [n]).


.. _op_logical:

Logical
-------

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Operator
     - Description
   * - :ref:`AND <op_logical_and>`
     - Combines conditions, returning True if both conditions are True.
   * - :ref:`OR <op_logical_or>`
     - Combines conditions, returning True if either condition is True.
   * - :ref:`NOT <op_logical_not>`
     - Negates a condition, returning True if the condition is False.


.. _op_logical_and:

`AND`: Combines conditions, returning True if both conditions are True.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [10, 20, 50, 35], "bar": [10, 10, 50, 35]})
    df.sql("""
      SELECT * FROM self
      WHERE (foo >= bar) AND (bar < 50) AND (foo != 10)
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 20  ┆ 10  │
    # │ 35  ┆ 35  │
    # └─────┴─────┘


.. _op_logical_or:

`OR`: Combines conditions, returning True if either condition is True.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [10, 20, 50, 35], "bar": [10, 10, 50, 35]})
    df.sql("""
      SELECT * FROM self
      WHERE (foo % 2 != 0) OR (bar > 40)
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 50  ┆ 50  │
    # │ 35  ┆ 35  │
    # └─────┴─────┘


.. _op_logical_not:

`NOT`: Negates a condition, returning True if the condition is False.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"foo": [10, 20, 50, 35], "bar": [10, 10, 50, 35]})
    df.sql("""
      SELECT * FROM self
      WHERE NOT(foo % 2 != 0 OR bar > 40)
    """)
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ foo ┆ bar │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 10  ┆ 10  │
    # │ 20  ┆ 10  │
    # └─────┴─────┘


.. _op_comparison:

Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`>, >= <op_greater_than>`
     - Greater than (or equal to).
   * - :ref:`\<, \<= <op_less_than>`
     - Less than (or equal to).
   * - :ref:`==, !=, \<> <op_equal_to>`
     - Equal to (or not equal to).
   * - :ref:`IS [NOT] <op_is>`
     - Check if a value is one of the special values NULL, TRUE, or FALSE.
   * - :ref:`IS [NOT] DISTINCT FROM, \<=> <op_is_distinct_from>`
     - Check if a value is distinct from another value; this is the NULL-safe equivalent of `==` (and `!=`).
   * - :ref:`[NOT] BETWEEN <op_between>`
     - Check if a value is between two other values.
   * - :ref:`[NOT] IN <op_in>`
     - Check if a value is present in a list of values.
   * - :ref:`> ALL, \< ALL, ... <op_all>`
     - Compare a value against all values in a column.
   * - :ref:`> ANY, \< ANY, = ANY, ... <op_any>`
     - Compare a value against any value in a column.


.. _op_greater_than:

Returns True if the first value is greater than the second value.

| `>`: Greater than.
| `>=`: Greater than or equal to.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 3000, 5000]})
    df.sql("SELECT * FROM self WHERE n > 3000")
    # shape: (1, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 5000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n >= 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 3000 │
    # │ 5000 │
    # └──────┘


.. _op_less_than:

Returns True if the first value is less than the second value.

| `<`: Less than.
| `<=`: Less than or equal to.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 3000, 5000]})
    df.sql("SELECT * FROM self WHERE n < 3000")
    # shape: (1, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n <= 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # │ 3000 │
    # └──────┘


.. _op_equal_to:

Returns True if the two values are considered equal.

| `=`: Equal to.
| `!=`, `<>`: Not equal to.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 3000, 5000]})
    df.sql("SELECT * FROM self WHERE n = 3000")
    # shape: (1, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 3000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n != 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # │ 5000 │
    # └──────┘


.. _op_is:

`IS`: Returns True if the first value is identical to the second value (typically one of NULL, TRUE, or FALSE).
Unlike `==` (and `!=`), this operator will always return TRUE or FALSE, never NULL.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"lbl": ["aa", "bb", "cc"], "n": [1000, None, 5000]})
    df.sql("SELECT * FROM self WHERE n IS NULL")
    # shape: (1, 2)
    # ┌─────┬──────┐
    # │ lbl ┆ n    │
    # │ --- ┆ ---  │
    # │ str ┆ i64  │
    # ╞═════╪══════╡
    # │ bb  ┆ null │
    # └─────┴──────┘

    df.sql("SELECT * FROM self WHERE n IS NOT NULL")
    # shape: (2, 2)
    # ┌─────┬──────┐
    # │ lbl ┆ n    │
    # │ --- ┆ ---  │
    # │ str ┆ i64  │
    # ╞═════╪══════╡
    # │ aa  ┆ 1000 │
    # │ cc  ┆ 5000 │
    # └─────┴──────┘


.. _op_is_distinct_from:

| `IS DISTINCT FROM`: Compare two values; this operator assumes that NULL values are equal.
| The inverse, `IS NOT DISTINCT FROM`, can also be written using the `<=>` operator.
| Equivalent to `==` (or `!=`) for non-NULL values.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n1": [2222, None, 8888], "n2": [4444, None, 8888]})
    df.sql("SELECT * FROM self WHERE n1 IS DISTINCT FROM n2")
    # shape: (1, 2)
    # ┌──────┬──────┐
    # │ n1   ┆ n2   │
    # │ ---  ┆ ---  │
    # │ i64  ┆ i64  │
    # ╞══════╪══════╡
    # │ 2222 ┆ 4444 │
    # └──────┴──────┘

    df.sql("SELECT * FROM self WHERE n1 IS NOT DISTINCT FROM n2")
    df.sql("SELECT * FROM self WHERE n1 <=> n2")
    # shape: (2, 2)
    # ┌──────┬──────┐
    # │ n1   ┆ n2   │
    # │ ---  ┆ ---  │
    # │ i64  ┆ i64  │
    # ╞══════╪══════╡
    # │ null ┆ null │
    # │ 8888 ┆ 8888 │
    # └──────┴──────┘


.. _op_between:

`BETWEEN`: Returns True if the first value is between the second and third values (inclusive).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1000, 2000, 3000, 4000]})
    df.sql("SELECT * FROM self WHERE n BETWEEN 2000 AND 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 2000 │
    # │ 3000 │
    # └──────┘

    df.sql("SELECT * FROM self WHERE n NOT BETWEEN 2000 AND 3000")
    # shape: (2, 1)
    # ┌──────┐
    # │ n    │
    # │ ---  │
    # │ i64  │
    # ╞══════╡
    # │ 1000 │
    # │ 4000 │
    # └──────┘


.. _op_in:

`IN`: Returns True if the value is found in the given list (or subquery result).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1, 2, 3, 4, 5]})
    df.sql("SELECT * FROM self WHERE n IN (2, 4)")
    # shape: (2, 1)
    # ┌─────┐
    # │ n   │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 2   │
    # │ 4   │
    # └─────┘

    df.sql("SELECT * FROM self WHERE n NOT IN (2, 4)")
    # shape: (3, 1)
    # ┌─────┐
    # │ n   │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 1   │
    # │ 3   │
    # │ 5   │
    # └─────┘


.. _op_all:

| `ALL`: Compare a value against *all* values in a column expression.
| Supported comparison operators: ``>``, ``<``, ``>=``, ``<=``.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 5, 10], "b": [3, 3, 3]})
    df.sql("SELECT * FROM self WHERE a > ALL(b)")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 5   ┆ 3   │
    # │ 10  ┆ 3   │
    # └─────┴─────┘


.. _op_any:

| `ANY`: Compare a value against *any* value in a column expression.
| Supported comparison operators: ``>``, ``<``, ``>=``, ``<=``, ``=``, ``!=``.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [1, 5, 10], "b": [3, 7, 3]})
    df.sql("SELECT * FROM self WHERE a < ANY(b)")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ a   ┆ b   │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 1   ┆ 3   │
    # │ 5   ┆ 7   │
    # └─────┴─────┘


.. _op_numeric:

Numeric
-------

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`+, - <op_add_sub>`
     - Addition and subtraction.
   * - :ref:`*, / <op_mul_div>`
     - Multiplication and division.
   * - :ref:`% <op_modulo>`
     - Modulo (remainder).
   * - :ref:`Unary +, - <op_unary>`
     - Unary plus and minus.


.. _op_add_sub:

| ``+``: Addition.
| ``-``: Subtraction.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [10, 20, 30], "b": [3, 7, 4]})
    df.sql("SELECT a + b AS sum, a - b AS diff FROM self")
    # shape: (3, 2)
    # ┌─────┬──────┐
    # │ sum ┆ diff │
    # │ --- ┆ ---  │
    # │ i64 ┆ i64  │
    # ╞═════╪══════╡
    # │ 13  ┆ 7    │
    # │ 27  ┆ 13   │
    # │ 34  ┆ 26   │
    # └─────┴──────┘


.. _op_mul_div:

| ``*``: Multiplication.
| ``/``: Division.
| ``//``: Integer division.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [10, 20, 30], "b": [5.55, 4.25, 3.33]})
    df.sql("""
        SELECT
          *,
          a * b AS mul,
          a / b AS div,
          a // b AS int_div
        FROM self
    """)
    # shape: (3, 2)
    # ┌─────┬─────┐
    # │ mul ┆ div │
    # │ --- ┆ --- │
    # │ i64 ┆ i64 │
    # ╞═════╪═════╡
    # │ 30  ┆ 3   │
    # │ 140 ┆ 2   │
    # │ 120 ┆ 7   │
    # └─────┴─────┘

.. _op_modulo:

``%``: Returns the remainder of the division.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [10, 20, 30], "b": [3, 7, 4]})
    df.sql("SELECT a % b AS mod FROM self")
    # shape: (3, 1)
    # ┌─────┐
    # │ mod │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ 1   │
    # │ 6   │
    # │ 2   │
    # └─────┘


.. _op_unary:

| ``+``: Unary plus (no-op; returns the value unchanged).
| ``-``: Unary minus (negates the value).

**Example:**

.. code-block:: python

    df = pl.DataFrame({"n": [1, -2, 3]})
    df.sql("SELECT -n AS neg FROM self")
    # shape: (3, 1)
    # ┌─────┐
    # │ neg │
    # │ --- │
    # │ i64 │
    # ╞═════╡
    # │ -1  │
    # │ 2   │
    # │ -3  │
    # └─────┘


.. _op_bitwise:

Bitwise
-------

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`& <op_bitwise_and>`
     - Bitwise AND.
   * - :ref:`| <op_bitwise_or>`
     - Bitwise OR.
   * - :ref:`XOR <op_bitwise_xor>`
     - Bitwise exclusive OR.


.. _op_bitwise_and:

``&``: Performs a bitwise AND on two integer values.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [12, 10], "b": [10, 6]})
    df.sql("SELECT a & b AS bw_and FROM self")
    # shape: (2, 1)
    # ┌────────┐
    # │ bw_and │
    # │ ---    │
    # │ i64    │
    # ╞════════╡
    # │ 8      │
    # │ 2      │
    # └────────┘


.. _op_bitwise_or:

``|``: Performs a bitwise OR on two integer values.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [12, 10], "b": [10, 6]})
    df.sql("SELECT a | b AS bw_or FROM self")
    # shape: (2, 1)
    # ┌───────┐
    # │ bw_or │
    # │ ---   │
    # │ i64   │
    # ╞═══════╡
    # │ 14    │
    # │ 14    │
    # └───────┘


.. _op_bitwise_xor:

``XOR``: Performs a bitwise exclusive OR on two integer values.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"a": [12, 10], "b": [10, 6]})
    df.sql("SELECT a XOR b AS bw_xor FROM self")
    # shape: (2, 1)
    # ┌────────┐
    # │ bw_xor │
    # │ ---    │
    # │ i64    │
    # ╞════════╡
    # │ 6      │
    # │ 12     │
    # └────────┘


.. _op_string:

String
------

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`[NOT] LIKE, ~~, !~~ <op_like>`
     - Matches a SQL "LIKE" expression.
   * - :ref:`[NOT] ILIKE, ~~*, !~~* <op_ilike>`
     - Matches a SQL "ILIKE" expression (case-insensitive "LIKE").
   * - :ref:`|| <op_concat>`
     - Concatenate two string values.
   * - :ref:`^@ <op_starts_with>`
     - Check if a string starts with a given prefix.

.. note::

    The ``LIKE`` operators match on string patterns, using the following wildcards:

    * The percent sign ``%`` represents zero or more characters.
    * An underscore ``_`` represents one character.


.. _op_like:

| ``LIKE``, ``~~``: Matches a SQL "LIKE" pattern.
| ``NOT LIKE``, ``!~~``: Does *not* match a SQL "LIKE" pattern.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"name": ["Alice", "Bob", "Charlie", "David"]})
    df.sql("SELECT * FROM self WHERE name LIKE 'A%'")
    # shape: (1, 1)
    # ┌───────┐
    # │ name  │
    # │ ---   │
    # │ str   │
    # ╞═══════╡
    # │ Alice │
    # └───────┘

    df.sql("SELECT * FROM self WHERE name NOT LIKE '%o%'")
    # shape: (3, 1)
    # ┌─────────┐
    # │ name    │
    # │ ---     │
    # │ str     │
    # ╞═════════╡
    # │ Alice   │
    # │ Charlie │
    # │ David   │
    # └─────────┘

Using the PostgreSQL-style operator syntax:

.. code-block:: python

    df.sql("SELECT * FROM self WHERE name ~~ '_o%'")
    # shape: (1, 1)
    # ┌──────┐
    # │ name │
    # │ ---  │
    # │ str  │
    # ╞══════╡
    # │ Bob  │
    # └──────┘


.. _op_ilike:

| ``ILIKE``, ``~~*``: Matches a SQL "LIKE" pattern, case insensitively.
| ``NOT ILIKE``, ``!~~*``: Does *not* match a SQL "LIKE" pattern, case insensitively.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"name": ["Alice", "bob", "CHARLIE", "David"]})
    df.sql("SELECT * FROM self WHERE name ILIKE '%LI%'")
    # shape: (2, 1)
    # ┌─────────┐
    # │ name    │
    # │ ---     │
    # │ str     │
    # ╞═════════╡
    # │ Alice   │
    # │ CHARLIE │
    # └─────────┘

    df.sql("SELECT * FROM self WHERE name NOT ILIKE 'b%'")
    # shape: (3, 1)
    # ┌─────────┐
    # │ name    │
    # │ ---     │
    # │ str     │
    # ╞═════════╡
    # │ Alice   │
    # │ CHARLIE │
    # │ David   │
    # └─────────┘


.. _op_concat:

``||``: Concatenate two or more string values.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"first": ["Alice", "Bob"], "last": ["Smith", "Jones"]})
    df.sql("SELECT first || ' ' || last AS full_name FROM self")
    # shape: (2, 1)
    # ┌─────────────┐
    # │ full_name   │
    # │ ---         │
    # │ str         │
    # ╞═════════════╡
    # │ Alice Smith │
    # │ Bob Jones   │
    # └─────────────┘

.. note::

    Non-string values are automatically cast to string when used with the ``||`` operator.


.. _op_starts_with:

``^@``: Returns True if the first string value starts with the second string value.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"name": ["Alice", "Bob", "Arthur", "David"]})
    df.sql("SELECT * FROM self WHERE name ^@ 'A'")
    # shape: (2, 1)
    # ┌────────┐
    # │ name   │
    # │ ---    │
    # │ str    │
    # ╞════════╡
    # │ Alice  │
    # │ Arthur │
    # └────────┘


.. _op_regex:

Regex
-----

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`~, !~ <op_match_regex>`
     - Match a regular expression.
   * - :ref:`~*, !~* <op_match_regex_ci>`
     - Match a regular expression, case insensitively.
   * - :ref:`[NOT] REGEXP, [NOT] RLIKE <op_rlike>`
     - Match a regular expression using keyword syntax.


.. _op_match_regex:

| `~`: Match a regular expression.
| `!~`: Does *not* match a regular expression.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"idx": [0, 1, 2, 3], "lbl": ["foo", "bar", "baz", "zap"]})
    df.sql("SELECT * FROM self WHERE lbl ~ '.*[aeiou]$'")
    # shape: (1, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ foo │
    # └─────┴─────┘

    df.sql("SELECT * FROM self WHERE lbl !~ '^b[aeiou]'")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ foo │
    # │ 3   ┆ zap │
    # └─────┴─────┘

.. _op_match_regex_ci:

| `~*`: Match a regular expression, case insensitively.
| `!~*`: Does *not* match a regular expression, case insensitively.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"idx": [0, 1, 2, 3], "lbl": ["FOO", "bar", "Baz", "ZAP"]})
    df.sql("SELECT * FROM self WHERE lbl ~* '.*[aeiou]$'")
    # shape: (1, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ FOO │
    # └─────┴─────┘

    df.sql("SELECT * FROM self WHERE lbl !~* '^b[aeiou]'")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 0   ┆ FOO │
    # │ 3   ┆ ZAP │
    # └─────┴─────┘

.. _op_rlike:

| ``REGEXP``, ``RLIKE``: Match a regular expression using keyword syntax.
| ``NOT REGEXP``, ``NOT RLIKE``: Does *not* match a regular expression.

These are equivalent to the ``~`` and ``!~`` operators, but use keyword syntax
instead of symbolic operators.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"idx": [0, 1, 2, 3], "lbl": ["foo", "bar", "baz", "zap"]})
    df.sql("SELECT * FROM self WHERE lbl REGEXP '^b'")
    # shape: (2, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 1   ┆ bar │
    # │ 2   ┆ baz │
    # └─────┴─────┘

    df.sql("SELECT * FROM self WHERE lbl NOT RLIKE '[aeiou]$'")
    # shape: (3, 2)
    # ┌─────┬─────┐
    # │ idx ┆ lbl │
    # │ --- ┆ --- │
    # │ i64 ┆ str │
    # ╞═════╪═════╡
    # │ 1   ┆ bar │
    # │ 2   ┆ baz │
    # │ 3   ┆ zap │
    # └─────┴─────┘


.. _op_indexing:

Indexing
--------

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Operator
     - Description
   * - :ref:`-> <op_arrow>`
     - Access a struct field by name or index, returning the native type.
   * - :ref:`->> <op_long_arrow>`
     - Access a struct field by name or index, returning a string.
   * - :ref:`#>, #>> <op_hash_arrow>`
     - Access a struct field by path, returning the native type or a string.
   * - :ref:`[n] <op_subscript>`
     - Access an array element by index (1-indexed).


.. _op_arrow:

``->``: Access a struct field by name or by ordinal index, returning the value in its native type.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {"data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    )
    df.sql("SELECT data -> 'name' AS name, data -> 'age' AS age FROM self")
    # shape: (2, 2)
    # ┌───────┬─────┐
    # │ name  ┆ age │
    # │ ---   ┆ --- │
    # │ str   ┆ i64 │
    # ╞═══════╪═════╡
    # │ Alice ┆ 30  │
    # │ Bob   ┆ 25  │
    # └───────┴─────┘


.. _op_long_arrow:

``->>``: Access a struct field by name or by ordinal index, returning the value cast to a string.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {"data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    )
    df.sql("SELECT data ->> 'age' AS age FROM self")
    # shape: (2, 1)
    # ┌─────┐
    # │ age │
    # │ --- │
    # │ str │
    # ╞═════╡
    # │ 30  │
    # │ 25  │
    # └─────┘


.. _op_hash_arrow:

| ``#>``: Access a struct field by path, returning the native type.
| ``#>>``: Access a struct field by path, returning a string.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
        {"data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    )
    df.sql("SELECT data #>> 'name' AS name FROM self")
    # shape: (2, 1)
    # ┌───────┐
    # │ name  │
    # │ ---   │
    # │ str   │
    # ╞═══════╡
    # │ Alice │
    # │ Bob   │
    # └───────┘


.. _op_subscript:

``[n]``: Access an array element by index. SQL array indices are 1-based.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"arr": [[10, 20, 30], [40, 50, 60]]})
    df.sql("SELECT arr[1] AS first, arr[3] AS last FROM self")
    # shape: (2, 2)
    # ┌───────┬──────┐
    # │ first ┆ last │
    # │ ---   ┆ ---  │
    # │ i64   ┆ i64  │
    # ╞═══════╪══════╡
    # │ 10    ┆ 30   │
    # │ 40    ┆ 60   │
    # └───────┴──────┘
