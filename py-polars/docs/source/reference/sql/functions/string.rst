String
======

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - :ref:`BIT_LENGTH <bit_length>`
     - Returns the length of the input string in bits.
   * - :ref:`CONCAT <concat>`
     - Returns all input expressions concatenated together as a string.
   * - :ref:`CONCATWS <concat_ws>`
     - Returns all input expressions concatenated together (and interleaved with a separator) as a string.
   * - :ref:`ENDS_WITH <ends_with>`
     - Returns True if the value ends with the second argument.
   * - :ref:`INITCAP <initcap>`
     - Returns the value with the first letter capitalized.
   * - :ref:`LEFT <left>`
     - Returns the first (leftmost) `n` characters.
   * - :ref:`LENGTH <length>`
     - Returns the character length of the string.
   * - :ref:`LOWER <lower>`
     - Returns a lowercased column.
   * - :ref:`LTRIM <ltrim>`
     - Strips whitespaces from the left.
   * - :ref:`OCTET_LENGTH <octet_length>`
     - Returns the length of a given string in bytes.
   * - :ref:`REGEXP_LIKE <regexp_like>`
     - Returns True if `pattern` matches the value (optional: `flags`).
   * - :ref:`REPLACE <replace>`
     - Replaces a given substring with another string.
   * - :ref:`REVERSE <reverse>`
     - Returns the reversed string.
   * - :ref:`RIGHT <right>`
     - Returns the last (rightmost) `n` characters.
   * - :ref:`RTRIM <rtrim>`
     - Strips whitespaces from the right.
   * - :ref:`STARTS_WITH <starts_with>`
     - Returns True if the value starts with the second argument.
   * - :ref:`STRPOST <strpos>`
     - Returns the index of the given substring in the target string.
   * - :ref:`SUBSTRING <substring>`
     - Returns a portion of the data (first character = 0) in the range [start, start + length].
   * - :ref:`UPPER <upper>`
     - Returns an uppercased column.

.. _bit_length:

BIT_LENGTH
----------
Returns the length of the input string in bits.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
            "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT BIT_LENGTH(foo) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 16  │
    │ 8   │
    │ 8   │
    │ 16  │
    └─────┘

.. _concat:

CONCAT
------
Returns all input expressions concatenated together as a string.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT CONCAT(foo, bar) FROM self")
    shape: (4, 1)
    ┌──────┐
    │ foo  │
    │ ---  │
    │ str  │
    ╞══════╡
    │ aazz │
    │ byy  │
    │ cxx  │
    │ ddww │
    └──────┘

.. _concat_ws:

CONCAT_WS
---------
Returns all input expressions concatenated together (and interleaved with a separator) as a string.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT CONCAT_WS(',', foo, bar) FROM self")
    shape: (4, 1)
    ┌───────┐
    │ foo   │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ aa:zz │
    │ b:yy  │
    │ c:xx  │
    │ dd:ww │
    └───────┘

.. _ends_with:

ENDS_WITH
---------
Returns True if the value ends with the second argument.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql(""SELECT ENDS_WITH(foo, 'a') FROM self"")
    shape: (4, 1)
    ┌───────┐
    │ foo   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    │ false │
    │ false │
    └───────┘

.. _initcap:

INITCAP
-------
Returns the value with the first letter capitalized.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["aa", "b", "c", "dd"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT INITCAP(bar) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ str │
    ╞═════╡
    │ Zz  │
    │ Yy  │
    │ Xx  │
    │ Ww  │
    └─────┘

.. _left:

LEFT
----
Returns the first (leftmost) `n` characters.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["abcd", "efgh", "ijkl", "mnop"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT LEFT(foo, 2) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ ab  │
    │ ef  │
    │ ij  │
    │ mn  │
    └─────┘

.. _length:

LENGTH
------
Returns the character length of the string.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["ab", "efg", "i", "mnop"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT LENGTH(foo) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 2   │
    │ 3   │
    │ 1   │
    │ 4   │
    └─────┘

.. _lower:

LOWER
-----
Returns a lowercased column.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["AA", "BB", "CC", "DD"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT LOWER(foo) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ aa  │
    │ bb  │
    │ cc  │
    │ dd  │
    └─────┘

.. _ltrim:

LTRIM
-----
Strips whitespaces from the left.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["   AA", " BB", "CC", "  DD"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT LTRIM(foo) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ AA  │
    │ BB  │
    │ CC  │
    │ DD  │
    └─────┘

.. _octet_length:

OCTET_LENGTH
------------
Returns the length of a given string in bytes.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["AAa", "BB", "CCc", "DD"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT OCTET_LENGTH(foo) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 3   │
    │ 2   │
    │ 3   │
    │ 2   │
    └─────┘

.. _regexp_like:

REGEXP_LIKE
-----------
Returns True if `pattern` matches the value (optional: `flags`).

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["AA", "BB", "CC", "DD"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT REGEXP_LIKE(foo, 'BB') FROM self")
    shape: (4, 1)
    ┌───────┐
    │ foo   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ false │
    │ true  │
    │ false │
    │ false │
    └───────┘

.. _replace:

REPLACE
-------
Replaces a given substring with another string.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["AA", "BB", "CC", "DD"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT REPLACE(foo, 'BB', 'new_value') FROM self")
    shape: (4, 1)
    ┌───────────┐
    │ foo       │
    │ ---       │
    │ str       │
    ╞═══════════╡
    │ AA        │
    │ new_value │
    │ CC        │
    │ DD        │
    └───────────┘

.. _reverse:

REVERSE
-------
Returns the reversed string.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["apple", "banana", "orange", "grape"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT REVERSE(foo) FROM self")
    shape: (4, 1)
    ┌────────┐
    │ foo    │
    │ ---    │
    │ str    │
    ╞════════╡
    │ elppa  │
    │ ananab │
    │ egnaro │
    │ eparg  │
    └────────┘

.. _right:

RIGHT
-----
Returns the last (rightmost) `n` characters.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["abcd", "efgh", "ijkl", "mnop"],
		    "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT RIGHT(foo, 2) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ cd  │
    │ gh  │
    │ kl  │
    │ op  │
    └─────┘

.. _rtrim:

RTRIM
-----
Strips whitespaces from the right.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["abcd", "efgh", "ijkl", "mnop"],
		    "bar": ["zz    ", "yy  ", "xx ", "ww   "]
      }
    )
    >>> df.sql("SELECT RTRIM(bar) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ str │
    ╞═════╡
    │ zz  │
    │ yy  │
    │ xx  │
    │ ww  │
    └─────┘

.. _starts_with:

STARTS_WITH
-----------
Returns True if the value starts with the second argument.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["apple", "banana", "orange", "grape"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT STARTS_WITH(foo, 'a') FROM self")
    shape: (4, 1)
    ┌───────┐
    │ foo   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    │ false │
    │ false │
    └───────┘

.. _strpos:

STRPOS
------
Returns the index of the given substring in the target string.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["apple", "banana", "orange", "grape"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT STRPOS(foo, 'a') FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    │ 3   │
    └─────┘

.. _substring:

SUBSTRING
---------
Returns a portion of the data (first character = 0) in the range [start, start + length].

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["apple", "banana", "orange", "grape"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT SUBSTR(foo, 1, 3) FROM self")
    shape: (4, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ app │
    │ ban │
    │ ora │
    │ gra │
    └─────┘

.. _upper:

UPPER
-----
Returns an uppercased column.

**Example:**

.. code-block:: python
  
    >>> df = pl.DataFrame(
      {
        "foo": ["apple", "banana", "orange", "grape"],
        "bar": ["zz", "yy", "xx", "ww"]
      }
    )
    >>> df.sql("SELECT UPPER(foo) FROM self")
    shape: (4, 1)
    ┌────────┐
    │ foo    │
    │ ---    │
    │ str    │
    ╞════════╡
    │ APPLE  │
    │ BANANA │
    │ ORANGE │
    │ GRAPE  │
    └────────┘

