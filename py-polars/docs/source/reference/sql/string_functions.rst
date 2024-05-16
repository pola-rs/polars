String functions
=====================

.. list-table::

   * - :ref:`BitLength <bit_length>`
     - Returns the length of the input string in bits.
   * - :ref:`Concat <concat>`
     - Returns all input expressions concatenated together as a string.
   * - :ref:`ConcatWS <concat_ws>`
     - Returns all input expressions concatenated together (and interleaved with a separator) as a string.
   * - :ref:`EndsWith <ends_with>`
     - Returns True if the value ends with the second argument.
   * - :ref:`InitCap <initcap>`
     - Returns the value with the first letter capitalized.
   * - :ref:`Left <left>`
     - Returns the first (leftmost) `n` characters.
   * - :ref:`Length <length>`
     - Returns the character length of the string.
   * - :ref:`Lower <lower>`
     - Returns a lowercased column.
   * - :ref:`LTrim <ltrim>`
     - Strips whitespaces from the left.
   * - :ref:`OctetLength <octet_length>`
     - Returns the length of a given string in bytes.
   * - :ref:`RegexpLike <regexp_like>`
     - Returns True if `pattern` matches the value (optional: `flags`).
   * - :ref:`Replace <replace>`
     - Replaces a given substring with another string.
   * - :ref:`Reverse <reverse>`
     - Returns the reversed string.
   * - :ref:`Right <right>`
     - Returns the last (rightmost) `n` characters.
   * - :ref:`RTrim <rtrim>`
     - Strips whitespaces from the right.
   * - :ref:`StartsWith <starts_with>`
     - Returns True if the value starts with the second argument.
   * - :ref:`StrPos <strpos>`
     - Returns the index of the given substring in the target string.
   * - :ref:`Substring <substring>`
     - Returns a portion of the data (first character = 0) in the range [start, start + length].
   * - :ref:`Upper <upper>`
     - Returns an uppercased column.

.. _bit_length:

BitLength
-----------
Returns the length of the input string in bits.

**Example:**

.. code-block:: sql

    SELECT BIT_LENGTH(column_1) FROM df;

.. _concat:

Concat
-----------
Returns all input expressions concatenated together as a string.

**Example:**

.. code-block:: sql

    SELECT CONCAT(column_1, column_2) FROM df;

.. _concat_ws:

ConcatWS
-----------
Returns all input expressions concatenated together (and interleaved with a separator) as a string.

**Example:**

.. code-block:: sql

    SELECT CONCAT_WS(':', column_1, column_2, column_3) FROM df;

.. _ends_with:

EndsWith
-----------
Returns True if the value ends with the second argument.

**Example:**

.. code-block:: sql

    SELECT ENDS_WITH(column_1, 'a') FROM df;
    SELECT column_2 FROM df WHERE ENDS_WITH(column_1, 'a');

.. _initcap:

InitCap
-----------
Returns the value with the first letter capitalized.

**Example:**

.. code-block:: sql

    SELECT INITCAP(column_1) FROM df;

.. _left:

Left
-----------
Returns the first (leftmost) `n` characters.

**Example:**

.. code-block:: sql

    SELECT LEFT(column_1, 3) FROM df;

.. _length:

Length
-----------
Returns the character length of the string.

**Example:**

.. code-block:: sql

    SELECT LENGTH(column_1) FROM df;

.. _lower:

Lower
-----------
Returns a lowercased column.

**Example:**

.. code-block:: sql

    SELECT LOWER(column_1) FROM df;

.. _ltrim:

LTrim
-----------
Strips whitespaces from the left.

**Example:**

.. code-block:: sql

    SELECT LTRIM(column_1) FROM df;

.. _octet_length:

OctetLength
-----------
Returns the length of a given string in bytes.

**Example:**

.. code-block:: sql

    SELECT OCTET_LENGTH(column_1) FROM df;

.. _regexp_like:

RegexpLike
-----------
Returns True if `pattern` matches the value (optional: `flags`).

**Example:**

.. code-block:: sql

    SELECT REGEXP_LIKE(column_1, 'xyz', 'i') FROM df;

.. _replace:

Replace
-----------
Replaces a given substring with another string.

**Example:**

.. code-block:: sql

    SELECT REPLACE(column_1, 'old', 'new') FROM df;

.. _reverse:

Reverse
-----------
Returns the reversed string.

**Example:**

.. code-block:: sql

    SELECT REVERSE(column_1) FROM df;

.. _right:

Right
-----------
Returns the last (rightmost) `n` characters.

**Example:**

.. code-block:: sql

    SELECT RIGHT(column_1, 3) FROM df;

.. _rtrim:

RTrim
-----------
Strips whitespaces from the right.

**Example:**

.. code-block:: sql

    SELECT RTRIM(column_1) FROM df;

.. _starts_with:

StartsWith
-----------
Returns True if the value starts with the second argument.

**Example:**

.. code-block:: sql

    SELECT STARTS_WITH(column_1, 'a') FROM df;
    SELECT column_2 FROM df WHERE STARTS_WITH(column_1, 'a');

.. _strpos:

StrPos
-----------
Returns the index of the given substring in the target string.

**Example:**

.. code-block:: sql

    SELECT STRPOS(column_1, 'xyz') FROM df;

.. _substring:

Substring
-----------
Returns a portion of the data (first character = 0) in the range [start, start + length].

**Example:**

.. code-block:: sql

    SELECT SUBSTR(column_1, 3, 5) FROM df;

.. _upper:

Upper
-----------
Returns an uppercased column.

**Example:**

.. code-block:: sql

    SELECT UPPER(column_1) FROM df;

