String functions
=====================

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
-----------
Returns the length of the input string in bits.

**Example:**

.. code-block:: sql

    SELECT BIT_LENGTH(column_1) FROM df;

.. _concat:

CONCAT
-----------
Returns all input expressions concatenated together as a string.

**Example:**

.. code-block:: sql

    SELECT CONCAT(column_1, column_2) FROM df;

.. _concat_ws:

CONCAT_WS
-----------
Returns all input expressions concatenated together (and interleaved with a separator) as a string.

**Example:**

.. code-block:: sql

    SELECT CONCAT_WS(':', column_1, column_2, column_3) FROM df;
.. _ends_with:

ENDS_WITH
-----------
Returns True if the value ends with the second argument.

**Example:**

.. code-block:: sql

    SELECT ENDS_WITH(column_1, 'a') FROM df;
    SELECT column_2 FROM df WHERE ENDS_WITH(column_1, 'a');

.. _initcap:

INITCAP
-----------
Returns the value with the first letter capitalized.

**Example:**

.. code-block:: sql

    SELECT INITCAP(column_1) FROM df;

.. _left:

LEFT
-----------
Returns the first (leftmost) `n` characters.

**Example:**

.. code-block:: sql

    SELECT LEFT(column_1, 3) FROM df;

.. _length:

LENGTH
-----------
Returns the character length of the string.

**Example:**

.. code-block:: sql

    SELECT LENGTH(column_1) FROM df;

.. _lower:

LOWER
-----------
Returns a lowercased column.

**Example:**

.. code-block:: sql

    SELECT LOWER(column_1) FROM df;

.. _ltrim:

LTRIM
-----------
Strips whitespaces from the left.

**Example:**

.. code-block:: sql

    SELECT LTRIM(column_1) FROM df;

.. _octet_length:

OCTET_LENGTH
--------------
Returns the length of a given string in bytes.

**Example:**

.. code-block:: sql

    SELECT OCTET_LENGTH(column_1) FROM df;

.. _regexp_like:

REGEXP_LIKE
-------------
Returns True if `pattern` matches the value (optional: `flags`).

**Example:**

.. code-block:: sql

    SELECT REGEXP_LIKE(column_1, 'xyz', 'i') FROM df;

.. _replace:

REPLACE
-----------
Replaces a given substring with another string.

**Example:**

.. code-block:: sql

    SELECT REPLACE(column_1, 'old', 'new') FROM df;

.. _reverse:

REVERSE
-----------
Returns the reversed string.

**Example:**

.. code-block:: sql

    SELECT REVERSE(column_1) FROM df;

.. _right:

RIGHT
-----------
Returns the last (rightmost) `n` characters.

**Example:**

.. code-block:: sql

    SELECT RIGHT(column_1, 3) FROM df;

.. _rtrim:

RTRIM
-----------
Strips whitespaces from the right.

**Example:**

.. code-block:: sql

    SELECT RTRIM(column_1) FROM df;

.. _starts_with:

STARTS_WITH
------------
Returns True if the value starts with the second argument.

**Example:**

.. code-block:: sql

    SELECT STARTS_WITH(column_1, 'a') FROM df;
    SELECT column_2 FROM df WHERE STARTS_WITH(column_1, 'a');

.. _strpos:

STRPOS
-----------
Returns the index of the given substring in the target string.

**Example:**

.. code-block:: sql

    SELECT STRPOS(column_1, 'xyz') FROM df;

.. _substring:

SUBSTRING
-----------
Returns a portion of the data (first character = 0) in the range [start, start + length].

**Example:**

.. code-block:: sql

    SELECT SUBSTR(column_1, 3, 5) FROM df;

.. _upper:

UPPER
-----------
Returns an uppercased column.

**Example:**

.. code-block:: sql

    SELECT UPPER(column_1) FROM df;

