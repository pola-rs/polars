Temporal
========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`DATE <date>`
     - Converts a formatted string date to an actual Date type.
   * - :ref:`DATE_PART <date_part>`
     - Extracts a part of a date (or datetime) such as 'year', 'month', etc.
   * - :ref:`EXTRACT <extract>`
     - Offers the same functionality as `DATE_PART` with slightly different syntax.

.. _date:

DATE
----
Converts a formatted string date to an actual Date type; ISO-8601 format is assumed
unless a strftime-compatible formatting string is provided as the second parameter.

**Example:**

.. code-block:: sql

    SELECT DATE('2021-03-15') FROM df;
    SELECT DATE('2021-15-03', '%Y-d%-%m') FROM df;
    SELECT DATE('2021-03', '%Y-%m') FROM df;

.. _date_part:

DATE_PART
---------
Extracts a part of a date (or datetime) such as 'year', 'month', etc.

**Example:**

.. code-block:: sql

    SELECT DATE_PART('year', column_1) FROM df;
    SELECT DATE_PART('day', column_1) FROM df;

.. _extract:

EXTRACT
-------
Extracts a part of a date (or datetime) such as 'year', 'month', etc.

.. code-block:: sql

    SELECT EXTRACT(isoyear FROM column_1) FROM df;
    SELECT EXTRACT(minute FROM column_1) FROM df;
