Aggregate
=========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`AVG <avg>`
     - Returns the average (mean) of all the elements in the grouping.
   * - :ref:`COUNT <count>`
     - Returns the amount of elements in the grouping.
   * - :ref:`FIRST <first>`
     - Returns the first element of the grouping.
   * - :ref:`LAST <last>`
     - Returns the last element of the grouping.
   * - :ref:`MAX <max>`
     - Returns the greatest (maximum) of all the elements in the grouping.
   * - :ref:`MEDIAN <median>`
     - Returns the median element from the grouping.
   * - :ref:`MIN <min>`
     - Returns the smallest (minimum) of all the elements in the grouping.
   * - :ref:`STDDEV <stddev>`
     - Returns the standard deviation of all the elements in the grouping.
   * - :ref:`SUM <sum>`
     - Returns the sum of all the elements in the grouping.
   * - :ref:`VARIANCE <variance>`
     - Returns the variance of all the elements in the grouping.

.. _avg:

AVG
---
Returns the average (mean) of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT AVG(column_1) FROM df;

.. _count:

COUNT
-----
Returns the amount of elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT COUNT(column_1) FROM df;
    SELECT COUNT(*) FROM df;
    SELECT COUNT(DISTINCT column_1) FROM df;
    SELECT COUNT(DISTINCT *) FROM df;

.. _first:

FIRST
-----
Returns the first element of the grouping.

**Example:**

.. code-block:: sql

    SELECT FIRST(column_1) FROM df;

.. _last:

LAST
----
Returns the last element of the grouping.

**Example:**

.. code-block:: sql

    SELECT LAST(column_1) FROM df;

.. _max:

MAX
---
Returns the greatest (maximum) of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT MAX(column_1) FROM df;

.. _median:

MEDIAN
------
Returns the median element from the grouping.

**Example:**

.. code-block:: sql

    SELECT MEDIAN(column_1) FROM df;

.. _min:

MIN
---
Returns the smallest (minimum) of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT MIN(column_1) FROM df;

.. _stddev:

STDDEV
------
Returns the standard deviation of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT STDDEV(column_1) FROM df;

.. _sum:

SUM
---
Returns the sum of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT SUM(column_1) FROM df;

.. _variance:

VARIANCE
--------
Returns the variance of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT VARIANCE(column_1) FROM df;

