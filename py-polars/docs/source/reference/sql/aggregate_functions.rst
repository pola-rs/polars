Aggregate functions
=========================

.. list-table::
   
   * - :ref:`Avg <avg>`
     - Returns the average (mean) of all the elements in the grouping.
   * - :ref:`Count <count>`
     - Returns the amount of elements in the grouping.
   * - :ref:`First <first>`
     - Returns the first element of the grouping.
   * - :ref:`Last <last>`
     - Returns the last element of the grouping.
   * - :ref:`Max <max>`
     - Returns the greatest (maximum) of all the elements in the grouping.
   * - :ref:`Median <median>`
     - Returns the median element from the grouping.
   * - :ref:`Min <min>`
     - Returns the smallest (minimum) of all the elements in the grouping.
   * - :ref:`StdDev <stddev>`
     - Returns the standard deviation of all the elements in the grouping.
   * - :ref:`Sum <sum>`
     - Returns the sum of all the elements in the grouping.
   * - :ref:`Variance <variance>`
     - Returns the variance of all the elements in the grouping.

.. _avg:

Avg
-----------
Returns the average (mean) of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT AVG(column_1) FROM df;

.. _count:

Count
-----------
Returns the amount of elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT COUNT(column_1) FROM df;
    SELECT COUNT(*) FROM df;
    SELECT COUNT(DISTINCT column_1) FROM df;
    SELECT COUNT(DISTINCT *) FROM df;

.. _first:

First
-----------
Returns the first element of the grouping.

**Example:**

.. code-block:: sql

    SELECT FIRST(column_1) FROM df;

.. _last:

Last
-----------
Returns the last element of the grouping.

**Example:**

.. code-block:: sql

    SELECT LAST(column_1) FROM df;

.. _max:

Max
-----------
Returns the greatest (maximum) of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT MAX(column_1) FROM df;

.. _median:

Median
-----------
Returns the median element from the grouping.

**Example:**

.. code-block:: sql

    SELECT MEDIAN(column_1) FROM df;

.. _min:

Min
-----------
Returns the smallest (minimum) of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT MIN(column_1) FROM df;

.. _stddev:

StdDev
-----------
Returns the standard deviation of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT STDDEV(column_1) FROM df;

.. _sum:

Sum
-----------
Returns the sum of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT SUM(column_1) FROM df;

.. _variance:

Variance
-----------
Returns the variance of all the elements in the grouping.

**Example:**

.. code-block:: sql

    SELECT VARIANCE(column_1) FROM df;
