System
======

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`VERSION <version>`
     - Returns the version of Polars that plans the query.


.. _version:

VERSION
-------
Returns the version of Polars that plans the query: the Python Polars version, or in the
`polars` command-line tool, the version of that tool.

**Example:**

.. code-block:: python

    pl.sql("SELECT VERSION()").collect()
    # shape: (1, 1)
    # ┌─────────┐
    # │ version │
    # │ ---     │
    # │ str     │
    # ╞═════════╡
    # │ 2.2.0   │
    # └─────────┘
