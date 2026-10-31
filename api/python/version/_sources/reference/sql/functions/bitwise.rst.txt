Temporal
========

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description

   * - :ref:`BIT_AND <bit_and>`
     - Returns the bitwise AND of the given values.
   * - :ref:`BIT_COUNT <bit_count>`
     - Returns the number of bits set to 1 in the binary representation of the given value.
   * - :ref:`BIT_OR <bit_or>`
     - Returns the bitwise OR of the given values.
   * - :ref:`BIT_XOR <bit_xor>`
     - Returns the bitwise XOR of the given values.


.. _bit_and:

BIT_AND
-------
Returns the bitwise AND of the given values.
Also available as the `&` binary operator.

.. code-block:: python

    df = pl.DataFrame(
      {
          "i": [3, 10, 4, 8],
          "j": [4, 7, 9, 10],
      }
    )
    df.sql("""
      SELECT
        i,
        j,
        i & j AS i_bitand_op_j,
        BIT_AND(i, j) AS i_bitand_j
      FROM self
    """)
    # shape: (4, 4)
    # ┌─────┬─────┬───────────────┬────────────┐
    # │ i   ┆ j   ┆ i_bitand_op_j ┆ i_bitand_j │
    # │ --- ┆ --- ┆ ---           ┆ ---        │
    # │ i64 ┆ i64 ┆ i64           ┆ i64        │
    # ╞═════╪═════╪═══════════════╪════════════╡
    # │ 3   ┆ 4   ┆ 0             ┆ 0          │
    # │ 10  ┆ 7   ┆ 2             ┆ 2          │
    # │ 4   ┆ 9   ┆ 0             ┆ 0          │
    # │ 8   ┆ 10  ┆ 8             ┆ 8          │
    # └─────┴─────┴───────────────┴────────────┘

.. _bit_count:

BIT_COUNT
---------
Returns the number of bits set to 1 in the binary representation of the given value.

.. code-block:: python

    df = pl.DataFrame({"i": [16, 10, 55, 127]})
    df.sql("""
      SELECT
        i,
        BIT_COUNT(i) AS i_bitcount
      FROM self
    """)
    # shape: (4, 2)
    # ┌─────┬────────────┐
    # │ i   ┆ i_bitcount │
    # │ --- ┆ ---        │
    # │ i64 ┆ u32        │
    # ╞═════╪════════════╡
    # │ 16  ┆ 1          │
    # │ 10  ┆ 2          │
    # │ 55  ┆ 5          │
    # │ 127 ┆ 7          │
    # └─────┴────────────┘

.. _bit_or:

BIT_OR
------
Returns the bitwise OR of the given values.
Also available as the `|` binary operator.

.. code-block:: python

    df = pl.DataFrame(
      {
          "i": [3, 10, 4, 8],
          "j": [4, 7, 9, 10],
      }
    )
    df.sql("""
      SELECT
        i,
        j,
        i | j AS i_bitor_op_j,
        BIT_OR(i, j) AS i_bitor_j
      FROM self
    """)
    # shape: (4, 4)
    # ┌─────┬─────┬──────────────┬───────────┐
    # │ i   ┆ j   ┆ i_bitor_op_j ┆ i_bitor_j │
    # │ --- ┆ --- ┆ ---          ┆ ---       │
    # │ i64 ┆ i64 ┆ i64          ┆ i64       │
    # ╞═════╪═════╪══════════════╪═══════════╡
    # │ 3   ┆ 4   ┆ 7            ┆ 7         │
    # │ 10  ┆ 7   ┆ 15           ┆ 15        │
    # │ 4   ┆ 9   ┆ 13           ┆ 13        │
    # │ 8   ┆ 10  ┆ 10           ┆ 10        │
    # └─────┴─────┴──────────────┴───────────┘

.. _bit_xor:

BIT_XOR
-------
Returns the bitwise XOR of the given values.
Also available as the `XOR` binary operator.

.. code-block:: python

    df = pl.DataFrame(
      {
          "i": [3, 10, 4, 8],
          "j": [4, 7, 9, 10],
      }
    )
    df.sql("""
      SELECT
        i,
        j,
        i XOR j AS i_bitxor_op_j,
        BIT_XOR(i, j) AS i_bitxor_j
      FROM self
    """)
    # shape: (4, 4)
    # ┌─────┬─────┬───────────────┬────────────┐
    # │ i   ┆ j   ┆ i_bitxor_op_j ┆ i_bitxor_j │
    # │ --- ┆ --- ┆ ---           ┆ ---        │
    # │ i64 ┆ i64 ┆ i64           ┆ i64        │
    # ╞═════╪═════╪═══════════════╪════════════╡
    # │ 3   ┆ 4   ┆ 7             ┆ 7          │
    # │ 10  ┆ 7   ┆ 13            ┆ 13         │
    # │ 4   ┆ 9   ┆ 13            ┆ 13         │
    # │ 8   ┆ 10  ┆ 2             ┆ 2          │
    # └─────┴─────┴───────────────┴────────────┘
