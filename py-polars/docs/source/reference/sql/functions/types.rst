Types
=====

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Function
     - Description
   * - :ref:`CAST <cast>`
     - Convert a value to a different datatype.
   * - :ref:`TRY_CAST <try_cast>`
     - Convert a value to a different datatype, returning NULL if the conversion fails.


.. _cast:

CAST
----
Convert a value to a different datatype.

Note that the more compact PostgreSQL `<expr>::type` syntax is also supported.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [20, 10, 30],
        "bar": ["1999-12-31", "2012-07-05", "2024-01-01"],
      }
    )
    df.sql("""
      SELECT
        foo::float4,
        bar::date
      FROM self
    """)
    # shape: (3, 2)
    # ┌──────┬────────────┐
    # │ foo  ┆ bar        │
    # │ ---  ┆ ---        │
    # │ f32  ┆ date       │
    # ╞══════╪════════════╡
    # │ 20.0 ┆ 1999-12-31 │
    # │ 10.0 ┆ 2012-07-05 │
    # │ 30.0 ┆ 2024-01-01 │
    # └──────┴────────────┘

Nested types are written as `STRUCT<name TYPE, ...>` and `MAP(KEY, VALUE)`. A struct is cast
field by field. To reorder, drop or add fields, use `TRY_CAST`, which matches fields by name (an
added field is null).
Only a `STRUCT` value or `NULL` can be cast to `STRUCT`, and only a `MAP` value or `NULL` to `MAP`.
For a list of nested structs, separate the closing brackets: `STRUCT<a STRUCT<b INT> >[]`; the
parser currently reads `STRUCT<a STRUCT<b INT>>[]` as `STRUCT<a STRUCT<b INT>[]>`.

`CATEGORICAL` is a categorical, and `CATEGORICAL('name')` one with the categories of that name.
`ENUM('a', 'b')` is an enum of those values, in that order. A timestamp in a named time zone is
written as `DATETIME64(3, 'Europe/Amsterdam')`, with the precision (1 to 9) first.

**Example:**

.. code-block:: python

    df = pl.DataFrame({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
    df.sql("""
      SELECT TRY_CAST(s AS STRUCT<y VARCHAR, x BIGINT, z DOUBLE>) AS s
      FROM self
    """)
    # shape: (2, 1)
    # ┌──────────────┐
    # │ s            │
    # │ ---          │
    # │ struct[3]    │
    # ╞══════════════╡
    # │ {"a",1,null} │
    # │ {"b",2,null} │
    # └──────────────┘

Struct values are written as `{'a': 1, 'b': 'x'}` and map values as `MAP {'k': 1}`. Their
entries can be any expression. The field names of a struct must differ. A map key cannot be
`NULL`. Constant map keys must differ once converted to the map's key type. Keys from columns can
still be equal in a row. The map then keeps one entry for that key, where the key first appears,
with the last value given for it. An empty map needs a type: `CAST(MAP {} AS MAP(VARCHAR, INT))`.
Casting a map value written this way casts its keys and values. `TRY_CAST` still raises when a key
does not convert, as a key cannot be `NULL`.

**Example:**

.. code-block:: python

    pl.sql("""
      SELECT {'a': 1, 'b': 'x'} AS s, MAP {'k': 1, 'j': 2} AS m
    """).collect()
    # shape: (1, 2)
    # ┌───────────┬──────────────────┐
    # │ s         ┆ m                │
    # │ ---       ┆ ---              │
    # │ struct[2] ┆ map[str, i32]    │
    # ╞═══════════╪══════════════════╡
    # │ {1,"x"}   ┆ {"k": 1, "j": 2} │
    # └───────────┴──────────────────┘


.. _try_cast:

TRY_CAST
--------
Convert a value to a different datatype, returning `NULL` if the conversion fails.

**Example:**

.. code-block:: python

    df = pl.DataFrame(
      {
        "foo": [65432, 10101, -33333],
        "bar": ["1999-12-31", "N/A", "2024-01-01"],
      }
    )
    df.sql("""
      SELECT
        TRY_CAST(foo AS uint2),
        TRY_CAST(bar AS date)
      FROM self
    """)
    # shape: (3, 2)
    # ┌───────┬────────────┐
    # │ foo   ┆ bar        │
    # │ ---   ┆ ---        │
    # │ u16   ┆ date       │
    # ╞═══════╪════════════╡
    # │ 65432 ┆ 1999-12-31 │
    # │ 10101 ┆ null       │
    # │ null  ┆ 2024-01-01 │
    # └───────┴────────────┘

Note that with a regular `CAST` this would fail with the following error:

.. code-block::

    InvalidOperationError:
      conversion from `i64` to `u16` failed in column 'foo' for 1 out of 3 values: [-33333]
