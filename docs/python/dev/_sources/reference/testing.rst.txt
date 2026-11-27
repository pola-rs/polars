=======
Testing
=======
.. currentmodule:: polars

The ``testing`` module provides a number of functions and helpers for use with unit tests.

.. note::

    The ``testing`` module is not imported by default in order to optimise import speed of
    the primary ``polars`` module. Either import ``polars.testing`` and *then* use that
    namespace, or import the specific functions you need from the full module path, e.g.:

    .. code-block:: python

        from polars.testing import assert_frame_equal, assert_series_equal


Asserts
-------

Polars provides some standard asserts for use with unit tests:

.. autosummary::
   :toctree: api/

    testing.assert_frame_equal
    testing.assert_frame_not_equal
    testing.assert_series_equal
    testing.assert_series_not_equal


Parametric testing
------------------

See the Hypothesis library for more details about property-based testing, strategies,
and library integrations:

* `Overview <https://hypothesis.readthedocs.io/>`_
* `Quick start guide <https://hypothesis.readthedocs.io/en/latest/quickstart.html>`_


Polars strategies
~~~~~~~~~~~~~~~~~

Polars provides the following `hypothesis <https://hypothesis.readthedocs.io>`_
testing strategies:

.. autosummary::
   :toctree: api/

    testing.parametric.dataframes
    testing.parametric.dtypes
    testing.parametric.lists
    testing.parametric.series


Strategy helpers
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    testing.parametric.column
    testing.parametric.columns
    testing.parametric.create_list_strategy


Profiles
~~~~~~~~

Several standard/named `hypothesis <https://hypothesis.readthedocs.io>`_
profiles are provided:

* ``fast``: runs 100 iterations.
* ``balanced``: runs 1,000 iterations.
* ``expensive``: runs 10,000 iterations.

The load/set helper functions allow you to access these profiles directly,
set your preferred profile (default is ``fast``), or set a custom number
of iterations.

.. autosummary::
   :toctree: api/

    testing.parametric.load_profile
    testing.parametric.set_profile


**Approximate profile timings:**

Running polars' own parametric unit tests on ``0.17.6`` against release
and debug builds, on a machine with 12 cores, using ``xdist -n auto``
results in the following timings (these values are indicative only,
and may vary significantly depending on your own hardware setup):

+---------------+------------+----------------+-----------------+
| Profile       | Iterations | Release        | Debug           |
+===============+============+================+=================+
| ``fast``      | 100        | ~6 secs        | ~8 secs         |
+---------------+------------+----------------+-----------------+
| ``balanced``  | 1,000      | ~22 secs       | ~30 secs        |
+---------------+------------+----------------+-----------------+
| ``expensive`` | 10,000     | ~3 mins 5 secs | ~4 mins 45 secs |
+---------------+------------+----------------+-----------------+

Examples
~~~~~~~~

**Basic:** Create a parametric unit test that will receive a series of
generated DataFrames, each having 5 numeric columns with a 10% chance
of any generated value being ``null`` (this is distinct from ``NaN``).

.. code-block:: python

    import polars as pl
    from polars.testing.parametric import dataframes
    from polars import NUMERIC_DTYPES

    from hypothesis import given

    @given(
        dataframes(
            cols=5,
            allow_null=True,
            allowed_dtypes=NUMERIC_DTYPES,
        )
    )
    def test_numeric(df: pl.DataFrame):
        assert all(df[col].dtype.is_numeric() for col in df.columns)

        # Example frame:
        # ┌──────┬────────┬───────┬────────────┬────────────┐
        # │ col0 ┆ col1   ┆ col2  ┆ col3       ┆ col4       │
        # │ ---  ┆ ---    ┆ ---   ┆ ---        ┆ ---        │
        # │ u8   ┆ i16    ┆ u16   ┆ i32        ┆ f64        │
        # ╞══════╪════════╪═══════╪════════════╪════════════╡
        # │ 54   ┆ -29096 ┆ 485   ┆ 2147483647 ┆ -2.8257e14 │
        # │ null ┆ 7508   ┆ 37338 ┆ 7264       ┆ 1.5        │
        # │ 0    ┆ 321    ┆ null  ┆ 16996      ┆ NaN        │
        # │ 121  ┆ -361   ┆ 63204 ┆ 1          ┆ 1.1443e235 │
        # └──────┴────────┴───────┴────────────┴────────────┘

**Intermediate:** Integrate hypothesis-native strategies into specifically-named columns,
generating a series of LazyFrames, with a minimum size of five rows and values that
conform to the given strategies:

.. code-block:: python

    import polars as pl
    from polars.testing.parametric import column, dataframes

    import hypothesis.strategies as st
    from hypothesis import given
    from string import ascii_letters, digits

    id_chars = ascii_letters + digits

    @given(
        dataframes(
            cols=[
                column("id", strategy=st.text(min_size=4, max_size=4, alphabet=id_chars)),
                column("ccy", strategy=st.sampled_from(["GBP", "EUR", "JPY", "USD"])),
                column("price", strategy=st.floats(min_value=0.0, max_value=1000.0)),
            ],
            min_size=5,
            lazy=True,
        )
    )
    def test_price_calculations(lf: pl.LazyFrame):
        ...
        print(lf.collect())

        # Example frame:
        # ┌──────┬─────┬─────────┐
        # │ id   ┆ ccy ┆ price   │
        # │ ---  ┆ --- ┆ ---     │
        # │ str  ┆ str ┆ f64     │
        # ╞══════╪═════╪═════════╡
        # │ A101 ┆ GBP ┆ 1.1     │
        # │ 8nIn ┆ JPY ┆ 1.5     │
        # │ QHoO ┆ EUR ┆ 714.544 │
        # │ i0e0 ┆ GBP ┆ 0.0     │
        # │ 0000 ┆ USD ┆ 999.0   │
        # └──────┴─────┴─────────┘

**Advanced:** Create and use a ``List[UInt8]`` dtype strategy as a hypothesis
`composite <https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.composite>`_
that generates pairs of pairs of small integer values in which the first value in each nested pair
is always less than or equal to the second value:

.. code-block:: python

    import polars as pl
    from polars.testing.parametric import column, dataframes, lists

    import hypothesis.strategies as st
    from hypothesis import given

    @st.composite
    def uint8_pairs(draw: st.DrawFn):
        uints = lists(pl.UInt8, size=2)
        pairs = list(zip(draw(uints), draw(uints)))
        return [sorted(ints) for ints in pairs]

    @given(
        dataframes(
            cols=[
                column("colx", strategy=uint8_pairs()),
                column("coly", strategy=uint8_pairs()),
                column("colz", strategy=uint8_pairs()),
            ],
            min_size=3,
            max_size=3,
        )
    )
    def test_miscellaneous(df: pl.DataFrame): ...

        # Example frame:
        # ┌─────────────────────────┬─────────────────────────┬──────────────────────────┐
        # │ colx                    ┆ coly                    ┆ colz                     │
        # │ ---                     ┆ ---                     ┆ ---                      │
        # │ list[list[i64]]         ┆ list[list[i64]]         ┆ list[list[i64]]          │
        # ╞═════════════════════════╪═════════════════════════╪══════════════════════════╡
        # │ [[143, 235], [75, 101]] ┆ [[143, 235], [75, 101]] ┆ [[31, 41], [57, 250]]    │
        # │ [[87, 186], [174, 179]] ┆ [[87, 186], [174, 179]] ┆ [[112, 213], [149, 221]] │
        # │ [[23, 85], [7, 86]]     ┆ [[23, 85], [7, 86]]     ┆ [[22, 255], [27, 28]]    │
        # └─────────────────────────┴─────────────────────────┴──────────────────────────┘
