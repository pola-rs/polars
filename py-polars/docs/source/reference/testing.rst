
============
Testing
============
.. currentmodule:: polars


Asserts
-------

.. autosummary::
   :toctree: api/

    testing.assert_frame_equal
    testing.assert_series_equal


Parametric testing
------------------

See the Hypothesis library for more details about property-based testing, strategies,
and library integrations:

* `Overview <https://hypothesis.readthedocs.io/>`_
* `Quick start guide <https://hypothesis.readthedocs.io/en/latest/quickstart.html>`_


Polars primitives
~~~~~~~~~~~~~~~~~

Polars provides the following `hypothesis <https://hypothesis.readthedocs.io>`_
testing primitives and strategy generators/helpers to make it easy to generate
suitable test DataFrames and Series.

.. autosummary::
   :toctree: api/

    testing.parametric.dataframes
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
