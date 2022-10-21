
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


Property-based testing
----------------------

See the Hypothesis library for details:
https://hypothesis.readthedocs.io/

Strategies
~~~~~~~~~~

Polars provides the following hypothesis
testing strategies and strategy helpers:

.. autosummary::
   :toctree: api/

    testing._parametric.dataframes
    testing._parametric.series

Strategy helpers
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    testing._parametric.column
    testing._parametric.columns
