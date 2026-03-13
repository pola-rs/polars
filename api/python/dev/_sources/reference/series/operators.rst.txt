=========
Operators
=========

Polars supports native Python operators for all common operations;
many of these operators are also available as methods on the :class:`Series`
class.

Comparison
~~~~~~~~~~

.. currentmodule:: polars
.. autosummary::
   :toctree: api/

    Series.eq
    Series.eq_missing
    Series.ge
    Series.gt
    Series.le
    Series.lt
    Series.ne
    Series.ne_missing

Numeric
~~~~~~~

.. autosummary::
   :toctree: api/

    Series.pow
