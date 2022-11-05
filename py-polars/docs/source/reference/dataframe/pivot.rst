=====
Pivot
=====

.. currentmodule:: polars

This namespace is available after calling :code:`DataFrame.groupby(...).pivot`

.. deprecated:: 0.13.23

   Note that this API has been deprecated in favor of :meth:`DataFrame.pivot`
   and will be removed in a future version.

.. currentmodule:: polars.internals.dataframe.pivot
.. autosummary::
   :toctree: api/

    PivotOps.count
    PivotOps.first
    PivotOps.last
    PivotOps.max
    PivotOps.mean
    PivotOps.median
    PivotOps.min
    PivotOps.sum
