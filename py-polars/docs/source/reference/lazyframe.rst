=========
LazyFrame
=========
.. currentmodule:: polars

Aggregation
-----------
.. autosummary::
   :toctree: api/

    LazyFrame.std
    LazyFrame.var
    LazyFrame.max
    LazyFrame.min
    LazyFrame.sum
    LazyFrame.mean
    LazyFrame.median
    LazyFrame.quantile

Descriptive stats
-----------------
.. autosummary::
   :toctree: api/

    LazyFrame.describe_plan
    LazyFrame.describe_optimized_plan
    LazyFrame.show_graph

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    LazyFrame.with_row_count
    LazyFrame.inspect
    LazyFrame.filter
    LazyFrame.select
    LazyFrame.groupby
    LazyFrame.groupby_dynamic
    LazyFrame.groupby_rolling
    LazyFrame.join
    DataFrame.join_asof
    LazyFrame.with_columns
    LazyFrame.with_column
    LazyFrame.drop
    LazyFrame.rename
    LazyFrame.reverse
    LazyFrame.shift
    LazyFrame.shift_and_fill
    LazyFrame.slice
    LazyFrame.limit
    LazyFrame.head
    LazyFrame.tail
    LazyFrame.last
    LazyFrame.first
    LazyFrame.fill_null
    LazyFrame.fill_nan
    LazyFrame.explode
    LazyFrame.distinct
    LazyFrame.drop_nulls
    LazyFrame.sort
    LazyFrame.melt
    LazyFrame.interpolate

Apply
-----
.. autosummary::
   :toctree: api/

    LazyFrame.map

Various
-------
.. autosummary::
   :toctree: api/

    LazyFrame.pipe
    LazyFrame.collect
    LazyFrame.fetch
    LazyFrame.cache

GroupBy
-------
This namespace comes available by calling `LazyFrame.groupby(..)`.

.. currentmodule:: polars.internals.lazy_frame

.. autosummary::
   :toctree: api/

    LazyGroupBy.agg
    LazyGroupBy.apply
    LazyGroupBy.head
    LazyGroupBy.tail
