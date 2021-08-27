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

    LazyFrame.filter
    LazyFrame.select
    LazyFrame.groupby
    LazyFrame.join
    LazyFrame.with_columns
    LazyFrame.with_column
    LazyFrame.drop_columns
    LazyFrame.drop_column
    LazyFrame.with_column_renamed
    LazyFrame.reverse
    LazyFrame.shift
    LazyFrame.shift_and_fill
    LazyFrame.slice
    LazyFrame.limit
    LazyFrame.head
    LazyFrame.tail
    LazyFrame.last
    LazyFrame.first
    LazyFrame.fill_none
    LazyFrame.fill_nan
    LazyFrame.explode
    LazyFrame.drop_duplicates
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

.. currentmodule:: polars.lazy.frame

.. autosummary::
   :toctree: api/

    LazyGroupBy.agg
    LazyGroupBy.apply
    LazyGroupBy.head
    LazyGroupBy.tail
