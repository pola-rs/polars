=========
LazyFrame
=========
.. currentmodule:: polars

.. py:class:: LazyFrame
    :canonical: polars.internals.lazy_frame.LazyFrame

    Representation of a Lazy computation graph/ query.

Attributes
----------

.. autosummary::
   :toctree: api/

    LazyFrame.columns
    LazyFrame.dtypes
    LazyFrame.schema

Aggregation
-----------
.. autosummary::
   :toctree: api/

    LazyFrame.max
    LazyFrame.mean
    LazyFrame.median
    LazyFrame.min
    LazyFrame.quantile
    LazyFrame.std
    LazyFrame.sum
    LazyFrame.var

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

    LazyFrame.clone
    LazyFrame.distinct
    LazyFrame.drop
    LazyFrame.drop_nulls
    LazyFrame.explode
    LazyFrame.fill_nan
    LazyFrame.fill_null
    LazyFrame.filter
    LazyFrame.first
    LazyFrame.groupby
    LazyFrame.groupby_dynamic
    LazyFrame.groupby_rolling
    LazyFrame.head
    LazyFrame.inspect
    LazyFrame.interpolate
    LazyFrame.join
    LazyFrame.join_asof
    LazyFrame.last
    LazyFrame.limit
    LazyFrame.melt
    LazyFrame.rename
    LazyFrame.reverse
    LazyFrame.select
    LazyFrame.shift
    LazyFrame.shift_and_fill
    LazyFrame.slice
    LazyFrame.sort
    LazyFrame.tail
    LazyFrame.unique
    LazyFrame.unnest
    LazyFrame.with_column
    LazyFrame.with_columns
    LazyFrame.with_row_count

Conversion
----------
.. autosummary::
   :toctree: api/

    LazyFrame.from_json
    LazyFrame.read_json
    LazyFrame.write_json

Apply
-----
.. autosummary::
   :toctree: api/

    LazyFrame.map

Various
-------
.. autosummary::
   :toctree: api/

    LazyFrame.cache
    LazyFrame.collect
    LazyFrame.fetch
    LazyFrame.pipe

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
