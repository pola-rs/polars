=========
DataFrame
=========
.. currentmodule:: polars

Constructor
-----------
.. autosummary::
   :toctree: api/

   DataFrame

Attributes
----------

.. autosummary::
   :toctree: api/

    DataFrame.columns
    DataFrame.dtypes
    DataFrame.height
    DataFrame.schema
    DataFrame.shape
    DataFrame.width

Conversion
----------
.. autosummary::
   :toctree: api/

    DataFrame.to_arrow
    DataFrame.to_dict
    DataFrame.to_dicts
    DataFrame.to_numpy
    DataFrame.to_pandas
    DataFrame.to_struct

Aggregation
-----------
.. autosummary::
   :toctree: api/

    DataFrame.max
    DataFrame.mean
    DataFrame.median
    DataFrame.min
    DataFrame.product
    DataFrame.quantile
    DataFrame.std
    DataFrame.sum
    DataFrame.var

Descriptive stats
-----------------
.. autosummary::
   :toctree: api/

    DataFrame.describe
    DataFrame.estimated_size
    DataFrame.is_duplicated
    DataFrame.is_empty
    DataFrame.is_unique
    DataFrame.n_chunks
    DataFrame.null_count

Computations
------------
.. autosummary::
   :toctree: api/

    DataFrame.fold
    DataFrame.hash_rows

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    DataFrame.cleared
    DataFrame.clone
    DataFrame.drop
    DataFrame.drop_in_place
    DataFrame.drop_nulls
    DataFrame.explode
    DataFrame.extend
    DataFrame.fill_nan
    DataFrame.fill_null
    DataFrame.filter
    DataFrame.find_idx_by_name
    DataFrame.get_column
    DataFrame.get_columns
    DataFrame.groupby
    DataFrame.groupby_dynamic
    DataFrame.groupby_rolling
    DataFrame.head
    DataFrame.hstack
    DataFrame.insert_at_idx
    DataFrame.interpolate
    DataFrame.join
    DataFrame.join_asof
    DataFrame.limit
    DataFrame.melt
    DataFrame.partition_by
    DataFrame.pipe
    DataFrame.pivot
    DataFrame.rechunk
    DataFrame.rename
    DataFrame.replace
    DataFrame.replace_at_idx
    DataFrame.reverse
    DataFrame.row
    DataFrame.rows
    DataFrame.sample
    DataFrame.select
    DataFrame.shift
    DataFrame.shift_and_fill
    DataFrame.shrink_to_fit
    DataFrame.slice
    DataFrame.sort
    DataFrame.tail
    DataFrame.take_every
    DataFrame.to_dummies
    DataFrame.to_series
    DataFrame.transpose
    DataFrame.unique
    DataFrame.unnest
    DataFrame.unstack
    DataFrame.upsample
    DataFrame.vstack
    DataFrame.with_column
    DataFrame.with_columns
    DataFrame.with_row_count

Apply
-----
.. autosummary::
   :toctree: api/

    DataFrame.apply

Various
--------
.. autosummary::
   :toctree: api/

    DataFrame.frame_equal
    DataFrame.lazy

GroupBy
-------
This namespace comes available by calling `DataFrame.groupby(..)`.

.. currentmodule:: polars.internals.dataframe.groupby

.. autosummary::
   :toctree: api/

    GroupBy.agg
    GroupBy.agg_list
    GroupBy.apply
    GroupBy.count
    GroupBy.first
    GroupBy.head
    GroupBy.last
    GroupBy.max
    GroupBy.mean
    GroupBy.median
    GroupBy.min
    GroupBy.n_unique
    GroupBy.pivot
    GroupBy.quantile
    GroupBy.sum
    GroupBy.tail

Pivot
-----
This namespace comes available by calling `DataFrame.groupby(..).pivot`

*Note that this API is deprecated in favor of `DataFrame.pivot`*

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
