=========
DataFrame
=========
.. currentmodule:: polars

Constructor
-----------

   DataFrame

Attributes
----------

.. autosummary::
   :toctree: api/

    DataFrame.shape
    DataFrame.height
    DataFrame.width
    DataFrame.columns
    DataFrame.dtypes

Conversion
----------
.. autosummary::
   :toctree: api/

    DataFrame.to_arrow
    DataFrame.to_json
    DataFrame.to_pandas
    DataFrame.to_csv
    DataFrame.to_ipc
    DataFrame.to_parquet
    DataFrame.to_numpy
    DataFrame.to_dict

Aggregation
-----------
.. autosummary::
   :toctree: api/

    DataFrame.max
    DataFrame.min
    DataFrame.sum
    DataFrame.mean
    DataFrame.std
    DataFrame.var
    DataFrame.median
    DataFrame.quantile

Descriptive stats
-----------------
.. autosummary::
   :toctree: api/

    DataFrame.describe
    DataFrame.is_duplicated
    DataFrame.is_unique
    DataFrame.n_chunks
    DataFrame.null_count
    DataFrame.is_empty

Computations
------------
.. autosummary::
   :toctree: api/

    DataFrame.hash_rows
    DataFrame.fold

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    DataFrame.rename
    DataFrame.with_row_count
    DataFrame.insert_at_idx
    DataFrame.filter
    DataFrame.find_idx_by_name
    DataFrame.select_at_idx
    DataFrame.replace_at_idx
    DataFrame.sort
    DataFrame.replace
    DataFrame.slice
    DataFrame.limit
    DataFrame.head
    DataFrame.tail
    DataFrame.drop_nulls
    DataFrame.drop
    DataFrame.drop_in_place
    DataFrame.to_series
    DataFrame.clone
    DataFrame.get_columns
    DataFrame.get_column
    DataFrame.fill_null
    DataFrame.fill_nan
    DataFrame.explode
    DataFrame.melt
    DataFrame.shift
    DataFrame.shift_and_fill
    DataFrame.with_column
    DataFrame.hstack
    DataFrame.vstack
    DataFrame.groupby
    DataFrame.groupby_dynamic
    DataFrame.select
    DataFrame.with_columns
    DataFrame.with_column_renamed
    DataFrame.sample
    DataFrame.row
    DataFrame.rows
    DataFrame.to_dummies
    DataFrame.drop_duplicates
    DataFrame.shrink_to_fit
    DataFrame.rechunk
    DataFrame.pipe
    DataFrame.join
    DataFrame.interpolate
    DataFrame.transpose
    DataFrame.upsample

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

.. currentmodule:: polars.internals.frame

.. autosummary::
   :toctree: api/

    GroupBy.agg
    GroupBy.apply
    GroupBy.head
    GroupBy.tail
    GroupBy.get_group
    GroupBy.groups
    GroupBy.pivot
    GroupBy.first
    GroupBy.last
    GroupBy.sum
    GroupBy.min
    GroupBy.max
    GroupBy.count
    GroupBy.mean
    GroupBy.n_unique
    GroupBy.quantile
    GroupBy.median
    GroupBy.agg_list

Pivot
-----
This namespace comes available by calling `DataFrame.groupby(..).pivot`

.. currentmodule:: polars.internals.frame

.. autosummary::
   :toctree: api/

    PivotOps.first
    PivotOps.sum
    PivotOps.min
    PivotOps.max
    PivotOps.mean
    PivotOps.count
    PivotOps.median
