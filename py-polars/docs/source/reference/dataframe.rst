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
    DataFrame.insert_at_idx
    DataFrame.filter
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
    DataFrame.select_at_idx
    DataFrame.clone
    DataFrame.get_columns
    DataFrame.fill_none
    DataFrame.explode
    DataFrame.melt
    DataFrame.shift
    DataFrame.shift_and_fill
    DataFrame.with_column
    DataFrame.hstack
    DataFrame.vstack
    DataFrame.groupby
    DataFrame.downsample
    DataFrame.select
    DataFrame.with_columns
    DataFrame.sample
    DataFrame.row
    DataFrame.rows
    DataFrame.to_dummies
    DataFrame.drop_duplicates
    DataFrame.shrink_to_fit
    DataFrame.rechunk
    DataFrame.pipe
    DataFrame.join

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

.. currentmodule:: polars.eager.frame

.. autosummary::
   :toctree: api/

    GroupBy.agg
    GroupBy.apply
    GroupBy.head
    GroupBy.tail
    GroupBy.get_group
    GroupBy.groups
    GroupBy.select
    GroupBy.select_all
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

.. currentmodule:: polars.eager.frame

.. autosummary::
   :toctree: api/

    PivotOps.first
    PivotOps.sum
    PivotOps.min
    PivotOps.max
    PivotOps.mean
    PivotOps.count
    PivotOps.median
