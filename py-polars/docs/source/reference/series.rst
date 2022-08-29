======
Series
======
.. currentmodule:: polars

Constructor
-----------
.. autosummary::
   :toctree: api/

   Series

Attributes
----------

.. autosummary::
   :toctree: api/

   Series.arr
   Series.cat
   Series.dt
   Series.dtype
   Series.inner_dtype
   Series.name
   Series.shape
   Series.str
   Series.time_unit
   Series.flags

Conversion
----------
.. autosummary::
   :toctree: api/

   Series.to_arrow
   Series.to_frame
   Series.to_list
   Series.to_numpy
   Series.to_pandas


Aggregation
-----------
.. autosummary::
   :toctree: api/

    Series.arg_max
    Series.arg_min
    Series.max
    Series.mean
    Series.median
    Series.min
    Series.mode
    Series.product
    Series.quantile
    Series.std
    Series.sum
    Series.var

Descriptive stats
-----------------
.. autosummary::
   :toctree: api/

    Series.chunk_lengths
    Series.describe
    Series.estimated_size
    Series.has_validity
    Series.is_boolean
    Series.is_datelike
    Series.is_duplicated
    Series.is_empty
    Series.is_finite
    Series.is_first
    Series.is_float
    Series.is_in
    Series.is_infinite
    Series.is_nan
    Series.is_not_nan
    Series.is_not_null
    Series.is_null
    Series.is_numeric
    Series.is_unique
    Series.is_utf8
    Series.len
    Series.n_chunks
    Series.n_unique
    Series.null_count
    Series.unique_counts
    Series.value_counts

Boolean
-------
.. autosummary::
   :toctree: api/

    Series.all
    Series.any

Computations
------------
.. autosummary::
   :toctree: api/

    Series.abs
    Series.arccos
    Series.arccosh
    Series.arcsin
    Series.arcsinh
    Series.arctan
    Series.arctanh
    Series.arg_true
    Series.arg_unique
    Series.cos
    Series.cosh
    Series.cummax
    Series.cummin
    Series.cumprod
    Series.cumsum
    Series.cumulative_eval
    Series.diff
    Series.dot
    Series.entropy
    Series.ewm_mean
    Series.ewm_std
    Series.ewm_var
    Series.exp
    Series.hash
    Series.kurtosis
    Series.log
    Series.log10
    Series.pct_change
    Series.peak_max
    Series.peak_min
    Series.rank
    Series.rolling_apply
    Series.rolling_max
    Series.rolling_mean
    Series.rolling_median
    Series.rolling_min
    Series.rolling_quantile
    Series.rolling_skew
    Series.rolling_std
    Series.rolling_sum
    Series.rolling_var
    Series.search_sorted
    Series.sign
    Series.sin
    Series.sinh
    Series.skew
    Series.sqrt
    Series.tan
    Series.tanh
    Series.unique

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    Series.alias
    Series.append
    Series.arg_sort
    Series.argsort
    Series.cast
    Series.ceil
    Series.cleared
    Series.clip
    Series.clip_max
    Series.clip_min
    Series.clone
    Series.drop_nans
    Series.drop_nulls
    Series.explode
    Series.extend_constant
    Series.fill_nan
    Series.fill_null
    Series.filter
    Series.floor
    Series.head
    Series.interpolate
    Series.limit
    Series.rechunk
    Series.rename
    Series.reshape
    Series.reverse
    Series.round
    Series.sample
    Series.set
    Series.set_at_idx
    Series.shift
    Series.shift_and_fill
    Series.shrink_to_fit
    Series.shuffle
    Series.slice
    Series.sort
    Series.tail
    Series.take
    Series.take_every
    Series.to_dummies
    Series.top_k
    Series.view
    Series.zip_with

Various
--------
.. autosummary::
   :toctree: api/

    Series.apply
    Series.dt
    Series.reinterpret
    Series.series_equal
    Series.set_sorted
    Series.str
    Series.to_physical

TimeSeries
----------
The following methods are available under the `Series.dt` attribute.

.. currentmodule:: polars.internals.series.datetime

.. autosummary::
   :toctree: api/

    DateTimeNameSpace.cast_time_unit
    DateTimeNameSpace.day
    DateTimeNameSpace.days
    DateTimeNameSpace.epoch
    DateTimeNameSpace.hour
    DateTimeNameSpace.hours
    DateTimeNameSpace.max
    DateTimeNameSpace.mean
    DateTimeNameSpace.median
    DateTimeNameSpace.milliseconds
    DateTimeNameSpace.min
    DateTimeNameSpace.minute
    DateTimeNameSpace.minutes
    DateTimeNameSpace.month
    DateTimeNameSpace.nanosecond
    DateTimeNameSpace.nanoseconds
    DateTimeNameSpace.ordinal_day
    DateTimeNameSpace.offset_by
    DateTimeNameSpace.quarter
    DateTimeNameSpace.second
    DateTimeNameSpace.seconds
    DateTimeNameSpace.strftime
    DateTimeNameSpace.timestamp
    DateTimeNameSpace.truncate
    DateTimeNameSpace.week
    DateTimeNameSpace.weekday
    DateTimeNameSpace.with_time_unit
    DateTimeNameSpace.year


Strings
-------

The following methods are available under the `Series.str` attribute.

.. currentmodule:: polars.internals.series.string

.. autosummary::
   :toctree: api/

    StringNameSpace.concat
    StringNameSpace.contains
    StringNameSpace.count_match
    StringNameSpace.decode
    StringNameSpace.encode
    StringNameSpace.ends_with
    StringNameSpace.extract
    StringNameSpace.extract_all
    StringNameSpace.json_path_match
    StringNameSpace.lengths
    StringNameSpace.ljust
    StringNameSpace.lstrip
    StringNameSpace.replace
    StringNameSpace.replace_all
    StringNameSpace.rjust
    StringNameSpace.rstrip
    StringNameSpace.slice
    StringNameSpace.split
    StringNameSpace.split_exact
    StringNameSpace.splitn
    StringNameSpace.starts_with
    StringNameSpace.strip
    StringNameSpace.strptime
    StringNameSpace.to_lowercase
    StringNameSpace.to_uppercase
    StringNameSpace.zfill

Lists
-----

The following methods are available under the `Series.arr` attribute.

.. currentmodule:: polars.internals.series.list

.. autosummary::
   :toctree: api/

    ListNameSpace.arg_max
    ListNameSpace.arg_min
    ListNameSpace.concat
    ListNameSpace.contains
    ListNameSpace.diff
    ListNameSpace.eval
    ListNameSpace.first
    ListNameSpace.get
    ListNameSpace.head
    ListNameSpace.join
    ListNameSpace.last
    ListNameSpace.lengths
    ListNameSpace.max
    ListNameSpace.mean
    ListNameSpace.min
    ListNameSpace.reverse
    ListNameSpace.shift
    ListNameSpace.slice
    ListNameSpace.sort
    ListNameSpace.sum
    ListNameSpace.tail
    ListNameSpace.unique

Categories
----------
The following methods are available under the `Series.cat` attribute.

.. currentmodule:: polars.internals.series.categorical

.. autosummary::
   :toctree: api/

    CatNameSpace.set_ordering

Struct
------
The following methods are available under the `Series.struct` attribute.

.. currentmodule:: polars.internals.series.struct

.. autosummary::
   :toctree: api/

    StructNameSpace.field
    StructNameSpace.fields
    StructNameSpace.rename_fields
    StructNameSpace.to_frame
