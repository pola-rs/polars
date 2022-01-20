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

   Series.dtype
   Series.inner_dtype
   Series.name
   Series.shape
   Series.arr
   Series.dt
   Series.str
   Series.time_unit

Conversion
----------
.. autosummary::
   :toctree: api/

   Series.to_frame
   Series.to_list
   Series.to_numpy
   Series.to_arrow
   Series.to_numpy


Aggregation
-----------
.. autosummary::
   :toctree: api/

    Series.sum
    Series.mean
    Series.min
    Series.max
    Series.std
    Series.var
    Series.median
    Series.quantile
    Series.product
    Series.mode
    Series.arg_min
    Series.arg_max

Descriptive stats
-----------------
.. autosummary::
   :toctree: api/

    Series.describe
    Series.value_counts
    Series.chunk_lengths
    Series.n_chunks
    Series.null_count
    Series.is_null
    Series.is_not_null
    Series.is_finite
    Series.is_infinite
    Series.is_nan
    Series.is_not_nan
    Series.is_in
    Series.is_unique
    Series.is_first
    Series.is_duplicated
    Series.is_numeric
    Series.is_float
    Series.is_boolean
    Series.is_utf8
    Series.is_datelike
    Series.len
    Series.n_unique
    Series.has_validity

Boolean
-------
.. autosummary::
   :toctree: api/

    Series.any
    Series.all

Computations
------------
.. autosummary::
   :toctree: api/

    Series.cumsum
    Series.cummin
    Series.cummax
    Series.cumprod
    Series.arg_true
    Series.arg_unique
    Series.unique
    Series.rolling_min
    Series.rolling_max
    Series.rolling_mean
    Series.rolling_sum
    Series.rolling_apply
    Series.rolling_std
    Series.rolling_var
    Series.rolling_median
    Series.rolling_quantile
    Series.rolling_skew
    Series.ewm_mean
    Series.ewm_std
    Series.ewm_var
    Series.hash
    Series.peak_max
    Series.peak_min
    Series.dot
    Series.abs
    Series.rank
    Series.diff
    Series.pct_change
    Series.skew
    Series.kurtosis
    Series.sqrt
    Series.sin
    Series.cos
    Series.tan
    Series.arcsin
    Series.arccos
    Series.arctan
    Series.log
    Series.log10
    Series.exp

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    Series.alias
    Series.rename
    Series.limit
    Series.slice
    Series.append
    Series.filter
    Series.head
    Series.tail
    Series.take_every
    Series.sort
    Series.argsort
    Series.take
    Series.shrink_to_fit
    Series.explode
    Series.sample
    Series.view
    Series.set
    Series.clone
    Series.shift
    Series.shift_and_fill
    Series.drop_nulls
    Series.rechunk
    Series.cast
    Series.round
    Series.floor
    Series.ceil
    Series.set_at_idx
    Series.fill_null
    Series.fill_nan
    Series.zip_with
    Series.interpolate
    Series.clip
    Series.str_concat
    Series.reshape
    Series.to_dummies
    Series.shuffle
    Series.extend

Various
--------
.. autosummary::
   :toctree: api/

    Series.series_equal
    Series.apply
    Series.dt
    Series.str
    Series.reinterpret
    Series.to_physical

TimeSeries
----------
The following methods are available under the `Series.dt` attribute.

.. currentmodule:: polars.internals.series

.. autosummary::
   :toctree: api/

    DateTimeNameSpace.strftime
    DateTimeNameSpace.year
    DateTimeNameSpace.month
    DateTimeNameSpace.week
    DateTimeNameSpace.weekday
    DateTimeNameSpace.day
    DateTimeNameSpace.ordinal_day
    DateTimeNameSpace.hour
    DateTimeNameSpace.minute
    DateTimeNameSpace.second
    DateTimeNameSpace.nanosecond
    DateTimeNameSpace.timestamp
    DateTimeNameSpace.to_python_datetime
    DateTimeNameSpace.min
    DateTimeNameSpace.max
    DateTimeNameSpace.median
    DateTimeNameSpace.mean
    DateTimeNameSpace.truncate
    DateTimeNameSpace.epoch_days
    DateTimeNameSpace.epoch_milliseconds
    DateTimeNameSpace.epoch_seconds
    DateTimeNameSpace.and_time_unit
    DateTimeNameSpace.and_time_zone
    DateTimeNameSpace.days
    DateTimeNameSpace.hours
    DateTimeNameSpace.seconds
    DateTimeNameSpace.milliseconds
    DateTimeNameSpace.nanoseconds


Strings
-------

The following methods are available under the `Series.str` attribute.

.. currentmodule:: polars.internals.series

.. autosummary::
   :toctree: api/

    StringNameSpace.strptime
    StringNameSpace.lengths
    StringNameSpace.contains
    StringNameSpace.json_path_match
    StringNameSpace.extract
    StringNameSpace.replace
    StringNameSpace.replace_all
    StringNameSpace.to_lowercase
    StringNameSpace.to_uppercase
    StringNameSpace.strip
    StringNameSpace.rstrip
    StringNameSpace.lstrip
    StringNameSpace.slice
    StringNameSpace.encode
    StringNameSpace.decode

Lists
-----

The following methods are available under the `Series.arr` attribute.

.. currentmodule:: polars.internals.series

.. autosummary::
   :toctree: api/

    ListNameSpace.concat
    ListNameSpace.lengths
    ListNameSpace.sum
    ListNameSpace.min
    ListNameSpace.max
    ListNameSpace.mean
    ListNameSpace.sort
    ListNameSpace.reverse
    ListNameSpace.unique
    ListNameSpace.get
    ListNameSpace.first
    ListNameSpace.last
    ListNameSpace.contains
