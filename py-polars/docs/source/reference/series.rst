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
   Series.name
   Series.shape
   Series.dtype
   Series.dt
   Series.str

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
    Series.len
    Series.n_unique


Computations
------------
.. autosummary::
   :toctree: api/

    Series.cum_sum
    Series.cum_min
    Series.cum_max
    Series.arg_sort
    Series.arg_true
    Series.arg_unique
    Series.unique
    Series.rolling_min
    Series.rolling_max
    Series.rolling_mean
    Series.rolling_sum
    Series.rolling_apply
    Series.hash
    Series.peak_max
    Series.peak_min
    Series.dot

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

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
    Series.drop_nulls
    Series.rechunk
    Series.cast
    Series.round
    Series.set_at_idx
    Series.fill_none
    Series.zip_with
    Series.interpolate

Various
--------
.. autosummary::
   :toctree: api/

    Series.series_equal
    Series.apply
    Series.dt
    Series.str
    Series.reinterpret

TimeSeries
----------
The following methods are available under the `Series.dt` attribute.

.. currentmodule:: polars.eager.series

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
    DateTimeNameSpace.round


Strings
-------

The following methods are available under the `Series.str` attribute.

.. currentmodule:: polars.eager.series

.. autosummary::
   :toctree: api/

    StringNameSpace.strptime
    StringNameSpace.lengths
    StringNameSpace.contains
    StringNameSpace.json_path_match
    StringNameSpace.replace
    StringNameSpace.replace_all
    StringNameSpace.to_lowercase
    StringNameSpace.to_uppercase
    StringNameSpace.rstrip
    StringNameSpace.lstrip
    StringNameSpace.slice
