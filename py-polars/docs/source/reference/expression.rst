===========
Expressions
===========
.. currentmodule:: polars


Functions
---------
These functions can be used as expression and sometimes also in eager contexts.

.. autosummary::
   :toctree: api/

   select
   col
   element
   count
   list
   std
   var
   max
   min
   sum
   mean
   avg
   median
   n_unique
   first
   last
   head
   tail
   lit
   pearson_corr
   spearman_rank_corr
   cov
   map
   apply
   fold
   any
   all
   groups
   quantile
   arange
   repeat
   argsort_by
   concat_str
   concat_list
   format
   when
   exclude
   datetime
   duration
   date
   struct

Constructor
-----------
.. autosummary::
   :toctree: api/

   Expr

Attributes
----------

.. autosummary::
   :toctree: api/

   Expr.arr
   Expr.dt
   Expr.str
   Expr.cat


Aggregation
-----------
.. autosummary::
   :toctree: api/

    Expr.std
    Expr.var
    Expr.max
    Expr.min
    Expr.sum
    Expr.mean
    Expr.mean
    Expr.median
    Expr.first
    Expr.last
    Expr.product
    Expr.list
    Expr.agg_groups
    Expr.count
    Expr.len
    Expr.quantile
    Expr.arg_min
    Expr.arg_max

Boolean
-------
.. autosummary::
   :toctree: api/

    Expr.is_not
    Expr.is_null
    Expr.is_not_null
    Expr.is_finite
    Expr.is_infinite
    Expr.is_nan
    Expr.is_not_nan
    Expr.is_unique
    Expr.is_first
    Expr.is_duplicated
    Expr.is_between
    Expr.is_in
    Expr.any
    Expr.all


Computations
------------
.. autosummary::
   :toctree: api/

    Expr.cumsum
    Expr.cummin
    Expr.cummax
    Expr.cumprod
    Expr.cumcount
    Expr.cumulative_eval
    Expr.dot
    Expr.mode
    Expr.n_unique
    Expr.arg_unique
    Expr.unique
    Expr.pow
    Expr.rolling_min
    Expr.rolling_max
    Expr.rolling_mean
    Expr.rolling_sum
    Expr.rolling_apply
    Expr.rolling_std
    Expr.rolling_var
    Expr.rolling_median
    Expr.rolling_quantile
    Expr.rolling_skew
    Expr.ewm_mean
    Expr.ewm_std
    Expr.ewm_var
    Expr.hash
    Expr.abs
    Expr.rank
    Expr.diff
    Expr.pct_change
    Expr.skew
    Expr.kurtosis
    Expr.entropy
    Expr.sqrt
    Expr.sin
    Expr.cos
    Expr.tan
    Expr.arcsin
    Expr.arccos
    Expr.arctan
    Expr.log
    Expr.log10
    Expr.exp
    Expr.sign
    Expr.unique_counts
    Expr.value_counts
    Expr.null_count

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    Expr.inspect
    Expr.slice
    Expr.explode
    Expr.flatten
    Expr.take_every
    Expr.repeat_by
    Expr.round
    Expr.floor
    Expr.ceil
    Expr.cast
    Expr.sort
    Expr.arg_sort
    Expr.argsort
    Expr.sort_by
    Expr.take
    Expr.shift
    Expr.shift_and_fill
    Expr.fill_null
    Expr.fill_nan
    Expr.forward_fill
    Expr.backward_fill
    Expr.reverse
    Expr.filter
    Expr.where
    Expr.head
    Expr.tail
    Expr.reinterpret
    Expr.drop_nulls
    Expr.drop_nans
    Expr.interpolate
    Expr.arg_sort
    Expr.clip
    Expr.lower_bound
    Expr.upper_bound
    Expr.reshape
    Expr.to_physical
    Expr.shuffle
    Expr.sample
    Expr.extend_constant

Column names
------------
   Expressions that help renaming/ selecting columns by name.

   A wildcard `col("*")`/`pl.all()` selects all columns in a DataFrame.

   >>> df.select(pl.all())

.. autosummary::
   :toctree: api/

    Expr.alias
    Expr.keep_name
    Expr.prefix
    Expr.suffix
    Expr.map_alias
    Expr.exclude

Apply
-----
.. autosummary::
   :toctree: api/

    Expr.map
    Expr.apply

Window
------
.. autosummary::
   :toctree: api/

    Expr.over

Various
--------
.. autosummary::
   :toctree: api/

    Expr.set_sorted

TimeSeries
----------
The following methods are available under the `expr.dt` attribute.

.. currentmodule:: polars.internals.expr

.. autosummary::
   :toctree: api/

    ExprDateTimeNameSpace.strftime
    ExprDateTimeNameSpace.year
    ExprDateTimeNameSpace.month
    ExprDateTimeNameSpace.week
    ExprDateTimeNameSpace.weekday
    ExprDateTimeNameSpace.day
    ExprDateTimeNameSpace.ordinal_day
    ExprDateTimeNameSpace.hour
    ExprDateTimeNameSpace.minute
    ExprDateTimeNameSpace.second
    ExprDateTimeNameSpace.nanosecond
    ExprDateTimeNameSpace.to_python_datetime
    ExprDateTimeNameSpace.timestamp
    ExprDateTimeNameSpace.truncate
    ExprDateTimeNameSpace.epoch
    ExprDateTimeNameSpace.epoch_days
    ExprDateTimeNameSpace.epoch_milliseconds
    ExprDateTimeNameSpace.epoch_seconds
    ExprDateTimeNameSpace.and_time_unit
    ExprDateTimeNameSpace.and_time_zone
    ExprDateTimeNameSpace.with_time_unit
    ExprDateTimeNameSpace.cast_time_unit
    ExprDateTimeNameSpace.days
    ExprDateTimeNameSpace.hours
    ExprDateTimeNameSpace.minutes
    ExprDateTimeNameSpace.seconds
    ExprDateTimeNameSpace.milliseconds
    ExprDateTimeNameSpace.nanoseconds

Strings
-------

The following methods are available under the `Expr.str` attribute.

.. currentmodule:: polars.internals.expr

.. autosummary::
   :toctree: api/

    ExprStringNameSpace.strptime
    ExprStringNameSpace.lengths
    ExprStringNameSpace.to_uppercase
    ExprStringNameSpace.to_lowercase
    ExprStringNameSpace.concat
    ExprStringNameSpace.strip
    ExprStringNameSpace.lstrip
    ExprStringNameSpace.rstrip
    ExprStringNameSpace.contains
    ExprStringNameSpace.json_path_match
    ExprStringNameSpace.extract
    ExprStringNameSpace.split
    ExprStringNameSpace.split_exact
    ExprStringNameSpace.replace
    ExprStringNameSpace.replace_all
    ExprStringNameSpace.slice
    ExprStringNameSpace.encode
    ExprStringNameSpace.decode

Lists
-----
The following methods are available under the `expr.arr` attribute.

.. currentmodule:: polars.internals.expr

.. autosummary::
   :toctree: api/

    ExprListNameSpace.concat
    ExprListNameSpace.lengths
    ExprListNameSpace.sum
    ExprListNameSpace.min
    ExprListNameSpace.max
    ExprListNameSpace.mean
    ExprListNameSpace.sort
    ExprListNameSpace.reverse
    ExprListNameSpace.unique
    ExprListNameSpace.get
    ExprListNameSpace.first
    ExprListNameSpace.last
    ExprListNameSpace.contains
    ExprListNameSpace.join
    ExprListNameSpace.arg_min
    ExprListNameSpace.arg_max
    ExprListNameSpace.diff
    ExprListNameSpace.shift
    ExprListNameSpace.slice
    ExprListNameSpace.head
    ExprListNameSpace.tail
    ExprListNameSpace.to_struct
    ExprListNameSpace.eval

Categories
----------
The following methods are available under the `expr.cat` attribute.

.. currentmodule:: polars.internals.expr

.. autosummary::
   :toctree: api/

    ExprCatNameSpace.set_ordering

Struct
------
The following methods are available under the `expr.struct` attribute.

.. currentmodule:: polars.internals.expr

.. autosummary::
   :toctree: api/

    ExprStructNameSpace.field
    ExprStructNameSpace.rename_fields
