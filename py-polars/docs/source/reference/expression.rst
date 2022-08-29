===========
Expressions
===========
.. currentmodule:: polars


Functions
---------
These functions can be used as expression and sometimes also in eager contexts.

.. autosummary::
   :toctree: api/

   all
   any
   apply
   arange
   argsort_by
   avg
   col
   concat_list
   concat_str
   count
   cov
   date
   datetime
   duration
   element
   exclude
   first
   fold
   format
   groups
   head
   list
   lit
   map
   max
   mean
   median
   min
   n_unique
   pearson_corr
   quantile
   repeat
   select
   spearman_rank_corr
   std
   struct
   sum
   tail
   var
   when

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
   Expr.cat
   Expr.dt
   Expr.str


Aggregation
-----------
.. autosummary::
   :toctree: api/

    Expr.agg_groups
    Expr.arg_max
    Expr.arg_min
    Expr.count
    Expr.first
    Expr.last
    Expr.len
    Expr.list
    Expr.max
    Expr.mean
    Expr.median
    Expr.min
    Expr.product
    Expr.quantile
    Expr.std
    Expr.sum
    Expr.var

Boolean
-------
.. autosummary::
   :toctree: api/

    Expr.all
    Expr.any
    Expr.is_between
    Expr.is_duplicated
    Expr.is_finite
    Expr.is_first
    Expr.is_in
    Expr.is_infinite
    Expr.is_nan
    Expr.is_not
    Expr.is_not_nan
    Expr.is_not_null
    Expr.is_null
    Expr.is_unique


Computations
------------
.. autosummary::
   :toctree: api/

    Expr.abs
    Expr.arccos
    Expr.arccosh
    Expr.arcsin
    Expr.arcsinh
    Expr.arctan
    Expr.arctanh
    Expr.arg_unique
    Expr.cos
    Expr.cosh
    Expr.cumcount
    Expr.cummax
    Expr.cummin
    Expr.cumprod
    Expr.cumsum
    Expr.cumulative_eval
    Expr.diff
    Expr.dot
    Expr.entropy
    Expr.ewm_mean
    Expr.ewm_std
    Expr.ewm_var
    Expr.exp
    Expr.hash
    Expr.kurtosis
    Expr.log
    Expr.log10
    Expr.mode
    Expr.n_unique
    Expr.null_count
    Expr.pct_change
    Expr.pow
    Expr.rank
    Expr.rolling_apply
    Expr.rolling_max
    Expr.rolling_mean
    Expr.rolling_median
    Expr.rolling_min
    Expr.rolling_quantile
    Expr.rolling_skew
    Expr.rolling_std
    Expr.rolling_sum
    Expr.rolling_var
    Expr.search_sorted
    Expr.sign
    Expr.sin
    Expr.sinh
    Expr.skew
    Expr.sqrt
    Expr.tan
    Expr.tanh
    Expr.unique
    Expr.unique_counts
    Expr.value_counts

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    Expr.append
    Expr.arg_sort
    Expr.argsort
    Expr.backward_fill
    Expr.cast
    Expr.ceil
    Expr.clip
    Expr.clip_max
    Expr.clip_min
    Expr.drop_nans
    Expr.drop_nulls
    Expr.explode
    Expr.extend_constant
    Expr.fill_nan
    Expr.fill_null
    Expr.filter
    Expr.flatten
    Expr.floor
    Expr.forward_fill
    Expr.head
    Expr.inspect
    Expr.interpolate
    Expr.limit
    Expr.lower_bound
    Expr.rechunk
    Expr.reinterpret
    Expr.repeat_by
    Expr.reshape
    Expr.reverse
    Expr.round
    Expr.sample
    Expr.shift
    Expr.shift_and_fill
    Expr.shuffle
    Expr.slice
    Expr.sort
    Expr.sort_by
    Expr.tail
    Expr.take
    Expr.take_every
    Expr.to_physical
    Expr.top_k
    Expr.upper_bound
    Expr.where

Column names
------------
   Expressions that help renaming/ selecting columns by name.

   A wildcard ``col("*")``/:func:`polars.all()` selects all columns in a DataFrame.

   >>> df.select(pl.all())

.. autosummary::
   :toctree: api/

    Expr.alias
    Expr.exclude
    Expr.keep_name
    Expr.map_alias
    Expr.prefix
    Expr.suffix

Apply
-----
.. autosummary::
   :toctree: api/

    Expr.apply
    Expr.map

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

.. currentmodule:: polars.internals.expr.datetime

.. autosummary::
   :toctree: api/

    ExprDateTimeNameSpace.cast_time_unit
    ExprDateTimeNameSpace.day
    ExprDateTimeNameSpace.days
    ExprDateTimeNameSpace.epoch
    ExprDateTimeNameSpace.hour
    ExprDateTimeNameSpace.hours
    ExprDateTimeNameSpace.milliseconds
    ExprDateTimeNameSpace.minute
    ExprDateTimeNameSpace.minutes
    ExprDateTimeNameSpace.month
    ExprDateTimeNameSpace.nanosecond
    ExprDateTimeNameSpace.nanoseconds
    ExprDateTimeNameSpace.ordinal_day
    ExprDateTimeNameSpace.offset_by
    ExprDateTimeNameSpace.quarter
    ExprDateTimeNameSpace.second
    ExprDateTimeNameSpace.seconds
    ExprDateTimeNameSpace.strftime
    ExprDateTimeNameSpace.timestamp
    ExprDateTimeNameSpace.truncate
    ExprDateTimeNameSpace.week
    ExprDateTimeNameSpace.weekday
    ExprDateTimeNameSpace.with_time_unit
    ExprDateTimeNameSpace.year

Strings
-------

The following methods are available under the `Expr.str` attribute.

.. currentmodule:: polars.internals.expr.string

.. autosummary::
   :toctree: api/

    ExprStringNameSpace.concat
    ExprStringNameSpace.contains
    ExprStringNameSpace.count_match
    ExprStringNameSpace.decode
    ExprStringNameSpace.encode
    ExprStringNameSpace.ends_with
    ExprStringNameSpace.extract
    ExprStringNameSpace.extract_all
    ExprStringNameSpace.json_path_match
    ExprStringNameSpace.lengths
    ExprStringNameSpace.ljust
    ExprStringNameSpace.lstrip
    ExprStringNameSpace.replace
    ExprStringNameSpace.replace_all
    ExprStringNameSpace.rjust
    ExprStringNameSpace.rstrip
    ExprStringNameSpace.slice
    ExprStringNameSpace.split
    ExprStringNameSpace.split_exact
    ExprStringNameSpace.splitn
    ExprStringNameSpace.starts_with
    ExprStringNameSpace.strip
    ExprStringNameSpace.strptime
    ExprStringNameSpace.to_lowercase
    ExprStringNameSpace.to_uppercase
    ExprStringNameSpace.zfill

Lists
-----
The following methods are available under the `expr.arr` attribute.

.. currentmodule:: polars.internals.expr.list

.. autosummary::
   :toctree: api/

    ExprListNameSpace.arg_max
    ExprListNameSpace.arg_min
    ExprListNameSpace.concat
    ExprListNameSpace.contains
    ExprListNameSpace.diff
    ExprListNameSpace.eval
    ExprListNameSpace.first
    ExprListNameSpace.get
    ExprListNameSpace.head
    ExprListNameSpace.join
    ExprListNameSpace.last
    ExprListNameSpace.lengths
    ExprListNameSpace.max
    ExprListNameSpace.mean
    ExprListNameSpace.min
    ExprListNameSpace.reverse
    ExprListNameSpace.shift
    ExprListNameSpace.slice
    ExprListNameSpace.sort
    ExprListNameSpace.sum
    ExprListNameSpace.tail
    ExprListNameSpace.to_struct
    ExprListNameSpace.unique

Categories
----------
The following methods are available under the `expr.cat` attribute.

.. currentmodule:: polars.internals.expr.categorical

.. autosummary::
   :toctree: api/

    ExprCatNameSpace.set_ordering

Struct
------
The following methods are available under the `expr.struct` attribute.

.. currentmodule:: polars.internals.expr.struct

.. autosummary::
   :toctree: api/

    ExprStructNameSpace.field
    ExprStructNameSpace.rename_fields


Meta
----
The following methods are available under the `expr.meta` attribute.

.. currentmodule:: polars.internals.expr.meta

.. autosummary::
   :toctree: api/

    ExprMetaNameSpace.output_name
    ExprMetaNameSpace.pop
    ExprMetaNameSpace.root_names
    ExprMetaNameSpace.undo_aliases
