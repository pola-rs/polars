===========
Expressions
===========
.. currentmodule:: polars


Functions
---------
These functions can be used as expression and sometimes also in eager contexts.

.. autosummary::
   :toctree: api/

   col
   count
   to_list
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
   lit_date
   lit
   pearson_corr
   cov
   map_binary
   fold
   any
   all
   groups
   quantile
   arange
   argsort_by
   concat_str
   concat_list
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

   Expr.dt
   Expr.str


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
    Expr.median
    Expr.first
    Expr.last
    Expr.list
    Expr.agg_groups
    Expr.count
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


Computations
------------
.. autosummary::
   :toctree: api/

    Expr.cumsum
    Expr.cummin
    Expr.cummax
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
    Expr.hash
    Expr.abs
    Expr.rank
    Expr.diff
    Expr.skew
    Expr.kurtosis
    Expr.sqrt

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
    Expr.cast
    Expr.sort
    Expr.arg_sort
    Expr.sort_by
    Expr.take
    Expr.shift
    Expr.shift_and_fill
    Expr.fill_null
    Expr.forward_fill
    Expr.backward_fill
    Expr.reverse
    Expr.filter
    Expr.head
    Expr.tail
    Expr.reinterpret
    Expr.drop_nulls
    Expr.interpolate
    Expr.argsort
    Expr.clip

Column names
------------
   Expressions that help renaming/ selecting columns by name.

   A wildcard `col("*")` selects all columns in a DataFrame.

   Examples
   --------

   >>> df.select(col("*"))

.. autosummary::
   :toctree: api/

    Expr.alias
    Expr.keep_name
    Expr.prefix
    Expr.suffix
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

TimeSeries
----------
The following methods are available under the `expr.dt` attribute.

.. currentmodule:: polars.lazy.expr

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
    ExprDateTimeNameSpace.round
    ExprDateTimeNameSpace.to_python_datetime
    ExprDateTimeNameSpace.timestamp

Strings
-------

The following methods are available under the `Expr.str` attribute.

.. currentmodule:: polars.lazy.expr

.. autosummary::
   :toctree: api/

    ExprStringNameSpace.strptime
    ExprStringNameSpace.lengths
    ExprStringNameSpace.to_uppercase
    ExprStringNameSpace.to_lowercase
    ExprStringNameSpace.contains
    ExprStringNameSpace.json_path_match
    ExprStringNameSpace.replace
    ExprStringNameSpace.replace_all
    ExprStringNameSpace.slice

Lists
-----
The following methods are available under the `expr.arr` attribute.

.. currentmodule:: polars.lazy.expr

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
