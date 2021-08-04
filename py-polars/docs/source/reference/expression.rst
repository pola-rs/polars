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

    Expr.cum_sum
    Expr.cum_min
    Expr.cum_max
    Expr.dot
    Expr.mode
    Expr.n_unique
    Expr.arg_unique
    Expr.unique
    Expr.pow
    Expr.hash

Manipulation/ selection
-----------------------
.. autosummary::
   :toctree: api/

    Expr.slice
    Expr.explode
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
    Expr.fill_none
    Expr.forward_fill
    Expr.backward_fill
    Expr.reverse
    Expr.filter
    Expr.head
    Expr.tail
    Expr.reinterpret
    Expr.drop_nulls

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
    ExprDateTimeNameSpace.day
    ExprDateTimeNameSpace.ordinal_day
    ExprDateTimeNameSpace.hour
    ExprDateTimeNameSpace.minute
    ExprDateTimeNameSpace.second
    ExprDateTimeNameSpace.nanosecond
    ExprDateTimeNameSpace.round

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
