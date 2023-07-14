=========
Functions
=========

These functions are available from the polars module root and can be used as expressions, and sometimes also in eager contexts.

----

**Available in module namespace:**

.. currentmodule:: polars
.. autosummary::
   :toctree: api/

   all
   all_horizontal
   any
   any_horizontal
   apply
   approx_unique
   arange
   arg_sort_by
   arg_where
   avg
   coalesce
   concat_list
   concat_str
   corr
   count
   cov
   cumfold
   cumreduce
   cumsum
   cumsum_horizontal
   date
   datetime
   date_range
   duration
   element
   exclude
   first
   fold
   format
   from_epoch
   groups
   head
   implode
   int_range
   int_ranges
   last
   lit
   map
   max
   max_horizontal
   mean
   median
   min
   min_horizontal
   n_unique
   ones
   quantile
   reduce
   repeat
   rolling_corr
   rolling_cov
   select
   std
   struct
   sum
   sum_horizontal
   sql_expr
   tail
   time
   time_range
   var
   when
   zeros


**Available in expression namespace:**

.. autosummary::
   :toctree: api/

   Expr.all
   Expr.any
   Expr.apply
   Expr.approx_unique
   Expr.count
   Expr.cumsum
   Expr.exclude
   Expr.first
   Expr.head
   Expr.implode
   Expr.map
   Expr.max
   Expr.mean
   Expr.median
   Expr.min
   Expr.n_unique
   Expr.quantile
   Expr.std
   Expr.sum
   Expr.tail
   Expr.var
