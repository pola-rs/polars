=========
Functions
=========

These functions are available from the Polars module root and can be used as expressions, and sometimes also in eager contexts.

----

**Available in module namespace:**

.. currentmodule:: polars
.. autosummary::
   :toctree: api/

   all
   all_horizontal
   any
   any_horizontal
   approx_n_unique
   arange
   arctan2
   arctan2d
   arg_sort_by
   arg_where
   business_day_count
   coalesce
   concat_list
   concat_str
   corr
   count
   cov
   cum_count
   cum_fold
   cum_reduce
   cum_sum
   cum_sum_horizontal
   date
   datetime
   date_range
   date_ranges
   datetime_range
   datetime_ranges
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
   len
   lit
   map_batches
   map_groups
   max
   max_horizontal
   mean
   mean_horizontal
   median
   min
   min_horizontal
   n_unique
   nth
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
   sql
   sql_expr
   tail
   time
   time_range
   time_ranges
   var
   when
   zeros


**Available in expression namespace:**

.. autosummary::
   :toctree: api/

   Expr.all
   Expr.any
   Expr.approx_n_unique
   Expr.count
   Expr.exclude
   Expr.first
   Expr.head
   Expr.implode
   Expr.map_batches
   Expr.map_elements
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
