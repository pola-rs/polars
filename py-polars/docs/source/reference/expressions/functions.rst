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
   any
   apply
   approx_unique
   arange
   arg_sort_by
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
   date
   datetime
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
   lit
   map
   max
   mean
   median
   min
   n_unique
   quantile
   reduce
   repeat
   rolling_corr
   rolling_cov
   select
   std
   struct
   sum
   sql_expr
   tail
   time
   var
   when


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
