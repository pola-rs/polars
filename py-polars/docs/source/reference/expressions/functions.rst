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
   arange
   argsort_by
   avg
   coalesce
   concat_list
   concat_str
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
   reduce
   repeat
   select
   spearman_rank_corr
   std
   struct
   sum
   tail
   var
   when


**Available in expression namespace:**

.. autosummary::
   :toctree: api/

   Expr.all
   Expr.any
   Expr.apply
   Expr.count
   Expr.cumsum
   Expr.exclude
   Expr.first
   Expr.head
   Expr.list
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
