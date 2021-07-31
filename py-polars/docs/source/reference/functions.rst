=================
Functions
=================
.. currentmodule:: polars

Config
~~~~~~
.. autosummary::
   :toctree: api/

    toggle_string_cache
    StringCache

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

    from_dict
    from_records
    from_arrow
    from_pandas

Eager functions
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   get_dummies
   concat
   repeat
   arg_where

Expression functions
~~~~~~~~~~~~~~~~~~~~
These functions can be used as expression and sometimes also in eager contexts.

.. autosummary::
   :toctree: api/

   col
   except_
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
