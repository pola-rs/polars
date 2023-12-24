========
Plotting
========

Polars doesn't implementing plotting logic itself. Instead, it defer to plotting
backends which implement the `Dataframe Plotting API <https://github.com/data-apis/dataframe-plotting-api>`_.

The following methods are available under the `DataFrame.plot` attribute.


.. currentmodule:: polars
.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   DataFrame.plot.line
