=================
Extending the API
=================
.. currentmodule:: polars

Providing new functionality
---------------------------

These functions allow you to register custom functionality in a dedicated
namespace on the underlying polars classes without requiring subclassing
or mixins. Expr, DataFrame, LazyFrame, and Series are all supported targets.

This feature is primarily intended for use by library authors providing
domain-specific capabilities which may not exist (or belong) in the
core library.

Available registrations
-----------------------

.. currentmodule:: polars.api
.. autosummary::
   :toctree: api/

    register_expr_namespace
    register_dataframe_namespace
    register_lazyframe_namespace
    register_series_namespace

.. note::

   You cannot override existing polars namespaces (such as ``.str`` or ``.dt``), and attempting to do so
   will raise an `AttributeError <https://docs.python.org/3/library/exceptions.html#AttributeError>`_.
   However, you *can* override other custom namespaces (which will only generate a
   `UserWarning <https://docs.python.org/3/library/exceptions.html#UserWarning>`_).
