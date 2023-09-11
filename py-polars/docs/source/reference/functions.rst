=========
Functions
=========
.. currentmodule:: polars

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

    from_arrow
    from_dataframe
    from_dict
    from_dicts
    from_numpy
    from_pandas
    from_records
    from_repr

Miscellaneous
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

    align_frames
    concat

Parallelization
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   collect_all
   collect_all_async

Random
~~~~~~
.. autosummary::
   :toctree: api/

   set_random_seed

StringCache
~~~~~~~~~~~

Note that the `StringCache` can be used as both a context manager
and a decorator, in order to explicitly scope cache lifetime.

.. autosummary::
   :toctree: api/

    StringCache
    enable_string_cache
    using_string_cache
