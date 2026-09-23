=======
Engines
=======

These objects select and configure the engine used to execute a query when
calling `LazyFrame.collect()`, `LazyFrame.execute()` or a `LazyFrame.sink_*`
method with an `engine` argument, or when set through
:meth:`Config.set_engine_affinity`.

.. currentmodule:: polars

.. autosummary::
   :toctree: api/

    Engine
    InMemoryEngine
    StreamingEngine
    RemoteEngine

GPUEngine
---------

This object provides fine-grained control over the behavior of the
GPU engine.

.. currentmodule:: polars.lazyframe.engine_config

.. autosummary::
   :toctree: api/

    GPUEngine
