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
    ~lazyframe.engine_config.GPUEngine
    RemoteEngine
