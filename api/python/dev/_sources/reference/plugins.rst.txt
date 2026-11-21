=======
Plugins
=======
.. currentmodule:: polars

Polars allows you to extend its functionality with either Expression plugins or IO plugins.
See the `user guide <https://docs.pola.rs/user-guide/plugins/>`_ for more information and resources.

Expression plugins
------------------

Expression plugins are the preferred way to create user defined functions. They allow you to compile
a Rust function and register that as an expression into the Polars library. The Polars engine will
dynamically link your function at runtime and your expression will run almost as fast as native
expressions. Note that this works without any interference of Python and thus no GIL contention.

See the `expression plugins section of the user guide <https://docs.pola.rs/user-guide/plugins/expr_plugins/>`_
for more information.

.. autosummary::
    :toctree: api/

    plugins.register_plugin_function


IO plugins
----------

IO plugins allow you to register different file formats as sources to the Polars engines.

See the `IO plugins section of the user guide <https://docs.pola.rs/user-guide/plugins/io_plugins/>`_
for more information.

File readers
~~~~~~~~~~~~

.. warning::
    This functionality is considered **unstable**. It may be changed
    at any point without it being considered a breaking change.

A :class:`~polars.io.external_reader.FileReaderBuilder` creates one
:class:`~polars.io.external_reader.FileReader` per source, and :func:`scan_external_reader` turns a
builder and a list of sources into a :class:`LazyFrame`.

.. autosummary::
    :toctree: api/

    scan_external_reader

.. currentmodule:: polars.io.external_reader

.. autosummary::
    :toctree: api/

    FileReaderBuilder
    FileReaderBuilder.build_file_reader
    FileReaderBuilder.reader_capabilities
    FileReaderBuilder.explain_properties
    FileReader
    FileReader.schema
    FileReader.n_rows_in_file
    FileReader.fetch_metadata
    FileReader.initialize_serial
    FileReader.collect_batches
    FileReader.on_drop
    ReaderCapabilities

LazyFrame resolvers
~~~~~~~~~~~~~~~~~~~

.. warning::
    This functionality is considered **unstable**. It may be changed
    at any point without it being considered a breaking change.

A :class:`~polars.lazyframe.resolver.LazyFrameResolver` defers resolving the paths to scan until a
query runs. The resolver sees the columns, row limit and filters that the query requests, and
returns the :class:`~polars.LazyFrame` to scan.

.. currentmodule:: polars.lazyframe.resolver

.. autosummary::

    LazyFrameResolver
    LazyFrameResolver.lazy
    LazyFrameResolver.schema
    LazyFrameResolver.resolve_lazyframe
    LazyFrameResolver.cse_eq
    FilterExpr
    FilterExpr.pyarrow_expr
    ResolvedLazyFrameProps

.. currentmodule:: polars

Generator-based IO sources
~~~~~~~~~~~~~~~~~~~~~~~~~~

A generator-based IO source provides one generator of DataFrames for the whole source.

.. note::

    The ``io.plugins`` module is not imported by default in order to optimise import speed of
    the primary ``polars`` module. Either import ``polars.io.plugins`` and *then* use that
    namespace, or import ``register_io_source`` from the full module path, e.g.:

    .. code-block:: python

        from polars.io.plugins import register_io_source

.. autosummary::
    :toctree: api/

    io.plugins.register_io_source
    io.plugins.IOSourceScanFunction
    io.plugins.IOSourceScanFunction.io_source
