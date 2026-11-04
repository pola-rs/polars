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

A file reader plugin consists of a :class:`~polars.io.external_reader.FileReaderBuilder` and a
:class:`~polars.io.external_reader.FileReader`. The function :func:`scan_external_reader` connects
a builder to a list of sources and returns a :class:`LazyFrame`. Polars creates one reader per
source and drives the reading itself. Scans over several sources therefore support row indices and
slices in the same way as the built-in file formats.

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

A :class:`~polars.lazyframe.resolver.LazyFrameResolver` defers building a scan until a query runs.
Polars passes it the projection, the row limit and the filters of the query. The resolver returns a
:class:`~polars.LazyFrame` that takes the place of the scan, for example one created with
:func:`~polars.scan_external_reader`. This suits table formats and catalogs, whose list of files
can change over time and can be narrowed down by the filters of a query.

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

A generator-based IO source is registered with :func:`~polars.io.plugins.register_io_source`. The
plugin provides a function that returns a generator of DataFrames for the whole source.

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
