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
------------------

IO plugins allow you to register different file formats as sources to the Polars engines.

See the `IO plugins section of the user guide <https://docs.pola.rs/user-guide/plugins/io_plugins/>`_
for more information.

.. note::

    The ``io.plugins`` module is not imported by default in order to optimise import speed of
    the primary ``polars`` module. Either import ``polars.io.plugins`` and *then* use that
    namespace, or import ``register_io_source`` from the full module path, e.g.:

    .. code-block:: python

        from polars.io.plugins import register_io_source

.. autosummary::
    :toctree: api/

    io.plugins.register_io_source
