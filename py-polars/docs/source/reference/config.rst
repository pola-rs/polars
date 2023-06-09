============
Config
============
.. currentmodule:: polars

Config options
--------------

.. autosummary::
   :toctree: api/

    Config.activate_decimals
    Config.set_ascii_tables
    Config.set_fmt_float
    Config.set_fmt_str_lengths
    Config.set_streaming_chunk_size
    Config.set_tbl_cell_alignment
    Config.set_tbl_cols
    Config.set_tbl_column_data_type_inline
    Config.set_tbl_dataframe_shape_below
    Config.set_tbl_formatting
    Config.set_tbl_hide_column_data_types
    Config.set_tbl_hide_column_names
    Config.set_tbl_hide_dataframe_shape
    Config.set_tbl_hide_dtype_separator
    Config.set_tbl_rows
    Config.set_tbl_width_chars
    Config.set_verbose

Config load, save, and current state
------------------------------------
.. autosummary::
   :toctree: api/

    Config.load
    Config.save
    Config.state
    Config.restore_defaults

Use as a context manager
------------------------

Note that ``Config`` supports setting context-scoped options. These options
are valid *only* during scope lifetime, and are reset to their initial values
(whatever they were before entering the new context) on scope exit.

You can take advantage of this by initialising  a``Config`` instance and then
explicitly calling one or more of the available "set\_" methods on it...

.. code-block:: python

    with pl.Config() as cfg:
        cfg.set_verbose(True)
        do_various_things()

    # on scope exit any modified settings are restored to their previous state

...or, often cleaner, by setting the options in the ``Config`` init directly
(optionally omitting the "set\_" prefix for brevity):

.. code-block:: python

    with pl.Config(verbose=True):
        do_various_things()

Use as a function decorator
---------------------------

In the same vein, you can also use ``Config`` as a function decorator to
temporarily set options for the duration of the function call:

.. code-block:: python

    @pl.Config(set_ascii_tables=True)
    def write_ascii_frame_to_stdout(df: pl.DataFrame) -> None:
        sys.stdout.write(str(df))

"""
