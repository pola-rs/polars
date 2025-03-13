======
Config
======
.. currentmodule:: polars

Config options
--------------

.. autosummary::
   :toctree: api/

    Config.set_ascii_tables
    Config.set_auto_structify
    Config.set_decimal_separator
    Config.set_engine_affinity
    Config.set_float_precision
    Config.set_fmt_float
    Config.set_fmt_str_lengths
    Config.set_fmt_table_cell_list_len
    Config.set_streaming_chunk_size
    Config.set_tbl_cell_alignment
    Config.set_tbl_cell_numeric_alignment
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
    Config.set_thousands_separator
    Config.set_trim_decimal_zeros
    Config.set_verbose

Config load, save, state
------------------------
.. autosummary::
   :toctree: api/

    Config.load
    Config.load_from_file
    Config.save
    Config.save_to_file
    Config.state
    Config.restore_defaults

While it is easy to restore *all* configuration options to their default
value using ``restore_defaults``, it can also be useful to reset *individual*
options. This can be done by setting the related value to ``None``, eg:

.. code-block:: python

    pl.Config.set_tbl_rows(None)


Use as a context manager
------------------------

Note that ``Config`` supports setting context-scoped options. These options
are valid *only* during scope lifetime, and are reset to their initial values
(whatever they were before entering the new context) on scope exit.

You can take advantage of this by initialising a ``Config`` instance and then
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

Use as a decorator
------------------

In the same vein, you can also use a ``Config`` instance as a function decorator
to temporarily set options for the duration of the function call:

.. code-block:: python

    cfg_ascii_frames = pl.Config(ascii_tables=True, apply_on_context_enter=True)

    @cfg_ascii_frames
    def write_markdown_frame_to_stdout(df: pl.DataFrame) -> None:
        sys.stdout.write(str(df))

Multiple Config instances
-------------------------
You may want to establish related bundles of `Config` options for use in different
parts of your code. Usually options are set immediately on `Config` init, meaning
the `Config` instance cannot be reused; however, you can defer this so that options
are only invoked when entering context scope (which includes function entry if used
as a decorator)._

This allows you to create multiple *reusable* `Config` instances in one place, update
and modify them centrally, and apply them as needed throughout your codebase.

.. code-block:: python

    cfg_verbose = pl.Config(verbose=True, apply_on_context_enter=True)
    cfg_markdown = pl.Config(tbl_formatting="MARKDOWN", apply_on_context_enter=True)

    @cfg_markdown
    def write_markdown_frame_to_stdout(df: pl.DataFrame) -> None:
        sys.stdout.write(str(df))

    @cfg_verbose
    def do_various_things():
        ...
