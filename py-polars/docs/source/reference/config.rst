============
Config
============
.. currentmodule:: polars

Config options
--------------

.. autosummary::
   :toctree: api/

    Config.set_ascii_tables
    Config.set_fmt_float
    Config.set_fmt_str_lengths
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

.. code-block:: python

    with pl.Config() as cfg:
        cfg.set_verbose(True)
        do_various_things()

    # on scope exit any modified settings are restored to their previous state
