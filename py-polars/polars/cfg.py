from __future__ import annotations

import os
import sys

from polars.string_cache import toggle_string_cache

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class Config:
    """Configure polars."""

    # class-local boolean flags can be used for options that don't have
    # a Rust component (so no need to register environment variables).
    with_columns_kwargs: bool = False

    @classmethod
    def set_tbl_hide_column_names(cls) -> type[Config]:
        """Hide column names of tables."""
        os.environ["POLARS_FMT_TABLE_HIDE_COLUMN_NAMES"] = "1"
        return cls

    @classmethod
    def set_tbl_hide_column_data_types(cls) -> type[Config]:
        """Hide column data types (i64, f64, str etc.) of tables."""
        os.environ["POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES"] = "1"
        return cls

    @classmethod
    def set_tbl_hide_column_separator(cls) -> type[Config]:
        """Hide the '---' separator that separates column names from the table rows."""
        os.environ["POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR"] = "1"
        return cls

    @classmethod
    def set_tbl_hide_dataframe_shape(cls) -> type[Config]:
        """Hide the shape information of the dataframe when displaying tables."""
        os.environ["POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION"] = "1"
        return cls

    @classmethod
    def set_tbl_change_column_data_type_position_format(cls) -> type[Config]:
        """Changes the data type position / format to directly below column name."""
        os.environ["POLARS_FMT_TABLE_CHANGE_COLUMN_DATA_TYPE_POSITION_FORMAT"] = "1"
        return cls

    @classmethod
    def set_utf8_tables(cls) -> type[Config]:
        """Use utf8 characters to print tables."""
        # os.unsetenv is automatically called if we remove a key from os.environ,
        # see https://docs.python.org/3/library/os.html#os.environ. However, we cannot
        # call os.unsetenv directly, as that fails on Windows
        os.environ["POLARS_FMT_TABLE_FORMATTING"] = "UTF8_FULL"
        return cls

    @classmethod
    def set_ascii_tables(cls) -> type[Config]:
        """Use ascii characters to print tables."""
        os.environ["POLARS_FMT_TABLE_FORMATTING"] = "ASCII_FULL"
        return cls

    @classmethod
    def set_tbl_formatting(
        cls,
        format: Literal[
            "ASCII_FULL",
            "ASCII_NO_BORDERS",
            "ASCII_BORDERS_ONLY",
            "ASCII_BORDERS_ONLY_CONDENSED",
            "ASCII_HORIZONTAL_ONLY",
            "ASCII_MARKDOWN",
            "UTF8_FULL",
            "UTF8_NO_BORDERS",
            "UTF8_BORDERS_ONLY",
            "UTF8_HORIZONTAL_ONLY",
            "NOTHING",
        ],
    ) -> type[Config]:
        """
        Set table formatting style.

        Args
        ----
            "ASCII_FULL": ASCII borders / lines
            "ASCII_NO_BORDERS": ASCII no borders
            "ASCII_BORDERS_ONLY": ASCII borders only
            "ASCII_BORDERS_ONLY_CONDENSED": ASCII borders only condensed
            "ASCII_HORIZONTAL_ONLY": Horizontal lines only
            "ASCII_MARKDOWN": Markdown style
            "UTF8_FULL": UTF8 borders lines
            "UTF8_NO_BORDERS": UTF8 no borders
            "UTF8_BORDERS_ONLY": UTF8 borders only
            "UTF8_HORIZONTAL_ONLY": UTF8 horizontal only
            "NOTHING": No borders /lines

        Raises
        ------
            KeyError: Wrong key

        """
        os.environ["POLARS_FMT_TABLE_FORMATTING"] = format
        return cls

    @classmethod
    def set_tbl_cell_alignment(
        cls, format: Literal["LEFT", "CENTER", "RIGHT"]
    ) -> type[Config]:
        """
        Set table cell alignment.

        Args
        ----
            "LEFT": left aligned
            "CENTER": center aligned
            "RIGHT": right aligned

        Raises
        ------
            KeyError: Wrong key

        """
        os.environ["POLARS_FMT_TABLE_CELL_ALIGNMENT"] = format
        return cls

    @classmethod
    def set_tbl_width_chars(cls, width: int) -> type[Config]:
        """
        Set the number of character used to draw the table.

        Parameters
        ----------
        width
            number of chars

        """
        os.environ["POLARS_TABLE_WIDTH"] = str(width)
        return cls

    @classmethod
    def set_tbl_rows(cls, n: int) -> type[Config]:
        """
        Set the number of rows used to print tables (Dataframe and Series).

        Parameters
        ----------
        n
            number of rows to print.
            If n<0 print all rows (DataFrame) and all elements (Series).

        """
        os.environ["POLARS_FMT_MAX_ROWS"] = str(n)
        return cls

    @classmethod
    def set_tbl_cols(cls, n: int) -> type[Config]:
        """
        Set the number of columns used to print tables.

        Parameters
        ----------
        n
            number of columns to print. If n<0 print all the columns.

        Examples
        --------
        >>> pl.cfg.Config.set_tbl_cols(5)  # doctest: +IGNORE_RESULT
        >>> df = pl.DataFrame({str(i): [i] for i in range(100)})
        >>> df
        shape: (1, 100)
        ┌─────┬─────┬─────┬─────┬─────┬─────┐
        │ 0   ┆ 1   ┆ 2   ┆ ... ┆ 98  ┆ 99  │
        │ --- ┆ --- ┆ --- ┆     ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 ┆     ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╪═════╪═════╡
        │ 0   ┆ 1   ┆ 2   ┆ ... ┆ 98  ┆ 99  │
        └─────┴─────┴─────┴─────┴─────┴─────┘

        >>> pl.cfg.Config.set_tbl_cols(10)  # doctest: +IGNORE_RESULT
        >>> df
        shape: (1, 100)
        ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
        │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ 4   ┆ ... ┆ 95  ┆ 96  ┆ 97  ┆ 98  ┆ 99  │
        │ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆     ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 ┆     ┆ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╡
        │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ 4   ┆ ... ┆ 95  ┆ 96  ┆ 97  ┆ 98  ┆ 99  │
        └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

        """
        os.environ["POLARS_FMT_MAX_COLS"] = str(n)
        return cls

    @classmethod
    def set_global_string_cache(cls) -> type[Config]:
        """Turn on the global string cache."""
        toggle_string_cache(True)
        return cls

    @classmethod
    def unset_global_string_cache(cls) -> type[Config]:
        """Turn off the global string cache."""
        toggle_string_cache(False)
        return cls

    @classmethod
    def set_fmt_str_lengths(cls, n: int) -> type[Config]:
        """
        Set the number of characters used to print string values.

        Parameters
        ----------
        n
            number of characters to print

        """
        os.environ["POLARS_FMT_STR_LEN"] = str(n)
        return cls
