import os
from typing import Type

import polars as pl

__all__ = [
    "Config",
]


class Config:
    "Configure polars"

    @classmethod
    def set_utf8_tables(cls) -> "Type[Config]":
        """
        Use utf8 characters to print tables
        """
        os.environ.unsetenv("POLARS_FMT_NO_UTF8")  # type: ignore
        return cls

    @classmethod
    def set_ascii_tables(cls) -> "Type[Config]":
        """
        Use ascii characters to print tables
        """
        os.environ["POLARS_FMT_NO_UTF8"] = "1"
        return cls

    @classmethod
    def set_tbl_width_chars(cls, width: int) -> "Type[Config]":
        """
        Set the number of character used to draw the table

        Parameters
        ----------
        width
            number of chars
        """
        os.environ["POLARS_TABLE_WIDTH"] = str(width)
        return cls

    @classmethod
    def set_tbl_rows(cls, n: int) -> "Type[Config]":
        """
        Set the number of rows used to print tables

        Parameters
        ----------
        n
            number of rows to print
        """

        os.environ["POLARS_FMT_MAX_ROWS"] = str(n)
        return cls

    @classmethod
    def set_tbl_cols(cls, n: int) -> "Type[Config]":
        """
        Set the number of columns used to print tables

        Parameters
        ----------
        n
            number of columns to print
        """

        os.environ["POLARS_FMT_MAX_COLS"] = str(n)
        return cls

    @classmethod
    def set_global_string_cache(cls) -> "Type[Config]":
        """
        Turn on the global string cache
        """
        pl.toggle_string_cache(True)
        return cls

    @classmethod
    def unset_global_string_cache(cls) -> "Type[Config]":
        """
        Turn off the global string cache
        """
        pl.toggle_string_cache(False)
        return cls
