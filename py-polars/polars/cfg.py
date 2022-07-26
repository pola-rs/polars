from __future__ import annotations

import os

from polars.string_cache import toggle_string_cache


class Config:
    """Configure polars."""

    # class-local boolean flags can be used for options that don't have
    # a Rust component (so no need to register environment variables).
    with_columns_kwargs: bool = False

    @classmethod
    def set_utf8_tables(cls) -> type[Config]:
        """Use utf8 characters to print tables."""
        # os.unsetenv is automatically called if we remove a key from os.environ,
        # see https://docs.python.org/3/library/os.html#os.environ. However, we cannot
        # call os.unsetenv directly, as that fails on Windows
        os.environ.pop("POLARS_FMT_NO_UTF8", None)
        return cls

    @classmethod
    def set_ascii_tables(cls) -> type[Config]:
        """Use ascii characters to print tables."""
        os.environ["POLARS_FMT_NO_UTF8"] = "1"
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
        Set the number of rows used to print tables.

        Parameters
        ----------
        n
            number of rows to print

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
            number of columns to print

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
