from __future__ import annotations

import json
import os
import sys
from types import TracebackType

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


# note: register all Config-specific environment variable names here; need to constrain
# which 'POLARS_' environment variables are recognised, as there are other lower-level
# and/or experimental settings that should not be saved or reset with the Config vars.
POLARS_CFG_ENV_VARS = {
    "POLARS_FMT_MAX_COLS",
    "POLARS_FMT_MAX_ROWS",
    "POLARS_FMT_STR_LEN",
    "POLARS_FMT_TABLE_CELL_ALIGNMENT",
    "POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW",
    "POLARS_FMT_TABLE_FORMATTING",
    "POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES",
    "POLARS_FMT_TABLE_HIDE_COLUMN_NAMES",
    "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR",
    "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION",
    "POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE",
    "POLARS_TABLE_WIDTH",
    "POLARS_VERBOSE",
}
# register Config-local attributes (with their defaults) here
POLARS_CFG_LOCAL_VARS = {"with_columns_kwargs": False}


class Config:
    """
    Configure polars; offers options for table formatting and more.

    Notes
    -----
    Can also be used as a context manager in order to temporarily scope
    the lifetime of specific options. For example:

    >>> with pl.Config() as cfg:
    ...     # set verbose for more detailed output within the scope
    ...     cfg.set_verbose(True)  # doctest: +IGNORE_RESULT
    ...
    >>> # scope exit - no longer in verbose mode

    """

    def __enter__(self) -> Config:
        """Support setting temporary Config options that are reset on scope exit."""
        self._original_state = self.save()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Reset any Config options that were set within the scope."""
        self.restore_defaults().load(self._original_state)

    # note: class-local attributes can be used for options that don't have
    # a Rust component (so, no need to register environment variables).
    with_columns_kwargs: bool = False

    @classmethod
    def load(cls, cfg: str) -> type[Config]:
        """
        Load and set previously saved (or shared) Config options.

        Parameters
        ----------
        cfg : str
            json string produced by ``Config.save()``.

        """
        options = json.loads(cfg)
        os.environ.update(options.get("environment", {}))
        for flag, value in options.get("local", {}).items():
            if hasattr(cls, flag):
                setattr(cls, flag, value)
        return cls

    @classmethod
    def restore_defaults(cls) -> type[Config]:
        """
        Reset all polars Config settings to their default state.

        Notes
        -----
        This method operates by removing all Config options from the environment,
        and then setting any class-local flags back to their default value.

        Examples
        --------
        >>> cfg = pl.Config.restore_defaults()  # doctest: +SKIP

        """
        for var in POLARS_CFG_ENV_VARS:
            os.environ.pop(var, None)
        for flag, value in POLARS_CFG_LOCAL_VARS.items():
            setattr(cls, flag, value)
        return cls

    @classmethod
    def save(cls) -> str:
        """
        Save the current set of Config options as a json string.

        Examples
        --------
        >>> cfg = pl.Config.save()

        """
        environment_vars = {
            key: os.environ[key]
            for key in sorted(POLARS_CFG_ENV_VARS)
            if (key in os.environ)
        }
        config_vars = {attr: getattr(cls, attr) for attr in POLARS_CFG_LOCAL_VARS}
        return json.dumps(
            {"environment": environment_vars, "local": config_vars},
            separators=(",", ":"),
        )

    @classmethod
    def state(cls, if_set: bool = False) -> dict[str, str | None]:
        """
        Show the current state of all Config environment variables as a dict.

        Parameters
        ----------
        if_set : bool
            by default this will show the state of all ``Config`` environment variables.
            change this to ``True`` to restrict the returned dictionary to include only
            those that _have_ been set to a specific value.

        Examples
        --------
        >>> set_state = pl.Config.state(if_set=True)
        >>> all_state = pl.Config.state()

        """
        return {
            var: os.environ.get(var)
            for var in sorted(POLARS_CFG_ENV_VARS)
            if not if_set or (os.environ.get(var) is not None)
        }

    @classmethod
    def set_ascii_tables(cls, active: bool = True) -> type[Config]:
        """
        Use ASCII characters to print table outlines (set False to revert to UTF8).

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_ascii_tables(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        shape: (3, 2)
        # ┌─────┬───────┐      +-----+-------+
        # │ abc ┆ xyz   │      | abc | xyz   |
        # │ --- ┆ ---   │      | --- | ---   |
        # │ f64 ┆ bool  │      | f64 | bool  |
        # ╞═════╪═══════╡      +=============+
        # │ 1.0 ┆ true  │  >>  | 1.0 | true  |
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      |-----+-------|
        # │ 2.5 ┆ false │      | 2.5 | false |
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      |-----+-------|
        # │ 5.0 ┆ true  │      | 5.0 | true  |
        # └─────┴───────┘      +-----+-------+

        """
        fmt = "ASCII_FULL" if active else "UTF8_FULL"
        os.environ["POLARS_FMT_TABLE_FORMATTING"] = fmt
        return cls

    @classmethod
    def set_fmt_str_lengths(cls, n: int) -> type[Config]:
        """
        Set the number of characters used to print string values.

        Parameters
        ----------
        n : int
            number of characters to print

        """
        os.environ["POLARS_FMT_STR_LEN"] = str(n)
        return cls

    @classmethod
    def set_tbl_cell_alignment(
        cls, format: Literal["LEFT", "CENTER", "RIGHT"]
    ) -> type[Config]:
        """
        Set table cell alignment.

        Parameters
        ----------
        format : str
            * "LEFT": left aligned
            * "CENTER": center aligned
            * "RIGHT": right aligned

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"column_abc": [1.0, 2.5, 5.0], "column_xyz": [True, False, True]}
        ... )
        >>> pl.Config.set_tbl_cell_alignment("RIGHT")  # doctest: +SKIP
        # ...
        # shape: (3, 2)
        # ┌────────────┬────────────┐
        # │ column_abc ┆ column_xyz │
        # │        --- ┆        --- │
        # │        f64 ┆       bool │
        # ╞════════════╪════════════╡
        # │        1.0 ┆       true │
        # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        # │        2.5 ┆      false │
        # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        # │        5.0 ┆       true │
        # └────────────┴────────────┘

        Raises
        ------
        KeyError: if alignment string not recognised.

        """
        os.environ["POLARS_FMT_TABLE_CELL_ALIGNMENT"] = format
        return cls

    @classmethod
    def set_tbl_cols(cls, n: int) -> type[Config]:
        """
        Set the number of columns used to print tables.

        Parameters
        ----------
        n : int
            number of columns to print. If n<0 print all the columns.

        Examples
        --------
        >>> with pl.Config() as cfg:
        ...     cfg.set_tbl_cols(5)  # doctest: +IGNORE_RESULT
        ...     df = pl.DataFrame({str(i): [i] for i in range(100)})
        ...     df
        ...
        shape: (1, 100)
        ┌─────┬─────┬─────┬─────┬─────┬─────┐
        │ 0   ┆ 1   ┆ 2   ┆ ... ┆ 98  ┆ 99  │
        │ --- ┆ --- ┆ --- ┆     ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 ┆     ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╪═════╪═════╡
        │ 0   ┆ 1   ┆ 2   ┆ ... ┆ 98  ┆ 99  │
        └─────┴─────┴─────┴─────┴─────┴─────┘

        >>> with pl.Config() as cfg:
        ...     pl.cfg.Config.set_tbl_cols(10)  # doctest: +IGNORE_RESULT
        ...     df
        ...
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
    def set_tbl_column_data_type_inline(cls, active: bool = True) -> type[Config]:
        """
        Moves the data type inline with the column name (to the right, in parentheses).

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_tbl_column_data_type_inline(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        shape: (3, 2)
        # ┌─────┬───────┐      ┌───────────┬────────────┐
        # │ abc ┆ xyz   │      │ abc (f64) ┆ xyz (bool) │
        # │ --- ┆ ---   │      ╞═══════════╪════════════╡
        # │ f64 ┆ bool  │      │ 1.0       ┆ true       │
        # ╞═════╪═══════╡      ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        # │ 1.0 ┆ true  │  >>  │ 2.5       ┆ false      │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        # │ 2.5 ┆ false │      │ 5.0       ┆ true       │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      └───────────┴────────────┘
        # │ 5.0 ┆ true  │
        # └─────┴───────┘

        """
        os.environ["POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE"] = str(int(active))
        return cls

    @classmethod
    def set_tbl_dataframe_shape_below(cls, active: bool = True) -> type[Config]:
        """
        Print the dataframe shape below the dataframe when displaying tables.

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_tbl_dataframe_shape_below(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        ┌─────┬───────┐
        # ┌─────┬───────┐      │ abc ┆ xyz   │
        # │ abc ┆ xyz   │      │ --- ┆ ---   │
        # │ --- ┆ ---   │      │ f64 ┆ bool  │
        # │ f64 ┆ bool  │      ╞═════╪═══════╡
        # ╞═════╪═══════╡      │ 1.0 ┆ true  │
        # │ 1.0 ┆ true  │  >>  ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      │ 2.5 ┆ false │
        # │ 2.5 ┆ false │      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      │ 5.0 ┆ true  │
        # │ 5.0 ┆ true  │      └─────┴───────┘
        # └─────┴───────┘      shape: (3, 2)

        """
        os.environ["POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW"] = str(int(active))
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
            "UTF8_FULL_CONDENSED",
            "UTF8_NO_BORDERS",
            "UTF8_BORDERS_ONLY",
            "UTF8_HORIZONTAL_ONLY",
            "NOTHING",
        ],
    ) -> type[Config]:
        """
        Set table formatting style.

        Parameters
        ----------
        format : str
            * "ASCII_FULL": ASCII, borders / lines.
            * "ASCII_NO_BORDERS": ASCII, no borders.
            * "ASCII_BORDERS_ONLY": ASCII, borders only.
            * "ASCII_BORDERS_ONLY_CONDENSED": ASCII, borders only, dense row spacing.
            * "ASCII_HORIZONTAL_ONLY": ASCII, horizontal lines only.
            * "ASCII_MARKDOWN": ASCII, Markdown compatible.
            * "UTF8_FULL": UTF8, with all borders and lines (default).
            * "UTF8_FULL_CONDENSED": Same as UTF8_FULL, with dense row spacing.
            * "UTF8_NO_BORDERS": UTF8, no borders.
            * "UTF8_BORDERS_ONLY": UTF8, borders only.
            * "UTF8_HORIZONTAL_ONLY": UTF8, horizontal lines only.
            * "NOTHING": No borders or other lines.

        Notes
        -----
        The UTF8 styles all use one or more of the semigraphic box-drawing characters
        found in the Unicode Box Drawing block, which are not ASCII compatible:
        https://en.wikipedia.org/wiki/Box-drawing_character#Box_Drawing

        Raises
        ------
        KeyError: if format string not recognised.

        """
        # can see what the different styles look like in the comfy-table tests:
        # https://github.com/Nukesor/comfy-table/blob/main/tests/all/presets_test.rs
        os.environ["POLARS_FMT_TABLE_FORMATTING"] = format
        return cls

    @classmethod
    def set_tbl_hide_column_data_types(cls, active: bool = True) -> type[Config]:
        """
        Hide table column data types (i64, f64, str etc.).

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_tbl_hide_column_data_types(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        shape: (3, 2)
        # ┌─────┬───────┐      ┌─────┬───────┐
        # │ abc ┆ xyz   │      │ abc ┆ xyz   │
        # │ --- ┆ ---   │      ╞═════╪═══════╡
        # │ f64 ┆ bool  │      │ 1.0 ┆ true  │
        # ╞═════╪═══════╡      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # │ 1.0 ┆ true  │  >>  │ 2.5 ┆ false │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # │ 2.5 ┆ false │      │ 5.0 ┆ true  │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      └─────┴───────┘
        # │ 5.0 ┆ true  │
        # └─────┴───────┘

        """
        os.environ["POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES"] = str(int(active))
        return cls

    @classmethod
    def set_tbl_hide_column_names(cls, active: bool = True) -> type[Config]:
        """
        Hide table column names.

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_tbl_hide_column_names(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        shape: (3, 2)
        # ┌─────┬───────┐      ┌─────┬───────┐
        # │ abc ┆ xyz   │      │ f64 ┆ bool  │
        # │ --- ┆ ---   │      ╞═════╪═══════╡
        # │ f64 ┆ bool  │      │ 1.0 ┆ true  │
        # ╞═════╪═══════╡      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # │ 1.0 ┆ true  │  >>  │ 2.5 ┆ false │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # │ 2.5 ┆ false │      │ 5.0 ┆ true  │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      └─────┴───────┘
        # │ 5.0 ┆ true  │
        # └─────┴───────┘

        """
        os.environ["POLARS_FMT_TABLE_HIDE_COLUMN_NAMES"] = str(int(active))
        return cls

    @classmethod
    def set_tbl_hide_dtype_separator(cls, active: bool = True) -> type[Config]:
        """
        Hide the '---' separator between the column names and column types.

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_tbl_hide_dtype_separator(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        shape: (3, 2)
        # ┌─────┬───────┐      ┌─────┬───────┐
        # │ abc ┆ xyz   │      │ abc ┆ xyz   │
        # │ --- ┆ ---   │      │ f64 ┆ bool  │
        # │ f64 ┆ bool  │      ╞═════╪═══════╡
        # ╞═════╪═══════╡      │ 1.0 ┆ true  │
        # │ 1.0 ┆ true  │  >>  ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      │ 2.5 ┆ false │
        # │ 2.5 ┆ false │      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      │ 5.0 ┆ true  │
        # │ 5.0 ┆ true  │      └─────┴───────┘
        # └─────┴───────┘

        See Also
        --------
        set_tbl_column_data_type_inline

        """
        os.environ["POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR"] = str(int(active))
        return cls

    @classmethod
    def set_tbl_hide_dataframe_shape(cls, active: bool = True) -> type[Config]:
        """
        Hide the shape information of the dataframe when displaying tables.

        Examples
        --------
        >>> df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})
        >>> pl.Config.set_tbl_hide_dataframe_shape(True)  # doctest: +SKIP
        # ...
        # shape: (3, 2)        ┌─────┬───────┐
        # ┌─────┬───────┐      │ abc ┆ xyz   │
        # │ abc ┆ xyz   │      │ --- ┆ ---   │
        # │ --- ┆ ---   │      │ f64 ┆ bool  │
        # │ f64 ┆ bool  │      ╞═════╪═══════╡
        # ╞═════╪═══════╡      │ 1.0 ┆ true  │
        # │ 1.0 ┆ true  │  >>  ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      │ 2.5 ┆ false │
        # │ 2.5 ┆ false │      ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤      │ 5.0 ┆ true  │
        # │ 5.0 ┆ true  │      └─────┴───────┘
        # └─────┴───────┘

        """
        os.environ["POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION"] = str(
            int(active)
        )
        return cls

    @classmethod
    def set_tbl_rows(cls, n: int) -> type[Config]:
        """
        Set the max number of rows used to draw the table (both Dataframe and Series).

        Parameters
        ----------
        n : int
            number of rows to print; if n < 0, prints all rows (DataFrame) and
            all elements (Series).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"abc": [1.0, 2.5, 3.5, 5.0], "xyz": [True, False, True, False]}
        ... )
        >>> pl.Config.set_tbl_rows(2)  # doctest: +SKIP
        # ...
        # shape: (4, 2)
        # ┌─────┬───────┐
        # │ abc ┆ xyz   │
        # │ --- ┆ ---   │
        # │ f64 ┆ bool  │
        # ╞═════╪═══════╡
        # │ 1.0 ┆ true  │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # │ ... ┆ ...   │
        # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        # │ 5.0 ┆ false │
        # └─────┴───────┘

        """
        os.environ["POLARS_FMT_MAX_ROWS"] = str(n)
        return cls

    @classmethod
    def set_tbl_width_chars(cls, width: int) -> type[Config]:
        """
        Set the number of characters used to draw the table.

        Parameters
        ----------
        width : int
            number of chars

        """
        os.environ["POLARS_TABLE_WIDTH"] = str(width)
        return cls

    @classmethod
    def set_verbose(cls, active: bool = True) -> type[Config]:
        """Enable additional verbose/debug logging."""
        os.environ["POLARS_VERBOSE"] = str(int(active))
        return cls
