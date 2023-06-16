from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polars.dependencies import json
from polars.utils.various import normalise_filepath


# dummy func required (so docs build)
def _get_float_fmt() -> str:
    return "n/a"


# note: module not available when building docs
with contextlib.suppress(ImportError):
    from polars.polars import get_float_fmt as _get_float_fmt  # type: ignore[no-redef]
    from polars.polars import set_float_fmt as _set_float_fmt

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    from polars.type_aliases import FloatFmt

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


# note: register all Config-specific environment variable names here; need to constrain
# which 'POLARS_' environment variables are recognised, as there are other lower-level
# and/or experimental settings that should not be saved or reset with the Config vars.
_POLARS_CFG_ENV_VARS = {
    "POLARS_ACTIVATE_DECIMAL",
    "POLARS_AUTO_STRUCTIFY",
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
    "POLARS_FMT_TABLE_ROUNDED_CORNERS",
    "POLARS_STREAMING_CHUNK_SIZE",
    "POLARS_TABLE_WIDTH",
    "POLARS_VERBOSE",
}

# vars that set the rust env directly should declare themselves here as the Config
# method name paired with a callable that returns the current state of that value:
_POLARS_CFG_DIRECT_VARS = {"set_fmt_float": _get_float_fmt}


class Config(contextlib.ContextDecorator):
    """
    Configure polars; offers options for table formatting and more.

    Notes
    -----
    Can also be used as a context manager OR a function decorator in order to
    temporarily scope the lifetime of specific options. For example:

    >>> with pl.Config() as cfg:
    ...     # set verbose for more detailed output within the scope
    ...     cfg.set_verbose(True)  # doctest: +IGNORE_RESULT
    ...
    >>> # scope exit - no longer in verbose mode

    This can also be written more compactly as:

    >>> with pl.Config(verbose=True):
    ...     pass
    ...

    (The compact format is available for all `Config` methods that take a single value).

    Alternatively, you can use as a decorator in order to scope the duration of the
    selected options to a specific function:

    >>> @pl.Config(verbose=True)
    ... def test():
    ...     pass
    ...

    """

    _original_state: str = ""

    def __init__(self, *, restore_defaults: bool = False, **options: Any) -> None:
        """
        Initialise a Config object instance for context manager usage.

        Any `options` kwargs should correspond to the available named "set_"
        methods, but can optionally to omit the "set_" prefix for brevity.

        Parameters
        ----------
        restore_defaults
            set all options to their default values (this is applied before
            setting any other options).
        options
            keyword args that will set the option; equivalent to calling the
            named "set_<option>" method with the given value.

        Examples
        --------
        >>> with pl.Config(
        ...     # these options will be set for scope duration
        ...     tbl_formatting="ASCII_MARKDOWN",
        ...     tbl_width_chars=180,
        ... ):
        ...     pass

        """
        # save original state _before_ any changes are made
        self._original_state = self.save()

        if restore_defaults:
            self.restore_defaults()

        for opt, value in options.items():
            if not hasattr(self, opt) and not opt.startswith("set_"):
                opt = f"set_{opt}"
            if not hasattr(self, opt):
                raise AttributeError(f"Config has no {opt!r} option")
            getattr(self, opt)(value)

    def __enter__(self) -> Config:
        """Support setting temporary Config options that are reset on scope exit."""
        self._original_state = self._original_state or self.save()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Reset any Config options that were set within the scope."""
        self.restore_defaults().load(self._original_state)
        self._original_state = ""

    @classmethod
    def load(cls, cfg: Path | str) -> type[Config]:
        """
        Load and set previously saved (or shared) Config options from json/file.

        Parameters
        ----------
        cfg : str
            json string produced by ``Config.save()``, or a filepath to the same.

        """
        options = json.loads(
            Path(normalise_filepath(cfg)).read_text()
            if isinstance(cfg, Path) or os.path.exists(cfg)
            else cfg
        )
        os.environ.update(options.get("environment", {}))
        for cfg_methodname, value in options.get("direct", {}).items():
            if hasattr(cls, cfg_methodname):
                getattr(cls, cfg_methodname)(value)
        return cls

    @classmethod
    def restore_defaults(cls) -> type[Config]:
        """
        Reset all polars Config settings to their default state.

        Notes
        -----
        This method operates by removing all Config options from the environment,
        and then setting any local (non-env) options back to their default value.

        Examples
        --------
        >>> cfg = pl.Config.restore_defaults()  # doctest: +SKIP

        """
        # unset all Config environment variables
        for var in _POLARS_CFG_ENV_VARS:
            os.environ.pop(var, None)

        # apply any 'direct' setting values
        cls.set_fmt_float()
        return cls

    @classmethod
    def save(cls, file: Path | str | None = None) -> str:
        """
        Save the current set of Config options as a json string or file.

        Parameters
        ----------
        file
            optional path to a file into which the json string will be written.

        Examples
        --------
        >>> cfg = pl.Config.save()

        Returns
        -------
        str : json string containing current Config options, or filepath where saved.

        """
        environment_vars = {
            key: os.environ[key]
            for key in sorted(_POLARS_CFG_ENV_VARS)
            if (key in os.environ)
        }
        direct_vars = {
            cfg_methodname: get_value()
            for cfg_methodname, get_value in _POLARS_CFG_DIRECT_VARS.items()
        }
        options = json.dumps(
            {"environment": environment_vars, "direct": direct_vars},
            separators=(",", ":"),
        )
        if isinstance(file, (str, Path)):
            file = os.path.abspath(normalise_filepath(file))
            Path(file).write_text(options)
            return file

        return options

    @classmethod
    def state(
        cls, if_set: bool = False, env_only: bool = False
    ) -> dict[str, str | None]:
        """
        Show the current state of all Config variables as a dict.

        Parameters
        ----------
        if_set : bool
            by default this will show the state of all ``Config`` environment variables.
            change this to ``True`` to restrict the returned dictionary to include only
            those that have been set to a specific value.

        env_only : bool
            include only Config environment variables in the output; some options (such
            as "set_fmt_float") are set directly, not via an environment variable.

        Examples
        --------
        >>> set_state = pl.Config.state(if_set=True)
        >>> all_state = pl.Config.state()

        """
        config_state = {
            var: os.environ.get(var)
            for var in sorted(_POLARS_CFG_ENV_VARS)
            if not if_set or (os.environ.get(var) is not None)
        }
        if not env_only:
            for cfg_methodname, get_value in _POLARS_CFG_DIRECT_VARS.items():
                config_state[cfg_methodname] = get_value()

        return config_state

    @classmethod
    def activate_decimals(cls, active: bool = True) -> type[Config]:
        """
        Activate ``Decimal`` data types.

        This is a temporary setting that will be removed later once the
        ``Decimal`` type stabilize. This will happens without it being
        considered a breaking change.

        Currently, ``Decimal`` types are in an alpha state.

        """
        if not active:
            os.environ.pop("POLARS_ACTIVATE_DECIMAL", None)
        else:
            os.environ["POLARS_ACTIVATE_DECIMAL"] = "1"
        return cls

    @classmethod
    def set_ascii_tables(cls, active: bool = True) -> type[Config]:
        """
        Use ASCII characters to display table outlines (set False to revert to UTF8).

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
        # │ 2.5 ┆ false │      | 2.5 | false |
        # │ 5.0 ┆ true  │      | 5.0 | true  |
        # └─────┴───────┘      +-----+-------+

        """
        fmt = "ASCII_FULL_CONDENSED" if active else "UTF8_FULL_CONDENSED"
        os.environ["POLARS_FMT_TABLE_FORMATTING"] = fmt
        return cls

    @classmethod
    def set_auto_structify(cls, active: bool = False) -> type[Config]:
        """Allow multi-output expressions to be automatically turned into Structs."""
        os.environ["POLARS_AUTO_STRUCTIFY"] = str(int(active))
        return cls

    @classmethod
    def set_fmt_float(cls, fmt: FloatFmt = "mixed") -> type[Config]:
        """
        Control how floating  point values are displayed.

        Parameters
        ----------
        fmt : {"mixed", "full"}
            How to format floating point numbers

        """
        _set_float_fmt(fmt)
        return cls

    @classmethod
    def set_fmt_str_lengths(cls, n: int) -> type[Config]:
        """
        Set the number of characters used to display string values.

        Parameters
        ----------
        n : int
            number of characters to display

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "txt": [
        ...             "Play it, Sam. Play 'As Time Goes By'.",
        ...             "This is the beginning of a beautiful friendship.",
        ...         ]
        ...     }
        ... )
        >>> df.with_columns(pl.col("txt").str.lengths().alias("len"))
        shape: (2, 2)
        ┌───────────────────────────────────┬─────┐
        │ txt                               ┆ len │
        │ ---                               ┆ --- │
        │ str                               ┆ u32 │
        ╞═══════════════════════════════════╪═════╡
        │ Play it, Sam. Play 'As Time Goes… ┆ 37  │
        │ This is the beginning of a beaut… ┆ 48  │
        └───────────────────────────────────┴─────┘
        >>> with pl.Config(fmt_str_lengths=50):
        ...     print(df)
        ...
        shape: (2, 1)
        ┌──────────────────────────────────────────────────┐
        │ txt                                              │
        │ ---                                              │
        │ str                                              │
        ╞══════════════════════════════════════════════════╡
        │ Play it, Sam. Play 'As Time Goes By'.            │
        │ This is the beginning of a beautiful friendship. │
        └──────────────────────────────────────────────────┘

        """
        if n <= 0:
            raise ValueError("number of characters must be > 0")

        os.environ["POLARS_FMT_STR_LEN"] = str(n)
        return cls

    @classmethod
    def set_streaming_chunk_size(cls, size: int) -> type[Config]:
        """
        Overwrite chunk size used in ``streaming`` engine.

        By default, the chunk size is determined by the schema
        and size of the thread pool. For some datasets (esp.
        when you have large string elements) this can be too
        optimistic and lead to Out of Memory errors.

        Parameters
        ----------
        size
            Number of rows per chunk. Every thread will process chunks
            of this size.

        """
        if size < 1:
            raise ValueError("number of rows per chunk must be >= 1")

        os.environ["POLARS_STREAMING_CHUNK_SIZE"] = str(size)
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
        # │        2.5 ┆      false │
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
        Set the number of columns that are visible when displaying tables.

        Parameters
        ----------
        n : int
            number of columns to display; if ``n < 0`` (eg: -1), display all columns.

        Examples
        --------
        Set number of displayed columns to a low value:

        >>> with pl.Config() as cfg:
        ...     cfg.set_tbl_cols(5)
        ...     df = pl.DataFrame({str(i): [i] for i in range(100)})
        ...     print(df)
        ...
        <class 'polars.config.Config'>
        shape: (1, 100)
        ┌─────┬─────┬─────┬───┬─────┬─────┐
        │ 0   ┆ 1   ┆ 2   ┆ … ┆ 98  ┆ 99  │
        │ --- ┆ --- ┆ --- ┆   ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 ┆   ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═══╪═════╪═════╡
        │ 0   ┆ 1   ┆ 2   ┆ … ┆ 98  ┆ 99  │
        └─────┴─────┴─────┴───┴─────┴─────┘

        >>> with pl.Config(tbl_cols=10):
        ...     print(df)
        ...
        shape: (1, 100)
        ┌─────┬─────┬─────┬─────┬─────┬───┬─────┬─────┬─────┬─────┬─────┐
        │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ 4   ┆ … ┆ 95  ┆ 96  ┆ 97  ┆ 98  ┆ 99  │
        │ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆   ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 ┆   ┆ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╪═════╪═══╪═════╪═════╪═════╪═════╪═════╡
        │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ 4   ┆ … ┆ 95  ┆ 96  ┆ 97  ┆ 98  ┆ 99  │
        └─────┴─────┴─────┴─────┴─────┴───┴─────┴─────┴─────┴─────┴─────┘

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
        # ╞═════╪═══════╡  >>  │ 2.5       ┆ false      │
        # │ 1.0 ┆ true  │      │ 5.0       ┆ true       │
        # │ 2.5 ┆ false │      └───────────┴────────────┘
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
        # ╞═════╪═══════╡  >>  │ 1.0 ┆ true  │
        # │ 1.0 ┆ true  │      │ 2.5 ┆ false │
        # │ 2.5 ┆ false │      │ 5.0 ┆ true  │
        # │ 5.0 ┆ true  │      └─────┴───────┘
        # └─────┴───────┘      shape: (3, 2)

        """
        os.environ["POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW"] = str(int(active))
        return cls

    @classmethod
    def set_tbl_formatting(
        cls,
        format: (
            Literal[
                "ASCII_FULL",
                "ASCII_FULL_CONDENSED",
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
            ]
            | None
        ) = None,
        rounded_corners: bool = False,
    ) -> type[Config]:
        """
        Set table formatting style.

        Parameters
        ----------
        format : str
            * "ASCII_FULL": ASCII, with all borders and lines, including row dividers.
            * "ASCII_FULL_CONDENSED": Same as ASCII_FULL, but with dense row spacing.
            * "ASCII_NO_BORDERS": ASCII, no borders.
            * "ASCII_BORDERS_ONLY": ASCII, borders only.
            * "ASCII_BORDERS_ONLY_CONDENSED": ASCII, borders only, dense row spacing.
            * "ASCII_HORIZONTAL_ONLY": ASCII, horizontal lines only.
            * "ASCII_MARKDOWN": ASCII, Markdown compatible.
            * "UTF8_FULL": UTF8, with all borders and lines, including row dividers.
            * "UTF8_FULL_CONDENSED": Same as UTF8_FULL, but with dense row spacing.
            * "UTF8_NO_BORDERS": UTF8, no borders.
            * "UTF8_BORDERS_ONLY": UTF8, borders only.
            * "UTF8_HORIZONTAL_ONLY": UTF8, horizontal lines only.
            * "NOTHING": No borders or other lines.

        rounded_corners : bool
            apply rounded corners to UTF8-styled tables (no-op for ASCII formats).

        Notes
        -----
        The UTF8 styles all use one or more of the semigraphic box-drawing characters
        found in the Unicode Box Drawing block, which are not ASCII compatible:
        https://en.wikipedia.org/wiki/Box-drawing_character#Box_Drawing

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"abc": [-2.5, 5.0], "mno": ["hello", "world"], "xyz": [True, False]}
        ... )
        >>> with pl.Config(
        ...     tbl_formatting="ASCII_MARKDOWN",
        ...     tbl_hide_column_data_types=True,
        ...     tbl_hide_dataframe_shape=True,
        ... ):
        ...     print(df)
        | abc  | mno   | xyz   |
        |------|-------|-------|
        | -2.5 | hello | true  |
        | 5.0  | world | false |

        Raises
        ------
        KeyError: if format string not recognised.

        """
        # can see what the different styles look like in the comfy-table tests:
        # https://github.com/Nukesor/comfy-table/blob/main/tests/all/presets_test.rs
        if format:
            os.environ["POLARS_FMT_TABLE_FORMATTING"] = format
        os.environ["POLARS_FMT_TABLE_ROUNDED_CORNERS"] = str(int(rounded_corners))
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
        # ╞═════╪═══════╡  >>  │ 2.5 ┆ false │
        # │ 1.0 ┆ true  │      │ 5.0 ┆ true  │
        # │ 2.5 ┆ false │      └─────┴───────┘
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
        # ╞═════╪═══════╡  >>  │ 2.5 ┆ false │
        # │ 1.0 ┆ true  │      │ 5.0 ┆ true  │
        # │ 2.5 ┆ false │      └─────┴───────┘
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
        # │ 1.0 ┆ true  │  >>  │ 2.5 ┆ false │
        # │ 2.5 ┆ false │      │ 5.0 ┆ true  │
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
        # │ 1.0 ┆ true  │  >>  │ 2.5 ┆ false │
        # │ 2.5 ┆ false │      │ 5.0 ┆ true  │
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
            number of rows to display; if ``n < 0`` (eg: -1), display all
            rows (DataFrame) and all elements (Series).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"abc": [1.0, 2.5, 3.5, 5.0], "xyz": [True, False, True, False]}
        ... )
        >>> with pl.Config(tbl_rows=2):
        ...     print(df)
        ...
        shape: (4, 2)
        ┌─────┬───────┐
        │ abc ┆ xyz   │
        │ --- ┆ ---   │
        │ f64 ┆ bool  │
        ╞═════╪═══════╡
        │ 1.0 ┆ true  │
        │ …   ┆ …     │
        │ 5.0 ┆ false │
        └─────┴───────┘

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
