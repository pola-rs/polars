from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars.utils.deprecation import issue_deprecation_warning

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from types import TracebackType


class StringCache(contextlib.ContextDecorator):
    """
    Context manager for enabling and disabling the global string cache.

    :class:`Categorical` columns created under the same global string cache have
    the same underlying physical value when string values are equal. This allows the
    columns to be concatenated or used in a join operation, for example.

    Notes
    -----
    Enabling the global string cache introduces some overhead.
    The amount of overhead depends on the number of categories in your data.
    It is advised to enable the global string cache only when strictly necessary.

    If ``StringCache``s are nested, the global string cache will only be disabled and
    cleared when the outermost context exits.

    Examples
    --------
    Construct two dataframes using the same string cache.

    >>> with pl.StringCache():
    ...     df1 = pl.DataFrame(
    ...         data={
    ...             "color": ["red", "green", "blue", "orange"],
    ...             "value": [1, 2, 3, 4],
    ...         },
    ...         schema={"color": pl.Categorical, "value": pl.UInt8},
    ...     )
    ...     df2 = pl.DataFrame(
    ...         data={
    ...             "color": ["yellow", "green", "orange", "black", "red"],
    ...             "char": ["a", "b", "c", "d", "e"],
    ...         },
    ...         schema={"color": pl.Categorical, "char": pl.Utf8},
    ...     )
    ...

    As both dataframes use the same string cache for the categorical column,
    the column can be used in a join operation.

    >>> df1.join(df2, how="inner", on="color")
    shape: (3, 3)
    ┌────────┬───────┬──────┐
    │ color  ┆ value ┆ char │
    │ ---    ┆ ---   ┆ ---  │
    │ cat    ┆ u8    ┆ str  │
    ╞════════╪═══════╪══════╡
    │ green  ┆ 2     ┆ b    │
    │ orange ┆ 4     ┆ c    │
    │ red    ┆ 1     ┆ e    │
    └────────┴───────┴──────┘

    """

    def __enter__(self) -> StringCache:
        plr._set_string_cache(True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        plr._set_string_cache(False)


def enable_string_cache(enable: bool | None = None) -> None:
    """
    Enable the global string cache.

    :class:`Categorical` columns created under the same global string cache have
    the same underlying physical value when string values are equal. This allows the
    columns to be concatenated or used in a join operation, for example.

    Parameters
    ----------
    enable
        Enable or disable the global string cache.

        .. deprecated:: 0.19.3
            ``enable_string_cache`` no longer accepts an argument.
             Call ``enable_string_cache()`` to enable the string cache
             and ``disable_string_cache()`` to disable the string cache.

    See Also
    --------
    enable_string_cache : Function to disable the string cache.
    StringCache : Context manager for enabling and disabling the string cache.

    Notes
    -----
    Enabling the global string cache introduces some overhead.
    The amount of overhead depends on the number of categories in your data.
    It is advised to enable the global string cache only when strictly necessary.

    Consider using the :class:`StringCache` context manager for a more reliable way of
    enabling and disabling the string cache.

    Examples
    --------
    >>> pl.enable_string_cache()
    >>> df1 = pl.DataFrame(
    ...     data={"color": ["red", "green", "blue", "orange"], "value": [1, 2, 3, 4]},
    ...     schema={"color": pl.Categorical, "value": pl.UInt8},
    ... )
    >>> df2 = pl.DataFrame(
    ...     data={
    ...         "color": ["yellow", "green", "orange", "black", "red"],
    ...         "char": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={"color": pl.Categorical, "char": pl.Utf8},
    ... )
    >>> pl.disable_string_cache()
    >>> df1.join(df2, how="inner", on="color")
    shape: (3, 3)
    ┌────────┬───────┬──────┐
    │ color  ┆ value ┆ char │
    │ ---    ┆ ---   ┆ ---  │
    │ cat    ┆ u8    ┆ str  │
    ╞════════╪═══════╪══════╡
    │ green  ┆ 2     ┆ b    │
    │ orange ┆ 4     ┆ c    │
    │ red    ┆ 1     ┆ e    │
    └────────┴───────┴──────┘

    """
    if enable is not None:
        issue_deprecation_warning(
            "`enable_string_cache` no longer accepts an argument."
            " Call `enable_string_cache()` to enable the string cache"
            " and `disable_string_cache()` to disable the string cache.",
            version="0.19.3",
        )
        plr._set_string_cache(enable)
        return

    plr.enable_string_cache()


def disable_string_cache() -> bool:
    """
    Disable and clear the global string cache.

    See Also
    --------
    enable_string_cache : Function to enable the string cache.
    StringCache : Context manager for enabling and disabling the string cache.

    Notes
    -----
    Consider using the :class:`StringCache` context manager for a more reliable way of
    enabling and disabling the string cache.

    Examples
    --------
    >>> pl.enable_string_cache()
    >>> df1 = pl.DataFrame(
    ...     data={"color": ["red", "green", "blue", "orange"], "value": [1, 2, 3, 4]},
    ...     schema={"color": pl.Categorical, "value": pl.UInt8},
    ... )
    >>> df2 = pl.DataFrame(
    ...     data={
    ...         "color": ["yellow", "green", "orange", "black", "red"],
    ...         "char": ["a", "b", "c", "d", "e"],
    ...     },
    ...     schema={"color": pl.Categorical, "char": pl.Utf8},
    ... )
    >>> pl.disable_string_cache()
    >>> df1.join(df2, how="inner", on="color")
    shape: (3, 3)
    ┌────────┬───────┬──────┐
    │ color  ┆ value ┆ char │
    │ ---    ┆ ---   ┆ ---  │
    │ cat    ┆ u8    ┆ str  │
    ╞════════╪═══════╪══════╡
    │ green  ┆ 2     ┆ b    │
    │ orange ┆ 4     ┆ c    │
    │ red    ┆ 1     ┆ e    │
    └────────┴───────┴──────┘

    """
    return plr.disable_string_cache()


def using_string_cache() -> bool:
    """Check whether the global string cache is enabled."""
    return plr.using_string_cache()
