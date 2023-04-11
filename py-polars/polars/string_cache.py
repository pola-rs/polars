from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING

from polars.utils.various import find_stacklevel

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import enable_string_cache as _enable_string_cache
    from polars.polars import using_string_cache as _using_string_cache

if TYPE_CHECKING:
    from types import TracebackType


class StringCache:
    """
    Context manager that allows data sources to share the same categorical features.

    This will temporarily cache the string categories until the context manager is
    finished. If StringCaches are nested, the global cache will only be invalidated
    when the outermost context exits.

    Examples
    --------
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
    ...     # Both dataframes use the same string cache for the categorical column,
    ...     # so the join operation on that column will succeed.
    ...     df_join = df1.join(df2, how="inner", on="color")
    ...
    >>> df_join
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
        self._already_enabled = _using_string_cache()
        if not self._already_enabled:
            _enable_string_cache(True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # note: if global string cache was already enabled
        # on __enter__, do NOT reset it on __exit__
        if not self._already_enabled:
            _enable_string_cache(False)


def enable_string_cache(enable: bool) -> None:
    """
    Enable (or disable) the global string cache.

    This ensures that casts to Categorical dtypes will have
    the same category values when string values are equal.

    Parameters
    ----------
    enable
        Enable or disable the global string cache.

    Examples
    --------
    >>> pl.enable_string_cache(True)
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
    >>> df_join = df1.join(df2, how="inner", on="color")
    >>> df_join
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
    _enable_string_cache(enable)


def toggle_string_cache(toggle: bool) -> None:
    """
    Enable (or disable) the global string cache.

    This ensures that casts to Categorical dtypes will have
    the same category values when string values are equal.

    .. deprecated:: 0.17.0

    """
    warnings.warn(
        "`toggle_string_cache` has been renamed; this"
        " redirect is temporary, please use `enable_string_cache` instead",
        category=DeprecationWarning,
        stacklevel=find_stacklevel(),
    )
    enable_string_cache(toggle)


def using_string_cache() -> bool:
    """Return the current state of the global string cache (enabled/disabled)."""
    return _using_string_cache()
