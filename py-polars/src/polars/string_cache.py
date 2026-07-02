from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars._utils.deprecation import deprecated

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


__all__ = [
    "StringCache",
    "disable_string_cache",
    "enable_string_cache",
    "using_string_cache",
]


@deprecated("the string cache has been replaced by pl.Categories")
class StringCache(contextlib.ContextDecorator):
    """
    Does nothing.

    .. deprecated:: 1.41.0
        The string cache was used to maintain the mapping for the Categorical
        dtype, this is now done through ``pl.Categories``.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return


@deprecated("the string cache has been replaced by pl.Categories")
def enable_string_cache() -> None:
    """
    Does nothing.

    .. deprecated:: 1.41.0
        The string cache was used to maintain the mapping for the Categorical
        dtype, this is now done through ``pl.Categories``.
    """


@deprecated("the string cache has been replaced by pl.Categories")
def disable_string_cache() -> None:
    """
    Does nothing.

    .. deprecated:: 1.41.0
        The string cache was used to maintain the mapping for the Categorical
        dtype, this is now done through ``pl.Categories``.
    """


@deprecated("the string cache has been replaced by pl.Categories")
def using_string_cache() -> bool:
    """
    Always returns true.

    .. deprecated:: 1.41.0
        The string cache was used to maintain the mapping for the Categorical
        dtype, this is now done through ``pl.Categories``.
    """
    return True
