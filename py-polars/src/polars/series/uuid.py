from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars._plr import PySeries


@expr_dispatch
class UuidNameSpace:
    """Namespace for UUID-related series methods."""

    _accessor = "uuid"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    def version(self) -> Series:
        """Extract the four-bit UUID version field as an unsigned 8-bit integer."""

    def timestamp(self, *, strict: bool = True) -> Series:
        """Extract the UTC millisecond timestamp encoded by UUIDv7."""
