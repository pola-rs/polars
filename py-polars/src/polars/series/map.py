from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars._plr import PySeries


@expr_dispatch
class MapNameSpace:
    """Series.map namespace."""

    _accessor = "map"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    def entries(self) -> Series:
        """
        Convert the `Map` to a `List` of `Struct` entries.

        Each entry is a `Struct` with a `key` and a `value` field. Entry order is
        preserved. The inverse of :meth:`Series.list.to_map`.

        Examples
        --------
        >>> s = pl.Series([{"a": 1, "b": 2}], dtype=pl.Map(pl.String, pl.Int64))
        >>> s.map.entries()
        shape: (1,)
        Series: '' [list[struct[2]]]
        [
                [{"a",1}, {"b",2}]
        ]
        """
