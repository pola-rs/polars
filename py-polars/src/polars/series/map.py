from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars._plr import PySeries
    from polars._typing import IntoExpr


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
        preserved. The inverse of :meth:`Expr.list.to_map`.

        .. engine-support:: in-memory, streaming, distributed

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

    def keys(self) -> Series:
        """
        Get the keys of every map as a `List`, in entry order.

        .. engine-support:: in-memory, streaming, distributed

        Returns
        -------
        Series
            Series of data type :class:`List` of the map's key type.

        Examples
        --------
        >>> s = pl.Series(
        ...     [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ... )
        >>> s.map.keys()
        shape: (3,)
        Series: '' [list[str]]
        [
                ["a", "b"]
                []
                null
        ]
        """

    def values(self) -> Series:
        """
        Get the values of every map as a `List`, in entry order.

        .. engine-support:: in-memory, streaming, distributed

        Returns
        -------
        Series
            Series of data type :class:`List` of the map's value type.

        Examples
        --------
        >>> s = pl.Series(
        ...     [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ... )
        >>> s.map.values()
        shape: (3,)
        Series: '' [list[i64]]
        [
                [1, 2]
                []
                null
        ]
        """

    def len(self) -> Series:
        """
        Get the number of entries of every map.

        .. engine-support:: in-memory, streaming, distributed

        Returns
        -------
        Series
            Series of data type :class:`UInt32`.

        Examples
        --------
        >>> s = pl.Series(
        ...     [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ... )
        >>> s.map.len()
        shape: (3,)
        Series: '' [u32]
        [
                2
                0
                null
        ]
        """

    def contains_key(self, key: IntoExpr) -> Series:
        """
        Check whether every map holds `key`.

        .. engine-support:: in-memory, streaming, distributed

        Parameters
        ----------
        key
            Key to look up. A single value is broadcast over all maps.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series(
        ...     [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ... )
        >>> s.map.contains_key("a")
        shape: (3,)
        Series: '' [bool]
        [
                true
                false
                null
        ]
        """

    def get(self, key: IntoExpr) -> Series:
        """
        Look up `key` in every map.

        Maps that do not hold `key` yield null, as do null maps. Use
        :meth:`contains_key` to tell a missing key apart from a key whose value is null.

        .. engine-support:: in-memory, streaming, distributed

        Parameters
        ----------
        key
            Key to look up. A single value is broadcast over all maps.

        Returns
        -------
        Series
            Series of the map's value type.

        Examples
        --------
        >>> s = pl.Series(
        ...     [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ... )
        >>> s.map.get("a")
        shape: (3,)
        Series: '' [i64]
        [
                1
                null
                null
        ]
        """
