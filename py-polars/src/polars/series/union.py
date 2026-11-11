from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars._plr import PySeries


@expr_dispatch
class UnionNameSpace:
    """Series.union namespace."""

    _accessor = "union"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    def tag(self) -> Series:
        """
        Get the variant of every row.

        Returns an :class:`Enum` of the variant names, in variant order. Null rows
        give null.

        .. engine-support:: in-memory, streaming

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> s = pl.select(pl.union_value(dtype, i=pl.Series([1, None]))).to_series()
        >>> s.union.tag()
        shape: (2,)
        Series: '' [enum]
        [
                "i"
                null
        ]
        """

    def field(self, name: str) -> Series:
        """
        Get the values of one variant.

        Rows of other variants and null rows give null.

        .. engine-support:: in-memory, streaming

        Parameters
        ----------
        name
            The name of the variant.

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> s = pl.select(pl.union_value(dtype, i=pl.Series([1, None]))).to_series()
        >>> s.union.field("i")
        shape: (2,)
        Series: '' [i64]
        [
                1
                null
        ]
        """
