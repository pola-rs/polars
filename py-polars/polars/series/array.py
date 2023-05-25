from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars.polars import PySeries


@expr_dispatch
class ArrayNameSpace:
    """Namespace for list related methods."""

    _accessor = "arr"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def min(self) -> Series:
        """
        Compute the min values of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series(
        ...     "a", [[1, 2], [4, 3]], dtype=pl.Array(width=2, inner=pl.Int64)
        ... )
        >>> s.arr.min()
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            3
        ]

        """

    def max(self) -> Series:
        """
        Compute the max values of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series(
        ...     "a", [[1, 2], [4, 3]], dtype=pl.Array(width=2, inner=pl.Int64)
        ... )
        >>> s.arr.max()
        shape: (2,)
        Series: 'a' [i64]
        [
            2
            4
        ]

        """

    def sum(self) -> Series:
        """
        Compute the sum values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(width=2, inner=pl.Int64)},
        ... )
        >>> df.select(pl.col("a").arr.sum())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 7   │
        └─────┘

        """
