from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars.polars import PySeries


@expr_dispatch
class ArrayNameSpace:
    """A namespace for :class:`Array` `Series`."""

    _accessor = "arr"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def min(self) -> Series:
        """
        Get the minimum value of the elements in each array.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
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
        Get the maximum value of the elements in each array.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
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
        Get the sum of the elements in each array.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(pl.Int64, 2)},
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

    def unique(self, *, maintain_order: bool = False) -> Series:
        """
        Get the unique values that appear in each array, removing duplicates.

        Parameters
        ----------
        maintain_order
            Whether to keep the unique elements in the same order as in the input data.
            This is slower.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 1, 2]],
        ...     },
        ...     schema_overrides={"a": pl.Array(pl.Int64, 3)},
        ... )
        >>> df.select(pl.col("a").arr.unique())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2]    │
        └───────────┘

        """

    def to_list(self) -> Series:
        """
        Convert an :class:`Array` `Series` into a :class:`List` `Series`.

        Returns
        -------
        Series
            A :class:`List` `Series`.

        Examples
        --------
        >>> s = pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2))
        >>> s.arr.to_list()
        shape: (2,)
        Series: '' [list[i8]]
        [
                [1, 2]
                [3, 4]
        ]

        """

    def any(self) -> Series:
        """
        Evaluate whether any :class:`Boolean` value in each array is true.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series(
        ...     [[True, True], [False, True], [False, False], [None, None], None],
        ...     dtype=pl.Array(pl.Boolean, 2),
        ... )
        >>> s.arr.any()
        shape: (5,)
        Series: '' [bool]
        [
            true
            true
            false
            false
            null
        ]

        """

    def all(self) -> Series:
        """
        Evaluate whether all :class:`Boolean` values in each array are `true`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series(
        ...     [[True, True], [False, True], [False, False], [None, None], None],
        ...     dtype=pl.Array(pl.Boolean, 2),
        ... )
        >>> s.arr.all()
        shape: (5,)
        Series: '' [bool]
        [
            true
            false
            false
            true
            null
        ]

        """
