from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from datetime import date, datetime, time

    from polars import Series
    from polars.polars import PySeries
    from polars.type_aliases import IntoExpr, IntoExprColumn


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
        Compute the max values of the sub-arrays.

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
        Compute the sum values of the sub-arrays.

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

    def std(self, ddof: int = 1) -> Series:
        """
        Compute the std of the values of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
        >>> s.arr.std()
        shape: (2,)
        Series: 'a' [f64]
        [
            0.707107
            0.707107
        ]
        """

    def var(self, ddof: int = 1) -> Series:
        """
        Compute the var of the values of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
        >>> s.arr.var()
        shape: (2,)
        Series: 'a' [f64]
        [
                0.5
                0.5
        ]
        """

    def median(self) -> Series:
        """
        Compute the median of the values of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
        >>> s.arr.median()
        shape: (2,)
        Series: 'a' [f64]
        [
            1.5
            3.5
        ]
        """

    def unique(self, *, maintain_order: bool = False) -> Series:
        """
        Get the unique/distinct values in the array.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

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
        Convert an Array column into a List column with the same inner data type.

        Returns
        -------
        Series
            Series of data type :class:`List`.

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
        Evaluate whether any boolean value is true for every subarray.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

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
        Evaluate whether all boolean values are true for every subarray.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

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

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Series:
        """
        Sort the arrays in this column.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [9, 1, 2]], dtype=pl.Array(pl.Int64, 3))
        >>> s.arr.sort()
        shape: (2,)
        Series: 'a' [array[i64, 3]]
        [
            [1, 2, 3]
            [1, 2, 9]
        ]
        >>> s.arr.sort(descending=True)
        shape: (2,)
        Series: 'a' [array[i64, 3]]
        [
            [3, 2, 1]
            [9, 2, 1]
        ]

        """

    def reverse(self) -> Series:
        """
        Reverse the arrays in this column.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [9, 1, 2]], dtype=pl.Array(pl.Int64, 3))
        >>> s.arr.reverse()
        shape: (2,)
        Series: 'a' [array[i64, 3]]
        [
            [1, 2, 3]
            [2, 1, 9]
        ]

        """

    def arg_min(self) -> Series:
        """
        Retrieve the index of the minimal value in every sub-array.

        Returns
        -------
        Series
            Series of data type :class:`UInt32` or :class:`UInt64`
            (depending on compilation).

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [9, 1, 2]], dtype=pl.Array(pl.Int64, 3))
        >>> s.arr.arg_min()
        shape: (2,)
        Series: 'a' [u32]
        [
            2
            1
        ]

        """

    def arg_max(self) -> Series:
        """
        Retrieve the index of the maximum value in every sub-array.

        Returns
        -------
        Series
            Series of data type :class:`UInt32` or :class:`UInt64`
            (depending on compilation).

        Examples
        --------
        >>> s = pl.Series("a", [[0, 9, 3], [9, 1, 2]], dtype=pl.Array(pl.Int64, 3))
        >>> s.arr.arg_max()
        shape: (2,)
        Series: 'a' [u32]
        [
            1
            0
        ]

        """

    def get(self, index: int | IntoExprColumn) -> Series:
        """
        Get the value by index in the sub-arrays.

        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sublist

        Returns
        -------
        Series
            Series of innter data type.

        Examples
        --------
        >>> s = pl.Series(
        ...     "a", [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=pl.Array(pl.Int32, 3)
        ... )
        >>> s.arr.get(pl.Series([1, -2, 4]))
        shape: (3,)
        Series: 'a' [i32]
        [
            2
            5
            null
        ]

        """

    def first(self) -> Series:
        """
        Get the first value of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series(
        ...     "a", [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=pl.Array(pl.Int32, 3)
        ... )
        >>> s.arr.first()
        shape: (3,)
        Series: 'a' [i32]
        [
            1
            4
            7
        ]

        """

    def last(self) -> Series:
        """
        Get the last value of the sub-arrays.

        Examples
        --------
        >>> s = pl.Series(
        ...     "a", [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=pl.Array(pl.Int32, 3)
        ... )
        >>> s.arr.last()
        shape: (3,)
        Series: 'a' [i32]
        [
            3
            6
            9
        ]

        """

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> Series:
        """
        Join all string items in a sub-array and place a separator between them.

        This errors if inner type of array `!= String`.

        Parameters
        ----------
        separator
            string to separate the items with
        ignore_nulls
            Ignore null values (default).

            If set to ``False``, null values will be propagated.
            If the sub-list contains any null values, the output is ``None``.

        Returns
        -------
        Series
            Series of data type :class:`String`.

        Examples
        --------
        >>> s = pl.Series([["x", "y"], ["a", "b"]], dtype=pl.Array(pl.String, 2))
        >>> s.arr.join(separator="-")
        shape: (2,)
        Series: '' [str]
        [
            "x-y"
            "a-b"
        ]

        """

    def contains(
        self, item: float | str | bool | int | date | datetime | time | IntoExprColumn
    ) -> Series:
        """
        Check if sub-arrays contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series(
        ...     "a", [[3, 2, 1], [1, 2, 3], [4, 5, 6]], dtype=pl.Array(pl.Int32, 3)
        ... )
        >>> s.arr.contains(1)
        shape: (3,)
        Series: 'a' [bool]
        [
            true
            true
            false
        ]

        """

    def count_matches(self, element: IntoExpr) -> Series:
        """
        Count how often the value produced by `element` occurs.

        Parameters
        ----------
        element
            An expression that produces a single value

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3], [2, 2, 2]], dtype=pl.Array(pl.Int64, 3))
        >>> s.arr.count_matches(2)
        shape: (2,)
        Series: 'a' [u32]
        [
            1
            3
        ]

        """
