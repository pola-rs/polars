from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import polars.internals as pli
from polars.internals.series.utils import call_expr

if TYPE_CHECKING:
    from polars.internals.type_aliases import NullBehavior
    from polars.polars import PySeries


class ListNameSpace:
    """Series.arr namespace."""

    _s: PySeries

    def __init__(self, series: pli.Series):
        self._s = series._s

    @call_expr(namespace="arr")
    def lengths(self) -> pli.Series:
        """
        Get the length of the arrays as UInt32.

        Examples
        --------
        >>> s = pl.Series([[1, 2, 3], [5]])
        >>> s.arr.lengths()
        shape: (2,)
        Series: '' [u32]
        [
            3
            1
        ]

        """
        ...

    @call_expr(namespace="arr")
    def sum(self) -> pli.Series:
        """Sum all the arrays in the list."""
        ...

    @call_expr(namespace="arr")
    def max(self) -> pli.Series:
        """Compute the max value of the arrays in the list."""
        ...

    @call_expr(namespace="arr")
    def min(self) -> pli.Series:
        """Compute the min value of the arrays in the list."""
        ...

    @call_expr(namespace="arr")
    def mean(self) -> pli.Series:
        """Compute the mean value of the arrays in the list."""
        ...

    @call_expr(namespace="arr")
    def sort(self, reverse: bool = False) -> pli.Series:
        """Sort the arrays in the list."""
        ...

    @call_expr(namespace="arr")
    def reverse(self) -> pli.Series:
        """Reverse the arrays in the list."""
        ...

    @call_expr(namespace="arr")
    def unique(self) -> pli.Series:
        """Get the unique/distinct values in the list."""
        ...

    @call_expr(namespace="arr")
    def concat(self, other: list[pli.Series] | pli.Series | list[Any]) -> pli.Series:
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series

        """
        ...

    @call_expr(namespace="arr")
    def get(self, index: int) -> pli.Series:
        """
        Get the value by index in the sublists.
        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sublist

        """
        ...

    @call_expr(namespace="arr")
    def join(self, separator: str) -> pli.Series:
        """
        Join all string items in a sublist and place a separator between them.
        This errors if inner type of list `!= Utf8`.

        Parameters
        ----------
        separator
            string to separate the items with

        Returns
        -------
        Series of dtype Utf8

        Examples
        --------
        >>> s = pl.Series([["foo", "bar"], ["hello", "world"]])
        >>> s.arr.join(separator="-")
        shape: (2,)
        Series: '' [str]
        [
            "foo-bar"
            "hello-world"
        ]

        """
        ...

    @call_expr(namespace="arr")
    def first(self) -> pli.Series:
        """Get the first value of the sublists."""
        ...

    @call_expr(namespace="arr")
    def last(self) -> pli.Series:
        """Get the last value of the sublists."""
        ...

    @call_expr(namespace="arr")
    def contains(self, item: float | str | bool | int | date | datetime) -> pli.Series:
        """
        Check if sublists contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Boolean mask

        """
        ...

    @call_expr(namespace="arr")
    def arg_min(self) -> pli.Series:
        """
        Retrieve the index of the minimal value in every sublist

        Returns
        -------
        Series of dtype UInt32/UInt64 (depending on compilation)

        """
        ...

    @call_expr(namespace="arr")
    def arg_max(self) -> pli.Series:
        """
        Retrieve the index of the maximum value in every sublist

        Returns
        -------
        Series of dtype UInt32/UInt64 (depending on compilation)

        """
        ...

    @call_expr(namespace="arr")
    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> pli.Series:
        """
        Calculate the n-th discrete difference of every sublist.

        Parameters
        ----------
        n
            Number of slots to shift.
        null_behavior : {'ignore', 'drop'}
            How to handle null values.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.diff()
        shape: (2,)
        Series: 'a' [list]
        [
            [null, 1, ... 1]
            [null, -8, -1]
        ]

        """
        ...

    @call_expr(namespace="arr")
    def shift(self, periods: int = 1) -> pli.Series:
        """
        Shift the values by a given period and fill the parts that will be empty due to
        this operation with nulls.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.shift()
        shape: (2,)
        Series: 'a' [list]
        [
            [null, 1, ... 3]
            [null, 10, 2]
        ]

        """
        ...

    @call_expr(namespace="arr")
    def slice(self, offset: int, length: int) -> pli.Series:
        """
        Slice every sublist

        Parameters
        ----------
        offset
            Take the values from this index offset
        length
            The length of the slice to take

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.slice(1, 2)
        shape: (2,)
        Series: 'a' [list]
        [
            [2, 3]
            [2, 1]
        ]

        """
        ...

    @call_expr(namespace="arr")
    def head(self, n: int = 5) -> pli.Series:
        """
        Slice the head of every sublist

        Parameters
        ----------
        n
            How many values to take in the slice.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.head(2)
        shape: (2,)
        Series: 'a' [list]
        [
            [1, 2]
            [10, 2]
        ]

        """
        ...

    @call_expr(namespace="arr")
    def tail(self, n: int = 5) -> pli.Series:
        """
        Slice the tail of every sublist

        Parameters
        ----------
        n
            How many values to take in the slice.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.tail(2)
        shape: (2,)
        Series: 'a' [list]
        [
            [3, 4]
            [2, 1]
        ]

        """
        ...

    @call_expr(namespace="arr")
    def eval(self, expr: pli.Expr, parallel: bool = False) -> pli.Series:
        """
        Run any polars expression against the lists' elements

        Parameters
        ----------
        expr
            Expression to run. Note that you can select an element with `pl.first()`, or
            `pl.col()`
        parallel
            Run all expression parallel. Don't activate this blindly.
            Parallelism is worth it if there is enough work to do per thread.

            This likely should not be use in the groupby context, because we already
            parallel execution per group

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
        >>> df.with_column(
        ...     pl.concat_list(["a", "b"]).arr.eval(pl.element().rank()).alias("rank")
        ... )
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ a   ┆ b   ┆ rank       │
        │ --- ┆ --- ┆ ---        │
        │ i64 ┆ i64 ┆ list[f32]  │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 4   ┆ [1.0, 2.0] │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 8   ┆ 5   ┆ [2.0, 1.0] │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 2   ┆ [2.0, 1.0] │
        └─────┴─────┴────────────┘

        """
        ...
