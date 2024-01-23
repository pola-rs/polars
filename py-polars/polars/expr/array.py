from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from datetime import date, datetime, time

    from polars import Expr
    from polars.type_aliases import IntoExpr, IntoExprColumn


class ExprArrayNameSpace:
    """Namespace for array related expressions."""

    _accessor = "arr"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def min(self) -> Expr:
        """
        Compute the min values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.select(pl.col("a").arr.min())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 3   │
        └─────┘
        """
        return wrap_expr(self._pyexpr.arr_min())

    def max(self) -> Expr:
        """
        Compute the max values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.select(pl.col("a").arr.max())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 4   │
        └─────┘
        """
        return wrap_expr(self._pyexpr.arr_max())

    def sum(self) -> Expr:
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
        return wrap_expr(self._pyexpr.arr_sum())

    def unique(self, *, maintain_order: bool = False) -> Expr:
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
        ...     schema={"a": pl.Array(pl.Int64, 3)},
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
        return wrap_expr(self._pyexpr.arr_unique(maintain_order))

    def to_list(self) -> Expr:
        """
        Convert an Array column into a List column with the same inner data type.

        Returns
        -------
        Expr
            Expression of data type :class:`List`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [3, 4]]},
        ...     schema={"a": pl.Array(pl.Int8, 2)},
        ... )
        >>> df.select(pl.col("a").arr.to_list())
        shape: (2, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ list[i8] │
        ╞══════════╡
        │ [1, 2]   │
        │ [3, 4]   │
        └──────────┘
        """
        return wrap_expr(self._pyexpr.arr_to_list())

    def any(self) -> Expr:
        """
        Evaluate whether any boolean value is true for every subarray.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "a": [
        ...             [True, True],
        ...             [False, True],
        ...             [False, False],
        ...             [None, None],
        ...             None,
        ...         ]
        ...     },
        ...     schema={"a": pl.Array(pl.Boolean, 2)},
        ... )
        >>> df.with_columns(any=pl.col("a").arr.any())
        shape: (5, 2)
        ┌────────────────┬───────┐
        │ a              ┆ any   │
        │ ---            ┆ ---   │
        │ array[bool, 2] ┆ bool  │
        ╞════════════════╪═══════╡
        │ [true, true]   ┆ true  │
        │ [false, true]  ┆ true  │
        │ [false, false] ┆ false │
        │ [null, null]   ┆ false │
        │ null           ┆ null  │
        └────────────────┴───────┘
        """
        return wrap_expr(self._pyexpr.arr_any())

    def all(self) -> Expr:
        """
        Evaluate whether all boolean values are true for every subarray.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "a": [
        ...             [True, True],
        ...             [False, True],
        ...             [False, False],
        ...             [None, None],
        ...             None,
        ...         ]
        ...     },
        ...     schema={"a": pl.Array(pl.Boolean, 2)},
        ... )
        >>> df.with_columns(all=pl.col("a").arr.all())
        shape: (5, 2)
        ┌────────────────┬───────┐
        │ a              ┆ all   │
        │ ---            ┆ ---   │
        │ array[bool, 2] ┆ bool  │
        ╞════════════════╪═══════╡
        │ [true, true]   ┆ true  │
        │ [false, true]  ┆ false │
        │ [false, false] ┆ false │
        │ [null, null]   ┆ true  │
        │ null           ┆ null  │
        └────────────────┴───────┘
        """
        return wrap_expr(self._pyexpr.arr_all())

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     },
        ...     schema={"a": pl.Array(pl.Int64, 3)},
        ... )
        >>> df.with_columns(sort=pl.col("a").arr.sort())
        shape: (2, 2)
        ┌───────────────┬───────────────┐
        │ a             ┆ sort          │
        │ ---           ┆ ---           │
        │ array[i64, 3] ┆ array[i64, 3] │
        ╞═══════════════╪═══════════════╡
        │ [3, 2, 1]     ┆ [1, 2, 3]     │
        │ [9, 1, 2]     ┆ [1, 2, 9]     │
        └───────────────┴───────────────┘
        >>> df.with_columns(sort=pl.col("a").arr.sort(descending=True))
        shape: (2, 2)
        ┌───────────────┬───────────────┐
        │ a             ┆ sort          │
        │ ---           ┆ ---           │
        │ array[i64, 3] ┆ array[i64, 3] │
        ╞═══════════════╪═══════════════╡
        │ [3, 2, 1]     ┆ [3, 2, 1]     │
        │ [9, 1, 2]     ┆ [9, 2, 1]     │
        └───────────────┴───────────────┘

        """
        return wrap_expr(self._pyexpr.arr_sort(descending, nulls_last))

    def reverse(self) -> Expr:
        """
        Reverse the arrays in this column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     },
        ...     schema={"a": pl.Array(pl.Int64, 3)},
        ... )
        >>> df.with_columns(reverse=pl.col("a").arr.reverse())
        shape: (2, 2)
        ┌───────────────┬───────────────┐
        │ a             ┆ reverse       │
        │ ---           ┆ ---           │
        │ array[i64, 3] ┆ array[i64, 3] │
        ╞═══════════════╪═══════════════╡
        │ [3, 2, 1]     ┆ [1, 2, 3]     │
        │ [9, 1, 2]     ┆ [2, 1, 9]     │
        └───────────────┴───────────────┘

        """
        return wrap_expr(self._pyexpr.arr_reverse())

    def arg_min(self) -> Expr:
        """
        Retrieve the index of the minimal value in every sub-array.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32` or :class:`UInt64`
            (depending on compilation).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2], [2, 1]],
        ...     },
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.with_columns(arg_min=pl.col("a").arr.arg_min())
        shape: (2, 2)
        ┌───────────────┬─────────┐
        │ a             ┆ arg_min │
        │ ---           ┆ ---     │
        │ array[i64, 2] ┆ u32     │
        ╞═══════════════╪═════════╡
        │ [1, 2]        ┆ 0       │
        │ [2, 1]        ┆ 1       │
        └───────────────┴─────────┘

        """
        return wrap_expr(self._pyexpr.arr_arg_min())

    def arg_max(self) -> Expr:
        """
        Retrieve the index of the maximum value in every sub-array.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32` or :class:`UInt64`
            (depending on compilation).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2], [2, 1]],
        ...     },
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.with_columns(arg_max=pl.col("a").arr.arg_max())
        shape: (2, 2)
        ┌───────────────┬─────────┐
        │ a             ┆ arg_max │
        │ ---           ┆ ---     │
        │ array[i64, 2] ┆ u32     │
        ╞═══════════════╪═════════╡
        │ [1, 2]        ┆ 1       │
        │ [2, 1]        ┆ 0       │
        └───────────────┴─────────┘

        """
        return wrap_expr(self._pyexpr.arr_arg_max())

    def get(self, index: int | IntoExprColumn) -> Expr:
        """
        Get the value by index in the sub-arrays.

        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sub-array

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"arr": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "idx": [1, -2, 4]},
        ...     schema={"arr": pl.Array(pl.Int32, 3), "idx": pl.Int32},
        ... )
        >>> df.with_columns(get=pl.col("arr").arr.get("idx"))
        shape: (3, 3)
        ┌───────────────┬─────┬──────┐
        │ arr           ┆ idx ┆ get  │
        │ ---           ┆ --- ┆ ---  │
        │ array[i32, 3] ┆ i32 ┆ i32  │
        ╞═══════════════╪═════╪══════╡
        │ [1, 2, 3]     ┆ 1   ┆ 2    │
        │ [4, 5, 6]     ┆ -2  ┆ 5    │
        │ [7, 8, 9]     ┆ 4   ┆ null │
        └───────────────┴─────┴──────┘

        """
        index = parse_as_expression(index)
        return wrap_expr(self._pyexpr.arr_get(index))

    def first(self) -> Expr:
        """
        Get the first value of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        ...     schema={"a": pl.Array(pl.Int32, 3)},
        ... )
        >>> df.with_columns(first=pl.col("a").arr.first())
        shape: (3, 2)
        ┌───────────────┬───────┐
        │ a             ┆ first │
        │ ---           ┆ ---   │
        │ array[i32, 3] ┆ i32   │
        ╞═══════════════╪═══════╡
        │ [1, 2, 3]     ┆ 1     │
        │ [4, 5, 6]     ┆ 4     │
        │ [7, 8, 9]     ┆ 7     │
        └───────────────┴───────┘

        """
        return self.get(0)

    def last(self) -> Expr:
        """
        Get the last value of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        ...     schema={"a": pl.Array(pl.Int32, 3)},
        ... )
        >>> df.with_columns(last=pl.col("a").arr.last())
        shape: (3, 2)
        ┌───────────────┬──────┐
        │ a             ┆ last │
        │ ---           ┆ ---  │
        │ array[i32, 3] ┆ i32  │
        ╞═══════════════╪══════╡
        │ [1, 2, 3]     ┆ 3    │
        │ [4, 5, 6]     ┆ 6    │
        │ [7, 8, 9]     ┆ 9    │
        └───────────────┴──────┘

        """
        return self.get(-1)

    def join(self, separator: IntoExprColumn) -> Expr:
        """
        Join all string items in a sub-array and place a separator between them.

        This errors if inner type of array `!= String`.

        Parameters
        ----------
        separator
            string to separate the items with

        Returns
        -------
        Expr
            Expression of data type :class:`String`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"s": [["a", "b"], ["x", "y"]], "separator": ["*", "_"]},
        ...     schema={
        ...         "s": pl.Array(pl.String, 2),
        ...         "separator": pl.String,
        ...     },
        ... )
        >>> df.with_columns(join=pl.col("s").arr.join(pl.col("separator")))
        shape: (2, 3)
        ┌───────────────┬───────────┬──────┐
        │ s             ┆ separator ┆ join │
        │ ---           ┆ ---       ┆ ---  │
        │ array[str, 2] ┆ str       ┆ str  │
        ╞═══════════════╪═══════════╪══════╡
        │ ["a", "b"]    ┆ *         ┆ a*b  │
        │ ["x", "y"]    ┆ _         ┆ x_y  │
        └───────────────┴───────────┴──────┘

        """
        separator = parse_as_expression(separator, str_as_lit=True)
        return wrap_expr(self._pyexpr.arr_join(separator))

    def explode(self) -> Expr:
        """
        Returns a column with a separate row for every array element.

        Returns
        -------
        Expr
            Expression with the data type of the array elements.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[1, 2, 3], [4, 5, 6]]}, schema={"a": pl.Array(pl.Int64, 3)}
        ... )
        >>> df.select(pl.col("a").arr.explode())
        shape: (6, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        │ 4   │
        │ 5   │
        │ 6   │
        └─────┘
        """
        return wrap_expr(self._pyexpr.explode())

    def contains(
        self, item: float | str | bool | int | date | datetime | time | IntoExprColumn
    ) -> Expr:
        """
        Check if sub-arrays contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Expr
            Expression of data type :class:`Boolean`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [["a", "b"], ["x", "y"], ["a", "c"]]},
        ...     schema={"a": pl.Array(pl.String, 2)},
        ... )
        >>> df.with_columns(contains=pl.col("a").arr.contains("a"))
        shape: (3, 2)
        ┌───────────────┬──────────┐
        │ a             ┆ contains │
        │ ---           ┆ ---      │
        │ array[str, 2] ┆ bool     │
        ╞═══════════════╪══════════╡
        │ ["a", "b"]    ┆ true     │
        │ ["x", "y"]    ┆ false    │
        │ ["a", "c"]    ┆ true     │
        └───────────────┴──────────┘

        """
        item = parse_as_expression(item, str_as_lit=True)
        return wrap_expr(self._pyexpr.arr_contains(item))

    def count_matches(self, element: IntoExpr) -> Expr:
        """
        Count how often the value produced by `element` occurs.

        Parameters
        ----------
        element
            An expression that produces a single value

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[1, 2], [1, 1], [2, 2]]}, schema={"a": pl.Array(pl.Int64, 2)}
        ... )
        >>> df.with_columns(number_of_twos=pl.col("a").arr.count_matches(2))
        shape: (3, 2)
        ┌───────────────┬────────────────┐
        │ a             ┆ number_of_twos │
        │ ---           ┆ ---            │
        │ array[i64, 2] ┆ u32            │
        ╞═══════════════╪════════════════╡
        │ [1, 2]        ┆ 1              │
        │ [1, 1]        ┆ 0              │
        │ [2, 2]        ┆ 2              │
        └───────────────┴────────────────┘
        """
        element = parse_as_expression(element, str_as_lit=True)
        return wrap_expr(self._pyexpr.arr_count_matches(element))
