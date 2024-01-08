from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr


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

    def std(self, ddof: int = 1) -> Expr:
        """
        Compute the std of the values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.select(pl.col("a").arr.std())
        shape: (2, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.707107 │
        │ 0.707107 │
        └──────────┘
        """
        return wrap_expr(self._pyexpr.arr_std(ddof))

    def var(self, ddof: int = 1) -> Expr:
        """
        Compute the var of the values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.select(pl.col("a").arr.var())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.5 │
        │ 0.5 │
        └─────┘
        """
        return wrap_expr(self._pyexpr.arr_var(ddof))

    def median(self) -> Expr:
        """
        Compute the median of the values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(pl.Int64, 2)},
        ... )
        >>> df.select(pl.col("a").arr.median())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.5 │
        │ 3.5 │
        └─────┘
        """
        return wrap_expr(self._pyexpr.arr_median())

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

    def sort(self, *, descending: bool = False) -> Expr:
        """
        Sort the arrays in this column.

        Parameters
        ----------
        descending
            Sort in descending order.

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
        return wrap_expr(self._pyexpr.arr_sort(descending))

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
