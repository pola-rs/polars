from __future__ import annotations

import copy
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, Callable

import polars.internals as pli

if TYPE_CHECKING:
    from polars.internals.type_aliases import NullBehavior, ToStructStrategy


class ExprListNameSpace:
    """Namespace for list related expressions."""

    _accessor = "arr"

    def __init__(self, expr: pli.Expr):
        self._pyexpr = expr._pyexpr

    def lengths(self) -> pli.Expr:
        """
        Get the length of the arrays as UInt32.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2], "bar": [["a", "b"], ["c"]]})
        >>> df.select(pl.col("bar").arr.lengths())
        shape: (2, 1)
        ┌─────┐
        │ bar │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        │ 1   │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.arr_lengths())

    def sum(self) -> pli.Expr:
        """
        Sum all the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").arr.sum())
        shape: (2, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 1      │
        │ 5      │
        └────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_sum())

    def max(self) -> pli.Expr:
        """
        Compute the max value of the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").arr.max())
        shape: (2, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 1      │
        │ 3      │
        └────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_max())

    def min(self) -> pli.Expr:
        """
        Compute the min value of the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").arr.min())
        shape: (2, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 1      │
        │ 2      │
        └────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_min())

    def mean(self) -> pli.Expr:
        """
        Compute the mean value of the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").arr.mean())
        shape: (2, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ f64    │
        ╞════════╡
        │ 1.0    │
        │ 2.5    │
        └────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_mean())

    def sort(self, reverse: bool = False) -> pli.Expr:
        """
        Sort the arrays in the list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.sort())
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        │ [1, 2, 9] │
        └───────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_sort(reverse))

    def reverse(self) -> pli.Expr:
        """
        Reverse the arrays in the list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.reverse())
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        │ [2, 1, 9] │
        └───────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_reverse())

    def unique(self) -> pli.Expr:
        """
        Get the unique/distinct values in the list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 1, 2]],
        ...     }
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
        return pli.wrap_expr(self._pyexpr.lst_unique())

    def concat(
        self, other: list[pli.Expr | str] | pli.Expr | str | pli.Series | list[Any]
    ) -> pli.Expr:
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [["a"], ["x"]],
        ...         "b": [["b", "c"], ["y", "z"]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.concat("b"))
        shape: (2, 1)
        ┌─────────────────┐
        │ a               │
        │ ---             │
        │ list[str]       │
        ╞═════════════════╡
        │ ["a", "b", "c"] │
        │ ["x", "y", "z"] │
        └─────────────────┘

        """
        if isinstance(other, list) and (
            not isinstance(other[0], (pli.Expr, str, pli.Series))
        ):
            return self.concat(pli.Series([other]))

        other_list: list[pli.Expr | str | pli.Series]
        if not isinstance(other, list):
            other_list = [other]
        else:
            other_list = copy.copy(other)  # type: ignore[arg-type]

        other_list.insert(0, pli.wrap_expr(self._pyexpr))
        return pli.concat_list(other_list)

    def get(self, index: int | pli.Expr | str) -> pli.Expr:
        """
        Get the value by index in the sublists.

        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sublist

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.get(0))
        shape: (3, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 3    │
        │ null │
        │ 1    │
        └──────┘

        """
        index = pli.expr_to_lit_or_expr(index, str_to_lit=False)._pyexpr
        return pli.wrap_expr(self._pyexpr.lst_get(index))

    def take(
        self,
        index: pli.Expr | pli.Series | list[int] | list[list[int]],
        null_on_oob: bool = False,
    ) -> pli.Expr:
        """
        Take sublists by multiple indices.

        The indices may be defined in a single column, or by sublists in another
        column of dtype ``List``.

        Parameters
        ----------
        index
            Indices to return per sublist
        null_on_oob
            Behavior if an index is out of bounds:
            True -> set as null
            False -> raise an error
            Note that defaulting to raising an error is much cheaper

        """
        if isinstance(index, list):
            index = pli.Series(index)
        index = pli.expr_to_lit_or_expr(index, str_to_lit=False)._pyexpr
        return pli.wrap_expr(self._pyexpr.lst_take(index, null_on_oob))

    def __getitem__(self, item: int) -> pli.Expr:
        return self.get(item)

    def first(self) -> pli.Expr:
        """
        Get the first value of the sublists.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.first())
        shape: (3, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 3    │
        │ null │
        │ 1    │
        └──────┘

        """
        return self.get(0)

    def last(self) -> pli.Expr:
        """
        Get the last value of the sublists.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.last())
        shape: (3, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 1    │
        │ null │
        │ 2    │
        └──────┘

        """
        return self.get(-1)

    def contains(
        self, item: float | str | bool | int | date | datetime | time | pli.Expr
    ) -> pli.Expr:
        """
        Check if sublists contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Boolean mask

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").arr.contains(1))
        shape: (3, 1)
        ┌───────┐
        │ foo   │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ false │
        │ true  │
        └───────┘

        """
        return pli.wrap_expr(
            self._pyexpr.arr_contains(pli.expr_to_lit_or_expr(item)._pyexpr)
        )

    def join(self, separator: str) -> pli.Expr:
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
        >>> df = pl.DataFrame({"s": [["a", "b", "c"], ["x", "y"]]})
        >>> df.select(pl.col("s").arr.join(" "))
        shape: (2, 1)
        ┌───────┐
        │ s     │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a b c │
        │ x y   │
        └───────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_join(separator))

    def arg_min(self) -> pli.Expr:
        """
        Retrieve the index of the minimal value in every sublist.

        Returns
        -------
        Series of dtype UInt32/UInt64 (depending on compilation)

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2], [2, 1]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.arg_min())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        │ 1   │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_arg_min())

    def arg_max(self) -> pli.Expr:
        """
        Retrieve the index of the maximum value in every sublist.

        Returns
        -------
        Series of dtype UInt32/UInt64 (depending on compilation)

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2], [2, 1]],
        ...     }
        ... )
        >>> df.select(pl.col("a").arr.arg_max())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        │ 0   │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_arg_max())

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> pli.Expr:
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
        Series: 'a' [list[i64]]
        [
            [null, 1, ... 1]
            [null, -8, -1]
        ]

        """
        return pli.wrap_expr(self._pyexpr.lst_diff(n, null_behavior))

    def shift(self, periods: int = 1) -> pli.Expr:
        """
        Shift values by the given period.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.shift()
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [null, 1, ... 3]
            [null, 10, 2]
        ]

        """
        return pli.wrap_expr(self._pyexpr.lst_shift(periods))

    def slice(
        self, offset: int | str | pli.Expr, length: int | str | pli.Expr | None = None
    ) -> pli.Expr:
        """
        Slice every sublist.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None`` (default), the slice is taken to the
            end of the list.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.slice(1, 2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [2, 3]
            [2, 1]
        ]

        """
        offset = pli.expr_to_lit_or_expr(offset, str_to_lit=False)._pyexpr
        length = pli.expr_to_lit_or_expr(length, str_to_lit=False)._pyexpr
        return pli.wrap_expr(self._pyexpr.lst_slice(offset, length))

    def head(self, n: int | str | pli.Expr = 5) -> pli.Expr:
        """
        Slice the first `n` values of every sublist.

        Parameters
        ----------
        n
            Number of values to return for each sublist.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.head(2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [1, 2]
            [10, 2]
        ]

        """
        return self.slice(0, n)

    def tail(self, n: int | str | pli.Expr = 5) -> pli.Expr:
        """
        Slice the last `n` values of every sublist.

        Parameters
        ----------
        n
            Number of values to return for each sublist.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.arr.tail(2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [3, 4]
            [2, 1]
        ]

        """
        offset = -pli.expr_to_lit_or_expr(n, str_to_lit=False)
        return self.slice(offset, n)

    def to_struct(
        self,
        n_field_strategy: ToStructStrategy = "first_non_null",
        name_generator: Callable[[int], str] | None = None,
        upper_bound: int = 0,
    ) -> pli.Expr:
        """
        Convert the series of type ``List`` to a series of type ``Struct``.

        Parameters
        ----------
        n_field_strategy : {'first_non_null', 'max_width'}
            Strategy to determine the number of fields of the struct.
        name_generator
            A custom function that can be used to generate the field names.
            Default field names are `field_0, field_1 .. field_n`
        upper_bound
            A polars `LazyFrame` needs to know the schema at all time.
            The caller therefore must provide an `upper_bound` of
            struct fields that will be set.
            If this is incorrectly downstream operation may fail.
            For instance an `all().sum()` expression will look in
            the current schema to determine which columns to select.
            It is adviced to set this value in a lazy query.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3], [1, 2]]})
        >>> df.select([pl.col("a").arr.to_struct()])
        shape: (2, 1)
        ┌────────────┐
        │ a          │
        │ ---        │
        │ struct[3]  │
        ╞════════════╡
        │ {1,2,3}    │
        │ {1,2,null} │
        └────────────┘
        >>> df.select(
        ...     [
        ...         pl.col("a").arr.to_struct(
        ...             name_generator=lambda idx: f"col_name_{idx}"
        ...         )
        ...     ]
        ... ).to_series().to_list()
        [{'col_name_0': 1, 'col_name_1': 2, 'col_name_2': 3},
        {'col_name_0': 1, 'col_name_1': 2, 'col_name_2': None}]

        """
        return pli.wrap_expr(
            self._pyexpr.lst_to_struct(n_field_strategy, name_generator, upper_bound)
        )

    def eval(self, expr: pli.Expr, parallel: bool = False) -> pli.Expr:
        """
        Run any polars expression against the lists' elements.

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
        │ 8   ┆ 5   ┆ [2.0, 1.0] │
        │ 3   ┆ 2   ┆ [2.0, 1.0] │
        └─────┴─────┴────────────┘

        """
        return pli.wrap_expr(self._pyexpr.lst_eval(expr._pyexpr, parallel))
