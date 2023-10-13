from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, Sequence

import polars._reexport as pl
from polars import functions as F
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import deprecate_renamed_function

if TYPE_CHECKING:
    from datetime import date, datetime, time

    from polars import Expr, Series
    from polars.type_aliases import (
        IntoExpr,
        IntoExprColumn,
        NullBehavior,
        ToStructStrategy,
    )


class ExprListNameSpace:
    """Namespace for list related expressions."""

    _accessor = "list"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def __getitem__(self, item: int) -> Expr:
        return self.get(item)

    def all(self) -> Expr:
        """
        Evaluate whether all boolean values in a list are true.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[True, True], [False, True], [False, False], [None], [], None]}
        ... )
        >>> df.select(pl.col("a").list.all())
        shape: (6, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ false │
        │ false │
        │ true  │
        │ true  │
        │ null  │
        └───────┘

        """
        return wrap_expr(self._pyexpr.list_all())

    def any(self) -> Expr:
        """
        Evaluate whether any boolean value in a list is true.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[True, True], [False, True], [False, False], [None], [], None]}
        ... )
        >>> df.select(pl.col("a").list.any())
        shape: (6, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ true  │
        │ false │
        │ false │
        │ false │
        │ null  │
        └───────┘

        """
        return wrap_expr(self._pyexpr.list_any())

    def len(self) -> Expr:
        """
        Return the number of elements in each list.

        Null values are treated like regular elements in this context.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, None], [5]]})
        >>> df.select(pl.col("a").list.len())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 3   │
        │ 1   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.list_len())

    def drop_nulls(self) -> Expr:
        """
        Drop all null values in the list.

        The original order of the remaining elements is preserved.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[None, 1, None, 2], [None], [3, 4]]})
        >>> df.select(pl.col("values").list.drop_nulls())
        shape: (3, 1)
        ┌───────────┐
        │ values    │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2]    │
        │ []        │
        │ [3, 4]    │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.list_drop_nulls())

    def sum(self) -> Expr:
        """
        Sum all the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").list.sum())
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
        return wrap_expr(self._pyexpr.list_sum())

    def max(self) -> Expr:
        """
        Compute the max value of the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").list.max())
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
        return wrap_expr(self._pyexpr.list_max())

    def min(self) -> Expr:
        """
        Compute the min value of the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").list.min())
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
        return wrap_expr(self._pyexpr.list_min())

    def mean(self) -> Expr:
        """
        Compute the mean value of the lists in the array.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.select(pl.col("values").list.mean())
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
        return wrap_expr(self._pyexpr.list_mean())

    def sort(self, *, descending: bool = False) -> Expr:
        """
        Sort the lists in this column.

        Parameters
        ----------
        descending
            Sort in descending order.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").list.sort())
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        │ [1, 2, 9] │
        └───────────┘
        >>> df.select(pl.col("a").list.sort(descending=True))
        shape: (2, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [3, 2, 1] │
        │ [9, 2, 1] │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.list_sort(descending))

    def reverse(self) -> Expr:
        """
        Reverse the arrays in the list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").list.reverse())
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
        return wrap_expr(self._pyexpr.list_reverse())

    def unique(self, *, maintain_order: bool = False) -> Expr:
        """
        Get the unique/distinct values in the list.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 1, 2]],
        ...     }
        ... )
        >>> df.select(pl.col("a").list.unique())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2]    │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.list_unique(maintain_order))

    def concat(self, other: list[Expr | str] | Expr | str | Series | list[Any]) -> Expr:
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
        >>> df.select(pl.col("a").list.concat("b"))
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
            not isinstance(other[0], (pl.Expr, str, pl.Series))
        ):
            return self.concat(pl.Series([other]))

        other_list: list[Expr | str | Series]
        other_list = [other] if not isinstance(other, list) else copy.copy(other)  # type: ignore[arg-type]

        other_list.insert(0, wrap_expr(self._pyexpr))
        return F.concat_list(other_list)

    def get(self, index: int | Expr | str) -> Expr:
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
        >>> df.select(pl.col("foo").list.get(0))
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
        index = parse_as_expression(index)
        return wrap_expr(self._pyexpr.list_get(index))

    def take(
        self,
        index: Expr | Series | list[int] | list[list[int]],
        *,
        null_on_oob: bool = False,
    ) -> Expr:
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
            index = pl.Series(index)
        index = parse_as_expression(index)
        return wrap_expr(self._pyexpr.list_take(index, null_on_oob))

    def first(self) -> Expr:
        """
        Get the first value of the sublists.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").list.first())
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

    def last(self) -> Expr:
        """
        Get the last value of the sublists.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").list.last())
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
        self, item: float | str | bool | int | date | datetime | time | Expr
    ) -> Expr:
        """
        Check if sublists contain the given item.

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
        >>> df = pl.DataFrame({"foo": [[3, 2, 1], [], [1, 2]]})
        >>> df.select(pl.col("foo").list.contains(1))
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
        item = parse_as_expression(item, str_as_lit=True)
        return wrap_expr(self._pyexpr.list_contains(item))

    def join(self, separator: IntoExpr) -> Expr:
        """
        Join all string items in a sublist and place a separator between them.

        This errors if inner type of list `!= Utf8`.

        Parameters
        ----------
        separator
            string to separate the items with

        Returns
        -------
        Expr
            Expression of data type :class:`Utf8`.

        Examples
        --------
        >>> df = pl.DataFrame({"s": [["a", "b", "c"], ["x", "y"]]})
        >>> df.select(pl.col("s").list.join(" "))
        shape: (2, 1)
        ┌───────┐
        │ s     │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a b c │
        │ x y   │
        └───────┘

        >>> df = pl.DataFrame(
        ...     {"s": [["a", "b", "c"], ["x", "y"]], "separator": ["*", "_"]}
        ... )
        >>> df.select(pl.col("s").list.join(pl.col("separator")))
        shape: (2, 1)
        ┌───────┐
        │ s     │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a*b*c │
        │ x_y   │
        └───────┘
        """
        separator = parse_as_expression(separator, str_as_lit=True)
        return wrap_expr(self._pyexpr.list_join(separator))

    def arg_min(self) -> Expr:
        """
        Retrieve the index of the minimal value in every sublist.

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
        ...     }
        ... )
        >>> df.select(pl.col("a").list.arg_min())
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
        return wrap_expr(self._pyexpr.list_arg_min())

    def arg_max(self) -> Expr:
        """
        Retrieve the index of the maximum value in every sublist.

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
        ...     }
        ... )
        >>> df.select(pl.col("a").list.arg_max())
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
        return wrap_expr(self._pyexpr.list_arg_max())

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Expr:
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
        >>> df = pl.DataFrame({"n": [[1, 2, 3, 4], [10, 2, 1]]})
        >>> df.select(pl.col("n").list.diff())
        shape: (2, 1)
        ┌────────────────┐
        │ n              │
        │ ---            │
        │ list[i64]      │
        ╞════════════════╡
        │ [null, 1, … 1] │
        │ [null, -8, -1] │
        └────────────────┘

        >>> df.select(pl.col("n").list.diff(n=2))
        shape: (2, 1)
        ┌───────────────────┐
        │ n                 │
        │ ---               │
        │ list[i64]         │
        ╞═══════════════════╡
        │ [null, null, … 2] │
        │ [null, null, -9]  │
        └───────────────────┘

        >>> df.select(pl.col("n").list.diff(n=2, null_behavior="drop"))
        shape: (2, 1)
        ┌───────────┐
        │ n         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [2, 2]    │
        │ [-9]      │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.list_diff(n, null_behavior))

    def shift(self, periods: int | IntoExprColumn = 1) -> Expr:
        """
        Shift values by the given period.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.shift()
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [null, 1, … 3]
            [null, 10, 2]
        ]

        """
        periods = parse_as_expression(periods)
        return wrap_expr(self._pyexpr.list_shift(periods))

    def slice(
        self, offset: int | str | Expr, length: int | str | Expr | None = None
    ) -> Expr:
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
        >>> s.list.slice(1, 2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [2, 3]
            [2, 1]
        ]

        """
        offset = parse_as_expression(offset)
        length = parse_as_expression(length)
        return wrap_expr(self._pyexpr.list_slice(offset, length))

    def head(self, n: int | str | Expr = 5) -> Expr:
        """
        Slice the first `n` values of every sublist.

        Parameters
        ----------
        n
            Number of values to return for each sublist.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.head(2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [1, 2]
            [10, 2]
        ]

        """
        return self.slice(0, n)

    def tail(self, n: int | str | Expr = 5) -> Expr:
        """
        Slice the last `n` values of every sublist.

        Parameters
        ----------
        n
            Number of values to return for each sublist.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.tail(2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [3, 4]
            [2, 1]
        ]

        """
        n = parse_as_expression(n)
        return wrap_expr(self._pyexpr.list_tail(n))

    def explode(self) -> Expr:
        """
        Returns a column with a separate row for every list element.

        Returns
        -------
        Expr
            Expression with the data type of the list elements.

        See Also
        --------
        ExprNameSpace.reshape: Reshape this Expr to a flat Series or a Series of Lists.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]})
        >>> df.select(pl.col("a").list.explode())
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

    def count_matches(self, element: IntoExpr) -> Expr:
        """
        Count how often the value produced by ``element`` occurs.

        Parameters
        ----------
        element
            An expression that produces a single value

        Examples
        --------
        >>> df = pl.DataFrame({"listcol": [[0], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]})
        >>> df.select(pl.col("listcol").list.count_matches(2).alias("number_of_twos"))
        shape: (5, 1)
        ┌────────────────┐
        │ number_of_twos │
        │ ---            │
        │ u32            │
        ╞════════════════╡
        │ 0              │
        │ 0              │
        │ 2              │
        │ 1              │
        │ 0              │
        └────────────────┘

        """
        element = parse_as_expression(element, str_as_lit=True)
        return wrap_expr(self._pyexpr.list_count_matches(element))

    def to_struct(
        self,
        n_field_strategy: ToStructStrategy = "first_non_null",
        fields: Sequence[str] | Callable[[int], str] | None = None,
        upper_bound: int = 0,
    ) -> Expr:
        """
        Convert the series of type ``List`` to a series of type ``Struct``.

        Parameters
        ----------
        n_field_strategy : {'first_non_null', 'max_width'}
            Strategy to determine the number of fields of the struct.

            * "first_non_null": set number of fields equal to the length of the
              first non zero-length sublist.
            * "max_width": set number of fields as max length of all sublists.
        fields
            If the name and number of the desired fields is known in advance
            a list of field names can be given, which will be assigned by index.
            Otherwise, to dynamically assign field names, a custom function can be
            used; if neither are set, fields will be `field_0, field_1 .. field_n`.
        upper_bound
            A polars ``LazyFrame`` needs to know the schema at all times, so the
            caller must provide an upper bound of the number of struct fields that
            will be created; if set incorrectly, subsequent operations may fail.
            (For example, an ``all().sum()`` expression will look in the current
            schema to determine which columns to select).

            When operating on a ``DataFrame``, the schema does not need to be
            tracked or pre-determined, as the result will be eagerly evaluated,
            so you can leave this parameter unset.

        Examples
        --------
        Convert list to struct with default field name assignment:

        >>> df = pl.DataFrame({"n": [[0, 1, 2], [0, 1]]})
        >>> df.select(pl.col("n").list.to_struct())
        shape: (2, 1)
        ┌────────────┐
        │ n          │
        │ ---        │
        │ struct[3]  │
        ╞════════════╡
        │ {0,1,2}    │
        │ {0,1,null} │
        └────────────┘

        Convert list to struct with field name assignment by function/index:

        >>> df.select(pl.col("n").list.to_struct(fields=lambda idx: f"n{idx}")).rows(
        ...     named=True
        ... )
        [{'n': {'n0': 0, 'n1': 1, 'n2': 2}}, {'n': {'n0': 0, 'n1': 1, 'n2': None}}]

        Convert list to struct with field name assignment by index from a list of names:

        >>> df.select(pl.col("n").list.to_struct(fields=["one", "two", "three"])).rows(
        ...     named=True
        ... )
        [{'n': {'one': 0, 'two': 1, 'three': 2}},
        {'n': {'one': 0, 'two': 1, 'three': None}}]

        """
        if isinstance(fields, Sequence):
            field_names = list(fields)

            def fields(idx: int) -> str:
                return field_names[idx]

        return wrap_expr(
            self._pyexpr.list_to_struct(n_field_strategy, fields, upper_bound)
        )

    def eval(self, expr: Expr, *, parallel: bool = False) -> Expr:
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

            This likely should not be used in the group by context, because we already
            parallel execution per group

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
        >>> df.with_columns(
        ...     pl.concat_list(["a", "b"]).list.eval(pl.element().rank()).alias("rank")
        ... )
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ a   ┆ b   ┆ rank       │
        │ --- ┆ --- ┆ ---        │
        │ i64 ┆ i64 ┆ list[f64]  │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 4   ┆ [1.0, 2.0] │
        │ 8   ┆ 5   ┆ [2.0, 1.0] │
        │ 3   ┆ 2   ┆ [2.0, 1.0] │
        └─────┴─────┴────────────┘

        """
        return wrap_expr(self._pyexpr.list_eval(expr._pyexpr, parallel))

    def set_union(self, other: IntoExpr) -> Expr:
        """
        Compute the SET UNION between the elements in this list and the elements of ``other``.

        Parameters
        ----------
        other
            Right hand side of the set operation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("a").list.set_union("b").alias("union")
        ... )  # doctest: +IGNORE_RESULT
        shape: (4, 3)
        ┌───────────┬──────────────┬───────────────┐
        │ a         ┆ b            ┆ union         │
        │ ---       ┆ ---          ┆ ---           │
        │ list[i64] ┆ list[i64]    ┆ list[i64]     │
        ╞═══════════╪══════════════╪═══════════════╡
        │ [1, 2, 3] ┆ [2, 3, 4]    ┆ [1, 2, 3, 4]  │
        │ []        ┆ [3]          ┆ [3]           │
        │ [null, 3] ┆ [3, 4, null] ┆ [null, 3, 4]  │
        │ [5, 6, 7] ┆ [6, 8]       ┆ [5, 6, 7, 8]  │
        └───────────┴──────────────┴───────────────┘

        """  # noqa: W505.
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "union"))

    def set_difference(self, other: IntoExpr) -> Expr:
        """
        Compute the SET DIFFERENCE between the elements in this list and the elements of ``other``.

        Parameters
        ----------
        other
            Right hand side of the set operation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(pl.col("a").list.set_difference("b").alias("difference"))
        shape: (4, 3)
        ┌───────────┬──────────────┬────────────┐
        │ a         ┆ b            ┆ difference │
        │ ---       ┆ ---          ┆ ---        │
        │ list[i64] ┆ list[i64]    ┆ list[i64]  │
        ╞═══════════╪══════════════╪════════════╡
        │ [1, 2, 3] ┆ [2, 3, 4]    ┆ [1]        │
        │ []        ┆ [3]          ┆ []         │
        │ [null, 3] ┆ [3, 4, null] ┆ []         │
        │ [5, 6, 7] ┆ [6, 8]       ┆ [5, 7]     │
        └───────────┴──────────────┴────────────┘

        See Also
        --------
        polars.Expr.list.diff: Calculates the n-th discrete difference of every sublist.

        """  # noqa: W505.
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "difference"))

    def set_intersection(self, other: IntoExpr) -> Expr:
        """
        Compute the SET INTERSECTION between the elements in this list and the elements of ``other``.

        Parameters
        ----------
        other
            Right hand side of the set operation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("a").list.set_intersection("b").alias("intersection")
        ... )
        shape: (4, 3)
        ┌───────────┬──────────────┬──────────────┐
        │ a         ┆ b            ┆ intersection │
        │ ---       ┆ ---          ┆ ---          │
        │ list[i64] ┆ list[i64]    ┆ list[i64]    │
        ╞═══════════╪══════════════╪══════════════╡
        │ [1, 2, 3] ┆ [2, 3, 4]    ┆ [2, 3]       │
        │ []        ┆ [3]          ┆ []           │
        │ [null, 3] ┆ [3, 4, null] ┆ [null, 3]    │
        │ [5, 6, 7] ┆ [6, 8]       ┆ [6]          │
        └───────────┴──────────────┴──────────────┘

        """  # noqa: W505.
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "intersection"))

    def set_symmetric_difference(self, other: IntoExpr) -> Expr:
        """
        Compute the SET SYMMETRIC DIFFERENCE between the elements in this list and the elements of ``other``.

        Parameters
        ----------
        other
            Right hand side of the set operation.


        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("b").list.set_symmetric_difference("a").alias("sdiff")
        ... )
        shape: (4, 3)
        ┌───────────┬──────────────┬───────────┐
        │ a         ┆ b            ┆ sdiff     │
        │ ---       ┆ ---          ┆ ---       │
        │ list[i64] ┆ list[i64]    ┆ list[i64] │
        ╞═══════════╪══════════════╪═══════════╡
        │ [1, 2, 3] ┆ [2, 3, 4]    ┆ [4, 1]    │
        │ []        ┆ [3]          ┆ [3]       │
        │ [null, 3] ┆ [3, 4, null] ┆ [4]       │
        │ [5, 6, 7] ┆ [6, 8]       ┆ [8, 5, 7] │
        └───────────┴──────────────┴───────────┘
        """  # noqa: W505.
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "symmetric_difference"))

    @deprecate_renamed_function("set_union", version="0.18.10")
    def union(self, other: IntoExpr) -> Expr:
        """
        Compute the SET UNION between the elements in this list and the elements of ``other``.

        .. deprecated:: 0.18.10
            This method has been renamed to ``Expr.list.set_union``.

        """  # noqa: W505
        return self.set_union(other)

    @deprecate_renamed_function("set_difference", version="0.18.10")
    def difference(self, other: IntoExpr) -> Expr:
        """
        Compute the SET DIFFERENCE between the elements in this list and the elements of ``other``.

        .. deprecated:: 0.18.10
            This method has been renamed to ``Expr.list.set_difference``.

        """  # noqa: W505
        return self.set_difference(other)

    @deprecate_renamed_function("set_intersection", version="0.18.10")
    def intersection(self, other: IntoExpr) -> Expr:
        """
        Compute the SET INTERSECTION between the elements in this list and the elements of ``other``.

        .. deprecated:: 0.18.10
            This method has been renamed to ``Expr.list.set_intersection``.

        """  # noqa: W505
        return self.set_intersection(other)

    @deprecate_renamed_function("set_symmetric_difference", version="0.18.10")
    def symmetric_difference(self, other: IntoExpr) -> Expr:
        """
        Compute the SET SYMMETRIC DIFFERENCE between the elements in this list and the elements of ``other``.

        .. deprecated:: 0.18.10
            This method has been renamed to ``Expr.list.set_symmetric_difference``.

        """  # noqa: W505
        return self.set_symmetric_difference(other)

    @deprecate_renamed_function("count_matches", version="0.19.3")
    def count_match(self, element: IntoExpr) -> Expr:
        """
        Count how often the value produced by ``element`` occurs.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`count_matches`.

        Parameters
        ----------
        element
            An expression that produces a single value

        """
        return self.count_matches(element)

    @deprecate_renamed_function("len", version="0.19.8")
    def lengths(self) -> Expr:
        """
        Return the number of elements in each list.

        .. deprecated:: 0.19.8
            This method has been renamed to :func:`len`.

        """
        return self.len()
