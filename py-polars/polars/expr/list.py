from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, Sequence

import polars._reexport as pl
from polars import functions as F
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import (
    deprecate_renamed_function,
    deprecate_renamed_parameter,
)

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
    """A namespace for :class:`List` expressions."""

    _accessor = "list"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def __getitem__(self, item: int) -> Expr:
        return self.get(item)

    def all(self) -> Expr:
        """
        Evaluate whether all :class:`Boolean` values in each list are true.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[True, True], [False, True], [False, False], [None], [], None]}
        ... )
        >>> df.with_columns(all=pl.col("a").list.all())
        shape: (6, 2)
        ┌────────────────┬───────┐
        │ a              ┆ all   │
        │ ---            ┆ ---   │
        │ list[bool]     ┆ bool  │
        ╞════════════════╪═══════╡
        │ [true, true]   ┆ true  │
        │ [false, true]  ┆ false │
        │ [false, false] ┆ false │
        │ [null]         ┆ true  │
        │ []             ┆ true  │
        │ null           ┆ null  │
        └────────────────┴───────┘

        """
        return wrap_expr(self._pyexpr.list_all())

    def any(self) -> Expr:
        """
        Evaluate whether any :class:`Boolean` value in each list is true.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [[True, True], [False, True], [False, False], [None], [], None]}
        ... )
        >>> df.with_columns(any=pl.col("a").list.any())
        shape: (6, 2)
        ┌────────────────┬───────┐
        │ a              ┆ any   │
        │ ---            ┆ ---   │
        │ list[bool]     ┆ bool  │
        ╞════════════════╪═══════╡
        │ [true, true]   ┆ true  │
        │ [false, true]  ┆ true  │
        │ [false, false] ┆ false │
        │ [null]         ┆ false │
        │ []             ┆ false │
        │ null           ┆ null  │
        └────────────────┴───────┘

        """
        return wrap_expr(self._pyexpr.list_any())

    def len(self) -> Expr:
        """
        Get the number of elements in each list, including `null` elements.

        Returns
        -------
        Expr
            A :class:`UInt32` expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, None], [5]]})
        >>> df.with_columns(len=pl.col("a").list.len())
        shape: (2, 2)
        ┌──────────────┬─────┐
        │ a            ┆ len │
        │ ---          ┆ --- │
        │ list[i64]    ┆ u32 │
        ╞══════════════╪═════╡
        │ [1, 2, null] ┆ 3   │
        │ [5]          ┆ 1   │
        └──────────────┴─────┘

        """
        return wrap_expr(self._pyexpr.list_len())

    def drop_nulls(self) -> Expr:
        """
        Remove all `null` values in each list.

        The original order of the remaining list elements is preserved.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[None, 1, None, 2], [None], [3, 4]]})
        >>> df.with_columns(drop_nulls=pl.col("values").list.drop_nulls())
        shape: (3, 2)
        ┌────────────────┬────────────┐
        │ values         ┆ drop_nulls │
        │ ---            ┆ ---        │
        │ list[i64]      ┆ list[i64]  │
        ╞════════════════╪════════════╡
        │ [null, 1, … 2] ┆ [1, 2]     │
        │ [null]         ┆ []         │
        │ [3, 4]         ┆ [3, 4]     │
        └────────────────┴────────────┘

        """
        return wrap_expr(self._pyexpr.list_drop_nulls())

    def sample(
        self,
        n: int | IntoExprColumn | None = None,
        *,
        fraction: float | IntoExprColumn | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Expr:
        """
        Randomly sample elements from each list.

        Parameters
        ----------
        n
            The number of elements to return. Cannot be used with `fraction`. Defaults
            to `1` if `fraction` is `None`.
        fraction
            The fraction of elements to return. Cannot be used with `n`.
        with_replacement
            Whether to allow elements to be sampled more than once.
        shuffle
            Whether to shuffle the order of the sampled elements. If `shuffle=False`
            (the default), the order will be neither stable nor fully random.
        seed
            The seed for the random number generator. If `seed=None` (the default), a
            random seed is generated anew for each `sample` operation. Set to an integer
            (e.g. `seed=0`) for fully reproducible results.

        Warnings
        --------
        `sample(fraction=1)` returns the expression as-is! To properly shuffle the
        values, add `shuffle=True`.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1, 2, 3], [4, 5]], "n": [2, 1]})
        >>> df.with_columns(sample=pl.col("values").list.sample(n=pl.col("n"), seed=1))
        shape: (2, 3)
        ┌───────────┬─────┬───────────┐
        │ values    ┆ n   ┆ sample    │
        │ ---       ┆ --- ┆ ---       │
        │ list[i64] ┆ i64 ┆ list[i64] │
        ╞═══════════╪═════╪═══════════╡
        │ [1, 2, 3] ┆ 2   ┆ [2, 1]    │
        │ [4, 5]    ┆ 1   ┆ [5]       │
        └───────────┴─────┴───────────┘

        """
        if n is not None and fraction is not None:
            raise ValueError("cannot specify both `n` and `fraction`")

        if fraction is not None:
            fraction = parse_as_expression(fraction)
            return wrap_expr(
                self._pyexpr.list_sample_fraction(
                    fraction, with_replacement, shuffle, seed
                )
            )

        if n is None:
            n = 1
        n = parse_as_expression(n)
        return wrap_expr(self._pyexpr.list_sample_n(n, with_replacement, shuffle, seed))

    def sum(self) -> Expr:
        """
        Get the sum of the elements in each list.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.with_columns(sum=pl.col("values").list.sum())
        shape: (2, 2)
        ┌───────────┬─────┐
        │ values    ┆ sum │
        │ ---       ┆ --- │
        │ list[i64] ┆ i64 │
        ╞═══════════╪═════╡
        │ [1]       ┆ 1   │
        │ [2, 3]    ┆ 5   │
        └───────────┴─────┘

        """
        return wrap_expr(self._pyexpr.list_sum())

    def max(self) -> Expr:
        """
        Get the maximum value of the elements in each list.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.with_columns(max=pl.col("values").list.max())
        shape: (2, 2)
        ┌───────────┬─────┐
        │ values    ┆ max │
        │ ---       ┆ --- │
        │ list[i64] ┆ i64 │
        ╞═══════════╪═════╡
        │ [1]       ┆ 1   │
        │ [2, 3]    ┆ 3   │
        └───────────┴─────┘

        """
        return wrap_expr(self._pyexpr.list_max())

    def min(self) -> Expr:
        """
        Get the minimum value of the elements in each list.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.with_columns(min=pl.col("values").list.min())
        shape: (2, 2)
        ┌───────────┬─────┐
        │ values    ┆ min │
        │ ---       ┆ --- │
        │ list[i64] ┆ i64 │
        ╞═══════════╪═════╡
        │ [1]       ┆ 1   │
        │ [2, 3]    ┆ 2   │
        └───────────┴─────┘

        """
        return wrap_expr(self._pyexpr.list_min())

    def mean(self) -> Expr:
        """
        Get the mean of the elements in each list.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [[1], [2, 3]]})
        >>> df.with_columns(mean=pl.col("values").list.mean())
        shape: (2, 2)
        ┌───────────┬──────┐
        │ values    ┆ mean │
        │ ---       ┆ ---  │
        │ list[i64] ┆ f64  │
        ╞═══════════╪══════╡
        │ [1]       ┆ 1.0  │
        │ [2, 3]    ┆ 2.5  │
        └───────────┴──────┘

        """
        return wrap_expr(self._pyexpr.list_mean())

    def sort(self, *, descending: bool = False) -> Expr:
        """
        Sort each list.

        Parameters
        ----------
        descending
            Whether to sort in descending instead of ascending order.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.with_columns(sort=pl.col("a").list.sort())
        shape: (2, 2)
        ┌───────────┬───────────┐
        │ a         ┆ sort      │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [3, 2, 1] ┆ [1, 2, 3] │
        │ [9, 1, 2] ┆ [1, 2, 9] │
        └───────────┴───────────┘
        >>> df.with_columns(sort=pl.col("a").list.sort(descending=True))
        shape: (2, 2)
        ┌───────────┬───────────┐
        │ a         ┆ sort      │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [3, 2, 1] ┆ [3, 2, 1] │
        │ [9, 1, 2] ┆ [9, 2, 1] │
        └───────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.list_sort(descending))

    def reverse(self) -> Expr:
        """
        Reverse the order of the elements in each list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[3, 2, 1], [9, 1, 2]],
        ...     }
        ... )
        >>> df.with_columns(reverse=pl.col("a").list.reverse())
        shape: (2, 2)
        ┌───────────┬───────────┐
        │ a         ┆ reverse   │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [3, 2, 1] ┆ [1, 2, 3] │
        │ [9, 1, 2] ┆ [2, 1, 9] │
        └───────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.list_reverse())

    def unique(self, *, maintain_order: bool = False) -> Expr:
        """
        Get the unique values that appear in each list, removing duplicates.

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
        ...     }
        ... )
        >>> df.with_columns(unique=pl.col("a").list.unique())
        shape: (1, 2)
        ┌───────────┬───────────┐
        │ a         ┆ unique    │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [1, 1, 2] ┆ [1, 2]    │
        └───────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.list_unique(maintain_order))

    def concat(self, other: list[Expr | str] | Expr | str | Series | list[Any]) -> Expr:
        """
        Concatenate the list elements in two or more :class:`List` columns.

        Parameters
        ----------
        other
            The other column(s).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [["a"], ["x"]],
        ...         "b": [["b", "c"], ["y", "z"]],
        ...     }
        ... )
        >>> df.with_columns(concat=pl.col("a").list.concat("b"))
        shape: (2, 3)
        ┌───────────┬────────────┬─────────────────┐
        │ a         ┆ b          ┆ concat          │
        │ ---       ┆ ---        ┆ ---             │
        │ list[str] ┆ list[str]  ┆ list[str]       │
        ╞═══════════╪════════════╪═════════════════╡
        │ ["a"]     ┆ ["b", "c"] ┆ ["a", "b", "c"] │
        │ ["x"]     ┆ ["y", "z"] ┆ ["x", "y", "z"] │
        └───────────┴────────────┴─────────────────┘

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
        Get a single element from each list by index.

        For instance, `list.get(0)` would return the first item of each list,
        and `list.get(-1)` would return the last item.

        If an index is out of bounds, the resulting element will be `null`.

        Parameters
        ----------
        index
            The index of the element to return from each list.

        See Also
        --------
        gather : Get multiple list elements by index.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 2, 1], [], [1, 2]]})
        >>> df.with_columns(get=pl.col("a").list.get(0))
        shape: (3, 2)
        ┌───────────┬──────┐
        │ a         ┆ get  │
        │ ---       ┆ ---  │
        │ list[i64] ┆ i64  │
        ╞═══════════╪══════╡
        │ [3, 2, 1] ┆ 3    │
        │ []        ┆ null │
        │ [1, 2]    ┆ 1    │
        └───────────┴──────┘

        """
        index = parse_as_expression(index)
        return wrap_expr(self._pyexpr.list_get(index))

    def gather(
        self,
        indices: Expr | Series | list[int] | list[list[int]],
        *,
        null_on_oob: bool = False,
    ) -> Expr:
        """
        Get multiple elements from each list by index.

        The indices may be defined in a single Python list, by lists in another
        :class:`List` column, or by a Python list of Python lists.

        Parameters
        ----------
        indices
            The indices of the elements to return from each list.
        null_on_oob
            Whether to set elements for out-of-bounds indices to `null`,
            rather than raising an error. The latter is much faster.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 2, 1], [], [1, 2, 3, 4, 5]]})
        >>> df.with_columns(gather=pl.col("a").list.gather([0, 4], null_on_oob=True))
        shape: (3, 2)
        ┌─────────────┬──────────────┐
        │ a           ┆ gather       │
        │ ---         ┆ ---          │
        │ list[i64]   ┆ list[i64]    │
        ╞═════════════╪══════════════╡
        │ [3, 2, 1]   ┆ [3, null]    │
        │ []          ┆ [null, null] │
        │ [1, 2, … 5] ┆ [1, 5]       │
        └─────────────┴──────────────┘
        """
        if isinstance(indices, list):
            indices = pl.Series(indices)
        indices = parse_as_expression(indices)
        return wrap_expr(self._pyexpr.list_gather(indices, null_on_oob))

    def first(self) -> Expr:
        """
        Get the first element of each list.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 2, 1], [], [1, 2]]})
        >>> df.with_columns(first=pl.col("a").list.first())
        shape: (3, 2)
        ┌───────────┬───────┐
        │ a         ┆ first │
        │ ---       ┆ ---   │
        │ list[i64] ┆ i64   │
        ╞═══════════╪═══════╡
        │ [3, 2, 1] ┆ 3     │
        │ []        ┆ null  │
        │ [1, 2]    ┆ 1     │
        └───────────┴───────┘

        """
        return self.get(0)

    def last(self) -> Expr:
        """
        Get the last element of each list.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 2, 1], [], [1, 2]]})
        >>> df.with_columns(last=pl.col("a").list.last())
        shape: (3, 2)
        ┌───────────┬──────┐
        │ a         ┆ last │
        │ ---       ┆ ---  │
        │ list[i64] ┆ i64  │
        ╞═══════════╪══════╡
        │ [3, 2, 1] ┆ 1    │
        │ []        ┆ null │
        │ [1, 2]    ┆ 2    │
        └───────────┴──────┘

        """
        return self.get(-1)

    def contains(
        self, item: float | str | bool | int | date | datetime | time | Expr
    ) -> Expr:
        """
        Check if each list contains the given item.

        Parameters
        ----------
        item
            The item that will be checked for membership.

        Returns
        -------
        Expr
            A :class:`Boolean` expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[3, 2, 1], [], [1, 2]]})
        >>> df.with_columns(contains=pl.col("a").list.contains(1))
        shape: (3, 2)
        ┌───────────┬──────────┐
        │ a         ┆ contains │
        │ ---       ┆ ---      │
        │ list[i64] ┆ bool     │
        ╞═══════════╪══════════╡
        │ [3, 2, 1] ┆ true     │
        │ []        ┆ false    │
        │ [1, 2]    ┆ true     │
        └───────────┴──────────┘

        """
        item = parse_as_expression(item, str_as_lit=True)
        return wrap_expr(self._pyexpr.list_contains(item))

    def join(self, separator: IntoExpr) -> Expr:
        """
        Join all string items in a list and place a separator between them.

        Raises an error if the inner dtype of the :class:`List` column is not
        :class:`String`.

        Parameters
        ----------
        separator
            A string to separate the items with.

        Returns
        -------
        Expr
            A :class:`String` expression.

        Examples
        --------
        >>> df = pl.DataFrame({"s": [["a", "b", "c"], ["x", "y"]]})
        >>> df.with_columns(join=pl.col("s").list.join(" "))
        shape: (2, 2)
        ┌─────────────────┬───────┐
        │ s               ┆ join  │
        │ ---             ┆ ---   │
        │ list[str]       ┆ str   │
        ╞═════════════════╪═══════╡
        │ ["a", "b", "c"] ┆ a b c │
        │ ["x", "y"]      ┆ x y   │
        └─────────────────┴───────┘

        >>> df = pl.DataFrame(
        ...     {"s": [["a", "b", "c"], ["x", "y"]], "separator": ["*", "_"]}
        ... )
        >>> df.with_columns(join=pl.col("s").list.join(pl.col("separator")))
        shape: (2, 3)
        ┌─────────────────┬───────────┬───────┐
        │ s               ┆ separator ┆ join  │
        │ ---             ┆ ---       ┆ ---   │
        │ list[str]       ┆ str       ┆ str   │
        ╞═════════════════╪═══════════╪═══════╡
        │ ["a", "b", "c"] ┆ *         ┆ a*b*c │
        │ ["x", "y"]      ┆ _         ┆ x_y   │
        └─────────────────┴───────────┴───────┘

        """
        separator = parse_as_expression(separator, str_as_lit=True)
        return wrap_expr(self._pyexpr.list_join(separator))

    def arg_min(self) -> Expr:
        """
        Get the index of the minimum value in each list.

        Returns
        -------
        Expr
            A :class:`UInt32` or :class:`UInt64` expression (depending on whether polars
            is compiled in `bigidx` mode).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2], [2, 1]],
        ...     }
        ... )
        >>> df.with_columns(arg_min=pl.col("a").list.arg_min())
        shape: (2, 2)
        ┌───────────┬─────────┐
        │ a         ┆ arg_min │
        │ ---       ┆ ---     │
        │ list[i64] ┆ u32     │
        ╞═══════════╪═════════╡
        │ [1, 2]    ┆ 0       │
        │ [2, 1]    ┆ 1       │
        └───────────┴─────────┘

        """
        return wrap_expr(self._pyexpr.list_arg_min())

    def arg_max(self) -> Expr:
        """
        Get the index of the maximum value in each list.

        Returns
        -------
        Expr
            A :class:`UInt32` or :class:`UInt64` expression (depending on whether polars
            is compiled in `bigidx` mode).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2], [2, 1]],
        ...     }
        ... )
        >>> df.with_columns(arg_max=pl.col("a").list.arg_max())
        shape: (2, 2)
        ┌───────────┬─────────┐
        │ a         ┆ arg_max │
        │ ---       ┆ ---     │
        │ list[i64] ┆ u32     │
        ╞═══════════╪═════════╡
        │ [1, 2]    ┆ 1       │
        │ [2, 1]    ┆ 0       │
        └───────────┴─────────┘

        """
        return wrap_expr(self._pyexpr.list_arg_max())

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Expr:
        """
        Get the first discrete difference between shifted elements of each list.

        Parameters
        ----------
        n
            The number of elements to shift by when calculating the difference.
        null_behavior : {'ignore', 'drop'}
            How to handle `null` values.

        Examples
        --------
        >>> df = pl.DataFrame({"n": [[1, 2, 3, 4], [10, 2, 1]]})
        >>> df.with_columns(diff=pl.col("n").list.diff())
        shape: (2, 2)
        ┌─────────────┬────────────────┐
        │ n           ┆ diff           │
        │ ---         ┆ ---            │
        │ list[i64]   ┆ list[i64]      │
        ╞═════════════╪════════════════╡
        │ [1, 2, … 4] ┆ [null, 1, … 1] │
        │ [10, 2, 1]  ┆ [null, -8, -1] │
        └─────────────┴────────────────┘

        >>> df.with_columns(diff=pl.col("n").list.diff(n=2))
        shape: (2, 2)
        ┌─────────────┬───────────────────┐
        │ n           ┆ diff              │
        │ ---         ┆ ---               │
        │ list[i64]   ┆ list[i64]         │
        ╞═════════════╪═══════════════════╡
        │ [1, 2, … 4] ┆ [null, null, … 2] │
        │ [10, 2, 1]  ┆ [null, null, -9]  │
        └─────────────┴───────────────────┘

        >>> df.with_columns(diff=pl.col("n").list.diff(n=2, null_behavior="drop"))
        shape: (2, 2)
        ┌─────────────┬───────────┐
        │ n           ┆ diff      │
        │ ---         ┆ ---       │
        │ list[i64]   ┆ list[i64] │
        ╞═════════════╪═══════════╡
        │ [1, 2, … 4] ┆ [2, 2]    │
        │ [10, 2, 1]  ┆ [-9]      │
        └─────────────┴───────────┘

        """
        return wrap_expr(self._pyexpr.list_diff(n, null_behavior))

    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift(self, n: int | IntoExprColumn = 1) -> Expr:
        """
        Shift list elements by the given number of indices.

        Parameters
        ----------
        n
            The number of indices to shift forward by. If negative, elements are shifted
            backward instead.

        Notes
        -----
        This method is similar to the `LAG` operation in SQL when the value for `n`
        is positive. With a negative value for `n`, it is similar to `LEAD`.

        Examples
        --------
        By default, list elements are shifted forward by one index:

        >>> df = pl.DataFrame({"a": [[1, 2, 3], [4, 5]]})
        >>> df.with_columns(shift=pl.col("a").list.shift())
        shape: (2, 2)
        ┌───────────┬──────────────┐
        │ a         ┆ shift        │
        │ ---       ┆ ---          │
        │ list[i64] ┆ list[i64]    │
        ╞═══════════╪══════════════╡
        │ [1, 2, 3] ┆ [null, 1, 2] │
        │ [4, 5]    ┆ [null, 4]    │
        └───────────┴──────────────┘

        Pass a negative value to shift backwards instead:

        >>> df.with_columns(shift=pl.col("a").list.shift(-2))
        shape: (2, 2)
        ┌───────────┬─────────────────┐
        │ a         ┆ shift           │
        │ ---       ┆ ---             │
        │ list[i64] ┆ list[i64]       │
        ╞═══════════╪═════════════════╡
        │ [1, 2, 3] ┆ [3, null, null] │
        │ [4, 5]    ┆ [null, null]    │
        └───────────┴─────────────────┘

        """
        n = parse_as_expression(n)
        return wrap_expr(self._pyexpr.list_shift(n))

    def slice(
        self, offset: int | str | Expr, length: int | str | Expr | None = None
    ) -> Expr:
        """
        Get a contiguous set of elements from each list.

        Parameters
        ----------
        offset
            The start index. Negative indexing is supported.
        length
            The length of the slice. If `length=None`, all elements starting from the
            `offset` will be selected.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4], [10, 2, 1]]})
        >>> df.with_columns(slice=pl.col("a").list.slice(1, 2))
        shape: (2, 2)
        ┌─────────────┬───────────┐
        │ a           ┆ slice     │
        │ ---         ┆ ---       │
        │ list[i64]   ┆ list[i64] │
        ╞═════════════╪═══════════╡
        │ [1, 2, … 4] ┆ [2, 3]    │
        │ [10, 2, 1]  ┆ [2, 1]    │
        └─────────────┴───────────┘

        """
        offset = parse_as_expression(offset)
        length = parse_as_expression(length)
        return wrap_expr(self._pyexpr.list_slice(offset, length))

    def head(self, n: int | str | Expr = 5) -> Expr:
        """
        Get the first `n` elements of each list.

        Parameters
        ----------
        n
            The number of elements to return. Negative values are not supported.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4], [10, 2, 1]]})
        >>> df.with_columns(head=pl.col("a").list.head(2))
        shape: (2, 2)
        ┌─────────────┬───────────┐
        │ a           ┆ head      │
        │ ---         ┆ ---       │
        │ list[i64]   ┆ list[i64] │
        ╞═════════════╪═══════════╡
        │ [1, 2, … 4] ┆ [1, 2]    │
        │ [10, 2, 1]  ┆ [10, 2]   │
        └─────────────┴───────────┘

        """
        return self.slice(0, n)

    def tail(self, n: int | str | Expr = 5) -> Expr:
        """
        Get the last `n` elements of each list.

        Parameters
        ----------
        n
            The number of elements to return. Negative values are not supported.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4], [10, 2, 1]]})
        >>> df.with_columns(tail=pl.col("a").list.tail(2))
        shape: (2, 2)
        ┌─────────────┬───────────┐
        │ a           ┆ tail      │
        │ ---         ┆ ---       │
        │ list[i64]   ┆ list[i64] │
        ╞═════════════╪═══════════╡
        │ [1, 2, … 4] ┆ [3, 4]    │
        │ [10, 2, 1]  ┆ [2, 1]    │
        └─────────────┴───────────┘

        """
        n = parse_as_expression(n)
        return wrap_expr(self._pyexpr.list_tail(n))

    def explode(self) -> Expr:
        """
        Put every element of every list on its own row.

        Returns
        -------
        Expr
            An expression with the same data type as the inner data type of the list
            elements.

        See Also
        --------
        Expr.explode : Explode a :class:`List` column.
        Expr.str.explode : Explode a :class:`String` column.
        ExprNameSpace.reshape: Reshape each column to a flat `Series` or a :class:`List`
                               `Series`.

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
        Count the number of occurrences of `element` in each list.

        Parameters
        ----------
        element
            An expression that evaluates to a single, scalar value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [[0], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]})
        >>> df.with_columns(number_of_twos=pl.col("a").list.count_matches(2))
        shape: (5, 2)
        ┌─────────────┬────────────────┐
        │ a           ┆ number_of_twos │
        │ ---         ┆ ---            │
        │ list[i64]   ┆ u32            │
        ╞═════════════╪════════════════╡
        │ [0]         ┆ 0              │
        │ [1]         ┆ 0              │
        │ [1, 2, … 2] ┆ 2              │
        │ [1, 2, 1]   ┆ 1              │
        │ [4, 4]      ┆ 0              │
        └─────────────┴────────────────┘

        """
        element = parse_as_expression(element, str_as_lit=True)
        return wrap_expr(self._pyexpr.list_count_matches(element))

    def to_array(self, width: int) -> Expr:
        """
        Convert :class:`List` columns to :class:`Array` columns.

        Parameters
        ----------
        width
            The widths of the resulting :class:`Array` columns.

        Returns
        -------
        Expr
            An :class:`Array` expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [3, 4]]},
        ...     schema={"a": pl.List(pl.Int8)},
        ... )
        >>> df.with_columns(array=pl.col("a").list.to_array(2))
        shape: (2, 2)
        ┌──────────┬──────────────┐
        │ a        ┆ array        │
        │ ---      ┆ ---          │
        │ list[i8] ┆ array[i8, 2] │
        ╞══════════╪══════════════╡
        │ [1, 2]   ┆ [1, 2]       │
        │ [3, 4]   ┆ [3, 4]       │
        └──────────┴──────────────┘

        """
        return wrap_expr(self._pyexpr.list_to_array(width))

    def to_struct(
        self,
        n_field_strategy: ToStructStrategy = "first_non_null",
        fields: Sequence[str] | Callable[[int], str] | None = None,
        upper_bound: int = 0,
    ) -> Expr:
        """
        Convert :class:`List` columns to :class:`Struct` columns.

        Parameters
        ----------
        n_field_strategy : {'first_non_null', 'max_width'}
            Whether to set the number of :class:`Struct` fields to:

            * `"first_non_null"`: the length of the first non zero-length list.
            * `"max_width"`: the maximum length of all lists.
        fields
            If the name and number of the desired fields is known in advance
            a list of field names can be given, which will be assigned by index.
            Otherwise, to dynamically assign field names, a custom function can be
            used; if neither are set, fields will be `field_0, field_1 .. field_n`.
        upper_bound
            A polars `LazyFrame` needs to know the schema at all times, so the
            caller must provide an upper bound of the number of :class:`Struct` fields
            that will be created; if set incorrectly, subsequent operations may fail.
            (For example, an `all().sum()` expression will look in the current
            schema to determine which columns to select).

            When operating on a `DataFrame`, the schema does not need to be
            tracked or pre-determined, as the result will be eagerly evaluated,
            so you can leave this parameter unset.

        Notes
        -----
        For performance reasons, the length of the first non-`null` list is used to
        determine the number of output fields. If the list can be of different lengths,
        then `n_field_strategy="max_width"` must be used to obtain the expected result.

        Examples
        --------
        Convert :class:`List` to :class:`Struct` with default field name assignment:

        >>> df = pl.DataFrame({"n": [[0, 1], [0, 1, 2]]})
        >>> df.with_columns(
        ...     struct=pl.col("n").list.to_struct()
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌───────────┬───────────┐
        │ n         ┆ struct    │
        │ ---       ┆ ---       │
        │ list[i64] ┆ struct[2] │ # <- struct with 2 fields
        ╞═══════════╪═══════════╡
        │ [0, 1]    ┆ {0,1}     │ # OK
        │ [0, 1, 2] ┆ {0,1}     │ # NOT OK - last value missing
        └───────────┴───────────┘

        As the shorter list comes first, we must use the `max_width`
        strategy to force a search for the longest.

        >>> df.with_columns(
        ...     struct=pl.col("n").list.to_struct(n_field_strategy="max_width")
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌───────────┬────────────┐
        │ n         ┆ struct     │
        │ ---       ┆ ---        │
        │ list[i64] ┆ struct[3]  │ # <- struct with 3 fields
        ╞═══════════╪════════════╡
        │ [0, 1]    ┆ {0,1,null} │ # OK
        │ [0, 1, 2] ┆ {0,1,2}    │ # OK
        └───────────┴────────────┘

        Convert :class:`List` to :class:`Struct` with field name assignment by
        function/index:

        >>> df = pl.DataFrame({"n": [[0, 1], [2, 3]]})
        >>> df.select(pl.col("n").list.to_struct(fields=lambda idx: f"n{idx}")).rows(
        ...     named=True
        ... )
        [{'n': {'n0': 0, 'n1': 1}}, {'n': {'n0': 2, 'n1': 3}}]

        Convert :class:`List` to :class:`Struct` with field name assignment by index
        from a list of names:

        >>> df.select(pl.col("n").list.to_struct(fields=["one", "two"])).rows(
        ...     named=True
        ... )
        [{'n': {'one': 0, 'two': 1}}, {'n': {'one': 2, 'two': 3}}]

        """
        if isinstance(fields, Sequence):
            field_names = list(fields)
            pyexpr = self._pyexpr.list_to_struct(n_field_strategy, None, upper_bound)
            return wrap_expr(pyexpr).struct.rename_fields(field_names)
        else:
            pyexpr = self._pyexpr.list_to_struct(n_field_strategy, fields, upper_bound)
            return wrap_expr(pyexpr)

    def eval(self, expr: Expr, *, parallel: bool = False) -> Expr:
        """
        Evaluate any polars expression across each list's elements.

        Use :func:`polars.element()` to refer to the list element, similar to how
        you usually would use `pl.col()` to refer to a column in an expression.

        Parameters
        ----------
        expr
            The expression to evaluate.
        parallel
            Whether to execute the computation in parallel.

            .. note::
                This option should likely not be enabled in an aggregation context,
                as the computation is already parallelized per group.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
        >>> df.with_columns(
        ...     rank=pl.concat_list("a", "b").list.eval(pl.element().rank())
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
        Compute set unions between elements in two :class:`List` columns.

        Parameters
        ----------
        other
            The other column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(
        ...     union=pl.col("a").list.set_union("b")
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

        """  # .
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "union"))

    def set_difference(self, other: IntoExpr) -> Expr:
        """
        Compute set differences between elements in two :class:`List` columns.

        Parameters
        ----------
        other
            The other column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(difference=pl.col("a").list.set_difference("b"))
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
        polars.Expr.list.diff: Get the first discrete difference between shifted
                               elements of each list.

        """  # .
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "difference"))

    def set_intersection(self, other: IntoExpr) -> Expr:
        """
        Compute set intersections between elements in two :class:`List` columns.

        Parameters
        ----------
        other
            The other column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(intersection=pl.col("a").list.set_intersection("b"))
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

        """  # .
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "intersection"))

    def set_symmetric_difference(self, other: IntoExpr) -> Expr:
        """
        Compute set symmetric differences between elements in two :class:`List` columns.

        Parameters
        ----------
        other
            The other column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
        ...         "b": [[2, 3, 4], [3], [3, 4, None], [6, 8]],
        ...     }
        ... )
        >>> df.with_columns(sdiff=pl.col("b").list.set_symmetric_difference("a"))
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
        """  # .
        other = parse_as_expression(other, str_as_lit=False)
        return wrap_expr(self._pyexpr.list_set_operation(other, "symmetric_difference"))

    @deprecate_renamed_function("count_matches", version="0.19.3")
    def count_match(self, element: IntoExpr) -> Expr:
        """
        Count how often the value produced by `element` occurs.

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

    @deprecate_renamed_function("gather", version="0.19.14")
    @deprecate_renamed_parameter("index", "indices", version="0.19.14")
    def take(
        self,
        indices: Expr | Series | list[int] | list[list[int]],
        *,
        null_on_oob: bool = False,
    ) -> Expr:
        """
        Take sublists by multiple indices.

        The indices may be defined in a single column, or by sublists in another
        column of dtype :class:`List`.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`gather`.

        Parameters
        ----------
        indices
            Indices to return per sublist
        null_on_oob
            Behavior if an index is out of bounds:
            True -> set as null
            False -> raise an error
            Note that defaulting to raising an error is much cheaper
        """
        return self.gather(indices)
