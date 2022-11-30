from __future__ import annotations

import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, Sequence, TypeVar

import polars.internals as pli
from polars.internals.dataframe.pivot import PivotOps
from polars.utils import _timedelta_to_pl_duration

if TYPE_CHECKING:
    from polars.datatypes import DataType
    from polars.internals.type_aliases import (
        ClosedWindow,
        RollingInterpolationMethod,
        StartBy,
    )
    from polars.polars import PyDataFrame

# A type variable used to refer to a polars.DataFrame or any subclass of it.
# Used to annotate DataFrame methods which returns the same type as self.
DF = TypeVar("DF", bound="pli.DataFrame")


class GroupBy(Generic[DF]):
    """
    Starts a new GroupBy operation.

    You can also loop over this Object to loop over `DataFrames` with unique groups.

    Examples
    --------
    >>> df = pl.DataFrame({"foo": ["a", "a", "b"], "bar": [1, 2, 3]})
    >>> for group in df.groupby("foo"):
    ...     print(group)
    ... # doctest: +IGNORE_RESULT
    ...
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ a   ┆ 2   │
    └─────┴─────┘
    shape: (1, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ b   ┆ 3   │
    └─────┴─────┘

    """

    _df: PyDataFrame
    _dataframe_class: type[DF]
    by: str | list[str]
    maintain_order: bool

    def __init__(
        self,
        df: PyDataFrame,
        by: str | list[str],
        dataframe_class: type[DF],
        maintain_order: bool = False,
    ):
        """
        Construct class representing a group by operation over the given dataframe.

        Parameters
        ----------
        df
            PyDataFrame to perform operation over.
        by
            Column(s) to group by.
        dataframe_class
            The class used to wrap around the given dataframe. Used to construct new
            dataframes returned from the group by operation.
        maintain_order
            Make sure that the order of the groups remain consistent. This is more
            expensive than a default groupby. Note that this only works in expression
            aggregations.

        """
        self._df = df
        self._dataframe_class = dataframe_class
        self.by = by
        self.maintain_order = maintain_order

    def __iter__(self) -> Iterable[Any]:
        groups_df = self._groups()
        groups = groups_df["groups"]
        df = self._dataframe_class._from_pydf(self._df)
        for i in range(groups_df.height):
            yield df[groups[i]]

    def _select(self, columns: str | list[str]) -> GBSelection[DF]:  # pragma: no cover
        """
        Select the columns that will be aggregated.

        Parameters
        ----------
        columns
            One or multiple columns.

        """
        warnings.warn(
            "accessing GroupBy by index is deprecated, consider using the `.agg`"
            " method",
            DeprecationWarning,
        )
        if isinstance(columns, str):
            columns = [columns]
        return GBSelection(
            self._df,
            self.by,
            columns,
            dataframe_class=self._dataframe_class,
        )

    def _select_all(self) -> GBSelection[DF]:
        """Select all columns for aggregation."""
        return GBSelection(
            self._df,
            self.by,
            None,
            dataframe_class=self._dataframe_class,
        )

    def _groups(self) -> DF:  # pragma: no cover
        """
        Get keys and group indices for each group in the groupby.

        Returns
        -------
        DataFrame
            A DataFrame with:

            - the groupby keys
            - the group indexes aggregated as lists

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, True, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )

        >>> df.groupby("d")._groups().sort(by="d")
        shape: (3, 2)
        ┌────────┬───────────┐
        │ d      ┆ groups    │
        │ ---    ┆ ---       │
        │ str    ┆ list[u32] │
        ╞════════╪═══════════╡
        │ Apple  ┆ [0, 2, 3] │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ Banana ┆ [4, 5]    │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ Orange ┆ [1]       │
        └────────┴───────────┘

        """
        warnings.warn(
            "accessing GroupBy by index is deprecated, consider using the `.agg`"
            " method",
            DeprecationWarning,
        )
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, None, "groups")
        )

    def apply(self, f: Callable[[pli.DataFrame], pli.DataFrame]) -> DF:
        """
        Apply a custom/user-defined function (UDF) over the groups as a sub-DataFrame.

        Implementing logic using a Python function is almost always _significantly_
        slower and more memory intensive than implementing the same logic using
        the native expression API because:

        - The native expression engine runs in Rust; UDFs run in Python.
        - Use of Python UDFs forces the DataFrame to be materialized in memory.
        - Polars-native expressions can be parallelised (UDFs cannot).
        - Polars-native expressions can be logically optimised (UDFs cannot).

        Wherever possible you should strongly prefer the native expression API
        to achieve the best performance.

        Parameters
        ----------
        f
            Custom function.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "id": [0, 1, 2, 3, 4],
        ...         "color": ["red", "green", "green", "red", "red"],
        ...         "shape": ["square", "triangle", "square", "triangle", "square"],
        ...     }
        ... )
        >>> df
        shape: (5, 3)
        ┌─────┬───────┬──────────┐
        │ id  ┆ color ┆ shape    │
        │ --- ┆ ---   ┆ ---      │
        │ i64 ┆ str   ┆ str      │
        ╞═════╪═══════╪══════════╡
        │ 0   ┆ red   ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ green ┆ triangle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ green ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ red   ┆ triangle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ red   ┆ square   │
        └─────┴───────┴──────────┘

        For each color group sample two rows:

        >>> (
        ...     df.groupby("color").apply(lambda group_df: group_df.sample(2))
        ... )  # doctest: +IGNORE_RESULT
        shape: (4, 3)
        ┌─────┬───────┬──────────┐
        │ id  ┆ color ┆ shape    │
        │ --- ┆ ---   ┆ ---      │
        │ i64 ┆ str   ┆ str      │
        ╞═════╪═══════╪══════════╡
        │ 1   ┆ green ┆ triangle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ green ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ red   ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ red   ┆ triangle │
        └─────┴───────┴──────────┘

        It is better to implement this with an expression:

        >>> (
        ...     df.filter(pl.arange(0, pl.count()).shuffle().over("color") < 2)
        ... )  # doctest: +IGNORE_RESULT

        """
        return self._dataframe_class._from_pydf(self._df.groupby_apply(self.by, f))

    def agg(self, aggs: pli.Expr | Sequence[pli.Expr]) -> pli.DataFrame:
        """
        Use multiple aggregations on columns.

        This can be combined with complete lazy API and is considered idiomatic polars.

        Parameters
        ----------
        aggs
            Single / multiple aggregation expression(s).

        Returns
        -------
        Result of groupby split apply operations.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": ["one", "two", "two", "one", "two"], "bar": [5, 3, 2, 4, 1]}
        ... )
        >>> df.groupby("foo", maintain_order=True).agg(
        ...     [
        ...         pl.sum("bar").suffix("_sum"),
        ...         pl.col("bar").sort().tail(2).sum().suffix("_tail_sum"),
        ...     ]
        ... )
        shape: (2, 3)
        ┌─────┬─────────┬──────────────┐
        │ foo ┆ bar_sum ┆ bar_tail_sum │
        │ --- ┆ ---     ┆ ---          │
        │ str ┆ i64     ┆ i64          │
        ╞═════╪═════════╪══════════════╡
        │ one ┆ 9       ┆ 9            │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ two ┆ 6       ┆ 5            │
        └─────┴─────────┴──────────────┘

        """
        df = (
            pli.wrap_df(self._df)
            .lazy()
            .groupby(self.by, maintain_order=self.maintain_order)
            .agg(aggs)
            .collect(no_optimization=True)
        )
        return self._dataframe_class._from_pydf(df._df)

    def head(self, n: int = 5) -> DF:
        """
        Get the first `n` rows of each group.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["c", "c", "a", "c", "a", "b"],
        ...         "nrs": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df
        shape: (6, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 4   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        └─────────┴─────┘
        >>> df.groupby("letters").head(2).sort("letters")
        shape: (5, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        └─────────┴─────┘

        """
        df = (
            pli.wrap_df(self._df)
            .lazy()
            .groupby(self.by, self.maintain_order)
            .head(n)
            .collect(no_optimization=True)
        )
        return self._dataframe_class._from_pydf(df._df)

    def tail(self, n: int = 5) -> DF:
        """
        Get the last `n` rows of each group.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["c", "c", "a", "c", "a", "b"],
        ...         "nrs": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df
        shape: (6, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 4   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        └─────────┴─────┘
        >>> (df.groupby("letters").tail(2).sort("letters"))
        shape: (5, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 4   │
        └─────────┴─────┘

        """
        df = (
            pli.wrap_df(self._df)
            .lazy()
            .groupby(self.by, self.maintain_order)
            .tail(n)
            .collect(no_optimization=True)
        )
        return self._dataframe_class._from_pydf(df._df)

    def pivot(
        self, pivot_column: str | list[str], values_column: str | list[str]
    ) -> PivotOps[DF]:
        """
        Do a pivot operation.

        The pivot operation is based on the group key, a pivot column and an aggregation
        function on the values column.

        Parameters
        ----------
        pivot_column
            Column to pivot.
        values_column
            Column that will be aggregated.

        Notes
        -----
        Polars'/arrow memory is not ideal for transposing operations like pivots.
        If you have a relatively large table, consider using a groupby over a pivot.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["one", "one", "one", "two", "two", "two"],
        ...         "bar": ["A", "B", "C", "A", "B", "C"],
        ...         "baz": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df.groupby("foo", maintain_order=True).pivot(  # doctest: +SKIP
        ...     pivot_column="bar", values_column="baz"
        ... ).first()
        shape: (2, 4)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ A   ┆ B   ┆ C   │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╡
        │ one ┆ 1   ┆ 2   ┆ 3   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ two ┆ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┴─────┘

        """
        if isinstance(pivot_column, str):
            pivot_column = [pivot_column]
        if isinstance(values_column, str):
            values_column = [values_column]
        return PivotOps(
            self._df,
            self.by,
            pivot_column,
            values_column,
            dataframe_class=self._dataframe_class,
        )

    def first(self) -> pli.DataFrame:
        """
        Aggregate the first values in the group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).first()
        shape: (3, 4)
        ┌────────┬─────┬──────┬───────┐
        │ d      ┆ a   ┆ b    ┆ c     │
        │ ---    ┆ --- ┆ ---  ┆ ---   │
        │ str    ┆ i64 ┆ f64  ┆ bool  │
        ╞════════╪═════╪══════╪═══════╡
        │ Apple  ┆ 1   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 4   ┆ 13.0 ┆ false │
        └────────┴─────┴──────┴───────┘

        """
        return self.agg(pli.all().first())

    def last(self) -> pli.DataFrame:
        """
        Aggregate the last values in the group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).last()
        shape: (3, 4)
        ┌────────┬─────┬──────┬───────┐
        │ d      ┆ a   ┆ b    ┆ c     │
        │ ---    ┆ --- ┆ ---  ┆ ---   │
        │ str    ┆ i64 ┆ f64  ┆ bool  │
        ╞════════╪═════╪══════╪═══════╡
        │ Apple  ┆ 3   ┆ 10.0 ┆ false │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 5   ┆ 14.0 ┆ true  │
        └────────┴─────┴──────┴───────┘

        """
        return self.agg(pli.all().last())

    def sum(self) -> pli.DataFrame:
        """
        Reduce the groups to the sum.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).sum()
        shape: (3, 4)
        ┌────────┬─────┬──────┬─────┐
        │ d      ┆ a   ┆ b    ┆ c   │
        │ ---    ┆ --- ┆ ---  ┆ --- │
        │ str    ┆ i64 ┆ f64  ┆ u32 │
        ╞════════╪═════╪══════╪═════╡
        │ Apple  ┆ 6   ┆ 14.5 ┆ 2   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ 1   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Banana ┆ 9   ┆ 27.0 ┆ 1   │
        └────────┴─────┴──────┴─────┘

        """
        return self.agg(pli.all().sum())

    def min(self) -> pli.DataFrame:
        """
        Reduce the groups to the minimal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).min()
        shape: (3, 4)
        ┌────────┬─────┬──────┬───────┐
        │ d      ┆ a   ┆ b    ┆ c     │
        │ ---    ┆ --- ┆ ---  ┆ ---   │
        │ str    ┆ i64 ┆ f64  ┆ bool  │
        ╞════════╪═════╪══════╪═══════╡
        │ Apple  ┆ 1   ┆ 0.5  ┆ false │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 4   ┆ 13.0 ┆ false │
        └────────┴─────┴──────┴───────┘

        """
        return self.agg(pli.all().min())

    def max(self) -> pli.DataFrame:
        """
        Reduce the groups to the maximal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).max()
        shape: (3, 4)
        ┌────────┬─────┬──────┬──────┐
        │ d      ┆ a   ┆ b    ┆ c    │
        │ ---    ┆ --- ┆ ---  ┆ ---  │
        │ str    ┆ i64 ┆ f64  ┆ bool │
        ╞════════╪═════╪══════╪══════╡
        │ Apple  ┆ 3   ┆ 10.0 ┆ true │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ true │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 5   ┆ 14.0 ┆ true │
        └────────┴─────┴──────┴──────┘

        """
        return self.agg(pli.all().max())

    def count(self) -> pli.DataFrame:
        """
        Count the number of values in each group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).count()
        shape: (3, 2)
        ┌────────┬───────┐
        │ d      ┆ count │
        │ ---    ┆ ---   │
        │ str    ┆ u32   │
        ╞════════╪═══════╡
        │ Apple  ┆ 3     │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 1     │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 2     │
        └────────┴───────┘

        """
        return self.agg(pli.lazy_functions.count())

    def mean(self) -> pli.DataFrame:
        """
        Reduce the groups to the mean values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )

        >>> df.groupby("d", maintain_order=True).mean()
        shape: (3, 4)
        ┌────────┬─────┬──────────┬──────┐
        │ d      ┆ a   ┆ b        ┆ c    │
        │ ---    ┆ --- ┆ ---      ┆ ---  │
        │ str    ┆ f64 ┆ f64      ┆ bool │
        ╞════════╪═════╪══════════╪══════╡
        │ Apple  ┆ 2.0 ┆ 4.833333 ┆ null │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Orange ┆ 2.0 ┆ 0.5      ┆ null │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 4.5 ┆ 13.5     ┆ null │
        └────────┴─────┴──────────┴──────┘

        """
        return self.agg(pli.all().mean())

    def n_unique(self) -> pli.DataFrame:
        """
        Count the unique values per group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 1, 3, 4, 5],
        ...         "b": [0.5, 0.5, 0.5, 10, 13, 14],
        ...         "d": ["Apple", "Banana", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )

        >>> df.groupby("d", maintain_order=True).n_unique()
        shape: (2, 3)
        ┌────────┬─────┬─────┐
        │ d      ┆ a   ┆ b   │
        │ ---    ┆ --- ┆ --- │
        │ str    ┆ u32 ┆ u32 │
        ╞════════╪═════╪═════╡
        │ Apple  ┆ 2   ┆ 2   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ Banana ┆ 3   ┆ 3   │
        └────────┴─────┴─────┘

        """
        return self.agg(pli.all().n_unique())

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> pli.DataFrame:
        """
        Compute the quantile per group.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).quantile(1)
        shape: (3, 3)
        ┌────────┬─────┬──────┐
        │ d      ┆ a   ┆ b    │
        │ ---    ┆ --- ┆ ---  │
        │ str    ┆ f64 ┆ f64  │
        ╞════════╪═════╪══════╡
        │ Apple  ┆ 3.0 ┆ 10.0 │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Orange ┆ 2.0 ┆ 0.5  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 5.0 ┆ 14.0 │
        └────────┴─────┴──────┘

        """
        return self.agg(pli.all().quantile(quantile, interpolation))

    def median(self) -> pli.DataFrame:
        """
        Return the median per group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "d": ["Apple", "Banana", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).median()
        shape: (2, 3)
        ┌────────┬─────┬──────┐
        │ d      ┆ a   ┆ b    │
        │ ---    ┆ --- ┆ ---  │
        │ str    ┆ f64 ┆ f64  │
        ╞════════╪═════╪══════╡
        │ Apple  ┆ 2.0 ┆ 4.0  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 4.0 ┆ 13.0 │
        └────────┴─────┴──────┘

        """
        return self.agg(pli.all().median())

    def agg_list(self) -> pli.DataFrame:
        """
        Aggregate the groups into Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
        >>> df.groupby("a", maintain_order=True).agg_list()
        shape: (2, 2)
        ┌─────┬───────────┐
        │ a   ┆ b         │
        │ --- ┆ ---       │
        │ str ┆ list[i64] │
        ╞═════╪═══════════╡
        │ one ┆ [1, 3]    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ two ┆ [2, 4]    │
        └─────┴───────────┘

        """
        return self.agg(pli.all().list())


class RollingGroupBy(Generic[DF]):
    """
    A rolling grouper.

    This has an `.agg` method which will allow you to run all polars expressions in a
    groupby context.
    """

    def __init__(
        self,
        df: DF,
        index_column: str,
        period: str | timedelta,
        offset: str | timedelta | None,
        closed: ClosedWindow = "none",
        by: str | Sequence[str] | pli.Expr | Sequence[pli.Expr] | None = None,
    ):
        period = _timedelta_to_pl_duration(period)
        offset = _timedelta_to_pl_duration(offset)

        self.df = df
        self.time_column = index_column
        self.period = period
        self.offset = offset
        self.closed = closed
        self.by = by

    def agg(self, aggs: pli.Expr | Sequence[pli.Expr]) -> pli.DataFrame:
        return (
            self.df.lazy()
            .groupby_rolling(
                index_column=self.time_column,
                period=self.period,
                offset=self.offset,
                closed=self.closed,
                by=self.by,
            )
            .agg(aggs)
            .collect(no_optimization=True)
        )


class DynamicGroupBy(Generic[DF]):
    """
    A dynamic grouper.

    This has an `.agg` method which allows you to run all polars expressions in a
    groupby context.
    """

    def __init__(
        self,
        df: DF,
        index_column: str,
        every: str | timedelta,
        period: str | timedelta | None,
        offset: str | timedelta | None,
        truncate: bool = True,
        include_boundaries: bool = True,
        closed: ClosedWindow = "none",
        by: str | Sequence[str] | pli.Expr | Sequence[pli.Expr] | None = None,
        start_by: StartBy = "window",
    ):

        period = _timedelta_to_pl_duration(period)
        offset = _timedelta_to_pl_duration(offset)
        every = _timedelta_to_pl_duration(every)

        self.df = df
        self.time_column = index_column
        self.every = every
        self.period = period
        self.offset = offset
        self.truncate = truncate
        self.include_boundaries = include_boundaries
        self.closed = closed
        self.by = by
        self.start_by = start_by

    def agg(self, aggs: pli.Expr | Sequence[pli.Expr]) -> pli.DataFrame:
        return (
            self.df.lazy()
            .groupby_dynamic(
                index_column=self.time_column,
                every=self.every,
                period=self.period,
                offset=self.offset,
                truncate=self.truncate,
                include_boundaries=self.include_boundaries,
                closed=self.closed,
                by=self.by,
                start_by=self.start_by,
            )
            .agg(aggs)
            .collect(no_optimization=True)
        )


class GBSelection(Generic[DF]):
    """Utility class returned in a groupby operation."""

    def __init__(
        self,
        df: PyDataFrame,
        by: str | Sequence[str],
        selection: Sequence[str] | None,
        dataframe_class: type[DF],
    ):
        self._df = df
        self.by = by
        self.selection = selection
        self._dataframe_class = dataframe_class

    def first(self) -> DF:
        """Aggregate the first values in the group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "first")
        )

    def last(self) -> DF:
        """Aggregate the last values in the group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "last")
        )

    def sum(self) -> DF:
        """Reduce the groups to the sum."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "sum")
        )

    def min(self) -> DF:
        """Reduce the groups to the minimal value."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "min")
        )

    def max(self) -> DF:
        """Reduce the groups to the maximal value."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "max")
        )

    def count(self) -> DF:
        """
        Count the number of values in each group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 3, 4],
        ...         "bar": ["a", "b", "c", "a"],
        ...     }
        ... )
        >>> df.groupby("bar", maintain_order=True).count()  # counts nulls
        shape: (3, 2)
        ┌─────┬───────┐
        │ bar ┆ count │
        │ --- ┆ ---   │
        │ str ┆ u32   │
        ╞═════╪═══════╡
        │ a   ┆ 2     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b   ┆ 1     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ c   ┆ 1     │
        └─────┴───────┘

        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "count")
        )

    def mean(self) -> DF:
        """Reduce the groups to the mean values."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "mean")
        )

    def n_unique(self) -> DF:
        """Count the unique values per group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "n_unique")
        )

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> DF:
        """
        Compute the quantile per group.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        """
        return self._dataframe_class._from_pydf(
            self._df.groupby_quantile(self.by, self.selection, quantile, interpolation)
        )

    def median(self) -> DF:
        """Return the median per group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "median")
        )

    def agg_list(self) -> DF:
        """Aggregate the groups into Series."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "agg_list")
        )

    def apply(
        self,
        func: Callable[[Any], Any],
        return_dtype: type[DataType] | None = None,
    ) -> DF:
        """Apply a function over the groups."""
        df = self.agg_list()
        if self.selection is None:
            raise TypeError(
                "apply not available for Groupby.select_all(). Use select() instead."
            )
        for name in self.selection:
            s = df.drop_in_place(name + "_agg_list").apply(func, return_dtype)
            s.rename(name, in_place=True)
            df.with_column(s)

        return df
