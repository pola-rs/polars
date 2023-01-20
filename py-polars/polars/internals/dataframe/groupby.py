from __future__ import annotations

import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Generic, Sequence, TypeVar

import polars.internals as pli
from polars.internals.dataframe.pivot import PivotOps
from polars.utils import _timedelta_to_pl_duration, is_str_sequence

if TYPE_CHECKING:
    from polars.internals.type_aliases import (
        ClosedInterval,
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
    >>> for group in df.groupby("foo", maintain_order=True):
    ...     print(group)
    ...
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ a   ┆ 1   │
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

    def __init__(
        self,
        df: PyDataFrame,
        by: str | pli.Expr | Sequence[str | pli.Expr],
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

    def __iter__(self) -> GroupBy[DF]:
        warnings.warn(
            "Return type of groupby iteration will change in the next breaking release."
            " Iteration will returns tuples of (group_key, data) instead of just data.",
            category=FutureWarning,
            stacklevel=2,
        )

        by = [self.by] if isinstance(self.by, (str, pli.Expr)) else self.by

        # Find any single column that is not specified as 'by'
        columns = self._df.columns()
        by_names = {c if isinstance(c, str) else c.meta.output_name() for c in by}
        try:
            non_by_col = next(c for c in columns if c not in by_names)
        except StopIteration:
            non_by_col = None

        # Get the group indices using that column
        if non_by_col is not None:
            groups_df = self.agg(pli.col(non_by_col).agg_groups())
            group_indices = groups_df.select(non_by_col).to_series()
        else:
            # TODO: Properly handle expression input
            group_indices = pli.Series([[i] for i in range(self._df.height())])

        self._group_indices = group_indices
        self._current_index = 0

        return self

    def __next__(self) -> DF:
        if self._current_index >= len(self._group_indices):
            raise StopIteration

        df = self._dataframe_class._from_pydf(self._df)
        group = df[self._group_indices[self._current_index]]
        self._current_index += 1
        return group

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
        │ 1   ┆ green ┆ triangle │
        │ 2   ┆ green ┆ square   │
        │ 3   ┆ red   ┆ triangle │
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
        │ 2   ┆ green ┆ square   │
        │ 4   ┆ red   ┆ square   │
        │ 3   ┆ red   ┆ triangle │
        └─────┴───────┴──────────┘

        It is better to implement this with an expression:

        >>> (
        ...     df.filter(pl.arange(0, pl.count()).shuffle().over("color") < 2)
        ... )  # doctest: +IGNORE_RESULT

        """
        by: Sequence[str]
        if isinstance(self.by, str):
            by = [self.by]
        elif is_str_sequence(self.by):
            by = self.by
        else:
            raise TypeError("Cannot call `apply` when grouping by an expression.")

        return self._dataframe_class._from_pydf(self._df.groupby_apply(by, f))

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
        │ c       ┆ 2   │
        │ a       ┆ 3   │
        │ c       ┆ 4   │
        │ a       ┆ 5   │
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
        │ a       ┆ 5   │
        │ b       ┆ 6   │
        │ c       ┆ 1   │
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
        │ c       ┆ 2   │
        │ a       ┆ 3   │
        │ c       ┆ 4   │
        │ a       ┆ 5   │
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
        │ a       ┆ 5   │
        │ b       ┆ 6   │
        │ c       ┆ 2   │
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

        .. deprecated:: 0.13.23
            `DataFrame.groupby.pivot` will be removed in favour of `DataFrame.pivot`.

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
        │ two ┆ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┴─────┘

        """
        warnings.warn(
            "`DataFrame.groupby.pivot` is deprecated and will be removed in a future"
            " version. Use `DataFrame.pivot` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(pivot_column, str):
            pivot_column = [pivot_column]
        if isinstance(values_column, str):
            values_column = [values_column]
        return PivotOps(
            self._df,
            self.by,  # type: ignore[arg-type]
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
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
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
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
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
        │ Orange ┆ 2   ┆ 0.5  ┆ 1   │
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
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
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
        │ Orange ┆ 2   ┆ 0.5  ┆ true │
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
        │ Orange ┆ 1     │
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
        ┌────────┬─────┬──────────┬──────────┐
        │ d      ┆ a   ┆ b        ┆ c        │
        │ ---    ┆ --- ┆ ---      ┆ ---      │
        │ str    ┆ f64 ┆ f64      ┆ f64      │
        ╞════════╪═════╪══════════╪══════════╡
        │ Apple  ┆ 2.0 ┆ 4.833333 ┆ 0.666667 │
        │ Orange ┆ 2.0 ┆ 0.5      ┆ 1.0      │
        │ Banana ┆ 4.5 ┆ 13.5     ┆ 0.5      │
        └────────┴─────┴──────────┴──────────┘

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
        │ Orange ┆ 2.0 ┆ 0.5  │
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
        closed: ClosedInterval = "none",
        by: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
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
        closed: ClosedInterval = "none",
        by: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
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
