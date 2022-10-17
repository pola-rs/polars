from __future__ import annotations

from typing import Callable, Generic, Sequence, TypeVar

import polars.internals as pli
from polars.datatypes import Schema
from polars.internals import selection_to_pyexpr_list
from polars.utils import is_expr_sequence

try:
    from polars.polars import PyLazyGroupBy

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

# Used to type any type or subclass of LazyFrame.
# Used to indicate when LazyFrame methods return the same type as self,
# including sub-classes.
LDF = TypeVar("LDF", bound="pli.LazyFrame")


class LazyGroupBy(Generic[LDF]):
    """Created by `df.lazy().groupby("foo)"`."""

    def __init__(self, lgb: PyLazyGroupBy, lazyframe_class: type[LDF]) -> None:
        self.lgb = lgb
        self._lazyframe_class = lazyframe_class

    def agg(self, aggs: pli.Expr | Sequence[pli.Expr]) -> LDF:
        """
        Describe the aggregation that need to be done on a group.

        Parameters
        ----------
        aggs
            Single / multiple aggregation expression(s).

        Examples
        --------
        >>> (
        ...     pl.scan_csv("data.csv")
        ...     .groupby("groups")
        ...     .agg(
        ...         [
        ...             pl.col("name").n_unique().alias("unique_names"),
        ...             pl.max("values"),
        ...         ]
        ...     )
        ... )  # doctest: +SKIP

        """
        if not (isinstance(aggs, pli.Expr) or is_expr_sequence(aggs)):
            msg = f"expected 'Expr | Sequence[Expr]', got '{type(aggs)}'"
            raise TypeError(msg)

        pyexprs = selection_to_pyexpr_list(aggs)
        return self._lazyframe_class._from_pyldf(self.lgb.agg(pyexprs))

    def head(self, n: int = 5) -> LDF:
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
        return self._lazyframe_class._from_pyldf(self.lgb.head(n))

    def tail(self, n: int = 5) -> LDF:
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
        >>> df.groupby("letters").tail(2).sort("letters")
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
        return self._lazyframe_class._from_pyldf(self.lgb.tail(n))

    def apply(
        self, f: Callable[[pli.DataFrame], pli.DataFrame], schema: Schema | None
    ) -> LDF:
        """
        Apply a custom/user-defined function (UDF) over the groups as a new DataFrame.

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
            Function to apply over each group of the `LazyFrame`.
        schema
            Schema of the output function. This has to be known statically.
            If the schema provided is incorrect, this is a bug in the callers
            query and may lead to errors.
            If set to None, polars assumes the schema is unchanged.


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
        ...     df.lazy()
        ...     .groupby("color")
        ...     .apply(lambda group_df: group_df.sample(2), schema=None)
        ...     .collect()
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
        ...     df.lazy()
        ...     .filter(pl.arange(0, pl.count()).shuffle().over("color") < 2)
        ...     .collect()
        ... )  # doctest: +IGNORE_RESULT

        """
        return self._lazyframe_class._from_pyldf(self.lgb.apply(f, schema))
