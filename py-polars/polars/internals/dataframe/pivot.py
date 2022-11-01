from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import polars.internals as pli
from polars.internals.type_aliases import PivotAgg

if TYPE_CHECKING:
    from polars.polars import PyDataFrame

# A type variable used to refer to a polars.DataFrame or any subclass of it.
# Used to annotate DataFrame methods which returns the same type as self.
DF = TypeVar("DF", bound="pli.DataFrame")


class PivotOps(Generic[DF]):
    """
    Utility class returned in a pivot operation.

    .. deprecated:: 0.13.23
          `PivotOps` will be removed in favour of `DataFrame.pivot`.

    """

    def __init__(
        self,
        df: PyDataFrame,
        by: str | list[str],
        pivot_column: str | list[str],
        values_column: str | list[str],
        dataframe_class: type[DF],
    ):
        self._df = df
        self.by = by
        self.pivot_column = pivot_column
        self.values_column = values_column
        self._dataframe_class = dataframe_class

    def _execute(self, aggregate_fn: PivotAgg) -> DF:
        return self._dataframe_class._from_pydf(
            pli.wrap_df(self._df)
            .pivot(
                index=self.by,
                columns=self.pivot_column,
                values=self.values_column,
                aggregate_fn=aggregate_fn,
            )
            ._df
        )

    def first(self) -> DF:
        """Get the first value per group."""
        return self._execute("first")

    def sum(self) -> DF:
        """Get the sum per group."""
        return self._execute("sum")

    def min(self) -> DF:
        """Get the minimal value per group."""
        return self._execute("min")

    def max(self) -> DF:
        """Get the maximal value per group."""
        return self._execute("max")

    def mean(self) -> DF:
        """Get the mean value per group."""
        return self._execute("mean")

    def count(self) -> DF:
        """Count the values per group."""
        return self._execute("count")

    def median(self) -> DF:
        """Get the median value per group."""
        return self._execute("median")

    def last(self) -> DF:
        """Get the last value per group."""
        return self._execute("last")
