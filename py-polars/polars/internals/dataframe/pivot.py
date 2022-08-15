from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import polars.internals as pli

if TYPE_CHECKING:
    from polars.polars import PyDataFrame

# A type variable used to refer to a polars.DataFrame or any subclass of it.
# Used to annotate DataFrame methods which returns the same type as self.
DF = TypeVar("DF", bound="pli.DataFrame")


class PivotOps(Generic[DF]):
    """Utility class returned in a pivot operation."""

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

    def first(self) -> DF:
        """Get the first value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "first")
        )

    def sum(self) -> DF:
        """Get the sum per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "sum")
        )

    def min(self) -> DF:
        """Get the minimal value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "min")
        )

    def max(self) -> DF:
        """Get the maximal value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "max")
        )

    def mean(self) -> DF:
        """Get the mean value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "mean")
        )

    def count(self) -> DF:
        """Count the values per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "count")
        )

    def median(self) -> DF:
        """Get the median value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )

    def last(self) -> DF:
        """Get the last value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "last")
        )
