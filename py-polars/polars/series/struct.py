from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

from polars.series.utils import expr_dispatch
from polars.utils._wrap import wrap_df
from polars.utils.various import sphinx_accessor

if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.datatypes import PolarsDataType
    from polars.polars import PySeries
elif os.getenv("BUILDING_SPHINX_DOCS"):
    property = sphinx_accessor


@expr_dispatch
class StructNameSpace:
    """Series.struct namespace."""

    _accessor = "struct"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def __getitem__(self, item: int | str) -> Series:
        if isinstance(item, int):
            return self.field(self.fields[item])
        elif isinstance(item, str):
            return self.field(item)
        else:
            raise ValueError(f"expected type 'int | str', got {type(item)}")

    def _ipython_key_completions_(self) -> list[str]:
        return self.fields

    @property
    def fields(self) -> list[str]:
        """Get the names of the fields."""
        if getattr(self, "_s", None) is None:
            return []
        return self._s.struct_fields()

    def field(self, name: str) -> Series:
        """
        Retrieve one of the fields of this `Struct` as a new Series.

        Parameters
        ----------
        name
            Name of the field

        """

    def rename_fields(self, names: Sequence[str]) -> Series:
        """
        Rename the fields of the struct.

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        """

    @property
    def schema(self) -> dict[str, PolarsDataType]:
        """Get the struct definition as a name/dtype schema dict."""
        if getattr(self, "_s", None) is None:
            return {}
        return self._s.dtype().to_schema()

    def unnest(self) -> DataFrame:
        """
        Convert this struct Series to a DataFrame with a separate column for each field.

        Examples
        --------
        >>> s = pl.Series([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        >>> s.struct.unnest()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 2   │
        │ 3   ┆ 4   │
        └─────┴─────┘

        """
        return wrap_df(self._s.struct_unnest())
