from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import call_expr

if TYPE_CHECKING:
    from polars.polars import PySeries


class StructNameSpace:
    """Series.struct namespace."""

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    @property
    def namespace(self) -> str:
        return "struct"

    def to_frame(self) -> pli.DataFrame:
        """Convert this Struct Series to a DataFrame."""
        return pli.wrap_df(self._s.struct_to_frame())

    @property
    def fields(self) -> list[str]:
        """Get the names of the fields."""
        return self._s.struct_fields()

    @call_expr
    def field(self, name: str) -> pli.Series:
        """
        Retrieve one of the fields of this `Struct` as a new Series.

        Parameters
        ----------
        name
            Name of the field

        """
        ...

    @call_expr
    def rename_fields(self, names: list[str]) -> pli.Series:
        """
        Rename the fields of the struct

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        """
        ...
