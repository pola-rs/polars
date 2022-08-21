from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars.polars import PySeries

    if sys.version_info >= (3, 8):
        from typing import Final
    else:
        from typing_extensions import Final


@expr_dispatch
class StructNameSpace:
    """Series.struct namespace."""

    _accessor: Final = "struct"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    def to_frame(self) -> pli.DataFrame:
        """Convert this Struct Series to a DataFrame."""
        return pli.wrap_df(self._s.struct_to_frame())

    @property
    def fields(self) -> list[str]:
        """Get the names of the fields."""
        return self._s.struct_fields()

    def field(self, name: str) -> pli.Series:
        """
        Retrieve one of the fields of this `Struct` as a new Series.

        Parameters
        ----------
        name
            Name of the field

        """

    def rename_fields(self, names: list[str]) -> pli.Series:
        """
        Rename the fields of the struct

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        """
