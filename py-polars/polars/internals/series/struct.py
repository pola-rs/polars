from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import expr_dispatch
from polars.utils import accessor

if TYPE_CHECKING:
    from polars.polars import PySeries


@expr_dispatch
class StructNameSpace:
    """Series.struct namespace."""

    _accessor = "struct"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    def __getitem__(self, item: int | str) -> pli.Series:
        if isinstance(item, int):
            return self.field(self.fields[item])
        elif isinstance(item, str):
            return self.field(item)
        else:
            raise ValueError(f"expected type 'int | str', got {type(item)}")

    def _ipython_key_completions_(self) -> list[str]:
        return self.fields

    def to_frame(self) -> pli.DataFrame:
        """Convert this Struct Series to a DataFrame."""
        return pli.wrap_df(self._s.struct_to_frame())

    @accessor
    def fields(self) -> list[str]:
        """Get the names of the fields."""
        if getattr(self, "_s", None) is None:
            return []
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
        Rename the fields of the struct.

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        """
