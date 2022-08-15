from __future__ import annotations

import polars.internals as pli


class StructNameSpace:
    def __init__(self, s: pli.Series):
        self.s = s

    def to_frame(self) -> pli.DataFrame:
        """Convert this Struct Series to a DataFrame."""
        return pli.wrap_df(self.s._s.struct_to_frame())

    def field(self, name: str) -> pli.Series:
        """
        Retrieve one of the fields of this `Struct` as a new Series.

        Parameters
        ----------
        name
            Name of the field

        """
        return pli.select(pli.lit(self.s).struct.field(name)).to_series()

    @property
    def fields(self) -> list[str]:
        """Get the names of the fields."""
        return self.s._s.struct_fields()

    def rename_fields(self, names: list[str]) -> pli.Series:
        """
        Rename the fields of the struct

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        """
        return pli.select(pli.lit(self.s).struct.rename_fields(names)).to_series()
