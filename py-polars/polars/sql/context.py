from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_ldf

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PySQLContext

if TYPE_CHECKING:
    from polars.dataframe import DataFrame
    from polars.lazyframe import LazyFrame


class SQLContext:
    """
    Run SQL query against a LazyFrame.

    Warnings
    --------
    This feature is experimental and may change without it being
    considered a breaking change.

    """

    def __init__(self) -> None:
        self._ctxt = PySQLContext.new()

    def register(self, name: str, lf: LazyFrame) -> None:
        """
        Register a ``LazyFrame`` in this ``SQLContext`` under a given ``name``.

        Parameters
        ----------
        name
            Name of the table
        lf
            LazyFrame to add as this table name.

        """
        self._ctxt.register(name, lf._ldf)

    def execute(self, query: str) -> LazyFrame:
        """
        Parse the givens SQL query and transform that to a ``LazyFrame``.

        Parameters
        ----------
        query
            A SQL query

        """
        return wrap_ldf(self._ctxt.execute(query))

    def query(self, query: str) -> DataFrame:
        return self.execute(query).collect()
