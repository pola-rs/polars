from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_ldf
from polars.utils.decorators import deprecated_alias

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PySQLContext

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame


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

    def execute(self, query: str) -> LazyFrame:
        """
        Parse the given SQL query and apply it lazily, returning a ``LazyFrame``.

        Parameters
        ----------
        query
            A valid string SQL query.

        """
        return wrap_ldf(self._ctxt.execute(query))

    def query(self, query: str) -> DataFrame:
        """
        Parse the given SQL query and execute it eagerly to return a ``DataFrame``.

        Parameters
        ----------
        query
            A valid string SQL query.

        """
        return self.execute(query).collect()

    @deprecated_alias(lf="frame")
    def register(self, name: str, frame: LazyFrame) -> None:
        """
        Register a ``LazyFrame`` in this ``SQLContext`` under a given ``name``.

        Parameters
        ----------
        name
            Name of the table
        frame
            LazyFrame to add as this table name.

        """
        self._ctxt.register(name, frame._ldf)

    def register_many(
        self, frames: dict[str, LazyFrame] | None = None, **named_frames: LazyFrame
    ) -> None:
        """
        Register multiple named ``LazyFrame`` objects in this ``SQLContext``.

        Parameters
        ----------
        frames
            A ``{name:lazyframe, ...}`` mapping.
        **named_frames
            Named ``LazyFrame`` objects, provided as kwargs.

        """
        frames = frames or {}
        frames.update(named_frames)

        for name, lf in frames.items():
            self._ctxt.register(name, lf._ldf)
