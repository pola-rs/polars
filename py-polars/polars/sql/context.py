from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Collection

from polars.dataframe import DataFrame
from polars.lazyframe import LazyFrame
from polars.utils._wrap import wrap_ldf
from polars.utils.decorators import deprecated_alias
from polars.utils.various import _get_stack_locals

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PySQLContext

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class SQLContext:
    """
    Run a SQL query against a LazyFrame.

    Warnings
    --------
    This feature is experimental and may change without it being considered breaking.

    """

    def __init__(
        self,
        frames: dict[str, LazyFrame] | None = None,
        *,
        register_globals: bool | int = False,
        **named_frames: LazyFrame,
    ) -> None:
        """
        Initialise a new ``SQLContext``, optionally registering ``LazyFrame`` objects.

        Parameters
        ----------
        frames
            A ``{name:lazyframe, ...}`` mapping.
        register_globals
            Register all``LazyFrame`` objects found in the globals, automatically
            mapping their variable name to a table name. If given an integer then
            only the most recent "n" frames found will be registered.
        **named_frames
            Named ``LazyFrame`` objects, provided as kwargs.

        Examples
        --------
        >>> lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})
        >>> res = pl.SQLContext(frame=lf).execute(
        ...     "SELECT b, a*2 AS two_a FROM frame WHERE b IS NOT NULL"
        ... )
        >>> res.collect()
        shape: (2, 2)
        ┌─────┬───────┐
        │ b   ┆ two_a │
        │ --- ┆ ---   │
        │ str ┆ i64   │
        ╞═════╪═══════╡
        │ x   ┆ 2     │
        │ z   ┆ 6     │
        └─────┴───────┘

        """
        self._ctxt = PySQLContext.new()

        frames = frames or {}
        if register_globals:
            n = None if register_globals is True else None
            for name, obj in _get_stack_locals(of_type=LazyFrame, n_objects=n).items():
                if name not in frames and name not in named_frames:
                    named_frames[name] = obj

        if frames or named_frames:
            self.register_many(frames, **named_frames)

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
    def register(self, name: str, frame: LazyFrame) -> Self:
        """
        Register a ``LazyFrame`` in this ``SQLContext`` under a given ``name``.

        Parameters
        ----------
        name
            Name of the table.
        frame
            LazyFrame to add as this table name.

        """
        if isinstance(frame, DataFrame):
            raise TypeError(
                "Cannot register an eager DataFrame in an SQLContext; use LazyFrame instead"
            )
        self._ctxt.register(name, frame._ldf)
        return self

    def register_globals(self, n: int | None = None) -> Self:
        """
        Register ``LazyFrame`` objects present in the current globals scope.

        Automatically maps variable names to table names.

        Parameters
        ----------
        n
            Register only the most recent "n" ``LazyFrame`` objects.

        Examples
        --------
        >>> lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})
        >>> lf2 = pl.LazyFrame({"a": [2, 3, 4], "c": ["t", "w", "v"]})
        >>>
        >>> pl.SQLContext(register_globals=True).execute(
        ...     "SELECT a, b, c FROM lf1 LEFT JOIN lf2 USING (a) ORDER BY a DESC"
        ... ).collect()
        shape: (3, 3)
        ┌─────┬──────┬──────┐
        │ a   ┆ b    ┆ c    │
        │ --- ┆ ---  ┆ ---  │
        │ i64 ┆ str  ┆ str  │
        ╞═════╪══════╪══════╡
        │ 3   ┆ z    ┆ w    │
        │ 2   ┆ null ┆ t    │
        │ 1   ┆ x    ┆ null │
        └─────┴──────┴──────┘

        """
        return self.register_many(
            frames=_get_stack_locals(of_type=LazyFrame, n_objects=n)
        )

    def register_many(
        self, frames: dict[str, LazyFrame] | None = None, **named_frames: LazyFrame
    ) -> Self:
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

        for name, frame in frames.items():
            self.register(name, frame)
        return self

    def unregister(self, names: str | Collection[str]) -> Self:
        """
        Unregister one or more table names from this ``SQLContext``.

        Parameters
        ----------
        names
            Names of the tables to unregister.

        """
        if isinstance(names, str):
            names = [names]
        for nm in names:
            self._ctxt.unregister(nm)
        return self
