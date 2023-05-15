from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    Collection,
    Generic,
    Mapping,
    overload,
)

from polars.dataframe import DataFrame
from polars.lazyframe import LazyFrame
from polars.type_aliases import FrameType
from polars.utils._wrap import wrap_ldf
from polars.utils.decorators import deprecated_alias, redirect
from polars.utils.various import _get_stack_locals

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PySQLContext

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 8):
        from typing import Final, Literal
    else:
        from typing_extensions import Final, Literal

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


@redirect({"query": ("execute", {"eager": True})})
class SQLContext(Generic[FrameType]):
    """
    Run a SQL query against a LazyFrame.

    Warnings
    --------
    This feature is experimental and may change without it being considered breaking.

    """

    _ctxt: PySQLContext
    _eager_execution: Final[bool]

    # note: the type-overloaded methods are required to support accurate typing
    # of the frame return from "execute" (which may be DataFrame or LazyFrame),
    # as that is influenced by both the "eager_execution" flag at init-time AND
    # the "eager" flag at query-time.

    @overload
    def __init__(
        self: SQLContext[LazyFrame],
        frames: Mapping[str, DataFrame | LazyFrame] | None = ...,
        *,
        register_globals: bool | int = ...,
        eager_execution: Literal[False] = False,
        **named_frames: DataFrame | LazyFrame,
    ) -> None:
        ...

    @overload
    def __init__(
        self: SQLContext[DataFrame],
        frames: Mapping[str, DataFrame | LazyFrame] | None = ...,
        *,
        register_globals: bool | int = ...,
        eager_execution: Literal[True],
        **named_frames: DataFrame | LazyFrame,
    ) -> None:
        ...

    def __init__(
        self,
        frames: Mapping[str, DataFrame | LazyFrame] | None = None,
        *,
        register_globals: bool | int = False,
        eager_execution: bool = False,
        **named_frames: DataFrame | LazyFrame,
    ) -> None:
        """
        Initialise a new ``SQLContext``.

        Parameters
        ----------
        frames
            A ``{name:lazyframe, ...}`` mapping.
        register_globals
            Register all``LazyFrame`` objects found in the globals, automatically
            mapping their variable name to a table name. If given an integer then
            only the most recent "n" frames found will be registered.
        eager_execution
            Always execute queries in this context eagerly (returning a `` DataFrame``
            instead of ``LazyFrame``).
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
        self._eager_execution = eager_execution

        frames = dict(frames or {})
        if register_globals:
            for name, obj in _get_stack_locals(
                of_type=(DataFrame, LazyFrame),
                n_objects=None if (register_globals is True) else None,
            ).items():
                if name not in frames and name not in named_frames:
                    named_frames[name] = obj

        if frames or named_frames:
            self.register_many(frames, **named_frames)

    # these overloads are necessary to cover the possible permutations
    # of the init-time "eager_execution" param, and the "eager" param.

    @overload
    def execute(
        self: SQLContext[DataFrame], query: str, eager: Literal[None] = None
    ) -> DataFrame:
        ...

    @overload
    def execute(
        self: SQLContext[DataFrame], query: str, eager: Literal[False]
    ) -> LazyFrame:
        ...

    @overload
    def execute(
        self: SQLContext[DataFrame], query: str, eager: Literal[True]
    ) -> DataFrame:
        ...

    @overload
    def execute(
        self: SQLContext[LazyFrame], query: str, eager: Literal[None] = None
    ) -> LazyFrame:
        ...

    @overload
    def execute(
        self: SQLContext[LazyFrame], query: str, eager: Literal[False]
    ) -> LazyFrame:
        ...

    @overload
    def execute(
        self: SQLContext[LazyFrame], query: str, eager: Literal[True]
    ) -> DataFrame:
        ...

    def execute(self, query: str, eager: bool | None = None) -> LazyFrame | DataFrame:
        """
        Parse the given SQL query and execute it against the underlying frame data.

        Parameters
        ----------
        query
            A valid string SQL query.
        eager
            Apply the query eagerly, returning ``DataFrame`` instead of ``LazyFrame``.
            If unset, the value of the init-time parameter "eager_execution" will be
            used (default is False).

        """
        res = wrap_ldf(self._ctxt.execute(query))
        return res.collect() if (eager or self._eager_execution) else res

    @deprecated_alias(lf="frame")
    def register(self, name: str, frame: DataFrame | LazyFrame) -> Self:
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
            frame = frame.lazy()
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
            frames=_get_stack_locals(of_type=(DataFrame, LazyFrame), n_objects=n)
        )

    def register_many(
        self,
        frames: Mapping[str, DataFrame | LazyFrame] | None = None,
        **named_frames: DataFrame | LazyFrame,
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
        frames = dict(frames or {})
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
