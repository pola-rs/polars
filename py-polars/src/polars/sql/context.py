from __future__ import annotations

import contextlib
import importlib.util
from typing import (
    TYPE_CHECKING,
    Generic,
    overload,
)

from polars._dependencies import _check_for_pandas, _check_for_pyarrow
from polars._dependencies import pandas as pd
from polars._dependencies import pyarrow as pa
from polars._typing import FrameType
from polars._utils.expired import RenamedParameter, removed_parameters
from polars._utils.pycapsule import is_pycapsule
from polars._utils.unstable import issue_unstable_warning
from polars._utils.various import _get_stack_locals, qualified_type_name
from polars._utils.wrap import wrap_ldf
from polars.convert import from_pandas
from polars.dataframe import DataFrame
from polars.lazyframe import LazyFrame
from polars.series import Series

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PySQLContext

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from types import TracebackType
    from typing import Any, Final, Literal, Self, TypeAlias

    import pyiceberg.catalog
    import pyiceberg.schema
    import pyiceberg.table

    from polars._plr import PyLazyFrame
    from polars._typing import StorageOptionsDict
    from polars.io.iceberg import IcebergCatalogConfig

    CompatibleFrameType: TypeAlias = (
        DataFrame
        | LazyFrame
        | Series
        | pd.DataFrame
        | pd.Series[Any]
        | pa.Table
        | pa.RecordBatch
    )

__all__ = ["SQLContext"]


def _compatible_frame(obj: Any) -> bool:
    """Check if the object can be converted to DataFrame."""
    return (
        is_pycapsule(obj)
        or isinstance(obj, LazyFrame)
        or (_check_for_pandas(obj) and isinstance(obj, (pd.DataFrame, pd.Series)))
        or (_check_for_pyarrow(obj) and isinstance(obj, (pa.Table, pa.RecordBatch)))
    )


def _ensure_lazyframe(obj: Any) -> LazyFrame:
    """Return LazyFrame from compatible input."""
    if isinstance(obj, (DataFrame, LazyFrame)):
        return obj.lazy()
    elif isinstance(obj, Series):
        return obj.to_frame().lazy()
    elif _check_for_pandas(obj) and isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(frame := from_pandas(obj), Series):
            frame = frame.to_frame()
        return frame.lazy()
    elif is_pycapsule(obj) or (
        _check_for_pyarrow(obj) and isinstance(obj, (pa.Table, pa.RecordBatch))
    ):
        return DataFrame(obj).lazy()  # type: ignore[union-attr]
    else:
        msg = f"unrecognised frame type: {qualified_type_name(obj)}"
        raise ValueError(msg)


def _get_frame_locals(
    *,
    all_compatible: bool,
    n_objects: int | None = None,
    named: str | Collection[str] | Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Return compatible frame objects from the local stack."""
    of_type = _compatible_frame if all_compatible else (DataFrame, LazyFrame, Series)
    return _get_stack_locals(of_type=of_type, n_objects=n_objects, named=named)  # type: ignore[arg-type]


class SQLContext(Generic[FrameType]):
    """
    Run SQL queries against DataFrame, LazyFrame, and Series data.

    .. warning::
        This functionality is considered **unstable**, although it is close to being
        considered stable. It may be changed at any point without it being considered
        a breaking change.
    """

    _ctxt: PySQLContext
    _eager_execution: Final[bool]
    _tables_scope_stack: list[set[str]]
    # Attached catalogs and the default catalog on scope entry.
    _catalogs_scope_stack: list[tuple[set[str], str | None]]
    # Attached catalogs by name: config, instance (if known) and storage options.
    _catalogs: dict[
        str,
        tuple[
            IcebergCatalogConfig,
            pyiceberg.catalog.Catalog | None,
            StorageOptionsDict | None,
        ],
    ]

    # note: the type-overloaded methods are required to support accurate typing
    # of the frame return from "execute" (which may be DataFrame or LazyFrame),
    # as that is influenced by both the "eager" flag at init-time AND the "eager"
    # flag at query-time (if anyone can find a lighter-weight set of annotations
    # that successfully resolves this, please go for it... ;)

    @overload
    def __init__(
        self: SQLContext[LazyFrame],
        frames: Mapping[str, CompatibleFrameType | None] | None = ...,
        *,
        register_globals: bool | int = ...,
        eager: Literal[False] = False,
        **named_frames: CompatibleFrameType | None,
    ) -> None: ...

    @overload
    def __init__(
        self: SQLContext[DataFrame],
        frames: Mapping[str, CompatibleFrameType | None] | None = ...,
        *,
        register_globals: bool | int = ...,
        eager: Literal[True],
        **named_frames: CompatibleFrameType | None,
    ) -> None: ...

    @overload
    def __init__(
        self: SQLContext[DataFrame],
        frames: Mapping[str, CompatibleFrameType | None] | None = ...,
        *,
        register_globals: bool | int = ...,
        eager: bool,
        **named_frames: CompatibleFrameType | None,
    ) -> None: ...

    @removed_parameters(
        RenamedParameter(
            name="eager_execution",
            new_name="eager",
            deprecated_in="0.20.31",
            removed_in="2.0",
        ),
    )
    def __init__(
        self,
        frames: Mapping[str, CompatibleFrameType | None] | None = None,
        *,
        register_globals: bool | int = False,
        eager: bool = False,
        **named_frames: CompatibleFrameType | None,
    ) -> None:
        """
        Initialize a new `SQLContext`.

        Parameters
        ----------
        frames
            A `{name:frame, ...}` mapping which can include Polars frames *and*
            pandas DataFrames, Series and pyarrow Table and RecordBatch objects.
        register_globals
            Register compatible objects (polars DataFrame, LazyFrame, and Series) found
            in the globals, automatically mapping their variable name to a table name.
            To register other objects (pandas/pyarrow data) pass them explicitly, or
            call the `execute_global` classmethod. If given an integer then only the
            most recent "n" objects found will be registered.
        eager
            If True, returns execution results as `DataFrame` instead of `LazyFrame`.
            (Note that the query itself is always executed in lazy-mode; this parameter
            impacts whether :meth:`execute` returns an eager or lazy result frame).
        **named_frames
            Named eager/lazy frames, provided as kwargs.

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
        issue_unstable_warning(
            "`SQLContext` is considered **unstable**, although it is close to being considered stable."
        )
        self._ctxt = PySQLContext.new()
        self._eager_execution = eager
        self._catalogs = {}

        frames = dict(frames or {})
        if register_globals:
            for name, obj in _get_frame_locals(
                all_compatible=False,
            ).items():
                if name not in frames and name not in named_frames:
                    named_frames[name] = obj

        if frames or named_frames:
            frames.update(named_frames)
            self.register_many(frames)

    @overload
    @classmethod
    def execute_global(
        cls, query: str, *, eager: Literal[False] = False
    ) -> LazyFrame: ...

    @overload
    @classmethod
    def execute_global(cls, query: str, *, eager: Literal[True]) -> DataFrame: ...

    @overload
    @classmethod
    def execute_global(cls, query: str, *, eager: bool) -> DataFrame | LazyFrame: ...

    @classmethod
    def execute_global(
        cls, query: str, *, eager: bool = False
    ) -> DataFrame | LazyFrame:
        """
        Immediately execute a SQL query, automatically registering frame globals.

        Notes
        -----
        * This convenience method automatically registers all compatible objects in
          the local stack that are referenced in the query, mapping their variable name
          to a table name. Note that in addition to polars DataFrame, LazyFrame, and
          Series this method *also* registers pandas DataFrame, Series, and pyarrow
          Table and RecordBatch objects.
        * Instead of calling this classmethod you should consider using `pl.sql`,
          which will use this code internally.

        Parameters
        ----------
        query
            A valid SQL query string.
        eager
            If True, returns execution results as `DataFrame` instead of `LazyFrame`.
            (Note that the query itself is always executed in lazy-mode).

        Examples
        --------
        >>> import pandas as pd
        >>> df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df_pandas = pd.DataFrame({"a": [2, 3, 4], "c": [7, 8, 9]})

        Join a polars LazyFrame with a pandas DataFrame (note use of the preferred
        `pl.sql` method, which is equivalent to `SQLContext.execute_global`):

        >>> pl.sql(
        ...     "SELECT df.*, c FROM df JOIN df_pandas USING(a) ORDER BY ALL"
        ... ).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 5   ┆ 7   │
        │ 3   ┆ 6   ┆ 8   │
        └─────┴─────┴─────┘
        """
        # extract table names from the query, checking against them so that
        # we don't register unnecessary objects found in the globals
        table_names = set(
            PySQLContext.table_identifiers(
                query,
                include_schema=False,
                unique=True,
            )
        )
        # get compatible objects from the globals, constraining by possible names
        named_frames = _get_frame_locals(all_compatible=True, named=table_names)
        with cls(frames=named_frames, register_globals=False) as ctx:
            return ctx.execute(query=query, eager=eager)

    def __enter__(self) -> SQLContext[FrameType]:
        """Track tables and catalogs on scope entry; supports nested scopes."""
        self._tables_scope_stack = getattr(self, "_tables_scope_stack", [])
        self._tables_scope_stack.append(set(self.tables()))
        self._catalogs_scope_stack = getattr(self, "_catalogs_scope_stack", [])
        self._catalogs_scope_stack.append(
            (set(self._catalogs), self._ctxt.default_catalog())
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Unregister any tables and catalogs created within the given scope on exit.

        If that detaches the default catalog, the default on scope entry is restored.

        See Also
        --------
        unregister
        unregister_catalog
        """
        self.unregister(
            names=(set(self.tables()) - self._tables_scope_stack.pop()),
        )
        attached, entry_default = self._catalogs_scope_stack.pop()
        detached = set(self._catalogs) - attached
        detaches_default = self._ctxt.default_catalog() in detached
        for name in detached:
            self._ctxt.unregister_catalog(name)
            del self._catalogs[name]
        if detaches_default and entry_default in self._catalogs:
            self._ctxt.set_default_catalog(entry_default)

    def __repr__(self) -> str:
        n_tables = len(self.tables())
        return f"<SQLContext [tables:{n_tables}] at 0x{id(self):x}>"

    # these overloads are necessary to cover the possible permutations
    # of the init-time "eager" param, and the local "eager" param.

    @overload
    def execute(
        self: SQLContext[DataFrame], query: str, *, eager: None = ...
    ) -> DataFrame: ...

    @overload
    def execute(
        self: SQLContext[DataFrame], query: str, *, eager: Literal[False]
    ) -> LazyFrame: ...

    @overload
    def execute(
        self: SQLContext[DataFrame], query: str, *, eager: Literal[True]
    ) -> DataFrame: ...

    @overload
    def execute(
        self: SQLContext[LazyFrame], query: str, *, eager: None = ...
    ) -> LazyFrame: ...

    @overload
    def execute(
        self: SQLContext[LazyFrame], query: str, *, eager: Literal[False]
    ) -> LazyFrame: ...

    @overload
    def execute(
        self: SQLContext[LazyFrame], query: str, *, eager: Literal[True]
    ) -> DataFrame: ...

    @overload
    def execute(
        self, query: str, *, eager: bool | None = ...
    ) -> LazyFrame | DataFrame: ...

    def execute(
        self, query: str, *, eager: bool | None = None
    ) -> LazyFrame | DataFrame:
        """
        Parse the given SQL query and execute it against the registered frame data.

        Parameters
        ----------
        query
            A valid string SQL query.
        eager
            Execute the query immediately, returning a `DataFrame` instead of
            `LazyFrame`.
            If unset, the value of the init-time "eager" parameter will be used.
            Note that the query itself is always executed in lazy-mode; this
            parameter only impacts the type of the returned frame.

        Examples
        --------
        Declare frame data and register with a SQLContext:

        >>> df = pl.DataFrame(
        ...     data=[
        ...         ("The Godfather", 1972, 6_000_000, 134_821_952, 9.2),
        ...         ("The Dark Knight", 2008, 185_000_000, 533_316_061, 9.0),
        ...         ("Schindler's List", 1993, 22_000_000, 96_067_179, 8.9),
        ...         ("Pulp Fiction", 1994, 8_000_000, 107_930_000, 8.9),
        ...         ("The Shawshank Redemption", 1994, 25_000_000, 28_341_469, 9.3),
        ...     ],
        ...     schema=["title", "release_year", "budget", "gross", "imdb_score"],
        ...     orient="row",
        ... )
        >>> ctx = pl.SQLContext(films=df)

        Execute a SQL query against the registered frame data:

        >>> ctx.execute(
        ...     '''
        ...     SELECT title, release_year, imdb_score
        ...     FROM films
        ...     WHERE release_year > 1990
        ...     ORDER BY imdb_score DESC
        ...     ''',
        ...     eager=True,
        ... )
        shape: (4, 3)
        ┌──────────────────────────┬──────────────┬────────────┐
        │ title                    ┆ release_year ┆ imdb_score │
        │ ---                      ┆ ---          ┆ ---        │
        │ str                      ┆ i64          ┆ f64        │
        ╞══════════════════════════╪══════════════╪════════════╡
        │ The Shawshank Redemption ┆ 1994         ┆ 9.3        │
        │ The Dark Knight          ┆ 2008         ┆ 9.0        │
        │ Schindler's List         ┆ 1993         ┆ 8.9        │
        │ Pulp Fiction             ┆ 1994         ┆ 8.9        │
        └──────────────────────────┴──────────────┴────────────┘

        Execute a GROUP BY query:

        >>> ctx.execute(
        ...     '''
        ...     SELECT
        ...         MAX(release_year // 10) * 10 AS decade,
        ...         SUM(gross) AS total_gross,
        ...         COUNT(title) AS n_films,
        ...     FROM films
        ...     GROUP BY (release_year // 10) -- decade
        ...     ORDER BY total_gross DESC
        ...     ''',
        ...     eager=True,
        ... )
        shape: (3, 3)
        ┌────────┬─────────────┬─────────┐
        │ decade ┆ total_gross ┆ n_films │
        │ ---    ┆ ---         ┆ ---     │
        │ i64    ┆ i64         ┆ i64     │
        ╞════════╪═════════════╪═════════╡
        │ 2000   ┆ 533316061   ┆ 1       │
        │ 1990   ┆ 232338648   ┆ 3       │
        │ 1970   ┆ 134821952   ┆ 1       │
        └────────┴─────────────┴─────────┘
        """
        if self._catalogs:
            res = wrap_ldf(self._execute_with_catalogs(query))
        else:
            res = wrap_ldf(self._ctxt.execute(query))
        return res.collect() if eager or self._eager_execution else res

    def _execute_with_catalogs(self, query: str) -> PyLazyFrame:
        from polars.io.iceberg.functions import _scan_catalog_table

        if (ddl := self._ctxt.catalog_ddl(query)) is not None:
            return self._execute_catalog_ddl(ddl)._ldf

        tables = []
        identifier: list[str]
        for catalog_name, identifier in self._ctxt.catalog_tables(query):
            config, instance, storage_options = self._catalogs[catalog_name]
            full_name = [catalog_name, *identifier]
            lf = _scan_catalog_table(
                tuple(identifier),
                config,
                instance,
                storage_options=storage_options,
                display_name=".".join(full_name),
            )
            tables.append((full_name, lf._ldf))
        return self._ctxt.execute_with_catalog_tables(query, tables)

    def _execute_catalog_ddl(self, ddl: dict[str, Any]) -> LazyFrame:
        """Perform the catalog DDL that `catalog_ddl` returned."""
        config, instance, storage_options = self._catalogs[ddl["catalog"]]
        catalog = (
            instance
            if instance is not None
            else config.class_(config.name, **config.properties)
        )
        if ddl["kind"] == "drop_table":
            response = _drop_catalog_table(catalog, ddl)
        elif ddl["kind"] == "create_table":
            response = _create_catalog_table(
                catalog,
                ddl,
                lambda query: wrap_ldf(self._execute_with_catalogs(query)),
                storage_options,
            )
        else:
            msg = f"unsupported catalog statement {ddl['kind']!r}"
            raise NotImplementedError(msg)
        return DataFrame({"Response": [response]}).lazy()

    def register_catalog(
        self,
        name: str,
        catalog: pyiceberg.catalog.Catalog | IcebergCatalogConfig,
        *,
        default_namespace: str | Sequence[str] | None = None,
        default: bool = False,
        storage_options: StorageOptionsDict | None = None,
    ) -> Self:
        """
        Attach an Iceberg catalog, so that queries can read its tables by name.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A table name that starts with `name` refers to a table of this catalog
        (`name.namespace.table`). With `default=True`, so does a table name that
        starts with no attached catalog (`namespace.table`). A table name without a
        namespace gets `default_namespace`. Registered tables and CTEs take
        precedence over catalog tables, and an attached catalog takes precedence
        over a namespace of the same name in the default catalog.

        Catalog tables are scanned lazily, as with :func:`scan_iceberg`: a table is
        loaded when the query's schema is first needed, and a table that does not
        exist raises then. The rest of the table name is passed to the catalog as
        written; catalog identifiers are case-sensitive.

        Parameters
        ----------
        name
            The catalog's name in SQL. Registering the same name again replaces
            the catalog: `catalog`, `default_namespace` and `storage_options` take
            the new values, including their defaults when omitted, while the
            catalog stays the default if it was (see `default`). A name that
            differs from an attached catalog's only in case is an error.
        catalog
            A PyIceberg catalog, or an `IcebergCatalogConfig`.
        default_namespace
            The namespace of table names without one: a string (split on ".") or a
            sequence of namespace parts.
        default
            Make this catalog the default: table names that start with no attached
            catalog refer to it. At most one catalog is the default; a later default
            replaces it. `False` never removes the default: re-registering the
            default catalog keeps it the default, and re-registering another
            catalog leaves it. Only `unregister_catalog` leaves no default.
        storage_options
            Options for the storage of the catalog's tables, as for
            :func:`scan_iceberg`; `CREATE TABLE ... AS` also writes the rows with
            them. Pass the storage credentials here when a plan is
            serialized after its tables loaded: the serialized plan then reads the
            tables' metadata directly, with these options rather than the catalog's
            properties. `DROP TABLE ... PURGE` does not use them: the catalog
            deletes the files with its own properties.

        Notes
        -----
        Catalog tables can be created and dropped with SQL, at `execute()`. The
        table name resolves as in a query: with a default catalog,
        `CREATE TABLE tpch.t ...` creates `tpch.t` in it. `CREATE TEMP TABLE`
        creates a registered table instead.

        * `CREATE TABLE <name> (<column>, ...)` creates an empty table, and
          `CREATE TABLE <name> AS <query>` one holding the query's rows. The rows
          are committed together with the create, so a failed query creates no
          table; a catalog that cannot stage a create gets an empty table first,
          dropped again if filling it fails, unless another writer changed it. A
          registered table of that name is an error, as it would hide the new
          table. See :ref:`CREATE TABLE <create_table>` for the options.
        * `DROP TABLE [IF EXISTS] <name> [PURGE]` drops a registered table of that
          name, else the catalog table, permanently. `PURGE` also deletes the
          table's data files.
        * `TRUNCATE` and `DELETE` are not supported on catalog tables.

        Catalogs attached inside a `with SQLContext()` block are detached on exit,
        like the tables registered there. If that detaches the default catalog, the
        default catalog on entry is restored.

        A serialized query plan over catalog tables can contain the catalog's
        properties and the storage options, which may include credentials: treat
        serialized plans as secret.

        `SQLContext` is not thread-safe: do not register catalogs or tables while
        another thread executes a query on the same context.

        Examples
        --------
        >>> from pyiceberg.catalog import load_catalog  # doctest: +SKIP
        >>> ctx = pl.SQLContext()  # doctest: +SKIP
        >>> ctx.register_catalog(
        ...     "lake", load_catalog("lake"), default_namespace="tpch", default=True
        ... )  # doctest: +SKIP
        >>> ctx.execute("SELECT COUNT(*) FROM lineitem")  # doctest: +SKIP

        Create a catalog table holding a query's rows, then drop it:

        >>> ctx.execute(
        ...     "CREATE TABLE big_orders AS SELECT * FROM orders WHERE amount > 1000"
        ... )  # doctest: +SKIP
        >>> ctx.execute("DROP TABLE big_orders")  # doctest: +SKIP
        """
        from polars.io.iceberg import IcebergCatalogConfig

        issue_unstable_warning("`SQLContext.register_catalog` is considered unstable.")
        pyiceberg_catalog = _pyiceberg_catalog_class()
        if not isinstance(catalog, IcebergCatalogConfig) and not (
            pyiceberg_catalog is not None and isinstance(catalog, pyiceberg_catalog)
        ):
            msg = (
                "`register_catalog` takes a PyIceberg catalog or an "
                f"`IcebergCatalogConfig`; found {qualified_type_name(catalog)!r}"
            )
            raise TypeError(msg)
        config, instance = (
            IcebergCatalogConfig._from_api_parameter_or_environment_default(
                catalog, fn_name="register_catalog"
            )
        )
        self._ctxt.register_catalog(name, _namespace_parts(default_namespace), default)
        self._catalogs[name] = (config, instance, storage_options)
        return self

    def unregister_catalog(self, name: str) -> Self:
        """
        Detach a catalog attached with :meth:`register_catalog`.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        name
            The catalog's name (matched ignoring case if no name matches exactly).
        """
        issue_unstable_warning(
            "`SQLContext.unregister_catalog` is considered unstable."
        )
        if (removed := self._ctxt.unregister_catalog(name)) is not None:
            del self._catalogs[removed]
        return self

    def register(self, name: str, frame: CompatibleFrameType | None) -> Self:
        """
        Register a single frame as a table, using the given name.

        Parameters
        ----------
        name
            Name of the table.
        frame
            eager/lazy frame to associate with this table name.

        See Also
        --------
        register_globals
        register_many
        unregister

        Examples
        --------
        >>> df = pl.DataFrame({"hello": ["world"]})
        >>> ctx = pl.SQLContext()
        >>> ctx.register("frame_data", df).execute("SELECT * FROM frame_data").collect()
        shape: (1, 1)
        ┌───────┐
        │ hello │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ world │
        └───────┘
        """
        frame = LazyFrame() if frame is None else _ensure_lazyframe(frame)
        self._ctxt.register(name, frame._ldf)
        return self

    def register_globals(
        self, n: int | None = None, *, all_compatible: bool = True
    ) -> Self:
        """
        Register all frames (lazy or eager) found in the current globals scope.

        Automatically maps variable names to table names.

        See Also
        --------
        register
        register_many
        unregister

        Parameters
        ----------
        n
            Register only the most recent "n" frames.
        all_compatible
            Control whether we *also* register pandas DataFrame, Series, and
            pyarrow Table and RecordBatch objects. If False, only Polars
            classes are registered with the SQL engine.

        Examples
        --------
        >>> df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})
        >>> df2 = pl.DataFrame({"a": [2, 3, 4], "c": ["t", "w", "v"]})

        Register frames directly from variables found in the current globals scope:

        >>> ctx = pl.SQLContext(register_globals=True)
        >>> ctx.tables()
        ['df1', 'df2']

        Query using the register variable/frame names

        >>> ctx.execute(
        ...     "SELECT a, b, c FROM df1 LEFT JOIN df2 USING (a) ORDER BY a DESC"
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
        frames = _get_frame_locals(all_compatible=all_compatible, n_objects=n)
        return self.register_many(frames=frames)

    def register_many(
        self,
        frames: Mapping[str, CompatibleFrameType | None] | None = None,
        **named_frames: CompatibleFrameType | None,
    ) -> Self:
        """
        Register multiple eager/lazy frames as tables, using the associated names.

        Parameters
        ----------
        frames
            A `{name:frame, ...}` mapping.
        **named_frames
            Named eager/lazy frames, provided as kwargs.

        See Also
        --------
        register
        register_globals
        unregister

        Examples
        --------
        >>> lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": ["m", "n", "o"]})
        >>> lf2 = pl.LazyFrame({"a": [2, 3, 4], "c": ["p", "q", "r"]})
        >>> lf3 = pl.LazyFrame({"a": [3, 4, 5], "b": ["s", "t", "u"]})
        >>> lf4 = pl.LazyFrame({"a": [4, 5, 6], "c": ["v", "w", "x"]})

        Register multiple frames at once, either by passing in as a dict...

        >>> ctx = pl.SQLContext().register_many({"tbl1": lf1, "tbl2": lf2})
        >>> ctx.tables()
        ['tbl1', 'tbl2']

        ...or using keyword args:

        >>> ctx.register_many(tbl3=lf3, tbl4=lf4).tables()
        ['tbl1', 'tbl2', 'tbl3', 'tbl4']
        """
        frames = dict(frames or {})
        frames.update(named_frames)
        for name, frame in frames.items():
            self.register(name, frame)
        return self

    def unregister(self, names: str | Collection[str]) -> Self:
        """
        Unregister one or more eager/lazy frames by name.

        Parameters
        ----------
        names
            Names of the tables to unregister.

        Notes
        -----
        You can also control table registration lifetime by using `SQLContext` as a
        context manager; this can often be more useful when such control is wanted:

        >>> df0 = pl.DataFrame({"colx": [0, 1, 2]})
        >>> df1 = pl.DataFrame({"colx": [1, 2, 3]})
        >>> df2 = pl.DataFrame({"colx": [2, 3, 4]})

        Frames registered in-scope are automatically unregistered on scope-exit. Note
        that frames registered on construction will persist through subsequent scopes.

        >>> # register one frame at construction time, and the other two in-scope
        >>> with pl.SQLContext(tbl0=df0) as ctx:
        ...     ctx.register_many(tbl1=df1, tbl2=df2).tables()
        ['tbl0', 'tbl1', 'tbl2']

        After scope exit, none of the tables registered in-scope remain:

        >>> ctx.tables()
        ['tbl0']

        See Also
        --------
        register
        register_globals
        register_many

        Examples
        --------
        >>> df0 = pl.DataFrame({"ints": [9, 8, 7, 6, 5]})
        >>> lf1 = pl.LazyFrame({"text": ["a", "b", "c"]})
        >>> lf2 = pl.LazyFrame({"misc": ["testing1234"]})

        Register with a SQLContext object:

        >>> ctx = pl.SQLContext(test1=df0, test2=lf1, test3=lf2)
        >>> ctx.tables()
        ['test1', 'test2', 'test3']

        Unregister one or more of the tables:

        >>> ctx.unregister(["test1", "test3"]).tables()
        ['test2']
        >>> ctx.unregister("test2").tables()
        []
        """
        if isinstance(names, str):
            names = [names]
        for nm in names:
            self._ctxt.unregister(nm)
        return self

    def tables(self) -> list[str]:
        """
        Return a list of the registered table names.

        Notes
        -----
        The :meth:`tables` method will return the same values as the
        "SHOW TABLES" SQL statement, but as a list instead of a frame.

        Executing as SQL:

        >>> frame_data = pl.DataFrame({"hello": ["world"]})
        >>> ctx = pl.SQLContext(hello_world=frame_data)
        >>> ctx.execute("SHOW TABLES", eager=True)
        shape: (1, 1)
        ┌─────────────┐
        │ name        │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ hello_world │
        └─────────────┘

        Calling the method:

        >>> ctx.tables()
        ['hello_world']

        Examples
        --------
        >>> df1 = pl.DataFrame({"hello": ["world"]})
        >>> df2 = pl.DataFrame({"foo": ["bar", "baz"]})
        >>> ctx = pl.SQLContext(hello_data=df1, foo_bar=df2)
        >>> ctx.tables()
        ['foo_bar', 'hello_data']
        """
        return sorted(self._ctxt.get_tables())


def _pyiceberg_catalog_class() -> type | None:
    """The PyIceberg catalog base class, if PyIceberg is installed."""
    if importlib.util.find_spec("pyiceberg") is None:
        return None
    import pyiceberg.catalog

    return pyiceberg.catalog.Catalog


def _drop_catalog_table(catalog: pyiceberg.catalog.Catalog, ddl: dict[str, Any]) -> str:
    """Drop the catalog table of a `DROP TABLE`, returning the response."""
    from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError

    identifier: tuple[str, ...] = tuple(ddl["identifier"])
    purge = ddl["purge"]
    full_name = _display_name([ddl["catalog"], *identifier])
    response = f"DROP TABLE {full_name}" + (" PURGE" if purge else "")
    try:
        if purge:
            catalog.purge_table(identifier)
        else:
            catalog.drop_table(identifier)
    except (NoSuchTableError, NoSuchNamespaceError) as error:
        if not ddl["if_exists"]:
            msg = (
                f"table '{ddl['name']}' does not exist: catalog table "
                f"'{_display_name(identifier)}' could not be dropped ({error}); "
                "catalog identifiers are case-sensitive"
            )
            raise NoSuchTableError(msg) from error
    return response


def _create_catalog_table(
    catalog: pyiceberg.catalog.Catalog,
    ddl: dict[str, Any],
    query_frame: Callable[[str], LazyFrame],
    storage_options: StorageOptionsDict | None,
) -> str:
    """Create the catalog table of a `CREATE TABLE`, returning the response."""
    from pyiceberg.exceptions import (
        CommitStateUnknownException,
        NoSuchNamespaceError,
        TableAlreadyExistsError,
    )

    from polars.io.iceberg._sink import _create_table_schema
    from polars.schema import Schema

    identifier: tuple[str, ...] = tuple(ddl["identifier"])
    name, table_name = ddl["name"], _display_name(identifier)
    response = f"CREATE TABLE {_display_name([ddl['catalog'], *identifier])}"
    # a query is planned, but not run, when the table exists
    lf = None if ddl["query"] is None else query_frame(ddl["query"])
    schema, casts = _create_table_schema(
        Schema(ddl["schema"]) if lf is None else lf.collect_schema(),
        required=ddl["required"],
        context="CREATE TABLE",
    )
    # checked first, as some catalogs report other errors before an existing table
    if catalog.table_exists(identifier):
        if ddl["if_not_exists"]:
            return response
        msg = (
            f"table '{name}' already exists: catalog table '{table_name}' "
            "could not be created"
        )
        raise TableAlreadyExistsError(msg)
    namespace = identifier[:-1]
    if lf is not None and casts:
        lf = lf.cast(casts)  # type: ignore[arg-type]
    created = None
    try:
        if lf is None:
            catalog.create_table(
                identifier,
                schema=schema,
                location=ddl["location"],
                properties=ddl["properties"],
            )
        else:
            created = _create_filled_table(
                catalog, identifier, schema, lf, ddl, storage_options
            )
    except Exception as error:
        if _created_meanwhile(catalog, identifier, error):
            if ddl["if_not_exists"]:
                return response
            msg = (
                f"table '{name}' already exists: catalog table '{table_name}' "
                f"could not be created ({error})"
            )
            raise TableAlreadyExistsError(msg) from error
        if isinstance(error, CommitStateUnknownException):
            error.add_note(
                f"table '{name}' may not have been created: catalog table '{table_name}'"
            )
            raise
        prefix = f"table '{name}' could not be created"
        # not every catalog raises NoSuchNamespaceError for a missing namespace
        if isinstance(error, NoSuchNamespaceError) or not _namespace_exists(
            catalog, namespace
        ):
            msg = (
                f"{prefix}: namespace '{_display_name(namespace)}' does not exist in "
                f"catalog '{ddl['catalog']}' ({error}); catalog identifiers are "
                "case-sensitive"
            )
            raise NoSuchNamespaceError(msg) from error
        error.add_note(f"{prefix}: catalog table '{table_name}'")
        raise
    if created is not None:
        assert lf is not None
        # outside the create's error handling: the table exists, it is ours
        _fill_created_table(catalog, identifier, created, lf, storage_options, name)
    return response


def _create_filled_table(
    catalog: pyiceberg.catalog.Catalog,
    identifier: tuple[str, ...],
    schema: pyiceberg.schema.Schema,
    lf: LazyFrame,
    ddl: dict[str, Any],
    storage_options: StorageOptionsDict | None,
) -> pyiceberg.table.Table | None:
    """Create a catalog table holding the rows of `lf`, in one commit.

    A failed query or commit leaves no table, and its data files are deleted. If the
    catalog does not stage creates, the table is created empty and returned, to be
    filled by the caller.
    """
    from pyiceberg.exceptions import (
        BadRequestError,
        CommitStateUnknownException,
        NoSuchNamespaceError,
    )

    from polars.io.iceberg._sink import IcebergSinkState

    # a staged create does not check the namespace on every catalog
    if not _namespace_exists(catalog, identifier[:-1]):
        msg = f"Namespace does not exist: {identifier[:-1]}"
        raise NoSuchNamespaceError(msg)
    try:
        transaction = catalog.create_table_transaction(
            identifier,
            schema=schema,
            location=ddl["location"],
            properties=ddl["properties"],
        )
    # some REST catalogs reject staged creates (400, or 501 per the spec)
    except (BadRequestError, NotImplementedError):
        return catalog.create_table(
            identifier,
            schema=schema,
            location=ddl["location"],
            properties=ddl["properties"],
        )
    state = IcebergSinkState.new_create(transaction, storage_options=storage_options)
    try:
        state.attach_sink(lf).collect()
    except CommitStateUnknownException as error:
        location, _ = state.written_files_dir()
        error.add_note(
            f"the create may have landed; data files kept under '{location}'"
        )
        raise
    except Exception as error:
        if (remaining := state.remove_written_files()) is not None:
            error.add_note(f"data files may remain under '{remaining}'")
        raise
    return None


def _fill_created_table(
    catalog: pyiceberg.catalog.Catalog,
    identifier: tuple[str, ...],
    table: pyiceberg.table.Table,
    lf: LazyFrame,
    storage_options: StorageOptionsDict | None,
    name: str,
) -> None:
    """Append the rows of `lf` to the new, empty `table`, dropping it on failure."""
    from pyiceberg.exceptions import CommitStateUnknownException

    from polars.io.iceberg._sink import IcebergSinkState

    table_name = _display_name(identifier)
    created_location = table.metadata_location
    state = None
    try:
        state = IcebergSinkState.new(
            table, mode="append", storage_options=storage_options
        )
        state.attach_sink(lf).collect()
    except CommitStateUnknownException as error:
        assert state is not None
        location, _ = state.written_files_dir()
        error.add_note(f"the rows may have landed; data files kept under '{location}'")
        error.add_note(
            f"table '{name}' was created, its rows may be missing: catalog table "
            f"'{table_name}'"
        )
        raise
    except Exception as error:
        if state is not None and (remaining := state.remove_written_files()):
            error.add_note(f"data files may remain under '{remaining}'")
        if _drop_if_unchanged(catalog, identifier, created_location):
            error.add_note(
                f"table '{name}' could not be created: catalog table '{table_name}'"
            )
        else:
            error.add_note(
                f"table '{name}' could not be filled: catalog table '{table_name}' "
                "was left in the catalog"
            )
        raise


def _drop_if_unchanged(
    catalog: pyiceberg.catalog.Catalog,
    identifier: tuple[str, ...],
    metadata_location: str,
) -> bool:
    """Drop a table only if no other writer changed it since `metadata_location`.

    Iceberg has no conditional drop: a write landing just before the drop is lost.
    """
    try:
        current = catalog.load_table(identifier)
        if (
            current.metadata_location != metadata_location
            or current.current_snapshot() is not None
        ):
            return False
        catalog.drop_table(identifier)
    except Exception:
        return False
    return True


def _created_meanwhile(
    catalog: pyiceberg.catalog.Catalog,
    identifier: tuple[str, ...],
    error: Exception,
) -> bool:
    """Whether a create failed because another writer created the table."""
    from pyiceberg.exceptions import CommitFailedException, TableAlreadyExistsError

    if isinstance(error, TableAlreadyExistsError):
        return True
    if not isinstance(error, CommitFailedException):
        return False
    try:
        return bool(catalog.table_exists(identifier))
    except Exception:
        return False


def _namespace_exists(
    catalog: pyiceberg.catalog.Catalog, namespace: Sequence[str]
) -> bool:
    """Whether the catalog has `namespace`; `True` if that can't be checked."""
    try:
        return bool(catalog.namespace_exists(tuple(namespace)))
    except Exception:
        return True


def _display_name(parts: Sequence[str]) -> str:
    """A table name for messages, with parts containing dots quoted."""
    return ".".join(f'"{part}"' if "." in part else part for part in parts)


def _namespace_parts(namespace: str | Sequence[str] | None) -> list[str] | None:
    """The parts of a namespace given as a dotted string or a sequence of parts."""
    if namespace is None:
        return None
    parts = namespace.split(".") if isinstance(namespace, str) else list(namespace)
    if not parts or not all(parts):
        msg = f"invalid `default_namespace`: {namespace!r}"
        raise ValueError(msg)
    return parts
