from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from contextlib import suppress
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Final, cast

from polars._utils.various import parse_version, qualified_type_name
from polars.datatypes import N_INFER_DEFAULT
from polars.exceptions import (
    DuplicateError,
    ModuleUpgradeRequiredError,
    UnsuitableSQLError,
)
from polars.io.database._arrow_registry import ARROW_DRIVER_REGISTRY
from polars.io.database._cursor_proxies import (
    ODBCCursorProxy,
    OracleCursorProxy,
    SurrealDBCursorProxy,
)
from polars.io.database._inference import dtype_from_cursor_description
from polars.io.database._utils import _run_async

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from types import TracebackType
    from typing import Self

    import pyarrow as pa
    from sqlalchemy.sql.elements import TextClause
    from sqlalchemy.sql.expression import Selectable

    from polars import DataFrame
    from polars._typing import ConnectionOrCursor, Cursor, SchemaDict
    from polars.io.database._arrow_registry import ArrowDriverProperties

_INVALID_QUERY_TYPES: Final[set[str]] = {
    "ALTER",
    "ANALYZE",
    "CREATE",
    "DELETE",
    "DROP",
    "GRANT",
    "INSERT",
    "REPLACE",
    "REVOKE",
    "UPDATE",
    "UPSERT",
    "USE",
    "VACUUM",
}

# the Databricks SQLAlchemy dialect's driver ("databricks-sql-python" before
# databricks-sqlalchemy 2.x)
_DATABRICKS_ALCHEMY_DRIVERS: Final = frozenset({"databricks", "databricks-sql-python"})


def _has_text(dtype: Any) -> bool:
    """Whether a dtype holds a String, also nested or in an extension."""
    from polars.datatypes import Array, BaseExtension, List, Map, String, Struct

    if dtype == String:
        return True
    if isinstance(dtype, BaseExtension):
        return _has_text(dtype.ext_storage())
    if isinstance(dtype, (List, Array)):
        return _has_text(dtype.inner)
    if isinstance(dtype, Map):
        return _has_text(dtype.key) or _has_text(dtype.value)
    if isinstance(dtype, Struct):
        return any(_has_text(f.dtype) for f in dtype.fields)
    return False


def _as_binary(dtype: Any) -> Any:
    """The physical dtype with Binary for each String, or None if it holds no String."""
    from polars.datatypes import Array, Binary, Field, List, String, Struct

    if dtype == String:
        return Binary
    if isinstance(dtype, (List, Array)):
        inner = _as_binary(dtype.inner)
        if inner is None:
            return None
        return List(inner) if isinstance(dtype, List) else Array(inner, dtype.size)
    if isinstance(dtype, Struct):
        fields = [_as_binary(f.dtype) for f in dtype.fields]
        if all(f is None for f in fields):
            return None
        return Struct(
            [
                Field(f.name, f.dtype if b is None else b)
                for f, b in zip(dtype.fields, fields, strict=True)
            ]
        )
    return None


def _check_utf8(df: DataFrame) -> DataFrame:
    """
    The frame, if its text is valid UTF-8, also nested in lists, structs and maps.

    A driver that hands the bytes of a text value over as they are (the Databricks
    connector and ADBC drivers over a `STRING` made of `unhex('FF')`) would otherwise
    reach the caller as a String that breaks its contract (memory unsafety in string
    operations).
    """
    from polars.exceptions import ComputeError

    for s in df.iter_columns():
        if not _has_text(s.dtype):
            continue
        # `to_physical` selects the column by name, which may be a pattern
        physical = s.rename("s").to_physical()
        try:
            physical.cast(_as_binary(physical.dtype)).cast(physical.dtype)
        except Exception as e:
            msg = (
                "the database returned text that isn't valid UTF-8 (a STRING made "
                "of other bytes); read the column as BINARY with "
                f"CAST(<column> AS BINARY) and decode it in Polars ({e})"
            )
            raise ComputeError(msg) from e
    return df


# the drivers that pass on the bytes of a text value without checking them
_UNCHECKED_TEXT_DRIVERS: Final = frozenset({"adbc", "databricks"})


def _close_after_frames(
    frames: Any, *, close: Callable[[], None]
) -> Generator[DataFrame]:
    """
    Yield the frames, then call `close` (also on an early stop or an error).

    The query has already run when this is returned. A generator that never started
    skips its `finally` when closed or collected, so it is primed here: it waits at
    a first `yield` inside the `try`, and `close()` before the first `next()` works.
    """

    def gen() -> Generator[DataFrame]:
        try:
            yield None  # type: ignore[misc]  # (priming)
            yield from frames
        finally:
            close()

    g = gen()
    next(g)
    return g


def _trino_field_names(value: tuple[Any, ...]) -> list[str | None]:
    attrs = vars(value)
    annotations = attrs.get("__annotations__")
    names_attr = attrs.get("_names")
    for metadata, names in (
        (
            annotations,
            annotations.get("names") if isinstance(annotations, dict) else None,
        ),
        (names_attr, names_attr),
    ):
        # Trino sets named fields as attributes. A field can replace a
        # metadata container or the whole instance dictionary (__dict__).
        if (
            isinstance(names, list)
            and len(names) == len(value)
            and all(name is None or isinstance(name, str) for name in names)
            and not any(
                metadata is field_value or attrs is field_value for field_value in value
            )
        ):
            return names
    msg = "Cannot determine field names for a Trino ROW: missing or shadowed driver metadata"
    raise ValueError(msg)


def _convert_trino_value(value: Any, row_type: type[tuple[Any, ...]]) -> Any:
    """Recursively convert trino `NamedRowTuple` values to dicts (struct-like)."""
    if isinstance(value, row_type):
        names = _trino_field_names(value)
        counts = Counter(names)
        reserved = {name for name, count in counts.items() if count == 1}
        fields: dict[str, Any] = {}
        fallback = 0
        for idx, (name, field_value) in enumerate(zip(names, value, strict=True)):
            if name is None or counts[name] > 1:
                fallback = max(fallback, idx)
                while (name := f"field_{fallback}") in reserved:
                    fallback += 1
                fallback += 1
            fields[name] = _convert_trino_value(field_value, row_type)
        return fields
    elif isinstance(value, list):
        return [_convert_trino_value(v, row_type) for v in value]
    elif isinstance(value, dict):
        return {k: _convert_trino_value(v, row_type) for k, v in value.items()}
    return value


def _convert_trino_rows(
    rows: Iterable[Sequence[Any] | Mapping[str, Any]],
    row_type: type[tuple[Any, ...]],
    row_columns: list[tuple[int, str]],
) -> list[Sequence[Any] | Mapping[str, Any]]:
    """Convert Trino ROW values in the columns identified by the cursor."""
    converted: list[Sequence[Any] | Mapping[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            mapping = dict(row)
            for _, name in row_columns:
                if name in mapping:
                    mapping[name] = _convert_trino_value(mapping[name], row_type)
            converted.append(mapping)
        else:
            values = list(row)
            for idx, _ in row_columns:
                values[idx] = _convert_trino_value(values[idx], row_type)
            converted.append(values)
    return converted


class ConnectionExecutor:
    """Abstraction for querying databases with user-supplied connection objects."""

    # indicate if we can/should close the cursor on scope exit. note that we
    # should never close the underlying connection, or a user-supplied cursor.
    can_close_cursor: bool = False
    # a SQLAlchemy result whose cursor is read as Arrow (Databricks): ours to close
    _owned_result: Any = None

    def __init__(self, connection: ConnectionOrCursor) -> None:
        self.driver_name = (
            "arrow_odbc_proxy"
            if isinstance(connection, ODBCCursorProxy)
            # compiled driver modules can have a leading underscore (eg: "_duckdb")
            else type(connection).__module__.split(".", 1)[0].lower().lstrip("_")
        )
        if self.driver_name == "surrealdb":
            connection = SurrealDBCursorProxy(client=connection)
        elif (
            # note: only the proxy can serve Arrow data (see `_arrow_registry`)
            self.driver_name == "oracledb"
            and OracleCursorProxy.supports_arrow(connection)
        ):
            connection = OracleCursorProxy(connection)
            self.driver_name = "oracledb_proxy"

        self.cursor = self._normalise_cursor(connection)
        self.result: Any = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # if we created it and are finished with it, we can close it
        # (but NOT a connection or cursor of the caller's)
        if (cleanup := self._take_cleanup()) is not None:
            cleanup()

    def _take_cleanup(self) -> Callable[[], None] | None:
        """
        Take over closing what the executor opened; the returned callable closes it.

        That is a SQLAlchemy result read as Arrow (`_owned_result`), then the cursor if
        it is ours (`can_close_cursor`). Once taken, `__exit__` leaves both to the
        callable, so a batch iterator can close them after its last batch.
        """
        result = self._owned_result
        cursor = self.cursor if self.can_close_cursor else None
        if result is None and cursor is None:
            return None
        self._owned_result = None
        self.can_close_cursor = False
        close_async = cursor is not None and self._is_alchemy_async(cursor)

        def cleanup() -> None:
            try:
                if result is not None:
                    result.close()
            finally:
                if close_async:
                    _run_async(self._close_async_cursor(cursor))
                elif cursor is not None and hasattr(cursor, "close"):
                    cursor.close()

        return cleanup

    def __repr__(self) -> str:
        return f"<{type(self).__name__} module={self.driver_name!r}>"

    @staticmethod
    def _apply_overrides(df: DataFrame, schema_overrides: SchemaDict) -> DataFrame:
        """Apply schema overrides to a DataFrame."""
        from polars._utils.construction.dataframe import cast_column_by_field_names

        existing_schema = df.schema
        if cast_cols := [
            cast_column_by_field_names(col, existing_schema[col], dtype, strict=True)
            for col, dtype in schema_overrides.items()
            if col in existing_schema and dtype != existing_schema[col]
        ]:
            df = df.with_columns(cast_cols)
        return df

    @staticmethod
    async def _close_async_cursor(cursor: Any) -> None:
        from sqlalchemy.ext.asyncio.exc import AsyncContextNotStarted

        with suppress(AsyncContextNotStarted):
            await cursor.close()

    @staticmethod
    def _check_module_version(module_name: str, minimum_version: str) -> None:
        """Check the module version against a minimum required version."""
        mod = __import__(module_name)
        with suppress(AttributeError):
            module_version: tuple[int, ...] | None = None
            for version_attr in ("__version__", "version"):
                if isinstance(ver := getattr(mod, version_attr, None), str):
                    module_version = parse_version(ver)
                    break
            if module_version and module_version < parse_version(minimum_version):
                msg = f"`read_database` queries require at least {module_name} version {minimum_version}"
                raise ModuleUpgradeRequiredError(msg)

    @staticmethod
    def _duckdb_has_lazy_relations() -> bool:
        """Check if DuckDB is version 2.0 or later."""
        try:
            ConnectionExecutor._check_module_version("duckdb", "2.0.0")
        except ModuleUpgradeRequiredError:
            return False
        return True

    def _fetch_arrow(
        self,
        driver_properties: ArrowDriverProperties,
        *,
        batch_size: int | None,
        iter_batches: bool,
    ) -> Iterable[pa.RecordBatch]:
        """Return Arrow data as an iterable of one or more RecordBatches or Tables."""
        # note: the fetch method is called here, not when iterating, as a DuckDB
        # relation runs its query (and needs an open cursor) at that call
        fetch_batches = driver_properties["fetch_batches"]
        if not iter_batches or fetch_batches is None:
            fetch_method = driver_properties["fetch_all"]
            res = getattr(self.result, fetch_method)()
            return res if isinstance(res, Iterable) else [res]

        size = [batch_size] if driver_properties["exact_batch_size"] else []
        fetchmany_arrow = getattr(self.result, fetch_batches)
        if not driver_properties["repeat_batch_calls"]:
            return fetchmany_arrow(*size)  # type: ignore[no-any-return]
        return self._repeat_fetch(fetchmany_arrow, size)

    @staticmethod
    def _repeat_fetch(
        fetchmany_arrow: Any, size: list[int | None]
    ) -> Iterator[pa.RecordBatch]:
        """Call a batch fetch method until it returns no more data."""
        while True:
            arrow = fetchmany_arrow(*size)
            if not arrow:
                break
            yield arrow

    @staticmethod
    def _fetchall_rows(
        result: Cursor, *, is_alchemy: bool
    ) -> Iterable[Sequence[Any] | Mapping[str, Any]]:
        """Fetch row data in a single call, returning the complete result set."""
        rows = result.fetchall()
        return (
            rows
            if rows and (is_alchemy or isinstance(rows[0], (list, tuple, Mapping)))
            else [tuple(row) for row in rows]
        )

    def _fetchmany_rows(
        self, result: Cursor, *, batch_size: int | None, is_alchemy: bool
    ) -> Iterable[Iterable[Sequence[Any] | Mapping[str, Any]]]:
        """Fetch row data incrementally, yielding over the complete result set."""
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            elif is_alchemy or isinstance(rows[0], (list, tuple, Mapping)):
                yield rows
            else:
                yield [tuple(row) for row in rows]

    def _from_arrow(
        self,
        *,
        batch_size: int | None,
        iter_batches: bool,
        schema_overrides: SchemaDict | None,
        infer_schema_length: int | None,  # noqa: ARG002
    ) -> DataFrame | Iterator[DataFrame] | None:
        """Return resultset data in Arrow format for frame init."""
        from polars import DataFrame

        try:
            # all ADBC drivers have the same method names
            driver = (
                "adbc" if self.driver_name.startswith("adbc_") else self.driver_name
            )
            driver_properties_list = ARROW_DRIVER_REGISTRY.get(driver, [])
            for i, driver_properties in enumerate(driver_properties_list, start=1):
                if ver := driver_properties["minimum_version"]:
                    # for ADBC drivers, the minimum version constraint is on the driver
                    # manager rather than the driver itself
                    driver_to_check = (
                        "adbc_driver_manager" if driver == "adbc" else self.driver_name
                    )
                    # if the minimum version constraint is not met, try additional
                    # driver properties with lower constraints
                    try:
                        self._check_module_version(driver_to_check, ver)
                    except ModuleUpgradeRequiredError:
                        if i < len(driver_properties_list):
                            continue
                        raise

                if iter_batches and (
                    driver_properties["exact_batch_size"] and not batch_size
                ):
                    msg = (
                        f"Cannot set `iter_batches` for {self.driver_name} "
                        "without also setting a non-zero `batch_size`"
                    )
                    raise ValueError(msg)  # noqa: TRY301

                check = driver in _UNCHECKED_TEXT_DRIVERS
                frames = (
                    self._apply_overrides(
                        _check_utf8(df) if check else df, (schema_overrides or {})
                    )
                    for df in (
                        batch if isinstance(batch, DataFrame) else DataFrame(batch)
                        for batch in self._fetch_arrow(
                            driver_properties,
                            iter_batches=iter_batches,
                            batch_size=batch_size,
                        )
                    )
                )
                return frames if iter_batches else next(frames)  # type: ignore[arg-type,return-value]
        except Exception as err:
            # eg: valid turbodbc/snowflake connection, but no arrow support
            # compiled in to the underlying driver (or on this connection)
            arrow_not_supported = (
                "does not support Apache Arrow",
                "Apache Arrow format is not supported",
            )
            if not any(e in str(err) for e in arrow_not_supported):
                raise

        return None

    def _from_rows(
        self,
        *,
        batch_size: int | None,
        iter_batches: bool,
        schema_overrides: SchemaDict | None,
        infer_schema_length: int | None,
    ) -> DataFrame | Iterator[DataFrame] | None:
        """Return resultset data row-wise for frame init."""
        from polars import DataFrame

        if iter_batches and not batch_size:
            msg = (
                "Cannot set `iter_batches` without also setting a non-zero `batch_size`"
            )
            raise ValueError(msg)

        if is_async := isinstance(original_result := self.result, Coroutine):
            self.result = _run_async(self.result)
        try:
            if hasattr(self.result, "fetchall"):
                if is_alchemy := (self.driver_name == "sqlalchemy"):
                    if hasattr(self.result, "cursor"):
                        cursor_desc = [
                            (d[0], d[1:]) for d in self.result.cursor.description
                        ]
                    elif hasattr(self.result, "_metadata"):
                        cursor_desc = [(k, None) for k in self.result._metadata.keys]
                    else:
                        msg = f"Unable to determine metadata from query result; {self.result!r}"
                        raise ValueError(msg)

                elif hasattr(self.result, "description"):
                    cursor_desc = [(d[0], d[1:]) for d in self.result.description]
                else:
                    cursor_desc = []

                schema_overrides = self._inject_type_overrides(
                    description=cursor_desc,
                    schema_overrides=(schema_overrides or {}),
                )
                result_columns = [nm for nm, _ in cursor_desc]
                row_batches: Iterable[Iterable[Sequence[Any] | Mapping[str, Any]]] = (
                    self._fetchmany_rows(
                        self.result,
                        batch_size=batch_size,
                        is_alchemy=is_alchemy,
                    )
                    if iter_batches
                    else [self._fetchall_rows(self.result, is_alchemy=is_alchemy)]
                )
                if trino_row_type := self._trino_row_type():
                    row_columns = [
                        (idx, name)
                        for idx, (name, desc) in enumerate(cursor_desc)
                        if isinstance(desc, tuple)
                        and desc
                        and isinstance(desc[0], str)
                        and re.search(r"\brow\s*\(", desc[0], flags=re.IGNORECASE)
                    ]
                    if row_columns:
                        row_batches = (
                            _convert_trino_rows(rows, trino_row_type, row_columns)
                            for rows in row_batches
                        )
                frames = (
                    DataFrame(
                        data=rows,
                        schema=result_columns or None,
                        schema_overrides=schema_overrides,
                        infer_schema_length=infer_schema_length,
                        orient="row",
                    )
                    for rows in row_batches
                )
                return frames if iter_batches else next(frames)  # type: ignore[arg-type]
            return None
        finally:
            if is_async:
                original_result.close()

    def _trino_row_type(self) -> type[tuple[Any, ...]] | None:
        """
        Return Trino's `NamedRowTuple` type if this is a Trino result.

        The trino driver returns `ROW` values as `NamedRowTuple`, a tuple subclass
        whose `__getattr__` returns `None` for unknown attributes; it therefore looks
        like a namedtuple (it appears to have `_fields`) without behaving as one,
        so these values need converting before frame init.
        """
        if (trino_types := sys.modules.get("trino.types")) is None:
            return None
        if self.driver_name == "sqlalchemy":
            dialect = getattr(getattr(self.result, "context", None), "dialect", None)
            if getattr(dialect, "name", None) != "trino":
                return None
        elif self.driver_name != "trino":
            cursor_type = getattr(sys.modules.get("trino.dbapi"), "Cursor", ())
            if not (
                isinstance(self.cursor, cursor_type)
                or isinstance(self.result, cursor_type)
            ):
                return None
        return getattr(trino_types, "NamedRowTuple", None)

    def _inject_type_overrides(
        self,
        description: list[tuple[str, Any]],
        schema_overrides: SchemaDict,
    ) -> SchemaDict:
        """
        Attempt basic dtype inference from a cursor description.

        Notes
        -----
        This is limited; the `type_code` description attr may contain almost anything,
        from strings or python types to driver-specific codes, classes, enums, etc.
        We currently only do the additional inference from string/python type values.
        (Further refinement will require per-driver module knowledge and lookups).
        """
        dupe_check = set()
        for nm, desc in description:
            if nm in dupe_check:
                msg = f"column {nm!r} appears more than once in the query/result cursor"
                raise DuplicateError(msg)
            elif desc is not None and nm not in schema_overrides:
                dtype = dtype_from_cursor_description(desc)
                if dtype is not None:
                    schema_overrides[nm] = dtype  # type: ignore[index]
            dupe_check.add(nm)

        return schema_overrides

    @staticmethod
    def _is_alchemy_async(conn: Any) -> bool:
        """Check if the given connection is SQLAlchemy async."""
        try:
            from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

            if isinstance(conn, (AsyncConnection, AsyncSession)):
                return True

            from sqlalchemy.ext.asyncio import async_sessionmaker

            return isinstance(conn, async_sessionmaker)
        except ImportError:
            return False

    @staticmethod
    def _is_alchemy_engine(conn: Any) -> bool:
        """Check if the given connection is a SQLAlchemy Engine."""
        from sqlalchemy.engine import Engine

        if isinstance(conn, Engine):
            return True
        try:
            from sqlalchemy.ext.asyncio import AsyncEngine

            return isinstance(conn, AsyncEngine)
        except ImportError:
            return False

    @staticmethod
    def _is_alchemy_object(conn: Any) -> bool:
        """Check if the given connection is a SQLAlchemy object (of any kind)."""
        return type(conn).__module__.split(".", 1)[0] == "sqlalchemy"

    @staticmethod
    def _is_alchemy_session(conn: Any) -> bool:
        """Check if the given connection is a SQLAlchemy Session object."""
        from sqlalchemy.orm import Session, sessionmaker

        if isinstance(conn, (Session, sessionmaker)):
            return True

        try:
            from sqlalchemy.ext.asyncio import AsyncSession

            if isinstance(conn, AsyncSession):
                return True

            from sqlalchemy.ext.asyncio import async_sessionmaker

            return isinstance(conn, async_sessionmaker)
        except ImportError:
            # SQLAlchemy 2.1 imports greenlet while loading the asyncio extra.
            # A synchronous connection must still be usable without it.
            return False

    @staticmethod
    def _is_alchemy_result(result: Any) -> bool:
        """Check if the given result is a SQLAlchemy Result object."""
        try:
            from sqlalchemy.engine import CursorResult

            if isinstance(result, CursorResult):
                return True

            from sqlalchemy.ext.asyncio import AsyncResult

            return isinstance(result, AsyncResult)
        except ImportError:
            return False

    def _normalise_cursor(self, conn: Any) -> Cursor:
        """Normalise a connection object such that we have the query executor."""
        if self.driver_name == "sqlalchemy":
            if self._is_alchemy_session(conn):
                return conn
            else:
                # where possible, use the raw connection to access arrow integration
                if conn.engine.driver == "duckdb_engine":
                    self.driver_name = "duckdb"
                    return conn
                elif self._is_alchemy_engine(conn):
                    # note: if we create it, we can close it
                    self.can_close_cursor = True
                    return conn.connect()
                else:
                    return conn

        elif hasattr(conn, "cursor"):
            # connection has a dedicated cursor; prefer over direct execute
            cursor = (
                cast("Cursor", cursor()) if callable(cursor := conn.cursor) else cursor
            )
            self.can_close_cursor = True
            return cursor

        elif hasattr(conn, "execute"):
            # can execute directly (given cursor, sqlalchemy connection, etc)
            return conn

        msg = (
            f"Unrecognised connection type {qualified_type_name(conn)!r}; no "
            "'execute' or 'cursor' method"
        )
        raise TypeError(msg)

    def _oracle_arrow_proxy(self) -> OracleCursorProxy | None:
        """Return an Arrow proxy for a SQLAlchemy "python-oracledb" connection."""
        if not self._is_alchemy_async(self.cursor):
            driver_name = getattr(getattr(self.cursor, "engine", None), "driver", None)
            if driver_name == "oracledb" and OracleCursorProxy.supports_arrow(
                # note: proxy the established DBAPI connection so the query
                # is executed in the caller's own session/transaction
                raw_conn := getattr(
                    getattr(self.cursor, "connection", None), "driver_connection", None
                )
            ):
                return OracleCursorProxy(raw_conn)
        return None

    async def _sqlalchemy_async_execute(self, query: TextClause, **options: Any) -> Any:
        """Execute a query using an async SQLAlchemy connection."""
        is_session = self._is_alchemy_session(self.cursor)
        cursor = self.cursor.begin() if is_session else self.cursor  # type: ignore[attr-defined]

        # check if connection is already started (eg: user awaited `engine.connect()`);
        # if so, use it directly without entering the context manager again
        if getattr(cursor, "sync_connection", None) is not None:
            return await cursor.execute(query, **options)

        async with cursor as conn:  # type: ignore[union-attr]
            if is_session and not hasattr(conn, "execute"):
                conn = conn.session
            result = await conn.execute(query, **options)
            return result

    def _sqlalchemy_setup(
        self, query: str | TextClause | Selectable, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any], str | TextClause | Selectable]:
        """Prepare a query for execution using a SQLAlchemy connection."""
        from sqlalchemy.orm import Session
        from sqlalchemy.sql import text
        from sqlalchemy.sql.elements import TextClause

        param_key = "parameters"
        cursor_execute = None
        if (
            isinstance(self.cursor, Session)
            and "parameters" in options
            and "params" not in options
        ):
            options = options.copy()
            options["params"] = options.pop("parameters")
            param_key = "params"

        params = options.get(param_key)
        is_async = self._is_alchemy_async(self.cursor)
        if (
            not is_async
            and hasattr(self.cursor, "exec_driver_sql")
            and (
                isinstance(params, Sequence)
                # The legacy Databricks dialect accepts native DBAPI placeholders.
                or (
                    isinstance(query, str)
                    and self.cursor.engine.driver == "databricks-sql-python"  # type: ignore[attr-defined]
                )
            )
        ):
            cursor_execute = self.cursor.exec_driver_sql
            if isinstance(query, TextClause):
                query = str(query)
            if isinstance(params, list) and not all(
                isinstance(p, (dict, tuple)) for p in params
            ):
                options[param_key] = tuple(params)

        elif isinstance(query, str):
            query = text(query)

        if cursor_execute is None:
            cursor_execute = (
                self._sqlalchemy_async_execute if is_async else self.cursor.execute
            )
        return cursor_execute, options, query

    def execute(
        self,
        query: str | TextClause | Selectable,
        *,
        options: dict[str, Any] | None = None,
        select_queries_only: bool = True,
    ) -> Self:
        """Execute a query and reference the result set."""
        if select_queries_only and isinstance(query, str):
            q = re.search(r"\w{3,}", re.sub(r"/\*(.|[\r\n])*?\*/", "", query))
            if (query_type := "" if not q else q.group(0)) in _INVALID_QUERY_TYPES:
                msg = f"{query_type} statements are not valid 'read' queries"
                raise UnsuitableSQLError(msg)

        options = options or {}

        databricks_arrow = False
        if self._is_alchemy_object(self.cursor):
            databricks_arrow = (
                isinstance(query, str)
                and getattr(getattr(self.cursor, "engine", None), "driver", None)
                in _DATABRICKS_ALCHEMY_DRIVERS
            )
            # note: the Arrow "fetch_df_*" methods take raw SQL, so the Oracle fast-path
            # is only available for string queries; anything else must be compiled first
            if (
                isinstance(query, str)
                and (oracle_proxy := self._oracle_arrow_proxy()) is not None
            ):
                # note: `self.cursor` is deliberately left as the SQLAlchemy connection,
                # so that it remains open (and closeable) for the proxy's lifetime
                self.driver_name = "oracledb_proxy"
                self.result = oracle_proxy.execute(query, **options)
                return self

            cursor_execute, options, query = self._sqlalchemy_setup(query, options)
        elif (
            self.driver_name == "duckdb"
            and isinstance(query, str)
            and self._duckdb_has_lazy_relations()
        ):
            # Arrow output from a DuckDB relation avoids copying the result rows
            if "parameters" in options and "params" not in options:
                options = options.copy()
                options["params"] = options.pop("parameters")
            relation = self.cursor.sql(query, **options)  # type: ignore[attr-defined]
            self.result = self.cursor if relation is None else relation
            return self
        else:
            cursor_execute = self.cursor.execute

        # note: some cursor execute methods (eg: sqlite3) only take positional
        # params, hence the slightly convoluted resolution of the 'options' dict
        try:
            params = signature(cursor_execute).parameters
        except ValueError:
            params = {}  # type: ignore[assignment]

        if not options or any(
            p.kind
            in (
                Parameter.KEYWORD_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_KEYWORD,
            )
            for p in params.values()
        ):
            result = cursor_execute(query, **options)
        else:
            positional_options = (
                options[o] for o in (params or options) if (not options or o in options)
            )
            result = cursor_execute(query, *positional_options)

        # note: some cursors execute in-place, some access results via a property, and
        # some (eg: pymysql, MySQLdb) return the affected row count
        if result is None or result is True or type(result) is int:
            result = self.cursor
        if self.driver_name == "duckdb" and self._is_alchemy_result(result):
            result = result.cursor  # type: ignore[union-attr]
        elif (
            databricks_arrow
            and self._is_alchemy_result(result)
            and result.returns_rows  # type: ignore[union-attr]
            and hasattr(result.cursor, "fetchall_arrow")  # type: ignore[union-attr]
        ):
            # Arrow from the driver's cursor (rows lose nested values and the time
            # zone of a TIMESTAMP); the result is kept open until the read is done
            self.driver_name = "databricks"
            self._owned_result = result
            result = result.cursor  # type: ignore[union-attr]

        self.result = result
        return self

    def to_polars(
        self,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: SchemaDict | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Convert the result set to a DataFrame.

        Wherever possible we try to return arrow-native data directly; only
        fall back to initialising with row-level data if no other option.
        """
        if self.result is None:
            msg = "cannot return a frame before executing a query"
            raise RuntimeError(msg)

        for frame_init in (
            self._from_arrow,  # init from arrow-native data (where support exists)
            self._from_rows,  # row-wise fallback (sqlalchemy, dbapi2, pyodbc, etc)
        ):
            frame = frame_init(
                batch_size=batch_size,
                iter_batches=iter_batches,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_schema_length,
            )
            if frame is not None:
                if iter_batches and (cleanup := self._take_cleanup()) is not None:
                    frame = _close_after_frames(frame, close=cleanup)
                return frame

        msg = (
            f"Currently no support for {self.driver_name!r} connection {self.cursor!r}"
        )
        raise NotImplementedError(msg)
