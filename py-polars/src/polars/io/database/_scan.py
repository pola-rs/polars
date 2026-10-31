from __future__ import annotations

import contextlib
import operator
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from polars._utils.unstable import unstable
from polars._utils.various import qualified_type_name
from polars._utils.wrap import wrap_ldf
from polars.expr import Expr
from polars.schema import Schema

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyLazyFrame

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars import DataFrame, LazyFrame
    from polars._typing import SchemaDict

# An ODBC connection string names its driver or data source in any of its keys.
_ODBC_KEY = re.compile(r"(^|;)\s*(driver|dsn|filedsn)\s*=", re.IGNORECASE)
_URI = re.compile(r"^[A-Za-z][\w+.-]*://")
# file databases also take `scheme:path`
_FILE_URI = re.compile(r"^(sqlite|duckdb)(\+\w+)?:", re.IGNORECASE)


def _connection_kind(connection: Any) -> tuple[str, Any]:
    """Classify a connection as `uri`, `odbc`, `factory` or `live`."""
    if isinstance(connection, str):
        if _ODBC_KEY.search(connection):
            return "odbc", connection
        if _URI.match(connection) or _FILE_URI.match(connection):
            return "uri", connection
        # the string isn't echoed: it may hold a password
        msg = (
            "expected a connection URI ('scheme://…', e.g. 'postgresql://user@host/db') "
            "or an ODBC connection string ('Driver={…};…' or 'DSN=…;…'); "
            "the string given is neither"
        )
        raise ValueError(msg)

    module = type(connection).__module__
    if module.startswith("sqlalchemy"):
        if hasattr(connection, "sync_engine"):
            msg = (
                "scan_database can't use a SQLAlchemy AsyncEngine; pass a sync Engine, "
                "or a URL with a sync driver"
            )
            raise TypeError(msg)
        # a URL is a URI with a driver pin
        if hasattr(connection, "render_as_string"):
            return "uri", connection.render_as_string(hide_password=False)
        # an Engine connects through its own pool, creator and connect_args
        if hasattr(connection, "url") and callable(
            getattr(connection, "connect", None)
        ):
            pool = type(getattr(connection, "pool", None)).__name__
            if pool in _SHARED_CONNECTION_POOLS:
                # one connection (per thread) that the user's own statements use too
                # (temporary tables, session settings, an open transaction): read as
                # a live connection
                return "live", _shared_dbapi_connection(connection)
            return "factory", connection.connect

    if (
        callable(connection)
        and not hasattr(connection, "cursor")
        and not hasattr(connection, "execute")
    ):
        return "factory", connection
    if hasattr(connection, "cursor") or hasattr(connection, "execute"):
        return "live", connection

    msg = f"unsupported connection type for scan_database: {qualified_type_name(connection)!r}"
    raise TypeError(msg)


# SQLAlchemy pools whose Engine hands out one shared connection (per thread).
_SHARED_CONNECTION_POOLS = ("StaticPool", "SingletonThreadPool", "AssertionPool")


def _shared_dbapi_connection(engine: Any) -> Any:
    """
    The DB-API connection that an Engine's shared-connection pool hands out.

    Each check-in of it would roll it back (the pool's reset on return), also the
    transaction of a user's own `engine.connect()`: it is checked in once, here, without
    that reset, and the scan uses the DB-API connection directly.
    """
    from sqlalchemy.pool.base import _ConnectionFairy

    if not callable(getattr(_ConnectionFairy, "_close_special", None)):
        msg = (
            "scan_database can't share the single pooled connection of this "
            "SQLAlchemy version; pass `engine.raw_connection().driver_connection`"
        )
        raise TypeError(msg)
    fairy = engine.raw_connection()
    dbapi_connection = fairy.driver_connection
    fairy._close_special(transaction_reset=True)
    return dbapi_connection


# Python package of the I/O plugin that implements `scan_database`.
PLUGIN_PACKAGE_NAME = "polars_database"


def _plugin() -> Any:
    """
    The `polars_database` I/O plugin, which implements `scan_database`.

    Polars calls the plugin directly from its engine, through the FFI contracts the
    package exports (see `polars-io-ext-ffi`); Python is only used to hand over the
    scan's arguments.
    """
    try:
        import polars_database
    except ImportError as e:
        msg = (
            "scan_database needs the `polars-database` package; "
            "install it with: pip install 'polars[database]'"
        )
        raise ModuleNotFoundError(msg) from e
    return polars_database


def _engine_pool_capacity(engine: Any) -> int | None:
    """
    How many connections an Engine's pool hands out at once; None if unbounded.

    Partitions beyond it would wait for the pool, and time out.
    """
    pool = getattr(engine, "pool", None)
    # a QueuePool: `pool_size + max_overflow`; a size of 0 or a negative overflow is
    # unbounded
    with contextlib.suppress(Exception):
        overflow = int(pool._max_overflow)  # type: ignore[union-attr]
        size = int(pool.size())  # type: ignore[union-attr]
        return None if overflow < 0 or size <= 0 else size + overflow
    return None


def _default_max_connections(dialect: str) -> int:
    """The plugin's default `max_connections` for a SQLAlchemy dialect name."""
    # as `DatabaseScanConfig::max_connections` (polars-database)
    if dialect in ("sqlite", "duckdb"):
        return 1
    if dialect.startswith(
        ("snowflake", "bigquery", "databricks", "trino", "clickhouse")
    ):
        return 8
    return 4


def _string_options(options: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    return [(str(k), str(v)) for k, v in (options or {}).items()]


def _positive_int(name: str, value: Any) -> None:
    if value is None:
        return
    msg = f"{name} must be a positive integer, got {value!r}"
    try:
        # (accepts NumPy integers)
        index = None if isinstance(value, bool) else operator.index(value)
    except TypeError:
        index = None
    if index is None or index < 1:
        raise ValueError(msg)


def _copy_containers(value: Any) -> Any:
    """`value` with its lists, tuples and mappings copied, their values as they are."""
    if isinstance(value, Mapping):
        return {k: _copy_containers(v) for k, v in value.items()}
    if isinstance(value, list) or type(value) is tuple:
        return type(value)(_copy_containers(v) for v in value)
    return value


def _check_live_sqlite3(connection: Any) -> None:
    """Sessions run on other threads: a default sqlite3 connection refuses them."""
    import sqlite3
    import threading

    if not isinstance(connection, sqlite3.Connection):
        return
    error: list[Exception] = []

    def probe() -> None:
        try:
            connection.cursor().close()
        except sqlite3.ProgrammingError as e:
            error.append(e)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    if error:
        if "thread" not in str(error[0]):
            # e.g. a closed connection
            raise error[0]
        msg = (
            "this sqlite3 connection can only be used from the thread that created it, "
            "and scan_database reads on other threads; open it with "
            "`sqlite3.connect(..., check_same_thread=False)`, or pass a URI"
        )
        raise ValueError(msg)


def _check_credential_provider(provider: Any) -> None:
    if provider is None:
        return
    from polars.io.cloud.credential_provider._providers import CredentialProvider

    if isinstance(provider, CredentialProvider):
        msg = (
            f"{type(provider).__name__} provides cloud storage credentials, which "
            "can't log in to a database; pass a callable returning "
            "{'user': ..., 'password': ...}"
        )
        raise TypeError(msg)
    if not callable(provider):
        msg = (
            "credential_provider must be a callable returning "
            f"{{'user': ..., 'password': ...}}, got {qualified_type_name(provider)!r}"
        )
        raise TypeError(msg)


@unstable()
def scan_database(
    source: str,
    connection: str | Callable[[], Any] | Any,
    *,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
    source_type: Literal["auto", "table", "query"] = "auto",
    driver: str | None = None,
    dialect: str | None = None,
    connect_options: Mapping[str, Any] | None = None,
    execute_options: Mapping[str, Any] | None = None,
    credential_provider: Callable[[], Any] | None = None,
    schema: SchemaDict | None = None,
    schema_overrides: SchemaDict | None = None,
    partition_on: str | Expr | None = "auto",
    partition_num: int | None = None,
    partition_range: tuple[Any, Any] | None = None,
    partition_predicates: Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
    consistency: Literal["none", "snapshot"] = "none",
    max_connections: int | None = None,
    batch_size: int | None = None,
    pushdown: bool = True,
) -> LazyFrame:
    """
    Lazily read from a database table or query.

    The scan runs on the streaming engine. Projections, filters and limits are pushed
    into the SQL sent to the database, and the read can be split into partitions that
    are fetched concurrently.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    source
        A table name (`"orders"`, `"public.orders"`, `'"My Schema"."Orders"'`) or a
        SQL query. A query is wrapped as a derived table so that filters and
        projections can still be pushed down. As in SQL, a bare (unquoted) name
        part is folded to the case the database stores it in: `orders` reads
        `ORDERS` on Oracle and Snowflake, and `Orders` reads `orders` on Postgres
        (also when a table `"Orders"` exists). If no table has the folded name,
        the name as written is used; with `schema=` (no lookup) it is always used
        as written. Quote a part to use it exactly.
    connection
        Where to connect:

        * a URI, e.g. `"postgresql://user@host/db"`, `"sqlite:///path.db"`; a
          SQLAlchemy-style `"scheme+driver://…"` URI pins the driver. As with
          SQLAlchemy, `"sqlite:///rel.db"` is a relative path and
          `"sqlite:////abs.db"` an absolute one. URI query parameters are passed
          to the driver (an unknown one raises);
        * an ODBC connection string (`"Driver={…};Server=…"`);
        * a SQLAlchemy `URL` (as its URI), or `Engine`: the engine opens the
          connections, with its pool, `creator` and `connect_args`. The session
          settings the scan changes (Postgres, MySQL and Oracle: read-only
          transactions, UTC, ...) are restored before a connection goes back to
          the pool. An Engine with a `StaticPool`, `SingletonThreadPool` or
          `AssertionPool` shares one DB-API connection: it is used as a live
          connection (see below), so the scan sees uncommitted changes made on it;
        * a zero-argument callable that returns a DB-API connection (a factory);
        * a live connection object. It is used for one partition at a time, from
          another thread: open a sqlite3 connection with `check_same_thread=False`.
          Its results are read whole, with client-side cursors: Polars may still
          read, or release, a scan after `collect()` returned, while the connection
          is the user's again. A transaction the scan opens on it is rolled back as
          soon as the query ran; one already open stays.
          On a MySQL connection whose cursors are unbuffered (e.g. PyMySQL's
          `SSCursor`), results are read with buffered cursors, as two scans can't
          stream from one connection at once.
    params
        Parameters of a query source, bound by the driver (its own placeholder style,
        e.g. `%s` for psycopg, `?` for sqlite3). They are never interpolated into the
        SQL. Needs a driver that binds parameters itself: not ADBC or ConnectorX.
        Their lists, tuples and dicts are copied when the scan is built: changing them
        afterwards doesn't change the scan.
        Filters with a `%` in a string (and `str.starts_with` / `ends_with` /
        `contains`) then stay in Polars, as `%s`-style drivers would read the `%`;
        with SQLAlchemy (`:name` placeholders), so do those with a `:`.
    source_type : {'auto', 'table', 'query'}
        Whether `source` is a table name or a query. `'auto'` decides from its form.
    driver
        Pin the driver instead of choosing one automatically, e.g. `"adbc"`,
        `"connectorx"`, `"sqlalchemy"`, `"psycopg"` (any case), one of the dialect's
        drivers listed by :func:`polars.io.database.show_drivers`; `"adbc:<name>"`
        or a path to an ADBC driver library; or `"python:<module>"` for another
        DB-API module. Not for connection objects, whose module is the driver.
    dialect
        The SQL dialect, if it can't be inferred from the connection.
    connect_options
        Driver-specific connection options: ADBC database options (as strings),
        Python driver `connect()` keyword arguments (as given), SQLAlchemy
        `create_engine()` arguments, ConnectorX `protocol`. Not for connection
        objects.
    execute_options
        Driver-specific statement options (Python drivers' `execute()` keyword
        arguments). ConnectorX takes none.
    credential_provider
        A callable returning `{"user": …, "password": …}` (either key may be left
        out, not both), evaluated wherever a connection is opened. The values
        override those of the URI. Not for connection objects; cloud storage
        credential providers such as `pl.CredentialProviderAWS` don't apply.
    schema
        The schema of the source: the database's column types, as Polars dtypes.
        When given, no metadata is read from the database while planning, and
        filters are pushed as if the database stores these types. To read a
        column as another type, use `schema_overrides`.
    schema_overrides
        Dtypes that override the discovered schema for some columns. The values are
        cast after reading; filters on these columns stay in Polars.
    partition_on
        `"auto"` partitions large tables on their integer or date primary key; a
        column name partitions on that column (`pl.col("auto")` for a column named
        "auto"); `None` reads with a single query.
    partition_num
        The number of partitions. Without it: `max_connections` partitions for a
        column, and for `"auto"` one per 256 MB of table, up to four per connection
        (tables under 128 MB are read with one query).
    partition_range
        `(low, high)` bounds of the partition column. They only set the stride: rows
        outside them still land in the first or last partition.
    partition_predicates
        One partition per SQL `WHERE` fragment. The fragments are inserted verbatim.
        Not combined with `partition_on`, `partition_num` or `partition_range`.
    order_by
        Order rows by these columns. Without it, row order is unspecified.
    consistency : {'none', 'snapshot'}
        `'snapshot'` requires every partition to see the same database state; it is
        only supported with a single partition for now, so `partition_on="auto"`
        reads with one query.
    max_connections
        The maximum number of concurrent connections. Defaults: 1 for SQLite and
        DuckDB and for a live connection, 8 for Snowflake, BigQuery, Databricks,
        Trino and ClickHouse, 4 otherwise. For a SQLAlchemy `Engine`, it is capped
        at the number of connections its pool hands out (`pool_size + max_overflow`).
        Each scan of a query (e.g. both sides of a join) reads with its own
        connections: the pool must hold at least one per scan.
    batch_size
        Rows per batch requested from the driver.
    pushdown
        Push projections, filters and limits into the SQL. With `False`, the source
        runs as given, with a single query, and Polars does all further work.

    See Also
    --------
    read_database
    polars.io.database.show_drivers

    Examples
    --------
    >>> lf = pl.scan_database("orders", "sqlite:///shop.db")  # doctest: +SKIP
    >>> lf.filter(pl.col("status") == "paid").select(
    ...     "id", "amount"
    ... ).collect()  # doctest: +SKIP
    """
    if not isinstance(source, str):
        msg = f"source must be a table name or a SQL query, got {qualified_type_name(source)!r}"
        raise TypeError(msg)
    if not source.strip():
        msg = "source must be a table name or a SQL query, got an empty string"
        raise ValueError(msg)
    if source_type not in ("auto", "table", "query"):
        msg = f"source_type must be one of {{'auto', 'table', 'query'}}, got {source_type!r}"
        raise ValueError(msg)
    if consistency not in ("none", "snapshot"):
        msg = f"consistency must be one of {{'none', 'snapshot'}}, got {consistency!r}"
        raise ValueError(msg)
    if isinstance(order_by, str):
        order_by = [order_by]
    if isinstance(partition_predicates, str):
        msg = "partition_predicates must be a sequence of SQL predicates, one per partition, not a str"
        raise TypeError(msg)
    for name, options in (
        ("connect_options", connect_options),
        ("execute_options", execute_options),
    ):
        if options is not None and not isinstance(options, Mapping):
            msg = f"{name} must be a mapping, got {qualified_type_name(options)!r}"
            raise TypeError(msg)
    _check_credential_provider(credential_provider)
    for key in ("parameters", "params"):
        if execute_options is not None and key in execute_options:
            # the query's parameters, keyed and copied as `params` are
            if params is not None:
                msg = f"pass the query parameters once: params, or execute_options[{key!r}]"
                raise ValueError(msg)
            params = execute_options[key]
            execute_options = {k: v for k, v in execute_options.items() if k != key}
    if params is not None:
        # the values when the scan is built, also those the plugin keys by identity
        params = _copy_containers(params)
    for name, value in (
        ("partition_num", partition_num),
        ("max_connections", max_connections),
        ("batch_size", batch_size),
    ):
        _positive_int(name, value)

    # `"auto"` picks the column; `pl.col("auto")` names a column called "auto"
    partition_auto = isinstance(partition_on, str) and partition_on == "auto"
    if partition_on is not None and not isinstance(partition_on, str):
        on: Any = partition_on  # (any object may be passed)
        names = on.meta.root_names() if isinstance(on, Expr) else []
        if len(names) != 1 or not on.meta.is_column():
            msg = "partition_on must be a column name, `pl.col(name)`, 'auto' or None"
            raise TypeError(msg)
        partition_on = names[0]
    if partition_predicates is not None:
        if (
            (partition_on is not None and not partition_auto)
            or partition_num is not None
            or partition_range
        ):
            msg = (
                "partition_predicates define the partitions: don't combine them with "
                "partition_on, partition_num or partition_range"
            )
            raise ValueError(msg)
        if consistency == "snapshot" and len(partition_predicates) > 1:
            msg = (
                "consistency='snapshot' reads with one query: don't combine it with "
                "partition_predicates"
            )
            raise ValueError(msg)
    elif partition_on is None and (partition_num is not None or partition_range):
        msg = (
            "partition_num and partition_range need partitioning: partition_on is None"
        )
        raise ValueError(msg)
    elif partition_auto and partition_range:
        msg = "partition_range needs partition_on=<column>"
        raise ValueError(msg)
    if not pushdown and (
        partition_predicates is not None
        or (partition_on is not None and not partition_auto)
        or partition_num is not None
        or partition_range
    ):
        msg = (
            "pushdown=False reads the source with a single query: don't combine it "
            "with partition_on, partition_num, partition_range or partition_predicates"
        )
        raise ValueError(msg)
    if partition_range is not None:
        if len(partition_range) != 2:
            msg = f"partition_range must be (low, high), got {partition_range!r}"
            raise ValueError(msg)
        lo, hi = partition_range
        with contextlib.suppress(TypeError):
            if lo > hi:
                msg = f"partition_range must be (low, high), got {partition_range!r}"
                raise ValueError(msg)

    engine = connection
    connection_kind, connection = _connection_kind(connection)
    if (
        connection_kind == "factory"
        and connection is not engine
        and (capacity := _engine_pool_capacity(engine)) is not None
    ):
        # an Engine: no more partitions read at once than its pool has connections
        if max_connections is None:
            engine_dialect = getattr(getattr(engine, "dialect", None), "name", "")
            max_connections = _default_max_connections(
                str(dialect or engine_dialect).lower()
            )
        max_connections = min(max_connections, capacity)
    if connection_kind in ("factory", "live"):
        # the connection object fixes the driver, its options and its login
        form = (
            "a connection factory"
            if connection_kind == "factory"
            else ("a live connection")
        )
        ignored = [
            name
            for name, value in (
                ("driver", driver),
                ("connect_options", connect_options),
                ("credential_provider", credential_provider),
            )
            if value
        ]
        if ignored:
            msg = (
                f"{' and '.join(ignored)} can't apply to {form}, which is already "
                "set up by its driver; configure the connection itself"
            )
            raise ValueError(msg)
    if connection_kind == "live":
        _check_live_sqlite3(connection)

    from polars import DataFrame, Series

    resolver = _plugin().scan_resolver(
        source=source,
        source_type=source_type,
        connection=connection,
        connection_kind=connection_kind,
        dialect=dialect,
        driver=driver,
        connect_options=_string_options(connect_options),
        execute_options=_string_options(execute_options),
        # Python drivers take the values as given (ints, bools, dicts)
        python_options=(dict(connect_options or {}), dict(execute_options or {})),
        credential_provider=credential_provider,
        params=params,
        # schemas and values cross as Arrow C streams
        schema=DataFrame(schema=Schema(schema)) if schema is not None else None,
        schema_overrides=DataFrame(schema=Schema(schema_overrides))
        if schema_overrides is not None
        else None,
        partition_on=partition_on,
        partition_auto=partition_auto,
        partition_num=partition_num,
        partition_range=Series(list(partition_range), strict=False)
        if partition_range is not None
        else None,
        partition_predicates=list(partition_predicates)
        if partition_predicates is not None
        else None,
        order_by=list(order_by or []),
        consistency=consistency,
        max_connections=max_connections,
        batch_size=batch_size,
        pushdown=pushdown,
    )
    return wrap_ldf(PyLazyFrame._from_io_plugin_resolver(PLUGIN_PACKAGE_NAME, resolver))


@unstable()
def show_drivers(dialect: str | None = None) -> DataFrame:
    """
    Show the database drivers `scan_database` can use on this machine.

    Candidates are listed per dialect, best first; `scan_database` picks the first
    available one unless a driver is pinned.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    dialect
        A dialect name (`"postgresql"`), or a URI whose scheme names one. `None`
        shows every dialect.

    Examples
    --------
    >>> pl.io.database.show_drivers("sqlite")  # doctest: +SKIP
    """
    from polars import DataFrame

    plugin = _plugin()
    if dialect is not None and ":" in dialect:
        dialect = plugin.dialect_from_scheme(dialect.split(":", 1)[0])
    rows = plugin.driver_statuses(dialect)
    return DataFrame(
        rows,
        schema={
            "dialect": str,
            "backend": str,
            "available": bool,
            "version": str,
            "location": str,
            "install": str,
        },
        orient="row",
    )
