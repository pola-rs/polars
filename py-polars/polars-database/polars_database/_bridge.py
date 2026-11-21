"""
Python database backends for `scan_database`, called from Rust.

Rust picks the backend and runs the SQL it compiled; this module only opens connections
for Python drivers and streams results back as DataFrames. Query execution reuses the
`ConnectionExecutor` that `read_database` uses, so every driver it supports works here.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import importlib.util
import inspect
import itertools
import os
import re
import sys
import threading
import time
import weakref
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

from polars_database import _lib

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from polars import DataFrame, Schema

# Rows per batch when the user gives no `batch_size`.
DEFAULT_BATCH_SIZE = 50_000

# Python package (distribution) names, for version lookups.
_DISTRIBUTIONS = {
    "MySQLdb": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "mysql.connector": "mysql-connector-python",
    "snowflake.connector": "snowflake-connector-python",
    "google.cloud.bigquery": "google-cloud-bigquery",
    "databricks.sql": "databricks-sql-connector",
    "clickhouse_connect": "clickhouse-connect",
    "clickhouse_driver": "clickhouse-driver",
    "arrow_odbc": "arrow-odbc",
    "mssql_python": "mssql-python",
    "redshift_connector": "redshift-connector",
}


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(module: str) -> str | None:
    root = module.split(".")[0]
    if module == "sqlite3":
        # the standard library's module: the SQLite library is what has a version
        import sqlite3

        return sqlite3.sqlite_version
    for dist in (_DISTRIBUTIONS.get(module), root, root.replace("_", "-")):
        if dist:
            with contextlib.suppress(PackageNotFoundError, ValueError):
                return _dist_version(dist)
    return None


def probe(kind: str) -> tuple[str | None, str | None] | None:
    """Whether a backend is installed: `(version, location)`, or None."""
    head, _, tail = kind.partition(":")
    if head == "adbc":
        # a pip-installed `adbc_driver_<name>` package carries the driver library
        module = f"adbc_driver_{tail}"
        if not re.fullmatch(r"\w+", tail) or not _has_module(module):
            return None
        try:
            driver = importlib.import_module(module)
            path = driver._driver_path()
        except Exception:
            return None
        return _version(module), str(path)
    if head == "connectorx":
        return (_version("connectorx"), None) if _has_module("connectorx") else None
    if head == "sqlalchemy":
        return (_version("sqlalchemy"), None) if _has_module("sqlalchemy") else None
    if head in ("python", "odbc") and tail:
        return (_version(tail), None) if _has_module(tail) else None
    return None


def _module_of(obj: Any) -> str:
    """
    The driver module of a connection (cursor, result); a user's subclass of a driver's
    class (`sqlite3.connect(factory=MyConn)`) is its base's.
    """
    own = None
    for cls in type(obj).__mro__:
        module = _top_module(cls)
        if module in _DRIVER_MODULES:
            return module
        if own is None:
            own = module
    return own or _top_module(type(obj))


def _top_module(cls: type) -> str:
    module = (getattr(cls, "__module__", None) or "builtins").split(".", 1)[0]
    # compiled driver modules can have a leading underscore (eg: "_duckdb")
    public = module.lstrip("_")
    return public if public != module and _has_module(public) else module


# Modules whose classes a user may subclass: a subclass is served as its base
_DRIVER_MODULES = frozenset(
    {
        "adbc_driver_manager",
        "sqlalchemy",
        "sqlite3",
        "duckdb",
        "psycopg",
        "psycopg2",
        "pg8000",
        "pymysql",
        "mysql",
        "MySQLdb",
        "mariadb",
        "oracledb",
        "cx_Oracle",
        "clickhouse_connect",
        "pymssql",
        "pyodbc",
        "snowflake",
        "databricks",
        "google",
        "trino",
        "redshift_connector",
    }
)


def _pool_checkout(
    conn: Any,
) -> tuple[Any, Callable[[], None] | None, Callable[[], None] | None]:
    """
    A SQLAlchemy pool's checkout (`engine.raw_connection()`): its DB-API connection, the
    call that returns it to the pool, and the one that discards it.
    """
    if (
        type(conn).__name__ == "_ConnectionFairy"
        and (driver := getattr(conn, "driver_connection", None)) is not None
    ):
        return driver, conn.close, conn.invalidate
    pool = getattr(conn, "_pool", None)
    if (
        pool is not None
        and type(pool).__module__.startswith("psycopg_pool")
        and callable(putconn := getattr(pool, "putconn", None))
        and not inspect.iscoroutinefunction(putconn)
    ):
        # a psycopg_pool `getconn()`: its `close()` closes the connection but doesn't
        # give the slot back; `putconn` does (and replaces a closed connection)
        def invalidate() -> None:
            # `close_returns=True`: `close()` gives it back itself (public attribute
            # since psycopg_pool 3.2.x/3.3; the private spelling is from older ones)
            if getattr(pool, "close_returns", False) or getattr(
                pool, "_close_returns", False
            ):
                conn.close()
                return
            with contextlib.suppress(Exception):
                conn.close()
            putconn(conn)

        return conn, lambda: putconn(conn), invalidate
    return conn, None, None


# SQLAlchemy pools that hand out one DB-API connection (per thread) to every checkout.
SHARED_CONNECTION_POOLS = ("StaticPool", "SingletonThreadPool", "AssertionPool")


def _no_reset(*_args: Any, **_kwargs: Any) -> None:
    return None


def _release_without_reset(fairy: Any) -> None:
    """
    Check a pool checkout in without the pool's reset on return.

    `transaction_reset` only spares the default rollback: a `reset_on_return="commit"`
    pool would commit, and `reset` listeners would run, on a connection whose
    transaction is the user's.
    """
    if fairy._counter > 1:
        # a pool that hands the thread's checkout out again (SingletonThreadPool): this
        # only gives back our share of the user's own checkout, which stays theirs
        fairy._close_special(transaction_reset=True)
        return
    try:
        fairy._reset = _no_reset
    except AttributeError:
        pool = fairy._pool
        if pool._reset_on_return.name == "reset_commit" or pool.dispatch.reset:
            msg = (
                "scan_database can't share the pooled connection of this engine "
                "without its pool's reset on return (`pool_reset_on_return` other "
                "than 'rollback', or a `reset` event listener), which would end the "
                "user's transaction: pass `engine.raw_connection().driver_connection`"
            )
            raise TypeError(msg) from None
    try:
        fairy._close_special(transaction_reset=True)
    finally:
        # (never left on a fairy that is checked out again)
        fairy.__dict__.pop("_reset", None)


def _is_orm_session(obj: Any) -> bool:
    return any(
        c.__name__ == "Session" and c.__module__.startswith("sqlalchemy.orm")
        for c in type(obj).__mro__
    )


def _bound_connection(session: Any) -> Any:
    """
    The SQLAlchemy Connection an ORM Session is bound to, or None.

    Its transaction and results may be the user's while the Session has none.
    """
    with contextlib.suppress(Exception):
        bind = session.get_bind()
        if _module_of(bind) == "sqlalchemy" and type(bind).__name__ == "Connection":
            return bind
    return None


def _session_connection(session: Any) -> Any:
    """
    The SQLAlchemy Connection an ORM Session's statements run on, without beginning a
    transaction: its bound Connection, or its own in a transaction; else None.
    """
    if (bound := _bound_connection(session)) is not None:
        return bound
    with contextlib.suppress(Exception):
        if session.in_transaction():
            return session.connection()
    return None


def _refuse_bound_session(checkout: Any) -> None:
    """Refuse a factory's ORM Session bound to a Connection, which is the caller's."""
    if _is_orm_session(checkout) and _bound_connection(checkout) is not None:
        msg = (
            "a connection function that returns Sessions bound to one Connection "
            "can't be scanned: that connection is the caller's. Pass the Session "
            "or the Connection itself"
        )
        raise TypeError(msg)


def _has_pool(obj: Any, name: str) -> bool:
    pool = getattr(obj, "pool", None)
    return pool is not None and any(c.__name__ == name for c in type(pool).__mro__)


def _singleton_pool_error() -> TypeError:
    return TypeError(
        "a connection function over a SingletonThreadPool can't be scanned: the pool "
        "opens a connection for each thread that asks, and closes other threads' "
        "(the caller's too) beyond its `pool_size`, while the scan connects from its "
        "own threads. Pass the Engine itself (read through its one connection), or "
        "use another pool"
    )


def shared_pool_connection(obj: Any) -> Any:
    """
    The DB-API connection of a checkout (`engine.raw_connection()`, `engine.connect()`,
    an ORM Session of the engine) from a pool that shares one connection, released
    without the pool's reset on return (which would end the transaction of whoever else
    uses the connection); None for any other object. The connection is then read as a
    live one. Raises for a SingletonThreadPool (see `_singleton_pool_error`).
    """
    fairy = obj
    if _module_of(obj) == "sqlalchemy" and type(obj).__name__ == "Connection":
        fairy = getattr(obj, "connection", None)
    elif _is_orm_session(obj):
        # a Session checks its connection out on first use, and gives it back on
        # `rollback()`/`close()`: read the Engine's one connection instead
        engine = None
        with contextlib.suppress(Exception):
            engine = obj.get_bind()
        pool = getattr(engine, "pool", None)
        if not any(c.__name__ in SHARED_CONNECTION_POOLS for c in type(pool).__mro__):
            return None
        if _has_pool(engine, "SingletonThreadPool"):
            raise _singleton_pool_error()
        return _engine_shared_connection(engine)
    if type(fairy).__name__ != "_ConnectionFairy":
        return None
    pool = getattr(fairy, "_pool", None)
    if not any(c.__name__ in SHARED_CONNECTION_POOLS for c in type(pool).__mro__):
        return None
    release = getattr(fairy, "_close_special", None)
    if not callable(release) or (driver := fairy.driver_connection) is None:
        return None
    if any(c.__name__ == "SingletonThreadPool" for c in type(pool).__mro__):
        _release_without_reset(fairy)
        raise _singleton_pool_error()
    _release_without_reset(fairy)
    return driver


def _engine_shared_connection(engine: Any) -> Any:
    """The one DB-API connection of an Engine whose pool shares it, left as it is."""
    pool = engine.pool
    if any(c.__name__ == "AssertionPool" for c in type(pool).__mro__) and getattr(
        pool, "_checked_out", False
    ):
        # checked out by a user's `engine.connect()` (a second checkout raises)
        record = getattr(pool, "_conn", None)
        if (driver := getattr(record, "driver_connection", None)) is not None:
            return driver
    fairy = engine.raw_connection()
    driver = fairy.driver_connection
    _release_without_reset(fairy)
    return driver


def _duckdb_engine_connection(obj: Any, session: dict[str, Any]) -> Any:
    """
    The DuckDB connection under a SQLAlchemy `Connection` of duckdb_engine, or None.

    duckdb_engine's cursors run on the one DuckDB connection, where a statement
    replaces a result the user has not read to the end: read as the DuckDB connection
    it is, the scan leaves that result alone (as for a DuckDB connection passed
    directly). Not with the user's own parameters or execute options, which are
    SQLAlchemy's.
    """
    if (
        _module_of(obj) != "sqlalchemy"
        or type(obj).__name__ != "Connection"
        or session.get("params") is not None
        or session.get("execute_options")
    ):
        return None
    with contextlib.suppress(Exception):
        wrapper = obj.connection.driver_connection
        raw = getattr(wrapper, "_ConnectionWrapper__c", None)
        if _module_of(raw) == "duckdb":
            return raw
    return None


# ADBC's vendor names, as dialect names
_ADBC_VENDORS = {
    "postgresql": "postgresql",
    "sqlite": "sqlite",
    "duckdb": "duckdb",
    "mysql": "mysql",
    "mariadb": "mysql",
    "snowflake": "snowflake",
    "microsoft sql server": "mssql",
    "clickhouse": "clickhouse",
}


def connection_info(obj: Any, kind: str) -> tuple[str, str]:
    """`(dialect, module)` of a connection factory or live connection object."""
    if kind == "factory":
        conn = obj()
        _refuse_bound_session(conn)
        if (shared := shared_pool_connection(conn)) is not None:
            return _connection_info(shared)
        driver, give_back, _ = _pool_checkout(conn)
        try:
            return _connection_info(driver)
        finally:
            # a factory may return one connection again: an open session's isn't ours
            # to close
            with _FACTORY_LOCK:
                held = id(driver) in _FACTORY_CONNECTIONS
                if not held:
                    with contextlib.suppress(Exception):
                        (give_back or conn.close)()
    if _module_of(obj) == "adbc_driver_manager":
        # Its autocommit is off: the scan's statements would leave a transaction open on
        # the user's connection, and a failed one abort it.
        msg = (
            "a live ADBC connection can't be scanned (the scan's statements and result "
            "streams would share its transaction); pass its URI (driver='adbc'), or a "
            "function that opens a connection"
        )
        raise TypeError(msg)
    return _connection_info(obj)


def _require_sqlalchemy() -> None:
    """
    SQLAlchemy 2.0 or later: sessions use its 2.0 `Connection` API (`commit()`,
    `exec_driver_sql` options). On 1.4 the session setup failed (no `commit`), and bound
    parameters and SQLite connections across threads didn't work.
    """
    from polars._dependencies import import_optional

    import_optional("sqlalchemy", err_prefix="scan_database needs", min_version="2.0")


def _connection_info(obj: Any) -> tuple[str, str]:
    module = _module_of(obj)
    if module == "adbc_driver_manager":
        # its driver says which database it is
        with contextlib.suppress(Exception):
            vendor = str(obj.adbc_get_info().get("vendor_name", "")).lower()
            return _ADBC_VENDORS.get(vendor, ""), module
    if module == "pyodbc":
        # an ODBC driver of any DBMS: `SQL_DBMS_NAME` (17) says which. Only SQL Server's
        # is checked; the others stay generic.
        with contextlib.suppress(Exception):
            if str(obj.getinfo(17)).lower() == "microsoft sql server":
                return "mssql", module
    if module == "sqlalchemy":
        _require_sqlalchemy()
        engine = getattr(obj, "engine", None) or getattr(obj, "bind", None)
        dialect = getattr(getattr(engine, "dialect", None), "name", "") or ""
        return dialect, "sqlalchemy"
    return "", module


def credentials(provider: Callable[[], Any]) -> tuple[str | None, str | None] | None:
    """Evaluates a credential provider: `(user, password)`."""
    creds = provider()
    if isinstance(creds, tuple) and len(creds) == 2:
        if all(isinstance(c, str) for c in creds):
            msg = (
                "a database credential provider must return a dict, e.g. "
                "{'user': ..., 'password': ...}, not a (user, password) tuple"
            )
            raise TypeError(msg)
        # `(credentials, expiry)`, as the cloud storage credential providers return
        creds = creds[0]
    if creds is None:
        return None
    if isinstance(creds, dict):
        user = creds.get("user", creds.get("username"))
        password = creds.get("password", creds.get("token"))
        if user is None and password is None:
            # e.g. cloud storage keys: using them would drop the URI's password
            msg = (
                "the database credential provider returned neither 'user' nor "
                f"'password' (keys: {sorted(map(str, creds))})"
            )
            raise ValueError(msg)
        return user, password
    msg = f"a database credential provider must return a dict, got {type(creds).__name__!r}"
    raise TypeError(msg)


def _delimiter(s: str) -> int:
    """The index of the first `/`, `?` or `#` (the end of a URI's authority)."""
    return min((i for i in map(s.find, "/?#") if i >= 0), default=len(s))


def _split_authority(uri: str) -> tuple[str, str | None, str, str, bool] | None:
    """
    `(scheme, userinfo, host[:port], rest, unencoded)` of a `scheme://` URI, or None.

    `unencoded` marks a password holding an unencoded `/`, `?` or `#` (see
    `userinfo_end` in Rust, `connection.rs`).
    """
    scheme, sep, rest = uri.partition("://")
    if not sep:
        return None
    at = _lib.userinfo_end(rest)
    unencoded = at is not None and at > _delimiter(rest)
    userinfo = rest[:at] if at is not None else None
    rest = rest[at + 1 :] if at is not None else rest
    end = _delimiter(rest)
    return scheme, userinfo, rest[:end], rest[end:], unencoded


def _with_credentials(uri: str, creds: tuple[str | None, str | None] | None) -> str:
    split = _split_authority(uri)
    if split is None or not (creds or split[4]):
        return uri
    scheme, userinfo, host, rest, _ = split
    old_user, colon, old_password = (userinfo or "").partition(":")
    user, password = creds or (None, None)
    auth = quote(user, safe="") if user is not None else old_user
    if password is not None:
        auth += ":" + quote(password, safe="")
    elif colon:
        # the provider gave no password: keep the URI's (encoded, see
        # `_split_authority`)
        auth += ":" + quote(unquote(old_password), safe="")
    if creds:
        # libpq, MySQL drivers and SQLAlchemy read a `?password=` after the userinfo:
        # the provider's must win
        rest = _without_query_keys(
            rest,
            ({"user", "username", "uid"} if user is not None else set())
            | ({"password", "passwd", "pwd"} if password is not None else set()),
        )
    return f"{scheme}://{auth}@{host}{rest}" if auth else f"{scheme}://{host}{rest}"


def _without_query_keys(rest: str, keys: set[str]) -> str:
    """The path and query `rest` without the query parameters in `keys` (any case)."""
    head, sep, query = rest.partition("?")
    if not sep or not keys:
        return rest
    query, hash_, fragment = query.partition("#")
    kept = [kv for kv in query.split("&") if kv.partition("=")[0].lower() not in keys]
    return head + ("?" + "&".join(kept) if kept else "") + hash_ + fragment


# Seconds before a connect attempt gives up, where the URI sets none: libpq waits for
# the operating system's TCP timeout (minutes) by default, and attempts are retried.
DEFAULT_CONNECT_TIMEOUT = 10

# SQLAlchemy's Postgres DB-API drivers built on libpq, which takes `connect_timeout`
_LIBPQ_SQLALCHEMY_DRIVERS = ("psycopg", "psycopg2", "psycopg2cffi")


def _with_connect_timeout(uri: str, dialect: str, kind: str) -> str:
    if dialect not in ("postgresql", "redshift") or "connect_timeout=" in uri:
        return uri
    head, _, tail = kind.partition(":")
    if head == "sqlalchemy":
        # `postgresql://` and sqlalchemy-redshift's `redshift://` default to psycopg2
        driver = uri.split(":", 1)[0].partition("+")[2] or "psycopg2"
        libpq = driver in _LIBPQ_SQLALCHEMY_DRIVERS
    else:
        # ConnectorX reads it too; pg8000 and redshift_connector reject it
        libpq = head == "connectorx" or tail in ("psycopg", "psycopg2")
    if not libpq:
        return uri
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}connect_timeout={DEFAULT_CONNECT_TIMEOUT}"


def _strip_driver(uri: str) -> str:
    scheme, sep, rest = uri.partition(":")
    return scheme.split("+", 1)[0] + sep + rest


# The scheme drivers read for a dialect whose aliases (`pgsql://`, `neon://`, `tidb://`)
# scan_database also takes
_CANONICAL_SCHEMES = {
    "postgresql": ("postgresql", "postgres"),
    "mysql": ("mysql",),
    "mssql": ("mssql",),
    "oracle": ("oracle",),
}


def _canonical_scheme(uri: str, dialect: str, head: str) -> str:
    """The URI with an alias scheme written as its dialect's (`pgsql://` as `postgresql://`)."""
    if (schemes := _CANONICAL_SCHEMES.get(dialect)) is None:
        return uri
    scheme, sep, rest = uri.partition(":")
    base, plus, driver = scheme.partition("+")
    if base.lower() in schemes or (head == "sqlalchemy" and base.lower() == "mariadb"):
        # SQLAlchemy has a `mariadb` dialect
        return uri
    if head == "sqlalchemy" and base.lower() == "oracledb" and not plus:
        # SQLAlchemy's `oracle://` is cx_Oracle
        return f"oracle+oracledb{sep}{rest}"
    return f"{schemes[0]}{plus}{driver}{sep}{rest}"


def _libpq_uri(uri: str) -> str:
    """A URI for libpq, which takes only `postgresql://` and `postgres://` schemes."""
    uri = _strip_driver(uri)
    scheme, sep, rest = uri.partition(":")
    return f"postgresql{sep}{rest}" if scheme.lower() == "redshift" else uri


# SQLite URI filename parameters: they make the path a `file:` URI
_SQLITE_URI_PARAMS = ("cache", "immutable", "mode", "modeof", "nolock", "psow", "vfs")


def _file_target(uri: str) -> tuple[str, dict[str, str]]:
    """
    The path of a `sqlite:` / `duckdb:` URI and its parameters, as SQLAlchemy reads it.

    `sqlite:///rel.db` is relative, `sqlite:////abs.db` absolute; `sqlite://` is
    ':memory:'. A `file:` URI (`sqlite:///file:a.db?mode=ro`) is returned whole.
    """
    path, params = _lib.file_target(uri)
    return path, dict(params)


def _file_path(uri: str) -> str:
    return _file_target(uri)[0]


def _sqlite_connect_args(uri: str) -> tuple[str, dict[str, Any]]:
    """`sqlite3.connect` arguments for a `sqlite:` URI."""
    path, params = _file_target(uri)
    # sessions move between threads; each is used by one thread at a time
    kw: dict[str, Any] = {"check_same_thread": False}
    file_params, unknown = {}, []
    for name, value in params.items():
        if name in _SQLITE_URI_PARAMS:
            file_params[name] = value
        elif name in ("timeout", "detect_types", "cached_statements"):
            kw[name] = _convert(
                "sqlite3", name, value, float if name == "timeout" else int
            )
        elif name == "isolation_level":
            kw[name] = value or None
        elif name not in ("uri", "check_same_thread"):
            # `uri=true` is implied by the file parameters, SQLAlchemy's form; the
            # thread check is always off
            unknown.append(name)
    _raise_unmapped("sqlite3", unknown)
    if file_params:
        safe = "/:\\"
        path = f"file:{quote(path, safe=safe)}?{urlencode(file_params)}"
    if path.startswith("file:"):
        kw["uri"] = True
    return path, kw


@functools.cache
def _sqlalchemy_decodes_sqlite_paths() -> bool:
    """
    Whether SQLAlchemy percent-decodes a SQLite URI's path: 2.1 does, 2.0 takes it as
    written (`sqlite:///a%20b.db` opened, and created, a file `a%20b.db`).
    """
    from sqlalchemy.engine import make_url

    return make_url("sqlite:///a%20b.db").database == "a b.db"


def _sqlalchemy_sqlite_uri(uri: str) -> str:
    """
    A `sqlite:` URI in SQLAlchemy's form, for the file the other backends open.

    SQLAlchemy reads `sqlite:rel.db` as no URI at all, and drops `?mode=ro` without
    `uri=true` (with `uri=true`, it wants a `file:` path).
    """
    path, kw = _sqlite_connect_args(uri)
    scheme = uri.split(":", 1)[0]
    if path == ":memory:":
        return f"{scheme}://"
    query = {
        k: str(v)
        for k, v in kw.items()
        if k not in ("check_same_thread", "uri") and v is not None
    }
    if kw.get("uri"):
        path, _, file_query = path.partition("?")
        query = {**dict(parse_qsl(file_query, keep_blank_values=True)), **query}
        query["uri"] = "true"
    if _sqlalchemy_decodes_sqlite_paths():
        path = quote(path, safe="/:\\")
    return f"{scheme}:///{path}" + (f"?{urlencode(query)}" if query else "")


def _duckdb_connect_args(uri: str) -> tuple[str, dict[str, Any]]:
    """`duckdb.connect` arguments for a `duckdb:` URI, whose parameters are settings."""
    path, params = _file_target(uri)
    kw: dict[str, Any] = {}
    if "read_only" in params:
        kw["read_only"] = _convert(
            "duckdb", "read_only", params.pop("read_only"), _bool
        )
    if params:
        # DuckDB rejects an unknown setting itself
        kw["config"] = params
    return path, kw


def _connectorx_uri(uri: str, dialect: str) -> str:
    uri = _strip_driver(uri)
    if dialect != "sqlite":
        return uri
    # ConnectorX opens the percent-decoded rest of its parsed `sqlite://…` URL, and
    # can't open `file:` URIs: hand it the same file as the other backends, absolute.
    # Fully encoded, the path is an opaque host the URL parser keeps as it is: as
    # written, it turns `C:\Users` into `C/\Users` on Windows
    path, params = _file_target(uri)
    if params or path.startswith("file:"):
        msg = (
            f"connectorx can't open SQLite with URI parameters ({uri.split('?', 1)[-1]!r}); "
            "pin another driver with driver='adbc' or driver='sqlite3'"
        )
        raise ValueError(msg)
    if path == ":memory:":
        return "sqlite://:memory:"
    return "sqlite://" + quote(os.path.abspath(path), safe="")


def _url_parts(uri: str) -> dict[str, Any]:
    split = _split_authority(_strip_driver(uri))
    if split is None:
        return dict.fromkeys(("host", "port", "user", "password", "database"))
    _, userinfo, hostport, rest, _ = split
    user, colon, password = (userinfo or "").partition(":")
    if "," in hostport:
        msg = (
            "this driver takes one host, and the URI names several; "
            "pin a libpq driver (driver='psycopg' or 'adbc') for multiple hosts"
        )
        raise ValueError(msg)
    location = urlsplit(f"//{hostport}")
    try:
        port = location.port
    except ValueError:
        # the text after `:` is shown nowhere: it may be part of a password
        msg = (
            "the connection URI has an invalid port; an unencoded `/`, `?`, `#` or "
            "`@` in the password must be percent-encoded"
        )
        raise ValueError(msg) from None
    path = rest.partition("?")[0].partition("#")[0]
    return {
        "host": location.hostname,
        "port": port,
        "user": unquote(user) if user else None,
        "password": unquote(password) if colon and password else None,
        "database": unquote(path.lstrip("/")) or None,
    }


def _url_query(uri: str) -> dict[str, str]:
    split = _split_authority(uri)
    rest = split[3] if split is not None else uri
    query = rest.partition("?")[2].partition("#")[0]
    return dict(parse_qsl(query, keep_blank_values=True))


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _bool(value: str) -> bool:
    low = value.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    msg = f"expected a boolean, got {value!r}"
    raise ValueError(msg)


def _convert(module: str, name: str, value: str, convert: Callable[[str], Any]) -> Any:
    try:
        return convert(value)
    except ValueError as e:
        msg = f"URI parameter {name}={value!r} for {module}: {e}"
        raise ValueError(msg) from None


def _raise_unmapped(module: str, names: list[str]) -> None:
    if names:
        # a name that isn't one may be part of a password the URI parse missed
        shown = [n for n in names if re.fullmatch(r"[A-Za-z_][\w.-]*", n)]
        listed = f"{sorted(shown)}" if len(shown) == len(names) else f"({len(names)})"
        msg = (
            f"{module} has no connect option for URI parameter(s) {listed}; "
            "pass the driver's own keyword arguments with connect_options={...}, "
            "or pin another driver with driver=..."
        )
        raise ValueError(msg)


def _tls_context(
    *,
    cafile: str | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
    key_password: str | None = None,
    verify: bool = True,
    check_hostname: bool = True,
) -> Any:
    """An `ssl.SSLContext`: `cafile="system"` (or None) uses the system's CAs."""
    import ssl

    try:
        ctx = ssl.create_default_context(cafile=None if cafile == "system" else cafile)
        if not verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        else:
            ctx.check_hostname = check_hostname
        if certfile is not None:
            ctx.load_cert_chain(certfile, keyfile, key_password)
    except OSError as e:
        files = [f for f in (cafile, certfile, keyfile) if f and f != "system"]
        msg = f"can't load the TLS certificate file(s) {files}: {e.strerror or e}"
        raise ValueError(msg) from None
    return ctx


def _pg8000_params(
    params: dict[str, str], kw: dict[str, Any], port: int | None
) -> None:
    # libpq's TLS parameters as an SSL context
    files = ("sslrootcert", "sslcert", "sslkey", "sslpassword")
    mode = params.pop("sslmode", None)
    if mode is None and any(f in params for f in files):
        mode = "require"
    if mode in ("require", "verify-ca", "verify-full"):
        cafile = params.get("sslrootcert")
        kw["ssl_context"] = _tls_context(
            cafile=cafile,
            certfile=params.get("sslcert"),
            keyfile=params.get("sslkey"),
            key_password=params.get("sslpassword"),
            # as libpq: `require` verifies the server when given a root certificate
            verify=mode != "require" or cafile is not None,
            check_hostname=mode == "verify-full",
        )
    elif mode not in (None, "disable", "allow", "prefer"):
        msg = f"URI parameter sslmode={mode!r} for pg8000: unknown mode"
        raise ValueError(msg)
    for f in files:
        params.pop(f, None)
    # pg8000 can't fall back between plain and TLS connections: `allow` and `prefer`
    # connect without TLS, as when there is no `sslmode`
    if (host := params.pop("host", None)) is not None:
        # libpq's socket directory
        if host.startswith("/"):
            kw["unix_sock"] = f"{host.rstrip('/')}/.s.PGSQL.{port or 5432}"
        else:
            kw["host"] = host
    if (options := params.pop("options", None)) is not None:
        kw["startup_params"] = {"options": options}
    if (timeout := params.pop("connect_timeout", None)) is not None:
        # pg8000's `timeout` holds for every socket operation: `_connect_uri` lifts it
        # once connected (unless `timeout` is given too)
        seconds = _convert("pg8000", "connect_timeout", timeout, float)
        kw[_PG8000_CONNECT_TIMEOUT] = seconds if seconds > 0 else None
    if (name := params.pop("fallback_application_name", None)) is not None:
        kw.setdefault("application_name", name)
    # libpq parameters pg8000 has no option for: accepted where pg8000 behaves the same,
    # rejected where it would ignore what they ask for
    same = {
        "target_session_attrs": ("any",),
        "client_encoding": ("utf8", "utf-8", "unicode"),
        "gssencmode": ("disable", "prefer"),
    }
    for name, values in same.items():
        value = params.pop(name, None)
        if value is not None and value.lower() not in values:
            msg = (
                f"URI parameter {name}={value!r} for pg8000: pg8000 can't honour it "
                f"(supported: {', '.join(values)})"
            )
            raise ValueError(msg)
    for name in ("keepalives_idle", "keepalives_interval", "keepalives_count"):
        # tuning of the TCP keepalives that `keepalives` turns on
        params.pop(name, None)


def _mysql_ssl_mode(module: str, params: dict[str, str], kw: dict[str, Any]) -> None:
    # MySQL's `--ssl-mode`, as the ADBC driver and ConnectorX read it
    mode = params.pop("ssl_mode", params.pop("ssl-mode", None))
    if module != "mariadb" and (ssl := params.pop("ssl", None)) is not None:
        # `ssl=true` (SQLAlchemy URLs, mariadb's option): TLS is required
        tls = _convert(module, "ssl", ssl, _bool)
        mode = mode or ("REQUIRED" if tls else "DISABLED")
    if mode is None:
        return
    mode = mode.upper()
    if mode not in (
        "DISABLED",
        "PREFERRED",
        "REQUIRED",
        "VERIFY_CA",
        "VERIFY_IDENTITY",
    ):
        msg = f"URI parameter ssl_mode={mode!r} for {module}: unknown mode"
        raise ValueError(msg)
    if module == "MySQLdb":
        kw["ssl_mode"] = mode
        return
    if mode == "DISABLED":
        kw["ssl" if module == "mariadb" else "ssl_disabled"] = module != "mariadb"
    elif mode == "REQUIRED":
        if module == "mariadb":
            kw["ssl"] = True
        elif module == "mysql.connector":
            # it only prefers TLS without a certificate check: `_connect_uri` checks
            kw["ssl_disabled"] = False
            kw[_REQUIRE_TLS] = True
        elif module == "pymysql" and not {"ssl_ca", "ssl_cert", "ssl_key"} & set(
            params
        ):
            # without a certificate parameter, pymysql only prefers TLS
            kw["ssl"] = _tls_context(verify=False)
    elif mode != "PREFERRED":
        if module == "mariadb":
            kw["ssl"] = True
        kw["ssl_verify_cert"] = True
        if mode == "VERIFY_IDENTITY" and module != "mariadb":
            kw["ssl_verify_identity"] = True
        if module == "mysql.connector":
            kw[_REQUIRE_TLS] = True


# `_query_kwargs` markers for `_connect_uri`, not driver arguments
_PG8000_CONNECT_TIMEOUT = "__polars_connect_timeout"
_REQUIRE_TLS = "__polars_require_tls"


def _check_mysql_tls(conn: Any) -> None:
    """Raises, closing the connection, if a MySQL connection isn't encrypted."""
    try:
        cur = conn.cursor()
        cur.execute("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
        row = cur.fetchone()
        cur.close()
    except BaseException:
        conn.close()
        raise
    if not row or not row[1]:
        conn.close()
        msg = "ssl_mode requires TLS, but the server doesn't support it"
        raise ValueError(msg)


def _trino_params(params: dict[str, str], kw: dict[str, Any], port: int | None) -> None:
    if (verify := params.pop("verify", None)) is not None:
        # a boolean, or the CA bundle to verify against
        try:
            kw["verify"] = _bool(verify)
        except ValueError:
            kw["verify"] = verify
    # the JDBC driver's form
    if (tls := params.pop("SSL", None)) is not None and _convert(
        "trino", "SSL", tls, _bool
    ):
        kw["http_scheme"] = "https"


def _redshift_params(
    params: dict[str, str], kw: dict[str, Any], port: int | None
) -> None:
    mode = params.pop("sslmode", None)
    if mode == "disable":
        kw["ssl"] = False
    elif mode in ("verify-ca", "verify-full"):
        kw["ssl"], kw["sslmode"] = True, mode
    elif mode is not None:
        msg = (
            f"URI parameter sslmode={mode!r} for redshift_connector: it supports "
            "only 'disable', 'verify-ca' and 'verify-full'"
        )
        raise ValueError(msg)


def _pymssql_params(
    params: dict[str, str], kw: dict[str, Any], port: int | None
) -> None:
    # the ODBC / JDBC drivers' `encrypt`
    if (encrypt := params.pop("encrypt", None)) is not None:
        kw["encryption"] = (
            "require" if _convert("pymssql", "encrypt", encrypt, _bool) else "off"
        )


_S, _I, _F, _B = str, int, float, _bool
_MYSQL_COMMON: dict[str, Any] = {
    "unix_socket": _S,
    "init_command": _S,
    "connect_timeout": _I,
    "read_timeout": _I,
    "write_timeout": _I,
    "ssl_ca": _S,
    "ssl_cert": _S,
    "ssl_key": _S,
    "compress": _B,
    "local_infile": _B,
    "autocommit": _B,
}

# URI query parameters a DB-API driver takes, as `{parameter: converter}` (the keyword
# argument of the same name) or `{parameter: (keyword, converter)}`; for drivers whose
# connect arguments come from the URI's parts. A parameter not listed raises.
_QUERY_KWARGS: dict[str, dict[str, Any]] = {
    "pg8000": {
        "application_name": _S,
        "timeout": _F,
        "tcp_keepalive": _B,
        "keepalives": ("tcp_keepalive", _B),
        "source_address": _S,
        "unix_sock": _S,
    },
    "pymysql": {
        **_MYSQL_COMMON,
        "charset": _S,
        "collation": _S,
        "sql_mode": _S,
        "program_name": _S,
        "bind_address": _S,
        "read_default_file": _S,
        "read_default_group": _S,
        "server_public_key": _S,
        "connect_timeout": _F,
        "read_timeout": _F,
        "write_timeout": _F,
        "max_allowed_packet": _I,
        "client_flag": _I,
        "ssl_key_password": _S,
        "ssl_verify_cert": _B,
        "ssl_verify_identity": _B,
        "ssl_check_hostname": ("ssl_verify_identity", _B),
        "ssl_disabled": _B,
        "binary_prefix": _B,
        "use_unicode": _B,
    },
    "mysql.connector": {
        **_MYSQL_COMMON,
        "charset": _S,
        "collation": _S,
        "sql_mode": _S,
        "time_zone": _S,
        "auth_plugin": _S,
        "ssl_cipher": _S,
        "tls_versions": ("tls_versions", lambda v: v.split(",")),
        "tls_ciphersuites": ("tls_ciphersuites", lambda v: v.split(",")),
        "connection_timeout": _I,
        "ssl_verify_cert": _B,
        "ssl_verify_identity": _B,
        "ssl_check_hostname": ("ssl_verify_identity", _B),
        "ssl_disabled": _B,
        "local_infile": ("allow_local_infile", _B),
        "allow_local_infile": _B,
        "use_pure": _B,
        "use_unicode": _B,
        "force_ipv6": _B,
    },
    "MySQLdb": {
        **{k: v for k, v in _MYSQL_COMMON.items() if not k.startswith("ssl_")},
        "charset": _S,
        "collation": _S,
        "sql_mode": _S,
        "auth_plugin": _S,
        "read_default_file": _S,
        "read_default_group": _S,
        "client_flag": _I,
        # TLS files: collected into MySQLdb's `ssl` dict
        "ssl_ca": _S,
        "ssl_cert": _S,
        "ssl_key": _S,
        "ssl_capath": _S,
        "ssl_cipher": _S,
    },
    "mariadb": {
        **_MYSQL_COMMON,
        "ssl_capath": _S,
        "ssl_cipher": _S,
        "ssl_crlpath": _S,
        "tls_version": _S,
        "default_file": _S,
        "default_group": _S,
        "plugin_dir": _S,
        "ssl": _B,
        "ssl_verify_cert": _B,
    },
    "pymssql": {
        "charset": _S,
        "appname": _S,
        "tds_version": _S,
        "conn_properties": _S,
        "encryption": _S,
        "timeout": _I,
        "login_timeout": _I,
        "read_only": _B,
        "autocommit": _B,
        "use_datetime2": _B,
    },
    "oracledb": {
        "service_name": _S,
        "sid": _S,
        "protocol": _S,
        "ssl_server_cert_dn": _S,
        "ssl_server_dn_match": _B,
        "use_sni": _B,
        "wallet_location": _S,
        "wallet_password": _S,
        "config_dir": _S,
        "https_proxy": _S,
        "https_proxy_port": _I,
        "tcp_connect_timeout": _F,
        "connect_timeout": ("tcp_connect_timeout", _F),
        "retry_count": _I,
        "retry_delay": _I,
        "expire_time": _I,
        "sdu": _I,
        "stmtcachesize": _I,
        "disable_oob": _B,
        "edition": _S,
        "program": _S,
        "machine": _S,
        "terminal": _S,
        "osuser": _S,
    },
    "redshift_connector": {
        "ssl": _B,
        "timeout": _I,
        "tcp_keepalive": _B,
        "application_name": _S,
        "region": _S,
        "cluster_identifier": _S,
        "db_user": _S,
        "iam": _B,
        "profile": _S,
        "max_prepared_statements": _I,
    },
    "trino": {
        "source": _S,
        "http_scheme": _S,
        "timezone": _S,
        "request_timeout": _F,
        "max_attempts": _I,
        "client_tags": ("client_tags", lambda v: v.split(",")),
        "legacy_primitive_types": _B,
    },
    "databricks.sql": {
        "catalog": _S,
        "schema": _S,
        "user_agent_entry": _S,
        "use_cloud_fetch": _B,
        "_tls_no_verify": _B,
    },
    "clickhouse_connect": {
        "interface": _S,
        "secure": _B,
        "verify": _B,
        "ca_cert": _S,
        "client_cert": _S,
        "client_cert_key": _S,
        "server_host_name": _S,
        "tls_mode": _S,
        "http_proxy": _S,
        "https_proxy": _S,
        "compress": _S,
        "connect_timeout": _F,
        "send_receive_timeout": _F,
        "query_limit": _I,
        "session_id": _S,
    },
}

# Parameters a driver reads in its own way: they're popped before the table lookup
_SPECIAL_PARAMS: dict[
    str, Callable[[dict[str, str], dict[str, Any], int | None], None]
] = {
    "pg8000": _pg8000_params,
    "redshift_connector": _redshift_params,
    "trino": _trino_params,
    "pymssql": _pymssql_params,
    **{
        m: (lambda m: lambda q, kw, _port: _mysql_ssl_mode(m, q, kw))(m)  # noqa: B023
        for m in ("pymysql", "mysql.connector", "MySQLdb", "mariadb")
    },
}


def _query_kwargs(module: str, uri: str, port: int | None = None) -> dict[str, Any]:
    """A driver's connect keyword arguments from the URI's query parameters."""
    params = _url_query(uri)
    kw: dict[str, Any] = {}
    if (special := _SPECIAL_PARAMS.get(module)) is not None:
        special(params, kw, port)
    table = _QUERY_KWARGS.get(module, {})
    unknown = []
    for name, value in params.items():
        if (spec := table.get(name)) is None:
            unknown.append(name)
            continue
        keyword, convert = spec if isinstance(spec, tuple) else (name, spec)
        kw[keyword] = _convert(module, name, value, convert)
    _raise_unmapped(module, unknown)
    if module == "MySQLdb":
        # TLS files go in one `ssl` dict
        ssl = {
            k[4:]: kw.pop(k)
            for k in list(kw)
            if k.startswith("ssl_") and k != "ssl_mode"
        }
        if ssl:
            kw["ssl"] = ssl
    return kw


# The keyword arguments that drivers name differently for the parts of a URI: `host`,
# `port`, `user`, `password` and `database` keep their names unless renamed here
_URI_PART_KWARGS: dict[str, dict[str, str]] = {
    "pg8000": {},
    "pymysql": {},
    "mysql.connector": {},
    "mariadb": {},
    "redshift_connector": {},
    "MySQLdb": {"password": "passwd", "database": "db"},
    "pymssql": {"host": "server"},
}


def _uri_part_kwargs(parts: dict[str, Any], renames: dict[str, str]) -> dict[str, Any]:
    """The URI's parts, as the driver's keyword arguments."""
    return {renames.get(k, k): v for k, v in parts.items() if v is not None}


# How long `_duckdb_connect` retries a file DuckDB still holds, and how often it asks
# first (doubling up to the maximum). The race with a closing instance lasts
# milliseconds; a file held for good (attached in another connection of this process)
# fails after this.
_DUCKDB_CONNECT_RETRY_SECONDS = 1.0
_DUCKDB_CONNECT_RETRY_INTERVAL = 0.005
_DUCKDB_CONNECT_RETRY_MAX_INTERVAL = 0.1


def _duckdb_connect(mod: Any, path: str, kwargs: dict[str, Any]) -> Any:
    """
    `duckdb.connect(path)`, retried while the file is still held by a closing database.

    DuckDB shares one database instance between the connections to a file in a process.
    When the last connection of one closes, the instance is destroyed; a `connect` from
    another thread at that moment finds no instance in its cache, but the file is still
    attached by the dying one: `Unique file handle conflict: Cannot attach ... already
    attached`. It lasts as long as the destruction does, so the connect is asked again,
    with a bound: a conflict that persists is another connection's (it attached it).
    """
    start = time.monotonic()
    interval = _DUCKDB_CONNECT_RETRY_INTERVAL
    while True:
        try:
            return mod.connect(path, **kwargs)
        except Exception as e:
            if "Unique file handle conflict" not in str(e):
                raise
            waited = time.monotonic() - start
            if waited >= _DUCKDB_CONNECT_RETRY_SECONDS:
                msg = (
                    f"{e} (it persisted for {waited:.1f} s: the file is attached by "
                    "another connection in this process, which has to be closed first)"
                )
                try:
                    err = type(e)(msg)
                except Exception:
                    raise e from None
                raise err from e
        time.sleep(interval)
        interval = min(interval * 2, _DUCKDB_CONNECT_RETRY_MAX_INTERVAL)


# The DuckDB databases this module has open, by file and connect arguments: the root
# connection that keeps the instance alive and how many sessions use it. Guarded by
# `_DUCKDB_ROOTS_LOCK`, which connecting and the last close both hold.
_DUCKDB_ROOTS: dict[tuple[str, str], list[Any]] = {}
_DUCKDB_ROOTS_LOCK = threading.Lock()


def _duckdb_shared_cursor(
    mod: Any, path: str, kwargs: dict[str, Any]
) -> tuple[Any, Callable[[], None]]:
    """
    A connection to a DuckDB file that shares its database instance with the others
    this module has open on it, and the function that closes it.

    A DuckDB instance is destroyed when its last connection closes, and for a while
    after that the file is still attached: a `connect` from another thread in that
    window fails with "Unique file handle conflict" (and not just for milliseconds: when
    the machine is loaded it can outlast `_duckdb_connect`'s retries). A root connection
    is kept open while any session uses the file, and the last session closes it under
    the lock that connecting takes, so a connect never meets a dying instance.
    """
    if path in ("", ":memory:") or path.startswith(":memory:"):
        conn = _duckdb_connect(mod, path, kwargs)
        return conn, conn.close
    # (a relative file is the working directory's, which may change between scans)
    # (resolved through symlinks, and opened as resolved: DuckDB's own instance cache
    # joins `link/../x.db` with `x.db`, where the file system doesn't)
    where = path
    if ":" not in path:
        where = path = os.path.realpath(path)
    key = (where, repr(sorted(kwargs.items(), key=lambda kv: kv[0])))
    with _DUCKDB_ROOTS_LOCK:
        entry = _DUCKDB_ROOTS.get(key)
        if entry is None:
            entry = _DUCKDB_ROOTS[key] = [_duckdb_connect(mod, path, kwargs), 0]
        try:
            cursor = entry[0].cursor()
        except BaseException:
            if entry[1] == 0:
                del _DUCKDB_ROOTS[key]
                with contextlib.suppress(Exception):
                    entry[0].close()
            raise
        entry[1] += 1
    closed = False

    def close() -> None:
        nonlocal closed
        with _DUCKDB_ROOTS_LOCK:
            if closed:
                return
            closed = True
            try:
                cursor.close()
            finally:
                entry[1] -= 1
                if entry[1] == 0:
                    _DUCKDB_ROOTS.pop(key, None)
                    entry[0].close()

    return cursor, close


def _connect_uri(module: str, uri: str, options: dict[str, Any]) -> Any:
    """A DB-API connection to a URI through a driver module."""
    mod = importlib.import_module(module)
    p = _url_parts(uri)
    if module == "sqlite3":
        path, kw = _sqlite_connect_args(uri)
        return mod.connect(path, **{**kw, **options})
    if module == "duckdb":
        path, kw = _duckdb_connect_args(uri)
        return _duckdb_connect(mod, path, {**kw, **options})
    if module == "psycopg":
        # libpq reads (and checks) the query parameters itself
        return mod.connect(_libpq_uri(uri), **options)
    if module == "psycopg2":
        conn = mod.connect(_libpq_uri(uri), **options)
        _psycopg2_bytes(conn)
        return conn

    # the driver's keyword arguments: the URI's parts, its query, then `connect_options`
    kw = _query_kwargs(module, uri, p["port"])
    if (renames := _URI_PART_KWARGS.get(module)) is not None:
        kw = {**_uri_part_kwargs(p, renames), **kw}
    elif module == "oracledb":
        # `/name` is the service name, as is `?service_name=name` (SQLAlchemy's form)
        if "sid" not in kw:
            kw.setdefault("service_name", p["database"] or "")
        base = _drop_none(user=p["user"], password=p["password"])
        kw = {**base, "host": p["host"], "port": p["port"] or 1521, **kw}
    elif module == "trino":
        from trino.dbapi import connect  # type: ignore[import-not-found]

        catalog, _, schema = (p["database"] or "").partition("/")
        base = _drop_none(host=p["host"], port=p["port"], user=p["user"])
        base.update(_drop_none(catalog=catalog or None, schema=schema or None))
        if p["password"] is not None:
            from trino.auth import BasicAuthentication  # type: ignore[import-not-found]

            # Trino takes a password only over HTTPS
            base["auth"] = BasicAuthentication(p["user"] or "", p["password"])
            base["http_scheme"] = "https"
        return connect(**{**base, **kw, **options})  # type: ignore[no-untyped-call]
    elif module == "databricks.sql":
        # `databricks://token:<access token>@<host>/<HTTP path>[?catalog=&schema=]`, the
        # ADBC driver's form
        if not p["host"] or not p["database"]:
            msg = (
                "a Databricks URI names the workspace host and the warehouse's HTTP path: "
                "databricks://token:<token>@<host>/sql/1.0/warehouses/<id>"
            )
            raise ValueError(msg)
        if p["user"] not in (None, "", "token"):
            # (the token is the password: it is never put in a message)
            msg = (
                "a Databricks URI authenticates with an access token: "
                "databricks://token:<token>@<host>/<HTTP path> (the user is `token`)"
            )
            raise ValueError(msg)
        base = _drop_none(
            server_hostname=p["host"],
            http_path="/" + p["database"],
            access_token=p["password"],
            _port=p["port"],
        )
        return mod.connect(**{**base, **kw, **options})
    elif module == "clickhouse_connect":
        from clickhouse_connect.dbapi import connect  # type: ignore[import-not-found]

        base = _uri_part_kwargs(p, {"user": "username"})
        return connect(**{**base, **kw, **options})  # type: ignore[no-untyped-call]
    else:
        msg = (
            f"scan_database can't open a {module!r} connection from a URI; "
            "pass a connection factory instead, e.g. `lambda: <module>.connect(...)`"
        )
        raise NotImplementedError(msg)
    require_tls = kw.pop(_REQUIRE_TLS, False)
    # libpq's `connect_timeout` for pg8000, whose `timeout` holds for every read
    lift_timeout = _PG8000_CONNECT_TIMEOUT in kw and not ({"timeout"} & {*kw, *options})
    if _PG8000_CONNECT_TIMEOUT in kw:
        kw.setdefault("timeout", kw.pop(_PG8000_CONNECT_TIMEOUT))
    conn = mod.connect(**{**kw, **options})
    if lift_timeout:
        with contextlib.suppress(AttributeError):
            conn._usock.settimeout(None)
    if require_tls:
        _check_mysql_tls(conn)
    return conn


def _secrets(
    target: str,
    creds: tuple[str | None, str | None] | None,
    options: dict[str, Any],
) -> list[str]:
    """
    The secrets of a connection (a URI or an ODBC connection string), to scrub from
    driver errors: longest first. Found by Rust (`ConnectionSpec::secrets`), which
    scrubs the messages of its own errors with them.
    """
    return _lib.connection_secrets(
        target,
        "uri" if "://" in target else "odbc",
        creds[1] if creds else None,
        [(k, v) for k, v in options.items() if isinstance(v, str)],
    )


_scrub = _lib.scrub


@contextlib.contextmanager
def _scrubbed(secrets: list[str]) -> Iterator[None]:
    """Driver errors without the connection's secrets: some echo the DSN."""
    try:
        yield
    except Exception as e:
        message = _scrub(str(e), secrets)
        if message == str(e):
            raise
        e.args = (message,)
        e.__cause__ = None
        e.__suppress_context__ = True
        if _scrub(str(e), secrets) != str(e):
            # a message not built from `args`
            msg = f"{type(e).__name__}: {message}"
            raise RuntimeError(msg) from None
        raise


def _no_panics(frames: Iterator[DataFrame]) -> Iterator[DataFrame]:
    """Batches, with a Polars panic on an Arrow type it can't import as an error."""
    from polars.exceptions import ComputeError

    try:
        yield from frames
    except BaseException as e:
        # e.g. Arrow month-day-nano intervals: a panic would abort the whole query
        if type(e).__name__ != "PanicException":
            raise
        msg = (
            f"the driver returned data Polars can't read ({e}); convert the column "
            "in a query, e.g. CAST(<column> AS VARCHAR)"
        )
        raise ComputeError(msg) from None


def _psycopg2_bytes(target: Any) -> None:
    """
    Reads psycopg2's bytea as bytes, not memoryview (which would become an Object
    column, and fails in a list), on a connection or a cursor: `bytea[]` (OID 1001) as
    lists.
    """
    mod = importlib.import_module("psycopg2")
    ext = importlib.import_module("psycopg2.extensions")
    as_bytes = ext.new_type(
        mod.BINARY.values,
        "POLARS_BYTES",
        lambda v, cur: None if v is None else mod.BINARY(v, cur).tobytes(),
    )
    ext.register_type(as_bytes, target)
    ext.register_type(
        ext.new_array_type((1001,), "POLARS_BYTES_ARRAY", as_bytes), target
    )


_SCAN_OPTION = "polars_database_scan"


@functools.cache
def _sqlalchemy_psycopg2_guard() -> None:
    """
    Makes the statements of a scan behind SQLAlchemy read bytea and infinity as a scan
    of psycopg2 does (`_psycopg2_bytes`, `_psycopg2_no_infinity`), on the driver's
    cursor that SQLAlchemy opens: a listener on every Engine, which acts on the
    statements run with the scan's execution option only.
    """
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "before_cursor_execute")
    def _on_cursor(
        conn: Any,
        cursor: Any,
        statement: Any,
        parameters: Any,
        context: Any,
        executemany: Any,
    ) -> None:
        if (
            context is not None
            and context.execution_options.get(_SCAN_OPTION)
            and _module_of(cursor) == "psycopg2"
        ):
            _psycopg2_bytes(cursor)
            _psycopg2_no_infinity(cursor)


def _psycopg2_no_infinity(cursor: Any) -> None:
    """
    Makes the cursor fail on a date or timestamp of `infinity` / `-infinity`, which
    psycopg2 reads as `date.max` / `date.min` (and Polars can't represent), whether or
    not the column's type was looked up: `date`, `timestamp`, `timestamptz` and their
    arrays.
    """
    ext = importlib.import_module("psycopg2.extensions")

    def guard(base: Any, name: str, oid: int, array_oid: int) -> None:
        def cast(value: Any, cur: Any) -> Any:
            if value is not None and value.strip().lower().endswith("infinity"):
                msg = (
                    f"the database holds '{value}' in a {name} column, which Polars can't "
                    "represent (the driver would read it as the largest or smallest "
                    "date): filter it out or cast it in a query"
                )
                raise ValueError(msg)
            return None if value is None else base(value, cur)

        scalar = ext.new_type((oid,), f"POLARS_{name.upper()}", cast)
        ext.register_type(scalar, cursor)
        ext.register_type(
            ext.new_array_type((array_oid,), f"POLARS_{name.upper()}_ARRAY", scalar),
            cursor,
        )

    guard(ext.DATE, "date", 1082, 1182)
    guard(ext.PYDATETIME, "timestamp", 1114, 1115)
    guard(ext.PYDATETIMETZ, "timestamptz", 1184, 1185)


def _object_columns_as_text(frames: Iterator[DataFrame]) -> Iterator[DataFrame]:
    """
    Batches without `Object` columns: a row driver's `uuid.UUID`, `ipaddress` or other
    Python values have no Arrow type (their export would carry pointers). They are read
    as text; `memoryview` and `bytearray` values (psycopg2's bytea) as binary.
    """
    import uuid
    from decimal import Decimal
    from ipaddress import _BaseAddress, _BaseNetwork

    import polars as pl
    from polars.exceptions import ComputeError

    as_text = (uuid.UUID, _BaseAddress, _BaseNetwork, Decimal)
    as_binary = (memoryview, bytearray, bytes)
    for df in frames:
        objects = [c for c, dt in df.schema.items() if dt == pl.Object]
        for name in objects:
            values = df[name].to_list()
            present = [v for v in values if v is not None]
            if present and all(isinstance(v, as_binary) for v in present):
                binary = [None if v is None else bytes(v) for v in values]
                df = df.with_columns(pl.Series(name, binary, pl.Binary))
                continue
            for v in present:
                if not isinstance(v, as_text):
                    msg = (
                        f"the driver returned column {name!r} as Python "
                        f"{type(v).__name__!r} values, which Polars can't read; "
                        "convert the column in a query, e.g. CAST(<column> AS VARCHAR)"
                    )
                    raise ComputeError(msg)
            df = df.with_columns(
                pl.Series(
                    name, [None if v is None else str(v) for v in values], pl.String
                )
            )
        yield df


def _valid_text(frames: Iterator[DataFrame]) -> Iterator[DataFrame]:
    """
    Batches whose text is valid UTF-8, also nested in lists, maps and structs.

    The Databricks connector hands the bytes of a `STRING` over as they are (a value
    made of `unhex('FF')` is one), and Polars' String assumes UTF-8: an invalid byte
    would reach the caller as a String that breaks its contract (memory unsafety in
    string operations).
    """
    import polars as pl
    from polars.exceptions import ComputeError

    def has_text(dtype: Any) -> bool:
        if dtype == pl.String:
            return True
        if isinstance(dtype, (pl.List, pl.Array)):
            return has_text(dtype.inner)
        if isinstance(dtype, pl.Map):
            return has_text(dtype.key) or has_text(dtype.value)
        if isinstance(dtype, pl.Struct):
            return any(has_text(f.dtype) for f in dtype.fields)
        return False

    for df in frames:
        names = [c for c, dt in df.schema.items() if has_text(dt)]
        if names:
            try:
                df.select(names).to_arrow().validate(full=True)
            except Exception as e:
                msg = (
                    "the database returned text that isn't valid UTF-8 (a STRING made "
                    "of other bytes); read the column as BINARY with "
                    f"CAST(<column> AS BINARY) and decode it in Polars ({e})"
                )
                raise ComputeError(msg) from e
        yield df


def _oracle_row_cursor(cx: Any) -> Any:
    """
    A cursor of the Oracle driver (oracledb without its Arrow fetch, cx_Oracle) that
    the executor reads rows from, or `None`: its own, or one on the DB-API connection
    of a SQLAlchemy connection (the same session, as the Arrow proxy uses).
    """
    cursor = cx.cursor
    if _module_of(cursor) in ("oracledb", "cx_Oracle"):
        return cursor
    if cx.driver_name == "sqlalchemy" and not cx._is_alchemy_session(cursor):
        with contextlib.suppress(Exception):
            driver = cursor.connection.driver_connection
            if _module_of(driver) in ("oracledb", "cx_Oracle"):
                return driver.cursor()
    return None


def _oracle_decimal_handler(cursor: Any) -> Callable[..., Any]:
    """An `outputtypehandler` that fetches NUMBER with a scale as `Decimal`."""
    from decimal import Decimal

    module = sys.modules[_module_of(cursor)]
    number = module.DB_TYPE_NUMBER

    def scaled(type_code: Any, precision: Any, scale: Any) -> bool:
        # integers (`NUMBER(p)`, `NUMBER(p, s)` with s <= 0) are exact as Python ints,
        # and `FLOAT(b)` (scale -127) is a float
        return type_code is number and not ((precision or 0) > 0 and (scale or 0) <= 0)

    if _module_of(cursor) == "oracledb":

        def handler(cursor: Any, metadata: Any) -> Any:
            if scaled(metadata.type_code, metadata.precision, metadata.scale):
                return cursor.var(Decimal, arraysize=cursor.arraysize)
            return None

        return handler

    def legacy_handler(
        cursor: Any, name: Any, default_type: Any, size: Any, precision: Any, scale: Any
    ) -> Any:
        if scaled(default_type, precision, scale):
            return cursor.var(Decimal, arraysize=cursor.arraysize)
        return None

    return legacy_handler


def _params_keyword(cx: Any) -> str:
    """The keyword of a cursor's `execute()` that takes the bound parameters."""
    from inspect import Parameter, signature

    if cx.driver_name in ("sqlalchemy", "duckdb") or "oracledb" in cx.driver_name:
        return "parameters"  # the executor maps it
    try:
        ps = [
            p
            for p in signature(cx.cursor.execute).parameters.values()
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        # no signature (C methods: sqlite3, psycopg2): passed positionally
        return "parameters"
    # (query, parameters, ...): psycopg `params`, PyMySQL and pg8000 `args`
    return ps[1].name if len(ps) >= 2 else "parameters"


def _transaction_idle(conn: Any) -> bool | None:
    """Whether a connection is outside a transaction; `None` if unknown."""
    if _is_orm_session(conn):
        with contextlib.suppress(Exception):
            if conn.in_transaction():
                return False
            if (bound := _bound_connection(conn)) is not None:
                return _transaction_idle(bound)
    with contextlib.suppress(Exception):
        if (info := getattr(conn, "info", None)) is not None and hasattr(
            info, "transaction_status"
        ):
            return bool(info.transaction_status == 0)  # psycopg: IDLE
        if callable(status := getattr(conn, "get_transaction_status", None)):
            return bool(status() == 0)  # psycopg2: TRANSACTION_STATUS_IDLE
        module = _module_of(conn)
        if module == "pg8000":
            # its DB-API connections send `BEGIN` before a statement outside one
            return not conn._in_transaction
        if module == "pymysql":
            in_trans = 1  # SERVER_STATUS_IN_TRANS
            if not conn.get_autocommit() and not conn.server_status & in_trans:
                # PyMySQL takes the status from OK packets only: a SELECT since may
                # have begun one, which `DO 0` (an OK packet) shows
                with conn.cursor() as cursor:
                    cursor.execute("DO 0")
            return not conn.server_status & in_trans
        if module in ("MySQLdb", "mariadb"):
            in_trans = 1  # SERVER_STATUS_IN_TRANS
            # MariaDB Connector/Python (and MySQLdb builds that have it): the status
            # flags of the last result
            if isinstance(status := getattr(conn, "server_status", None), int):
                return not status & in_trans
            # mysqlclient has no status: MariaDB servers say (`@@in_transaction`); a
            # MySQL server doesn't know the variable (an error that leaves no
            # transaction behind), and the answer is unknown
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT @@in_transaction")
                row = cursor.fetchone()
            finally:
                cursor.close()
            return None if row is None else not row[0]
        in_transaction = getattr(conn, "in_transaction", None)
        if callable(in_transaction):
            # A SQLAlchemy Connection, which begins one on its first statement, and
            # says so in AUTOCOMMIT too, where there is none
            # (an ORM Session has neither)
            if _sqlalchemy_autocommit(conn):
                # no transaction of SQLAlchemy's, but the user's own `BEGIN` may be open
                # on the driver's connection: that one says
                fairy = getattr(conn, "connection", None)
                driver = getattr(fairy, "driver_connection", None)
                if driver is not None and driver is not conn:
                    status = _transaction_idle(driver)
                    if status is not None:
                        return status
                return True
            if in_transaction():
                return False
            # none of SQLAlchemy's, but the driver's connection may have one open (a
            # `checkout` listener's `SET`, the user's own statements): that one stays
            fairy = getattr(conn, "connection", None)
            driver = getattr(fairy, "driver_connection", None)
            if driver is not None and driver is not conn:
                if _transaction_idle(driver) is False:
                    return False
            return True
        if isinstance(in_transaction, bool) and module in ("sqlite3", "mysql"):
            # sqlite3, and mysql.connector (the status of its last result)
            return not in_transaction
    return None


def _sqlalchemy_autocommit(conn: Any) -> bool:
    """Whether a SQLAlchemy Connection runs in AUTOCOMMIT."""
    autocommit = getattr(conn, "_is_autocommit_isolation", None)
    options = getattr(conn, "get_execution_options", None)
    return bool(
        (callable(autocommit) and autocommit())
        or (callable(options) and options().get("isolation_level") == "AUTOCOMMIT")
    )


def _refuse_pending_results(conn: Any) -> None:
    """
    Refuse a scan on a connection with an unread unbuffered result.

    A statement of the scan on such a connection drops the unread rows (PyMySQL, with a
    warning). (The pending result sets of a multi-statement query aren't told from the
    trailing OK packet of a `CALL` without reading it, so they stay a limit.)
    """
    if _module_of(conn) == "sqlalchemy":
        # a SQLAlchemy Connection, or an ORM Session's (only one in a transaction has
        # one): the driver's is what holds the result
        with contextlib.suppress(Exception):
            if _is_orm_session(conn) and (conn := _session_connection(conn)) is None:
                return
            conn = conn.connection.driver_connection
    if _module_of(conn) != "pymysql":
        return
    result = getattr(conn, "_result", None)
    if result is not None and getattr(result, "unbuffered_active", False):
        msg = (
            "the connection has an unbuffered result that has not been read to the "
            "end, and a scan would discard its rows: read it first (`fetchall()`), "
            "or pass a connection of its own"
        )
        raise RuntimeError(msg)


def _sqlalchemy_dialect(conn: Any) -> str:
    """The SQLAlchemy dialect name of a connection, engine or ORM Session; else ""."""
    # an ORM Session's dialect is its bind's
    bind = conn
    if callable(get_bind := getattr(conn, "get_bind", None)):
        with contextlib.suppress(Exception):
            bind = get_bind()
    return getattr(getattr(bind, "dialect", None), "name", "") or ""


def _cancel_target(conn: Any) -> Any:
    """What cancels a connection's statement: an Engine connection's driver connection
    (a SQLAlchemy Connection has no cancel).
    """
    if _module_of(conn) == "sqlalchemy":
        with contextlib.suppress(Exception):
            return conn.connection.driver_connection
    return conn


def _duckdb_needs_connection(conn: Any) -> bool:
    """
    Whether a DuckDB connection has what its cursors (connections of their own) don't
    see: temporary tables, views, macros or sequences, or an open transaction (its
    uncommitted writes). False if it can't be told, as with a result of the user's left
    unread: a cursor then.
    """
    temp = "database_name = 'temp'"
    sql = (
        f"SELECT EXISTS (SELECT 1 FROM duckdb_tables() WHERE {temp}) "
        f"OR EXISTS (SELECT 1 FROM duckdb_views() WHERE {temp} AND NOT internal) "
        f"OR EXISTS (SELECT 1 FROM duckdb_sequences() WHERE {temp}) "
        f"OR EXISTS (SELECT 1 FROM duckdb_functions() WHERE {temp} "
        "AND function_type IN ('macro', 'table_macro'))"
    )
    try:
        # A relation, not `execute`: `conn.execute` replaces a result of the user's left
        # unread (DuckDB 1 and 2, silently), a relation's query leaves it (DuckDB 1) or
        # is refused (DuckDB 2, `ResourceInUseException`)
        rows = conn.sql(sql).fetchall()
        if rows and rows[0][0]:
            return True
        # outside a transaction each statement is one of its own
        first = conn.sql("SELECT txid_current()").fetchall()
        second = conn.sql("SELECT txid_current()").fetchall()
    except Exception:
        return False
    return bool(first == second)


def _duckdb_copy_session(conn: Any, cursor: Any) -> bool:
    """
    Copies the session state of a DuckDB connection onto a cursor of it, which starts
    with the defaults: `search_path` (`USE`), `TimeZone`, `integer_division`, the
    variables of `SET VARIABLE`, and the rest of `duckdb_settings()` that differs. (Not
    by `scope`: a time zone set in the session is listed as global.) False if anything
    fails to read or copy: the scan then runs on the connection itself, not with some of
    its state.
    """

    def quote(v: Any) -> str:
        return "'" + str(v).replace("'", "''") + "'"

    def ident(v: Any) -> str:
        return '"' + str(v).replace('"', '""') + '"'

    skipped = {
        "schema",
        "enable_profiling",
        "profiling_output",
        "profiling_mode",
        "log_query_path",
    }
    # (qualified: a user's macro of the name doesn't hijack it)
    settings = "SELECT name, value FROM system.main.duckdb_settings()"
    variables = "SELECT name, value::VARCHAR, type FROM system.main.duckdb_variables()"
    try:
        wanted = dict(conn.sql(settings).fetchall())
        current = dict(cursor.execute(settings).fetchall())
        for name, value in wanted.items():
            # (`schema` can't be set: `search_path` has it; the profiling and log files
            # are the user's, and none of it changes rows)
            if name not in skipped and value != current.get(name):
                cursor.execute(f"SET {ident(name)} = {quote(value)}")
        for name, value, type_ in conn.sql(variables).fetchall():
            # DuckDB 2's catalog wraps typed NULLs in a VARIANT: its value is not
            # SQL NULL and casts to the text 'NULL', also a valid non-null VARCHAR.
            # getvariable requires a constant name, so inspect each variable itself.
            is_null = conn.sql(
                f"SELECT system.main.getvariable({quote(name)}) IS NULL"
            ).fetchall()[0][0]
            if type_.strip('"').upper() == "NULL":
                # an untyped NULL has no valid cast target
                cursor.execute(f"SET VARIABLE {ident(name)} = NULL")
            else:
                literal = "NULL" if is_null else quote(value)
                cursor.execute(f"SET VARIABLE {ident(name)} = {literal}::{type_}")
    except Exception as e:
        # DuckDB 2 refuses a relation's query while the user's result is unfinished
        # (even a `fetchone()` of a count): nothing of the session can be read, and the
        # cursor, which leaves that result alone, runs with the defaults
        return type(e).__name__ == "ResourceInUseException"
    return True


def _postgres_family(conn: Any) -> bool:
    """Whether a connection is a Postgres one, where a failed statement aborts the
    transaction it runs in.
    """
    module = _module_of(conn)
    if module in ("psycopg", "psycopg2", "pg8000"):
        return True
    if module == "sqlalchemy":
        bind = conn
        if callable(get_bind := getattr(conn, "get_bind", None)):
            # an ORM Session's dialect is its bind's
            with contextlib.suppress(Exception):
                bind = get_bind()
        dialect = getattr(bind, "dialect", None)
        # (also the forks: TimescaleDB, CockroachDB, ... subclass PGDialect)
        return getattr(dialect, "name", "") in ("postgresql", "redshift") or any(
            c.__name__ == "PGDialect" for c in type(dialect).__mro__
        )
    return False


def _run_statement(conn: Any, sql: str) -> None:
    """Run a statement without a result on a connection, a SQLAlchemy one too."""
    if callable(exec_driver_sql := _exec_driver_sql(conn)):
        options = {"no_parameters": True, "stream_results": False}
        exec_driver_sql(sql, execution_options=options).close()
        return
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    finally:
        with contextlib.suppress(Exception):
            cursor.close()


def _unbuffered_mysql(conn: Any, module: str) -> bool:
    """Whether a MySQL-family connection's cursors stream their result (unbuffered)."""
    with contextlib.suppress(Exception):
        if module in ("pymysql", "MySQLdb"):
            cursors = importlib.import_module(f"{module}.cursors")
            return issubclass(conn.cursorclass, cursors.SSCursor)
        if module == "mysql":
            # mysql.connector: unbuffered unless connected with `buffered=True`
            return not getattr(conn, "_buffered", False)
    return False


def _buffered_cursor(conn: Any, module: str) -> Any:
    """A cursor that reads its whole result on `execute()`."""
    if module == "mysql":
        return conn.cursor(buffered=True)
    cursors = importlib.import_module(f"{module}.cursors")
    return conn.cursor(cursors.Cursor)


def _exec_driver_sql(target: Any) -> Callable[..., Any] | None:
    """`exec_driver_sql` of a SQLAlchemy Connection, or of an ORM Session's."""
    if callable(f := getattr(target, "exec_driver_sql", None)):
        return cast("Callable[..., Any]", f)
    if type(target).__module__.startswith("sqlalchemy.orm") and callable(
        connection := getattr(target, "connection", None)
    ):
        with contextlib.suppress(Exception):
            return cast("Callable[..., Any]", connection().exec_driver_sql)
    return None


def _sql_str(value: Any, *, dialect: str) -> str:
    """A string literal, read the same whatever the session's string settings."""
    s = str(value).replace("'", "''")
    if "\\" not in s:
        return f"'{s}'"
    s = s.replace("\\", "\\\\")
    # MySQL's backslash escapes are on (the setup removes NO_BACKSLASH_ESCAPES); a
    # Postgres escape string doesn't depend on `standard_conforming_strings`
    return f"E'{s}'" if dialect == "postgresql" else f"'{s}'"


# DB-API driver modules, by the dialect whose settings `_session_settings` restores
_DRIVER_DIALECTS = {
    "psycopg": "postgresql",
    "psycopg2": "postgresql",
    "pg8000": "postgresql",
    "pymysql": "mysql",
    "mysql": "mysql",
    "MySQLdb": "mysql",
    "mariadb": "mysql",
    "oracledb": "oracle",
    "cx_Oracle": "oracle",
}


def _session_settings(conn: Any) -> list[str] | None:
    """
    Statements that restore what the session setup changes on a SQLAlchemy or DB-API
    connection; `None` if unknown.
    """

    def rows(sql: str) -> list[Any]:
        if (run := _exec_driver_sql(conn)) is not None:
            options = {"no_parameters": True, "stream_results": False}
            result = run(sql, execution_options=options)
            try:
                return list(result.fetchall())
            finally:
                result.close()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            return list(cursor.fetchall())
        finally:
            with contextlib.suppress(Exception):
                cursor.close()

    def row(sql: str) -> Any:
        return rows(sql)[0]

    dialect = _sqlalchemy_dialect(conn) or _DRIVER_DIALECTS.get(_module_of(conn), "")
    try:
        if dialect == "postgresql":
            names = (
                "application_name",
                "TimeZone",
                "extra_float_digits",
                "default_transaction_read_only",
            )
            values = row(
                "SELECT " + ", ".join(f"current_setting('{n}')" for n in names)
            )
            # `set_config` is allowed in a read-only transaction; values are quoted
            # with standard strings
            sets = (
                f"set_config('{n}', {_sql_str(v, dialect='postgresql')}, false)"
                for n, v in zip(names, values, strict=True)
            )
            return ["SELECT " + ", ".join(sets)]
        if dialect in ("mysql", "mariadb"):
            tz, mode = row("SELECT @@SESSION.time_zone, @@SESSION.sql_mode")
            for var in ("transaction_read_only", "tx_read_only"):
                with contextlib.suppress(Exception):
                    (read_only,) = row(f"SELECT @@SESSION.{var}")
                    break
            else:
                return None
            # backslash escapes are on: the setup removes NO_BACKSLASH_ESCAPES
            return [
                f"SET SESSION time_zone = {_sql_str(tz, dialect='mysql')}, "
                f"SESSION sql_mode = {_sql_str(mode, dialect='mysql')}, "
                f"SESSION {var} = {int(read_only)}"
            ]
        if dialect == "oracle":
            values = {
                str(p): str(v)
                for p, v in rows(
                    "SELECT parameter, value FROM nls_session_parameters "
                    "WHERE parameter IN ('NLS_COMP', 'NLS_SORT')"
                )
            }
            if sorted(values) != ["NLS_COMP", "NLS_SORT"] or not all(
                re.fullmatch(r"\w+", v) for v in values.values()
            ):
                return None
            return [
                f"ALTER SESSION SET NLS_COMP = {values['NLS_COMP']} "
                f"NLS_SORT = {values['NLS_SORT']}"
            ]
    except Exception:
        with contextlib.suppress(Exception):
            conn.rollback()
    return None


def _shutdown_socket(conn: Any) -> None:
    """
    Stops a PyMySQL or pg8000 connection's result: the server stops sending once it
    can't.
    """
    import socket

    # pg8000's `_sock` is a buffered file of its socket `_usock`
    for name in ("_usock", "_sock"):
        sock: Any = getattr(conn, name, None)
        if callable(getattr(sock, "shutdown", None)):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            return


def _cast_null_columns(
    frames: Iterator[DataFrame], schema: Schema
) -> Iterator[DataFrame]:
    """`frames` with each column of dtype `Null` cast to its dtype in `schema`."""
    import polars as pl

    for df in frames:
        casts = [
            pl.col(name).cast(schema[name])
            for name, dtype in df.schema.items()
            if dtype == pl.Null and schema.get(name, pl.Null) != pl.Null
        ]
        yield df.with_columns(casts) if casts else df


def _empty_frame(result: Any, *, utc: bool) -> DataFrame:
    import polars as pl

    names: list[str] = []
    dtypes: list[Any] = []
    with contextlib.suppress(Exception):
        if (description := getattr(result, "description", None)) is not None:
            names = [d[0] for d in description]
            dtypes = _description_dtypes(result, description, utc=utc)
        elif callable(keys := getattr(result, "keys", None)):
            names = [str(k) for k in cast("Any", keys())]
            # a SQLAlchemy result: the DB-API cursor has the types
            if (cursor := getattr(result, "cursor", None)) is not None and (
                description := getattr(cursor, "description", None)
            ) is not None:
                dtypes = _description_dtypes(cursor, description, utc=utc)
    if len(dtypes) != len(names):
        dtypes = [pl.Null] * len(names)
    frame = pl.DataFrame(schema=dict(zip(names, dtypes, strict=True)))
    if _module_of(result) == "databricks":
        frame = _databricks_arrow_frame(result, frame)
    return frame


# Postgres type OIDs whose dtype is fixed, as ADBC reads them. `cursor.description` has
# them for an empty result too, where a row-wise driver has no values to type columns
# by.
_POSTGRES_OID_DTYPES = {
    16: "Boolean",  # bool
    17: "Binary",  # bytea
    18: "String",  # "char"
    19: "String",  # name
    20: "Int64",  # int8
    21: "Int16",  # int2
    23: "Int32",  # int4
    25: "String",  # text
    700: "Float32",  # float4
    701: "Float64",  # float8
    1042: "String",  # bpchar
    1043: "String",  # varchar
    1082: "Date",  # date
    1083: "Time",  # time
    1114: "Datetime",  # timestamp
    1184: "Datetime[UTC]",  # timestamptz (the session time zone is UTC)
    2950: "String",  # uuid, as a table's is read (as text)
    114: "String",  # json, as a table's is read (as text)
    3802: "String",  # jsonb
}

# Postgres array OIDs, by their element's OID: lists of the element dtype.
_POSTGRES_ARRAY_OIDS = {
    1000: 16,
    1001: 17,
    1002: 18,
    1003: 19,
    1016: 20,
    1005: 21,
    1007: 23,
    1009: 25,
    1021: 700,
    1022: 701,
    1014: 1042,
    1015: 1043,
    1182: 1082,
    1183: 1083,
    1115: 1114,
    1185: 1184,
    2951: 2950,
}

# MySQL field types (`cursor.description` of PyMySQL, MySQLdb and mysql.connector), as a
# table's declared types are read (`declared_dtype`): a query's columns read the same.
# These drivers read every integer as an Int64, `FLOAT` as a Python float: the values
# are cast. Integers are `(signed, unsigned)` (see `_mysql_unsigned`).
# Others (strings, which may be binary; decimals; `BIT`) are typed from values.
_MYSQL_FIELD_DTYPES = {
    1: ("Int8", "UInt8"),  # TINY
    2: ("Int16", "UInt16"),  # SHORT
    3: ("Int32", "UInt32"),  # LONG
    4: "Float32",  # FLOAT
    5: "Float64",  # DOUBLE
    7: "Datetime[UTC]",  # TIMESTAMP (the session time zone is UTC)
    8: ("Int64", "UInt64"),  # LONGLONG
    9: ("Int32", "UInt32"),  # INT24
    10: "Date",  # DATE
    11: "Duration",  # TIME (a duration, not a time of day)
    12: "Datetime",  # DATETIME
    13: "Int16",  # YEAR
    14: "Date",  # NEWDATE
}


# Databricks type names (`cursor.description`'s type codes), as its Arrow results type
# them. Left out, as the description is ambiguous (typed from values): `timestamp` is
# `TIMESTAMP_NTZ` too, `string` is `TIME` as well (and intervals, `VARIANT`), and the
# time zone of a `TIMESTAMP` is the session's. Decimals have their own rule.
_DATABRICKS_TYPE_DTYPES = {
    "tinyint": "Int8",
    "smallint": "Int16",
    "int": "Int32",
    "bigint": "Int64",
    "float": "Float32",
    "double": "Float64",
    "boolean": "Boolean",
    "binary": "Binary",
    "date": "Date",
}


# Trino type names (`cursor.description`'s type codes) whose dtype is fixed. Left out
# (typed from values): types with a zone or a variable precision, nested types,
# `json`, `uuid`, intervals.
_TRINO_TYPE_DTYPES = {
    "boolean": "Boolean",
    "tinyint": "Int8",
    "smallint": "Int16",
    "integer": "Int32",
    "bigint": "Int64",
    "real": "Float32",
    "double": "Float64",
    "varbinary": "Binary",
    "date": "Date",
    "json": "String",
}


def _databricks_arrow_frame(cursor: Any, frame: DataFrame) -> DataFrame:
    """
    The empty frame of a `databricks.sql` result, typed by the driver's Arrow schema.

    The cursor's description is ambiguous (`TIMESTAMP_NTZ` reads `timestamp`, `TIME`
    `string`) and lacks nested types; the Arrow table of a fetch has the dtypes of a
    non-empty read, and for zero rows, which consume none, has the schema too (the
    cursor is closed once `ConnectionExecutor` has read an empty result to its end).
    `frame` stays where that fails.
    """
    import polars as pl

    with contextlib.suppress(Exception):
        typed = pl.from_arrow(cursor.fetchmany_arrow(0))
        if isinstance(typed, pl.DataFrame) and typed.columns == frame.columns:
            return typed.clear()
    return frame


def _mysql_session_zone(conn: Any) -> str | None:
    """
    The time zone a live MySQL-family connection reads a `TIMESTAMP` in, as a name
    Polars takes (an IANA name or a fixed offset, `UTC` for UTC's spellings); `None`
    where it isn't known exactly (the server's own zone `SYSTEM` that isn't UTC, an
    abbreviation such as `CEST` or `EST`, which don't tell the zone's rules), or the
    lookup fails.
    """
    try:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT @@session.time_zone, @@system_time_zone")
            row = cursor.fetchone()
        finally:
            cursor.close()
        session, system = (str(v).strip() for v in row)
    except Exception:
        return None
    utc_names = {
        "UTC", "+00:00", "-00:00", "0:00", "Z", "ETC/UTC", "UCT", "ETC/UCT", "ZULU",
        "ETC/ZULU", "UNIVERSAL", "ETC/UNIVERSAL", "COORDINATED UNIVERSAL TIME",
    }  # fmt: skip
    zone = system if session.upper() == "SYSTEM" else session
    if zone.upper() in utc_names:
        return "UTC"
    if session.upper() == "SYSTEM":
        return None  # the OS's zone, by an abbreviation
    if re.fullmatch(r"[+-]\d\d:\d\d", zone):
        return zone
    try:
        import zoneinfo

        zoneinfo.ZoneInfo(zone)
    except Exception:
        return None
    # (a bare abbreviation that is also a fixed zone, `EST`, may be a daylight zone's)
    return zone if "/" in zone else None


def _description(result: Any) -> Any:
    """A result's DB-API description (a SQLAlchemy result: its cursor's)."""
    description = getattr(result, "description", None)
    if description is None:
        description = getattr(getattr(result, "cursor", None), "description", None)
    return description


def _time_with_zone_columns(result: Any) -> list[str]:
    """
    The names of a result's columns that are or hold a `TIMETZ` where the description
    tells (DuckDB's type name, Postgres drivers' OIDs 1266 and 1270 for its array): the
    drivers return such a value as the local time of day, its offset dropped.
    """
    module = _module_of(
        result if hasattr(result, "description") else getattr(result, "cursor", result)
    )
    with contextlib.suppress(Exception):
        return [
            d[0]
            for d in _description(result)
            if (module == "duckdb" and _duckdb_holds_timetz(d[1]))
            or (
                module in ("psycopg", "psycopg2", "pg8000")
                and type(d[1]) is int
                and d[1] in (1266, 1270)
            )
        ]
    return []


def _duckdb_holds_timetz(dtype: Any) -> bool:
    """Whether a DuckDB result type is or holds a `TIMETZ` (field names aside)."""
    if isinstance(dtype, str):
        return "TIME WITH TIME ZONE" in dtype.upper()
    kind = getattr(dtype, "id", None)
    if kind == "time with time zone":
        return True
    if kind not in ("list", "array", "struct", "map", "union"):
        return False
    return any(_duckdb_holds_timetz(child) for _, child in dtype.children)


def _mysql_timestamp_columns(result: Any) -> list[str]:
    """The names of a MySQL-family result's `TIMESTAMP` columns (field type 7)."""
    description = _description(result)
    with contextlib.suppress(Exception):
        return [d[0] for d in description if type(d[1]) is int and d[1] == 7]
    return []


def _mysql_timestamps_to_utc(
    frames: Iterator[DataFrame], columns: list[str], zone: str
) -> Iterator[DataFrame]:
    """
    Each batch's `TIMESTAMP` columns (the session's wall-clock time, naive) as the
    instants they are, in UTC. A wall-clock time the zone has twice (the hour its
    clocks go back) can't tell which instant it is: an error, never a guess.
    """
    import datetime as dt

    import polars as pl

    offset = None
    if zone == "UTC":
        offset = dt.timedelta(0)
    elif re.fullmatch(r"[+-]\d\d:\d\d", zone):
        hours, minutes = int(zone[1:3]), int(zone[4:6])
        offset = dt.timedelta(hours=hours, minutes=minutes) * (
            -1 if zone[0] == "-" else 1
        )

    def instant(c: str) -> Any:
        naive = pl.col(c).cast(pl.Datetime("us"))
        if offset is not None:
            return (naive - offset).dt.replace_time_zone("UTC")
        return naive.dt.replace_time_zone(
            zone, ambiguous="raise", non_existent="raise"
        ).dt.convert_time_zone("UTC")

    for df in frames:
        present = [c for c in columns if c in df.columns]
        try:
            yield df.with_columns(instant(c) for c in present)
        except pl.exceptions.PolarsError as e:
            msg = (
                f"the TIMESTAMP column(s) {present} are read in this connection's time "
                f"zone ({zone}), and a value in it can't be made an instant (a "
                f"wall-clock time the zone has twice?): {e}. Run SET time_zone = "
                "'+00:00' on the connection first"
            )
            raise RuntimeError(msg) from e


def _mysql_unsigned(cursor: Any, description: Any, i: int) -> bool | None:
    """Whether a MySQL result column is `UNSIGNED`; `None` if unknown."""
    unsigned_flag = 32
    with contextlib.suppress(Exception):
        if _module_of(cursor) == "mysql":
            # mysql.connector: the column flags are the description's 8th item
            return bool(description[i][7] & unsigned_flag)
        if _module_of(cursor) == "pymysql":
            return bool(cursor._result.fields[i].flags & unsigned_flag)
    return None


_DUCKDB_SCALAR_DTYPES = {
    "BOOLEAN": "Boolean",
    "TINYINT": "Int8",
    "SMALLINT": "Int16",
    "INTEGER": "Int32",
    "BIGINT": "Int64",
    "HUGEINT": "Int128",
    "UTINYINT": "UInt8",
    "USMALLINT": "UInt16",
    "UINTEGER": "UInt32",
    "UBIGINT": "UInt64",
    "FLOAT": "Float32",
    "DOUBLE": "Float64",
    "VARCHAR": "String",
    "BLOB": "Binary",
    "DATE": "Date",
    "TIME": "Time",
}


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """`text` split at `sep` outside parentheses, brackets and double quotes."""
    parts: list[str] = []
    depth = 0
    quoted = False
    start = 0
    for i, ch in enumerate(text):
        if ch == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _duckdb_type_dtype(type_name: str, *, nested_only: bool) -> Any:
    """
    The Polars dtype of a DuckDB type name, as `cursor.description` shows it.

    `Null` for a type this doesn't read (a map, a union, a time zone...): the column is
    then typed as before. With `nested_only`, a scalar type is `Null` too.
    """
    import polars as pl

    name = type_name.strip()
    try:
        if name.endswith("]") and (i := name.rfind("[")) > 0:
            element = _duckdb_type_dtype(name[:i], nested_only=False)
            if element == pl.Null:
                return pl.Null
            size = name[i + 1 : -1].strip()
            if not size:
                return pl.List(element)
            return pl.Array(element, int(size)) if size.isdigit() else pl.Null
        if name.upper().startswith("STRUCT(") and name.endswith(")"):
            fields: dict[str, Any] = {}
            for field in _split_top_level(name[len("STRUCT(") : -1]):
                if field.startswith('"'):
                    end = 1
                    while end < len(field):
                        if field[end] == '"':
                            if field[end + 1 : end + 2] != '"':
                                break
                            end += 1
                        end += 1
                    field_name = field[1:end].replace('""', '"')
                    field_type = field[end + 1 :]
                else:
                    field_name, _, field_type = field.partition(" ")
                dtype = _duckdb_type_dtype(field_type, nested_only=False)
                if dtype == pl.Null:
                    return pl.Null
                fields[field_name] = dtype
            return pl.Struct(fields) if fields else pl.Null
        if nested_only:
            return pl.Null
        upper = name.upper()
        if (scalar := _DUCKDB_SCALAR_DTYPES.get(upper)) is not None:
            return getattr(pl, scalar)
        if upper == "TIMESTAMP":
            return pl.Datetime("us")
        if m := re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", upper):
            p, s = int(m.group(1)), int(m.group(2))
            return pl.Decimal(p, s) if 0 < p <= 38 else pl.Null
    except (ValueError, TypeError):
        pass
    return pl.Null


def _description_dtypes(cursor: Any, description: Any, *, utc: bool) -> list[Any]:
    """Dtypes of a Postgres-, MySQL-family or Databricks result's columns.

    From their type codes, which an empty result has too; `Null` (typed later from a
    sample) for other types and other drivers. `utc`: the session time zone is UTC (the
    session setup ran).
    """
    import polars as pl

    def named(name: str) -> Any:
        if name.startswith("Datetime"):
            return pl.Datetime("us", "UTC" if name.endswith("[UTC]") else None)
        return getattr(pl, name)

    def postgres(d: Any) -> Any:
        type_code = d[1]
        if type(type_code) is not int:
            return pl.Null
        if type_code == 1700:  # numeric(p, s); without a precision, typed from values
            p, s = d[4], d[5]
            if type(p) is int and type(s) is int and 0 < p <= 38 and 0 <= s <= p:
                return pl.Decimal(p, s)
            return pl.Null
        if (element := _POSTGRES_ARRAY_OIDS.get(type_code)) is not None:
            return pl.List(named(_POSTGRES_OID_DTYPES[element]))
        name = _POSTGRES_OID_DTYPES.get(type_code)
        return pl.Null if name is None else named(name)

    def mysql(i: int, d: Any) -> Any:
        if d[1] == 16 and _module_of(cursor) == "pymysql":
            return pl.Binary  # BIT: pymysql reads it as bytes, as with rows
        name = _MYSQL_FIELD_DTYPES.get(d[1]) if type(d[1]) is int else None
        if isinstance(name, tuple):
            unsigned = _mysql_unsigned(cursor, description, i)
            if unsigned is None:
                return pl.Null  # typed from values
            name = name[unsigned]
        if name == "Datetime[UTC]" and not utc:
            # in the session's time zone, which is unknown: as read
            name = "Datetime"
        return pl.Null if name is None else named(name)  # type: ignore[arg-type]

    def databricks(d: Any) -> Any:
        type_name = str(d[1]).lower()
        if type_name == "decimal":
            p, s = d[4], d[5]
            if type(p) is int and type(s) is int and 0 < p <= 38 and 0 <= s <= p:
                return pl.Decimal(p, s)
            return pl.Null
        name = _DATABRICKS_TYPE_DTYPES.get(type_name)
        return pl.Null if name is None else named(name)

    def trino(d: Any) -> Any:
        # `d[1]` is Trino's type name: `bigint`, `decimal(10,2)`, `varchar(5)`
        type_name = str(d[1]).lower().strip()
        if (m := re.fullmatch(r"decimal\((\d+),\s*(\d+)\)", type_name)) is not None:
            p, s = int(m[1]), int(m[2])
            return pl.Decimal(p, s) if 0 < p <= 38 and 0 <= s <= p else pl.Null
        # (a precision above 6 is read as text, and an instant in UTC: the planner's
        # conversions, `Dialect.text_conversion`, `column_conversion`)
        if (
            m := re.fullmatch(r"timestamp\((\d+)\)( with time zone)?", type_name)
        ) is not None:
            if int(m[1]) > 6:
                return pl.String
            return pl.Datetime("us", "UTC") if m[2] else pl.Datetime("us")
        if (m := re.fullmatch(r"time\((\d+)\)", type_name)) is not None:
            return pl.Time if int(m[1]) <= 6 else pl.String
        if re.fullmatch(r"(varchar|char)(\(\d+\))?", type_name):
            return pl.String
        name = _TRINO_TYPE_DTYPES.get(type_name)
        return pl.Null if name is None else getattr(pl, name)

    module = _module_of(cursor)
    if module == "trino":
        return [trino(d) for d in description]
    if module == "duckdb":
        # (only nested types: a scalar column is typed from the catalog's type)
        return [_duckdb_type_dtype(str(d[1]), nested_only=True) for d in description]
    if module == "databricks":
        return [databricks(d) for d in description]
    if module in ("psycopg", "psycopg2", "pg8000"):
        return [postgres(d) for d in description]
    if module in ("pymysql", "MySQLdb", "mysql"):
        return [mysql(i, d) for i, d in enumerate(description)]
    return [pl.Null] * len(description)


# Names of server-side cursors, unique in the process.
_CURSOR_IDS = itertools.count()


class _ConnectionState:
    """Shared by every session over one connection."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # executions in progress
        self.active = 0


# A live connection is the user's one object: the scans of a query (a self-join, a
# union) each have a session over it and read at the same time, from other threads.
_LIVE_STATES: weakref.WeakKeyDictionary[Any, _ConnectionState] = (
    weakref.WeakKeyDictionary()
)
_LIVE_STATES_BY_ID: dict[int, _ConnectionState] = {}
_LIVE_STATES_LOCK = threading.Lock()

# The connections (by identity) of the factory's open sessions, with how many sessions
# hold each: a factory may return one connection again.
_FACTORY_CONNECTIONS: dict[int, int] = {}
# reentrant: a session's finalizer may run (garbage collection) inside it
_FACTORY_LOCK = threading.RLock()


def _release_factory_key(key: int) -> None:
    with _FACTORY_LOCK:
        if (left := _FACTORY_CONNECTIONS.get(key, 1) - 1) > 0:
            _FACTORY_CONNECTIONS[key] = left
        else:
            _FACTORY_CONNECTIONS.pop(key, None)


def _live_state(conn: Any) -> _ConnectionState:
    with _LIVE_STATES_LOCK:
        try:
            return _LIVE_STATES.setdefault(conn, _ConnectionState())
        except TypeError:
            # not weak-referenceable or not hashable: kept for the process
            return _LIVE_STATES_BY_ID.setdefault(id(conn), _ConnectionState())


_CANCELLED = "the database read was cancelled"

# A cancel asks again until its driver call returns (the stopped statement ends it),
# however long that takes: a statement past the check of `_cancelled` but not yet at the
# driver isn't there to stop. At interpreter exit every cancel stops asking after this
# long, together: a driver call that ignores the interrupt must not hold the exit.
_EXIT_CANCEL_SECONDS = 2.0
_exit_cancel_deadline: float | None = None


def begin_exit_cancel() -> None:
    """Ends the retries of every cancel `_EXIT_CANCEL_SECONDS` from now (`_at_exit`)."""
    global _exit_cancel_deadline
    if _exit_cancel_deadline is None:
        _exit_cancel_deadline = time.monotonic() + _EXIT_CANCEL_SECONDS


def end_exit_cancel() -> None:
    """
    Lifts the bound: a scan in a user's `atexit` hook after `_at_exit` cancels as
    usual.
    """
    global _exit_cancel_deadline
    _exit_cancel_deadline = None


def _cancel_expired() -> bool:
    deadline = _exit_cancel_deadline
    return deadline is not None and time.monotonic() >= deadline


_at_exit_before_duckdb = False
_at_exit_before_duckdb_lock = threading.Lock()


def _register_at_exit_before_duckdb(conn: Any) -> None:
    """
    Registers `_at_exit` once more for the first DuckDB connection, so it runs before
    DuckDB's own exit hooks, which its import registered (the last registered runs
    first).

    DuckDB 2 closes its databases at exit, which waits for their running statements:
    `_at_exit` must have interrupted them by then. The registration from this
    package's import stays, as a registration made during exit never runs.
    """
    global _at_exit_before_duckdb
    if _at_exit_before_duckdb or (
        _module_of(conn) != "duckdb" and _sqlalchemy_dialect(conn) != "duckdb"
    ):
        return
    from polars_database import _run_at_exit

    with _at_exit_before_duckdb_lock:
        if not _at_exit_before_duckdb:
            _at_exit_before_duckdb = True
            atexit.register(_run_at_exit)


class _Session:
    """A connection used by one partition reader at a time."""

    can_cancel = False

    def __init__(
        self,
        conn: Any,
        *,
        owned: bool,
        execute_options: dict[str, Any],
        close: Callable[[], None] | None = None,
        params: Any = None,
        factory: bool = False,
        invalidate: Callable[[], None] | None = None,
    ) -> None:
        _register_at_exit_before_duckdb(conn)
        self.conn = conn
        self.owned = owned
        # from the user's function, whose `close` may return it to a pool of theirs
        self.factory = factory
        self.execute_options = execute_options
        self.params = params
        self._close = close
        # discards it from its pool instead (when its socket was shut down)
        self._invalidate = invalidate
        self._state = _ConnectionState() if owned else _live_state(conn)
        self._factory_key: int | None = None
        if factory:
            self._register_factory_connection()
        # the session setup changed the connection's settings (see `close`)
        self._setup_ran = False
        # statements that restore what the setup changed on a pooled connection
        # (`None`: unknown, so the connection is discarded instead)
        self._restore: list[str] | None = None
        # stopped by `cancel` through its socket: the connection is unusable
        self._aborted = False
        # `cancel` is sticky: a statement not yet issued then never is (a cancel before
        # it started had nothing to stop). A cancelled session is never given back to a
        # pool, so the flag can't reach another scan.
        self._cancel_lock = threading.Lock()
        self._cancelled = False
        # what runs the statement where that isn't the connection: a DuckDB cursor
        # (a connection of its own, which `interrupt` on `conn` doesn't reach)
        self._running: Any = None
        # between the check of `_cancelled` and the driver call's return
        self._issuing = False
        module = _module_of(conn)
        # PyMySQL and pg8000 have no cancel, and closing PyMySQL's unbuffered cursor
        # reads the rest of the result; a cancelled session is never reused, so its
        # socket can go
        self._abort_socket = owned and module in ("pymysql", "pg8000")
        # A live connection is never cancelled: a cancel stops whatever runs on it (the
        # user's next statement, once ours ended), and on Postgres aborts the
        # transaction. Its scans' cursors are closed instead.
        self.can_cancel = owned and (
            self._abort_socket
            or any(
                callable(getattr(_cancel_target(conn), m, None))
                for m in ("cancel", "interrupt")
            )
        )
        # the last statement's result columns that are a `TIMETZ` (offset dropped)
        self.timetz_columns: list[str] = []
        # a live MySQL driver's session reads a TIMESTAMP in its own time zone, which
        # `execute` converts to UTC (a SQLAlchemy Connection is its driver's connection)
        self.mysql_converts = not owned and _module_of(_cancel_target(conn)) in (
            "pymysql",
            "MySQLdb",
            "mysql",
            "mariadb",
        )
        # SQLAlchemy's `text()` reads `:name` and `\:` inside the string literals
        # written into the SQL; without parameters, the SQL goes to the driver as is
        self._raw_sqlalchemy = (
            module == "sqlalchemy"
            and params is None
            and "parameters" not in execute_options
            and "params" not in execute_options
        )

    def execute(
        self, sql: str, batch_size: int | None, bind: bool = True
    ) -> Iterator[DataFrame]:
        """
        Run a statement. `bind`: the user's parameters (`params=`, or the execute
        options' `parameters`) are bound to it; the plugin's own statements (catalog
        lookups) hold none of the user's placeholders.
        """
        from polars.io.database._executor import ConnectionExecutor

        batch_size = batch_size or DEFAULT_BATCH_SIZE
        params = self.params if bind else None
        raw_sqlalchemy = self._raw_sqlalchemy or (
            not bind and _module_of(self.conn) == "sqlalchemy"
        )
        state = self._state
        self.timetz_columns = []
        entered = False
        cursor = None
        cx_cursor: Any = None
        cx: Any = None
        frames: Iterator[DataFrame] | None = None
        idle = False
        savepoint: str | None = None
        try:
            # driver calls hold the lock, yields don't: other scans of the same live
            # connection interleave their calls, and none waits for another's consumer
            with state.lock:
                state.active += 1
                entered = True
                if not self.owned:
                    _refuse_pending_results(self.conn)
                # A live connection outside a transaction is left without one. Polars
                # may release a read after `collect()` returned (an early stop), while
                # the user runs statements on the connection again: a ROLLBACK then
                # could end theirs. So the result is read whole and the transaction it
                # opened rolled back here, before any batch is handed over.
                status = None if self.owned else _transaction_idle(self.conn)
                idle = status is True
                in_database = status is False
                if in_database and _module_of(self.conn) == "sqlalchemy":
                    # a SQLAlchemy transaction in AUTOCOMMIT (SQLAlchemy's, or the
                    # driver's own) has none in the database
                    run_on = (
                        _session_connection(self.conn)
                        if _is_orm_session(self.conn)
                        else self.conn
                    )
                    driver = getattr(
                        getattr(run_on, "connection", None), "driver_connection", None
                    )
                    in_database = not (
                        driver is not None
                        and (
                            _sqlalchemy_autocommit(run_on)
                            or getattr(driver, "autocommit", False) is True
                        )
                        and _transaction_idle(driver) is True
                    )
                if in_database and _postgres_family(self.conn):
                    # In the user's transaction, a failed statement (a catalog lookup
                    # of a name that isn't there, a bad query) would abort it, their
                    # writes lost: it runs in a savepoint, rolled back to on an error.
                    name = f"polars_scan_{next(_CURSOR_IDS)}"
                    _run_statement(self.conn, f"SAVEPOINT {name}")
                    savepoint = name
                cursor = self._server_cursor(batch_size)
                cx = ConnectionExecutor(cursor if cursor is not None else self.conn)
                # before the statement: a connection reads one result at a time
                mysql_live = self.mysql_converts
                live_zone = (
                    _mysql_session_zone(_cancel_target(self.conn))
                    if mysql_live
                    else None
                )
                timestamps: list[str] = []
                # A DuckDB cursor is a connection of its own: it sees neither the
                # user's connection's temporary tables, views and macros nor its
                # uncommitted writes (a factory's connection has the same). Where it
                # has any, the statement runs on that connection, and its result is
                # read whole here (another statement on it, another scan's, would end
                # it). (A failing statement then aborts an open transaction.)
                # Otherwise on a cursor, which gets the connection's session state
                # first (`search_path`, time zone, variables), before any lookup of the
                # catalog.
                on_connection = (
                    (not self.owned or self.factory)
                    and cx.driver_name == "duckdb"
                    and _module_of(self.conn) == "duckdb"
                    and (
                        _duckdb_needs_connection(self.conn)
                        or not _duckdb_copy_session(self.conn, cx.cursor)
                    )
                )
                if on_connection:
                    with contextlib.suppress(Exception):
                        cx.cursor.close()
                    cx.cursor = self.conn
                    cx.can_close_cursor = False
                    # `execute` would replace a result of the user's left unread (DuckDB
                    # 1): a relation leaves it, and DuckDB 2 refuses
                    cx._duckdb_has_lazy_relations = lambda: True  # type: ignore[method-assign]
                cx_cursor = cx.cursor if cx.cursor is not cursor else None
                options = dict(self.execute_options)
                if not bind:
                    for key in {"parameters", "params", _params_keyword(cx)}:
                        options.pop(key, None)
                oracle_arrow = (
                    cx.driver_name == "oracledb_proxy"
                    or cx._oracle_arrow_proxy() is not None
                )
                if oracle_arrow:
                    # NUMBER(p, s) as decimals, not floats, like the other drivers
                    options.setdefault("fetch_decimals", True)
                elif (raw := _oracle_row_cursor(cx)) is not None:
                    # Without Arrow (oracledb < 3.4, cx_Oracle) NUMBER(p, s) is
                    # fetched as a float, which rounds past 15 digits: as decimals
                    # instead, on a cursor of the driver's own
                    if raw is not cx.cursor:
                        cx = ConnectionExecutor(raw)
                        cx_cursor = raw
                    raw.outputtypehandler = _oracle_decimal_handler(raw)
                if (
                    _module_of(self.conn) == "clickhouse_connect"
                    and "parameters" not in options
                    and params is None
                ):
                    # it %-formats the SQL even without parameters, which would turn
                    # LIKE's '%%' into '%'; with parameters, '%' is the user's to write
                    sql = sql.replace("%", "%%")
                if params is not None:
                    # each driver names the parameters argument of `execute()` itself
                    options[_params_keyword(cx)] = params
                if (
                    _module_of(self.conn) == "sqlalchemy"
                    and not oracle_arrow
                    and cx.driver_name == "sqlalchemy"
                    and cx._is_alchemy_object(cx.cursor)
                ):
                    # (psycopg2's cursor behind SQLAlchemy, see the guard)
                    _sqlalchemy_psycopg2_guard()
                    options["execution_options"] = {
                        **(options.get("execution_options") or {}),
                        _SCAN_OPTION: True,
                    }
                runner = (
                    cx.cursor
                    if cx.driver_name == "duckdb" and cx.cursor is not self.conn
                    else None
                )
                with self._cancel_lock:
                    if self._cancelled:
                        raise RuntimeError(_CANCELLED)
                    self._running = runner
                    self._issuing = True
                exec_driver_sql = (
                    _exec_driver_sql(cx.cursor)
                    if raw_sqlalchemy
                    and not oracle_arrow
                    and cx.driver_name in ("sqlalchemy", "duckdb")
                    else None
                )
                if exec_driver_sql is not None:
                    # as `cx.execute`, but without `text()` (see `__init__`); with
                    # `no_parameters`, `%`-style drivers don't format the SQL either
                    # (`text()` escaped `%` for them)
                    execution_options = options.pop("execution_options", None) or {}
                    result = exec_driver_sql(
                        sql,
                        execution_options={**execution_options, "no_parameters": True},
                        **options,
                    )
                    if cx.driver_name == "duckdb" and cx._is_alchemy_result(result):
                        result = result.cursor
                    cx.result = result
                else:
                    cx.execute(sql, options=options or None)
                if cursor is not None and cursor.description is None:
                    # psycopg2's named cursors describe the result only after a fetch;
                    # `FETCH FORWARD 0` returns no row
                    cursor.fetchmany(0)
                frames = _object_columns_as_text(
                    _no_panics(
                        cx.to_polars(  # type: ignore[arg-type]
                            iter_batches=True,
                            batch_size=batch_size,
                            # every row types a batch: a column whose first rows are
                            # NULL would fail on its first value otherwise
                            infer_schema_length=None,
                        )
                    )
                )
                # A live MySQL session reads a TIMESTAMP in its own time zone, as naive
                # wall-clock time: the instants (UTC), as the table source reads them
                self.timetz_columns = _time_with_zone_columns(cx.result)
                timestamps = _mysql_timestamp_columns(cx.result) if mysql_live else []
                if timestamps:
                    if live_zone is None:
                        msg = (
                            f"the TIMESTAMP column(s) {timestamps} are read in this "
                            "connection's time zone, which can't be told exactly (the "
                            "server's own zone, or an abbreviation): run SET time_zone "
                            "= '+00:00' on the connection first, or give an IANA name "
                            "or an offset"
                        )
                        raise RuntimeError(msg)
                    frames = _mysql_timestamps_to_utc(frames, timestamps, live_zone)
                if _module_of(cx.result) == "databricks":
                    frames = _valid_text(frames)
                # some drivers yield no batch for an empty result, and close the
                # cursor when done: take the column names first
                empty_frame = _empty_frame(
                    cx.result, utc=self.owned or bool(timestamps)
                )
                if _module_of(cx.result) == "trino":
                    # a Trino batch types a column of NULLs only as `Null`
                    frames = _cast_null_columns(frames, empty_frame.schema)
                if on_connection:
                    frames = iter(list(frames))
                if (
                    not self.owned
                    and _module_of(_cancel_target(self.conn)) == "pymssql"
                ):
                    # pymssql has no MARS: a connection runs one statement at a time,
                    # and another scan's statement while this result is unread would
                    # discard it (no rows, the other's rows, or a shape error). Read
                    # whole here, under the lock.
                    frames = iter(list(frames))
                if idle and _transaction_idle(self.conn) is False:
                    frames = iter(list(frames))
                    if _module_of(cx.result) == "sqlalchemy":
                        with contextlib.suppress(Exception):
                            cx.result.close()
                    self.conn.rollback()
                if savepoint is not None:
                    # read whole by the client-side cursor already
                    frames = iter(list(frames))
                    _run_statement(self.conn, f"RELEASE SAVEPOINT {savepoint}")
                    savepoint = None
            empty = True
            while True:
                # Every server-side fetch may issue another statement. Register it
                # before entering the driver: cancellation is sticky and is repeated
                # through delayed startup until this fetch returns.
                with self._cancel_lock:
                    if self._cancelled:
                        raise RuntimeError(_CANCELLED)
                    self._issuing = True
                try:
                    with state.lock:
                        df = next(frames, None)
                finally:
                    with self._cancel_lock:
                        self._issuing = False
                if df is None:
                    break
                empty = False
                yield df
            if empty:
                yield empty_frame
        except Exception:
            # A failed statement aborts a Postgres transaction: the session's next one
            # (another catalog lookup) would fail with `InFailedSqlTransaction`, hiding
            # this error. Rolled back where the transaction is the scan's.
            if (self.owned and not self._aborted) or idle:
                with state.lock, contextlib.suppress(Exception):
                    self.conn.rollback()
            elif savepoint is not None:
                with state.lock, contextlib.suppress(Exception):
                    _run_statement(self.conn, f"ROLLBACK TO SAVEPOINT {savepoint}")
                    _run_statement(self.conn, f"RELEASE SAVEPOINT {savepoint}")
            raise
        finally:
            # Also runs when the reader stops early and drops this generator, which
            # Polars may do after `collect()` returned: on a live connection, nothing
            # here may then send a statement that could fail in, or end, a transaction
            # of the user's.
            with self._cancel_lock:
                self._running = None
                self._issuing = False
            with state.lock:
                if (
                    self._aborted
                    and (result := getattr(self.conn, "_result", None)) is not None
                ):
                    # the socket is gone: nothing more to read (closing the cursor,
                    # or the result's `__del__`, would try)
                    result.unbuffered_active = False
                if frames is not None:
                    with contextlib.suppress(Exception):
                        frames.close()  # type: ignore[attr-defined]
                if cursor is not None:
                    with contextlib.suppress(Exception):
                        cursor.close()
                if (
                    cx is not None
                    and _module_of(cx.result) == "sqlalchemy"
                    and callable(getattr(cx.result, "close", None))
                ):
                    # a SQLAlchemy result (a server-side cursor with `stream_results`)
                    # left unfinished: closed before its connection goes on
                    with contextlib.suppress(Exception):
                        cx.result.close()
                if cx_cursor is not None and cx_cursor is not self.conn:
                    # the executor's own cursor (or an Engine's connection), which it
                    # closes only after the last batch
                    with contextlib.suppress(Exception):
                        cx_cursor.close()
                if entered:
                    state.active -= 1
                if self.owned and cursor is not None and not self._aborted:
                    with contextlib.suppress(Exception):
                        # the server-side cursor's transaction holds a snapshot
                        self.conn.rollback()

    def _reset_pooled(self) -> None:
        """
        A pooled connection (an Engine's) goes back to the user's pool: the settings
        the session setup changed (READ ONLY, UTC, sql_mode, ...) are restored, or the
        connection is discarded.
        """
        conn = self.conn
        if not callable(invalidate := getattr(conn, "invalidate", None)):
            return
        if self._restore is not None:
            try:
                conn.rollback()
                for sql in self._restore:
                    # not on the server-side cursor that `stream_results` would open
                    options = {"no_parameters": True, "stream_results": False}
                    conn.exec_driver_sql(sql, execution_options=options).close()
                conn.commit()
            except Exception:
                pass
            else:
                return
        with contextlib.suppress(Exception):
            invalidate()

    def _server_cursor(self, batch_size: int) -> Any:
        """A server-side cursor, so rows stream instead of being fetched up front."""
        module = _module_of(self.conn)
        if not self.owned:
            # No server-side cursor on the user's connection: Polars may go on reading
            # (or stop) after `collect()` returned, while the user runs statements on
            # it again; a FETCH or CLOSE then, after their transaction ended, would
            # fail in their next one. A client-side cursor reads the result whole.
            if _unbuffered_mysql(self.conn, module):
                # MySQL can't keep two results open on one connection: another scan's
                # statement would end this one's result early, silently with PyMySQL
                return _buffered_cursor(self.conn, module)
            if module == "psycopg2":
                # a client-side cursor of the scan's own: its readings of bytea and
                # infinity are the cursor's, and the user's statements keep theirs
                cursor = self.conn.cursor()
                _psycopg2_bytes(cursor)
                _psycopg2_no_infinity(cursor)
                return cursor
            return None
        if module in ("psycopg", "psycopg2"):
            name = f"polars_scan_{next(_CURSOR_IDS)}"
            if getattr(self.conn, "autocommit", False) is True:
                # a named cursor needs a transaction, and there is none in autocommit:
                # one that outlives each statement
                cursor = self.conn.cursor(name=name, withhold=True)
            else:
                cursor = self.conn.cursor(name=name)
            cursor.itersize = batch_size
            if module == "psycopg2":
                # also for a connection the plugin didn't open (factory, live)
                _psycopg2_bytes(cursor)
                _psycopg2_no_infinity(cursor)
            return cursor
        if module == "pymysql":
            import pymysql.cursors  # type: ignore[import-untyped]

            return self.conn.cursor(pymysql.cursors.SSCursor)
        if module == "sqlalchemy" and callable(
            opts := getattr(self.conn, "execution_options", None)
        ):
            # the dialect's own streaming cursor (named, SSCursor, ...), but not in
            # AUTOCOMMIT: a Postgres named cursor needs a transaction there is none of
            autocommit = getattr(self.conn, "_is_autocommit_isolation", None)
            if not (
                callable(autocommit) and autocommit() and _postgres_family(self.conn)
            ):
                opts(stream_results=True, max_row_buffer=batch_size)
        return None

    def execute_update(self, sql: str) -> None:
        """Run a session setup statement (read-only transactions, time zone, ...)."""
        conn = self.conn
        if not self.owned:
            # a live connection is the user's: its session settings are left alone
            return
        if _module_of(conn) == "sqlalchemy":
            if not self._setup_ran and self._close is None:
                # an Engine's pooled connection: its settings are restored on `close`
                self._restore = _session_settings(conn)
            self._setup_ran = True
            # a Connection's, or an ORM Session's (a `sessionmaker` factory)
            run = _exec_driver_sql(conn)
            if run is None:
                msg = f"can't run SQL on a SQLAlchemy {type(conn).__name__}"
                raise TypeError(msg)
            run(sql)
            conn.commit()
            return
        if self.factory and not self._setup_ran:
            # restored on `close`, in case it goes back to a pool
            self._restore = _session_settings(conn)
        self._setup_ran = True
        cursor = conn.cursor() if hasattr(conn, "cursor") else conn
        try:
            cursor.execute(sql)
        finally:
            if cursor is not conn:
                with contextlib.suppress(Exception):
                    cursor.close()
        # drivers that open a transaction implicitly (psycopg, pymysql, ...) apply
        # session characteristics like READ ONLY to the transactions after this one
        if callable(commit := getattr(conn, "commit", None)):
            with contextlib.suppress(Exception):
                commit()

    def cancel(self) -> None:
        if not self.owned:
            # see `can_cancel`
            return
        with self._cancel_lock:
            self._cancelled = True
            running = self._running
        if self._abort_socket:
            self._aborted = True
            _shutdown_socket(self.conn)
            return
        target = running if running is not None else _cancel_target(self.conn)
        for m in ("cancel", "interrupt"):
            if callable(f := getattr(target, m, None)):
                break
        else:
            return
        delay = 0.02
        while True:
            try:
                f()
            except Exception:
                # the statement ended, and its cursor with it, meanwhile
                if running is None or self._running is running:
                    raise
                return
            # A statement past the check of `_cancelled` but not yet at the driver isn't
            # there to stop: asked again until the driver call returns (the stopped
            # statement ends it).
            if not self._issuing or _cancel_expired():
                return
            # a cursor's cancel may open a connection of its own: backing off
            time.sleep(delay)
            delay = min(delay * 2, 0.25)

    def _register_factory_connection(self) -> None:
        """
        Counts the factory's connection among the open sessions'. One that another open
        session holds is refused, for any driver: two sessions on one connection end
        each other's results (PyMySQL's unbuffered cursors, silently), transactions
        and named cursors (`commit()`, `rollback()`), or close it under the other.
        """
        key = id(self.conn)
        with _FACTORY_LOCK:
            if key in _FACTORY_CONNECTIONS:
                msg = (
                    "the connection function returned a connection that another read "
                    "of this scan, or of another scan in the same query, still uses, "
                    "and sessions can't share one: return a new connection on each "
                    "call, or pass the connection itself (max_connections=1 only helps "
                    "within one scan)"
                )
                raise ValueError(msg)
            _FACTORY_CONNECTIONS[key] = _FACTORY_CONNECTIONS.get(key, 0) + 1
            self._factory_key = key
        # a session dropped without `close()` must not keep refusing the connection
        # (or a later object at the same `id()`)
        self._factory_finalizer = weakref.finalize(self, _release_factory_key, key)

    def _release_factory_connection(self) -> None:
        if self._factory_key is None:
            return
        self._factory_key = None
        # runs once, here or when the session is collected
        self._factory_finalizer()

    def close(self) -> None:
        try:
            self._close_connection()
        finally:
            # only now another session may take the connection: while the driver closes
            # it, it is still this one's
            self._release_factory_connection()

    def _close_connection(self) -> None:
        if self._setup_ran and self._close is None:
            self._reset_pooled()
        # A factory's connection may go back to a pool of the user's: the settings the
        # setup changed are restored, or, when that is unknown or fails, it is discarded
        # (invalidated, or its own connection closed) rather than handed back changed.
        discard = self._aborted
        if (
            self._setup_ran
            and self.factory
            and not self._aborted
            and _module_of(self.conn) != "sqlalchemy"  # (`_reset_pooled`)
        ):
            discard = True
            if self._restore:
                try:
                    self.conn.rollback()
                    for sql in self._restore:
                        _run_statement(self.conn, sql)
                    self.conn.commit()
                except Exception:
                    pass
                else:
                    discard = False
        if discard and not self._aborted and not _is_psycopg_pooled(self.conn):
            # (a psycopg_pool connection is closed once, by `_invalidate`)
            _close_for_real(self.conn)
        # The connection is the session's until it is given back: once it is in a pool
        # another reader may be handed it at once, so it is released just before that. A
        # connection the session closes itself is released after the driver close.
        if discard and self._invalidate is not None:
            self._release_factory_connection()
            with contextlib.suppress(Exception):
                self._invalidate()
        elif self._close is not None:
            self._release_factory_connection()
            self._close()
        elif self.owned and callable(getattr(self.conn, "close", None)):
            if discard:
                with contextlib.suppress(Exception):
                    self.conn.close()
            else:
                if (
                    getattr(self.conn, "_cnx", None) is not None
                    or getattr(self.conn, "_pool", None) is not None
                ):
                    # pooled (mysql.connector, oracledb): `close()` gives it back
                    self._release_factory_connection()
                self.conn.close()


def _is_psycopg_pooled(conn: Any) -> bool:
    pool = getattr(conn, "_pool", None)
    return pool is not None and type(pool).__module__.startswith("psycopg_pool")


def _close_for_real(conn: Any) -> None:
    """
    Closes the connection itself, also where `close()` would return it to a pool:
    mysql.connector's pooled connections hold it as `_cnx`, oracledb's are dropped from
    their pool.
    """
    inner = getattr(conn, "_cnx", None)
    # an oracledb pooled connection: `close()` releases it to the pool, `drop` closes it
    pool = getattr(conn, "_pool", None)
    if (
        inner is None
        and pool is not None
        and type(conn).__module__.startswith("oracledb")
        and callable(drop := getattr(pool, "drop", None))
    ):
        with contextlib.suppress(Exception):
            drop(conn)
            return
    with contextlib.suppress(Exception):
        (inner if inner is not None else conn).close()


# the oldest ConnectorX the tests pass with (also in Polars' `database` extra)
_MIN_CONNECTORX = "0.4.6"

# (URI, protocol) pairs ConnectorX has connected to in this process
_CONNECTORX_CHECKED: set[tuple[str, str | None]] = set()
_CONNECTORX_CHECKED_LOCK = threading.Lock()


class _ConnectorXSession:
    """ConnectorX has no sessions: every query opens its own connections."""

    can_cancel = False

    def __init__(self, uri: str, dialect: str, protocol: str | None = None) -> None:
        from polars._dependencies import import_optional

        # ConnectorX 0.3 read decimals wider than a float's digits rounded, silently
        import_optional(
            "connectorx",
            err_prefix="scan_database needs",
            min_version=_MIN_CONNECTORX,
        )
        self.uri = uri
        self.dialect = dialect
        self.protocol = protocol
        self._check_connection()

    def _check_connection(self) -> None:
        """
        Connects once per URI: a failed login surfaces here, after ConnectorX's ~30s
        connect timeout, instead of after the planner's catalog lookups (which carry on
        past a failure) each waited for it.
        """
        if self.dialect == "sqlite":
            # ConnectorX opens SQLite files read-only, never creating one: a missing
            # file fails each connect after the same wait
            path = unquote(self.uri.removeprefix("sqlite://"))
            if path != ":memory:" and not os.path.isfile(path):
                msg = f"no SQLite database file at {path!r}"
                raise FileNotFoundError(msg)
            return
        key = (self.uri, self.protocol)
        with _CONNECTORX_CHECKED_LOCK:
            if key in _CONNECTORX_CHECKED:
                return
        from polars.io.database._utils import _read_sql_connectorx

        sql = "SELECT 1 FROM DUAL" if self.dialect == "oracle" else "SELECT 1"
        _read_sql_connectorx(sql, connection_uri=self.uri, protocol=self.protocol)
        with _CONNECTORX_CHECKED_LOCK:
            _CONNECTORX_CHECKED.add(key)

    def execute(
        self, sql: str, batch_size: int | None, bind: bool = True
    ) -> Iterator[DataFrame]:
        # ConnectorX returns whole results; there is no batch size to apply
        from polars.io.database._utils import _read_sql_connectorx

        try:
            # ConnectorX connects per query: its errors can echo the URI
            with _scrubbed(_secrets(self.uri, None, {})):
                df = _read_sql_connectorx(
                    sql, connection_uri=self.uri, protocol=self.protocol
                )
        except RuntimeError as e:
            # ConnectorX can't type an empty SQLite result (or a leading NULL), and
            # types a column by its declared type: a number in a `TIMESTAMP` column
            # is "Invalid column type Integer". sqlite3 reads the same file, by the
            # values' own types
            if self.dialect != "sqlite" or not any(
                m in str(e) for m in ("Cannot infer type", "Invalid column type")
            ):
                raise
            # `sqlite://<path>` (see `_connectorx_uri`) as SQLAlchemy's form
            sqlite_uri = "sqlite:///" + self.uri.removeprefix("sqlite://")
            conn = _connect_uri("sqlite3", sqlite_uri, {})
            try:
                yield from _Session(conn, owned=True, execute_options={}).execute(
                    sql, batch_size
                )
            finally:
                conn.close()
            return
        yield df

    def execute_update(self, sql: str) -> None:
        msg = "connectorx can't run session statements"
        raise NotImplementedError(msg)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


class _ArrowOdbcSession:
    """`arrow-odbc` reads batches straight from an ODBC connection string."""

    can_cancel = False

    def __init__(self, connection_string: str, options: dict[str, Any]) -> None:
        self.connection_string = connection_string
        self.options = options

    def execute(
        self, sql: str, batch_size: int | None, bind: bool = True
    ) -> Iterator[DataFrame]:
        import arrow_odbc

        import polars as pl

        reader = arrow_odbc.read_arrow_batches_from_odbc(
            query=sql,
            connection_string=self.connection_string,
            batch_size=batch_size or DEFAULT_BATCH_SIZE,
            **self.options,
        )
        for batch in reader:
            yield pl.DataFrame(batch)

    def execute_update(self, sql: str) -> None:
        msg = "arrow-odbc can't run session statements"
        raise NotImplementedError(msg)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


class _AdbcSession:
    """
    An ADBC connection opened through `adbc_driver_manager`.

    Rust finds the driver and builds the options; results go back as the driver's
    Arrow C stream, which Rust reads without the GIL.
    """

    def __init__(
        self,
        driver: str,
        entrypoint: str | None,
        options: list[tuple[str, str]],
        execute_options: list[tuple[str, str]],
    ) -> None:
        try:
            from adbc_driver_manager import AdbcConnection, AdbcDatabase
            from adbc_driver_manager._lib import ArrowArrayStreamHandle
        except ImportError:
            msg = (
                "ADBC drivers are loaded through adbc_driver_manager: "
                "pip install adbc-driver-manager"
            )
            raise ModuleNotFoundError(msg) from None
        if not hasattr(ArrowArrayStreamHandle, "__arrow_c_stream__"):
            msg = (
                "ADBC drivers need a newer adbc_driver_manager "
                f"(found {_version('adbc_driver_manager')}): "
                "pip install -U adbc-driver-manager"
            )
            raise ModuleNotFoundError(msg)

        # the driver manager's own arguments: as options they would load another driver
        reserved = {"driver", "entrypoint", "load_flags", "init_func"} & dict(
            options
        ).keys()
        if reserved:
            msg = (
                f"connect option(s) {sorted(reserved)} can't be used with ADBC: "
                "adbc_driver_manager reads them as its own arguments"
            )
            raise ValueError(msg)
        kwargs: dict[str, Any] = {"driver": driver, **dict(options)}
        if entrypoint:
            kwargs["entrypoint"] = entrypoint
        self._db = AdbcDatabase(**kwargs)
        try:
            self._conn = AdbcConnection(self._db)
        except BaseException:
            self._db.close()
            raise
        self._execute_options = dict(execute_options)
        self._lock = threading.Lock()
        # held while a statement or the connection is cancelled or closed: the driver
        # manager calls the driver for both without a lock of its own, and a cancel must
        # never run on a handle being released
        self._handle_lock = threading.Lock()
        # statements whose result streams Rust still reads; the connection closes after
        # the last one
        self._open: list[Any] = []
        self._closed = False
        # `cancel` is sticky (see `_Session`): no statement starts after it
        self._cancelled = False
        # a statement is being executed: a cancel may find the driver not running it yet
        # (before the query is sent, or between its round trips), so it is repeated
        self._issuing = False

    def _statement(self, sql: str, hint: tuple[str, str] | None) -> Any:
        from adbc_driver_manager import AdbcStatement

        stmt = AdbcStatement(self._conn)
        try:
            if hint is not None:
                # a hint: drivers that reject it still work
                with contextlib.suppress(Exception):
                    stmt.set_options(**{hint[0]: hint[1]})
            if self._execute_options:
                stmt.set_options(**self._execute_options)
            stmt.set_sql_query(sql)
        except BaseException:
            stmt.close()
            raise
        return stmt

    def execute(self, sql: str, hint: tuple[str, str] | None) -> tuple[Any, Any]:
        """Run a query: `(Arrow C stream capsule, statement)`; see `release`."""
        stmt = self._statement(sql, hint)
        self._begin(stmt)
        try:
            handle, _ = stmt.execute_query()
            with self._lock:
                self._issuing = False
            # a cancel that came as the statement started may have found it not running;
            # the handle goes before the statement is closed
            if self._cancelled:
                del handle
                raise RuntimeError(_CANCELLED)
            return handle.__arrow_c_stream__(), stmt
        except BaseException:
            with self._lock:
                self._issuing = False
            self.release(stmt)
            raise

    def execute_update(self, sql: str) -> None:
        stmt = self._statement(sql, None)
        self._begin(stmt)
        try:
            stmt.execute_update()
        finally:
            self.release(stmt)

    def _begin(self, stmt: Any) -> None:
        """Registers a statement about to run, unless the session was cancelled."""
        with self._lock:
            if not self._cancelled:
                self._open.append(stmt)
                self._issuing = True
                return
        stmt.close()
        raise RuntimeError(_CANCELLED)

    def release(self, stmt: Any) -> None:
        """Close a statement once its result stream is released."""
        close_conn = False
        try:
            # The handle lock is held from the statement leaving `_open` until it is
            # closed: a `close()` that finds `_open` empty waits for it (the driver
            # manager refuses to close a connection with a statement open).
            with self._handle_lock:
                with self._lock:
                    with contextlib.suppress(ValueError):
                        self._open.remove(stmt)
                    self._issuing = False
                    close_conn = self._closed and not self._open
                stmt.close()
        finally:
            if close_conn:
                self._close_connection()

    def cancel(self) -> None:
        from adbc_driver_manager import AdbcStatusCode, NotSupportedError

        with self._lock:
            self._cancelled = True
        delay = 0.02
        while True:
            with self._lock:
                stmt = self._open[-1] if self._open else None
                issuing = self._issuing
            if stmt is None:
                return
            try:
                with self._handle_lock:
                    # released meanwhile: nothing to cancel
                    if stmt not in self._open:
                        return
                    try:
                        stmt.cancel()
                    except NotSupportedError:
                        try:
                            self._conn.cancel()
                        except NotSupportedError:
                            # an ADBC 1.0 driver: it can't cancel
                            return
            except Exception as e:
                # nothing running (yet): asked again while the statement is executing
                if getattr(e, "status_code", None) != AdbcStatusCode.INVALID_STATE:
                    raise
            if not issuing or _cancel_expired():
                return
            time.sleep(delay)
            delay = min(delay * 2, 0.25)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._open:
                return
        self._close_connection()

    def _close_connection(self) -> None:
        with self._handle_lock:
            try:
                self._conn.close()
            finally:
                self._db.close()


def has_adbc_driver_manager() -> bool:
    """Whether an `adbc_driver_manager` version `_AdbcSession` can use is installed."""
    try:
        # the package first, as `_AdbcSession` imports it: a cached `_lib` alone is
        # found without it
        import adbc_driver_manager  # noqa: F401
        from adbc_driver_manager._lib import ArrowArrayStreamHandle
    except ImportError:
        return False
    return hasattr(ArrowArrayStreamHandle, "__arrow_c_stream__")


def adbc_connect(
    driver: str,
    entrypoint: str | None,
    options: list[tuple[str, str]],
    execute_options: list[tuple[str, str]],
) -> _AdbcSession:
    """Open an ADBC connection through `adbc_driver_manager`."""
    return _AdbcSession(driver, entrypoint, options, execute_options)


_ASSERTION_POOL_LOCK = threading.Lock()

# the dialects with a session setup (`DialectName::session_setup`)
_SESSION_SETUP_DIALECTS = ("postgresql", "mysql", "oracle")


def _assertion_pool_busy(e: AssertionError) -> bool:
    return str(e).startswith("connection is already checked out")


def connect(
    kind: str,
    target_kind: str,
    target: Any,
    connect_options: list[tuple[str, str]],
    execute_options: list[tuple[str, str]],
    dialect: str,
    credential_provider: Callable[[], Any] | None,
    params: Any = None,
    python_options: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> Any:
    """Opens a session for a Python backend; the batch size is given per query."""
    # the values as given in Python: driver keyword arguments aren't all strings
    connect_opts: dict[str, Any]
    execute_opts: dict[str, Any]
    if python_options is not None:
        connect_opts, execute_opts = dict(python_options[0]), dict(python_options[1])
    else:
        connect_opts, execute_opts = dict(connect_options), dict(execute_options)
    # bound to the statements that hold the user's query (`_Session.execute`)
    session = {"execute_options": execute_opts, "params": params}

    if target_kind == "live":
        if (raw := _duckdb_engine_connection(target, session)) is not None:
            target = raw
        return _Session(target, owned=False, **session)
    if target_kind == "factory":
        try:
            checkout = target()
        except AssertionError as e:
            # an AssertionPool refuses a second checkout while another is out: ours are
            # given back at once (below), and wait for each other
            if not _assertion_pool_busy(e):
                raise
            with _ASSERTION_POOL_LOCK:
                for attempt in range(100):
                    try:
                        checkout = target()
                        break
                    except AssertionError as e:
                        if attempt == 99 or not _assertion_pool_busy(e):
                            raise
                        time.sleep(0.002)
        _refuse_bound_session(checkout)
        if (shared := shared_pool_connection(checkout)) is not None:
            if dialect in _SESSION_SETUP_DIALECTS:
                # the plan relies on the session setup, which the user's connection
                # must not get
                msg = (
                    "the connection function returns the connection that its pool "
                    "shares with the user's own (StaticPool, AssertionPool), whose "
                    "session the scan must not set up: pass the Engine itself, which "
                    "is read as a live connection"
                )
                raise TypeError(msg)
            # one connection that the user's own statements use too (an open
            # transaction): a live connection, never rolled back or closed
            return _Session(shared, owned=False, **session)
        conn, close, invalidate = _pool_checkout(checkout)
        return _Session(
            conn,
            owned=True,
            close=close,
            invalidate=invalidate,
            factory=True,
            **session,
        )

    creds = credentials(credential_provider) if credential_provider else None
    with _scrubbed(_secrets(target, creds, connect_opts)):
        return _open(kind, target_kind, target, creds, connect_opts, dialect, session)


def _open(
    kind: str,
    target_kind: str,
    target: str,
    creds: tuple[str | None, str | None] | None,
    connect_opts: dict[str, Any],
    dialect: str,
    session: dict[str, Any],
) -> Any:
    """A session for a URI or ODBC connection string."""
    head, _, tail = kind.partition(":")
    if target_kind == "odbc":
        if tail == "arrow_odbc":
            # (its `parameters` are text only: an integer would bind as a string)
            if session.get("params") or session.get("execute_options"):
                msg = (
                    "arrow_odbc takes no `params` or `execute_options`: write the "
                    "values into the query, or use another ODBC driver (pyodbc)"
                )
                raise ValueError(msg)
            return _ArrowOdbcSession(target, connect_opts)
        mod = importlib.import_module(tail)
        if tail == "turbodbc":
            conn = mod.connect(connection_string=target, **connect_opts)
        else:
            conn = mod.connect(target, **connect_opts)
        return _Session(conn, owned=True, **session)

    uri = _canonical_scheme(_with_credentials(target, creds), dialect, head)
    uri = _with_connect_timeout(uri, dialect, kind)
    if head == "connectorx":
        unknown = sorted(set(connect_opts) - {"protocol"}) + sorted(
            session["execute_options"]
        )
        if unknown:
            msg = (
                f"connectorx takes no options {unknown}: only "
                "connect_options={'protocol': ...}"
            )
            raise ValueError(msg)
        return _ConnectorXSession(
            _connectorx_uri(uri, dialect),
            dialect,
            protocol=connect_opts.get("protocol"),
        )
    if head == "sqlalchemy":
        _require_sqlalchemy()
        from sqlalchemy import create_engine

        if dialect == "sqlite":
            uri = _sqlalchemy_sqlite_uri(uri)
        try:
            engine = create_engine(uri, **connect_opts)
        except Exception as e:
            if type(e).__name__ != "NoSuchModuleError":
                raise
            scheme = uri.split(":", 1)[0]
            msg = (
                f"SQLAlchemy has no dialect for {scheme!r}; install the SQLAlchemy "
                "dialect package for this database, pin another driver with "
                "driver=..., or pass a connection factory"
            )
            raise ValueError(msg) from None
        conn = engine.connect()

        def close() -> None:
            conn.close()
            engine.dispose()

        return _Session(conn, owned=True, close=close, **session)
    if head == "python":
        if tail == "duckdb":
            path, kw = _duckdb_connect_args(uri)
            conn, close_duckdb = _duckdb_shared_cursor(
                importlib.import_module("duckdb"), path, {**kw, **connect_opts}
            )
            return _Session(conn, owned=True, close=close_duckdb, **session)
        conn = _connect_uri(tail, uri, connect_opts)
        return _Session(conn, owned=True, **session)

    msg = f"no Python backend for {kind!r} (dialect {dialect!r})"
    raise ValueError(msg)
