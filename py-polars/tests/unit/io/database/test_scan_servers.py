"""
`scan_database` against real database servers: the differential suite.

These run only when server URIs are given:

    POLARS_TEST_POSTGRES_URI, POLARS_TEST_POSTGRES14_URI, POLARS_TEST_MYSQL_URI,
    POLARS_TEST_MARIADB_URI, POLARS_TEST_ORACLE_URI, POLARS_TEST_CLICKHOUSE_URI,
    POLARS_TEST_MSSQL_URI, POLARS_TEST_TRINO_URI,
    POLARS_TEST_DATABRICKS_URI (a workspace's warehouse, see `_databricks.py`: it seeds
    its own schema, and each query takes seconds)

Every (server, driver) cell must pass: a failing cell is a bug, and an xfail would hide
a regression in it. Cells that can't run here are skipped with the reason
(`KNOWN_UNRUNNABLE`), at once rather than after driver timeouts.

By default the `live` cell of each server runs the whole predicate corpus, and the
other cells what they plan differently (see `_scan_corpus`).
`POLARS_TEST_DATABASE_FULL=1` runs it on every cell, as the nightly job does.

A server whose URI is set but which doesn't accept connections fails the run when the
tests are collected; `POLARS_TEST_DATABASE_SKIP_UNREACHABLE=1` skips its tests instead.

Run them in parallel (`pytest -n 4`); `test_scan_load.py` runs on its own, serially.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo

import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal
from tests.unit.io.database._scan_corpus import (
    CORPUS_MODES,
    FULL,
    N_EDGE_ROWS,
    SAMPLE,
    CorpusPlans,
    cell_predicates,
    check_corpus,
    check_filter_ids,
    corpus_for_mode,
    expected_ids,
    ids,
    predicate_strategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from polars._typing import PolarsDataType

pytest.importorskip("polars_database")
pytest.importorskip("sqlglot")

# Random predicate trees per server; the nightly job raises it.
N_EXAMPLES = int(
    os.environ.get("POLARS_TEST_DATABASE_EXAMPLES", "200" if FULL else "50")
)

# The cell that runs the whole corpus: a live connection doesn't connect per query, and
# plans the same SQL as the other cells (the SQL is the dialect's, not the driver's)
BASE_CELL = "live"


def _parts(uri: str) -> dict[str, Any]:
    p = urlsplit(uri)
    return {
        "host": p.hostname,
        "port": p.port,
        "user": p.username,
        "password": p.password,
        "database": p.path.lstrip("/"),
    }


def _pg_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import psycopg

        return psycopg.connect(uri)

    return connect


def _mysql_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import pymysql  # type: ignore[import-untyped]

        return pymysql.connect(**_parts(uri))

    return connect


def _oracle_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import oracledb

        p = _parts(uri)
        return oracledb.connect(
            user=p["user"],
            password=p["password"],
            dsn=f"{p['host']}:{p['port']}/FREEPDB1",
        )

    return connect


def _clickhouse_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        from clickhouse_connect.dbapi import connect

        p = _parts(uri)
        return connect(
            host=p["host"],
            port=p["port"],
            username=p["user"],
            password=p["password"],
            database=p["database"],
        )

    return connect


def _mssql_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import pymssql  # type: ignore[import-not-found]

        p = _parts(uri)
        return pymssql.connect(
            server=p["host"],
            port=p["port"],
            user=p["user"],
            password=p["password"],
            database=p["database"],
        )

    return connect


def _trino_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import trino  # type: ignore[import-not-found]

        p = _parts(uri)
        catalog, _, schema = p["database"].partition("/")
        return trino.dbapi.connect(
            host=p["host"],
            port=p["port"],
            user=p["user"],
            catalog=catalog,
            schema=schema or "default",
            http_scheme="http",
        )

    return connect


def _mssql_adbc_uri(uri: str) -> str:
    p = urlsplit(uri)
    return p._replace(path="", query=f"database={p.path.lstrip('/')}").geturl()


def _mssql_odbc_string(uri: str) -> str:
    p = _parts(uri)
    return (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={p['host']},{p['port']};Database={p['database']};"
        f"UID={p['user']};PWD={p['password']};TrustServerCertificate=yes"
    )


def _databricks_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        from tests.unit.io.database._databricks import connect

        return connect(uri)

    return connect


@functools.cache
def _reachable(uri: str) -> bool:
    """
    Whether the server accepts connections, checked once per run.

    A server that is down skips its tests at once, instead of each one waiting for
    connect retries and driver timeouts.
    """
    import socket

    p = urlsplit(uri)
    scheme = p.scheme.split("+", 1)[0]
    port = p.port or _DEFAULT_PORTS.get(scheme)
    if not p.hostname or port is None:
        # a socket path or an unknown scheme: let the tests find out
        return True
    try:
        with socket.create_connection((p.hostname, port), timeout=2):
            return True
    except OSError:
        return False


_DEFAULT_PORTS = {
    "postgresql": 5432,
    "postgres": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "oracle": 1521,
    "clickhouse": 8123,
    "mssql": 1433,
    "trino": 8080,
    "databricks": 443,
}


@dataclass
class Server:
    """A server of the backend matrix and its driver cells."""

    env: str
    # the edge table as a query, in the dialect's quoting
    query: str
    # driver pin -> blocking?
    drivers: dict[str, bool]
    factory: Callable[[str], Callable[[], Any]] | None = None
    factory_blocking: bool = False
    live_blocking: bool = False
    # a driver pin that needs a different URI form (SQLAlchemy dialect+driver)
    uri_for: dict[str, Callable[[str], str]] = field(default_factory=dict)
    # extra `scan_database` arguments of a driver pin
    kwargs_for: dict[str, dict[str, Any]] = field(default_factory=dict)
    # whether an explicit `partition_on` is accepted: a dialect without a catalog lookup
    # (Trino) can't prove a source repeatable, and refuses it
    partitions: bool = True

    @property
    def uri(self) -> str | None:
        """The server's URI, or None when it isn't given or isn't reachable."""
        uri = os.environ.get(self.env)
        if uri is None or _reachable(uri):
            return uri
        if os.environ.get("POLARS_TEST_DATABASE_SKIP_UNREACHABLE"):
            return None
        # a server that didn't start must not turn its tests into skips
        msg = (
            f"{self.env} is set, but the server is not reachable "
            "(POLARS_TEST_DATABASE_SKIP_UNREACHABLE=1 skips its tests)"
        )
        raise RuntimeError(msg)


SERVERS = {
    "postgres17": Server(
        "POLARS_TEST_POSTGRES_URI",
        "SELECT * FROM edge",
        {
            "adbc": True,
            "connectorx": True,
            "psycopg": True,
            "psycopg2": False,
            "pg8000": False,
            "sqlalchemy": True,
        },
        factory=_pg_factory,
        factory_blocking=True,
        live_blocking=True,
    ),
    "postgres14": Server(
        "POLARS_TEST_POSTGRES14_URI",
        "SELECT * FROM edge",
        {"adbc": True, "connectorx": False, "psycopg": False, "sqlalchemy": False},
        factory=_pg_factory,
    ),
    "mysql": Server(
        "POLARS_TEST_MYSQL_URI",
        "SELECT * FROM edge",
        {
            "adbc": True,
            "connectorx": True,
            "pymysql": True,
            "MySQLdb": False,
            "mysql.connector": False,
            "sqlalchemy": True,
        },
        factory=_mysql_factory,
        factory_blocking=True,
        uri_for={"sqlalchemy": lambda u: u.replace("mysql://", "mysql+pymysql://", 1)},
    ),
    "mariadb": Server(
        "POLARS_TEST_MARIADB_URI",
        "SELECT * FROM edge",
        {"adbc": False, "connectorx": False, "pymysql": False, "mariadb": False},
        factory=_mysql_factory,
    ),
    "oracle": Server(
        "POLARS_TEST_ORACLE_URI",
        'SELECT * FROM "edge"',
        {"oracledb": False, "connectorx": False, "sqlalchemy": False},
        factory=_oracle_factory,
        uri_for={
            "sqlalchemy": lambda u: u.replace("oracle://", "oracle+oracledb://", 1)
        },
    ),
    "clickhouse": Server(
        "POLARS_TEST_CLICKHOUSE_URI",
        "SELECT * FROM edge",
        {"adbc": False, "clickhouse_connect": False, "connectorx": False},
        factory=_clickhouse_factory,
    ),
    # `connectorx` first: the reference the other cells are compared with, and the
    # one driver whose catalog lookups work
    # (see `test_mssql_catalog_lookup_names_its_columns`)
    "mssql": Server(
        "POLARS_TEST_MSSQL_URI",
        "SELECT * FROM edge",
        {
            "connectorx": False,
            "pymssql": False,
            "adbc": False,
            "pyodbc": False,
            "arrow_odbc": False,
            "turbodbc": False,
            "sqlalchemy": False,
        },
        factory=_mssql_factory,
        uri_for={
            # the ADBC driver reads the URI's path as an instance name: `?database=`
            # (see `test_mssql_adbc_reads_the_database_of_the_uri_path`)
            "adbc": _mssql_adbc_uri,
            "pyodbc": _mssql_odbc_string,
            "arrow_odbc": _mssql_odbc_string,
            "turbodbc": _mssql_odbc_string,
            "sqlalchemy": lambda u: u.replace("mssql://", "mssql+pymssql://", 1),
        },
        # arrow-odbc needs an upper bound for `nvarchar(max)` and `varbinary(max)`
        kwargs_for={
            "arrow_odbc": {
                "connect_options": {"max_text_size": 65536, "max_binary_size": 65536}
            }
        },
    ),
    # the `memory` catalog (see `seed.py`): URI `trino://user@host:port/catalog/schema`
    "trino": Server(
        "POLARS_TEST_TRINO_URI",
        "SELECT * FROM edge",
        {"trino": False, "adbc": False, "sqlalchemy": False, "connectorx": False},
        factory=_trino_factory,
        partitions=False,
        uri_for={
            # the ADBC driver speaks HTTPS unless told not to
            "adbc": lambda u: u + ("&" if "?" in u else "?") + "SSL=false",
        },
    ),
    "databricks": Server(
        "POLARS_TEST_DATABRICKS_URI",
        "SELECT * FROM edge",
        {"adbc": False, "databricks.sql": False},
        factory=_databricks_factory,
    ),
}

DRIVER_MODULES = {
    "adbc": None,
    "connectorx": "connectorx",
    "sqlalchemy": "sqlalchemy",
    "clickhouse_connect": "clickhouse_connect",
    "mysql.connector": "mysql.connector",
}


# Oracle Free's listener refuses bursts of connections (ORA-12516) beyond what connect
# retries absorb: its tests run on one xdist worker, one at a time.
ORACLE_GROUP = pytest.mark.xdist_group("oracle")


def _server_marks(name: str) -> list[Any]:
    return [ORACLE_GROUP] if name == "oracle" else []


def _no_odbc_driver() -> str | None:
    try:
        with open("/etc/odbcinst.ini") as f:  # noqa: PTH123
            if "[ODBC Driver 18 for SQL Server]" in f.read():
                return None
    except OSError:
        pass
    return "the ODBC driver `msodbcsql18` is not installed"


def _no_adbc_driver(dialect: str) -> Callable[[], str | None]:
    def check() -> str | None:
        import polars_database

        if polars_database.find_adbc_driver(dialect) is None:
            return f"no ADBC driver for {dialect} (`dbc install {dialect}`)"
        return None

    return check


def _no_connectorx_trino() -> str | None:
    return (
        "ConnectorX's Trino source takes no schema from the URI (unqualified table "
        "names fail with MISSING_SCHEMA_NAME) and can't parse a bigint beyond 2**53"
    )


def _no_oracle_client() -> str | None:
    import ctypes.util

    if ctypes.util.find_library("clntsh") is None and not os.environ.get("ORACLE_HOME"):
        return "ConnectorX reads Oracle through Oracle Instant Client, not installed"
    return None


# Cells that may not run here: (server, driver) -> the reason, or None if it can.
KNOWN_UNRUNNABLE: dict[tuple[str, str], Callable[[], str | None]] = {
    ("oracle", "connectorx"): _no_oracle_client,
    ("mssql", "adbc"): _no_adbc_driver("mssql"),
    ("mssql", "pyodbc"): _no_odbc_driver,
    ("mssql", "arrow_odbc"): _no_odbc_driver,
    ("mssql", "turbodbc"): _no_odbc_driver,
    ("trino", "adbc"): _no_adbc_driver("trino"),
    ("trino", "connectorx"): _no_connectorx_trino,
}


def _cells(*, connections: bool = True, group: bool = False) -> list[Any]:
    """
    (server, driver) pairs; `factory` and `live` are connection forms.

    `group` keeps the tests of a server on one xdist worker, where they share the cached
    references, baselines and plans of its cells, the base cell's above all.
    """
    out = []
    for name, server in SERVERS.items():
        cells = list(server.drivers.items())
        if connections and server.factory is not None:
            cells += [
                ("factory", server.factory_blocking),
                ("live", server.live_blocking),
            ]
        for driver, _blocking in cells:
            marks = [
                pytest.mark.skipif(
                    server.uri is None, reason=f"{server.env} unset or unreachable"
                )
            ]
            marks += _server_marks(name)
            if group and name != "oracle":
                marks.append(pytest.mark.xdist_group(f"corpus-{name}"))
            unrunnable = KNOWN_UNRUNNABLE.get((name, driver))
            if unrunnable is not None and (reason := unrunnable()) is not None:
                marks.append(pytest.mark.skip(reason=reason))
            out.append(pytest.param(name, driver, id=f"{name}-{driver}", marks=marks))
    return out


# One live connection per server, shared by the module's tests (closed after them).
_LIVE: dict[str, Any] = {}


def _seeded(name: str, uri: str) -> None:
    """Creates the test schema of servers that don't have one from a seed script."""
    if name == "databricks":
        from tests.unit.io.database._databricks import ensure_seeded

        ensure_seeded(uri)


def _live(name: str) -> Any:
    if name not in _LIVE:
        server = SERVERS[name]
        assert server.factory is not None
        assert server.uri is not None
        _seeded(name, server.uri)
        _LIVE[name] = server.factory(server.uri)()
    return _LIVE[name]


@pytest.fixture(scope="module", autouse=True)
def _close_live_connections() -> Iterator[None]:
    yield
    for conn in _LIVE.values():
        with contextlib.suppress(Exception):
            conn.close()
    _LIVE.clear()


def _scan(name: str, driver: str, source: str = "edge", **kwargs: Any) -> pl.LazyFrame:
    server = SERVERS[name]
    uri = server.uri
    if uri is None:
        pytest.skip(f"no {name} URI")
    _seeded(name, uri)
    if name == "oracle" and source.isidentifier() and source.islower():
        # the shared tables are created quoted lower-case; a bare name means upper case
        source = f'"{source}"'
    if driver == "factory":
        assert server.factory is not None
        return pl.scan_database(source, server.factory(uri), **kwargs)
    if driver == "live":
        return pl.scan_database(source, _live(name), **kwargs)
    module = DRIVER_MODULES.get(driver, driver)
    if module is not None:
        pytest.importorskip(module)
    convert = server.uri_for.get(driver)
    if convert is not None:
        uri = convert(uri)
    kwargs = {**server.kwargs_for.get(driver, {}), **kwargs}
    return pl.scan_database(source, uri, driver=driver, **kwargs)


@functools.cache
def _reference(name: str, driver: str) -> pl.DataFrame:
    # the table as read without pushdown; Polars then does all the filtering
    return _scan(name, driver, pushdown=False, partition_on=None).collect()


@functools.cache
def _plans(name: str, driver: str, source: str) -> CorpusPlans:
    lf = _scan(name, driver, source, partition_on=None)
    return CorpusPlans.of(lf, _reference(name, driver))


def _default_driver(name: str) -> str:
    return next(iter(SERVERS[name].drivers))


@pytest.mark.parametrize(("name", "driver"), _cells(group=True))
def test_pushdown_off_reads_every_row(name: str, driver: str) -> None:
    # harness self-check: the reference really is the whole table
    assert ids(_reference(name, driver)) == list(range(N_EDGE_ROWS))


def _corpus_cells() -> list[Any]:
    """
    (server, driver, mode) of the corpus test.

    Outside the full run, partitioned modes run on the base cell only: the other cells
    plan the same SQL plus a range, and `test_partitions` reads partitions with each
    driver. `factory` cells don't run: they read through the driver module of a pinned
    cell, and the schema and round-trip tests cover the connection form.
    """
    return [
        pytest.param(*cell.values, mode, id=f"{cell.id}-{mode}", marks=cell.marks)
        for cell in _cells(group=True)
        for mode in CORPUS_MODES
        if (SERVERS[cell.values[0]].partitions or not mode.endswith("p4"))
        and (
            FULL
            or cell.values[1] == BASE_CELL
            or (cell.values[1] != "factory" and not mode.endswith("p4"))
        )
    ]


@pytest.mark.parametrize(("name", "driver", "mode"), _corpus_cells())
def test_differential_corpus(name: str, driver: str, mode: str) -> None:
    """Pushed predicates may only ever widen what the database returns."""
    server = SERVERS[name]
    source = server.query if mode.startswith("query") else "edge"
    kwargs: dict[str, Any] = (
        {"partition_on": "id", "partition_num": 4}
        if mode.endswith("p4")
        else {"partition_on": None}
    )
    reference = _reference(name, driver)
    lf = _scan(name, driver, source, **kwargs)
    predicates = corpus_for_mode(mode)
    # the base cell's table runs the whole corpus; its query is compared with it
    if not FULL and (driver != BASE_CELL or source != "edge"):
        base = (BASE_CELL, source) if driver != BASE_CELL else (BASE_CELL, "edge")
        predicates = cell_predicates(
            mode,
            _plans(name, driver, source),
            _plans(name, *base),
            sample=SAMPLE[mode],
            offset=[*server.drivers, "factory", BASE_CELL].index(driver),
        )
    failures, stats = check_corpus(
        lambda: lf, reference, predicates, check_exact=mode == "table"
    )
    if mode == "table":
        print(f"\n{name}-{driver}: {stats}")
    assert not failures, f"of {len(predicates)}:\n" + "\n".join(failures[:40])


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=[
                pytest.mark.skipif(
                    s.uri is None, reason=f"{s.env} unset or unreachable"
                ),
                *_server_marks(n),
            ],
        )
        for n, s in SERVERS.items()
    ],
)
def test_property(name: str) -> None:
    """Random predicate trees; a failing example belongs in the fixed corpus."""
    from hypothesis import HealthCheck, given, settings

    driver = _default_driver(name) if FULL else BASE_CELL
    reference = _reference(name, driver)
    lf = _scan(name, driver, partition_on=None)

    @settings(
        max_examples=N_EXAMPLES,
        deadline=None,
        suppress_health_check=list(HealthCheck),
        database=None,
    )
    @given(predicate_strategy(boolean_column=reference["b"].dtype == pl.Boolean))
    def check(p: pl.Expr) -> None:
        expected = expected_ids(reference, p)
        if isinstance(expected, Exception):
            with pytest.raises(Exception):  # noqa: B017
                lf.filter(p).select("id").collect()
            return
        assert ids(lf.filter(p).select("id").collect()) == expected, str(p)

    check()


@pytest.mark.parametrize("table", ["edge", "orders"])
@pytest.mark.parametrize(("name", "driver"), _cells(group=True))
def test_same_schema_for_every_driver(name: str, driver: str, table: str) -> None:
    # the dialect fixes the dtypes, not the driver
    assert _scan(name, driver, table).collect_schema() == _baseline_schema(name, table)


@functools.cache
def _baseline_schema(name: str, table: str) -> pl.Schema:
    return _scan(name, _default_driver(name), table).collect_schema()


# The rows `test_partitions` reads: all of `orders` (100k) in the full run, else a
# pushed range of them, still split over every partition
ORDERS = pl.lit(value=True) if FULL else pl.col("id") < 5_000


@pytest.mark.parametrize(("name", "driver"), _cells(connections=False))
def test_partitions(name: str, driver: str) -> None:
    if not SERVERS[name].partitions:
        pytest.skip(f"{name} has no catalog lookup: see `test_unproven_dialect_*`")
    lf = _scan(
        name, driver, "orders", partition_on="id", partition_num=8, max_connections=4
    ).filter(ORDERS)
    assert "partitions: 8" in lf.explain()
    reference = _orders(name)
    assert lf.select(pl.len()).collect().item() == reference.height
    assert_frame_equal(lf.collect().sort("id"), reference, check_dtypes=False)


@functools.cache
def _orders(name: str) -> pl.DataFrame:
    lf = _scan(name, _default_driver(name), "orders", partition_on=None)
    return lf.filter(ORDERS).collect().sort("id")


POSTGRES = SERVERS["postgres17"].uri
needs_postgres = pytest.mark.skipif(POSTGRES is None, reason="no Postgres URI")


def active_polars_queries(uri: str, marker: str = "") -> int:
    """
    Active Polars statements, only those whose SQL holds `marker` if given.

    Tests run in parallel: other tests' statements may be running too.
    """
    like = f" AND query LIKE '%{marker}%' AND pid != pg_backend_pid()" if marker else ""
    return int(
        pl.read_database_uri(
            "SELECT COUNT(*) AS n FROM pg_stat_activity "
            "WHERE application_name = 'polars' AND backend_type = 'client backend' "
            f"AND state = 'active'{like}",
            uri,
        ).item()
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_cancel_on_early_stop(driver: str) -> None:
    assert POSTGRES is not None
    marker = f"cancel_on_early_stop_{driver}"
    slow = (
        f"SELECT i AS id, pg_sleep(0.002) IS NULL AS {marker} "
        "FROM generate_series(1, 5000) AS i"
    )
    start = time.monotonic()
    out = (
        _scan(
            "postgres17",
            driver,
            slow,
            batch_size=100,
            schema={"id": pl.Int32, marker: pl.Boolean},
        )
        .filter(pl.col("id") > 0)
        .head(5)
        .collect()
    )
    assert out.height == 5
    # the whole query takes 10s; stopping early must not wait for it (the bound leaves
    # room for the other workers' load on the same server)
    assert time.monotonic() - start < 9
    # the cancel reaches the server asynchronously: wait a little, not 10s
    deadline = time.monotonic() + 3
    while active_polars_queries(POSTGRES, marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert active_polars_queries(POSTGRES, marker) == 0


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_credential_provider(driver: str) -> None:
    assert POSTGRES is not None
    password = urlsplit(POSTGRES).password
    without_password = POSTGRES.replace(f":{password}@", "@", 1)
    lf = pl.scan_database(
        "orders",
        without_password,
        driver=driver,
        credential_provider=lambda: {"password": password},
    )
    assert lf.select(pl.len()).collect().item() > 0
    assert str(password) not in lf.explain()


# Declared types and their dtypes, per dialect, whichever driver reads them
EDGE_TYPES: dict[str, dict[str, PolarsDataType]] = {
    "postgres17": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.Binary,
        # unconstrained `numeric` with 12 decimal places: Decimal(38, 9) would round it,
        # which is an error; read as text, it is the same for every driver
        "nu": pl.String,
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.Time,
    },
    "mysql": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.Binary,
        "nu": pl.Decimal(30, 12),
        "tstz": pl.Datetime("us", "UTC"),
        # a duration, not a time of day
        "t": pl.Duration("us"),
    },
    "oracle": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.Binary,
        "nu": pl.String,
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.String,
    },
    "clickhouse": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.String,
        "nu": pl.Decimal(38, 12),
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.String,
    },
}
EDGE_TYPES["mssql"] = {
    # SQL Server 2022 has no JSON type; a `uniqueidentifier` is read as text
    "j": pl.String,
    "u": pl.String,
    "bin": pl.Binary,
    "nu": pl.Decimal(30, 12),
    "tstz": pl.Datetime("us", "UTC"),
    "t": pl.Time,
}
EDGE_TYPES["trino"] = {
    "j": pl.String,
    "u": pl.String,
    "bin": pl.Binary,
    "nu": pl.Decimal(38, 12),
    "tstz": pl.Datetime("us", "UTC"),
    "t": pl.Time,
}
EDGE_TYPES["databricks"] = {
    "j": pl.String,
    "bin": pl.Binary,
    "nu": pl.Decimal(30, 12),
    "d38": pl.Decimal(38, 0),
    "tstz": pl.Datetime("us", "UTC"),
    "tntz": pl.Datetime("us"),
    "t": pl.Time,
    "arr": pl.List(pl.Int32),
    "st": pl.Struct({"a": pl.Int32, "b": pl.String}),
    "mp": pl.Map(pl.String, pl.Int32),
    # a VARIANT is its JSON text
    "v": pl.String,
}
EDGE_TYPES["postgres14"] = EDGE_TYPES["postgres17"]
EDGE_TYPES["mariadb"] = EDGE_TYPES["mysql"]


@pytest.mark.parametrize(("name", "driver"), _cells(group=True))
def test_edge_types_round_trip(name: str, driver: str) -> None:
    schema = _scan_edge_types(name, driver).collect_schema()
    expected = {"id": schema["id"], **EDGE_TYPES[name]}
    assert dict(schema) == expected
    got = _scan_edge_types(name, driver).collect().sort("id")
    assert_frame_equal(got, _edge_types(name))


def _scan_edge_types(name: str, driver: str) -> pl.LazyFrame:
    # unconstrained `numeric` / `NUMBER` with 12 decimal places: read as text
    kwargs: dict[str, Any] = (
        {"schema_overrides": {"nu": pl.String}}
        if name.startswith("postgres") or name == "oracle"
        else {}
    )
    return _scan(name, driver, "edge_types", **kwargs)


@functools.cache
def _edge_types(name: str) -> pl.DataFrame:
    return _scan_edge_types(name, _default_driver(name)).collect().sort("id")


# Pushdown regressions: each check compares a pushed-down
# filter with the same filter applied by Polars to the whole table.


def _execute(name: str, statements: list[str], *, root: bool = False) -> None:
    # (`root`: MySQL's `polars` can't create functions (binlog))
    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    assert server.factory is not None
    uri = server.uri.replace("polars:polars_Test1", "root:polars_Test1")
    conn = server.factory(uri if root else server.uri)()
    try:
        cur = conn.cursor()
        for sql in statements:
            cur.execute(sql)
            if name == "trino":
                # a statement runs to its end only when its result is read
                cur.fetchall()
        conn.commit()
    finally:
        conn.close()


@needs_postgres
def test_postgres_converted_and_padded_columns() -> None:
    from decimal import Decimal

    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS review_pg",
            "DROP COLLATION IF EXISTS review_ci",
            "CREATE COLLATION review_ci "
            "(provider = icu, locale = 'und-u-ks-level2', deterministic = false)",
            "CREATE TABLE review_pg (id int, u uuid, j jsonb, ch char(3), "
            "ci text COLLATE review_ci, nu numeric, d numeric(10,2))",
            "INSERT INTO review_pg VALUES "
            "(1, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{\"a\":1}', 'a', 'abc', "
            "0.100000001, 1.50), "
            "(2, 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{\"b\":2}', 'b  ', 'ABC', 0.1, 2.00), "
            "(3, NULL, NULL, NULL, 'x', 0.2, 3.00)",
        ],
    )
    for driver in ["adbc", "psycopg"]:
        lf = _scan("postgres17", driver, "review_pg")
        check_filter_ids(
            lf,
            [
                pl.col("u") == "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11",
                pl.col("u") != "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
                pl.col("u").str.starts_with("a0"),
                pl.col("j") == '{"a":1}',
                pl.col("ch") == "a",
                pl.col("ch") != "a",
                pl.col("ci") == "abc",
                pl.col("ci") != "abc",
                ~pl.col("ci").is_in(["abc"]),
                pl.col("nu") == Decimal("0.1"),
                pl.col("nu") > Decimal("0.1"),
                pl.col("d") < pl.lit(Decimal("1.501")),
                pl.col("d") > pl.lit(Decimal("1.499")),
            ],
        )


@needs_postgres
def test_postgres_scans_differing_in_options_are_not_merged() -> None:
    _execute(
        "postgres17",
        [
            "CREATE SCHEMA IF NOT EXISTS review_s1",
            "CREATE SCHEMA IF NOT EXISTS review_s2",
            "DROP TABLE IF EXISTS review_s1.t",
            "DROP TABLE IF EXISTS review_s2.t",
            "CREATE TABLE review_s1.t AS SELECT 's1' AS s",
            "CREATE TABLE review_s2.t AS SELECT 's2' AS s",
        ],
    )

    def scan(schema: str) -> pl.LazyFrame:
        return _scan(
            "postgres17",
            "psycopg",
            "SELECT s FROM t",
            connect_options={"options": f"-c search_path={schema}"},
        )

    both = pl.concat([scan("review_s1"), scan("review_s2")])
    for engine in ["in-memory", "streaming"]:
        assert both.collect(engine=engine)["s"].to_list() == ["s1", "s2"]  # type: ignore[call-overload]


@needs_postgres
def test_postgres_bare_table_names_fold_to_lower_case() -> None:
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS e_fold",
            'DROP TABLE IF EXISTS "E_Fold_Quoted"',
            "CREATE TABLE e_fold AS SELECT 1 AS id",
            'CREATE TABLE "E_Fold_Quoted" AS SELECT 2 AS id',
        ],
    )
    for driver in ["adbc", "psycopg"]:
        assert _scan("postgres17", driver, "E_Fold").collect()["id"].to_list() == [1]
        assert _scan("postgres17", driver, "Public.E_FOLD").collect().height == 1
        quoted = _scan("postgres17", driver, '"E_Fold_Quoted"')
        assert quoted.collect()["id"].to_list() == [2]


@needs_postgres
def test_postgres_bare_names_created_quoted_are_found_as_written() -> None:
    _execute(
        "postgres17",
        [
            'DROP TABLE IF EXISTS "E_Written"',
            'CREATE TABLE "E_Written" AS SELECT 1 AS id',
        ],
    )
    for driver in ["adbc", "psycopg"]:
        assert _scan("postgres17", driver, "E_Written").collect()["id"].to_list() == [1]
        # a user schema means no lookup: the name as written
        lf = _scan("postgres17", driver, "E_Written", schema={"id": pl.Int32})
        assert lf.collect()["id"].to_list() == [1]


@needs_postgres
def test_postgres_credential_provider_with_an_at_in_a_parameter() -> None:
    assert POSTGRES is not None
    uri = POSTGRES + ("&" if "?" in POSTGRES else "?") + "application_name=a@b"
    for driver in ["adbc", "psycopg"]:
        lf = pl.scan_database(
            "SELECT 1 AS one",
            uri,
            driver=driver,
            credential_provider=lambda: {"user": urlsplit(POSTGRES).username},
        )
        assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_bare_table_names_fold_to_upper_case() -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE e_fold'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE e_fold ("id" NUMBER(9))',
            "INSERT INTO e_fold VALUES (1)",
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE \"e_written\"'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE "e_written" ("id" NUMBER(9))',
            'INSERT INTO "e_written" VALUES (1)',
        ],
    )
    # a bare name created quoted is found as written
    for source in ["e_fold", "E_Fold", '"E_FOLD"', "e_written"]:
        lf = pl.scan_database(source, SERVERS["oracle"].uri)
        assert lf.collect()["id"].to_list() == [1], source


@needs_postgres
def test_postgres_date_partitions_past_year_9999() -> None:
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS e_far_dates",
            "CREATE TABLE e_far_dates (id int, d date)",
            "INSERT INTO e_far_dates VALUES (1, '2000-01-01'), (2, '5000-06-01'), "
            "(3, '20000-01-01'), (4, '900000-01-01'), (5, NULL)",
        ],
    )
    lf = _scan("postgres17", "adbc", "e_far_dates", partition_on="d", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3, 4, 5]


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_far_date_literals() -> None:
    lf = _scan("mysql", "adbc", "edge")
    far = pl.lit(2_933_897).cast(pl.Date)
    old = pl.lit(date(1, 6, 1))
    check_filter_ids(lf, [pl.col("d") < far, pl.col("d") != far, pl.col("d") > old])


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_char_and_float_columns() -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_ora'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_ora ("id" NUMBER(9), ch CHAR(3), fl FLOAT)',
            "INSERT INTO review_ora VALUES (1, 'a', 0.1000000000000000001)",
            "INSERT INTO review_ora VALUES (2, 'b', 0.1)",
            "INSERT INTO review_ora VALUES (3, NULL, NULL)",
        ],
    )
    lf = _scan("oracle", "oracledb", "REVIEW_ORA")
    check_filter_ids(
        lf,
        [
            pl.col("CH") == "a",
            pl.col("CH") != "a",
            pl.col("FL") == 0.1,
            pl.col("FL") > 0.1,
            pl.col("FL") != 0.1,
        ],
    )


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
@pytest.mark.parametrize("driver", ["adbc", "clickhouse_connect"])
def test_clickhouse_datetime_and_enum(driver: str) -> None:
    from datetime import datetime

    table = f"review_ch_{driver}"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, dt DateTime('UTC'), "
            "e Enum8('a' = 1, 'b' = 2)) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES "
            "(1, '2020-01-02 03:04:05', 'a'), (2, '2021-01-01 00:00:00', 'b')",
        ],
    )
    lf = _scan("clickhouse", driver, table)
    assert lf.collect_schema()["dt"] == pl.Datetime("ms", "UTC")
    assert lf.collect_schema()["e"] == pl.String
    df = lf.collect().sort("id")
    assert df["dt"][0] == datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert df["e"].to_list() == ["a", "b"]
    check_filter_ids(
        lf,
        [
            pl.col("dt") > datetime(2020, 6, 1, tzinfo=UTC),
            pl.col("e") == "b",
        ],
    )


@needs_postgres
def test_postgres_live_connection_left_outside_a_transaction() -> None:
    psycopg = pytest.importorskip("psycopg")

    assert POSTGRES is not None
    with psycopg.connect(POSTGRES) as conn:
        assert pl.scan_database("orders", conn).select(pl.len()).collect().item() > 0
        assert conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
        # a transaction the user opened is theirs: it stays open
        conn.execute("SELECT 1")
        pl.scan_database("orders", conn).head(1).collect()
        status = conn.info.transaction_status
        assert status == psycopg.pq.TransactionStatus.INTRANS  # type: ignore[comparison-overlap]


# `params=` with each Postgres and MySQL driver: placeholders, and a pushed filter
# around the query that would hold a `%`
PARAM_CELLS = [
    ("postgres17", d) for d in ["psycopg", "psycopg2", "pg8000", "sqlalchemy"]
] + [("mysql", d) for d in ["pymysql", "mysql.connector", "sqlalchemy"]]
# clickhouse_connect %-formats every query: the query's own placeholders must survive
PARAM_CELLS += [("clickhouse", "clickhouse_connect")]
# the ADBC driver doesn't bind here (an error says so, see `test_databricks_params`)
PARAM_CELLS += [("databricks", "databricks.sql")]
PARAM_CELLS += [("mssql", d) for d in ["pymssql", "pyodbc", "sqlalchemy"]]
PARAM_CELLS += [("trino", d) for d in ["trino", "sqlalchemy"]]


@pytest.mark.parametrize(("name", "driver"), PARAM_CELLS)
def test_params(name: str, driver: str) -> None:
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    placeholder = {
        "sqlalchemy": ":n",
        "databricks.sql": "?",
        "pyodbc": "?",
        "trino": "?",
    }.get(driver, "%s")
    params: Any = {"n": 30} if driver == "sqlalchemy" else [30]
    lf = _scan(
        name,
        driver,
        f"SELECT id, s FROM edge WHERE id < {placeholder}",
        params=params,
        partition_on=None,
    )
    out = lf.filter(pl.col("id") >= 10, pl.col("s").str.starts_with("a")).collect()
    expected = _reference(name, _default_driver(name)).filter(
        pl.col("id").is_between(10, 29), pl.col("s").str.starts_with("a")
    )
    assert sorted(out["id"].to_list()) == sorted(expected["id"].to_list())


@pytest.mark.parametrize("name", ["mysql", "postgres17", "mssql", "trino"])
def test_scans_sharing_a_live_connection(name: str) -> None:
    # each scan has its own session over the one connection, and they read at once
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    conn = _live(name)
    lo = pl.scan_database("edge", conn).filter(pl.col("id") < 40)
    hi = pl.scan_database("edge", conn).filter(pl.col("id") >= 10)
    out = lo.select("id", "s").join(hi.select("id"), on="id").collect()
    expected = _reference(name, _default_driver(name)).filter(
        pl.col("id").is_between(10, 39)
    )
    assert sorted(out["id"].to_list()) == sorted(expected["id"].to_list())


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_live_connection_without_backslash_escapes() -> None:
    # a live session is the user's: its sql_mode stays, so literals that need
    # backslash escapes aren't pushed (a connection of its own: it changes sql_mode)
    assert SERVERS["mysql"].uri is not None
    conn = _mysql_factory(SERVERS["mysql"].uri)()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SET SESSION sql_mode = "
                "CONCAT(@@SESSION.sql_mode, ',NO_BACKSLASH_ESCAPES')"
            )
        expected = _reference("mysql", _default_driver("mysql"))
        for predicate in [
            pl.col("s") == "\\",
            pl.col("s").is_in(["\\", "abc"]),
            pl.col("s").str.contains("\\", literal=True),
            pl.col("s").str.starts_with("a"),
        ]:
            out = pl.scan_database("edge", conn).filter(predicate).collect()
            want = expected.filter(predicate)
            assert sorted(out["id"].to_list()) == sorted(want["id"].to_list())
    finally:
        conn.close()


# Types outside the ones every driver reads natively are read as text (or a fixed
# dtype), so all drivers agree; none of them may crash or return other values.
UNUSUAL_TYPES: dict[str, tuple[list[str], str, list[str]]] = {
    "postgres17": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "DROP TYPE IF EXISTS review_mood CASCADE",
            "CREATE EXTENSION IF NOT EXISTS citext",
            "CREATE TYPE review_mood AS ENUM ('sad', 'ok')",
            "CREATE TABLE review_unusual (id int, ci citext, ip inet, mac macaddr, "
            "iv interval, m money, arr int[], e review_mood, b bit(4), r int4range, "
            "oi oid, w numeric(50, 10), n numeric)",
            "INSERT INTO review_unusual VALUES (1, 'AbC', '10.0.0.1/24', "
            "'08:00:2b:01:02:03', '1 day 2 hours', 12.34, '{1,2}', 'ok', B'1010', "
            "'[1,5)', 42, 1234567890123456789012345678901234567890.0123456789, "
            "12345678901234567890.123)",
        ],
        "review_unusual",
        ["adbc", "psycopg", "psycopg2", "connectorx", "pg8000", "sqlalchemy"],
    ),
    "mysql": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "CREATE TABLE review_unusual (id int, y year, bt bit(4), st set('x','y'), "
            "g point, fl float, dc decimal(65, 30))",
            "INSERT INTO review_unusual VALUES (1, 2024, b'1010', 'x,y', "
            "ST_GeomFromText('POINT(1 2)'), 1.5, 1.123456789012345678901234567890)",
        ],
        "review_unusual",
        ["adbc", "pymysql", "connectorx", "mysql.connector", "sqlalchemy"],
    ),
    "clickhouse": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "CREATE TABLE review_unusual (id Int32, arr Array(Int32), "
            "mp Map(String, Int32), tp Tuple(Int32, String), ip IPv4, fs FixedString(3), "
            "i128 Int128, u256 UInt256, d256 Decimal(60, 10)) "
            "ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_unusual VALUES (1, [1, 2], {'a': 1}, (1, 'x'), "
            "'10.0.0.1', 'abc', 170141183460469231731687303715884105727, 1, 1.5)",
        ],
        "review_unusual",
        ["adbc", "clickhouse_connect", "connectorx"],
    ),
    # (the types the drivers don't agree on are in `test_mssql_exotic_types`)
    "mssql": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "CREATE TABLE review_unusual (id int, tx text, ntx ntext, img image, "
            "ch char(5), nch nchar(5), ti tinyint, bt bit, w decimal(38, 10))",
            "INSERT INTO review_unusual VALUES (1, 'text', N'ntext', 0x0102, "
            "'ab', N'cd', 255, 1, 1234567890123456789012345678.1234567891)",
        ],
        "review_unusual",
        ["connectorx", "pymssql", "adbc", "pyodbc", "sqlalchemy"],
    ),
    # (see `test_trino_exotic_types`)
    "trino": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "CREATE TABLE review_unusual (id integer, ia array(integer), "
            "ip ipaddress, ch char(5), ti tinyint, ts3 timestamp(3), "
            "w decimal(38, 10), hl varbinary)",
            "INSERT INTO review_unusual VALUES (1, ARRAY[1, 2], IPADDRESS '10.0.0.1', "
            "CAST('ab' AS char(5)), TINYINT '7', TIMESTAMP '2020-01-02 03:04:05.123', "
            "DECIMAL '1234567890123456789012345678.1234567891', X'0102')",
        ],
        "review_unusual",
        ["trino", "adbc", "sqlalchemy"],
    ),
    "oracle": (
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_unusual'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_unusual ("id" NUMBER(9), ids INTERVAL DAY TO SECOND, '
            "iym INTERVAL YEAR TO MONTH, ri ROWID, n NUMBER)",
            "INSERT INTO review_unusual VALUES (1, INTERVAL '1 02:03:04' DAY TO SECOND, "
            "INTERVAL '1-2' YEAR TO MONTH, NULL, 12345678901234567890.123)",
        ],
        "REVIEW_UNUSUAL",
        ["oracledb", "sqlalchemy"],
    ),
}


def _installed(module: str) -> bool:
    # `find_spec("a.b")` imports `a`, and raises when `a` is missing
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=_server_marks(n)) for n in sorted(UNUSUAL_TYPES)]
)
def test_unusual_types_read_the_same_by_every_driver(name: str) -> None:
    statements, table, drivers = UNUSUAL_TYPES[name]
    _execute(name, statements)
    frames = {}
    for driver in drivers:
        module = DRIVER_MODULES.get(driver, driver)
        if module is not None and not _installed(module):
            continue
        frames[driver] = _scan(name, driver, table).collect()
    assert len(frames) >= 2, frames.keys()
    first, *rest = frames
    for driver in rest:
        assert_frame_equal(frames[driver], frames[first]), driver
    row = frames[first].row(0, named=True)
    assert frames[first].height == 1
    if name == "postgres17":
        assert row["iv"] == "1 day 02:00:00"
        assert row["w"] == "1234567890123456789012345678901234567890.0123456789"
        assert row["n"] == Decimal("12345678901234567890.123")
    if name == "oracle":
        assert row["N"] == Decimal("12345678901234567890.123")
    if name == "mysql":
        assert row["g"] == "POINT(1 2)"
        assert row["bt"] == "1010"


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "connectorx"])
def test_decimal_that_would_round_is_an_error(driver: str) -> None:
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS review_round_{driver}",
            f"CREATE TABLE review_round_{driver} (id int, n numeric, d numeric(20, 15))",
            f"INSERT INTO review_round_{driver} VALUES (1, 0.1000000001, 1.123456789012345)",
        ],
    )
    with pytest.raises(pl.exceptions.PolarsError, match="would round"):
        _scan("postgres17", driver, f"review_round_{driver}").select("n").collect()
    # a query has no declared types; ConnectorX writes every decimal with 10 places
    query = _scan("postgres17", driver, f"SELECT id, d FROM review_round_{driver}")
    if driver == "psycopg":
        assert query.collect()["d"].to_list() == [Decimal("1.123456789012345")]
    else:
        with pytest.raises(pl.exceptions.PolarsError, match="would round"):
            query.collect()


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_local_time_zone_and_numbers_on_a_live_session() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_tz'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_tz ("id" NUMBER(9), tsl TIMESTAMP WITH LOCAL TIME ZONE, '
            "n NUMBER)",
            "INSERT INTO review_tz VALUES (1, TIMESTAMP '2020-01-02 03:04:05 +00:00', 0.5)",
        ],
    )
    server = SERVERS["oracle"]
    assert server.factory is not None
    assert server.uri is not None
    conn = server.factory(server.uri)()
    try:
        cur = conn.cursor()
        cur.execute("ALTER SESSION SET TIME_ZONE = 'Asia/Tokyo'")
        cur.execute("ALTER SESSION SET NLS_NUMERIC_CHARACTERS = ',.'")
        row = pl.scan_database("REVIEW_TZ", conn).collect().row(0)
    finally:
        conn.close()
    assert row == (1, datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC), Decimal("0.5"))


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
def test_clickhouse_nan_in_a_float_partition_column_and_naive_timestamps() -> None:
    from datetime import datetime

    _execute(
        "clickhouse",
        [
            "DROP TABLE IF EXISTS review_nan",
            "CREATE TABLE review_nan (id Int32, f Float64, ts DateTime64(6)) "
            "ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_nan VALUES (1, 0, '2020-01-01 00:00:00'), "
            "(2, nan, '2020-01-02 00:00:00'), (3, 10, '2020-01-03 00:00:00')",
        ],
    )
    lf = _scan("clickhouse", "adbc", "review_nan", partition_on="f", partition_num=3)
    assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3]
    # a literal without a zone would be read in the server's time zone: not pushed
    naive = _scan("clickhouse", "adbc", "review_nan").filter(
        pl.col("ts") >= datetime(2020, 1, 2)
    )
    assert "filters 0/1" in naive.explain()
    assert sorted(naive.collect()["id"].to_list()) == [2, 3]


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_ordered_read_with_more_partitions_than_sessions(
    driver: str, tmp_path: Path
) -> None:
    # An ordered sink consumes partition 0 first: a later partition must never hold a
    # session it waits for. Run in a subprocess, so a deadlock fails instead of hanging.
    import subprocess
    import sys

    assert POSTGRES is not None
    # a real table: a query over generate_series() isn't proven repeatable, so it can't
    # be partitioned
    table = f"rvc_pg_ordered_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} AS SELECT i AS id, md5(i::text) AS s "
            "FROM generate_series(1, 1000000) AS i",
        ],
    )
    out = tmp_path / "out.parquet"
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_database({table!r}, {POSTGRES!r}, driver={driver!r}, "
        "partition_on='id', partition_num=16, max_connections=4, batch_size=10000)\n"
        f"for _ in range({3 if FULL else 1}):\n"
        f"    lf.sink_parquet({str(out)!r})\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("the ordered partitioned read hung")
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])
    assert pl.scan_parquet(out).select(pl.len()).collect().item() == 1_000_000


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_year_literals(name: str) -> None:
    # MySQL reads an integer compared with YEAR as a year: 70 is 1970, 1 is 2001
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_year",
            "CREATE TABLE review_year (id int, y year)",
            "INSERT INTO review_year VALUES (1, 1970), (2, 2001), (3, 2050), (4, 0), "
            "(5, NULL)",
        ],
    )
    predicates = [
        pl.col("y") == 70,
        pl.col("y") == 1,
        pl.col("y") > 50,
        pl.col("y") > 99,
        pl.col("y") < 1,
        pl.col("y").is_in([1, 70]),
        pl.col("y").is_in([5, 1970]),
        pl.col("y").is_between(1, 2100),
        pl.col("y") == 2001,
        pl.col("y") == 0,
        # every other integer compares as itself
        pl.col("y") > 100,
        pl.col("y") < 1000,
        pl.col("y") == 2156,
        pl.col("y") == -1,
        pl.col("y").is_between(100, 1950),
        pl.col("y").is_in([0, 1970]),
    ]
    check_filter_ids(_scan(name, "pymysql", "review_year"), predicates)
    # a query's columns read by a row driver have no type: any integer may be a YEAR,
    # so only literals 1-99 are kept from the database
    query = _scan(name, "pymysql", "SELECT id, y FROM review_year")
    check_filter_ids(query, predicates)
    assert "filters 1/1 (1 exact)" in query.filter(pl.col("id") == 500).explain()
    assert "filters 0/1" in query.filter(pl.col("id") == 50).explain()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "connectorx", "live"])
def test_postgres_literals_without_standard_conforming_strings(driver: str) -> None:
    # a table per driver: the parametrizations run on parallel workers
    table = f"review_scs_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, s text)",
            f"INSERT INTO {table} VALUES (1, 'a\\b'), (2, 'ab'), (3, '50%'), "
            "(4, 'a''b')",
        ],
    )
    off = "-c standard_conforming_strings=off"
    conn = None
    if driver == "live":
        # a connection of its own: it changes a session setting
        server = SERVERS["postgres17"]
        assert server.factory is not None
        assert server.uri is not None
        conn = server.factory(server.uri)()
        conn.execute("SET standard_conforming_strings = off")
        conn.commit()
        lf = pl.scan_database(table, conn)
    elif driver == "psycopg":
        lf = _scan("postgres17", driver, table, connect_options={"options": off})
    else:
        assert POSTGRES is not None
        if driver == "connectorx":
            pytest.importorskip(driver)
        uri = f"{POSTGRES}?options={quote(off)}"
        lf = pl.scan_database(table, uri, driver=driver)
    try:
        check_filter_ids(
            lf,
            [
                pl.col("s") == "a\\b",
                pl.col("s") != "a\\b",
                pl.col("s").str.contains("%", literal=True),
                pl.col("s").str.starts_with("a\\"),
                pl.col("s") == "a'b",
                # Postgres text can't hold NUL
                pl.col("s") == "a\0b",
                pl.col("s") != "a\0b",
            ],
        )
    finally:
        if conn is not None:
            conn.close()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_statement_timeout_is_not_retried(driver: str) -> None:
    timeout = "-c statement_timeout=300"
    if driver == "adbc":
        assert POSTGRES is not None
        lf = pl.scan_database(
            "SELECT pg_sleep(2) AS x", f"{POSTGRES}?options={quote(timeout)}"
        )
    else:
        lf = _scan(
            "postgres17",
            driver,
            "SELECT pg_sleep(2) AS x",
            connect_options={"options": timeout},
        )
    start = time.monotonic()
    with pytest.raises(pl.exceptions.PolarsError, match="statement timeout"):
        lf.collect()
    # retried, it would take 4 x 0.3 s plus 2.6 s of backoff
    assert time.monotonic() - start < 2


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
@pytest.mark.parametrize("connection", ["live", "factory"])
def test_oracle_linguistic_session(connection: str) -> None:
    # a table per connection: parametrizations may run at once on other workers
    table = f"review_nls_{connection}"
    _execute(
        "oracle",
        [
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            f'CREATE TABLE {table} ("id" NUMBER(9), s VARCHAR2(10))',
            f"INSERT INTO {table} VALUES (1, 'abc')",
            f"INSERT INTO {table} VALUES (2, 'ABC')",
            f"INSERT INTO {table} VALUES (3, 'x')",
        ],
    )
    assert SERVERS["oracle"].uri is not None
    factory = _oracle_factory(SERVERS["oracle"].uri)

    def linguistic() -> Any:
        conn = factory()
        conn.cursor().execute(
            "ALTER SESSION SET NLS_COMP = LINGUISTIC NLS_SORT = BINARY_CI"
        )
        return conn

    # a live session keeps its settings; a factory's sessions get the setup
    source = linguistic() if connection == "live" else linguistic
    lf = pl.scan_database(table.upper(), source)
    check_filter_ids(
        lf,
        [
            pl.col("S") == "abc",
            pl.col("S") != "abc",
            pl.col("S").is_in(["abc"]),
            pl.col("S").str.starts_with("ab"),
            pl.col("S") > "a",
        ],
    )


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_time_zone_offsets_and_bc_dates() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_tstz'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_tstz ("id" NUMBER(9), ts TIMESTAMP WITH TIME ZONE, '
            "d DATE)",
            "INSERT INTO review_tstz VALUES "
            "(1, TIMESTAMP '2020-01-02 12:00:00 +05:00', DATE '2020-01-02')",
            "INSERT INTO review_tstz VALUES "
            "(2, TIMESTAMP '2020-01-02 12:00:00 -08:00', "
            "TO_DATE('-0044-03-15', 'SYYYY-MM-DD'))",
        ],
    )
    lf = _scan("oracle", "oracledb", "REVIEW_TSTZ")
    # the offsets are applied, not dropped
    assert lf.select("id", "TS").sort("id").collect().rows() == [
        (1, datetime(2020, 1, 2, 7, tzinfo=UTC)),
        (2, datetime(2020, 1, 2, 20, tzinfo=UTC)),
    ]
    noon = datetime(2020, 1, 2, 12, tzinfo=UTC)
    assert lf.filter(pl.col("TS") < noon).select("id").collect().item() == 1
    # Python's datetime has no year -44
    with pytest.raises(pl.exceptions.PolarsError, match=r"outside years 1..9999"):
        lf.select("D").collect()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["adbc", "connectorx", "pymysql"])
def test_mysql_unsigned_decimals_and_zero_dates(driver: str) -> None:
    from datetime import datetime

    table = f"review_d_{driver}"
    _execute(
        "mysql",
        [
            "SET SESSION sql_mode = ''",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, a decimal(30,15) unsigned, "
            "b decimal(20,5) unsigned zerofill, d date, dt datetime, ts timestamp NULL)",
            f"INSERT INTO {table} VALUES "
            "(1, 0.123456789012345, 123456789012345.12345, '0000-00-00', "
            "'0000-00-00 00:00:00', '0000-00-00 00:00:00'), "
            "(2, 1, 1, '2020-00-15', '2020-01-02 03:04:05', '2020-01-02 03:04:05'), "
            "(3, 1, 1, '2019-01-01', NULL, NULL), (4, 1, 1, '2021-01-01', NULL, NULL)",
        ],
    )
    lf = _scan("mysql", driver, table)
    assert lf.collect_schema()["a"] == pl.Decimal(30, 15)
    assert lf.collect_schema()["b"] == pl.Decimal(20, 5)
    # every digit, not cut to 10 places or to 19 digits
    exact = Decimal("0.123456789012345")
    assert lf.filter(pl.col("a") == exact).select("id").collect().item() == 1
    rows = lf.sort("id").collect().rows()
    assert rows[0][1:3] == (exact, Decimal("123456789012345.12345"))
    # zero dates (and a zero day or month) are no date, also to pushed filters
    assert rows[0][3:] == (None, None, None)
    assert rows[1][3:5] == (None, datetime(2020, 1, 2, 3, 4, 5))
    # bounds and partitions take zero dates (and a zero day or month) as NULL, and so
    # does the order, also when the column isn't read
    cases3: list[dict[str, Any]] = [
        {"partition_on": None},
        {"partition_on": "d", "partition_num": 3},
    ]
    for kwargs in cases3:
        ordered = _scan("mysql", driver, table, order_by="d", **kwargs)
        assert ordered.collect()["d"].to_list() == [
            None,
            None,
            date(2019, 1, 1),
            date(2021, 1, 1),
        ], kwargs
        ids = ordered.select("id").collect()["id"].to_list()
        assert ids[2:] == [3, 4], kwargs
    check_filter_ids(
        lf,
        [
            pl.col("d").is_null(),
            pl.col("d").is_not_null(),
            pl.col("dt") < datetime(2021, 1, 1),
            pl.col("dt").is_null() | (pl.col("ts") > datetime(2000, 1, 1, tzinfo=UTC)),
        ],
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "pg8000", "sqlalchemy"])
def test_postgres_empty_table_is_typed(driver: str) -> None:
    from datetime import datetime

    table = f"review_empty_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, s text, v float8, ts timestamptz, "
            "n numeric(10, 2))",
        ],
    )
    expected = {
        "id": pl.Int32,
        "s": pl.String,
        "v": pl.Float64,
        "ts": pl.Datetime("us", "UTC"),
        "n": pl.Decimal(10, 2),
    }
    row = (1, "x", 1.5, datetime(2020, 1, 2, tzinfo=UTC), Decimal("1.25"))
    # a query's `numeric` has no declared precision for every driver
    for source, n in [(table, 5), (f"SELECT id, s, v, ts FROM {table}", 4)]:
        lf = _scan("postgres17", driver, source)
        assert dict(lf.collect_schema()) == dict(list(expected.items())[:n]), source
        # rows added after planning still fit the planned schema
        _execute(
            "postgres17",
            [f"INSERT INTO {table} VALUES (1, 'x', 1.5, '2020-01-02', 1.25)"],
        )
        assert lf.collect().rows() == [row[:n]]
        _execute("postgres17", [f"DELETE FROM {table}"])


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "connectorx"])
def test_postgres_query_columns_compare_by_their_type(driver: str) -> None:
    # a query has no catalog: its result types decide what is compared, and how
    table = f"review_pgq_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"DROP TYPE IF EXISTS {table}_mood",
            f"DROP COLLATION IF EXISTS {table}_ci",
            f"CREATE COLLATION {table}_ci "
            "(provider = icu, locale = 'und-u-ks-level2', deterministic = false)",
            f"CREATE TYPE {table}_mood AS ENUM ('a', 'b')",
            f"CREATE TABLE {table} (id int, code char(5), u uuid, ip inet, "
            f"m {table}_mood, ci text COLLATE {table}_ci, v text, n int)",
            f"INSERT INTO {table} VALUES "
            "(1, 'ab', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '10.0.0.1', 'a', "
            "'abc', 'x', 1), "
            "(2, 'cd', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '10.0.0.2/31', 'b', "
            "'ABC', 'y', 2)",
        ],
    )
    lf = _scan(
        "postgres17", driver, f"SELECT id, code, u, ip, m, ci, v, n FROM {table}"
    )
    predicates = [
        pl.col("code") == "ab",
        pl.col("m") == "nope",
        pl.col("ci") == "abc",
        pl.col("v") == "x",
        pl.col("v") < "y",
        pl.col("n") > 1,
    ]
    if lf.collect_schema()["u"] == pl.String:
        predicates += [
            pl.col("u") == "{a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11}",
            pl.col("u") == "zzz",
            pl.col("u") < "b",
            pl.col("ip") == "10.0.0.1/32",
        ]
    check_filter_ids(lf, predicates)
    # text and numbers are still compared in the database
    for p in [pl.col("v") == "x", pl.col("n") > 1]:
        assert "filters 1/1 (1 exact)" in lf.filter(p).explain(), str(p)


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_integer_is_a_38_digit_integer() -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_ora_int'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_ora_int ("id" NUMBER(9), i INTEGER, s NUMBER(*, 2))',
            "INSERT INTO review_ora_int VALUES "
            "(1, 12345678901234567890123456789012345, 1.25)",
            "INSERT INTO review_ora_int VALUES (2, 7, NULL)",
        ],
    )
    lf = _scan("oracle", "oracledb", "REVIEW_ORA_INT")
    assert lf.collect_schema()["I"] == pl.Decimal(38, 0)
    assert lf.collect_schema()["S"] == pl.Decimal(38, 2)
    df = lf.collect().sort("id")
    assert df["I"].to_list() == [Decimal(12345678901234567890123456789012345), 7]
    check_filter_ids(lf, [pl.col("I") > 7, pl.col("S") == Decimal("1.25")])


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
@pytest.mark.parametrize("driver", ["adbc", "clickhouse_connect"])
def test_clickhouse_query_columns_compare_by_their_type(driver: str) -> None:
    lf = _scan(
        "clickhouse",
        driver,
        "SELECT number AS id, toString(number) AS s, "
        "CAST(if(number = 1, 'b', 'a') AS Enum8('a' = 1, 'b' = 2)) AS e "
        "FROM numbers(3)",
    )
    if lf.collect_schema()["e"] == pl.String:
        # the database casts the literal to the enum
        check_filter_ids(lf, [pl.col("e") == "nope", pl.col("e") > "a"])
        assert "filters 0/1" in lf.filter(pl.col("e") == "b").explain()
    assert "filters 1/1 (1 exact)" in lf.filter(pl.col("s") == "1").explain()


def _sqlalchemy_url(uri: str, driver: str) -> str:
    scheme, rest = uri.split("://", 1)
    return f"{scheme}+{driver}://{rest}"


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "psycopg2"])
def test_postgres_engine_connections_dont_keep_the_session_setup(driver: str) -> None:
    # the scan's READ ONLY and UTC settings don't go back into the user's pool: they
    # are restored, and the pooled connection is kept
    sqlalchemy = pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    engine = sqlalchemy.create_engine(
        _sqlalchemy_url(POSTGRES, driver), pool_size=1, max_overflow=0
    )

    @sqlalchemy.event.listens_for(engine, "connect")  # type: ignore[untyped-decorator]
    def setup(dbapi_conn: Any, _: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("SET TIME ZONE 'Asia/Tokyo'")
        cur.execute("SET application_name = 'it''s 100%'")
        cur.close()
        dbapi_conn.commit()

    try:
        with engine.connect() as conn:
            pid = conn.exec_driver_sql("SELECT pg_backend_pid()").scalar()
        assert pl.scan_database("edge", engine).select(pl.len()).collect().item() > 0
        with engine.connect() as conn:
            show = conn.exec_driver_sql
            assert show("SELECT pg_backend_pid()").scalar() == pid
            assert show("SHOW default_transaction_read_only").scalar() == "off"
            assert show("SHOW TimeZone").scalar() == "Asia/Tokyo"
            assert show("SHOW application_name").scalar() == "it's 100%"
            conn.exec_driver_sql("CREATE TEMPORARY TABLE review_engine_rw (i int)")
            conn.exec_driver_sql("INSERT INTO review_engine_rw VALUES (1)")
            conn.rollback()
    finally:
        engine.dispose()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_engine_connections_dont_keep_the_session_setup() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    uri = SERVERS["mysql"].uri
    assert uri is not None
    engine = sqlalchemy.create_engine(
        _sqlalchemy_url(uri, "pymysql"), pool_size=1, max_overflow=0
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SET time_zone = '+09:00'")
            conn.exec_driver_sql(
                "SET SESSION sql_mode = CONCAT(@@sql_mode, ',NO_BACKSLASH_ESCAPES')"
            )
            before = conn.exec_driver_sql(
                "SELECT CONNECTION_ID(), @@time_zone, @@sql_mode, "
                "@@transaction_read_only"
            ).one()
            conn.commit()
        assert pl.scan_database("edge", engine).select(pl.len()).collect().item() > 0
        with engine.connect() as conn:
            after = conn.exec_driver_sql(
                "SELECT CONNECTION_ID(), @@time_zone, @@sql_mode, "
                "@@transaction_read_only"
            ).one()
        assert tuple(after) == tuple(before)
    finally:
        engine.dispose()


def _postgres_in_transaction(conn: Any) -> bool:
    # libpq's transaction status, in psycopg and psycopg2: 2 is INTRANS
    return int(conn.info.transaction_status) == 2


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "psycopg2"])
@pytest.mark.parametrize("in_transaction", [False, True])
def test_postgres_live_connection_after_an_early_stop(
    module: str, in_transaction: bool
) -> None:
    # Polars may release an early-stopped read after `collect()` returned: nothing
    # the scan sends then may hit the user's next statements. A transaction the scan
    # opened is rolled back before `collect()` returns; in a transaction of the
    # user's, it reads in that transaction.
    import importlib

    driver = importlib.import_module(module)
    table = f"review_early_stop_{module}_{in_transaction}"
    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (k int)",
        ],
    )
    conn = driver.connect(POSTGRES)
    try:
        q = "SELECT g AS id FROM generate_series(1, 300000) g"
        for k in range(10):
            cur = conn.cursor()
            if in_transaction:
                cur.execute("SELECT 1")
            lf = pl.scan_database(q, conn, batch_size=1000)
            lf.filter((pl.col("id").hash() % 7) == 0).head(2).collect()
            assert _postgres_in_transaction(conn) is in_transaction
            cur.execute(f"INSERT INTO {table} VALUES (%s)", (k,))
            cur.close()
            conn.commit()
        cur = conn.cursor()
        cur.execute(f"SELECT count(*) FROM {table}")
        assert cur.fetchone()[0] == 10
        cur.close()
        conn.commit()
    finally:
        conn.close()


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "psycopg2"])
@pytest.mark.parametrize("in_transaction", [False, True])
@pytest.mark.parametrize("how", ["batches", "dropped_batches", "polars_error"])
def test_postgres_live_connection_keeps_the_users_writes(
    module: str, in_transaction: bool, how: str
) -> None:
    # The user's statements between batches, after an iterator dropped early, or after
    # a query that failed in Polars mid-read are theirs: the scan never rolls them back
    # (nor aborts their transaction) with one of its own.
    import importlib

    driver = importlib.import_module(module)
    table = f"review_writes_{module}_{in_transaction}_{how}"
    assert POSTGRES is not None
    _execute(
        "postgres17", [f"DROP TABLE IF EXISTS {table}", f"CREATE TABLE {table} (k int)"]
    )
    conn = driver.connect(POSTGRES)
    try:
        q = "SELECT g AS id FROM generate_series(1, 200000) g"
        lf = pl.scan_database(q, conn, batch_size=1000)
        for k in range(5):
            cur = conn.cursor()
            if in_transaction:
                cur.execute("SELECT 1")
            insert = f"INSERT INTO {table} VALUES ({k})"
            if how == "batches":
                for i, _ in enumerate(lf.collect_batches(chunk_size=1000)):
                    if i == 1:
                        cur.execute(insert)
            elif how == "dropped_batches":
                batches = lf.collect_batches(chunk_size=1000)
                next(batches)
                del batches
                cur.execute(insert)
            else:
                with pytest.raises(pl.exceptions.PolarsError):
                    lf.select(pl.col("id").cast(pl.Int16, strict=True)).collect()
                cur.execute(insert)
            if in_transaction or how != "batches":
                cur.execute(
                    "SELECT count(*) FROM pg_catalog.pg_cursors "
                    "WHERE name LIKE 'polars_scan_%'"
                )
                assert cur.fetchone()[0] == 0
            cur.close()
            conn.commit()
        cur = conn.cursor()
        cur.execute(f"SELECT count(*) FROM {table}")
        assert cur.fetchone()[0] == 5
        cur.close()
        conn.commit()
    finally:
        conn.close()
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
def test_postgres_engine_params_and_literals_with_colons() -> None:
    # `text()` binds `:lo`, also inside a pushed literal: those stay in Polars
    sqlalchemy = pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS review_colons",
            "CREATE TABLE review_colons (id int PRIMARY KEY, s text)",
            "INSERT INTO review_colons VALUES "
            "(1, ':lo'), (2, '0'), (3, 'ab'), (4, '\\:x'), (5, ':x')",
        ],
    )
    engine = sqlalchemy.create_engine(_sqlalchemy_url(POSTGRES, "psycopg2"))
    try:
        q = pl.scan_database(
            "SELECT id, s FROM review_colons WHERE id > :lo", engine, params={"lo": 0}
        )
        t = pl.scan_database("review_colons", engine)
        # parameters given as `execute()`'s own argument
        e = pl.scan_database(
            "SELECT id, s FROM review_colons WHERE id > :lo",
            engine,
            execute_options={"parameters": {"lo": 0}},
        )
        with sqlalchemy.orm.Session(engine) as session:
            # an ORM Session has no `exec_driver_sql`: its connection's is used
            o = pl.scan_database("review_colons", session)
            for lf in (q, t, e, o):
                for value, want in [
                    (":lo", [1]),
                    ("\\:x", [4]),
                    (":abc", []),
                    ("ab", [3]),
                ]:
                    out = lf.filter(pl.col("s") == value).collect()
                    assert out["id"].to_list() == want, value
    finally:
        engine.dispose()


@needs_postgres
def test_postgres_percent_literals_with_execute_options_parameters() -> None:
    # psycopg binds `parameters` given in `execute_options` and reads `%` in the SQL
    assert POSTGRES is not None
    lf = pl.scan_database(
        "SELECT id, s FROM edge WHERE id < %s",
        POSTGRES,
        driver="psycopg",
        execute_options={"params": [30]},
        partition_on=None,
    )
    out = lf.filter(pl.col("s") != "a%b").collect()
    expected = _reference("postgres17", _default_driver("postgres17")).filter(
        pl.col("id") < 30, pl.col("s") != "a%b"
    )
    assert sorted(out["id"].to_list()) == sorted(expected["id"].to_list())


@needs_postgres
def test_postgres_cancel_on_a_shared_live_connection() -> None:
    # one scan's cancel must not stop another scan's statement on the same connection:
    # a live connection is never cancelled
    import threading

    psycopg = pytest.importorskip("psycopg")

    from polars_database import _bridge

    assert POSTGRES is not None
    with psycopg.connect(POSTGRES) as conn:
        running = _bridge._Session(conn, owned=False, execute_options={})
        other = _bridge._Session(conn, owned=False, execute_options={})
        assert not other.can_cancel
        other_batches = other.execute("SELECT i FROM generate_series(1, 10) i", 1)
        next(other_batches)
        stop = threading.Event()

        def cancel() -> None:
            while not stop.is_set():
                other.cancel()

        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            sql = "SELECT i, md5(i::text) AS s FROM generate_series(1, 200000) i ORDER BY 2"
            n = sum(df.height for df in running.execute(sql, 5000))
        finally:
            stop.set()
            thread.join()
            other_batches.close()  # type: ignore[attr-defined]
        assert n == 200_000
        # the scans' transactions were rolled back once their queries ran
        assert conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_pymysql_cancel_doesnt_read_the_rest() -> None:
    # PyMySQL's unbuffered cursor would read the rest of the result when closed
    from polars_database import _bridge

    uri = SERVERS["mysql"].uri
    assert uri is not None
    conn = _mysql_factory(uri)()
    session = _bridge._Session(conn, owned=True, execute_options={})
    assert session.can_cancel
    sql = (
        "WITH RECURSIVE r (i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r "
        "WHERE i < 1000) SELECT a.i, MD5(a.i * b.i) AS s FROM r a CROSS JOIN r b"
    )
    batches = session.execute(sql, 1000)
    next(batches)
    packets = 0
    read_packet = conn._read_packet

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal packets
        packets += 1
        return read_packet(*args, **kwargs)

    conn._read_packet = counted
    session.cancel()
    batches.close()  # type: ignore[attr-defined]
    session.close()
    # not the ~1M rows left
    assert packets < 100


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_scans_sharing_an_unbuffered_live_connection(driver: str) -> None:
    # MySQL keeps one result open per connection: interleaved unbuffered reads would
    # end each other's results early
    uri = SERVERS["mysql"].uri
    assert uri is not None
    if driver == "pymysql":
        pymysql = pytest.importorskip("pymysql")

        conn = pymysql.connect(**_parts(uri), cursorclass=pymysql.cursors.SSCursor)
    else:
        mysql_connector = pytest.importorskip("mysql.connector")
        conn = mysql_connector.connect(**_parts(uri))
    try:
        a = pl.scan_database("edge", conn, batch_size=7)
        b = pl.scan_database("edge", conn, batch_size=7)
        n = pl.concat([a, b]).select(pl.len()).collect().item()
        assert n == 2 * a.select(pl.len()).collect().item()
        assert a.join(b, on="id").select(pl.len()).collect().item() == n // 2
    finally:
        conn.close()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_user_statements_during_an_unbuffered_live_scan(driver: str) -> None:
    # the scan's results are buffered: the user's statements between its batches,
    # or after it was dropped early, don't meet an unread result
    import gc

    uri = SERVERS["mysql"].uri
    assert uri is not None
    if driver == "pymysql":
        pymysql = pytest.importorskip("pymysql")

        conn = pymysql.connect(**_parts(uri), cursorclass=pymysql.cursors.SSCursor)
    else:
        mysql_connector = pytest.importorskip("mysql.connector")
        conn = mysql_connector.connect(**_parts(uri))

    def user_statement() -> None:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM edge")
        cur.fetchall()
        cur.close()

    try:
        lf = pl.scan_database("edge", conn, batch_size=7)
        total = 0
        for df in lf.collect_batches():
            total += df.height
            user_statement()
        assert total == lf.select(pl.len()).collect().item()
        batches = lf.collect_batches()
        next(iter(batches))
        del batches
        gc.collect()
        user_statement()
    finally:
        conn.close()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("how", ["pymysql", "sqlalchemy"])
def test_pymysql_scan_refuses_to_discard_the_users_pending_result(how: str) -> None:
    # PyMySQL drops unread rows of an unbuffered cursor (with a warning) when another
    # statement runs: also through a SQLAlchemy Connection. (The later results of a
    # multi-statement query are dropped silently, and not detected.)
    pymysql = pytest.importorskip("pymysql")
    uri = SERVERS["mysql"].uri
    assert uri is not None
    engine = None
    cur: Any = None
    result: Any = None
    if how == "pymysql":
        conn = pymysql.connect(**_parts(uri))
        cur = conn.cursor(pymysql.cursors.SSCursor)
        cur.execute("SELECT id FROM edge")
        cur.fetchone()
        source: Any = conn
    else:
        sqlalchemy = pytest.importorskip("sqlalchemy")
        engine = sqlalchemy.create_engine(_sqlalchemy_url(uri, "pymysql"))
        conn = engine.connect()
        result = conn.execution_options(stream_results=True).exec_driver_sql(
            "SELECT id FROM edge"
        )
        result.fetchmany(1)
        source = conn
    try:
        with pytest.raises(Exception, match="not been read to the end"):
            pl.scan_database("edge", source).collect()
        # the user's result is intact; read to the end, the scan works again
        if how == "pymysql":
            assert len(cur.fetchall()) > 0
            cur.close()
        else:
            assert len(result.fetchall()) > 0
        assert pl.scan_database("edge", source).collect().height > 0
    finally:
        conn.close()
        if engine is not None:
            engine.dispose()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("how", ["execute", "callproc"])
def test_pymysql_scan_after_a_stored_procedure_call(how: str) -> None:
    # the trailing OK packet of a CALL is not a pending result
    pymysql = pytest.importorskip("pymysql")
    uri = SERVERS["mysql"].uri
    assert uri is not None
    conn = pymysql.connect(**_parts(uri), autocommit=True)
    try:
        cur = conn.cursor()
        cur.execute("DROP PROCEDURE IF EXISTS polars_scan_call_p")
        cur.execute("CREATE PROCEDURE polars_scan_call_p() BEGIN SELECT 1 AS a; END")
        if how == "execute":
            cur.execute("CALL polars_scan_call_p()")
        else:
            cur.callproc("polars_scan_call_p")
        assert cur.fetchall() == ((1,),)
        assert pl.scan_database("SELECT 7 AS x", conn).collect().item() == 7
        cur.execute("DROP PROCEDURE polars_scan_call_p")
    finally:
        conn.close()


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_autocommit_engine_still_streams_outside_postgres(server: str) -> None:
    # only a Postgres named cursor needs a transaction: other dialects stream in
    # AUTOCOMMIT too (the whole result in memory otherwise)
    uri = SERVERS[server].uri
    if uri is None:
        pytest.skip(f"no {server} URI")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("pymysql")
    engine = sqlalchemy.create_engine(
        _sqlalchemy_url(uri, "pymysql"), isolation_level="AUTOCOMMIT"
    )
    seen: list[str] = []

    @sqlalchemy.event.listens_for(engine, "before_cursor_execute")  # type: ignore[untyped-decorator]
    def hook(conn: Any, cursor: Any, stmt: str, *args: Any) -> None:
        if "polars_stream_probe" in stmt:
            seen.append(type(cursor).__name__)

    try:
        pl.scan_database("SELECT 1 AS polars_stream_probe", engine).collect()
    finally:
        engine.dispose()
    assert seen
    assert all("SS" in name or "Unbuffered" in name for name in seen), seen


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "psycopg2"])
def test_postgres_autocommit_connections_are_read(module: str) -> None:
    # a named cursor needs a transaction: an autocommit connection (a factory's, an
    # Engine's with isolation_level="AUTOCOMMIT") is read with one that outlives the
    # statement
    import importlib

    driver = importlib.import_module(module)
    assert POSTGRES is not None
    q = "SELECT g AS id FROM generate_series(1, 5000) g"

    def connect() -> Any:
        conn = driver.connect(POSTGRES)
        conn.autocommit = True
        return conn

    assert pl.scan_database(q, connect, batch_size=1000).collect().height == 5000
    # (a table: a query over a table function (`generate_series`) is not proven to
    # repeat, so it is not partitioned)
    lf = pl.scan_database("orders", connect, partition_on="id", partition_num=3)
    whole = pl.scan_database("orders", connect, partition_on=None)
    assert lf.select("id").collect().height == whole.select("id").collect().height > 0
    sqlalchemy = pytest.importorskip("sqlalchemy")
    url = POSTGRES.replace("postgresql://", f"postgresql+{module}://", 1)
    engine = sqlalchemy.create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        assert pl.scan_database(q, engine, batch_size=1000).collect().height == 5000
    finally:
        engine.dispose()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "pg8000", "sqlalchemy"])
def test_postgres_credential_provider_with_user_only(driver: str) -> None:
    # the URI's password stays
    pytest.importorskip(
        {"adbc": "polars", "sqlalchemy": "sqlalchemy"}.get(driver, driver)
    )
    assert POSTGRES is not None
    user = urlsplit(POSTGRES).username
    lf = pl.scan_database(
        "SELECT 1 AS x",
        POSTGRES,
        driver=driver,
        credential_provider=lambda: {"user": user},
    )
    assert lf.collect().item() == 1


@needs_postgres
def test_postgres_uri_query_parameters_reach_pg8000() -> None:
    pytest.importorskip("pg8000")
    assert POSTGRES is not None
    lf = pl.scan_database(
        "SELECT current_setting('search_path') AS p",
        f"{POSTGRES}?options=-c%20search_path%3Da_path",
        driver="pg8000",
    )
    assert lf.collect().item() == "a_path"
    # TLS is required: pg8000 must not connect without it
    tls = f"{POSTGRES}?sslmode=verify-full&sslrootcert=/nonexistent"
    with pytest.raises(pl.exceptions.ComputeError, match="/nonexistent"):
        pl.scan_database("SELECT 1 AS x", tls, driver="pg8000").collect()
    with pytest.raises(pl.exceptions.ComputeError, match=r"\['bogus'\]"):
        pl.scan_database(
            "SELECT 1 AS x", f"{POSTGRES}?bogus=1", driver="pg8000"
        ).collect()


@needs_postgres
def test_postgres_sqlalchemy_pg8000_gets_no_connect_timeout() -> None:
    pytest.importorskip("pg8000")
    pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    uri = POSTGRES.replace("postgresql://", "postgresql+pg8000://", 1)
    lf = pl.scan_database("SELECT 1 AS x", uri, driver="sqlalchemy")
    assert lf.collect().item() == 1


@needs_postgres
def test_redshift_uri_through_psycopg2() -> None:
    # a Postgres server stands in for Redshift: libpq takes only `postgresql://`
    pytest.importorskip("psycopg2")
    assert POSTGRES is not None
    uri = POSTGRES.replace("postgresql://", "redshift://", 1)
    lf = pl.scan_database("SELECT 1 AS x", uri, driver="psycopg2")
    assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_uri_query_parameters_reach_the_driver(driver: str) -> None:
    pytest.importorskip(driver)
    uri = SERVERS["mysql"].uri
    assert uri is not None
    lf = pl.scan_database(
        "SELECT @@character_set_client AS c", f"{uri}?charset=latin1", driver=driver
    )
    assert lf.collect().item() == "latin1"
    with pytest.raises(pl.exceptions.ComputeError, match=r"\['bogus'\]"):
        pl.scan_database("SELECT 1 AS x", f"{uri}?bogus=1", driver=driver).collect()


@needs_postgres
def test_postgres_common_libpq_parameters_with_pg8000() -> None:
    pytest.importorskip("pg8000")
    assert POSTGRES is not None
    query = "connect_timeout=5&target_session_attrs=any&client_encoding=utf8"
    lf = pl.scan_database("SELECT 1 AS x", f"{POSTGRES}?{query}", driver="pg8000")
    assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_ssl_parameter(driver: str) -> None:
    pytest.importorskip(driver)
    uri = SERVERS["mysql"].uri
    assert uri is not None
    for q in ["ssl=true", "ssl=false", "ssl_mode=REQUIRED"]:
        lf = pl.scan_database("SELECT 1 AS x", f"{uri}?{q}", driver=driver)
        assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["adbc", "pymysql"])
def test_mysql_empty_table_is_typed(driver: str) -> None:
    table = f"review_empty_{driver}"
    _execute(
        "mysql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, d date, dt datetime(6), x double)",
        ],
    )
    for source in [table, f"SELECT d, dt, x FROM {table}"]:
        schema = _scan("mysql", driver, source).collect_schema()
        assert schema["d"] == pl.Date, source
        assert schema["dt"] == pl.Datetime("us"), source
        assert schema["x"] == pl.Float64, source


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "pg8000"])
def test_postgres_empty_query_arrays_are_typed(driver: str) -> None:
    table = f"review_empty_arr_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (a int4[], t text[], f float8[])",
        ],
    )
    lf = _scan("postgres17", driver, f"SELECT * FROM {table}")
    assert dict(lf.collect_schema()) == {
        "a": pl.List(pl.Int32),
        "t": pl.List(pl.String),
        "f": pl.List(pl.Float64),
    }
    _execute(
        "postgres17", [f"INSERT INTO {table} VALUES ('{{1}}', '{{x}}', '{{1.5}}')"]
    )
    assert lf.collect().rows() == [([1], ["x"], [1.5])]


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_query_time_zone_offsets() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_qtz'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_qtz ("id" NUMBER(9), ts TIMESTAMP WITH TIME ZONE)',
            "INSERT INTO review_qtz VALUES (1, TIMESTAMP '2020-01-02 12:00:00 +05:00')",
            "INSERT INTO review_qtz VALUES (2, TIMESTAMP '2020-01-02 12:00:00 -08:00')",
        ],
    )
    noon = datetime(2020, 1, 2, 12, tzinfo=UTC)
    # read as UTC with pushdown or without
    for pushdown in [True, False]:
        lf = _scan(
            "oracle", "oracledb", 'SELECT "id", ts FROM review_qtz', pushdown=pushdown
        )
        assert lf.select("TS").sort("TS").collect()["TS"].to_list() == [
            datetime(2020, 1, 2, 7, tzinfo=UTC),
            datetime(2020, 1, 2, 20, tzinfo=UTC),
        ]
        assert lf.filter(pl.col("TS") < noon).select("id").collect().item() == 1


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
@pytest.mark.parametrize("driver", ["adbc", "pymysql", "connectorx"])
def test_mysql_query_year_and_json_columns(name: str, driver: str) -> None:
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    table = f"review_myq_{driver}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, y year, j json, s varchar(5))",
            f"INSERT INTO {table} VALUES (1, 1970, '{{\"a\": 1}}', 'x'), "
            "(2, 2001, '\"x\"', 'y'), (3, 0, NULL, NULL), (4, NULL, NULL, NULL)",
        ],
    )
    sources = [
        (f"SELECT id, y, j, s FROM {table}", {}),
        # the user's schema doesn't say an integer isn't a YEAR
        (table, {"schema": {"id": pl.Int32, "y": pl.Int16}}),
    ]
    for source, kwargs in sources:
        lf = _scan(name, driver, source, **kwargs)
        dtype = lf.collect_schema()["y"]
        if dtype.is_integer():
            predicates = [pl.col("y") == 70, pl.col("y") == 1, pl.col("y") > 50]
            predicates += [pl.col("y").is_in([1, 70]), pl.col("y") == 1970]
        else:
            predicates = [pl.col("y") == "0", pl.col("y") == "70"]
        if "j" in lf.collect_schema():
            predicates += [pl.col("j") == '{"a": 1}', pl.col("j") == "x"]
            predicates += [pl.col("s") == "x"]
        check_filter_ids(lf, predicates)


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_query_types_and_collations() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_ora_q'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_ora_q ("id" NUMBER(9), ts TIMESTAMP WITH TIME ZONE, '
            "s VARCHAR2(5), cl CLOB)",
            "INSERT INTO review_ora_q VALUES "
            "(1, TIMESTAMP '2024-01-01 12:00:00 +05:00', 'a', 'x')",
            "INSERT INTO review_ora_q VALUES "
            "(2, TIMESTAMP '2024-01-01 12:00:00 -08:00', 'b', 'y')",
            "INSERT INTO review_ora_q VALUES (3, NULL, NULL, NULL)",
        ],
    )
    lf = _scan("oracle", "oracledb", 'SELECT "id", ts, s, cl FROM review_ora_q')
    utc = datetime(2024, 1, 1, 7)
    check_filter_ids(
        lf,
        [
            pl.col("TS") == pl.lit(utc).dt.replace_time_zone("UTC"),
            pl.col("TS") > pl.lit(utc).dt.replace_time_zone("UTC"),
            pl.col("CL") == "x",
            pl.col("S") == "a",
        ],
    )
    # the table's column collations are looked up: a default one is compared
    table = _scan("oracle", "oracledb", "REVIEW_ORA_Q")
    assert "filters 1/1" in table.filter(pl.col("S") == "a").explain()


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "pg8000", "sqlalchemy"])
def test_postgres_uuid_from_row_drivers(driver: str) -> None:
    # row drivers return `uuid.UUID` objects: read as text, not as an Object column
    table = f"review_uuid_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, u uuid)",
            f"INSERT INTO {table} VALUES "
            "(1, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'), (2, NULL)",
        ],
    )
    for source in [table, f"SELECT * FROM {table}"]:
        lf = _scan("postgres17", driver, source)
        assert lf.collect().sort("id")["u"].to_list() == [
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            None,
        ]
        uuid = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
        out = lf.filter(pl.col("u") == uuid).collect()
        assert out["id"].to_list() == [1]


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_nul_literal_without_backslash_escapes() -> None:
    # NUL is written as `\0`, which NO_BACKSLASH_ESCAPES reads as backslash, zero; so
    # are the other control characters sqlglot escapes (`\n`, `\t`, `\r`, `\b`, `\Z`)
    _execute(
        "mysql",
        [
            "DROP TABLE IF EXISTS review_nul",
            "CREATE TABLE review_nul (id int, s varchar(10) COLLATE utf8mb4_bin)",
            "INSERT INTO review_nul VALUES (1, 'a\\0b'), (2, 'a'), (3, 'a\\\\0b'), "
            "(4, 'a\\nb'), (5, 'a\\tb'), (6, 'a\\Zb'), (7, 'a\\\\nb')",
        ],
    )
    assert SERVERS["mysql"].uri is not None
    conn = _mysql_factory(SERVERS["mysql"].uri)()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SET SESSION sql_mode = "
                "CONCAT(@@SESSION.sql_mode, ',NO_BACKSLASH_ESCAPES')"
            )
        check_filter_ids(
            pl.scan_database("review_nul", conn),
            [
                pl.col("s") == "a\0b",
                pl.col("s").is_in(["a\0b"]),
                pl.col("s") == "a\\0b",
                pl.col("s") == "a\nb",
                pl.col("s") != "a\nb",
                pl.col("s").is_in(["a\tb", "a\x1ab"]),
                pl.col("s") == "a\\nb",
            ],
        )
    finally:
        conn.close()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_domains_compare_as_their_base_type(driver: str) -> None:
    table = f"review_pgdom_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"DROP DOMAIN IF EXISTS {table}_d2",
            f"DROP DOMAIN IF EXISTS {table}_d1",
            f"CREATE DOMAIN {table}_d1 AS varchar(5)",
            f"CREATE DOMAIN {table}_d2 AS {table}_d1",
            f"CREATE TABLE {table} (id int, a {table}_d1, b {table}_d2)",
            f"INSERT INTO {table} VALUES (1, 'x', 'y'), (2, 'z', NULL)",
        ],
    )
    for source in [table, f"SELECT * FROM {table}"]:
        lf = _scan("postgres17", driver, source)
        check_filter_ids(lf, [pl.col("a") == "x", pl.col("b") == "y"])
        # a domain over a domain over varchar: compared as text
        assert "(1 exact)" in lf.filter(pl.col("b") == "y").explain(), source


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_wide_query_is_described() -> None:
    # the description of 300 columns is longer than a VARCHAR2
    cols = ", ".join(f"{i} AS some_longer_column_name_{i}" for i in range(300))
    lf = _scan(
        "oracle",
        "oracledb",
        f"SELECT {cols}, CAST('x' AS VARCHAR2(5)) AS s FROM dual",
    )
    assert "filters 1/1" in lf.filter(pl.col("S") == "x").explain()
    assert lf.filter(pl.col("S") == "x").collect().height == 1


@needs_postgres
def test_postgres_live_early_stops_dont_leave_cursors_behind() -> None:
    # early-stopped scans' cursors are closed in the transaction they live in, and
    # names aren't reused: another scan never declares a cursor that still exists
    psycopg = pytest.importorskip("psycopg")

    assert POSTGRES is not None
    q = "SELECT g AS id FROM generate_series(1, 200000) g"
    with psycopg.connect(POSTGRES) as conn:
        for _ in range(20):
            lf = pl.scan_database(q, conn, batch_size=1000, pushdown=False)
            assert lf.head(3).collect().height == 3
            open_cursors = conn.execute(
                "SELECT count(*) FROM pg_cursors WHERE name LIKE 'polars_scan_%'"
            ).fetchone()
            assert open_cursors is not None
            assert open_cursors[0] <= 2
        conn.rollback()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_bigint_unsigned_query_source(driver: str) -> None:
    uri = SERVERS["mysql"].uri
    assert uri is not None
    table = f"review_unsigned_{driver.replace('.', '_')}"
    _execute(
        "mysql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT, bu BIGINT UNSIGNED, iu INT UNSIGNED)",
            f"INSERT INTO {table} VALUES (1, 18446744073709551615, 4294967295), "
            "(2, 5, 0)",
        ],
    )
    for source in [table, f"SELECT id, bu, iu FROM {table}"]:
        df = _scan("mysql", driver, source).collect().sort("id")
        assert df["bu"].to_list() == [2**64 - 1, 5]
        assert df["iu"].to_list() == [2**32 - 1, 0]


@needs_postgres
@pytest.mark.parametrize("kind", ["live", "factory"])
def test_postgres_psycopg2_bytea(kind: str) -> None:
    # psycopg2 returns bytea as memoryview values
    psycopg2 = pytest.importorskip("psycopg2")

    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS review_bytea_{kind}",
            f"CREATE TABLE review_bytea_{kind} (id int, b bytea)",
            f"INSERT INTO review_bytea_{kind} VALUES (1, '\\x00ff'), (2, NULL)",
        ],
    )
    for source in [f"review_bytea_{kind}", f"SELECT * FROM review_bytea_{kind}"]:
        if kind == "live":
            with contextlib.closing(psycopg2.connect(POSTGRES)) as conn:
                df = pl.scan_database(source, conn).collect()
        else:
            df = pl.scan_database(source, lambda: psycopg2.connect(POSTGRES)).collect()
        assert df.sort("id")["b"].to_list() == [b"\x00\xff", None]


@needs_postgres
def test_postgres_unparsable_write_runs_nowhere_and_queries_run_once() -> None:
    # sqlglot can't parse `USING <`: a WITH it can't check is refused where sessions
    # aren't read-only, and a query is never run whole to find its schema
    psycopg = pytest.importorskip("psycopg")

    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS fix_once_w",
            "DROP SEQUENCE IF EXISTS fix_once_s",
            "CREATE TABLE fix_once_w (id serial, v int)",
            "CREATE SEQUENCE fix_once_s",
        ],
    )
    write = (
        "WITH d AS (INSERT INTO fix_once_w (v) VALUES (1) RETURNING id, v) "
        "SELECT * FROM d WHERE id IS NOT NULL ORDER BY id USING <"
    )
    live = psycopg.connect(POSTGRES, autocommit=True)
    try:
        for source in [POSTGRES, live]:
            kw: dict[str, Any] = {"driver": "connectorx"} if source is POSTGRES else {}
            for pushdown in [True, False]:
                lf = pl.scan_database(
                    write, source, source_type="query", pushdown=pushdown, **kw
                )
                with pytest.raises(
                    pl.exceptions.InvalidOperationError, match="can't be parsed"
                ):
                    lf.collect()
        # these sessions are read-only: the database refuses the write itself
        for driver in ["adbc", "psycopg"]:
            with pytest.raises(pl.exceptions.ComputeError):
                pl.scan_database(write, POSTGRES, driver=driver).collect()
        assert live.execute("SELECT count(*) FROM fix_once_w").fetchone() == (0,)

        nextval = "SELECT nextval('fix_once_s') AS n"
        expected = 0
        for source in [POSTGRES, live]:
            kw = {"driver": "connectorx"} if source is POSTGRES else {}
            for pushdown in [False, True]:
                n = pl.scan_database(nextval, source, pushdown=pushdown, **kw).collect()
                expected += 1
                assert n.item() == expected
                last = live.execute("SELECT last_value FROM fix_once_s").fetchone()
                assert last == (expected,)
    finally:
        live.close()
        _execute(
            "postgres17",
            ["DROP TABLE IF EXISTS fix_once_w", "DROP SEQUENCE IF EXISTS fix_once_s"],
        )


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_query_order_by_is_kept(name: str) -> None:
    # MariaDB drops the ORDER BY of a derived table without LIMIT: such a query isn't
    # wrapped there
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS fix_order",
            "CREATE TABLE fix_order (id int PRIMARY KEY, x int, KEY (x))",
            "INSERT INTO fix_order VALUES "
            + ", ".join(f"({i}, {(i * 31) % 100 + 1})" for i in range(100)),
        ],
    )
    try:
        for driver, query in [
            ("pymysql", "SELECT id, x FROM fix_order ORDER BY x DESC"),
            ("factory", "SELECT id, x FROM fix_order ORDER BY x DESC"),
            # parenthesised, the order is on the inner query
            ("pymysql", "((SELECT id, x FROM fix_order ORDER BY x DESC))"),
        ]:
            lf = _scan(name, driver, query)
            assert lf.head(3).collect()["x"].to_list() == [100, 99, 98]
            assert lf.collect()["x"].to_list() == list(range(100, 0, -1))
            out = lf.filter(pl.col("x") < 50).head(2).collect()
            assert out["x"].to_list() == [49, 48]
            pushed = "pushdown: off" not in lf.explain()
            assert pushed is (name == "mysql")
    finally:
        _execute(name, ["DROP TABLE IF EXISTS fix_order"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_year_partitions(name: str) -> None:
    # a cut of 1-99 would be read as a year 1970-2069
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS fix_year_parts",
            "CREATE TABLE fix_year_parts (id int, y year)",
            "INSERT INTO fix_year_parts VALUES "
            "(1, 0), (2, 2000), (3, 2155), (4, 1901), (5, NULL)",
        ],
    )
    try:
        for kw in [{}, {"partition_range": (0, 2155)}]:
            lf = _scan(
                name,
                "pymysql",
                "fix_year_parts",
                partition_on="y",
                partition_num=40,
                **kw,
            )
            assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3, 4, 5]
    finally:
        _execute(name, ["DROP TABLE IF EXISTS fix_year_parts"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_float_with_digits_and_integer_booleans(name: str) -> None:
    # FLOAT(7,3) and FLOAT: row drivers and ADBC read them rounded (FLOAT to 6
    # digits), the database compares the stored value; booleans are integers, any of
    # which but 0 is true
    table = "fix_float_bool"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, f2 FLOAT(7,3), b TINYINT(1), f FLOAT)",
            f"INSERT INTO {table} VALUES (1, 1.234, 1, 1234.567), "
            "(2, 3.3, 2, 0.1), (3, NULL, 0, NULL), (4, 2.5, -1, 99.9995), "
            "(5, 0.1, NULL, 1.5)",
        ],
    )
    drivers = ["pymysql", "mysql.connector", "adbc"]
    try:
        for driver in drivers if name == "mysql" else drivers[:1]:
            for source in [table, f"SELECT * FROM {table}"]:
                lf = _scan(name, driver, source)
                f = lf.collect().sort("id")["f"]
                assert f.dtype == pl.Float32
                stored = pl.Series(
                    [1234.567, 0.1, None, 99.9995, 1.5], dtype=pl.Float32
                )
                assert f.to_list() == stored.to_list()
                check_filter_ids(
                    lf,
                    [
                        pl.col("f2").is_in([1.234, 3.3]),
                        pl.col("f2") == pl.lit(1.234, pl.Float32),
                        pl.col("f2") == 1.234,
                        pl.col("f2") > 2.5,
                        pl.col("f2") <= 0.1,
                        pl.col("f") == pl.lit(1234.567, pl.Float32),
                        pl.col("f") > 1234.5699,
                        pl.col("f") < pl.lit(99.9995, pl.Float32),
                    ],
                )
                schema = {
                    "id": pl.Int32,
                    "f2": pl.Float32,
                    "b": pl.Boolean,
                    "f": pl.Float32,
                }
                lf = _scan(name, driver, source, schema=schema)
                check_filter_ids(
                    lf,
                    [
                        pl.col("b"),
                        ~pl.col("b"),
                        pl.col("b") == True,  # noqa: E712
                        pl.col("b") != True,  # noqa: E712
                        pl.col("b") == False,  # noqa: E712
                        pl.col("b").is_in([True]),
                        pl.col("b") | (pl.col("id") == 3),
                    ],
                )
                assert "<> 0" in lf.filter(pl.col("b")).explain()
    finally:
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


def _in_transaction(conn: Any) -> bool:
    module = type(conn).__module__.split(".")[0]
    if module == "pg8000":
        return bool(conn._in_transaction)
    cur = conn.cursor()
    try:
        # an OK packet with the server's status (PyMySQL keeps only theirs)
        cur.execute("DO 0")
    finally:
        cur.close()
    if module == "pymysql":
        return bool(conn.server_status & 1)
    return bool(conn.in_transaction)


@pytest.mark.parametrize(
    ("name", "module"),
    [("postgres17", "pg8000"), ("mysql", "pymysql"), ("mysql", "mysql.connector")],
)
def test_live_connection_left_outside_a_transaction(name: str, module: str) -> None:
    # the transaction a scan opens on a live connection is rolled back: it would hold
    # locks that block another session's DDL
    import importlib

    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    driver = importlib.import_module(module)
    p = _parts(server.uri)
    if module == "pg8000":
        conn = driver.dbapi.connect(**p)
    else:
        conn = driver.connect(**p)
    table = f"fix_live_tx_{module.replace('.', '_')}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, v int)",
            f"INSERT INTO {table} VALUES (1, 1), (2, 2)",
        ],
    )
    try:
        for source in [table, f"SELECT * FROM {table}"]:
            lf = pl.scan_database(source, conn).filter(pl.col("id") < 2)
            assert lf.collect().height == 1
            assert not _in_transaction(conn)
        # a transaction of the user's stays theirs
        cur = conn.cursor()
        cur.execute(f"UPDATE {table} SET v = 3 WHERE id = 2")
        cur.close()
        assert pl.scan_database(table, conn).collect().height == 2
        assert _in_transaction(conn)
        conn.rollback()
    finally:
        conn.close()
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
def test_postgres_engine_pool_of_one_and_a_failed_lookup() -> None:
    # The name as written is looked up on a new session once the first is closed; a
    # pool timeout isn't retried. A failed catalog lookup doesn't hide the error.
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("pg8000")
    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            'DROP TABLE IF EXISTS "FixFold"',
            'CREATE TABLE "FixFold" (id int PRIMARY KEY)',
        ],
    )
    try:
        for driver in ["psycopg", "pg8000"]:
            url = _sqlalchemy_url(POSTGRES, driver)
            engine = sqlalchemy.create_engine(
                url, pool_size=1, max_overflow=0, pool_timeout=3
            )
            try:
                start = time.monotonic()
                assert pl.scan_database("FixFold", engine).collect().height == 0
                assert time.monotonic() - start < 3
                with pytest.raises(pl.exceptions.ComputeError, match="does not exist"):
                    pl.scan_database("fix_no_such_table", engine).collect()
            finally:
                engine.dispose()
    finally:
        _execute("postgres17", ['DROP TABLE IF EXISTS "FixFold"'])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_query_ending_in_comments(name: str) -> None:
    for query in [
        "SELECT id FROM edge WHERE id < 3 # a comment",
        "SELECT id FROM edge WHERE id < 3; -- done",
        "SELECT id FROM edge WHERE id < 3; # done",
    ]:
        lf = _scan(name, "pymysql", query)
        assert sorted(lf.filter(pl.col("id") >= 1).collect()["id"].to_list()) == [1, 2]


needs_clickhouse = pytest.mark.skipif(
    SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI"
)


@needs_clickhouse
def test_clickhouse_connectorx_reads_every_row_or_fails() -> None:
    # ConnectorX dropped rows silently: a table with a decimal, an array of nullable
    # values; text that isn't UTF-8 stopped the read (clickhouse_connect hex-encoded it)
    pytest.importorskip("connectorx")
    table = "rvb_ch_rows"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, d Decimal(38, 10), "
            "a Array(Nullable(Int64)), ai Array(Int32), s String) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} SELECT number, number / 2, [number, NULL], [1], "
            "toString(number) FROM numbers(1000)",
        ],
    )
    try:
        lf = _scan("clickhouse", "connectorx", table).select("id", "d", "ai", "s")
        assert lf.collect().height == 1000
        query = _scan("clickhouse", "connectorx", f"SELECT id, d FROM {table}")
        assert query.collect()["d"].sum() == Decimal("249750")
        for source in [table, f"SELECT id, a FROM {table}"]:
            with pytest.raises(pl.exceptions.InvalidOperationError, match="column 'a'"):
                _scan("clickhouse", "connectorx", source).collect()
        assert _scan("clickhouse", "adbc", table).collect().height == 1000
        _execute(
            "clickhouse", [f"INSERT INTO {table} VALUES (1000, 0, [], [], unhex('ff'))"]
        )
        for driver in ["connectorx", "clickhouse_connect"]:
            for source in [table, f"SELECT id, s FROM {table}"]:
                lf = _scan("clickhouse", driver, source).select("id", "s")
                # (ConnectorX loses the message once the result streams)
                with pytest.raises(
                    pl.exceptions.ComputeError, match=r"not valid UTF-8|500 Internal"
                ):
                    lf.collect()
                # without the column, the rows read
                assert lf.select("id").collect().height == 1001
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_connectorx_array_of_decimals_fails_only_when_read() -> None:
    # #442: ConnectorX fails on an array of decimals even for no rows, so the table
    # couldn't be planned: any projection failed
    pytest.importorskip("connectorx")
    table = "rvb_ch_array_decimal"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, ad Array(Decimal(38, 20)), "
            "d Decimal(38, 20)) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, [1.12345678901234567890], "
            "1.12345678901234567890)",
        ],
    )
    try:
        lf = _scan("clickhouse", "connectorx", table)
        assert lf.select("id", "d").collect().rows() == [
            (1, Decimal("1.12345678901234567890"))
        ]
        with pytest.raises(pl.exceptions.InvalidOperationError, match="column 'ad'"):
            lf.collect()
        # a query that selects it can't be planned
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="array of decimals"
        ):
            _scan("clickhouse", "connectorx", f"SELECT * FROM {table}").collect()
        query = _scan("clickhouse", "connectorx", f"SELECT id, d FROM {table}")
        assert query.collect().rows() == [(1, Decimal("1.12345678901234567890"))]
        assert _scan("clickhouse", "adbc", table).collect().height == 1
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_time_zones_and_nanoseconds_by_every_driver() -> None:
    # A zone keeps its case (it was lower-cased, and the driver's dtype used), and
    # DateTime64(7..9) keeps its nanoseconds (clickhouse_connect and ConnectorX cut
    # them)
    from datetime import datetime
    from zoneinfo import ZoneInfo

    table = "rvb_ch_tz"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, dt DateTime('Asia/Tokyo'), "
            "ns DateTime64(9, 'America/New_York'), naive Nullable(DateTime64(7))) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, '2024-01-02 03:04:05', "
            "'2024-11-03 01:30:00.123456789', '2200-01-01 00:00:00.1234567'), "
            "(2, '2024-06-02 03:04:05', '2001-01-01 00:00:00.000000001', NULL)",
        ],
    )
    try:
        frames = {}
        for driver in ["adbc", "clickhouse_connect", "connectorx"]:
            for source in [table, f"SELECT * FROM {table}"]:
                lf = _scan("clickhouse", driver, source)
                df = lf.collect().sort("id")
                assert lf.collect_schema() == df.schema
                frames[driver, source] = df
        first, *rest = frames.values()
        assert first.schema["dt"] == pl.Datetime("ms", "Asia/Tokyo")
        assert first.schema["ns"] == pl.Datetime("ns", "America/New_York")
        assert first["dt"][0] == datetime(
            2024, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("Asia/Tokyo")
        )
        assert first["ns"].dt.nanosecond().to_list() == [123456789, 1]
        assert first["naive"].dt.nanosecond()[0] == 123456700
        for df in rest:
            assert_frame_equal(df, first)
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_nanoseconds_and_numbers_read_exactly() -> None:
    # TIMESTAMP(9) kept its microseconds only and wrapped outside 1677-2262; an
    # unconstrained NUMBER in scientific notation read as 0, and one in a query was a
    # rounded Float64; NUMBER(5) read differently from a table and a query
    from datetime import datetime

    table = "rvb_ora_exact"
    _execute(
        "oracle",
        [
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            f'CREATE TABLE {table} ("id" NUMBER(5), ts TIMESTAMP(9), '
            "tz TIMESTAMP(9) WITH TIME ZONE, n NUMBER)",
            f"INSERT INTO {table} VALUES (1, TIMESTAMP '2024-01-02 03:04:05.123456789', "
            "TIMESTAMP '2024-01-02 03:04:05.123456789 +05:00', 12345678901234567891)",
            f"INSERT INTO {table} VALUES (2, NULL, NULL, 1.5e25)",
        ],
    )
    try:
        frames = []
        for driver in ["oracledb", "sqlalchemy"]:
            for source in [
                table.upper(),
                f'SELECT "id", ts, tz, n FROM {table}',
                f'SELECT "id", ts, tz, n * 1 AS n FROM {table}',
            ]:
                lf = _scan("oracle", driver, source)
                df = lf.collect().sort("id")
                assert lf.collect_schema() == df.schema
                frames.append(df)
        for df in frames:
            assert_frame_equal(df, frames[0])
        df = frames[0]
        assert df.schema == pl.Schema(
            {  # type: ignore[arg-type]
                "id": pl.Int64,
                "TS": pl.Datetime("ns"),
                "TZ": pl.Datetime("ns", "UTC"),
                "N": pl.Decimal(38, 9),
            }
        )
        assert df["TS"].dt.nanosecond()[0] == 123456789
        assert df["TZ"].dt.replace_time_zone(None)[0] == datetime(
            2024, 1, 1, 22, 4, 5, 123456
        )
        assert df["N"].to_list() == [
            Decimal("12345678901234567891"),
            Decimal("15000000000000000000000000"),
        ]
        _execute(
            "oracle",
            [
                f"INSERT INTO {table} VALUES (3, TIMESTAMP '9999-12-31 23:59:59.5', "
                "NULL, 1e-100)"
            ],
        )
        for source in [table.upper(), f'SELECT "id", ts, n FROM {table}']:
            lf = _scan("oracle", "oracledb", source)
            with pytest.raises(
                pl.exceptions.SchemaError, match="'TS' has the value 9999"
            ):
                lf.select("TS").collect()
            with pytest.raises(
                pl.exceptions.SchemaError, match="'N' has the value 1E-100"
            ):
                lf.select("N").collect()
    finally:
        _execute(
            "oracle",
            [
                f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; EXCEPTION WHEN OTHERS THEN NULL; END;"
            ],
        )


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("module", ["pymysql", "mysql.connector"])
def test_mysql_live_connection_in_another_time_zone(module: str) -> None:
    # a live connection has no session setup: its TIMESTAMP values were read in its
    # time zone, but labelled UTC
    from datetime import datetime

    driver = pytest.importorskip(module)
    server = SERVERS["mysql"]
    assert server.uri is not None
    table = f"rvb_my_tz_{module.replace('.', '_')}"
    _execute(
        "mysql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, ts timestamp(6) NULL)",
            "SET time_zone = '+00:00'",
            f"INSERT INTO {table} VALUES (1, '2024-01-02 03:04:05.5'), (2, NULL)",
        ],
    )
    conn = driver.connect(**_parts(server.uri))
    try:
        cur = conn.cursor()
        cur.execute("SET time_zone = '+09:00'")
        cur.close()
        instant = datetime(2024, 1, 2, 3, 4, 5, 500000, tzinfo=UTC)
        lf = pl.scan_database(table, conn)
        assert lf.collect_schema()["ts"] == pl.Datetime("us", "UTC")
        assert lf.collect().sort("id")["ts"].to_list() == [instant, None]
        assert lf.filter(pl.col("ts") == instant).collect()["id"].to_list() == [1]
        # a query's TIMESTAMP is the same instants, as the table reads them
        query = pl.scan_database(f"SELECT id, ts FROM {table}", conn)
        assert query.collect_schema()["ts"] == pl.Datetime("us", "UTC")
        assert query.collect().sort("id")["ts"].to_list() == [instant, None]
    finally:
        conn.close()
        _execute("mysql", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
@pytest.mark.parametrize(
    "driver", ["psycopg", "psycopg2", "pg8000", "adbc", "connectorx"]
)
def test_postgres_time_declared_as_text_under_schema(driver: str) -> None:
    # `schema=` has no type to look up: a plain `time` declared String reads its text; a
    # `timetz` never loses its offset (a Python driver's description tells it; an Arrow
    # driver can't, and refuses a String that is a time of day, naming Postgres' cast)
    table = f"r44_pg_timetz_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, t time, tz timetz)",
            f"INSERT INTO {table} VALUES (1, '10:00:00', '10:00:00+02'), (2, NULL, NULL)",
        ],
    )
    python = driver in ("psycopg", "psycopg2", "pg8000")
    try:
        schema: dict[str, PolarsDataType] = {"id": pl.Int32, "t": pl.String}
        lf = _scan("postgres17", driver, table, schema=schema, partition_on=None)
        if python:
            assert lf.collect().sort("id")["t"].to_list() == ["10:00:00", None]
            for dtype in [pl.String, pl.Time]:
                lf = _scan(
                    "postgres17",
                    driver,
                    table,
                    schema={"id": pl.Int32, "tz": dtype},
                    partition_on=None,
                )
                with pytest.raises(
                    pl.exceptions.InvalidOperationError, match=r"TIMETZ.*CAST.*AS TEXT"
                ):
                    lf.collect()
            # the advice works
            out = _scan(
                "postgres17",
                driver,
                f"SELECT id, CAST(tz AS TEXT) AS tz FROM {table}",
                schema={"id": pl.Int32, "tz": pl.String},
                partition_on=None,
            ).collect()
            assert out.sort("id")["tz"].to_list() == ["10:00:00+02", None]
        else:
            with pytest.raises(
                pl.exceptions.SchemaError, match=r"TIMETZ.*CAST.*AS TEXT"
            ):
                lf.collect()
        # (the table's own types read the same: the timetz as text, with its offset)
        out = _scan("postgres17", driver, table, partition_on=None).collect()
        assert out.sort("id")["tz"].to_list() == ["10:00:00+02", None]
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "connectorx", "psycopg", "pg8000"])
def test_postgres_time_24_is_an_error(driver: str) -> None:
    # Arrow drivers read it as NULL or 00:00:00, also in a time[] (ADBC)
    table = f"rvb_pg_time_{driver}"
    # (ConnectorX can't read a time[] at all)
    arrays = driver != "connectorx"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, t time, t3 time(3)"
            + (", ta time[])" if arrays else ")"),
            f"INSERT INTO {table} (id, t, t3) VALUES (1, '01:02:03.5', '01:02:03.125'), "
            "(2, NULL, NULL)",
        ],
    )
    try:
        for source in [table, f"SELECT * FROM {table}"]:
            lf = _scan("postgres17", driver, source)
            assert lf.collect_schema()["t3"] == pl.Time
            assert lf.collect().height == 2
        _execute("postgres17", [f"INSERT INTO {table} (id, t) VALUES (3, '24:00:00')"])
        if arrays:
            _execute(
                "postgres17",
                [
                    f"UPDATE {table} SET ta = '{{01:00:00,NULL}}' WHERE id = 1",
                    # in an array, of any dimension
                    f"INSERT INTO {table} (id, ta) VALUES "
                    "(4, '{{01:00:00},{24:00:00}}')",
                ],
            )
        for column in ["t", "ta"] if arrays else ["t"]:
            for source in [table, f"SELECT id, {column} FROM {table}"]:
                with pytest.raises(
                    pl.exceptions.PolarsError, match=f'column "{column}" holds 24:00:00'
                ):
                    _scan("postgres17", driver, source).select(column).collect()
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "connectorx", "psycopg", "pg8000"])
def test_postgres_query_types_read_as_a_tables(driver: str) -> None:
    # json was a Struct from a query on row drivers; numeric(5,2)[] failed on ADBC; a
    # query's numeric planned another dtype on each driver
    table = f"rvb_pg_qtypes_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, j json, jb jsonb, a numeric(5,2)[], "
            "n numeric(10,3), u numeric, i smallint)",
            f"""INSERT INTO {table} VALUES (1, '{{"a": 0.1000000000000000055, "a": 2}}', """
            """'[1, "x"]', '{1.5,NULL}', 1.5, 2.25, 3), (2, 'null', '"s"', NULL, NULL, NULL, NULL)""",
        ],
    )
    try:
        df = _scan("postgres17", driver, table).collect().sort("id")
        assert df.schema["a"] == pl.List(pl.Decimal(5, 2))
        assert df["j"].to_list() == ['{"a": 0.1000000000000000055, "a": 2}', "null"]
        assert df["jb"].to_list() == ['[1, "x"]', '"s"']
        for source in [
            f"SELECT * FROM {table}",
            f"SELECT * FROM {table} WHERE id > 10",
        ]:
            lf = _scan("postgres17", driver, source)
            got = lf.collect().sort("id")
            assert got.schema == lf.collect_schema()
            # a query's numeric shows no precision (psycopg's description has it)
            n = pl.Decimal(10, 3) if driver == "psycopg" else pl.Decimal(38, 9)
            assert got.schema["n"] == n
            assert got.schema["u"] == pl.Decimal(38, 9)
            assert got.schema["i"] == pl.Int16
            assert got.schema["j"] == pl.String
            if got.height:
                assert got["j"].to_list() == df["j"].to_list()
                assert got["u"].to_list() == [Decimal("2.25"), None]
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["adbc", "pymysql", "mysql.connector"])
def test_mysql_query_dtypes_are_the_tables(driver: str) -> None:
    # row drivers read every integer as Int64, FLOAT as a Float64, TIME as a duration
    # and TIMESTAMP naive; ADBC read a query's YEAR and BIT as text and binary
    table = f"rvb_my_qtypes_{driver.replace('.', '_')}"
    _execute(
        "mysql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, t tinyint, tu tinyint unsigned, sm smallint, "
            "f float, tm time(3), ts timestamp(6) NULL, y year, b bit(10))",
            f"INSERT INTO {table} VALUES (1, -5, 200, 7, 1.5, '12:34:56.789', "
            "'2024-01-02 03:04:05', 2024, b'1010101010'), "
            "(2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
        ],
    )
    try:
        tables = _scan("mysql", driver, table)
        # a row driver's query has no types: BIT's text form needs the width
        bit = [] if driver == "adbc" else ["b"]
        expected = tables.collect().drop(bit).schema
        for source in [
            f"SELECT * FROM {table}",
            f"SELECT * FROM {table} WHERE id > 10",
        ]:
            lf = _scan("mysql", driver, source)
            df = lf.collect()
            assert lf.collect_schema() == df.schema
            assert df.drop(bit).schema == expected
        df = _scan("mysql", driver, f"SELECT * FROM {table}").collect().sort("id")
        assert_frame_equal(df.drop(bit), tables.collect().sort("id").drop(bit))
    finally:
        _execute("mysql", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_query_with_side_effects_runs_once(name: str) -> None:
    # MySQL evaluates a derived table of one row (`SELECT f()`) before the schema
    # probe's `WHERE 1 = 0` applies: a function that writes ran twice, a sequence
    # advanced twice. Sessions that aren't read-only (ConnectorX, live) ran them.
    pytest.importorskip("connectorx")
    table = "rvf_once_log"
    _execute(name, [f"DROP TABLE IF EXISTS {table}"])
    try:
        _execute(
            name,
            [
                "DROP FUNCTION IF EXISTS rvf_once_bump",
                f"CREATE TABLE {table} (id int AUTO_INCREMENT PRIMARY KEY)",
                "CREATE FUNCTION rvf_once_bump() RETURNS INT MODIFIES SQL DATA "
                f"DETERMINISTIC BEGIN INSERT INTO {table} VALUES (NULL); "
                "RETURN LAST_INSERT_ID(); END",
            ],
        )
    except Exception as e:
        # MySQL with binary logging needs SUPER for a function that writes
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])
        if "SUPER" not in str(e):
            raise
        # (without one: a query that takes a second runs once, given its schema)
        for driver in ["connectorx", "live"]:
            start = time.monotonic()
            _scan(
                name, driver, "SELECT SLEEP(1) AS x", schema={"x": pl.Int64}
            ).collect()
            assert time.monotonic() - start < 1.9, driver
        return
    server = SERVERS[name]
    assert server.uri is not None
    assert server.factory is not None
    live = server.factory(server.uri)()
    live.autocommit(True)
    try:
        cur = live.cursor()

        def calls() -> int:
            cur.execute(f"SELECT count(*) FROM {table}")
            return int(cur.fetchone()[0])

        expected = 0
        cases: list[tuple[Any, dict[str, Any]]] = [
            (server.uri, {"driver": "connectorx"}),
            (live, {}),
        ]
        for source, kw in cases:
            # MySQL may evaluate the function to find the schema: inside a CTE, and on
            # MariaDB in a constant WHERE, a HAVING or a UNION branch, even under the
            # query's own LIMIT 0. Refused without a schema where sessions can write.
            flat = [
                "SELECT rvf_once_bump() AS x",
                "SELECT rvf_once_bump() AS x UNION ALL SELECT 0",
                "SELECT 1 AS x FROM DUAL WHERE rvf_once_bump() > 0",
                "(SELECT rvf_once_bump() AS x LIMIT 1)",
                "SELECT rvf_once_bump() AS x LIMIT 1",
            ]
            nested = [
                "WITH c AS (SELECT rvf_once_bump() AS x) SELECT x FROM c",
                "SELECT * FROM (SELECT rvf_once_bump() AS x) d",
            ]
            for query in flat + nested:
                if name == "mysql" and query in flat:
                    # MySQL doesn't run them under the query's own LIMIT 0
                    pl.scan_database(query, source, **kw).collect()
                    expected += 1
                    assert calls() == expected, query
                    continue
                with pytest.raises(
                    pl.exceptions.InvalidOperationError, match="schema="
                ):
                    pl.scan_database(query, source, **kw).collect()
                assert calls() == expected
                for pushdown in [True, False]:
                    pl.scan_database(
                        query, source, schema={"x": pl.Int64}, pushdown=pushdown, **kw
                    ).collect()
                    expected += 1
                    assert calls() == expected, (query, pushdown)
        if name == "mariadb":
            cur.execute("DROP SEQUENCE IF EXISTS rvf_once_seq")
            cur.execute("CREATE SEQUENCE rvf_once_seq START WITH 1 NOCACHE")
            cases2: list[tuple[Any, dict[str, Any]]] = [
                (server.uri, {"driver": "connectorx"}),
                (live, {}),
            ]
            for source, kw in cases2:
                v = pl.scan_database(
                    "SELECT NEXTVAL(rvf_once_seq) AS v",
                    source,
                    schema={"v": pl.Int64},
                    **kw,
                ).collect()
                cur.execute("SELECT next_not_cached_value FROM rvf_once_seq")
                assert cur.fetchone()[0] == v.item() + 1
    finally:
        live.close()
        _execute(
            name,
            [
                f"DROP TABLE IF EXISTS {table}",
                "DROP FUNCTION IF EXISTS rvf_once_bump",
                "DROP SEQUENCE IF EXISTS rvf_once_seq" if name == "mariadb" else "DO 0",
            ],
        )


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "pg8000", "sqlalchemy"])
def test_postgres_failed_lookup_keeps_the_users_transaction(module: str) -> None:
    # A failed statement in the user's transaction (the lookup of a name that isn't
    # there, a query that fails) aborted it: their writes were lost, and the next
    # lookup failed with InFailedSqlTransaction, hiding the error
    pytest.importorskip(module)
    assert POSTGRES is not None
    table = f"rvf_tx_{module}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f'DROP TABLE IF EXISTS "Rvf{module}"',
            f"CREATE TABLE {table} (k int)",
            f'CREATE TABLE "Rvf{module}" (id int)',
            f'INSERT INTO "Rvf{module}" VALUES (1)',
        ],
    )
    if module == "sqlalchemy":
        import sqlalchemy

        engine = sqlalchemy.create_engine(_sqlalchemy_url(POSTGRES, "psycopg"))
        conn: Any = engine.connect()

        def run(sql: str) -> Any:
            result = conn.exec_driver_sql(sql)
            return (
                [tuple(r) for r in result.fetchall()] if result.returns_rows else None
            )
    else:
        import importlib

        engine = None
        if module == "pg8000":
            p = _parts(POSTGRES)
            conn = importlib.import_module("pg8000").connect(**p)
        else:
            conn = importlib.import_module(module).connect(POSTGRES)

        def run(sql: str) -> Any:
            cur = conn.cursor()
            try:
                cur.execute(sql)
                return [tuple(r) for r in cur.fetchall()] if cur.description else None
            finally:
                cur.close()

    try:
        run(f"INSERT INTO {table} VALUES (1)")
        # found by the name as written, after the folded name's lookup failed
        assert pl.scan_database(f"Rvf{module}", conn).collect().height == 1
        with pytest.raises(pl.exceptions.ComputeError, match="does not exist"):
            pl.scan_database("rvf_no_such_table", conn).collect()
        with pytest.raises(pl.exceptions.ComputeError, match="division by zero"):
            pl.scan_database("SELECT 1 / 0 AS x", conn).collect()
        # the transaction goes on, with the user's write
        run(f"INSERT INTO {table} VALUES (2)")
        assert run(f"SELECT count(*) FROM {table}") == [(2,)]
        conn.commit()
    finally:
        conn.close()
        if engine is not None:
            engine.dispose()
        _execute(
            "postgres17",
            [f"DROP TABLE IF EXISTS {table}", f'DROP TABLE IF EXISTS "Rvf{module}"'],
        )


def test_converted_columns_order_by_their_values() -> None:
    # ORDER BY "n" sorted the select item of that name, read as text: 10, 100, -20, ...
    values = ["10", "100", "-20", "2.5", "-5", "9"]
    expected = sorted(values, key=Decimal)
    cases = []
    if POSTGRES is not None:
        cases += [
            ("postgres17", "rvf_ord", d, f"CREATE TABLE rvf_ord (id int, n {t})")
            for d in ["psycopg", "adbc", "pg8000"]
            for t in ["numeric"]
        ]
    if SERVERS["mysql"].uri is not None:
        cases.append(
            (
                "mysql",
                "rvf_ord",
                "connectorx",
                "CREATE TABLE rvf_ord (id int, n decimal(30,15))",
            )
        )
    if SERVERS["clickhouse"].uri is not None:
        cases.append(
            (
                "clickhouse",
                "rvf_ord",
                "connectorx",
                "CREATE TABLE rvf_ord (id Int32, n Decimal(30, 15)) "
                "ENGINE = MergeTree ORDER BY id",
            )
        )
    # (ConnectorX reads no bounds of a ClickHouse decimal)
    partitioned = {"partition_on": "n", "partition_num": 3}
    if not cases:
        pytest.skip("no Postgres, MySQL or ClickHouse URI")
    for name, table, driver, create in cases:
        if driver == "connectorx":
            pytest.importorskip("connectorx")
        _execute(
            name,
            [
                f"DROP TABLE IF EXISTS {table}",
                create,
                f"INSERT INTO {table} VALUES "
                + ", ".join(f"({i}, {v})" for i, v in enumerate(values)),
            ],
        )
        try:
            for source in [table, f"SELECT id, n FROM {table}"]:
                for kw in [{}, {} if name == "clickhouse" else partitioned]:
                    lf = _scan(name, driver, source, order_by="n", **kw)
                    got = [Decimal(str(v)) for v in lf.collect()["n"].to_list()]
                    assert got == [Decimal(v) for v in expected], (name, driver, source)
        finally:
            _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_numbers_order_and_timestamps_in_another_calendar() -> None:
    # ORDER BY "N" sorted TO_CHAR(N); TIMESTAMP(9) WITH LOCAL TIME ZONE was cut to
    # microseconds; TO_CHAR followed a live session's NLS_CALENDAR (543 years later)
    from datetime import datetime

    table = "rvf_ora_cal"
    _execute(
        "oracle",
        [
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            f'CREATE TABLE {table} ("id" NUMBER(5), n NUMBER, ts TIMESTAMP(9), '
            "ltz TIMESTAMP(9) WITH LOCAL TIME ZONE)",
        ]
        + [
            f"INSERT INTO {table} VALUES ({i}, {n}, "
            "TIMESTAMP '2024-01-02 03:04:05.123456789', "
            "TIMESTAMP '2024-01-02 03:04:05.123456789 +00:00')"
            for i, n in enumerate(["10", "100", "-20", "2.5", "-5", "9"])
        ],
    )
    server = SERVERS["oracle"]
    assert server.uri is not None
    assert server.factory is not None
    live = server.factory(server.uri)()
    try:
        cur = live.cursor()
        cur.execute("ALTER SESSION SET NLS_CALENDAR = 'Thai Buddha'")
        cur.execute("ALTER SESSION SET TIME_ZONE = '+05:00'")
        for source in [table.upper(), f'SELECT "id", n, ts, ltz FROM {table}']:
            for conn in [live, server.factory(server.uri)]:
                kws: list[dict[str, Any]] = [
                    {},
                    {"partition_on": "N", "partition_num": 3},
                ]
                for kw in kws:
                    lf = pl.scan_database(source, conn, order_by="N", **kw)
                    df = lf.collect()
                    assert df["N"].cast(pl.Float64).to_list() == [
                        -20.0,
                        -5.0,
                        2.5,
                        9.0,
                        10.0,
                        100.0,
                    ]
                    assert df["TS"][0] == datetime(2024, 1, 2, 3, 4, 5, 123456)
                    assert df["TS"].dt.nanosecond()[0] == 123456789
                    assert df.schema["LTZ"] == pl.Datetime("ns", "UTC")
                    assert df["LTZ"].dt.nanosecond()[0] == 123456789
                    assert df["LTZ"].dt.replace_time_zone(None)[0] == datetime(
                        2024, 1, 2, 3, 4, 5, 123456
                    )
    finally:
        live.close()
        _execute(
            "oracle",
            [
                f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; EXCEPTION WHEN OTHERS THEN NULL; END;"
            ],
        )


@needs_clickhouse
def test_clickhouse_arrays_of_text_and_nanoseconds() -> None:
    # an array element that isn't UTF-8 made ConnectorX return 1 row of 5, and
    # clickhouse_connect hex-encode it; clickhouse_connect cut Array(DateTime64(9)) to
    # microseconds. A real column named like the UTF-8 checks collided with them.
    from datetime import datetime

    table = "rvf_ch_arrays"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, a Array(String), "
            "t Array(DateTime64(9, 'UTC')), s String, __pl_utf8_0 String) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} SELECT number, [toString(number)], "
            "[toDateTime64('2200-01-01 00:00:00.123456789', 9, 'UTC')], 'x', "
            "toString(number) FROM numbers(5)",
        ],
    )
    try:
        frames = []
        for driver in ["adbc", "clickhouse_connect", "connectorx"]:
            for source in [table, f"SELECT * FROM {table}"]:
                lf = _scan("clickhouse", driver, source)
                if driver == "connectorx":
                    lf = lf.drop("t")  # ConnectorX can't read it: an error
                df = lf.filter(pl.col("__pl_utf8_0") != "3").collect().sort("id")
                assert df["id"].to_list() == [0, 1, 2, 4]
                frames.append(df.drop("t", strict=False))
                if driver != "connectorx":
                    assert df.schema["t"] == pl.List(pl.Datetime("ns", "UTC"))
                    t = df["t"].explode()
                    assert t.dt.nanosecond().to_list() == [123456789] * 4
                    assert t[0].replace(tzinfo=None) == datetime(
                        2200, 1, 1, 0, 0, 0, 123456
                    )
        for df in frames:
            assert_frame_equal(df, frames[0])
        _execute(
            "clickhouse",
            [f"INSERT INTO {table} VALUES (5, ['ok', unhex('ff')], [], 'x', 'y')"],
        )
        for driver in ["connectorx", "clickhouse_connect"]:
            for source in [table, f"SELECT id, a FROM {table}"]:
                lf = _scan("clickhouse", driver, source).select("id", "a")
                with pytest.raises(
                    pl.exceptions.ComputeError, match=r"not valid UTF-8|500 Internal"
                ):
                    lf.collect()
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_time_is_a_duration(name: str) -> None:
    # TIME is a duration (-838:59:59 to 838:59:59): a query's TIME was planned as a
    # time of day, which TIMEDIFF, SEC_TO_TIME or '25:30:00' failed
    from datetime import timedelta

    table = "rvf_my_time"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, t time(6), t0 time)",
            f"INSERT INTO {table} VALUES (1, '12:00:00.5', '12:00:00'), "
            "(2, '25:30:00', '838:59:59'), (3, '-01:00:00.25', '-838:59:59'), "
            "(4, NULL, NULL)",
        ],
    )
    expected = [
        timedelta(hours=12, microseconds=500000),
        timedelta(hours=25, minutes=30),
        -timedelta(hours=1, microseconds=250000),
        None,
    ]
    drivers = ["pymysql", "adbc", "connectorx", "mysql.connector"]
    try:
        for driver in drivers if name == "mysql" else ["pymysql", "connectorx"]:
            if driver == "connectorx":
                pytest.importorskip("connectorx")
            sources = [table]
            # ConnectorX has no query types: it reads a TIME as a time of day
            if driver != "connectorx":
                sources.append(f"SELECT * FROM {table}")
            for source in sources:
                lf = _scan(name, driver, source)
                df = lf.collect().sort("id")
                assert lf.collect_schema()["t"] == pl.Duration("us")
                assert df["t"].to_list() == expected, (driver, source)
                assert df["t0"][1] == timedelta(hours=838, minutes=59, seconds=59)
                check_filter_ids(
                    lf,
                    [
                        pl.col("t") > timedelta(hours=1),
                        pl.col("t") < timedelta(0),
                        pl.col("t").is_null(),
                    ],
                )
            if driver != "connectorx":
                q = (
                    "SELECT 1 AS id, TIMEDIFF(TIMESTAMP '2024-01-01 00:00:00', "
                    "TIMESTAMP '2024-01-01 01:00:00') AS td, "
                    "SEC_TO_TIME(-5) AS st, MAKETIME(-2, 0, 0) AS mt"
                )
                df = _scan(name, driver, q).collect()
                assert df.row(0) == (
                    1,
                    -timedelta(hours=1),
                    -timedelta(seconds=5),
                    -timedelta(hours=2),
                ), driver
    finally:
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg2", "psycopg", "pg8000", "connectorx"])
def test_postgres_infinity_is_an_error(driver: str) -> None:
    # psycopg2 read ±infinity as date.max / date.min (ADBC checks them itself)
    table = f"rvc_pg_inf_{driver}"
    arrays = driver != "connectorx"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, d date, ts timestamp, tz timestamptz"
            + (", da date[])" if arrays else ")"),
            f"INSERT INTO {table} (id, d, ts, tz) VALUES "
            "(1, '9999-12-31', '2024-01-01', '2024-01-01 00:00:00+00'), (2, NULL, NULL, NULL)",
        ],
    )
    try:
        for source in [table, f"SELECT * FROM {table}"]:
            assert _scan("postgres17", driver, source).collect().height == 2
        columns = {"d": "'-infinity'", "ts": "'infinity'", "tz": "'-infinity'"}
        if arrays:
            columns["da"] = "'{2024-01-01,infinity}'"
        for i, (column, value) in enumerate(columns.items(), start=3):
            _execute(
                "postgres17",
                [f"INSERT INTO {table} (id, {column}) VALUES ({i}, {value})"],
            )
        for column in columns:
            for source in [table, f"SELECT id, {column} FROM {table}"]:
                with pytest.raises(
                    pl.exceptions.PolarsError, match=f'column "{column}" holds infinity'
                ):
                    _scan("postgres17", driver, source).select(column).collect()
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
@pytest.mark.parametrize("how", ["live", "engine", "sa_connection", "sa_uri"])
def test_postgres_infinity_is_an_error_on_psycopg2_connections_of_the_user(
    how: str,
) -> None:
    # psycopg2 read `-infinity` as 0001-01-01 and `infinity` as 9999-12-31 on a
    # connection the plugin didn't open too (also behind SQLAlchemy)
    psycopg2 = pytest.importorskip("psycopg2")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    table = f"r42_pg_inf_{how}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, d date, ts timestamp, tz timestamptz, "
            "da date[])",
            f"INSERT INTO {table} VALUES (1, '2024-01-01', '2024-01-01', "
            "'2024-01-01 00:00:00+00', '{2024-01-01}'), (2, '-infinity', NULL, NULL, NULL), "
            "(3, NULL, 'infinity', NULL, NULL), (4, NULL, NULL, '-infinity', NULL), "
            "(5, NULL, NULL, NULL, '{2024-01-01,infinity}')",
        ],
    )
    url = _sqlalchemy_url(POSTGRES, "psycopg2")
    engine = None
    conn: Any = None
    if how == "live":
        conn = source = psycopg2.connect(POSTGRES)
    elif how == "engine":
        source = engine = sqlalchemy.create_engine(url)
    elif how == "sa_connection":
        engine = sqlalchemy.create_engine(url)
        conn = source = engine.connect()
    else:
        source = url
    try:
        # (`schema=`: nothing looks the columns' types up, as a typed read would)
        types = {
            "d": pl.Date,
            "ts": pl.Datetime("us"),
            "tz": pl.Datetime("us", "UTC"),
            "da": pl.List(pl.Date),
        }
        for column, dtype in types.items():
            for query in [table, f"SELECT id, {column} FROM {table}"]:
                schemas: list[Any] = [None, {"id": pl.Int32, column: dtype}]
                for schema in schemas:
                    lf = pl.scan_database(
                        query, source, partition_on=None, schema=schema
                    )
                    if schema is None or query != table:
                        lf = lf.select("id", column)
                    with pytest.raises(pl.exceptions.PolarsError, match="infinity"):
                        lf.collect()
        # the user's own statements keep the driver's reading of them
        if how == "live":
            cur = conn.cursor()
            cur.execute(f"SELECT d FROM {table} WHERE id = 2")
            assert cur.fetchone()[0].year == 1
            conn.rollback()
        # rows without one are read
        lf = pl.scan_database(
            f"SELECT * FROM {table} WHERE id = 1", source, partition_on=None
        )
        assert lf.collect()["d"].to_list() == [date(2024, 1, 1)]
    finally:
        if conn is not None:
            conn.close()
        if engine is not None:
            engine.dispose()
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
def test_postgres_multidimensional_array_is_an_error_on_adbc() -> None:
    # ADBC flattened {{1,2},{3,4}} to [1, 2, 3, 4]
    table = "rvc_pg_ndims"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, a int[])",
            f"INSERT INTO {table} VALUES (1, '{{1,2}}'), (2, '[0:1]={{5,6}}'), (3, NULL)",
        ],
    )
    try:
        for source in [table, f"SELECT * FROM {table}"]:
            df = _scan("postgres17", "adbc", source).collect().sort("id")
            assert df["a"].to_list() == [[1, 2], [5, 6], None]
        _execute(
            "postgres17", [f"INSERT INTO {table} VALUES (4, '{{{{1,2}},{{3,4}}}}')"]
        )
        for source in [table, f"SELECT * FROM {table}"]:
            with pytest.raises(
                pl.exceptions.PolarsError, match='column "a" holds a multi-dimensional'
            ):
                _scan("postgres17", "adbc", source).collect()
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_nanoseconds_before_1900() -> None:
    # read through their text, they were clamped to 1900-01-01, or a second late
    table = "rvc_ch_ns_1900"
    stored = [-2208988800123456789, -3000000000250000000, 1]
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, t DateTime64(9, 'UTC')) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} SELECT number, fromUnixTimestamp64Nano(toInt64(v), 'UTC') "
            f"FROM (SELECT arrayJoin(arrayEnumerate({stored})) - 1 AS number, "
            f"{stored}[number + 1] AS v)",
        ],
    )
    try:
        for driver in ["adbc", "clickhouse_connect", "connectorx"]:
            for source in [table, f"SELECT * FROM {table}"]:
                df = _scan("clickhouse", driver, source).collect().sort("id")
                assert df["t"].dt.epoch("ns").to_list() == stored, (driver, source)
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_float_and_negative_scale_numbers() -> None:
    # NUMBER(p, -s) failed; FLOAT (a decimal NUMBER) is read as the nearest Float64
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE rvc_or_num'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            "CREATE TABLE rvc_or_num (id NUMBER(9), f FLOAT, neg NUMBER(10, -2), "
            "wide NUMBER(30, -10))",
            "INSERT INTO rvc_or_num VALUES (1, 9007199254740993, 123456789012, "
            "1234567890123456789012345678900000000000)",
            "INSERT INTO rvc_or_num VALUES (2, 0.5, NULL, NULL)",
        ],
    )
    try:
        # (a query's columns are probed as the driver reads them: NUMBER(p, -s) fails)
        for source in ["RVC_OR_NUM", "SELECT ID, F FROM rvc_or_num"]:
            lf = _scan("oracle", "oracledb", source)
            assert lf.select("ID").collect().height == 2
            df = lf.collect().sort("ID")
            assert df["F"].to_list() == [9007199254740992.0, 0.5]
            if source == "RVC_OR_NUM":
                assert df["NEG"].to_list() == [123456789000, None]
                assert df["WIDE"].to_list() == [
                    "1234567890123456789012345678900000000000",
                    None,
                ]
    finally:
        _execute("oracle", ["DROP TABLE rvc_or_num"])


@needs_postgres
def test_postgres_query_with_a_writing_function_runs_once() -> None:
    # a count (tail) and partition bounds ran it twice where sessions aren't read-only
    pytest.importorskip("connectorx")
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS rvc_pg_log",
            "DROP TABLE IF EXISTS rvc_pg_src",
            "CREATE TABLE rvc_pg_log (n int)",
            "CREATE TABLE rvc_pg_src AS SELECT g AS id FROM generate_series(1, 5) g",
            "CREATE OR REPLACE FUNCTION rvc_pg_logged(i int) RETURNS int LANGUAGE sql "
            "VOLATILE AS $$ INSERT INTO rvc_pg_log VALUES (i); SELECT i $$",
        ],
    )
    psycopg = pytest.importorskip("psycopg")

    assert POSTGRES is not None
    query = "SELECT id, rvc_pg_logged(id) AS f FROM rvc_pg_src"
    try:
        with psycopg.connect(POSTGRES, autocommit=True) as conn:

            def logged() -> int:
                n = conn.execute("SELECT count(*) FROM rvc_pg_log").fetchone()
                conn.execute("DELETE FROM rvc_pg_log")
                return int(n[0])  # type: ignore[index]

            assert (
                _scan("postgres17", "connectorx", query).tail(2).collect().height == 2
            )
            assert logged() == 5
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                _scan(
                    "postgres17",
                    "connectorx",
                    query,
                    partition_on="id",
                    partition_num=2,
                ).collect()
            assert logged() == 0
    finally:
        _execute(
            "postgres17",
            [
                "DROP TABLE IF EXISTS rvc_pg_log",
                "DROP TABLE IF EXISTS rvc_pg_src",
                "DROP FUNCTION IF EXISTS rvc_pg_logged(int)",
            ],
        )


@needs_postgres
def test_postgres_like_on_a_nondeterministic_collation() -> None:
    # Postgres rejects LIKE there: the whole scan failed
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS rvc_pg_nd",
            "DROP COLLATION IF EXISTS rvc_pg_ci",
            "CREATE COLLATION rvc_pg_ci (provider = icu, locale = 'und-u-ks-level2', "
            "deterministic = false)",
            "CREATE TABLE rvc_pg_nd (id int, s text COLLATE rvc_pg_ci)",
            "INSERT INTO rvc_pg_nd VALUES (1, 'abc'), (2, 'ABD'), (3, 'xa')",
        ],
    )
    try:
        lf = _scan("postgres17", "adbc", "rvc_pg_nd")
        for p in [
            pl.col("s").str.starts_with("a"),
            pl.col("s").str.ends_with("a"),
            pl.col("s").str.contains("b", literal=True),
        ]:
            assert lf.filter(p).collect().sort("id")["id"].to_list() == (
                lf.collect().filter(p).sort("id")["id"].to_list()
            )
    finally:
        _execute(
            "postgres17",
            ["DROP TABLE IF EXISTS rvc_pg_nd", "DROP COLLATION IF EXISTS rvc_pg_ci"],
        )


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
def test_mariadb_query_with_a_function_is_not_probed_on_writable_sessions() -> None:
    # MariaDB runs a constant WHERE, a HAVING or a UNION branch under LIMIT 0: the
    # schema probe ran the function, and the read again
    lf = _scan(
        "mariadb", "live", "SELECT id FROM edge WHERE rvc_unknown_fn() = 1"
    ).select("id")
    with pytest.raises(pl.exceptions.InvalidOperationError, match="schema="):
        lf.collect()


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "psycopg2"])
def test_sqlalchemy_autocommit_connection_after_a_statement(driver: str) -> None:
    # SQLAlchemy says it is in a transaction under AUTOCOMMIT: a SAVEPOINT failed
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip(driver)
    assert POSTGRES is not None
    engine = sa.create_engine(
        POSTGRES.replace("postgresql://", f"postgresql+{driver}://"),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
            assert pl.scan_database("SELECT 1 AS x", conn).collect()["x"].item() == 1
    finally:
        engine.dispose()


@needs_postgres
def test_live_adbc_connection_is_refused() -> None:
    # its autocommit is off: the scan left a transaction open, and a failed statement
    # aborted the user's
    dbapi = pytest.importorskip("adbc_driver_postgresql.dbapi")
    assert POSTGRES is not None
    with dbapi.connect(POSTGRES) as conn:
        with pytest.raises(pl.exceptions.ComputeError, match="live ADBC connection"):
            pl.scan_database("SELECT 1 AS x", conn).collect()
        # a function opening ADBC connections plans the database's dialect
        lf = pl.scan_database("edge", lambda: dbapi.connect(POSTGRES))
        assert lf.collect_schema()["id"].is_integer()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_factory_connection_returns_to_its_pool_as_it_was() -> None:
    # a pool that doesn't reset sessions got its connection back READ ONLY, in UTC
    pooling = pytest.importorskip("mysql.connector.pooling")
    mysql_uri = SERVERS["mysql"].uri
    assert mysql_uri is not None
    uri = urlsplit(mysql_uri)
    pool = pooling.MySQLConnectionPool(
        pool_name="rvc_pool",
        pool_size=1,
        pool_reset_session=False,
        host=uri.hostname,
        port=uri.port,
        user=uri.username,
        password=uri.password,
        database=uri.path.lstrip("/"),
    )
    conn = pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT @@SESSION.time_zone, @@SESSION.transaction_read_only")
    before = cursor.fetchall()
    cursor.close()
    conn.close()
    assert pl.scan_database("edge", pool.get_connection).collect().height > 0
    conn = pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT @@SESSION.time_zone, @@SESSION.transaction_read_only")
    assert cursor.fetchall() == before
    cursor.close()
    conn.close()


@needs_postgres
def test_engine_raw_connection_as_a_factory() -> None:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg")
    assert POSTGRES is not None
    engine = sa.create_engine(
        POSTGRES.replace("postgresql://", "postgresql+psycopg://")
    )
    try:
        assert pl.scan_database("edge", engine.raw_connection).collect().height > 0
        with engine.connect() as conn:
            assert conn.exec_driver_sql("SHOW transaction_read_only").scalar() == "off"
    finally:
        engine.dispose()


_EARLY_STOP_SOURCES = {
    "postgres17": "SELECT g AS id FROM generate_series(1, 5000000) g",
    "mysql": "SELECT a.id + b.id * 1000 AS id FROM edge a CROSS JOIN edge b CROSS JOIN edge c",
}


@pytest.mark.parametrize(
    ("server", "driver"),
    [
        ("postgres17", "psycopg2"),
        ("postgres17", "sqlalchemy"),
        ("mysql", "adbc"),
        ("mysql", "pymysql"),
    ],
)
@pytest.mark.parametrize("stop", ["head", "collect_batches"])
def test_stopped_early_exits_cleanly(server: str, driver: str, stop: str) -> None:
    # The cleanup of a stopped read ran on a thread of its own after `collect()`
    # returned, and a dropped `collect_batches()` iterator's query runs on until it
    # notices: their driver calls during Python's finalization aborted the process on
    # Python <= 3.13 (psycopg2 4 runs in 5, SQLAlchemy over psycopg2 and MySQL over ADBC
    # 5 in 5).
    uri = SERVERS[server].uri
    if uri is None:
        pytest.skip(f"no {server} URI")
    if driver == "adbc":
        import polars_database

        if polars_database.find_adbc_driver(server) is None:
            pytest.skip(f"no ADBC driver for {server}")
    elif driver == "sqlalchemy":
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("psycopg2")
        uri = uri.replace("postgresql://", "postgresql+psycopg2://")
    else:
        pytest.importorskip(driver)
    read = {
        "head": "assert lf.head(3).collect().height == 3",
        "collect_batches": "for df in lf.collect_batches():\n    break",
    }[stop]
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_database({_EARLY_STOP_SOURCES[server]!r}, {uri!r}, "
        f"driver={driver!r}, batch_size=1000)\n"
    ) + read
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr


@needs_postgres
def test_engine_partitions_are_cancelled_on_an_early_stop() -> None:
    # an Engine's connections had no cancel: head() waited for the slowest partition
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg")
    # A plain table: a slow function in a view or key isn't proven repeatable, so it
    # couldn't be partitioned. The last partition is slow because it holds most rows.
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS rvc_pg_slow",
            "CREATE UNLOGGED TABLE rvc_pg_slow (id int, k int, s text)",
            "INSERT INTO rvc_pg_slow SELECT g, 1, md5(g::text) "
            "FROM generate_series(1, 300) g",
            "INSERT INTO rvc_pg_slow SELECT 301 + g % 100, 1, md5(g::text) "
            "FROM generate_series(1, 12000000) g",
        ],
    )
    assert POSTGRES is not None
    engine = sa.create_engine(
        POSTGRES.replace("postgresql://", "postgresql+psycopg://"), pool_size=4
    )
    try:
        lf = pl.scan_database(
            "rvc_pg_slow",
            engine,
            partition_on="id",
            partition_range=(1, 400),
            partition_num=4,
        )
        start = time.perf_counter()
        # k matches in every partition: none is pruned, the first answers at once
        assert lf.filter(pl.col("k") == 1).head(1).collect().height == 1
        # the last partition alone takes several seconds
        assert time.perf_counter() - start < 3
        # The cancelled partitions gave their connections back after `collect()`
        # returned, from threads of their own: an Engine disposed then let them go
        # unclosed. They are back before it returns.
        assert engine.pool.checkedout() == 0
    finally:
        engine.dispose()
        _execute("postgres17", ["DROP TABLE IF EXISTS rvc_pg_slow"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_builtin_functions_and_executable_comments(name: str) -> None:
    # built-ins sqlglot doesn't know were refused on writable sessions; a trailing
    # executable comment was dropped
    pytest.importorskip("connectorx")
    for driver in ["connectorx", "live"]:
        lf = _scan(name, driver, "SELECT id, CRC32(CAST(id AS CHAR)) AS c FROM edge")
        assert lf.collect().height > 0
    lf = _scan(name, "pymysql", "SELECT id FROM edge /*! WHERE id = 1 */")
    assert lf.collect()["id"].to_list() == [1]


@needs_postgres
def test_sqlalchemy_orm_session_left_outside_a_transaction() -> None:
    # an idle ORM Session was left in the scan's transaction
    sa = pytest.importorskip("sqlalchemy")
    orm = pytest.importorskip("sqlalchemy.orm")
    pytest.importorskip("psycopg")
    assert POSTGRES is not None
    engine = sa.create_engine(
        POSTGRES.replace("postgresql://", "postgresql+psycopg://")
    )
    try:
        with orm.Session(engine) as session:
            assert pl.scan_database("SELECT 1 AS a", session).collect()["a"].item() == 1
            assert not session.in_transaction()
    finally:
        engine.dispose()


# Round 7 review findings.


def _connect(name: str) -> Any:
    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    assert server.factory is not None
    return server.factory(server.uri)()


@needs_clickhouse
def test_clickhouse_negated_filters_on_nullable_columns() -> None:
    # `NULL IN (…)` is 0 on ClickHouse, not NULL: `NOT (x IN …)` kept the NULL rows
    pytest.importorskip("clickhouse_connect")
    table = "r7_ch_not_in"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, x Nullable(Int32), "
            "s Nullable(String), b Nullable(Bool)) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, 1, 'ab', true), (2, 2, 'cd', false), "
            "(3, NULL, NULL, NULL)",
        ],
    )
    try:
        lf = _scan("clickhouse", "clickhouse_connect", table)
        check_filter_ids(
            lf,
            [
                ~pl.col("x").is_in([1]),
                ~pl.col("x").is_in([1, None]),
                ~pl.col("x").is_in([1], nulls_equal=True),
                ~(pl.col("x").is_in([1]) | (pl.col("id") == 3)),
                ~(pl.col("x").is_in([1]) & (pl.col("id") == 3)),
                ~pl.col("s").is_in(["ab"]),
                ~pl.col("s").str.starts_with("a"),
                ~pl.col("s").str.contains("b", literal=True),
                ~(pl.col("x") == 1),
                ~(pl.col("x") > 1),
                ~pl.col("x").is_between(0, 1),
                ~pl.col("b"),
                ~(pl.col("x").eq_missing(1)),
                ~(pl.col("s") == "ab"),
            ],
        )
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_backslashes_in_names() -> None:
    # the catalog and UTF-8 check queries didn't escape `\` in string literals, nor
    # the fixed select lists in quoted identifiers
    pytest.importorskip("clickhouse_connect")
    table = "r7_ch_bs"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, `x\\\\` Int32, `s\\\\` String, "
            "`t\\\\` DateTime64(3, 'UTC')) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, 10, 'a', 0), (2, 20, 'b', 1000)",
        ],
    )
    try:
        for source in [table, f"SELECT * FROM {table}"]:
            lf = _scan("clickhouse", "clickhouse_connect", source)
            assert lf.collect_schema().names() == ["id", "x\\", "s\\", "t\\"]
            df = lf.collect().sort("id")
            assert df["x\\"].to_list() == [10, 20]
            assert df["s\\"].to_list() == ["a", "b"]
            check_filter_ids(lf, [pl.col("x\\") > 10, pl.col("s\\") == "a"])
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


_FLOAT_DRIVERS = {"postgres17": "psycopg", "mysql": "pymysql", "mariadb": "pymysql"}


@pytest.mark.parametrize("name", list(_FLOAT_DRIVERS))
def test_float_literals_compare_exactly(name: str) -> None:
    # a float literal the database reads as a decimal must compare as the double itself
    import random

    rng = random.Random(145)
    values = [rng.random() for _ in range(200)]
    values += [rng.uniform(-1, 1) * 10.0 ** rng.randint(-40, 40) for _ in range(200)]
    table = "r7_floats"
    float_type = "double precision" if name == "postgres17" else "DOUBLE"
    conn = _connect(name)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} (id int, f {float_type})")
        cur.executemany(f"INSERT INTO {table} VALUES (%s, %s)", list(enumerate(values)))
        conn.commit()
    finally:
        conn.close()
    lf = _scan(name, _FLOAT_DRIVERS[name], table)
    assert_frame_equal(
        lf.collect().sort("id"),
        pl.DataFrame({"id": range(len(values)), "f": values}),
        check_dtypes=False,
    )
    got = lf.filter(pl.col("f").is_in(values)).collect()
    assert sorted(got["id"].to_list()) == list(range(len(values)))
    for i, v in list(enumerate(values))[::25]:
        assert lf.filter(pl.col("f") == v).collect()["id"].to_list() == [i], v


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "adbc"])
def test_postgres_user_float_schema_over_real_and_numeric(driver: str) -> None:
    # `schema=` Float64 over `real` / `numeric`: the database compares values Polars
    # doesn't read
    table = f"r7_pg_schema_float_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, r real, n numeric, d numeric(40, 30))",
            f"INSERT INTO {table} VALUES (1, 0.1, 0.10000000000000001, "
            "0.100000000000000000000000000001), (2, 0.5, 0.5, 0.5), (3, NULL, NULL, NULL)",
        ],
    )
    schema = {"id": pl.Int32, "r": pl.Float64, "n": pl.Float64, "d": pl.Float64}
    lf = _scan("postgres17", driver, table, schema=schema)
    check_filter_ids(
        lf,
        [
            pl.col("r") == 0.1,
            pl.col("r") != 0.1,
            pl.col("r") > 0.1,
            pl.col("r") <= 0.1,
            pl.col("n") == 0.1,
            pl.col("n") <= 0.1,
            pl.col("d") == 0.1,
            pl.col("d") > 0.1,
        ],
    )


_WIDE_DECIMAL = "123456789012345678901234.1234567891"


@pytest.mark.parametrize("name", ["postgres17", "mysql"])
def test_connectorx_wide_decimals_are_exact_or_an_error(name: str) -> None:
    # ConnectorX reads decimals through rust_decimal (28-29 digits): a wider value must
    # be read exactly or fail, never be rounded
    pytest.importorskip("connectorx")
    table = "r7_cx_wide"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, d numeric(38, 10))",
            f"INSERT INTO {table} VALUES (1, {_WIDE_DECIMAL}), (2, -{_WIDE_DECIMAL}), "
            "(3, 1.5)",
        ],
    )
    expected = [Decimal(_WIDE_DECIMAL), Decimal("-" + _WIDE_DECIMAL), Decimal("1.5")]
    for source in [table, f"SELECT id, d FROM {table}"]:
        lf = _scan(name, "connectorx", source)
        try:
            got = lf.collect().sort("id")["d"].to_list()
        except pl.exceptions.PolarsError:
            # refused rather than rounded: ConnectorX's rounding lowers the scale,
            # which its Arrow writer rejects ("decimal scale is not equal"), and the
            # plugin's cast of a query's decimals fails where it would round
            continue
        assert got == expected, (source, got)


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_like_under_a_collation_with_contractions() -> None:
    # Czech `ch` is one letter: `'chata' LIKE 'c%'` is false under XCZECH, so LIKE isn't
    # a superset of the byte-wise match there
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE r7_czech'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE r7_czech ("id" NUMBER(9), s VARCHAR2(10))',
            "INSERT INTO r7_czech VALUES (1, 'chata')",
            "INSERT INTO r7_czech VALUES (2, 'cesta')",
            "INSERT INTO r7_czech VALUES (3, 'hrad')",
            "INSERT INTO r7_czech VALUES (4, 'CHATA')",
        ],
    )
    assert SERVERS["oracle"].uri is not None
    conn = _oracle_factory(SERVERS["oracle"].uri)()
    try:
        conn.cursor().execute(
            "ALTER SESSION SET NLS_COMP = LINGUISTIC NLS_SORT = XCZECH"
        )
        lf = pl.scan_database("R7_CZECH", conn)
        check_filter_ids(
            lf,
            [
                pl.col("S").str.starts_with("c"),
                pl.col("S").str.ends_with("h"),
                pl.col("S").str.contains("c", literal=True),
                pl.col("S").str.starts_with("h"),
                pl.col("S") == "chata",
                pl.col("S").is_in(["cesta"]),
            ],
        )
    finally:
        conn.close()


@needs_clickhouse
def test_clickhouse_naive_datetimes_read_alike_by_every_driver() -> None:
    # a naive DateTime is stored as an instant and shown in the server's time zone;
    # every driver must read the same values (checked by hand on a server in Asia/Tokyo)
    pytest.importorskip("clickhouse_connect")
    table = "r7_ch_naive"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, dt DateTime, dt64 DateTime64(3)) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, '2024-01-02 03:04:05', "
            "'2024-01-02 03:04:05.123')",
        ],
    )
    try:
        frames = []
        for driver in ["clickhouse_connect", "connectorx", "adbc"]:
            module = DRIVER_MODULES.get(driver, driver)
            if module is not None and not _installed(module):
                continue
            for source in [table, f"SELECT * FROM {table}"]:
                lf = _scan("clickhouse", driver, source)
                frames.append((lf.collect_schema(), lf.collect()))
        (first_schema, first), *rest = frames
        assert first_schema["dt"] == pl.Datetime("ms")
        for schema, df in rest:
            assert schema == first_schema
            assert_frame_equal(df, first)
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


class _PooledConnection:
    """A DB-API connection of a user's pool: `close()` hands it back to the pool."""

    def __init__(self, cnx: Any, pool: list[Any]) -> None:
        self._cnx = cnx
        self._pool = pool

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cnx, name)

    def close(self) -> None:
        self._pool.append(self)


def _session_state(conn: Any) -> Any:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT @@SESSION.time_zone, @@SESSION.transaction_read_only, "
        "@@SESSION.sql_mode"
    )
    rows = cursor.fetchall()
    cursor.close()
    return rows


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
@pytest.mark.parametrize("restore", ["works", "fails"])
def test_factory_connection_of_the_mariadb_module_is_restored_or_discarded(
    restore: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The session setup ran on a `mariadb` (or `cx_Oracle`) connection, but nothing
    # restored it: it went back to the user's pool READ ONLY, in UTC. A restore that
    # fails discards the connection instead.
    import importlib.machinery
    import sys
    import types

    pymysql = pytest.importorskip("pymysql")

    from polars_database import _bridge

    mod = types.ModuleType("mariadb")
    mod.__spec__ = importlib.machinery.ModuleSpec("mariadb", None)
    monkeypatch.setitem(sys.modules, "mariadb", mod)
    # a connection type of the `mariadb` module, around a PyMySQL connection
    wrapper = type(
        "Connection", (_PooledConnection,), {"__module__": "mariadb.connections"}
    )
    if restore == "fails":
        monkeypatch.setattr(
            _bridge, "_session_settings", lambda conn: ["SELECT no_such_function()"]
        )
    mariadb_uri = SERVERS["mariadb"].uri
    assert mariadb_uri is not None
    pool: list[Any] = []
    opened: list[Any] = []

    def get() -> Any:
        conn = pool.pop() if pool else None
        if conn is None or not conn._cnx.open:
            # a pool reconnects a connection that was closed
            opened.append(pymysql.connect(**_parts(mariadb_uri)))
            conn = wrapper(opened[-1], pool)
        return conn

    probe = pymysql.connect(**_parts(mariadb_uri))
    before = _session_state(probe)
    probe.close()
    lf = pl.scan_database("edge", get, partition_on=None)
    assert lf.collect().height > 0
    assert opened
    for cnx in opened:
        if restore == "works":
            assert _session_state(cnx) == before
            cnx.close()
        else:
            # closed for real, not handed back as the setup left it
            assert not cnx.open


@needs_postgres
def test_postgres_c_collation_in_another_server_encoding() -> None:
    # `COLLATE "C"` orders by the bytes of the server encoding: code point order only in
    # UTF-8. In WIN1251, '№' (0xB9, U+2116) sorts before U+0410 (0xC0).
    psycopg = pytest.importorskip("psycopg")

    assert POSTGRES is not None
    with psycopg.connect(POSTGRES, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS r7_win1251")
        conn.execute(
            "CREATE DATABASE r7_win1251 ENCODING 'WIN1251' LC_COLLATE 'C' "
            "LC_CTYPE 'C' TEMPLATE template0"
        )
    uri = POSTGRES.rsplit("/", 1)[0] + "/r7_win1251"
    try:
        with psycopg.connect(uri) as conn:
            conn.execute('CREATE TABLE t (id int, s text COLLATE "C")')
            conn.execute(
                "INSERT INTO t VALUES (1, 'А'), (2, '№'), (3, 'Я'), (4, 'a'), (5, NULL)"  # noqa: RUF001
            )
        for source in ["t", "SELECT id, s FROM t"]:
            lf = pl.scan_database(source, uri, driver="psycopg")
            check_filter_ids(
                lf,
                [
                    pl.col("s") > "А",  # noqa: RUF001
                    pl.col("s") < "Я",
                    pl.col("s") == "№",
                    pl.col("s").str.starts_with("№"),
                ],
            )
    finally:
        with psycopg.connect(POSTGRES, autocommit=True) as conn:
            conn.execute("DROP DATABASE IF EXISTS r7_win1251 WITH (FORCE)")


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
@pytest.mark.parametrize("connection", ["oracledb", "sqlalchemy", "factory", "live"])
def test_oracle_numbers_without_arrow_are_exact(
    connection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # oracledb < 3.4 has no Arrow fetch: NUMBER(p, s) came as floats, 1234567890123456
    # 789012345678.12 read as ...778777195970.56
    from polars.io.database._cursor_proxies import OracleCursorProxy

    monkeypatch.setattr(
        OracleCursorProxy, "supports_arrow", classmethod(lambda cls, conn: False)
    )
    # a table per connection: parametrizations may run at once on other workers
    table = f"r7_ora_num_{connection}"
    _execute(
        "oracle",
        [
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            f'CREATE TABLE {table} ("id" NUMBER(9), n NUMBER(30, 2), '
            "m NUMBER(12, 4), i NUMBER(20), f BINARY_DOUBLE)",
            f"INSERT INTO {table} VALUES "
            "(1, 1234567890123456789012345678.12, 12345678.1234, "
            "12345678901234567890, 0.5)",
            f"INSERT INTO {table} VALUES (2, -0.07, 0.0001, -1, NULL)",
        ],
    )
    lf = _scan("oracle", connection, table.upper())
    df = lf.collect().sort("id")
    assert df["N"].to_list() == [
        Decimal("1234567890123456789012345678.12"),
        Decimal("-0.07"),
    ]
    assert df["M"].to_list() == [Decimal("12345678.1234"), Decimal("0.0001")]
    assert df["I"].cast(pl.String).to_list() == ["12345678901234567890", "-1"]
    assert df["F"].to_list() == [0.5, None]


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_query_decimal_literals_finer_than_the_scale(name: str) -> None:
    # A query source has no result types: `DECIMAL(6, 2)` plans a wider scale, and
    # MySQL 8 folded `x < -0.005` to the column's scale with the wrong sign, keeping
    # 0.00
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS r9_dec",
            "CREATE TABLE r9_dec (id INT, x DECIMAL(6, 2))",
            "INSERT INTO r9_dec VALUES (1, 0.00), (2, -0.01), (3, 0.01), (4, NULL)",
        ],
    )
    lf = _scan(name, "pymysql", "SELECT id, x FROM r9_dec")
    check_filter_ids(
        lf,
        [
            pl.col("x") < Decimal("-0.005"),
            pl.col("x") <= Decimal("-0.005"),
            pl.col("x") > Decimal("-0.005"),
            pl.col("x") >= Decimal("-0.005"),
            pl.col("x").is_between(Decimal("-0.005"), Decimal("0.005")),
            pl.col("x") < -5e-3,
        ],
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "sqlalchemy"])
def test_postgres_query_bpchar_keeps_its_padding(driver: str) -> None:
    # `pg_typeof` writes `character(n)` as `bpchar`, which was read through a cast to
    # text that strips the padding; `char(n)[]` was a String
    table = f"r9_bpchar_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, ch char(5), cha char(4)[])",
            f"INSERT INTO {table} VALUES (1, 'ab', ARRAY['a', 'bb'])",
        ],
    )
    for source in [table, f"SELECT id, ch, cha FROM {table}"]:
        df = _scan("postgres17", driver, source).collect()
        assert df["ch"].to_list() == ["ab   "], source
        assert df["cha"].dtype == pl.List(pl.String), source
        assert df["cha"].to_list() == [["a   ", "bb  "]], source


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "pg8000", "psycopg2"])
def test_postgres_live_timestamptz_in_a_historical_zone(module: str) -> None:
    # psycopg returned the session zone's wall-clock time, its instant recomputed from
    # Python's tzdata: 1900 in Amsterdam read 00:19:32 UTC; pg8000 failed
    table = f"r9_tstz_{module}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, ts timestamptz)",
            f"INSERT INTO {table} VALUES (1, '1900-01-01 00:00:00+00'), "
            "(2, '1800-01-01 00:00:00+00'), (3, '2026-06-01 12:00:00+00')",
        ],
    )
    assert POSTGRES is not None
    conn = _live_postgres(module)
    try:
        cur = conn.cursor()
        cur.execute("SET TIME ZONE 'Europe/Amsterdam'")
        cur.close()
        df = pl.scan_database(table, conn).collect().sort("id")
    finally:
        conn.close()
    assert df["ts"].dtype == pl.Datetime("us", "UTC")
    assert df["ts"].dt.replace_time_zone(None).to_list() == [
        datetime(1900, 1, 1),
        datetime(1800, 1, 1),
        datetime(2026, 6, 1, 12),
    ]


def _live_postgres(module: str) -> Any:
    assert POSTGRES is not None
    if module == "psycopg":
        psycopg = pytest.importorskip("psycopg")

        return psycopg.connect(POSTGRES)
    if module == "psycopg2":
        psycopg2 = pytest.importorskip("psycopg2")

        return psycopg2.connect(POSTGRES)
    pg8000_dbapi = pytest.importorskip("pg8000.dbapi")

    u = urlsplit(POSTGRES)
    return pg8000_dbapi.connect(
        user=u.username,
        password=u.password,
        host=u.hostname,
        port=u.port or 5432,
        database=u.path.lstrip("/"),
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["connectorx", "psycopg", "psycopg2", "sqlalchemy"])
def test_postgres_uri_password_parameter_doesnt_override_the_provider(
    driver: str,
) -> None:
    assert POSTGRES is not None
    password = urlsplit(POSTGRES).password
    without_password = POSTGRES.replace(f":{password}@", "@", 1)
    lf = pl.scan_database(
        "orders",
        f"{without_password}?password=WRONG",
        driver=driver,
        credential_provider=lambda: {"password": password},
    )
    assert lf.select(pl.len()).collect().item() > 0


@needs_postgres
@pytest.mark.parametrize("scheme", ["pgsql", "neon", "cockroachdb"])
@pytest.mark.parametrize("driver", ["adbc", "connectorx", "psycopg", "sqlalchemy"])
def test_postgres_alias_schemes(scheme: str, driver: str) -> None:
    assert POSTGRES is not None
    uri = scheme + POSTGRES[POSTGRES.index(":") :]
    lf = pl.scan_database("orders", uri, driver=driver)
    assert lf.select(pl.len()).collect().item() > 0


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
@pytest.mark.parametrize("driver", [None, "pymysql", "connectorx"])
def test_mariadb_scheme(driver: str | None) -> None:
    # `mariadb://` (a SQLAlchemy URL) failed on the default driver
    uri = SERVERS["mariadb"].uri
    assert uri is not None
    uri = "mariadb" + uri[uri.index(":") :]
    lf = pl.scan_database("orders", uri, driver=driver)
    assert lf.select(pl.len()).collect().item() > 0


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
@pytest.mark.parametrize("connection", ["oracledb", "sqlalchemy"])
def test_oracle_number_with_scale_above_precision(connection: str) -> None:
    # Arrow refused the drivers' `decimal(5, 7)`, failing the whole scan
    table = f"r9_ora_ns_{connection}"
    _execute(
        "oracle",
        [
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table}'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            f'CREATE TABLE {table} ("id" NUMBER(9), ns NUMBER(5, 7))',
            f"INSERT INTO {table} VALUES (1, 0.0012345)",
            f"INSERT INTO {table} VALUES (2, -0.0000001)",
        ],
    )
    # (a query's such column still fails: its type is known only after it is read, #105)
    df = _scan("oracle", connection, table.upper()).collect().sort("id")
    assert df["NS"].dtype == pl.Decimal(7, 7)
    assert df["NS"].to_list() == [Decimal("0.0012345"), Decimal("-0.0000001")]
    assert df.drop("NS")["id"].to_list() == [1, 2]


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_mysql_decimal_partition_cuts_near_zero(server: str) -> None:
    # MySQL 8 misorders a float cut in (-0.01, 0) on DECIMAL(10,2): `0.00` was read
    # twice
    _execute(
        server,
        [
            "DROP TABLE IF EXISTS review_dec_cuts",
            "CREATE TABLE review_dec_cuts (x DECIMAL(10,2))",
            "INSERT INTO review_dec_cuts "
            "WITH RECURSIVE r(i) AS (SELECT -300 UNION ALL SELECT i + 1 FROM r "
            "WHERE i < 300) SELECT i / 100 FROM r",
        ],
    )
    for n in [600, 7]:
        lf = _scan(
            server, "pymysql", "review_dec_cuts", partition_on="x", partition_num=n
        )
        out = lf.collect()["x"]
        assert out.len() == 601, n
        assert out.n_unique() == 601, n


@needs_postgres
@pytest.mark.parametrize(
    "driver", ["adbc", "connectorx", "psycopg", "psycopg2", "pg8000"]
)
def test_postgres_arrays_of_types_read_as_text(driver: str) -> None:
    # the whole array was one string: `'{6b0b9f4e-…}'`
    table = f"review_text_arrays_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"DROP TYPE IF EXISTS {table}_mood",
            f"CREATE TYPE {table}_mood AS ENUM ('happy', 'sad')",
            f"CREATE TABLE {table} (id int PRIMARY KEY, u uuid[], j jsonb[], "
            f"m {table}_mood[], w numeric(50,2)[])",
            f"INSERT INTO {table} VALUES (1, "
            "ARRAY['a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11']::uuid[], "
            """ARRAY['{"a": 1}']::jsonb[], ARRAY['happy', 'sad']::"""
            f"{table}_mood[], ARRAY[1.50]::numeric(50,2)[]), (2, NULL, NULL, NULL, NULL)",
        ],
    )
    # (ConnectorX can't read a query's `uuid[]`, #105; a query's `numeric(50,2)[]` is
    # `numeric[]`, without its precision)
    sources = [table] if driver == "connectorx" else [table, f"SELECT * FROM {table}"]
    for source in sources:
        out = _scan("postgres17", driver, source).collect().sort("id")
        assert out.schema["u"] == pl.List(pl.String), source
        assert out.drop("w").rows() == [
            (
                1,
                ["a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"],
                ['{"a": 1}'],
                ["happy", "sad"],
            ),
            (2, None, None, None),
        ], source
    out = _scan("postgres17", driver, table).collect().sort("id")
    assert out["w"].to_list() == [["1.50"], None]


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
def test_clickhouse_array_of_decimals() -> None:
    # `Array(Decimal(10, 2))` was inferred as a List of a Decimal without precision,
    # which failed the whole table's discovery
    from decimal import Decimal

    _execute(
        "clickhouse",
        [
            "DROP TABLE IF EXISTS review_ch_array_decimal",
            "CREATE TABLE review_ch_array_decimal (id Int32, "
            "v Array(Decimal(10, 2)), ok Int32) ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_ch_array_decimal VALUES (1, [1.5, -2.25], 7)",
        ],
    )
    lf = _scan("clickhouse", "clickhouse_connect", "review_ch_array_decimal")
    assert lf.collect_schema()["v"] == pl.List(pl.Decimal(10, 2))
    assert lf.select("ok").collect()["ok"].to_list() == [7]
    assert lf.collect()["v"].to_list() == [[Decimal("1.50"), Decimal("-2.25")]]


@needs_clickhouse
def test_clickhouse_connectorx_low_cardinality_array_is_refused() -> None:
    # ConnectorX reads an `Array(LowCardinality(String))` as one string
    pytest.importorskip("connectorx")
    table = "review_ch_lc_array"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, a Array(LowCardinality(String))) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, ['x', 'y'])",
        ],
    )
    for source in [table, f"SELECT id, a FROM {table}"]:
        with pytest.raises(pl.exceptions.InvalidOperationError, match="column 'a'"):
            _scan("clickhouse", "connectorx", source).collect()
        out = _scan("clickhouse", "clickhouse_connect", source).collect()
        assert out["a"].to_list() == [["x", "y"]]


# Round 11 review findings.


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_mysql_decimal_read_as_float_partition_cuts(server: str) -> None:
    # a DECIMAL read as Float64 had float cuts again, which MySQL 8 misorders (#201)
    _execute(
        server,
        [
            "DROP TABLE IF EXISTS r11_dec_float_cuts",
            "CREATE TABLE r11_dec_float_cuts (x DECIMAL(5,2))",
            "INSERT INTO r11_dec_float_cuts "
            "WITH RECURSIVE r(i) AS (SELECT -300 UNION ALL SELECT i + 1 FROM r "
            "WHERE i < 300) SELECT i / 100 FROM r",
        ],
    )
    for dtype in [pl.Float64, pl.Float32]:
        for kwargs in [{"schema_overrides": {"x": dtype}}, {"schema": {"x": dtype}}]:
            lf = _scan(
                server,
                "pymysql",
                "r11_dec_float_cuts",
                partition_on="x",
                partition_num=600,
                **kwargs,
            )
            out = lf.collect()["x"]
            assert out.len() == 601, kwargs
            assert out.n_unique() == 601, kwargs


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_mysql_indexed_columns_round_literals(server: str) -> None:
    # an index lookup stores the literal into the column's type: a fractional datetime
    # literal on DATETIME(0), a DOUBLE(7,2) equality (MariaDB) and a YEAR beyond 2155
    # (MariaDB) matched other rows; `LIKE 'a%'` under a PAD SPACE binary collation
    # skipped `'a\tb'`
    _execute(
        server,
        [
            "DROP TABLE IF EXISTS r11_indexed",
            "CREATE TABLE r11_indexed (id int, d0 DATETIME(0), d3 DATETIME(3), "
            "ts0 TIMESTAMP(0) NULL, f DOUBLE(7,2), y YEAR, "
            "s VARCHAR(20) COLLATE utf8mb4_bin, "
            "KEY(d0), KEY(d3), KEY(ts0), KEY(f), KEY(y), KEY(s))",
            "INSERT INTO r11_indexed VALUES "
            "(1, '2020-01-01 00:00:00', '2020-01-01 00:00:00.500', "
            "'2020-01-01 00:00:00', 1.25, 2155, 'a'), "
            "(2, '2020-01-01 00:00:01', '2020-01-01 00:00:01', "
            "'2020-01-01 00:00:01', 1.5, 1901, 'a\\tb'), "
            "(3, NULL, NULL, NULL, NULL, 2000, 'ab'), "
            "(4, NULL, NULL, NULL, NULL, NULL, 'a\\nx'), "
            "(5, NULL, NULL, NULL, NULL, NULL, 'b')",
        ],
    )
    t = datetime(2020, 1, 1, 0, 0, 0)
    utc = t.replace(tzinfo=UTC)
    predicates = [
        pl.col("d0") == t + timedelta(microseconds=400_000),
        pl.col("d0") == t + timedelta(microseconds=500_000),
        pl.col("d0").is_in([t + timedelta(microseconds=500_000)]),
        pl.col("d0") == t + timedelta(seconds=1),
        pl.col("d0") < t + timedelta(microseconds=500_000),
        pl.col("d3") == t + timedelta(microseconds=500_400),
        pl.col("d3") == t + timedelta(microseconds=500_000),
        pl.col("ts0") == utc + timedelta(microseconds=500_000),
        pl.col("f") == 1.2501,
        pl.col("f") == 1.25,
        pl.col("f") > 1.2501,
        pl.col("y") < 2200,
        pl.col("y") <= 3000,
        pl.col("y") > 1800,
        pl.col("y") == 2155,
        pl.col("s").str.starts_with("a"),
        pl.col("s").str.ends_with("b"),
    ]
    check_filter_ids(_scan(server, "pymysql", "r11_indexed"), predicates)


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_mysql_high_not_precedence(server: str) -> None:
    # `NOT x IS NULL` is `(NOT x) IS NULL` under HIGH_NOT_PRECEDENCE
    uri = SERVERS[server].uri
    if uri is None:
        pytest.skip(f"no {server} URI")
    pymysql = pytest.importorskip("pymysql")
    _execute(
        server,
        [
            "DROP TABLE IF EXISTS r11_not_prec",
            "CREATE TABLE r11_not_prec (id int, n int)",
            "INSERT INTO r11_not_prec VALUES (1, 0), (2, NULL), (3, 5)",
        ],
    )

    parts = _parts(uri)

    def connect() -> Any:
        return pymysql.connect(
            **parts,
            init_command="SET SESSION sql_mode = "
            "CONCAT(@@sql_mode, ',HIGH_NOT_PRECEDENCE')",
        )

    check_filter_ids(
        pl.scan_database("r11_not_prec", connect),
        [
            pl.col("n").is_not_null(),
            ~pl.col("n").is_null(),
            pl.col("n").eq_missing(5),
            ~pl.col("n").is_in([5]),
            pl.col("n").is_in([5], nulls_equal=True),
        ],
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "psycopg2", "pg8000"])
def test_postgres_extra_float_digits(driver: str) -> None:
    # with extra_float_digits = 0 a text-protocol driver read 0.1 + 0.2 as 0.3, while
    # the pushed filter compared the exact double
    pytest.importorskip(driver)
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS r11_float_digits",
            "CREATE TABLE r11_float_digits (id int, d float8)",
            "INSERT INTO r11_float_digits VALUES (1, 0.1::float8 + 0.2::float8), (2, 0.3)",
        ],
    )
    assert POSTGRES is not None
    sep = "&" if "?" in POSTGRES else "?"
    uri = f"{POSTGRES}{sep}options=-c%20extra_float_digits%3D0"
    lf = pl.scan_database("r11_float_digits", uri, driver=driver)
    out = lf.collect().sort("id")
    assert out["d"].to_list() == [0.1 + 0.2, 0.3]
    check_filter_ids(lf, [pl.col("d") == 0.1 + 0.2, pl.col("d") == 0.3])


@needs_clickhouse
def test_clickhouse_like_patterns_with_nul() -> None:
    # `s LIKE '%\0'` matched every row
    pytest.importorskip("clickhouse_connect")
    table = "r11_ch_nul"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, s String) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, 'a'), (2, 'a\\0'), (3, 'a\\0b'), "
            "(4, '\\0b'), (5, 'b'), (6, '')",
        ],
    )
    try:
        lf = _scan("clickhouse", "clickhouse_connect", table)
        check_filter_ids(
            lf,
            [
                pl.col("s").str.ends_with("\x00"),
                pl.col("s").str.ends_with("a\x00"),
                ~pl.col("s").str.ends_with("\x00b"),
                pl.col("s").str.starts_with("\x00"),
                pl.col("s").str.contains("\x00", literal=True),
            ],
        )
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_connectorx_question_marks() -> None:
    # ConnectorX's ClickHouse client reads `?` as a placeholder: "unbound query
    # argument"
    pytest.importorskip("connectorx")
    table = "r11_ch_question"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f'CREATE TABLE {table} (id Int32, s String, "q?" Int32) '
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, 'a?', 1), (2, 'b', 2)",
        ],
    )
    try:
        lf = _scan("clickhouse", "connectorx", table)
        out = lf.select("id", "s").filter(pl.col("s") == "a?").collect()
        assert out["id"].to_list() == [1]
        with pytest.raises(pl.exceptions.InvalidOperationError, match=r"column 'q\?'"):
            lf.select("q?").collect()
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
@pytest.mark.parametrize("driver", ["clickhouse_connect", "adbc", "connectorx"])
def test_clickhouse_long_is_in_of_long_strings(driver: str) -> None:
    # 9000 UUIDs in `IN (…)`: "Max query size exceeded" (256 KiB)
    import uuid

    table = f"r13_ch_long_in_{driver}"  # each its own: xdist runs them side by side
    values = [str(uuid.UUID(int=k)) for k in range(9000)]
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, s String) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, '{values[5]}'), (2, 'x')",
        ],
    )
    try:
        lf = _scan("clickhouse", driver, table)
        out = lf.filter(pl.col("s").is_in(values)).collect()
        assert out["id"].to_list() == [1]
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_connectorx_quotes_and_backslashes() -> None:
    # ConnectorX parses the SQL itself: "Single quoted string is not closed"
    pytest.importorskip("connectorx")
    table = "r13_ch_quotes"
    values = ["a\\'b", "\\'", "''", "it's"]
    rows = ", ".join(
        "({}, '{}')".format(i, v.replace("\\", "\\\\").replace("'", "\\'"))
        for i, v in enumerate(values)
    )
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, s String) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES {rows}",
        ],
    )
    try:
        lf = _scan("clickhouse", "connectorx", table)
        for i, v in enumerate(values):
            for predicate in [
                pl.col("s") == v,
                pl.col("s").str.starts_with(v),
                pl.col("s").str.contains(v, literal=True),
            ]:
                assert i in lf.filter(predicate).collect()["id"].to_list(), predicate
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
def test_adbc_connect_errors_scrub_query_passwords() -> None:
    # libpq quoted a `?password=` it couldn't decode: `invalid percent-encoded token`
    pytest.importorskip("adbc_driver_postgresql")
    assert POSTGRES is not None
    password = urlsplit(POSTGRES).password
    base = POSTGRES.replace(f":{password}@", "@", 1)
    sep = "&" if "?" in base else "?"
    for key in ["password", "sslpassword"]:
        with pytest.raises(Exception) as e:
            pl.scan_database(
                "SELECT 1 AS x", f"{base}{sep}{key}=Sekr3tPw%zz", driver="adbc"
            ).collect()
        assert "Sekr3tPw" not in str(e.value), key


@needs_postgres
def test_psycopg2_bytea_arrays() -> None:
    # "not yet implemented: Nested object types"
    pytest.importorskip("psycopg2")
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS r11_bytea_arrays",
            "CREATE TABLE r11_bytea_arrays (id int, b bytea[])",
            "INSERT INTO r11_bytea_arrays VALUES (1, ARRAY['\\x01'::bytea, NULL]), "
            "(2, NULL)",
        ],
    )
    for driver in ["psycopg2", "factory"]:
        connection: Any = POSTGRES
        kwargs: dict[str, Any] = {"driver": driver}
        if driver == "factory":
            psycopg2 = pytest.importorskip("psycopg2")

            connection, kwargs = (lambda: psycopg2.connect(POSTGRES)), {}  # noqa: B023
        for source in ["r11_bytea_arrays", "SELECT id, b FROM r11_bytea_arrays"]:
            out = pl.scan_database(source, connection, **kwargs).collect().sort("id")
            assert out["b"].to_list() == [[b"\x01", None], None], (driver, source)


@needs_postgres
def test_sqlalchemy_url_with_a_space_in_the_password() -> None:
    # SQLAlchemy renders a space unencoded, which libpq rejects
    sqlalchemy = pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            "DROP ROLE IF EXISTS r11_space",
            "CREATE ROLE r11_space LOGIN PASSWORD 'two words'",
        ],
    )
    url = sqlalchemy.engine.make_url(POSTGRES).set(
        username="r11_space", password="two words"
    )
    for driver in ["adbc", "psycopg"]:
        lf = pl.scan_database("SELECT 1 AS x", url, driver=driver)
        assert lf.collect()["x"].to_list() == [1], driver


@needs_postgres
@pytest.mark.parametrize("server", ["postgres17", "mysql"])
def test_sessionmaker_factory(server: str) -> None:
    # the session setup called `exec_driver_sql` on the ORM Session
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.orm import sessionmaker

    uri = SERVERS[server].uri
    if uri is None:
        pytest.skip(f"no {server} URI")
    if server == "mysql":
        uri = uri.replace("mysql://", "mysql+pymysql://", 1)
    engine = sqlalchemy.create_engine(uri)
    try:
        out = pl.scan_database("SELECT 1 AS x", sessionmaker(engine)).collect()
        assert out["x"].to_list() == [1]
    finally:
        engine.dispose()


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
def test_clickhouse_adbc_provider_credentials_win_over_the_query() -> None:
    # "not allowed to use X-ClickHouse HTTP headers and authentication via parameters"
    uri = SERVERS["clickhouse"].uri
    assert uri is not None
    parts = urlsplit(uri)
    base = uri.replace(f"{parts.username}:{parts.password}@", "", 1)
    for query, creds in [
        ("?password=stale", {"user": parts.username, "password": parts.password}),
        ("?user=nobody", {"user": parts.username, "password": parts.password}),
    ]:
        lf = pl.scan_database(
            "SELECT 1 AS x",
            base + query,
            driver="adbc",
            credential_provider=lambda creds=creds: creds,  # type: ignore[misc]
        )
        assert lf.collect()["x"].to_list() == [1], query


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracledb_scheme_alias() -> None:
    # "no dialect for 'oracledb'" (ConnectorX: "Source Unknown not supported")
    pytest.importorskip("sqlalchemy")
    uri = SERVERS["oracle"].uri
    assert uri is not None
    alias = uri.replace("oracle://", "oracledb://", 1)
    query = "SELECT 1 AS x FROM dual"
    out = pl.scan_database(query, alias, driver="sqlalchemy").collect()
    assert out.item() == 1
    if importlib.util.find_spec("connectorx") is not None:
        # (its Oracle source needs the Oracle client libraries)
        try:
            out = pl.scan_database(query, alias, driver="connectorx").collect()
        except Exception as e:
            assert "Source Unknown" not in str(e)  # noqa: PT017
        else:
            assert out.item() == 1


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracledb_pool_session_after_a_failed_read() -> None:
    # the cancelled session went back to the pool with NLS_COMP = BINARY
    oracledb = pytest.importorskip("oracledb")
    p = _parts(SERVERS["oracle"].uri or "")

    def linguistic(conn: Any, _tag: Any) -> None:
        conn.cursor().execute(
            "ALTER SESSION SET NLS_COMP = LINGUISTIC NLS_SORT = BINARY_CI"
        )

    pool = oracledb.create_pool(
        user=p["user"],
        password=p["password"],
        dsn=f"{p['host']}:{p['port']}/FREEPDB1",
        min=0,
        max=1,
        session_callback=linguistic,
    )
    query = (
        "SELECT CASE WHEN level < 5000 THEN 1 ELSE 1 / (level - level) END AS x "
        "FROM dual CONNECT BY level <= 6000"
    )
    try:
        with pytest.raises(Exception, match="ORA-01476"):
            pl.scan_database(
                query, pool.acquire, max_connections=1, batch_size=1000
            ).collect()
        with pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT value FROM nls_session_parameters "
                "WHERE parameter IN ('NLS_COMP', 'NLS_SORT') ORDER BY parameter"
            )
            assert [r[0] for r in cur.fetchall()] == ["LINGUISTIC", "BINARY_CI"]
    finally:
        pool.close(force=True)


# A view's query runs each time the view is read: MySQL merges it into the reading
# query, calling rand() again for a filter, and each partition reads other values
VOLATILE_VIEWS: dict[str, tuple[list[str], float]] = {
    "postgres17": (
        [
            "DROP VIEW IF EXISTS review_volatile_v",
            "DROP TABLE IF EXISTS review_volatile_ids",
            "CREATE TABLE review_volatile_ids (id int PRIMARY KEY)",
            "INSERT INTO review_volatile_ids SELECT generate_series(0, 299)",
            "CREATE VIEW review_volatile_v AS "
            "SELECT id, random() AS r FROM review_volatile_ids",
        ],
        0.5,
    ),
    "mysql": (
        [
            "DROP VIEW IF EXISTS review_volatile_v",
            "DROP TABLE IF EXISTS review_volatile_ids",
            "CREATE TABLE review_volatile_ids (id int PRIMARY KEY)",
            "INSERT INTO review_volatile_ids VALUES "
            + ", ".join(f"({i})" for i in range(300)),
            "CREATE VIEW review_volatile_v AS "
            "SELECT id, rand() AS r FROM review_volatile_ids",
            # MySQL's view definition leaves out the column list
            "DROP VIEW IF EXISTS review_volatile_cl",
            "CREATE VIEW review_volatile_cl (id, b) AS "
            "SELECT id, rand() AS r FROM review_volatile_ids",
        ],
        0.5,
    ),
    "clickhouse": (
        [
            "DROP VIEW IF EXISTS review_volatile_v",
            "DROP TABLE IF EXISTS review_volatile_ids",
            "CREATE TABLE review_volatile_ids (id Int32) ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_volatile_ids SELECT number FROM numbers(300)",
            "CREATE VIEW review_volatile_v AS "
            "SELECT id, rand() AS r FROM review_volatile_ids",
            "DROP VIEW IF EXISTS review_volatile_pv",
            "CREATE VIEW review_volatile_pv AS SELECT id FROM review_volatile_ids "
            "WHERE randCanonical() < {p:Float64}",
        ],
        2.0**31,
    ),
    # `RAND()` is one value per statement: a value per row needs
    # `RAND(CHECKSUM(NEWID()))`
    "mssql": (
        [
            "DROP SYNONYM IF EXISTS review_volatile_syn",
            "DROP SYNONYM IF EXISTS review_volatile_syn_f",
            "DROP FUNCTION IF EXISTS review_volatile_itvf",
            "DROP FUNCTION IF EXISTS review_volatile_itvf_dt",
            "DROP FUNCTION IF EXISTS review_volatile_mstvf",
            "DROP FUNCTION IF EXISTS review_volatile_mstvf_dt",
            "DROP VIEW IF EXISTS review_volatile_v",
            "DROP TABLE IF EXISTS review_volatile_ids",
            "CREATE TABLE review_volatile_ids (id int PRIMARY KEY)",
            "INSERT INTO review_volatile_ids "
            "SELECT TOP 300 ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1 "
            "FROM sys.all_objects",
            "CREATE VIEW review_volatile_v AS "
            "SELECT id, RAND(CHECKSUM(NEWID())) AS r FROM review_volatile_ids",
            "CREATE SYNONYM review_volatile_syn FOR review_volatile_v",
            # inline and multi-statement table-valued functions
            # (NEWID() and RAND() aren't allowed in the body of any function:
            # through the view)
            "CREATE FUNCTION review_volatile_itvf() RETURNS TABLE AS RETURN "
            "SELECT id, r FROM review_volatile_v",
            "CREATE FUNCTION review_volatile_itvf_dt() RETURNS TABLE AS RETURN "
            "SELECT id, SYSDATETIME() AS t FROM review_volatile_ids",
            "CREATE FUNCTION review_volatile_mstvf() RETURNS @t TABLE "
            "(id int, r float) AS BEGIN "
            "INSERT INTO @t SELECT id, r FROM review_volatile_v; RETURN END",
            "CREATE FUNCTION review_volatile_mstvf_dt() RETURNS @t TABLE "
            "(id int, t datetime2) AS BEGIN "
            "INSERT INTO @t SELECT id, SYSDATETIME() FROM review_volatile_ids; "
            "RETURN END",
            "CREATE SYNONYM review_volatile_syn_f FOR review_volatile_itvf",
        ],
        0.5,
    ),
    "trino": (
        [
            "DROP VIEW IF EXISTS review_volatile_v",
            "DROP TABLE IF EXISTS review_volatile_ids",
            "CREATE TABLE review_volatile_ids (id integer)",
            "INSERT INTO review_volatile_ids "
            "SELECT x FROM UNNEST(sequence(0, 299)) AS t(x)",
            "CREATE VIEW review_volatile_v AS "
            "SELECT id, random() AS r FROM review_volatile_ids",
        ],
        0.5,
    ),
    "oracle": (
        [
            'DROP VIEW IF EXISTS "review_volatile_v"',
            'DROP TABLE IF EXISTS "review_volatile_ids"',
            'CREATE TABLE "review_volatile_ids" ("id" NUMBER(10) PRIMARY KEY)',
            'INSERT INTO "review_volatile_ids" '
            "SELECT level - 1 FROM dual CONNECT BY level <= 300",
            'CREATE VIEW "review_volatile_v" AS '
            'SELECT "id", CAST(DBMS_RANDOM.VALUE AS BINARY_DOUBLE) AS "r" '
            'FROM "review_volatile_ids"',
            'CREATE OR REPLACE SYNONYM "review_volatile_s" FOR "review_volatile_v"',
        ],
        0.5,
    ),
}
VOLATILE_VIEWS["mariadb"] = VOLATILE_VIEWS["mysql"]


def _runnable_drivers(name: str, drivers: list[str] | None = None) -> list[str]:
    out = []
    for driver in drivers or list(SERVERS[name].drivers):
        module = DRIVER_MODULES.get(driver, driver)
        if module is not None and not _installed(module):
            continue
        unrunnable = KNOWN_UNRUNNABLE.get((name, driver))
        if unrunnable is not None and unrunnable() is not None:
            continue
        out.append(driver)
    return out


@pytest.mark.parametrize(
    "name",
    [pytest.param(name, marks=_server_marks(name)) for name in sorted(VOLATILE_VIEWS)],
)
def test_volatile_views(name: str) -> None:
    statements, threshold = VOLATILE_VIEWS[name]
    _execute(name, statements)
    for driver in _runnable_drivers(name):
        df = (
            _scan(name, driver, "review_volatile_v")
            .filter(pl.col("r") > threshold)
            .collect()
        )
        assert (df["r"] > threshold).all(), driver
        quote = '"' if name == "oracle" else ""
        for source in [
            "review_volatile_v",
            f"SELECT * FROM {quote}review_volatile_v{quote}",
        ]:
            lf = _scan(name, driver, source, partition_on="id", partition_num=4)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()
    # through a synonym; a parameterized view; unquoted names in another case
    more = {
        "oracle": ['"review_volatile_s"', 'SELECT * FROM "review_volatile_s"'],
        "clickhouse": ["SELECT * FROM review_volatile_pv(p = 0.5)"],
        "postgres17": ["SELECT * FROM REVIEW_Volatile_V"],
        "mssql": [
            "review_volatile_syn",
            "SELECT * FROM review_volatile_syn",
            "SELECT * FROM review_volatile_itvf()",
            "SELECT * FROM review_volatile_itvf_dt()",
            "SELECT * FROM review_volatile_mstvf()",
            "SELECT * FROM review_volatile_mstvf_dt()",
            "SELECT * FROM review_volatile_syn_f()",
            "SELECT id, NEWID() AS u FROM review_volatile_ids",
            "SELECT id, SYSDATETIME() AS t FROM review_volatile_ids",
            "SELECT id, RAND(CHECKSUM(NEWID())) AS r FROM review_volatile_ids",
        ],
    }.get(name, [])
    for source in more:
        lf = _scan(
            name, _default_driver(name), source, partition_on="id", partition_num=4
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
    if name in ("mysql", "mariadb"):
        for driver in _runnable_drivers(name):
            lf = _scan(name, driver, "review_volatile_cl")
            df = lf.filter(pl.col("b") > threshold).collect()
            assert (df["b"] > threshold).all(), driver
    # a table and a view without volatile functions still partition
    if SERVERS[name].partitions:
        assert _partitions(name, "review_volatile_ids") == list(range(300))


@pytest.mark.parametrize("name", ["postgres17", "mariadb"])
def test_partition_over_a_volatile_user_function(name: str) -> None:
    # a read-only session can't write, but each partition calls the function again
    if name == "postgres17":
        functions = [
            "CREATE OR REPLACE FUNCTION review_myrand() RETURNS float8 "
            "LANGUAGE sql VOLATILE AS 'SELECT random()'",
            "CREATE OR REPLACE FUNCTION review_twice(x int) RETURNS int "
            "LANGUAGE sql IMMUTABLE AS 'SELECT 2 * x'",
        ]
        drivers = ["psycopg", "adbc"]
    else:
        functions = [
            "DROP FUNCTION IF EXISTS review_myrand",
            "DROP FUNCTION IF EXISTS review_twice",
            "CREATE FUNCTION review_myrand() RETURNS double NOT DETERMINISTIC NO SQL "
            "RETURN rand()",
            "CREATE FUNCTION review_twice(x int) RETURNS int DETERMINISTIC NO SQL "
            "RETURN 2 * x",
        ]
        drivers = ["pymysql", "adbc"]
    # a table of its own: `test_volatile_views` may drop the other meanwhile
    tables = [
        t.replace("review_volatile_", "review_volatile_fn_")
        for t in VOLATILE_VIEWS[name][0][1:4]
    ]
    _execute(name, tables + functions)
    for driver in _runnable_drivers(name, drivers):
        lf = _scan(
            name,
            driver,
            "SELECT id, review_myrand() AS r FROM review_volatile_fn_ids",
            partition_on="r",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="review_myrand"):
            lf.collect()
        # a user's function is refused whatever its IMMUTABLE / DETERMINISTIC label,
        # which the database doesn't check; a plain column key still partitions
        lf = _scan(
            name,
            driver,
            "SELECT id, review_twice(id) AS d FROM review_volatile_fn_ids",
            partition_on="d",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
        lf = _scan(
            name,
            driver,
            "review_volatile_fn_ids",
            partition_on="id",
            partition_num=4,
        )
        assert sorted(lf.collect()["id"].to_list()) == list(range(300))


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_temporary_table_hiding_a_table(name: str) -> None:
    # `information_schema` doesn't list temporary tables: the hidden table's binary
    # collation was taken, and `s == "abc"` compared exactly matched 'ABC' too
    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    pytest.importorskip("pymysql")
    table = f"review_r14_temp_{name}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, s varchar(10) COLLATE utf8mb4_bin)",
            f"INSERT INTO {table} VALUES (1, 'zzz')",
        ],
    )
    assert server.factory is not None
    conn = server.factory(server.uri)()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TEMPORARY TABLE {table} "
                "(id int, s varchar(10) COLLATE utf8mb4_general_ci)"
            )
            cur.execute(f"INSERT INTO {table} VALUES (1, 'abc'), (2, 'ABC')")
        conn.commit()
        lf = pl.scan_database(table, conn)
        assert lf.filter(pl.col("s") == "abc").collect()["id"].to_list() == [1]
        assert lf.filter(pl.col("s") != "abc").collect()["id"].to_list() == [2]
        # code point order: 'ABC' < 'abd'
        assert lf.filter(pl.col("s") < "abd").collect()["id"].sort().to_list() == [1, 2]
    finally:
        conn.close()


@needs_clickhouse
def test_clickhouse_row_numbers_arent_filtered_in_the_database() -> None:
    # a filter on another column, pushed into the query, ran before the numbering:
    # `rowNumberInAllBlocks()` numbered the rows it kept
    _execute(
        "clickhouse",
        [
            "DROP TABLE IF EXISTS review_r14_rownum",
            "CREATE TABLE review_r14_rownum (id Int64, x Int64) ENGINE = MergeTree "
            "ORDER BY id",
            "INSERT INTO review_r14_rownum VALUES (1, 10), (2, 15), (3, 30), (4, 31)",
        ],
    )
    query = (
        "SELECT id, x, rowNumberInAllBlocks() AS r, blockNumber() AS b "
        "FROM review_r14_rownum"
    )
    for driver in _runnable_drivers(
        "clickhouse", ["adbc", "clickhouse_connect", "connectorx"]
    ):
        lf = _scan("clickhouse", driver, query)
        full = lf.collect().sort("id")
        assert full["r"].to_list() == [0, 1, 2, 3], driver
        for predicate in [pl.col("id") >= 2, pl.col("x") > 12, pl.col("r") > 0]:
            got = lf.filter(predicate).collect().sort("id")
            assert got.equals(full.filter(predicate)), (driver, predicate)


@needs_clickhouse
def test_clickhouse_connectorx_date_arrays_are_refused() -> None:
    # ConnectorX reads `Array(Date)` as garbage text, and may lose every row: an empty
    # frame, no error
    pytest.importorskip("connectorx")
    table = "review_r14_ch_date_array"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id UInt32, ad Array(Date)) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, ['2149-06-06']), (2, [])",
        ],
    )
    for source in [table, f"SELECT id, ad FROM {table}"]:
        with pytest.raises(pl.exceptions.InvalidOperationError, match="column 'ad'"):
            _scan("clickhouse", "connectorx", source).collect()
    assert _scan("clickhouse", "adbc", table).collect().height == 2


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_zero_dates_in_queries_and_under_schema(name: str) -> None:
    # Zero dates were read and compared as NULL only from a table, whose types the
    # catalog gives: a query's columns (on a driver without the type lookup) and those
    # under `schema=` were compared as stored, `dn < 2000-01-01` keeping '0000-00-00'
    # read as NULL; PyMySQL failed reading a query's
    table = "review_r15_zero_dates"
    _execute(
        name,
        [
            "SET SESSION sql_mode = ''",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, d date NOT NULL, "
            "dn date NULL, ts datetime NULL)",
            f"INSERT INTO {table} VALUES "
            "(1, '0000-00-00', '0000-00-00', '0000-00-00 00:00:00'), "
            "(2, '2020-01-05', '2020-01-05', '2020-01-05 10:00:00'), "
            "(3, '1999-01-01', NULL, NULL), "
            "(4, '2020-00-00', '2020-00-00', '2020-00-00 00:00:00')",
        ],
    )
    schema = {"id": pl.Int32, "d": pl.Date, "dn": pl.Date, "ts": pl.Datetime("us")}
    expected = {
        "all": [1, 2, 3, 4],
        "dn_before_2000": [],
        "dn_null": [1, 3, 4],
        "d_not_null": [2, 3],
        "ts_before_2000": [],
    }
    predicates = {
        "all": pl.lit(True),
        "dn_before_2000": pl.col("dn") < date(2000, 1, 1),
        "dn_null": pl.col("dn").is_null(),
        "d_not_null": pl.col("d").is_not_null(),
        "ts_before_2000": pl.col("ts") < datetime(2000, 1, 1),
    }
    query = f"SELECT * FROM {table}"
    for driver in _runnable_drivers(name, ["pymysql", "mysql.connector"]):
        for source, kwargs in [
            (query, {}),
            (table, {"schema": schema}),
            (query, {"schema": schema}),
        ]:
            lf = _scan(name, driver, source, **kwargs)
            for what, predicate in predicates.items():
                ids = lf.filter(predicate).collect()["id"].sort().to_list()
                assert ids == expected[what], (driver, source, kwargs, what)
        # bounds leave zero dates out
        lf = _scan(
            name, driver, table, schema=schema, partition_on="d", partition_num=3
        )
        assert lf.collect()["id"].sort().to_list() == [1, 2, 3, 4], driver


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
@pytest.mark.parametrize("dtype", [pl.Int32, pl.String, pl.Date])
def test_mysql_zero_dates_are_never_silently_dropped_by_a_null_test(
    name: str, dtype: pl.DataType
) -> None:
    # under `schema=` of a dtype other than a date, `IS NULL` was pushed as stored and
    # returned only the NULL row, the zero date (which the read refuses) left out
    # silently
    table = f"r460_zero_{name}_{str(dtype).lower()}"
    _execute(
        name,
        [
            "SET SESSION sql_mode = ''",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, dd date NULL)",
            f"INSERT INTO {table} VALUES (1, '0000-00-00'), (2, NULL), (3, '2020-01-02')",
        ],
    )
    try:
        lf = _scan(
            name,
            "adbc",
            table,
            schema={"id": pl.Int32, "dd": dtype},
        )
        for predicate, expected in [
            (pl.col("dd").is_null(), [1, 2]),
            (pl.col("dd").is_not_null(), [3]),
        ]:
            if dtype == pl.Date:
                # the zero date reads as NULL
                got = lf.filter(predicate).collect()["id"].sort().to_list()
                assert got == expected
            else:
                # the read refuses a zero date: it is not left out silently
                with pytest.raises(pl.exceptions.ComputeError, match="zero"):
                    lf.filter(predicate).collect()
    finally:
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize(
    ("name", "driver"),
    [
        pytest.param("postgres17", "psycopg"),
        pytest.param("mysql", "pymysql"),
        pytest.param("mysql", "mysql.connector"),
        pytest.param("mariadb", "pymysql"),
    ],
)
def test_params_dont_hide_volatile_views(name: str, driver: str) -> None:
    # With params=, every catalog lookup was given the parameters too, and failed; on
    # MySQL sqlglot didn't parse the `%s` placeholder either: the volatile view a query
    # reads went unchecked, and partitions were accepted, each reading other values
    if driver not in _runnable_drivers(name, [driver]):
        pytest.skip(f"{driver} isn't installed")
    # per driver: the cells run in parallel (`-n`) against one server
    suffix = driver.replace(".", "_")
    ids, view = f"review_r15_params_ids_{suffix}", f"review_r15_params_v_{suffix}"
    if name == "postgres17":
        statements = [
            f"DROP VIEW IF EXISTS {view}",
            f"DROP TABLE IF EXISTS {ids}",
            f"CREATE TABLE {ids} AS SELECT g AS id FROM generate_series(1, 300) g",
            f"CREATE VIEW {view} AS SELECT id, random() AS r FROM {ids}",
        ]
    else:
        statements = [
            f"DROP VIEW IF EXISTS {view}",
            f"DROP TABLE IF EXISTS {ids}",
            f"CREATE TABLE {ids} (id int PRIMARY KEY)",
            f"INSERT INTO {ids} WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL "
            "SELECT n + 1 FROM s WHERE n < 300) SELECT n FROM s",
            f"CREATE VIEW {view} AS SELECT id, RAND() AS r FROM {ids}",
        ]
    _execute(name, statements)
    query = f"SELECT * FROM {view} WHERE id > %s"
    lf = _scan(name, driver, query, params=[0], partition_on="id", partition_num=4)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        lf.collect()
    df = _scan(name, driver, query, params=[0]).filter(pl.col("r") < 0.5).collect()
    assert df.height > 0
    assert (df["r"] < 0.5).all()


@needs_postgres
def test_postgres_partitions_over_stable_functions() -> None:
    # A STABLE function returns the same values within a statement only, but each
    # partition is a statement (and a session) of its own: a user's over `now()`, and
    # `pg_backend_pid()`, were taken as repeatable (8 partitions read 8 values)
    ids = "review_r15_stable_ids"
    _execute(
        "postgres17",
        [
            "DROP VIEW IF EXISTS review_r15_stable_now",
            "DROP VIEW IF EXISTS review_r15_stable_pid",
            f"DROP TABLE IF EXISTS {ids}",
            f"CREATE TABLE {ids} AS SELECT g AS id FROM generate_series(1, 300) g",
            "CREATE OR REPLACE FUNCTION review_r15_now() RETURNS timestamptz "
            "LANGUAGE sql STABLE AS 'SELECT now()'",
            "CREATE VIEW review_r15_stable_now AS "
            f"SELECT id, review_r15_now() AS t FROM {ids}",
            "CREATE VIEW review_r15_stable_pid AS "
            f"SELECT id, pg_backend_pid() AS p FROM {ids}",
        ],
    )
    for source in [
        "review_r15_stable_now",
        f"SELECT id, review_r15_now() AS t FROM {ids}",
        "review_r15_stable_pid",
    ]:
        lf = _scan("postgres17", "psycopg", source, partition_on="id", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
    # one statement reads one value, and a filter is still run in the database
    lf = _scan("postgres17", "psycopg", "review_r15_stable_now")
    assert lf.collect()["t"].n_unique() == 1
    assert lf.filter(pl.col("id") <= 10).collect().height == 10


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_partitions_over_the_connection_id(name: str) -> None:
    # `CONNECTION_ID()` is the session's: each partition read another
    view = "review_r15_connection_v"
    _execute(
        name,
        [
            f"DROP VIEW IF EXISTS {view}",
            f"CREATE VIEW {view} AS SELECT 1 AS id, CONNECTION_ID() AS c",
        ],
    )
    for source in [view, "SELECT 1 AS id, CONNECTION_ID() AS c"]:
        lf = _scan(
            name, _default_driver(name), source, partition_on="id", partition_num=4
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()


@needs_clickhouse
def test_clickhouse_block_and_query_functions() -> None:
    # `blockSize()` and `blockSerializedSize()` are the block read's: a filter pushed
    # into the query (or a partition) read smaller blocks (5 for 300). `queryID()` and
    # `nowInBlock()` weren't volatile: each partition read another query's ID.
    table = "review_r15_ch_blocks"
    _execute(
        "clickhouse",
        [
            f"DROP VIEW IF EXISTS {table}_qid",
            f"DROP VIEW IF EXISTS {table}_nib",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int64) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} SELECT number FROM numbers(300)",
            f"CREATE VIEW {table}_qid AS SELECT id, queryID() AS q FROM {table}",
            f"CREATE VIEW {table}_nib AS SELECT id, nowInBlock() AS n FROM {table}",
        ],
    )
    query = f"SELECT id, blockSize() AS bs, blockSerializedSize(id) AS bss FROM {table}"
    for driver in _runnable_drivers(
        "clickhouse", ["adbc", "clickhouse_connect", "connectorx"]
    ):
        lf = _scan("clickhouse", driver, query)
        full = lf.collect().sort("id")
        got = lf.filter(pl.col("id") >= 295).collect().sort("id")
        assert got.equals(full.filter(pl.col("id") >= 295)), driver
    for source in [query, f"{table}_qid", f"{table}_nib"]:
        lf = _scan(
            "clickhouse",
            _default_driver("clickhouse"),
            source,
            partition_on="id",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()


@needs_postgres
@pytest.mark.parametrize("during_fetch", [False, True])
def test_psycopg_cancel_stops_later_server_fetch(
    monkeypatch: pytest.MonkeyPatch, during_fetch: bool
) -> None:
    import threading

    psycopg = pytest.importorskip("psycopg")
    from polars_database._bridge import _Session

    from polars.io.database._executor import ConnectionExecutor

    entered = threading.Event()
    original = ConnectionExecutor.to_polars

    def delayed_batches(self: Any, *args: Any, **kwargs: Any) -> Any:
        batches = iter(original(self, *args, **kwargs))
        yield next(batches)
        entered.set()
        time.sleep(1.6)  # a later FETCH starts after the former retry deadline
        yield from batches

    if during_fetch:
        monkeypatch.setattr(ConnectionExecutor, "to_polars", delayed_batches)
    conn = psycopg.connect(POSTGRES)
    conn.execute("SET statement_timeout = '5s'")
    session = _Session(conn, owned=True, execute_options={})
    frames = session.execute(
        "SELECT i, CASE WHEN i > 1 THEN pg_sleep(2) END AS pause "
        "FROM generate_series(1,3) i",
        1,
    )
    thread = None
    try:
        assert next(frames)["i"].to_list() == [1]
        assert not session._issuing
        started = time.monotonic()
        if during_fetch:

            def cancel() -> None:
                assert entered.wait(5)
                session.cancel()

            thread = threading.Thread(target=cancel)
            thread.start()
        else:
            session.cancel()
        with pytest.raises(Exception, match=r"cancelled|canceling statement"):
            next(frames)
        assert time.monotonic() - started < 3.5
        assert not session._issuing
    finally:
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive()
        frames.close()  # type: ignore[attr-defined]
        session.close()


@needs_postgres
@pytest.mark.parametrize("kind", ["cte", "join"])
def test_postgres_distinct_quoted_view_names_are_inspected(kind: str) -> None:
    # Lowercasing the CTE names hid the real view; lowercasing the seen-view key
    # skipped it after the stable uppercase view. Both allowed wrong-row partitions.
    view = f"review_r20_{kind}_case_v"
    _execute(
        "postgres17",
        [
            (
                f"CREATE OR REPLACE VIEW {view} AS "
                "SELECT id, random() AS r FROM generate_series(1, 1000) AS id"
            ),
            f'CREATE OR REPLACE VIEW "{view.upper()}" AS SELECT 1 AS dummy',
        ],
    )
    if kind == "cte":
        query = f'WITH "{view.upper()}" AS (SELECT 1) SELECT * FROM {view}'
    else:
        query = f'SELECT v.* FROM {view} v CROSS JOIN "{view.upper()}" s'
    for driver in ["live", *_runnable_drivers("postgres17", ["psycopg", "adbc"])]:
        assert _scan("postgres17", driver, query).collect().height == 1000
        for schema in [None, {"id": pl.Int32, "r": pl.Float64}]:
            lf = _scan(
                "postgres17",
                driver,
                query,
                schema=schema,
                partition_on="r",
                partition_num=2,
                partition_range=(0, 1),
            )
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()


@needs_postgres
def test_postgres_name_columns_are_not_compared_as_text() -> None:
    # #409: an untyped literal resolves to `name` and is cut to 63 bytes, so `n == L`
    # and `n < L` / `n != L` would disagree with Polars for a 70-byte `L`. (A `name`
    # column is read as text through a conversion, which isn't compared in the database
    # either: a guard for the catalog's classification.)
    long = "a" * 70
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS review_name_col",
            "CREATE TABLE review_name_col (id int, n name)",
            f"INSERT INTO review_name_col VALUES (1, '{'a' * 63}')",
        ],
    )
    for driver in ["live", *_runnable_drivers("postgres17", ["psycopg", "adbc"])]:
        for source in ["review_name_col", "SELECT * FROM review_name_col"]:
            lf = _scan("postgres17", driver, source)
            n = pl.col("n")
            assert lf.filter(n == long).collect().height == 0, (driver, source)
            assert lf.filter(n != long).collect().height == 1, (driver, source)
            assert lf.filter(n < long).collect().height == 1, (driver, source)
            assert lf.filter(n.is_in([long])).collect().height == 0, (driver, source)


@needs_clickhouse
@pytest.mark.parametrize("driver", ["adbc", "clickhouse_connect"])
def test_clickhouse_datetime_arrays_are_read_in_seconds(driver: str) -> None:
    # #429: ADBC read the elements' epoch seconds as milliseconds
    table = f"r31_ch_dt_arr_{driver}"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, a Array(DateTime('UTC')), "
            "n Array(Nullable(DateTime('Asia/Tokyo')))) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES "
            "(1, ['2024-01-01 00:00:00', '2024-06-01 12:30:00'], "
            "[NULL, '2024-01-01 00:00:00']), (2, [], [])",
        ],
    )
    for source in [table, f"SELECT * FROM {table}"]:
        df = _scan("clickhouse", driver, source).collect().sort("id")
        assert df["a"].to_list() == [
            [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 6, 1, 12, 30, tzinfo=UTC),
            ],
            [],
        ]
        n = df["n"][0].to_list()
        assert n[0] is None
        # (written in the column's zone)
        assert n[1] == datetime(2024, 1, 1, tzinfo=ZoneInfo("Asia/Tokyo"))


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "psycopg2"])
def test_postgres_timestamptz_arrays_read_in_utc_on_a_live_session(
    module: str,
) -> None:
    # #430: the array lacked the `AT TIME ZONE 'UTC'` of the scalar: in a session of
    # Europe/Amsterdam, 1900 reads as the zone's local mean time (00:19:32)
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS r31_tstz_arr",
            "CREATE TABLE r31_tstz_arr (id int, t timestamptz, a timestamptz[])",
            "INSERT INTO r31_tstz_arr VALUES "
            "(1, '1900-01-01 00:00+00', "
            "ARRAY['1900-01-01 00:00+00', '2024-06-01 12:00+00']::timestamptz[]), "
            "(2, NULL, NULL), "
            "(3, NULL, ARRAY[NULL, '1900-01-01 00:00+00']::timestamptz[])",
        ],
    )
    conn = _live_postgres(module)
    try:
        conn.cursor().execute("SET TIME ZONE 'Europe/Amsterdam'")
        conn.commit()
        for source in ["r31_tstz_arr", "SELECT * FROM r31_tstz_arr"]:
            df = pl.scan_database(source, conn).collect().sort("id")
            assert df.schema["a"] == pl.List(pl.Datetime("us", "UTC"))
            assert df["t"][0] == datetime(1900, 1, 1, tzinfo=UTC)
            assert df["a"][0].to_list() == [
                datetime(1900, 1, 1, tzinfo=UTC),
                datetime(2024, 6, 1, 12, tzinfo=UTC),
            ]
            assert df["a"][1] is None
            assert df["a"][2].to_list() == [None, datetime(1900, 1, 1, tzinfo=UTC)]
    finally:
        conn.close()


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
@pytest.mark.parametrize("pushdown", [True, False])
def test_mariadb_ordered_query_with_a_date_keeps_its_order(pushdown: bool) -> None:
    # #423: MariaDB drops the ORDER BY of a derived table: a query with a date column
    # was wrapped for the zero-date conversion, and read in another order
    table = f"r31_ord_{int(pushdown)}"  # (the parametrized runs may be parallel)
    _execute(
        "mariadb",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, d DATE)",
            f"INSERT INTO {table} VALUES "
            + ", ".join(f"({i}, '2024-01-0{i}')" for i in range(1, 6)),
        ],
    )
    for driver in ["pymysql", "live"]:
        lf = _scan(
            "mariadb",
            driver,
            f"SELECT id, d FROM {table} ORDER BY id DESC",
            pushdown=pushdown,
        )
        assert lf.collect()["id"].to_list() == [5, 4, 3, 2, 1], driver
        assert lf.head(1).collect()["id"].to_list() == [5], driver
        assert lf.collect()["d"][0] == date(2024, 1, 5)


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
@pytest.mark.parametrize("driver", ["clickhouse_connect", "connectorx"])
def test_clickhouse_datetime64_nanoseconds_with_a_schema_override(driver: str) -> None:
    # #435: the number the plugin's own conversion reads (nanoseconds since the epoch)
    # is no guess: an override of the column's type to ns must not refuse it
    _execute(
        "clickhouse",
        [
            "DROP TABLE IF EXISTS r31_dt64",
            "CREATE TABLE r31_dt64 (id Int32, ts DateTime64(9, 'UTC')) "
            "ENGINE = MergeTree ORDER BY id",
            "INSERT INTO r31_dt64 VALUES (1, '2024-01-01 20:00:00.123456789')",
        ],
    )
    out = _scan(
        "clickhouse",
        driver,
        "r31_dt64",
        schema_overrides={"ts": pl.Datetime("ns", "UTC")},
    ).collect()
    assert out["ts"].dtype == pl.Datetime("ns", "UTC")
    assert out["ts"].dt.nanosecond().item() == 123456789


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
@pytest.mark.parametrize("driver", ["pymysql", "adbc"])
def test_mariadb_ordered_query_sorts_by_the_derived_table_column(driver: str) -> None:
    # #423: the outer ORDER BY names the derived table's column (`_pl`.`d`), not the
    # select item that converts it (a decimal or SET read as text would sort as text);
    # a query's own conversions are the zero-date one, whose order is the column's
    table = f"r31_ordcol_{driver}"
    _execute(
        "mariadb",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, d DATE)",
            f"INSERT INTO {table} VALUES (1, '2024-03-01'), (2, '2024-01-01'), "
            "(3, '2024-02-01')",
        ],
    )
    lf = _scan(
        "mariadb", driver, f"SELECT id, d FROM {table} ORDER BY d", pushdown=False
    )
    assert "ORDER BY `_pl`.`d`" in lf.explain()
    assert lf.collect()["id"].to_list() == [2, 3, 1]


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
def test_mariadb_star_order_by_position_under_a_reordered_schema() -> None:
    # `ORDER BY 1` of a `SELECT *` is the database's first column, not the first of the
    # user's `schema=`: it sorted by the date, silently
    _execute(
        "mariadb",
        [
            "DROP TABLE IF EXISTS r42_star",
            "CREATE TABLE r42_star (id INT PRIMARY KEY, d DATE)",
            "INSERT INTO r42_star VALUES (1, '2024-01-03'), (2, '2024-01-02'), "
            "(3, '2024-01-01')",
        ],
    )
    schema = {"d": pl.Date, "id": pl.Int32}
    for driver in ["pymysql", "live"]:
        lf = _scan(
            "mariadb",
            driver,
            "SELECT * FROM r42_star ORDER BY 1 DESC",
            schema=schema,
            partition_on=None,
        )
        with pytest.raises(pl.exceptions.PolarsError, match="can't be kept"):
            lf.collect()
        lf = _scan(
            "mariadb",
            driver,
            "SELECT * FROM r42_star ORDER BY id DESC",
            schema=schema,
            partition_on=None,
        )
        assert lf.collect()["id"].to_list() == [3, 2, 1], driver
        # the database's own columns: the position is theirs
        lf = _scan(
            "mariadb",
            driver,
            "SELECT * FROM r42_star ORDER BY 1 DESC",
            partition_on=None,
        )
        assert lf.collect()["id"].to_list() == [3, 2, 1], driver


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_timestamp_read_as_text_is_the_same_on_every_live_path(name: str) -> None:
    # a TIMESTAMP typed String read `...+00:00` through a query or `schema=` and no
    # offset through a table (the bridge converted its instants to UTC, the SQL did)
    pymysql = pytest.importorskip("pymysql")
    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    u = urlsplit(server.uri)
    table = f"r42_ts_text_{name}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, ts TIMESTAMP NULL)",
            f"INSERT INTO {table} VALUES (1, '2024-01-01 22:00:00'), (2, NULL)",
        ],
    )
    conn = pymysql.connect(
        host=u.hostname,
        port=u.port,
        user=u.username,
        password=u.password,
        database=u.path.lstrip("/"),
        autocommit=True,
    )
    try:
        # (a session zone that isn't UTC: the instant is 17:00 there)
        conn.cursor().execute("SET time_zone = '+05:00'")
        texts = set()
        for source in [table, f"SELECT id, ts FROM {table}"]:
            kwargs_list: list[Any] = [
                {"schema_overrides": {"ts": pl.String}},
                {"schema": {"id": pl.Int32, "ts": pl.String}},
            ]
            for kwargs in kwargs_list:
                out = pl.scan_database(
                    source, conn, partition_on=None, **kwargs
                ).collect()
                texts.add(tuple(out.sort("id")["ts"].to_list()))
        assert texts == {("2024-01-01 22:00:00.000000+00:00", None)}
        # (through a URI the digits are the driver's, the instant the same)
        for source in [table, f"SELECT id, ts FROM {table}"]:
            out = pl.scan_database(
                source,
                server.uri,
                partition_on=None,
                schema_overrides={"ts": pl.String},
            ).collect()
            ts = out.sort("id")["ts"].to_list()
            assert ts[0].startswith("2024-01-01 22:00:00")
            assert ts[0].endswith("+00:00")
    finally:
        conn.close()
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
def test_mariadb_ordered_query_with_a_timestamp_in_a_live_session() -> None:
    # #423: the wrapper of a query's column conversions repeats the ORDER BY outside
    # (MariaDB drops it inside a derived table); a key that isn't a selected column
    # can't be both
    _execute(
        "mariadb",
        [
            "SET time_zone = '+00:00'",
            "DROP TABLE IF EXISTS r31_ots",
            "CREATE TABLE r31_ots (id INT PRIMARY KEY, ts TIMESTAMP NULL)",
            "INSERT INTO r31_ots VALUES "
            + ", ".join(f"({i}, '2024-01-0{i} 20:00:00')" for i in range(1, 6)),
        ],
    )
    conn = _mysql_connection("mariadb", "pymysql", "-08:00")
    try:
        out = pl.scan_database(
            "SELECT id, ts FROM r31_ots ORDER BY id DESC", conn
        ).collect()
        assert out["id"].to_list() == [5, 4, 3, 2, 1]
        lf = pl.scan_database("SELECT id, ts FROM r31_ots ORDER BY id DESC", conn)
        assert "ORDER BY `_pl`.`id` DESC" in lf.explain()
        with pytest.raises(pl.exceptions.InvalidOperationError, match="ORDER BY"):
            pl.scan_database("SELECT id, ts FROM r31_ots ORDER BY -id", conn).collect()
    finally:
        conn.close()


def _mysql_connection(name: str, module: str, time_zone: str | None) -> Any:
    uri = SERVERS[name].uri
    assert uri is not None
    if module == "pymysql":
        conn = pytest.importorskip("pymysql").connect(**_parts(uri))
    else:
        conn = pytest.importorskip("mysql.connector").connect(**_parts(uri))
    if time_zone is not None:
        conn.cursor().execute(f"SET time_zone = '{time_zone}'")
    return conn


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("module", ["pymysql", "mysql.connector"])
def test_mysql_live_session_zone_with_an_aware_datetime_of_unknown_type(
    module: str,
) -> None:
    # #440: a TIMESTAMP is read in the session's time zone: under `schema=` its type
    # isn't known, so the instants were shifted silently. The result's field type tells
    # a TIMESTAMP, and the session's zone is read: the instants are right (a table of
    # its own: the parametrizations run side by side)
    table = f"r31_tz_{module.replace('.', '_')}"
    _execute(
        "mysql",
        [
            "SET time_zone = '+00:00'",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, ts TIMESTAMP NULL)",
            f"INSERT INTO {table} VALUES (1, '2024-01-01 20:00:00')",
        ],
    )
    schema: dict[str, PolarsDataType] = {
        "id": pl.Int64,
        "ts": pl.Datetime("us", "UTC"),
    }
    expected = datetime(2024, 1, 1, 20, tzinfo=UTC)
    conn = _mysql_connection("mysql", module, "-08:00")
    try:
        out = pl.scan_database(table, conn, schema=schema).collect()
        assert out["ts"].item() == expected
        out = pl.scan_database(
            f"SELECT id, ts FROM {table}",
            conn,
            schema_overrides={"ts": pl.Datetime("us", "UTC")},
        ).collect()
        assert out["ts"].item() == expected
        # a table's declared type is known: the instant is read right
        out = pl.scan_database(table, conn).collect()
        assert out["ts"].item() == expected
    finally:
        conn.close()
    conn = _mysql_connection("mysql", module, "+00:00")
    try:
        out = pl.scan_database(table, conn, schema=schema).collect()
        assert out["ts"].item() == expected
    finally:
        conn.close()


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_sqlalchemy_connection_in_another_time_zone(name: str) -> None:
    # a live SQLAlchemy `Connection` over a MySQL driver reads a TIMESTAMP in its
    # session's zone as every other live MySQL connection: the instants, in UTC, as the
    # table source reads them, for text and for an unknown type; a DATETIME stays naive
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("pymysql")
    uri = SERVERS[name].uri
    if uri is None:
        pytest.skip(f"no {name} URI")
    table = f"r44_sa_tz_{name}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, ts TIMESTAMP NULL, "
            "dt DATETIME NULL)",
            "SET time_zone = '+00:00'",
            f"INSERT INTO {table} VALUES (1, '2024-01-01 22:00:00', "
            "'2024-01-01 22:00:00')",
        ],
    )
    engine = sa.create_engine(_sqlalchemy_url(uri, "pymysql"))
    instant = datetime(2024, 1, 1, 22, tzinfo=UTC)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SET time_zone = '+05:00'")
            for source in [table, f"SELECT id, ts, dt FROM {table}"]:
                lf = pl.scan_database(source, conn, partition_on=None)
                assert lf.collect_schema()["ts"] == pl.Datetime("us", "UTC")
                out = lf.collect()
                assert out["ts"].to_list() == [instant]
                assert out["dt"].to_list() == [datetime(2024, 1, 1, 22)]
                texts = set()
                kwargs_list: list[Any] = [
                    {"schema_overrides": {"ts": pl.String}},
                    {"schema": {"id": pl.Int32, "ts": pl.String}},
                ]
                for kwargs in kwargs_list:
                    out = pl.scan_database(
                        source, conn, partition_on=None, **kwargs
                    ).collect()
                    texts.add(out["ts"].item())
                assert texts == {"2024-01-01 22:00:00.000000+00:00"}, source
                out = pl.scan_database(
                    source,
                    conn,
                    partition_on=None,
                    schema={
                        "id": pl.Int32,
                        "ts": pl.Datetime("us", "UTC"),
                        "dt": pl.Datetime("us"),
                    },
                ).collect()
                assert out["ts"].to_list() == [instant]
                assert out["dt"].to_list() == [datetime(2024, 1, 1, 22)]
    finally:
        engine.dispose()
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize("module", ["pymysql", "mysql.connector"])
@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_params_with_percent_in_names(module: str, name: str) -> None:
    # #425: a `%` in a column's name, written by the plugin into a statement the driver
    # %-formats (`params=`), is doubled. Connector/Python replaces only `%s` and leaves
    # `%%` as it is: a column named `x%s` can't be written there (pushdown is off)
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    connector = module == "mysql.connector"
    table = f"r31_pct_{name}_{module.replace('.', '_')}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, `a%b` TEXT, `rate%` INT, "
            + ("d DATE)" if connector else "`x%s` INT, d DATE)"),
            f"INSERT INTO {table} (id, `a%b`, `rate%`, d) VALUES "
            "(1, 'x', 5, '2024-01-01'), (2, 'y', 7, '2024-01-02')",
        ],
    )
    conn = _mysql_connection(name, module, None)
    escaped = "a%b" if connector else "a%%b"
    rate = "rate%" if connector else "rate%%"
    try:
        for source in [
            f"SELECT * FROM {table} WHERE id >= %s",
            f"SELECT id, `{escaped}`, `{rate}`, d FROM {table} WHERE id >= %s",
        ]:
            kwargs_list: list[dict[str, Any]] = [
                {},
                {"pushdown": False},
                {"partition_on": None},
            ]
            for kwargs in kwargs_list:
                lf = pl.scan_database(source, conn, params=[2], **kwargs)
                out = lf.filter(pl.col("id") >= 2).collect()
                assert out["id"].to_list() == [2], (source, kwargs)
                assert out["d"].to_list() == [date(2024, 1, 2)], (source, kwargs)
                assert out["a%b"].to_list() == ["y"], (source, kwargs)
                assert lf.select("id", "rate%").collect()["rate%"].to_list() == [7]
    finally:
        conn.close()


@needs_postgres
def test_postgres_user_collations_named_like_builtin_ones() -> None:
    # a collation is known by its identity, not its name: a user's ICU `"C"` (or
    # `"default"`, `"POSIX"`, `ucs_basic`) is not the byte-wise built-in one
    _execute(
        "postgres17",
        [
            "DROP SCHEMA IF EXISTS r_coll_det CASCADE",
            "DROP SCHEMA IF EXISTS r_coll_nd CASCADE",
            "CREATE SCHEMA r_coll_det",
            "CREATE SCHEMA r_coll_nd",
            "CREATE COLLATION r_coll_det.\"C\" (provider = icu, locale = 'en-US')",
            'CREATE TABLE r_coll_det.keys (id int, s text COLLATE r_coll_det."C")',
            "INSERT INTO r_coll_det.keys VALUES (1, 'Z'), (2, 'a'), (3, 'z'), (4, 'A')",
        ]
        + [
            f'CREATE COLLATION r_coll_nd."{name}" (provider = icu, '
            "locale = 'und-u-ks-level2', deterministic = false)"
            for name in ["C", "default", "POSIX", "ucs_basic"]
        ]
        + [
            f'CREATE TABLE r_coll_nd.keys_{i} (id int, s text COLLATE r_coll_nd."{name}")'
            for i, name in enumerate(["C", "default", "POSIX", "ucs_basic"])
        ]
        + [
            f"INSERT INTO r_coll_nd.keys_{i} VALUES (1, 'Z'), (2, 'a'), (3, 'z'), (4, 'A')"
            for i in range(4)
        ],
    )
    try:
        # the table path: ordering over a user collation named `C`
        lf = _scan("postgres17", "psycopg", "r_coll_det.keys")
        check_filter_ids(
            lf,
            [
                pl.col("s") < "a",
                pl.col("s") > "a",
                pl.col("s") >= "Z",
                pl.col("s") == "a",
            ],
        )
        # the query path: the printed collation name is the user's, visible first on the
        # search path
        for i in range(4):
            lf = _scan(
                "postgres17",
                "psycopg",
                f"SELECT id, s FROM keys_{i}",
                connect_options={"options": "-c search_path=r_coll_nd,pg_catalog"},
            )
            check_filter_ids(lf, [pl.col("s") == "a", pl.col("s") != "a"])
    finally:
        _execute(
            "postgres17",
            [
                "DROP SCHEMA IF EXISTS r_coll_det CASCADE",
                "DROP SCHEMA IF EXISTS r_coll_nd CASCADE",
            ],
        )


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_literals_in_the_1582_calendar_gap() -> None:
    # Oracle reads a literal in 1582-10-05..14 back as 1582-10-15, which the stored
    # values and Polars' comparison don't: it is not pushed
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE r512_gap'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE r512_gap ("id" NUMBER(9), d DATE, t TIMESTAMP(6))',
            "INSERT INTO r512_gap VALUES (1, DATE '1582-10-04', TIMESTAMP '1582-10-04 00:00:00')",
            "INSERT INTO r512_gap VALUES (2, DATE '1582-10-15', TIMESTAMP '1582-10-15 00:00:00')",
            "INSERT INTO r512_gap VALUES (3, DATE '1582-10-16', TIMESTAMP '1582-10-16 00:00:00')",
        ],
    )
    try:
        lf = _scan("oracle", "oracledb", "R512_GAP")
        literals = [
            datetime(1582, 10, 10),
            datetime(1582, 10, 5),
            datetime(1582, 10, 14),
        ]
        predicates = []
        for col in ("D", "T"):
            for lit in literals:
                c = pl.col(col)
                predicates += [
                    c == lit,
                    c != lit,
                    c < lit,
                    c <= lit,
                    c > lit,
                    c >= lit,
                    c.is_in([lit]),
                ]
        check_filter_ids(lf, predicates)
    finally:
        _execute("oracle", ["DROP TABLE r512_gap"])


# Silent value and type conversions.


@pytest.mark.skipif(SERVERS["mariadb"].uri is None, reason="no MariaDB URI")
@pytest.mark.parametrize("driver", ["pymysql", "connectorx", "live"])
def test_mariadb_query_with_order_by_reads_the_dtypes_of_one_without(
    driver: str,
) -> None:
    # the result types of a query with a top-level ORDER BY weren't looked up: a FLOAT
    # holding 16777216 read as 6-digit text (16777200.0), a YEAR as String
    if driver == "connectorx":
        pytest.importorskip("connectorx")
    table = f"t455_order_{driver}"
    _execute(
        "mariadb",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, f FLOAT, y YEAR)",
            f"INSERT INTO {table} VALUES (1, 16777216, 2024), (2, 1.5, 1999)",
        ],
    )
    try:
        plain = _scan("mariadb", driver, f"SELECT id, f, y FROM {table}").collect()
        ordered = _scan(
            "mariadb", driver, f"SELECT id, f, y FROM {table} ORDER BY id"
        ).collect()
        assert plain.schema == ordered.schema
        assert ordered["f"].to_list() == [16777216.0, 1.5]
        assert ordered["y"].to_list() == plain["y"].to_list()
    finally:
        _execute("mariadb", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
@pytest.mark.parametrize("driver", ["pymysql", "live"])
def test_mysql_select_star_order_by_is_kept_with_converted_columns(
    name: str, driver: str
) -> None:
    # `SELECT * … ORDER BY id DESC` of a table with a FLOAT / DATE / DATETIME /
    # TIMESTAMP column raised "the query's ORDER BY can't be kept" on MariaDB: the star
    # is the planned columns
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    table = f"t457_star_{name}_{driver}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, f FLOAT, d DATE, "
            "dt DATETIME, ts TIMESTAMP NULL)",
            f"INSERT INTO {table} VALUES "
            "(1, 16777216, '2024-01-01', '2024-01-01 01:02:03', '2024-01-01 01:02:03'), "
            "(3, 1.5, '2024-01-03', '2024-01-03 01:02:03', '2024-01-03 01:02:03'), "
            "(2, NULL, NULL, NULL, NULL)",
        ],
    )
    try:
        for query in [
            f"SELECT * FROM {table} ORDER BY id DESC",
            f"SELECT * FROM {table} ORDER BY {table}.id DESC",
            f"SELECT * FROM {table} AS t ORDER BY 1 DESC",
        ]:
            lf = _scan(name, driver, query, partition_on=None)
            out = lf.collect()
            assert out["id"].to_list() == [3, 2, 1], query
            assert out["f"].to_list() == [1.5, None, 16777216.0], query
            assert out["d"].to_list() == [date(2024, 1, 3), None, date(2024, 1, 1)]
            assert lf.head(2).collect()["id"].to_list() == [3, 2], query
    finally:
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_live_query_timestamps_are_instants_in_a_non_utc_session(
    name: str,
) -> None:
    # a query's TIMESTAMP on a live connection read as naive wall-clock time of the
    # session's zone, where the table reads UTC instants: dt.date() put a row on another
    # day. The zone is read from the session; where it can't be told apart, or a
    # wall-clock time is ambiguous, the read fails
    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    assert server.factory is not None
    table = f"t76_live_{name}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT PRIMARY KEY, ts TIMESTAMP NULL, "
            "dt DATETIME)",
            "SET time_zone = '+00:00'",
            f"INSERT INTO {table} VALUES "
            "(1, '2024-01-01 22:00:00', '2024-01-01 22:00:00'), "
            "(2, NULL, NULL), "
            "(3, '2024-11-03 09:30:00', '2024-11-03 09:30:00'), "
            "(4, '2024-07-01 12:00:00', '2024-07-01 12:00:00')",
        ],
    )
    conn = server.factory(server.uri)()
    try:

        def run(statement: str) -> None:
            cur = conn.cursor()
            cur.execute(statement)
            cur.close()

        run("SET time_zone = '+00:00'")
        utc = (
            pl.scan_database(f"SELECT * FROM {table}", conn, partition_on=None)
            .collect()
            .sort("id")
        )
        assert utc.schema["ts"] == pl.Datetime("us", "UTC")
        for zone in ["-08:00", "+05:30", "SYSTEM"]:
            run(f"SET time_zone = '{zone}'")
            for source in [f"SELECT * FROM {table}", table]:
                lf = pl.scan_database(source, conn, partition_on=None)
                got = lf.collect().sort("id")
                assert_frame_equal(got, utc)
                assert got["ts"].dt.date().to_list()[0] == date(2024, 1, 1)
        # the zone is read for each query
        run("SET time_zone = '-08:00'")
        for source in [f"SELECT id, ts FROM {table} WHERE id = 1", table]:
            out = pl.scan_database(source, conn, partition_on=None).collect()
            assert out["ts"].dt.convert_time_zone("UTC")[0] == datetime(
                2024, 1, 1, 22, tzinfo=UTC
            )
        # a zone with daylight saving time: exact outside the repeated hour
        try:
            run("SET time_zone = 'America/Los_Angeles'")
        except Exception:
            pytest.skip("no time zone tables on this server")
        out = pl.scan_database(
            f"SELECT id, ts FROM {table} WHERE id IN (1, 4)", conn, partition_on=None
        ).collect()
        assert out.sort("id")["ts"].to_list() == [
            datetime(2024, 1, 1, 22, tzinfo=UTC),
            datetime(2024, 7, 1, 12, tzinfo=UTC),
        ]
        # 2024-11-03 09:30 UTC is 01:30 PST, a wall-clock time that happened twice
        with pytest.raises(pl.exceptions.PolarsError):
            pl.scan_database(
                f"SELECT id, ts FROM {table} WHERE id = 3", conn, partition_on=None
            ).collect()
        # an abbreviation names no zone's rules
        run("SET time_zone = 'SYSTEM'")
    finally:
        conn.close()
        _execute(name, [f"DROP TABLE IF EXISTS {table}"])


@needs_clickhouse
def test_clickhouse_datetime64_arrays_retyped_to_a_coarser_unit_are_not_misread() -> (
    None
):
    # the nanosecond ticks of an Array(DateTime64(9)) read as microseconds (year 55972)
    # under a List(Datetime("us")) override, a "ms" one panicked: each is an error or
    # the right instant
    pytest.importorskip("clickhouse_connect")
    table = "t456_ch_ticks"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, a Array(DateTime64(9, 'UTC'))) "
            "ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES (1, ['2024-01-02 03:04:05.123456789'])",
        ],
    )
    try:
        for unit in ["us", "ms"]:
            for source in [table, f"SELECT * FROM {table}"]:
                lf = _scan(
                    "clickhouse",
                    "clickhouse_connect",
                    source,
                    schema_overrides={"a": pl.List(pl.Datetime(unit, "UTC"))},  # type: ignore[arg-type]
                )
                try:
                    df = lf.collect()
                except pl.exceptions.PolarsError:
                    continue
                [value] = df["a"][0].to_list()
                assert value.replace(microsecond=0) == datetime(
                    2024, 1, 2, 3, 4, 5, tzinfo=UTC
                ), (unit, source)
    finally:
        _execute("clickhouse", [f"DROP TABLE IF EXISTS {table}"])


@needs_postgres
def test_postgres_psycopg2_infinity_is_an_error_without_a_type_lookup() -> None:
    # under `schema=` no type lookup ran, and psycopg2 read ±infinity as date.max /
    # date.min
    table = "t99_pg_inf"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, d date, tz timestamptz[])",
            f"INSERT INTO {table} VALUES (1, '2024-01-01', '{{2024-01-01 00:00+00}}'), "
            "(2, '-infinity', NULL), (3, NULL, '{infinity}')",
        ],
    )
    schema = {
        "id": pl.Int32,
        "d": pl.Date,
        "tz": pl.List(pl.Datetime("us", "UTC")),
    }
    try:
        for column in ["d", "tz"]:
            lf = _scan("postgres17", "psycopg2", table, schema=schema)
            with pytest.raises(pl.exceptions.PolarsError, match="infinity"):
                lf.select(column).collect()
        ok = _scan("postgres17", "psycopg2", table, schema=schema).select("id")
        assert ok.collect()["id"].sort().to_list() == [1, 2, 3]
    finally:
        _execute("postgres17", [f"DROP TABLE IF EXISTS {table}"])


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_view_column_list_permuting_the_query_aliases(name: str) -> None:
    # under `schema=` a view's columns were matched to its query's by name, but MySQL's
    # definition leaves out the view's column list: `a` here is `rand()`
    _execute(
        name,
        [
            "DROP VIEW IF EXISTS review_volatile_perm",
            "DROP TABLE IF EXISTS review_volatile_perm_ids",
            "CREATE TABLE review_volatile_perm_ids (id int PRIMARY KEY)",
            "INSERT INTO review_volatile_perm_ids VALUES "
            + ", ".join(f"({i})" for i in range(300)),
            "CREATE VIEW review_volatile_perm (a, b) AS "
            "SELECT rand() AS b, id AS a FROM review_volatile_perm_ids",
        ],
    )
    # the view's column `a` is `rand()`; the schema may list the columns in either order
    for schema in (
        {"a": pl.Float64, "b": pl.Int64},
        {"b": pl.Int64, "a": pl.Float64},
    ):
        for driver in _runnable_drivers(name):
            lf = _scan(name, driver, "review_volatile_perm", schema=schema)
            # `a` is never filtered by the database (it would call rand() again)
            pred = pl.col("a") > 0.5
            assert "filters 0/1" in lf.filter(pred).explain(), (driver, schema)
            df = lf.filter(pred).collect()
            assert (df["a"] > 0.5).all(), (driver, schema)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                _scan(
                    name,
                    driver,
                    "review_volatile_perm",
                    schema=schema,
                    partition_on="b",
                    partition_num=4,
                ).collect()


def _refused(
    name: str,
    source: str,
    key: str = "id",
    driver: str | None = None,
    *,
    via: str | None = None,
    why: str | None = None,
) -> None:
    """
    The explicit `partition_on` is refused.

    `via`: which check fired, `"query"` (the
    SQL's own allowlist, `this query can't be partitioned`) or `"source"` (the catalog:
    a view, a table's kind, a user's function, a collation); `why`: a regex the message
    names the unproven item by.
    """
    lf = _scan(
        name, driver or _default_driver(name), source, partition_on=key, partition_num=4
    )
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="can't be partitioned"
    ) as e:
        lf.collect()
    message = str(e.value)
    if via is not None:
        assert f"this {via} can't be partitioned" in message, message
    if why is not None:
        assert re.search(why, message, re.IGNORECASE), message


def _partitions(name: str, source: str, key: str = "id") -> list[int]:
    lf = _scan(name, _default_driver(name), source, partition_on=key, partition_num=4)
    return sorted(lf.collect()["id"].to_list())


def _execute_as_root(name: str, statements: list[str]) -> None:
    _execute(name, statements, root=True)


def test_postgres_allowlist_kinds_of_relation() -> None:
    # only ordinary tables (whose children are ordinary), and views proven the same way,
    # partition; server views, user operators and casts do not
    name = "postgres17"
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_al_parent CASCADE",
            "DROP TABLE IF EXISTS review_al_ids CASCADE",
            "CREATE TABLE review_al_ids (id int PRIMARY KEY)",
            "INSERT INTO review_al_ids SELECT g FROM generate_series(0, 299) g",
            "CREATE TABLE review_al_parent (id int, p int) PARTITION BY RANGE (id)",
            "CREATE TABLE review_al_p1 PARTITION OF review_al_parent FOR VALUES "
            "FROM (0) TO (150)",
            "CREATE TABLE review_al_p2 PARTITION OF review_al_parent FOR VALUES "
            "FROM (150) TO (300)",
            "INSERT INTO review_al_parent SELECT g, 0 FROM generate_series(0, 299) g",
            "CREATE OR REPLACE VIEW review_al_v AS SELECT id FROM review_al_ids",
        ],
    )
    for source in ["review_al_ids", "review_al_parent", "review_al_v"]:
        assert _partitions(name, source) == list(range(300))
    # server-state views: their definitions call functions that aren't proven
    for source in [
        "SELECT datid FROM pg_stat_database",
        "SELECT pid AS id FROM pg_stat_activity",
    ]:
        _refused(name, source, "pid" if "activity" in source else "datid")
    # a user's operator over a volatile function, on a domain over `int`: it counts for
    # the sources with a column of that type only; one on `int` would count for every
    # int column of the tests running at the same time
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_al_dt",
            "DROP OPERATOR IF EXISTS ===(review_al_dom, review_al_dom)",
            "DROP FUNCTION IF EXISTS review_al_noisy(review_al_dom, review_al_dom)",
            "DROP DOMAIN IF EXISTS review_al_dom",
            "CREATE DOMAIN review_al_dom AS int",
            "CREATE TABLE review_al_dt (id review_al_dom PRIMARY KEY)",
            "INSERT INTO review_al_dt SELECT g FROM generate_series(0, 299) g",
            "CREATE OR REPLACE FUNCTION review_al_noisy(a review_al_dom, "
            "b review_al_dom) RETURNS bool "
            "LANGUAGE sql VOLATILE AS 'SELECT random() < 0.5'",
            "CREATE OPERATOR === (LEFTARG = review_al_dom, RIGHTARG = review_al_dom, "
            "FUNCTION = review_al_noisy)",
        ],
    )
    try:
        # the error names the operator
        _refused(
            name,
            "SELECT id FROM review_al_dt WHERE id === id",
            via="source",
            why=r"operator .*===",
        )
        # a source with no column of the operator's type is fine
        assert _partitions(name, "review_al_ids") == list(range(300))
        assert _partitions(name, "SELECT id FROM review_al_ids WHERE id = id") == list(
            range(300)
        )
    finally:
        _execute(
            name,
            [
                "DROP TABLE IF EXISTS review_al_dt",
                "DROP OPERATOR IF EXISTS ===(review_al_dom, review_al_dom)",
                "DROP FUNCTION IF EXISTS review_al_noisy(review_al_dom, review_al_dom)",
                "DROP DOMAIN IF EXISTS review_al_dom",
            ],
        )


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_allowlist_kinds_of_relation(name: str) -> None:
    # information_schema's system views, session variables, and a VIRTUAL column that
    # calls rand() are not proven
    statements = [
        "DROP TABLE IF EXISTS review_al_ids",
        "CREATE TABLE review_al_ids (id int PRIMARY KEY)",
        "INSERT INTO review_al_ids SELECT seq FROM (SELECT @r := @r + 1 AS seq FROM "
        "information_schema.columns a, (SELECT @r := -1) v LIMIT 300) t",
    ]
    _execute(name, statements)
    _refused(name, "SELECT id FROM review_al_ids WHERE id > @@pseudo_thread_id - 1")
    _refused(name, "SELECT ID AS id FROM information_schema.processlist", "id")
    if name == "mariadb":
        _execute(
            name,
            [
                "DROP TABLE IF EXISTS review_al_gen",
                "CREATE TABLE review_al_gen (id int PRIMARY KEY, "
                "k int AS (floor(rand() * 1000)) VIRTUAL)",
                "INSERT INTO review_al_gen (id) SELECT id FROM review_al_ids",
                "DROP TABLE IF EXISTS review_al_okgen",
                "CREATE TABLE review_al_okgen (id int PRIMARY KEY, "
                "k int AS (id % 7) VIRTUAL)",
                "INSERT INTO review_al_okgen (id) SELECT id FROM review_al_ids",
            ],
        )
        _refused(name, "review_al_gen", "k")
        _refused(name, "SELECT id, k FROM review_al_gen", "k")
        for key in ["id", "k"]:
            assert _partitions(name, "review_al_okgen", key) == list(range(300))
    assert _partitions(name, "review_al_ids") == list(range(300))


def test_clickhouse_allowlist_kinds_of_relation() -> None:
    # an ALIAS column, a Merge table over a volatile view, table functions and virtual
    # columns are not proven; MergeTree tables are
    name = "clickhouse"
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_al_alias",
            "DROP TABLE IF EXISTS review_al_merge",
            "DROP VIEW IF EXISTS review_al_v",
            "DROP TABLE IF EXISTS review_al_ids",
            "CREATE TABLE review_al_ids (id UInt32) ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_al_ids SELECT number FROM numbers(300)",
            "CREATE TABLE review_al_alias (id UInt32, k UInt32 ALIAS rand() % 1000) "
            "ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_al_alias (id) SELECT number FROM numbers(300)",
            "CREATE VIEW review_al_v AS SELECT id FROM review_al_ids "
            "WHERE rand() % 2 = 0",
            "CREATE TABLE review_al_merge (id UInt32) ENGINE = Merge(currentDatabase(), "
            "'^review_al_v$')",
        ],
    )
    _refused(name, "SELECT id, k FROM review_al_alias", "k")
    _refused(name, "review_al_alias", "id")
    _refused(name, "review_al_merge", "id")
    _refused(name, "SELECT id FROM merge(currentDatabase(), '^review_al_ids$')")
    _refused(name, "SELECT id, _part FROM review_al_ids", "id")
    assert _partitions(name, "review_al_ids") == list(range(300))


def test_oracle_allowlist_synonym_chain() -> None:
    # a synonym of a synonym of a volatile view is followed (or not proven)
    name = "oracle"
    statements, _ = VOLATILE_VIEWS[name]
    _execute(
        name,
        [
            *statements,
            'CREATE OR REPLACE SYNONYM "review_volatile_s2" FOR "review_volatile_s"',
            'CREATE OR REPLACE SYNONYM "review_volatile_s3" FOR "review_volatile_s2"',
        ],
    )
    for source in ['"review_volatile_s2"', 'SELECT * FROM "review_volatile_s3"']:
        _refused(name, source)
    assert _partitions(name, "review_volatile_ids") == list(range(300))


def _ids_table(name: str, table: str, extra: str = "") -> None:
    """`table (id)` of 300 rows, on the server's own syntax."""
    if name.startswith("postgres"):
        statements = [
            f"DROP TABLE IF EXISTS {table} CASCADE",
            f"CREATE TABLE {table} (id int PRIMARY KEY{extra})",
            f"INSERT INTO {table} (id) SELECT g FROM generate_series(0, 299) g",
        ]
    elif name == "clickhouse":
        statements = [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id UInt32{extra}) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} (id) SELECT number FROM numbers(300)",
        ]
    elif name == "mssql":
        statements = [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY{extra})",
            f"INSERT INTO {table} (id) SELECT TOP 300 ROW_NUMBER() OVER "
            "(ORDER BY (SELECT NULL)) - 1 FROM sys.all_objects",
        ]
    elif name == "trino":
        statements = [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id integer{extra})",
            f"INSERT INTO {table} (id) SELECT x FROM UNNEST(sequence(0, 299)) AS t(x)",
        ]
    else:
        statements = [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY{extra})",
            f"INSERT INTO {table} (id) SELECT seq FROM (SELECT @r := @r + 1 AS seq FROM "
            "information_schema.columns a, (SELECT @r := -1) v LIMIT 300) t",
        ]
    _execute(name, statements)


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_user_function_named_like_another_dialects_builtin(name: str) -> None:
    # `initcap`, `date_trunc` and `strpos` are sqlglot's, not MySQL's built-ins: a
    # stored function of the name runs there, and was proven because the allowlist
    # applied to every dialect. The built-ins TRUNCATE and CHAR are no user's.
    fns = ["initcap", "date_trunc", "strpos"]
    _ids_table(name, "review_vf_ids")
    _execute_as_root(
        name,
        [f"DROP FUNCTION IF EXISTS polars.{fn}" for fn in fns]
        + [
            f"CREATE FUNCTION polars.{fn}(x INT) RETURNS INT NOT DETERMINISTIC NO SQL "
            "RETURN FLOOR(RAND() * 1000)"
            for fn in fns
        ],
    )
    try:
        for fn in fns:
            via = "query" if fn == "initcap" else None
            _refused(
                name, f"SELECT id, {fn}(id) AS k FROM review_vf_ids", via=via, why=fn
            )
        assert _partitions(name, "review_vf_ids") == list(range(300))
        assert _partitions(
            name,
            "SELECT id, TRUNCATE(id / 3, 0) AS k, CHAR(65) AS c FROM review_vf_ids",
        ) == list(range(300))
    finally:
        _execute_as_root(name, [f"DROP FUNCTION IF EXISTS polars.{fn}" for fn in fns])


def test_clickhouse_user_functions_named_like_another_dialects_builtin() -> None:
    # none of these names is a ClickHouse built-in (`bool_and` is sqlglot's), where
    # `levenshteinDistance` and `editDistance` are (sqlglot calls both LEVENSHTEIN)
    name = "clickhouse"
    _ids_table(name, "review_vf_ids")
    names = [
        "total",
        "cot",
        "quote",
        "unicode",
        "levenshtein",
        "split_part",
        "bool_and",
    ]
    drops = [f"DROP FUNCTION IF EXISTS {n}" for n in names]
    _execute(name, drops)
    _execute(name, [f"CREATE FUNCTION {n} AS (x) -> rand() % 1000" for n in names])
    try:
        for n in names:
            _refused(
                name,
                f"SELECT id, {n}(id) AS k FROM review_vf_ids",
                via="query",
                why=n,
            )
        # (a user's `levenshtein` would stand for them both)
        _execute(name, ["DROP FUNCTION IF EXISTS levenshtein"])
        assert _partitions(
            name,
            "SELECT id, levenshteinDistance('a', 'b') AS a, "
            "editDistance('a', 'b') AS b FROM review_vf_ids",
        ) == list(range(300))
    finally:
        _execute(name, drops)
    assert _partitions(name, "review_vf_ids") == list(range(300))


@pytest.mark.parametrize(
    "name", ["postgres17", "mysql", "mariadb", "clickhouse", "mssql"]
)
def test_casts_of_time_columns_to_dates_still_partition(name: str) -> None:
    # a cast (or `date(col)`) of a column that holds no text is proven; a plain table, a
    # view and a GROUP BY over an int key too
    column = {
        "clickhouse": "DateTime",
        "postgres17": "timestamp",
        "mssql": "datetime2",
    }.get(name, "DATETIME")
    _ids_table(name, "review_cast_ids", extra=f", ts {column}")
    expected = list(range(300))
    for source in [
        "review_cast_ids",
        "SELECT id, CAST(ts AS DATE) AS d FROM review_cast_ids",
        "SELECT id, CAST(ts AS DATE) AS d FROM review_cast_ids GROUP BY id, ts",
        "SELECT id, COUNT(*) AS n FROM review_cast_ids GROUP BY id",
    ]:
        assert _partitions(name, source) == expected, source
    if name == "postgres17":
        assert (
            _partitions(name, "SELECT id, date(ts) AS d FROM review_cast_ids")
            == expected
        )


def _nopad_bin(name: str) -> str:
    """The binary collation of the server that doesn't pad `'a'` to `'a '`."""
    return "utf8mb4_nopad_bin" if name == "mariadb" else "utf8mb4_0900_bin"


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_collapsing_over_a_case_insensitive_column(name: str) -> None:
    # `GROUP BY s` over `utf8mb4_*_ci` returns any one spelling of 'a' / 'A', so
    # partitions disagree; binary ones don't
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_ci",
            "CREATE TABLE review_ci (id INT PRIMARY KEY, s VARCHAR(5) "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci, "
            f"b VARCHAR(5) CHARACTER SET utf8mb4 COLLATE {_nopad_bin(name)}, "
            "p VARCHAR(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin)",
            "INSERT INTO review_ci VALUES (1, 'a', 'a', 'a'), (2, 'A', 'A', 'a '), "
            "(3, 'b', 'b', 'b')",
        ],
    )
    for source in [
        "SELECT s, COUNT(*) AS id FROM review_ci GROUP BY s",
        "SELECT DISTINCT s, id FROM review_ci",
        "SELECT s, id FROM review_ci UNION SELECT s, id FROM review_ci",
    ]:
        _refused(name, source, via="source", why="collation|column 's'")
    # a PAD SPACE binary collation (`utf8mb4_bin`) equates `'a'` and `'a '` too, as a
    # COLLATE clause does
    for source in [
        "SELECT p, COUNT(*) AS id FROM review_ci GROUP BY p",
        "SELECT DISTINCT p, id FROM review_ci",
    ]:
        _refused(name, source, via="source", why="collation|column 'p'")
    # a NO PAD binary collation, a column the query doesn't name, and no collapsing:
    # proven
    for source in [
        "SELECT b, COUNT(*) AS id FROM review_ci GROUP BY b",
        "SELECT id % 2 AS g, COUNT(*) AS id FROM review_ci GROUP BY g",
        "SELECT s, id FROM review_ci",
    ]:
        lf = _scan(
            name,
            _default_driver(name),
            source,
            partition_on="id",
            partition_num=2,
        )
        lf.collect()


def test_postgres_collapsing_over_a_nondeterministic_collation() -> None:
    name = "postgres17"
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_ci",
            "DROP COLLATION IF EXISTS review_ci_coll",
            "CREATE COLLATION review_ci_coll (provider = icu, locale = "
            "'und-u-ks-level2', deterministic = false)",
            "CREATE TABLE review_ci (id int PRIMARY KEY, "
            'n text COLLATE "review_ci_coll", d text)',
            "INSERT INTO review_ci VALUES (1, 'a', 'a'), (2, 'A', 'A'), (3, 'b', 'b')",
        ],
    )
    try:
        for source in [
            "SELECT n, COUNT(*) AS id FROM review_ci GROUP BY n",
            "SELECT DISTINCT n, id FROM review_ci",
        ]:
            _refused(name, source, via="source", why="collation|column 'n'")
        # a table's name is its whole row, collated column included
        _refused(
            name,
            "SELECT DISTINCT r FROM review_ci r",
            via="source",
            why="collation|column 'n'",
        )
        lf = _scan(
            name,
            _default_driver(name),
            "SELECT d, COUNT(*) AS id FROM review_ci GROUP BY d",
            partition_on="id",
            partition_num=2,
        )
        lf.collect()
    finally:
        _execute(
            name,
            [
                "DROP TABLE IF EXISTS review_ci",
                "DROP COLLATION IF EXISTS review_ci_coll",
            ],
        )


def test_postgres_relative_time_text_through_casts() -> None:
    # 'now' reaches a cast to a time by routes that aren't a literal
    name = "postgres17"
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_rt",
            "CREATE TABLE review_rt (id int PRIMARY KEY, ts timestamptz, txt text)",
            "INSERT INTO review_rt SELECT g, now(), 'now' FROM generate_series(0, 299) g",
        ],
    )
    try:
        for cond in [
            "ts < (SELECT 'now')::timestamptz",
            "ts < 'now'::text::timestamptz",
            "ts < ('n' || 'ow')::timestamptz",
            "ts < lower('NOW')::timestamptz",
            "ts < 'yesterday 13:00'::timestamptz",
            "ts < 'today allballs'::timestamptz",
            "ts < (SELECT x FROM (VALUES ('now')) v(x))::timestamptz",
        ]:
            _refused(name, f"SELECT id FROM review_rt WHERE {cond}", via="query")
        _refused(
            name,
            "WITH c AS (SELECT lower('NOW') AS x) SELECT id FROM review_rt, c "
            "WHERE ts < c.x::timestamptz",
            via="query",
        )
        # stored text, cast later (also by a typed constructor: `date(txt)` is
        # `exp.Date`)
        _refused(
            name,
            "SELECT id FROM review_rt WHERE txt::timestamptz > ts",
            via="source",
            why="text column 'txt'",
        )
        for cond in ["date(txt) > ts::date", "txt::date > ts::date"]:
            _refused(
                name,
                f"SELECT id FROM review_rt WHERE {cond}",
                via="source",
                why="text column 'txt'",
            )
        # relative text that doesn't start with the word
        for text in ["today EST", "EST today", "13:00 yesterday", "04:05 tomorrow"]:
            _refused(
                name,
                f"SELECT id FROM review_rt WHERE ts < '{text}'::timestamptz",
                via="query",
            )
        # absolute text, and a cast of a time column, are proven
        for cond in [
            "ts < '2999-01-01 00:00:00'::timestamptz",
            "ts::date < date '2999-01-01'",
            "ts::date <= ts",
        ]:
            assert _partitions(name, f"SELECT id FROM review_rt WHERE {cond}") == list(
                range(300)
            ), cond
    finally:
        _execute(name, ["DROP TABLE IF EXISTS review_rt"])


def test_clickhouse_merge_on_read_engines_need_final() -> None:
    # Replacing/Collapsing/... MergeTree show other rows before and after a background
    # merge; a TTL drops rows in the background
    name = "clickhouse"
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_rep",
            "DROP TABLE IF EXISTS review_exp",
            "DROP TABLE IF EXISTS review_plain",
            "CREATE TABLE review_rep (id UInt32, v UInt32) "
            "ENGINE = ReplacingMergeTree ORDER BY id",
            "INSERT INTO review_rep SELECT number, 1 FROM numbers(300)",
            "INSERT INTO review_rep SELECT number, 2 FROM numbers(300)",
            "CREATE TABLE review_exp (id UInt32, d DateTime) ENGINE = MergeTree "
            "ORDER BY id TTL d + INTERVAL 100 YEAR",
            "INSERT INTO review_exp SELECT number, now() FROM numbers(300)",
            "CREATE TABLE review_plain (id UInt32) ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_plain SELECT number FROM numbers(300)",
            "DROP TABLE IF EXISTS review_expcol",
            "DROP TABLE IF EXISTS review_expcolumn",
            # a column named `ttl` is no TTL; a column's TTL expires its values
            "CREATE TABLE review_expcol (id UInt32, ttl UInt32) ENGINE = MergeTree "
            "ORDER BY id",
            "INSERT INTO review_expcol SELECT number, 1 FROM numbers(300)",
            "CREATE TABLE review_expcolumn (id UInt32, d DateTime, "
            "v UInt32 TTL d + INTERVAL 100 YEAR) ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_expcolumn SELECT number, now(), 1 FROM numbers(300)",
        ],
    )
    try:
        # said so, not "foreign or external table"
        _refused(name, "review_exp", via="source", why=r"\bTTL\b")
        _refused(name, "review_expcolumn", via="source", why=r"\bTTL\b")
        assert _partitions(name, "review_expcol") == list(range(300))
        _refused(name, "review_rep", via="source", why="merge")
        _refused(name, "SELECT id FROM review_rep", via="source", why="FINAL|merge")
        assert _partitions(name, "review_plain") == list(range(300))
        # with FINAL the visible rows don't depend on merges, also after one
        source = "SELECT id, v FROM review_rep FINAL"
        before = _partitions(name, source)
        _execute(name, ["OPTIMIZE TABLE review_rep FINAL"])
        assert _partitions(name, source) == before == list(range(300))
    finally:
        _execute(
            name,
            [
                "DROP TABLE IF EXISTS review_rep",
                "DROP TABLE IF EXISTS review_exp",
                "DROP TABLE IF EXISTS review_plain",
                "DROP TABLE IF EXISTS review_expcol",
                "DROP TABLE IF EXISTS review_expcolumn",
            ],
        )


def test_postgres_user_operators_and_casts_only_count_where_used() -> None:
    # a user's operator or cast elsewhere in the database doesn't make every source
    # unproven; one the SQL uses does, named in the error
    name = "postgres17"
    _ids_table(name, "review_uo_ids")
    _execute(
        name,
        [
            "DROP TYPE IF EXISTS review_uo_t CASCADE",
            "CREATE TYPE review_uo_t AS (a int)",
            "CREATE OR REPLACE FUNCTION review_uo_f(review_uo_t) RETURNS int "
            "LANGUAGE sql VOLATILE AS 'SELECT (random() * 10)::int'",
            "CREATE CAST (review_uo_t AS int) WITH FUNCTION review_uo_f(review_uo_t)",
        ],
    )
    try:
        assert _partitions(name, "review_uo_ids") == list(range(300))
        assert _partitions(name, "SELECT id + 1 AS id FROM review_uo_ids") == list(
            range(1, 301)
        )
        _execute(
            name,
            [
                "DROP TABLE IF EXISTS review_uo_c",
                "CREATE TABLE review_uo_c (id int PRIMARY KEY, v review_uo_t)",
                "INSERT INTO review_uo_c SELECT g, ROW(g)::review_uo_t "
                "FROM generate_series(0, 299) g",
            ],
        )
        _refused(
            name,
            "SELECT id FROM review_uo_c WHERE v::int > -1",
            via="source",
            why="cast .*review_uo_t",
        )
    finally:
        _execute(
            name,
            [
                "DROP TABLE IF EXISTS review_uo_c",
                "DROP TYPE IF EXISTS review_uo_t CASCADE",
            ],
        )


@pytest.mark.parametrize("name", ["postgres17", "postgres14"])
def test_postgres_user_operators_count_by_the_types_they_take(name: str) -> None:
    # LIKE, BETWEEN, IN, CASE, GROUP BY, ORDER BY ... run operators the SQL doesn't
    # spell, `x=-1` merges two; a user's operator counts when one of its operand types
    # is a type of a column the source reads (or one the SQL names). The operator is on
    # a domain over `text`, not on `int` or `text` (which would make every source of the
    # tests running at the same time unproven).
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_op_dom_t CASCADE",
            "DROP TABLE IF EXISTS review_op_ids CASCADE",
            "DROP VIEW IF EXISTS review_op_v",
            "DROP OPERATOR IF EXISTS public.~~ (review_op_dom, review_op_dom)",
            "DROP FUNCTION IF EXISTS review_op_f(review_op_dom, review_op_dom)",
            "DROP DOMAIN IF EXISTS review_op_dom",
            "CREATE DOMAIN review_op_dom AS text",
            "CREATE TABLE review_op_dom_t (id int PRIMARY KEY, s review_op_dom)",
            "INSERT INTO review_op_dom_t SELECT g, g::text FROM generate_series(0, 299) g",
            "CREATE TABLE review_op_ids (id int PRIMARY KEY)",
            "INSERT INTO review_op_ids SELECT g FROM generate_series(0, 299) g",
            "CREATE VIEW review_op_v AS SELECT id, s FROM review_op_dom_t",
            "CREATE FUNCTION review_op_f(review_op_dom, review_op_dom) RETURNS boolean "
            "LANGUAGE sql VOLATILE AS 'SELECT random() < 0.5'",
            "CREATE OPERATOR public.~~ (LEFTARG = review_op_dom, "
            "RIGHTARG = review_op_dom, FUNCTION = review_op_f)",
        ],
    )
    try:
        # `s LIKE '1%'` runs `public.~~(review_op_dom, review_op_dom)`: a volatile
        # user's operator
        _refused(
            name,
            "SELECT id FROM review_op_dom_t WHERE s LIKE '1%'",
            via="source",
            why=r"operator .*~~",
        )
        # (any source over a column of the type: the operator may run for any syntax)
        _refused(name, "review_op_dom_t", via="source", why=r"operator .*~~")
        _refused(name, "review_op_v", via="source", why=r"operator .*~~")
        _refused(
            name,
            "SELECT id FROM review_op_dom_t WHERE id BETWEEN 0 AND 400",
            via="source",
            why=r"operator .*~~",
        )
        # a type the SQL names is a user's: the query itself isn't proven
        _refused(
            name,
            "SELECT id FROM review_op_ids WHERE id::text::review_op_dom IN ('1', '2')",
        )
        # no column of the source, and no type the SQL names, is the domain
        assert _partitions(name, "review_op_ids") == list(range(300))
        assert _partitions(
            name, "SELECT id FROM review_op_ids WHERE id > 5 GROUP BY id"
        ) == list(range(6, 300))
    finally:
        _execute(
            name,
            [
                "DROP VIEW IF EXISTS review_op_v",
                "DROP OPERATOR IF EXISTS public.~~ (review_op_dom, review_op_dom)",
                "DROP FUNCTION IF EXISTS review_op_f(review_op_dom, review_op_dom)",
                "DROP TABLE IF EXISTS review_op_dom_t CASCADE",
                "DROP TABLE IF EXISTS review_op_ids CASCADE",
                "DROP DOMAIN IF EXISTS review_op_dom",
            ],
        )


def test_postgres_citext_domains_and_arrays_collapse_case_insensitively() -> None:
    # `citext` through a domain or an array
    name = "postgres17"
    try:
        _execute(name, ["CREATE EXTENSION IF NOT EXISTS citext"])
    except Exception:
        pytest.skip("no citext extension")
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_cd",
            "DROP DOMAIN IF EXISTS review_cd_dom",
            "CREATE DOMAIN review_cd_dom AS citext",
            "CREATE TABLE review_cd (id int PRIMARY KEY, d review_cd_dom, a citext[], "
            "t text)",
            "INSERT INTO review_cd VALUES (1, 'a', '{a}', 'a'), (2, 'A', '{A}', 'A')",
        ],
    )
    try:
        for source in [
            "SELECT d, COUNT(*) AS id FROM review_cd GROUP BY d",
            "SELECT a, COUNT(*) AS id FROM review_cd GROUP BY a",
        ]:
            _refused(name, source, via="source", why="citext|collation")
        assert _partitions(name, "SELECT t, COUNT(*) AS id FROM review_cd GROUP BY t")
    finally:
        _execute(
            name,
            ["DROP TABLE IF EXISTS review_cd", "DROP DOMAIN IF EXISTS review_cd_dom"],
        )


def test_sqlite_virtual_table_detection_is_by_create_statement(tmp_path: Any) -> None:
    # a column `is_virtual` or a name `virtual_machines` is no virtual table; a VIRTUAL
    # generated column is evaluated on each read
    import sqlite3

    path = tmp_path / "v.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE virtual_machines (id INTEGER PRIMARY KEY, is_virtual INT);
        CREATE TABLE gen_ok (id INTEGER PRIMARY KEY, k INT AS (id % 7) VIRTUAL);
        CREATE TABLE gen_stored (id INTEGER PRIMARY KEY, k INT AS (id % 5) STORED);
        CREATE VIRTUAL TABLE fts USING fts5(body);
        """
    )
    for t in ["virtual_machines", "gen_ok", "gen_stored"]:
        conn.executemany(f"INSERT INTO {t} (id) VALUES (?)", [(i,) for i in range(300)])
    conn.commit()
    conn.close()
    for table in ["virtual_machines", "gen_ok", "gen_stored"]:
        lf = pl.scan_database(
            table, f"sqlite:///{path}", partition_on="id", partition_num=4
        )
        assert sorted(lf.collect()["id"].to_list()) == list(range(300)), table
    for table, why in [("fts", "ordinary")]:
        lf = pl.scan_database(
            table, f"sqlite:///{path}", partition_on="rowid", partition_num=4
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match=why):
            lf.collect()


@pytest.mark.parametrize("name", ["postgres17", "duckdb"])
def test_array_of_a_subquery_isnt_proven(name: str) -> None:
    if name == "duckdb":
        pytest.skip("covered by the unit test")
    _ids_table(name, "review_arr_ids")
    _refused(
        name,
        "SELECT id, ARRAY(SELECT id FROM review_arr_ids) AS a FROM review_arr_ids",
        via="query",
    )
    assert _partitions(
        name, "SELECT id, ARRAY[id, 1] AS a FROM review_arr_ids"
    ) == list(range(300))


# -------------------------------------------------------------------------------------
# SQL Server and Trino
# -------------------------------------------------------------------------------------

needs_mssql = pytest.mark.skipif(
    SERVERS["mssql"].uri is None, reason="no SQL Server URI"
)
needs_trino = pytest.mark.skipif(SERVERS["trino"].uri is None, reason="no Trino URI")


def _read_by_every_driver(
    name: str, source: str, drivers: list[str] | None = None, **kwargs: Any
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """The source read with each runnable driver: the frames, and the errors."""
    frames: dict[str, pl.DataFrame] = {}
    errors: dict[str, str] = {}
    for driver in _runnable_drivers(name, drivers):
        try:
            frames[driver] = _scan(name, driver, source, **kwargs).collect()
        except pl.exceptions.PolarsError as e:
            errors[driver] = f"{type(e).__name__}: {str(e)[:300]}"
    return frames, errors


def _check_filters_by_driver(
    name: str,
    source: str,
    predicates: list[pl.Expr],
    drivers: list[str] | None = None,
) -> None:
    """`check_filter_ids` with each driver; every failing driver is reported."""
    problems = []
    for driver in _runnable_drivers(name, drivers):
        try:
            check_filter_ids(_scan(name, driver, source), predicates)
        except (AssertionError, pl.exceptions.PolarsError) as e:
            problems.append(f"{driver}: {type(e).__name__}: {str(e)[:400]}")
    assert not problems, "\n".join(problems)


def _assert_drivers_agree(
    frames: dict[str, pl.DataFrame], errors: dict[str, str], what: str
) -> None:
    """No driver failed, and all read the same dtypes and values as the first.

    Integer widths are not compared (a driver reading `integer` as Int32 or Int64
    would hide every other difference): see the tests of declared widths.
    """
    problems = [f"{d} failed: {e}" for d, e in errors.items()]
    frames = {
        d: df.with_columns(cs.integer().cast(pl.Int64)) for d, df in frames.items()
    }
    if frames:
        first, *rest = frames
        for driver in rest:
            try:
                assert_frame_equal(frames[driver], frames[first])
            except AssertionError as e:
                problems.append(f"{driver} differs from {first}: {e}")
    assert not problems, f"{what}:\n" + "\n".join(problems)


@needs_mssql
@pytest.mark.parametrize("database", ["polars"])
def test_mssql_adbc_reads_the_database_of_the_uri_path(database: str) -> None:
    # `mssql://user:pw@host:1433/polars` names the database in its path, as it does for
    # the other drivers; the ADBC driver reads a path as an instance name, and ran the
    # query in the login's default database (`master`): "Invalid object name 'edge'"
    pytest.importorskip("adbc_driver_manager")
    uri = SERVERS["mssql"].uri
    assert uri is not None
    if (reason := _no_adbc_driver("mssql")()) is not None:
        pytest.skip(reason)
    assert urlsplit(uri).path == f"/{database}"
    lf = pl.scan_database("edge", uri, driver="adbc")
    assert lf.select(pl.len()).collect().item() == N_EDGE_ROWS


@needs_mssql
def test_mssql_catalog_lookups_work_with_every_driver() -> None:
    # the lookups behind a partitioned read (table statistics, kind of relation) select
    # unnamed columns (`CASE … END`, `SELECT COUNT(*)`), which every driver that names
    # its columns `''` turns into "column '' appears more than once"; the lookup then
    # fails, and the table is refused as "not shown to be an ordinary table"
    _ids_table("mssql", "review_cat_ids")
    refused = {}
    for driver in _runnable_drivers("mssql"):
        lf = _scan(
            "mssql", driver, "review_cat_ids", partition_on="id", partition_num=4
        )
        try:
            assert sorted(lf.collect()["id"].to_list()) == list(range(300))
        except pl.exceptions.InvalidOperationError as e:
            refused[driver] = str(e)[:120]
    assert not refused, refused


@needs_mssql
def test_mssql_adbc_naive_datetimes_stay_naive() -> None:
    # a `datetime2` has no time zone: read by ADBC it was a UTC timestamp
    table = "review_naive_ts"
    _execute(
        "mssql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, a datetime2(6), b datetime, c smalldatetime)",
            f"INSERT INTO {table} VALUES (1, '2020-01-02 03:04:05.123456', "
            "'2020-01-02 03:04:05.003', '2020-01-02 03:04:00')",
        ],
    )
    frames, errors = _read_by_every_driver("mssql", table)
    _assert_drivers_agree(frames, errors, "datetime2 / datetime / smalldatetime")
    for df in frames.values():
        assert df.schema["a"] == pl.Datetime("us")


@needs_mssql
def test_mssql_legacy_and_offset_datetimes() -> None:
    # `datetime` (3.33 ms), `smalldatetime` (minutes), `datetimeoffset(7)` (100 ns, any
    # offset), `datetime2(7)` and `time(7)` (100 ns): read alike by every driver, and a
    # filter on them keeps every row Polars keeps, whatever the database rounds
    table = "review_legacy_dt"
    _execute(
        "mssql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, dt datetime, sdt smalldatetime, "
            "dto datetimeoffset(7), d2 datetime2(7), tm time(7))",
            f"INSERT INTO {table} VALUES "
            "(1, '2020-01-02 03:04:05.003', '2020-01-02 03:04:00', "
            "'2020-01-02 03:04:05.1234567 +05:30', '2020-01-02 03:04:05.1234567', "
            "'03:04:05.1234567'), "
            "(2, '1753-01-01 00:00:00', '1900-01-01 00:00:00', "
            "'0001-01-01 00:00:00 +00:00', '0001-01-01 00:00:00', '00:00:00'), "
            "(3, '9999-12-31 23:59:59.997', '2079-06-06 23:59:00', "
            "'9999-12-31 23:59:59.9999999 +00:00', '9999-12-31 23:59:59.9999999', "
            "'23:59:59.9999999'), "
            "(4, '2020-06-01 12:00:00', '2020-06-01 12:00:00', "
            "'2020-06-01 12:00:00 -08:00', '2020-06-01 12:00:00', '12:00:00'), "
            "(5, NULL, NULL, NULL, NULL, NULL)",
        ],
    )
    frames, errors = _read_by_every_driver("mssql", table)
    _assert_drivers_agree(frames, errors, "legacy date and time columns")
    df = next(iter(frames.values())).sort("id")
    assert df["dt"][0] == datetime(2020, 1, 2, 3, 4, 5, 3000)
    assert df["dto"][0] == datetime(2020, 1, 1, 21, 34, 5, 123456, tzinfo=UTC)
    assert df["dto"][3] == datetime(2020, 6, 1, 20, 0, tzinfo=UTC)
    predicates = [
        pl.col("dt") == datetime(2020, 1, 2, 3, 4, 5, 3000),
        pl.col("dt") == datetime(2020, 1, 2, 3, 4, 5, 3333),
        pl.col("dt") != datetime(2020, 1, 2, 3, 4, 5, 3000),
        pl.col("dt") > datetime(2020, 1, 2, 3, 4, 5, 2000),
        pl.col("dt") <= datetime(1753, 1, 1),
        pl.col("sdt") == datetime(2020, 1, 2, 3, 4),
        pl.col("sdt") != datetime(2020, 1, 2, 3, 4),
        pl.col("sdt") < datetime(2020, 1, 2, 3, 4, 29),
        pl.col("dto") == datetime(2020, 1, 1, 21, 34, 5, 123456, tzinfo=UTC),
        pl.col("dto") != datetime(2020, 1, 1, 21, 34, 5, 123456, tzinfo=UTC),
        pl.col("dto") > datetime(2020, 1, 1, 21, 34, 5, 123456, tzinfo=UTC),
        pl.col("dto") < datetime(2020, 6, 1, 20, 0, 0, 1, tzinfo=UTC),
        pl.col("d2") == datetime(2020, 1, 2, 3, 4, 5, 123456),
        pl.col("d2") != datetime(2020, 1, 2, 3, 4, 5, 123456),
        pl.col("d2") >= datetime(9999, 12, 31, 23, 59, 59, 999999),
        pl.col("tm") == dtime(3, 4, 5, 123456),
        pl.col("tm") != dtime(3, 4, 5, 123456),
        pl.col("tm") >= dtime(23, 59, 59, 999999),
        pl.col("dt").dt.date() == date(2020, 1, 2),
        pl.col("dto").dt.year() == 2020,
        pl.col("d2").dt.hour() == 3,
    ]
    _check_filters_by_driver("mssql", table, predicates)


@needs_mssql
def test_mssql_time_zones_by_every_driver() -> None:
    # `datetimeoffset` values at other offsets are instants, read in UTC; `AT TIME ZONE`
    # turns a `datetime2` or an offset value into another zone's local time
    table = "review_tz_mssql"
    _execute(
        "mssql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, dto datetimeoffset(6), "
            "ts datetime2(6))",
            f"INSERT INTO {table} VALUES "
            "(1, '2026-03-29 01:30:00 +00:00', '2026-03-29 01:30:00'), "
            "(2, '2026-03-29 01:30:00 +05:30', '2026-10-25 02:30:00.5'), "
            "(3, '2026-10-25 02:30:00 -07:00', '1900-01-01 00:00:00'), "
            "(4, NULL, NULL)",
        ],
    )
    for source in [
        table,
        f"SELECT id, dto AT TIME ZONE 'Tokyo Standard Time' AS tokyo, "
        f"ts AT TIME ZONE 'UTC' AS ts_utc FROM {table}",
        f"SELECT id, SWITCHOFFSET(dto, '-05:00') AS so, "
        f"TODATETIMEOFFSET(ts, '+09:00') AS td FROM {table}",
    ]:
        frames, errors = _read_by_every_driver("mssql", source)
        _assert_drivers_agree(frames, errors, source)
        df = next(iter(frames.values())).sort("id")
        assert df["id"].to_list() == [1, 2, 3, 4]
        first = df.columns[1]
        assert df[first][0] == datetime(2026, 3, 29, 1, 30, tzinfo=UTC), source
        assert df[first][1] == datetime(2026, 3, 28, 20, 0, tzinfo=UTC), source
    _check_filters_by_driver(
        "mssql",
        table,
        [
            pl.col("dto") == datetime(2026, 3, 28, 20, 0, tzinfo=UTC),
            pl.col("dto") > datetime(2026, 3, 29, tzinfo=UTC),
            pl.col("dto").dt.hour() == 20,
            pl.col("dto").dt.convert_time_zone("Asia/Tokyo").dt.hour() == 5,
            pl.col("ts") < datetime(2026, 10, 25, 2, 30, 0, 500000),
        ],
    )


@needs_mssql
def test_mssql_collations() -> None:
    # `s` is case and accent insensitive (CI_AS) and ignores trailing spaces; a `_SC`
    # collation counts a supplementary character once, a binary one by code point
    table = "review_coll"
    _execute(
        "mssql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, "
            "ci nvarchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS, "
            "cs nvarchar(20) COLLATE SQL_Latin1_General_CP1_CS_AS, "
            "bin nvarchar(20) COLLATE Latin1_General_100_BIN2, "
            "sc nvarchar(20) COLLATE Latin1_General_100_CI_AS_SC, "
            "ai nvarchar(20) COLLATE Latin1_General_100_CI_AI, "
            "vci varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS, "
            "utf8 varchar(20) COLLATE Latin1_General_100_CI_AS_SC_UTF8)",
        ]
        + [
            f"INSERT INTO {table} VALUES ({i}, N'{v}', N'{v}', N'{v}', N'{v}', "
            f"N'{v}', '{v if v.isascii() else 'x'}', N'{v}')"
            for i, v in enumerate(
                ["abc", "ABC", "Abc", "abc ", "é", "e", "É", "😀", "ß", "ss", "", "a_c"]
            )
        ]
        + [
            f"INSERT INTO {table} VALUES (99, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
        ],
    )
    predicates = []
    for c in ["ci", "cs", "bin", "sc", "ai", "vci", "utf8"]:
        col = pl.col(c)
        predicates += [
            col == "abc",
            col != "abc",
            col == "é",
            col != "é",
            col == "ss",
            col == "😀",
            col < "abd",
            col >= "abc",
            col > "É",
            col.is_in(["abc", "É"]),
            ~col.is_in(["abc", "É"]),
            col.str.starts_with("a"),
            col.str.starts_with("A"),
            col.str.ends_with("c"),
            col.str.contains("b", literal=True),
            col.str.contains("_", literal=True),
            col.str.contains("😀", literal=True),
            col.str.len_chars() == 1,
            col.str.len_chars() == 3,
            col.str.len_bytes() == 3,
            col.str.to_lowercase() == "abc",
            col.str.to_uppercase() == "ABC",
        ]
    _check_filters_by_driver("mssql", table, predicates)


@needs_mssql
def test_mssql_decimals() -> None:
    table = "review_dec_mssql"
    big = "9" * 38
    _execute(
        "mssql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, d38 decimal(38, 0), "
            "d18 decimal(18, 4), dmax decimal(38, 37), dn numeric(20, 10), "
            "m money, sm smallmoney)",
            # one INSERT per row: a multi-row VALUES unifies each column's literal
            # types, and decimal(38, 37) with the literal 0 overflows
            f"INSERT INTO {table} VALUES "
            f"(1, {big}, 12.5, 9.9999999999999999999999999999999999999, "
            "-0.0000000001, 922337203685477.5807, 214748.3647)",
            f"INSERT INTO {table} VALUES "
            "(2, -1, -0.0001, -9.9999999999999999999999999999999999999, "
            "0.0000000001, -922337203685477.5808, -214748.3648)",
            f"INSERT INTO {table} VALUES (3, 0, 0, 0, 0, 0, 0)",
            f"INSERT INTO {table} VALUES (4, NULL, NULL, NULL, NULL, NULL, NULL)",
        ],
    )
    frames, errors = _read_by_every_driver("mssql", table)
    _assert_drivers_agree(frames, errors, "decimals")
    df = next(iter(frames.values())).sort("id")
    assert df["d38"][0] == Decimal(big)
    assert df["dn"][0] == Decimal("-0.0000000001")
    _check_filters_by_driver(
        "mssql",
        table,
        [
            pl.col("d38") == Decimal(big),
            pl.col("d38") > Decimal(big[:-1]),
            pl.col("d38") < Decimal(-1),
            pl.col("d18") == Decimal("12.5"),
            pl.col("d18") < Decimal("0.00005"),
            pl.col("d18") > Decimal("-0.00005"),
            pl.col("d18") == Decimal("12.50001"),
            pl.col("dmax") > Decimal("9.99999999999999999999999999999999999"),
            pl.col("dn") == Decimal("-0.0000000001"),
            pl.col("dn") != Decimal("0.00000000010"),
            pl.col("dn") < Decimal("0.00000000005"),
            pl.col("m").cast(pl.Decimal(19, 4)) > Decimal("0"),
            pl.col("d18").cast(pl.Float64) > 12.0,
            pl.col("d38").cast(pl.Int64) == 0,
        ],
    )


@needs_mssql
@pytest.mark.parametrize(
    ("column", "sql_type", "literal"),
    [
        ("m", "money", "12.3456"),
        ("sm", "smallmoney", "7.5"),
        ("x", "xml", "'<a>1</a>'"),
        ("hid", "hierarchyid", "'/1/2/'"),
        ("geo", "geography", "geography::Point(1, 2, 4326)"),
        ("geom", "geometry", "geometry::Point(1, 2, 0)"),
        ("sv", "sql_variant", "CAST(5 AS int)"),
        ("rv", "rowversion", "DEFAULT"),
        ("r", "real", "1.5"),
        ("tx", "text", "'text'"),
        ("sn", "sysname", "N'name'"),
        ("ui", "uniqueidentifier", "'6F9619FF-8B86-D011-B42D-00C04FC964FF'"),
    ],
)
def test_mssql_exotic_types(column: str, sql_type: str, literal: str) -> None:
    # none of these may fail, and every driver reads a column alike (a type a driver
    # can't read should be cast to text or a plain type in the query it sends)
    table = f"review_exotic_{column}"
    values = f"(1, {literal})" if literal != "DEFAULT" else "(1, DEFAULT)"
    columns = "(id int, v " + sql_type + ")"
    if sql_type == "rowversion":
        columns = "(id int, v rowversion)"
        values = "(1)"
        insert = f"INSERT INTO {table} (id) VALUES {values}"
    else:
        insert = f"INSERT INTO {table} VALUES {values}"
    _execute(
        "mssql",
        [f"DROP TABLE IF EXISTS {table}", f"CREATE TABLE {table} {columns}", insert],
    )
    frames, errors = _read_by_every_driver("mssql", table)
    _assert_drivers_agree(frames, errors, f"{sql_type}")


@needs_mssql
def test_mssql_scan_on_a_live_pymssql_connection_isnt_corrupted_by_another() -> None:
    # pymssql runs one statement at a time per connection (no MARS): a second scan on
    # the same connection ran while the first was reading, and the first got no rows,
    # rows of the other query or "ShapeError: data does not match the number of columns"
    conn = _live("mssql")
    for _ in range(5):
        a = pl.scan_database("edge", conn).filter(pl.col("id") < 5).select("id")
        b = pl.scan_database("edge", conn).filter(pl.col("id") > 55).select("id")
        out_a, out_b = pl.collect_all([a, b])
        assert sorted(out_a["id"].to_list()) == [0, 1, 2, 3, 4]
        assert sorted(out_b["id"].to_list()) == [56, 57, 58, 59]


@needs_mssql
def test_mssql_python_factory() -> None:
    # `mssql_python` (Microsoft's driver) can only be given as a connection or a factory
    mssql_python = pytest.importorskip("mssql_python")
    uri = SERVERS["mssql"].uri
    assert uri is not None
    p = _parts(uri)
    conn = mssql_python.connect(
        f"SERVER={p['host']},{p['port']};DATABASE={p['database']};UID={p['user']};"
        f"PWD={p['password']};Encrypt=yes;TrustServerCertificate=yes"
    )
    try:
        lf = pl.scan_database("edge", conn)
        assert lf.collect().height == N_EDGE_ROWS
        assert_frame_equal(
            lf.collect().sort("id"),
            _reference("mssql", _default_driver("mssql")).sort("id"),
        )
        got = lf.filter(pl.col("s") == "abc").collect()["id"].to_list()
        want = _reference("mssql", _default_driver("mssql")).filter(
            pl.col("s") == "abc"
        )["id"]
        assert sorted(got) == sorted(want.to_list())
    finally:
        conn.close()


@needs_mssql
def test_mssql_scalar_function_over_a_volatile_view_is_refused_for_partitioning() -> (
    None
):
    # a scalar function (user-defined, whatever its labels) in the select list or the
    # partition key; NEWID() and RAND() can't be called in a function, but a view's can
    _execute(
        "mssql",
        [
            "DROP FUNCTION IF EXISTS review_myrand",
            "DROP FUNCTION IF EXISTS review_twice",
            "DROP VIEW IF EXISTS review_fn_v",
            "DROP TABLE IF EXISTS review_fn_ids",
            "CREATE TABLE review_fn_ids (id int PRIMARY KEY)",
            "INSERT INTO review_fn_ids SELECT TOP 300 ROW_NUMBER() OVER "
            "(ORDER BY (SELECT NULL)) - 1 FROM sys.all_objects",
            "CREATE VIEW review_fn_v AS SELECT RAND(CHECKSUM(NEWID())) AS r",
            "CREATE FUNCTION review_myrand() RETURNS float AS BEGIN "
            "RETURN (SELECT r FROM review_fn_v) END",
            "CREATE FUNCTION review_twice(@x int) RETURNS int WITH SCHEMABINDING AS "
            "BEGIN RETURN 2 * @x END",
        ],
    )
    for driver in _runnable_drivers("mssql", ["connectorx"]):
        lf = _scan(
            "mssql",
            driver,
            "SELECT id, dbo.review_myrand() AS r FROM review_fn_ids",
            partition_on="r",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="review_myrand"):
            lf.collect()
        lf = _scan(
            "mssql",
            driver,
            "SELECT id, dbo.review_twice(id) AS d FROM review_fn_ids",
            partition_on="d",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
        lf = _scan("mssql", driver, "review_fn_ids", partition_on="id", partition_num=4)
        assert sorted(lf.collect()["id"].to_list()) == list(range(300))


@needs_mssql
def test_mssql_views_with_volatile_table_functions_filter_after_the_read() -> None:
    # the filter must apply to the values the read returns: a pushed `r > 0.5` over a
    # function that calls NEWID() would be a second, independent draw
    _execute("mssql", VOLATILE_VIEWS["mssql"][0])
    problems: list[str] = []
    for source in [
        "SELECT * FROM review_volatile_itvf()",
        "SELECT * FROM review_volatile_mstvf()",
        "SELECT * FROM review_volatile_syn_f()",
        "review_volatile_syn",
    ]:
        for driver in _runnable_drivers("mssql"):
            try:
                df = _scan("mssql", driver, source).filter(pl.col("r") > 0.5).collect()
                assert (df["r"] > 0.5).all(), (driver, source)
                assert 0 < df.height < 300, (driver, source, df.height)
            except (AssertionError, pl.exceptions.PolarsError) as e:
                problems.append(
                    f"{driver} {source}: {type(e).__name__}: {str(e)[:200]}"
                )
    assert not problems, "\n".join(problems)


@needs_mssql
def test_mssql_edge_types_filters() -> None:
    # filters on the types of `edge_types` (time zone, time of day, text, bytes, GUID,
    # JSON text)
    _check_filters_by_driver(
        "mssql",
        "edge_types",
        [
            pl.col("tstz") == datetime(2026, 3, 29, 1, 30, tzinfo=UTC),
            pl.col("tstz") > datetime(2000, 1, 1, tzinfo=UTC),
            pl.col("t") == dtime(13, 45, 30, 123456),
            pl.col("t") < dtime(1),
            pl.col("bin") == b"\x00\x01\xff",
            pl.col("bin") == b"",
            pl.col("u") == "6F1B6A3E-9B8F-4F0E-8D2A-0C4B1F2E3A4D",
            pl.col("u") == "6f1b6a3e-9b8f-4f0e-8d2a-0c4b1f2e3a4d",
            pl.col("u").is_null(),
            pl.col("j") == "[]",
            pl.col("nu") == Decimal("123.456789012345"),
            pl.col("nu") == Decimal("-0.5"),
        ],
    )


@needs_trino
def test_unproven_dialect_refuses_explicit_partitions_and_reads_auto_once() -> None:
    # Trino has no catalog lookup: a source can't be shown to repeat its rows
    for driver in _runnable_drivers("trino"):
        lf = _scan("trino", driver, "orders", partition_on="id", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
        auto = _scan("trino", driver, "orders", partition_on="auto")
        assert auto.select(pl.len()).collect().item() == 100_000
        assert "partitions: 1" in auto.explain() or "partitions" not in auto.explain()
        assert auto.collect()["id"].n_unique() == 100_000


@needs_trino
def test_trino_orders_read_alike_with_and_without_pushdown() -> None:
    for driver in _runnable_drivers("trino"):
        lf = _scan("trino", driver, "orders")
        off = _scan("trino", driver, "orders", pushdown=False)
        pred = (pl.col("id") > 99_000) & pl.col("customer").str.starts_with("cust_1")
        assert_frame_equal(
            lf.filter(pred).collect().sort("id"), off.filter(pred).collect().sort("id")
        )
        assert lf.select(pl.len()).collect().item() == 100_000


@needs_trino
def test_trino_char_columns_pad_in_comparisons() -> None:
    # `char(3)` compares padded: 'a' = 'a  ' is true in Trino, and the column is read
    # padded, so Polars says the opposite; a pushed `<>` would drop a row Polars keeps
    table = "review_trino_char"
    _execute(
        "trino",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id integer, c char(3), v varchar(3))",
            f"INSERT INTO {table} VALUES (1, 'a', 'a'), (2, 'ab', 'ab '), "
            "(3, 'abc', 'abc'), (4, NULL, NULL), (5, '', ''), (6, ' a', ' a')",
        ],
    )
    _check_filters_by_driver(
        "trino",
        table,
        [
            pl.col("c") == "a",
            pl.col("c") != "a",
            pl.col("c") == "a  ",
            pl.col("c") == "ab",
            pl.col("c") < "abc",
            pl.col("c") >= "ab",
            pl.col("c").is_in(["a", "abc"]),
            ~pl.col("c").is_in(["a", "abc"]),
            pl.col("c").str.starts_with("a"),
            pl.col("c").str.ends_with("a"),
            pl.col("c").str.len_chars() == 3,
            pl.col("v") == "ab",
            pl.col("v") != "ab",
            pl.col("v").str.ends_with(" "),
        ],
    )


@needs_trino
def test_trino_strings_compare_by_code_point() -> None:
    # Trino has no collations: comparisons are by code point (UTF-8 bytes order)
    table = "review_trino_str"
    values = [
        "abc",
        "ABC",
        "Abc",
        "abc ",
        "é",
        "e",
        "É",
        "😀",
        "ß",
        "ss",
        "",
        "a_c",
        "ﬁ",
    ]
    _execute(
        "trino",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id integer, s varchar)",
            f"INSERT INTO {table} VALUES "
            + ", ".join(f"({i}, '{v}')" for i, v in enumerate(values))
            + f", ({len(values)}, NULL)",
        ],
    )
    predicates = []
    col = pl.col("s")
    for v in ["abc", "é", "😀", "ss", "ﬁ", "Z", "￿"]:
        predicates += [col == v, col != v, col < v, col <= v, col > v, col >= v]
    predicates += [
        col.is_in(["abc", "É"]),
        ~col.is_in(["abc", "É"]),
        col.str.starts_with("a"),
        col.str.contains("_", literal=True),
        col.str.contains("😀", literal=True),
        col.str.len_chars() == 1,
        col.str.len_bytes() == 4,
        col.str.to_lowercase() == "abc",
        col.str.to_uppercase() == "ABC",
        col.str.contains("^a.c$"),
    ]
    _check_filters_by_driver("trino", table, predicates)


@needs_trino
def test_trino_decimals() -> None:
    table = "review_dec_trino"
    big = "9" * 38
    _execute(
        "trino",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id integer, d38 decimal(38, 0), d18 decimal(18, 4), "
            "dmax decimal(38, 37), dn decimal(20, 10), d1 decimal(1, 0))",
            f"INSERT INTO {table} VALUES "
            f"(1, DECIMAL '{big}', DECIMAL '12.5', "
            "DECIMAL '9.9999999999999999999999999999999999999', "
            "DECIMAL '-0.0000000001', DECIMAL '9'), "
            "(2, DECIMAL '-1', DECIMAL '-0.0001', "
            "DECIMAL '-9.9999999999999999999999999999999999999', "
            "DECIMAL '0.0000000001', DECIMAL '0'), "
            "(3, DECIMAL '0', DECIMAL '0', DECIMAL '0', DECIMAL '0', DECIMAL '1'), "
            "(4, NULL, NULL, NULL, NULL, NULL)",
        ],
    )
    frames, errors = _read_by_every_driver("trino", table)
    _assert_drivers_agree(frames, errors, "decimals")
    df = next(iter(frames.values())).sort("id")
    assert df["d38"][0] == Decimal(big)
    assert df["dn"][0] == Decimal("-0.0000000001")
    _check_filters_by_driver(
        "trino",
        table,
        [
            pl.col("d38") == Decimal(big),
            pl.col("d38") > Decimal(big[:-1]),
            pl.col("d38") < Decimal(-1),
            pl.col("d18") == Decimal("12.5"),
            pl.col("d18") < Decimal("0.00005"),
            pl.col("d18") == Decimal("12.50001"),
            pl.col("dmax") > Decimal("9.99999999999999999999999999999999999"),
            pl.col("dn") == Decimal("-0.0000000001"),
            pl.col("dn") != Decimal("0.00000000010"),
            pl.col("d1") == Decimal(9),
            pl.col("d18").cast(pl.Float64) > 12.0,
        ],
    )


@needs_trino
def test_trino_time_zones_by_every_driver() -> None:
    # `timestamp with time zone` values carry a zone of their own: every driver reads
    # them as the same instants, in one dtype
    table = "review_tz_trino"
    _execute(
        "trino",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id integer, tz timestamp(6) with time zone, "
            "ts timestamp(6))",
            f"INSERT INTO {table} VALUES "
            "(1, TIMESTAMP '2026-03-29 01:30:00 UTC', TIMESTAMP '2026-03-29 01:30:00'), "
            "(2, TIMESTAMP '2026-03-29 01:30:00 Asia/Tokyo', "
            "TIMESTAMP '2026-10-25 02:30:00.5'), "
            "(3, TIMESTAMP '2026-10-25 02:30:00 America/New_York', "
            "TIMESTAMP '1900-01-01 00:00:00'), "
            "(4, TIMESTAMP '2026-03-29 01:30:00 +05:30', NULL), "
            "(5, NULL, NULL)",
        ],
    )
    for source in [
        table,
        f"SELECT id, tz AT TIME ZONE 'Europe/Paris' AS paris, "
        f"with_timezone(ts, 'Asia/Kolkata') AS kolkata FROM {table}",
    ]:
        frames, errors = _read_by_every_driver("trino", source)
        problems = [f"{d} failed: {e}" for d, e in errors.items()]
        instants = {
            d: df.sort("id")
            .select(
                pl.col(df.columns[1])
                .dt.convert_time_zone("UTC")
                .dt.replace_time_zone(None)
            )[df.columns[1]]
            .to_list()
            for d, df in frames.items()
        }
        first, *rest = instants
        problems.extend(
            f"{d}: {instants[d]} != {first}: {instants[first]}"
            for d in rest
            if instants[d] != instants[first]
        )
        dtypes = {d: str(df.schema[df.columns[1]]) for d, df in frames.items()}
        if len(set(dtypes.values())) > 1:
            problems.append(f"dtypes differ: {dtypes}")
        assert not problems, f"{source}:\n" + "\n".join(problems)
    _check_filters_by_driver(
        "trino",
        table,
        [
            pl.col("tz") == datetime(2026, 3, 28, 16, 30, tzinfo=UTC),
            pl.col("tz") > datetime(2026, 3, 29, tzinfo=UTC),
            pl.col("tz") != datetime(2026, 3, 29, 1, 30, tzinfo=UTC),
            pl.col("tz").dt.hour() == 16,
            pl.col("ts") < datetime(2026, 10, 25, 2, 30, 0, 500000),
            pl.col("ts") >= datetime(2026, 10, 25, 2, 30),
        ],
    )


@needs_trino
@pytest.mark.parametrize(
    ("column", "sql_type", "literal"),
    [
        ("ia", "array(integer)", "ARRAY[1, 2]"),
        ("mp", "map(varchar, integer)", "MAP(ARRAY['a'], ARRAY[1])"),
        ("rw", "row(a integer, b varchar)", "ROW(1, 'x')"),
        ("iv", "interval day to second", "INTERVAL '1 2:03:04' DAY TO SECOND"),
        ("ym", "interval year to month", "INTERVAL '1-2' YEAR TO MONTH"),
        ("ts12", "timestamp(12)", "TIMESTAMP '2020-01-02 03:04:05.123456789012'"),
        ("tz3", "time(3) with time zone", "TIME '03:04:05.123 +05:30'"),
        (
            "tsz9",
            "timestamp(9) with time zone",
            "TIMESTAMP '2020-01-02 03:04:05.123456789 Asia/Tokyo'",
        ),
        ("u", "uuid", "UUID '6f1b6a3e-9b8f-4f0e-8d2a-0c4b1f2e3a4d'"),
        ("r", "real", "REAL '1.5'"),
        ("i", "integer", "42"),
        ("j", "json", "JSON '{\"a\":1}'"),
    ],
)
def test_trino_exotic_types(column: str, sql_type: str, literal: str) -> None:
    # none of these may fail, and every driver reads a column alike
    table = f"review_trino_exotic_{column}"
    _execute(
        "trino",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id integer, v {sql_type})",
            f"INSERT INTO {table} VALUES (1, {literal})",
        ],
    )
    frames, errors = _read_by_every_driver("trino", table)
    _assert_drivers_agree(frames, errors, sql_type)


@needs_trino
def test_trino_declared_integer_width() -> None:
    # `integer` is Int32 for ADBC and `smallint` Int16 for every driver: the Python
    # driver read `integer` as Int64 (see `test_same_schema_for_every_driver`)
    table = "review_trino_ints"
    _execute(
        "trino",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (a tinyint, b smallint, c integer, d bigint)",
            f"INSERT INTO {table} VALUES (1, 2, 3, 4)",
        ],
    )
    schemas = {
        d: dict(_scan("trino", d, table).collect_schema())
        for d in _runnable_drivers("trino")
    }
    first, *rest = schemas
    for d in rest:
        assert schemas[d] == schemas[first], (d, schemas[d], first, schemas[first])


@needs_trino
def test_trino_edge_types_filters() -> None:
    _check_filters_by_driver(
        "trino",
        "edge_types",
        [
            pl.col("tstz") == datetime(2026, 3, 29, 1, 30, tzinfo=UTC),
            pl.col("tstz") > datetime(2000, 1, 1, tzinfo=UTC),
            pl.col("t") == dtime(13, 45, 30, 123456),
            pl.col("t") < dtime(1),
            pl.col("bin") == b"\x00\x01\xff",
            pl.col("bin") == b"",
            pl.col("nu") == Decimal("123.456789012345"),
            pl.col("nu") == Decimal("-0.5"),
            pl.col("j") == "[]",
            pl.col("u") == "6f1b6a3e-9b8f-4f0e-8d2a-0c4b1f2e3a4d",
            pl.col("u") == "6F1B6A3E-9B8F-4F0E-8D2A-0C4B1F2E3A4D",
        ],
    )


@needs_trino
def test_trino_connectorx_reads_a_qualified_table_without_big_integers() -> None:
    # ConnectorX can read what has no schema to set and no bigint beyond 2**53
    pytest.importorskip("connectorx")
    uri = SERVERS["trino"].uri
    assert uri is not None
    lf = pl.scan_database(
        "SELECT id, customer, amount, price FROM memory.default.orders WHERE id <= 1000",
        urlsplit(uri)._replace(path="/memory").geturl(),
        driver="connectorx",
    )
    assert lf.collect().height == 1000


@needs_trino
def test_trino_volatile_view_is_filtered_after_the_read() -> None:
    _execute("trino", VOLATILE_VIEWS["trino"][0])
    for driver in _runnable_drivers("trino"):
        df = (
            _scan("trino", driver, "SELECT * FROM review_volatile_v")
            .filter(pl.col("r") > 0.5)
            .collect()
        )
        assert (df["r"] > 0.5).all(), driver
        assert 0 < df.height < 300, driver


@needs_mssql
def test_mssql_empty_results_keep_their_schema_with_every_driver() -> None:
    # arrow-odbc yields no batch for an empty result: the scan found no schema
    for driver in _runnable_drivers("mssql"):
        df = _scan("mssql", driver, "SELECT * FROM orders WHERE 1 = 0").collect()
        assert df.height == 0, driver
        assert df.width > 1, driver


@needs_mssql
def test_mssql_exec_after_a_table_is_not_run_as_a_write() -> None:
    # sqlglot parses `… FROM c EXEC('…')` as an alias, so the query passed the read-only
    # check, and ran unwrapped (a CTE can't be wrapped on SQL Server) with the write
    # after it: a `DELETE` ran, an `INSERT` up to four times
    victim = "review_ro_victim"
    _execute(
        "mssql",
        [
            f"DROP TABLE IF EXISTS {victim}",
            f"CREATE TABLE {victim} (id int)",
            f"INSERT INTO {victim} VALUES (1), (2), (3)",
        ],
    )
    try:
        for write in (
            f"DELETE FROM {victim}",
            f"INSERT INTO {victim} VALUES (9)",
        ):
            for sql in (
                f"WITH c AS (SELECT id FROM edge) SELECT id FROM c EXEC('{write}')",
                f"WITH c AS (SELECT id FROM edge) SELECT c.id FROM c JOIN edge e "
                f"ON c.id = e.id EXEC('{write}')",
                f"SELECT id FROM (SELECT id FROM edge) EXEC('{write}')",
                f"SELECT id FROM (SELECT id FROM edge) AS d EXECUTE('{write}')",
            ):
                for driver in _runnable_drivers("mssql"):
                    with pytest.raises(Exception, match="only runs read"):
                        _scan("mssql", driver, sql).collect()
        left = _scan("mssql", _default_driver("mssql"), victim).collect()
        assert sorted(left["id"].to_list()) == [1, 2, 3]
    finally:
        _execute("mssql", [f"DROP TABLE IF EXISTS {victim}"])
