"""The `scan_database` arguments: their checks, and how each connection form is used."""

from __future__ import annotations

import gc
import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import PolarsDataType

pytest.importorskip("polars_database")


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, "auto" INTEGER)')
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(i, f"v{i}", i) for i in range(1, 101)]
        )
        conn.commit()
    return path


@pytest.fixture
def uri(db: Path) -> str:
    return f"sqlite:///{db}"


@pytest.mark.parametrize(
    "connection",
    [
        "polars:s3cret@127.0.0.1:5432/polars",  # no scheme
        "postgresql:polars:s3cret@127.0.0.1/polars",  # no `//`
        "user=polars password=s3cret host=h",  # libpq key-value form
    ],
)
def test_bad_connection_string_is_not_echoed(connection: str) -> None:
    with pytest.raises(ValueError, match="neither") as exc:
        pl.scan_database("t", connection)
    assert "s3cret" not in str(exc.value)


def test_odbc_string_in_any_key_order() -> None:
    lf = pl.scan_database(
        "t",
        "Server=h;Driver={ODBC Driver 18 for SQL Server};UID=u;PWD=s3cret",
        schema={"x": pl.Int64},
        dialect="mssql",
        driver="arrow_odbc",
    )
    try:
        plan = lf.explain()
    except pl.exceptions.PolarsError as e:  # the ODBC module may be missing
        plan = str(e)
    assert "s3cret" not in plan


def test_sqlalchemy_uri_with_an_odbc_string_is_a_uri() -> None:
    from polars.io.database._scan import _connection_kind

    uri = "mssql+pyodbc:///?odbc_connect=DRIVER={ODBC Driver 18};SERVER=h"
    assert _connection_kind(uri) == ("uri", uri)


def test_async_sqlalchemy_objects_are_refused() -> None:
    pytest.importorskip("sqlalchemy")
    asyncio = pytest.importorskip("sqlalchemy.ext.asyncio")
    for connection in (asyncio.AsyncSession(), asyncio.async_sessionmaker()):
        with pytest.raises(TypeError, match="sync Engine"):
            pl.scan_database("t", connection)


def test_missing_plugin_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from polars.io.database._scan import _plugin

    monkeypatch.setitem(sys.modules, "polars_database", None)
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        _plugin()


def test_invalid_arguments(uri: str) -> None:
    def scan(**kwargs: Any) -> pl.LazyFrame:
        return pl.scan_database("t", uri, **kwargs)

    invalid: list[dict[str, Any]] = [
        {"max_connections": 0},
        {"batch_size": 0},
        {"consistency": "strict"},
        {"source_type": "view"},
        {"dialect": "not_a_dialect"},
    ]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            scan(**kwargs)
    invalid_types: list[dict[str, Any]] = [
        {"connect_options": [("a", "b")]},
        {"execute_options": [("a", "b")]},
        {"order_by": pl.col("id")},
        {"order_by": [pl.col("id")]},
        {"order_by": 5},
    ]
    for kwargs in invalid_types:
        with pytest.raises(TypeError):
            scan(**kwargs)
    with pytest.raises(TypeError, match="source"):
        pl.scan_database(123, uri)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        pl.scan_database("", uri)
    with pytest.raises(TypeError):
        scan(credential_provider=123)
    with pytest.raises(TypeError, match="cloud storage"):
        scan(credential_provider=pl.CredentialProviderAWS())


def test_integer_arguments_accept_numpy_integers(uri: str) -> None:
    np = pytest.importorskip("numpy")
    lf = pl.scan_database(
        "t",
        uri,
        max_connections=np.int32(2),
        batch_size=np.uint16(10),
    )
    assert lf.collect().height == 100
    with pytest.raises(ValueError):
        pl.scan_database("t", uri, batch_size=np.float64(4))


def test_integer_arguments_have_an_upper_bound(uri: str) -> None:
    with pytest.raises(ValueError, match="must be at most"):
        pl.scan_database("t", uri, batch_size=2**70)
    with pytest.raises(ValueError, match="must be at most"):
        pl.scan_database("t", uri, max_connections=2**70)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"partition_on": "id"},
        {"partition_on": pl.col("id")},
        {"partition_on": pl.col("auto")},
        {"partition_num": 4},
        {"partition_on": None, "partition_num": 4},
        {"partition_on": "id", "partition_num": 4},
        {"partition_on": "id", "partition_range": (1, 100)},
        {"partition_range": (1, 100)},
        {"partition_predicates": ["id < 50", "id >= 50"]},
        {"partition_predicates": []},
    ],
)
def test_partitioning_is_not_supported_yet(uri: str, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="doesn't support partitioning yet"):
        pl.scan_database("t", uri, **kwargs)


@pytest.mark.parametrize("partition_on", ["auto", None])
def test_scan_reads_with_one_query(uri: str, partition_on: str | None) -> None:
    lf = pl.scan_database("t", uri, partition_on=partition_on, max_connections=4)
    assert "partitions: 1" in lf.explain()
    assert lf.collect().height == 100


def test_snapshot_reads_with_one_query(uri: str) -> None:
    lf = pl.scan_database("t", uri, consistency="snapshot")
    assert "partitions: 1" in lf.explain()
    assert lf.collect().height == 100


def test_driver_pins(uri: str) -> None:
    assert pl.scan_database("t", uri, driver="SQLITE3").collect().height == 100
    with pytest.raises(pl.exceptions.InvalidOperationError, match="not a driver for"):
        pl.scan_database("t", uri, driver="psycopg").collect()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="empty"):
        pl.scan_database("t", uri, driver=" ").collect()


def test_typed_options(uri: str) -> None:
    lf = pl.scan_database("t", uri, driver="sqlite3", connect_options={"timeout": 5})
    assert lf.collect().height == 100
    pytest.importorskip("sqlalchemy")
    lf = pl.scan_database(
        "t", uri, driver="sqlalchemy", connect_options={"echo": False}
    )
    assert lf.collect().height == 100


def test_connectorx_rejects_options(uri: str) -> None:
    pytest.importorskip("connectorx")
    lf = pl.scan_database("t", uri, driver="connectorx", execute_options={"x": 1})
    with pytest.raises(pl.exceptions.ComputeError, match="connectorx takes no options"):
        lf.collect()


def test_connection_objects_take_no_connect_settings(db: Path) -> None:
    def factory() -> sqlite3.Connection:
        return sqlite3.connect(db, check_same_thread=False)

    with pytest.raises(ValueError, match="driver"):
        pl.scan_database("t", factory, driver="adbc")
    with pytest.raises(ValueError, match="connect_options"):
        pl.scan_database("t", factory, connect_options={"timeout": 1})
    with (
        closing(factory()) as conn,
        pytest.raises(ValueError, match="credential_provider"),
    ):
        pl.scan_database("t", conn, credential_provider=lambda: {"user": "u"})


def test_live_sqlite3_connection(db: Path) -> None:
    with (
        closing(sqlite3.connect(db)) as conn,
        pytest.raises(ValueError, match="check_same_thread"),
    ):
        pl.scan_database("t", conn)
    with closing(sqlite3.connect(db, check_same_thread=False)) as conn:
        assert pl.scan_database("t", conn).collect().height == 100
    conn = sqlite3.connect(db, check_same_thread=False)
    conn.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        pl.scan_database("t", conn)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_collect_leaves_no_factory_running(db: Path, fail_at: int) -> None:
    lock = threading.Lock()
    calls = {"bad": 0, "slow": 0}
    running = 0

    def bad() -> sqlite3.Connection:
        with lock:
            calls["bad"] += 1
            if calls["bad"] >= fail_at:
                msg = "bad connection"
                raise RuntimeError(msg)
        return sqlite3.connect(db, check_same_thread=False)

    def slow() -> sqlite3.Connection:
        nonlocal running
        with lock:
            calls["slow"] += 1
            running += 1
            n = calls["slow"]
        try:
            if n >= fail_at:
                time.sleep(0.5)
            return sqlite3.connect(db, check_same_thread=False)
        finally:
            with lock:
                running -= 1

    schema = {"id": pl.Int64, "s": pl.String}
    args: dict[str, Any] = {"schema": schema}
    a = pl.scan_database("t", bad, **args)
    b = pl.scan_database("t", slow, **args)
    with pytest.raises(pl.exceptions.ComputeError, match="bad connection"):
        a.join(b, on="id").collect()
    with lock:
        assert running == 0
        n_calls = calls["slow"]
    time.sleep(1)
    assert calls["slow"] == n_calls


def test_stop_cancels_with_saturated_blocking_pool(tmp_path: Path) -> None:
    # Set the pool size before Polars initializes its runtime. Cancellation needs its
    # own thread even when the only fetch worker is blocked inside the database.
    code = f"""
import runpy
from pathlib import Path
tests = runpy.run_path({__file__!r})
tests["_stop_with_saturated_blocking_pool"](Path({str(tmp_path)!r}))
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "POLARS_MAX_BLOCKING_THREAD_COUNT": "1"},
        check=True,
        timeout=30,
    )


def _stop_with_saturated_blocking_pool(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(1, 1001)])
        conn.commit()

    cancelled = threading.Event()

    class Connection(sqlite3.Connection):
        def interrupt(self) -> None:
            cancelled.set()
            super().interrupt()

    # read as a `sqlite3` connection
    Connection.__module__ = "sqlite3"

    rows = 0
    blocked_at = 0.0

    def first_rows(_: int) -> int:
        # The read fails the query with its first batch, but it blocks
        # in the database on a later row (the driver hands a batch over once it has read
        # the next row), holding the only fetch thread.
        nonlocal rows, blocked_at
        rows += 1
        if rows > 6:
            blocked_at = blocked_at or time.monotonic()
            cancelled.wait(5)
        return 1

    def factory() -> sqlite3.Connection:
        conn = sqlite3.connect(path, factory=Connection, check_same_thread=False)
        conn.create_function("first_rows", 1, first_rows)
        return conn

    lf = pl.scan_database(
        "SELECT id FROM t WHERE first_rows(id) = 1",
        factory,
        schema={"id": pl.Int64},
        max_connections=1,
        batch_size=5,
    )

    # Fail the query natively (a Python callback or UDF would itself need a thread of
    # the saturated pool), but late: the read must reach its blocking
    # row first, as it does right after handing the first batch over. The failing cast
    # comes after a per-row sum over millions of values, which takes much longer.
    heavy = pl.col("id").repeat_by(2_000_000).list.sum()
    q = lf.select(heavy.cast(pl.UInt8, strict=True))
    with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion"):
        q.collect(engine="streaming")
    # The only fetch thread is blocked in the database: only the cancellation, which
    # cannot run on that pool, lets the query return.
    assert blocked_at
    assert cancelled.is_set()
    assert time.monotonic() - blocked_at < 3


def test_sqlalchemy_engine_connects_through_its_pool() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.pool import StaticPool

    # an in-memory database that only this engine's single connection holds
    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE m (id INTEGER)")
        conn.exec_driver_sql("INSERT INTO m VALUES (1), (2)")
    try:
        assert pl.scan_database("m", engine).collect()["id"].to_list() == [1, 2]
    finally:
        engine.dispose()


def test_in_memory_sqlite_engine_advice() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")

    # SingletonThreadPool, one connection per thread, refused on other threads: a URI
    # would open another, empty database, so the advice is to share the connection
    engine = sqlalchemy.create_engine("sqlite://")
    assert type(engine.pool).__name__ == "SingletonThreadPool"
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE m (id INTEGER)")
        with pytest.raises(ValueError, match="in-memory") as exc:
            pl.scan_database("m", engine)
        assert "check_same_thread" in str(exc.value)
        assert "pass a URI" not in str(exc.value)
    finally:
        engine.dispose()

    # the advice works
    engine = sqlalchemy.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}
    )
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE m (id INTEGER)")
            conn.exec_driver_sql("INSERT INTO m VALUES (1), (2)")
        assert pl.scan_database("m", engine).collect()["id"].to_list() == [1, 2]
    finally:
        engine.dispose()


def test_assertion_pool_reads_the_checked_out_connection() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.pool import AssertionPool

    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=AssertionPool,
        connect_args={"check_same_thread": False},
    )
    try:
        # a second checkout raises: the scan reads the user's connection, uncommitted
        # rows included
        with engine.connect() as conn:
            conn.exec_driver_sql("CREATE TABLE m (id INTEGER)")
            conn.exec_driver_sql("INSERT INTO m VALUES (1), (2)")
            out = pl.scan_database("m", engine).collect()
            assert out["id"].to_list() == [1, 2]
        # and without a checkout, through the pool
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE n (id INTEGER)")
            conn.exec_driver_sql("INSERT INTO n VALUES (3)")
        assert pl.scan_database("n", engine).collect()["id"].to_list() == [3]
    finally:
        engine.dispose()


def test_credential_provider_without_user_or_password(uri: str) -> None:
    lf = pl.scan_database(
        "t", uri, driver="sqlite3", schema={"id": pl.Int64}, credential_provider=dict
    )
    with pytest.raises(pl.exceptions.ComputeError, match="neither 'user' nor"):
        lf.collect()


class _Id(int):
    """An int the plugin can't key by value."""


def test_params_are_taken_when_the_scan_is_built(uri: str) -> None:
    # a changed container merged two scans, and execute_options' parameters were keyed
    # by their text only
    query = "SELECT id FROM t WHERE id = ?"
    params: list[Any] = [_Id(1)]
    first = pl.scan_database(query, uri, params=params, driver="sqlite3")
    params[0] = _Id(2)
    second = pl.scan_database(query, uri, params=params, driver="sqlite3")
    assert pl.concat([first, second]).collect()["id"].to_list() == [1, 2]
    options: dict[str, Any] = {"parameters": [3]}
    third = pl.scan_database(query, uri, execute_options=options, driver="sqlite3")
    options["parameters"] = [4]
    fourth = pl.scan_database(query, uri, execute_options=options, driver="sqlite3")
    assert pl.concat([third, fourth]).collect()["id"].to_list() == [3, 4]
    # any Mapping, also one that can't be copied
    proxy = pl.scan_database(
        "SELECT id FROM t WHERE id = :x",
        uri,
        params=MappingProxyType({"x": 5}),
        driver="sqlite3",
    )
    assert proxy.collect()["id"].to_list() == [5]
    with pytest.raises(ValueError, match="once"):
        pl.scan_database(
            query,
            uri,
            params=[1],
            execute_options={"parameters": [1]},
            driver="sqlite3",
        )


def test_driver_options_are_copied_when_the_scan_is_built(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a list or dict inside the options, changed after the call, changed later collects
    import polars_database

    seen: list[Any] = []
    real = polars_database.scan_resolver

    def spy(**kwargs: Any) -> Any:
        seen.append(kwargs["python_options"])
        return real(**kwargs)

    monkeypatch.setattr(polars_database, "scan_resolver", spy)
    connect: dict[str, Any] = {"nested": {"a": [1]}}
    execute: dict[str, Any] = {"nested": [{"b": 1}]}
    pl.scan_database(
        "t",
        uri,
        driver="sqlite3",
        connect_options=connect,
        execute_options=execute,
    )
    connect["nested"]["a"].append(2)
    execute["nested"][0]["b"] = 2
    assert seen == [({"nested": {"a": [1]}}, {"nested": [{"b": 1}]})]


class _Row(list):  # type: ignore[type-arg]
    """A list whose constructor takes something else."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__([a, b])


def test_params_with_container_subclasses_are_left_as_they_are(uri: str) -> None:
    row = _Row(1, 2)
    lf = pl.scan_database(
        "SELECT id FROM t WHERE id IN (?, ?)", uri, params=row, driver="sqlite3"
    )
    assert sorted(lf.collect()["id"].to_list()) == [1, 2]
    lf = pl.scan_database(
        "SELECT id FROM t WHERE id = ?", uri, params=[_Row(1, 2)[0]], driver="sqlite3"
    )
    assert lf.collect()["id"].to_list() == [1]


def test_params_keep_percent_literals_in_polars(uri: str) -> None:
    lf = pl.scan_database(
        "SELECT id, s FROM t WHERE id < ?", uri, params=[50], driver="sqlite3"
    ).filter(pl.col("s").str.starts_with("v1"), pl.col("s") != "50%")
    plan = lf.explain()
    assert "LIKE" not in plan
    assert "'50%'" not in plan
    assert sorted(lf.collect()["id"]) == [1, *range(10, 20)]


def test_arrow_type_polars_cannot_read_is_an_error(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT union_value(a := 1) AS u")
    lf = pl.scan_database("SELECT u FROM t", f"duckdb:///{path}")
    with pytest.raises(pl.exceptions.ComputeError, match="can't read"):
        lf.collect()


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_count_of_a_slice_of_a_source_beyond_idxsize(
    tmp_path: Path, engine: Any
) -> None:
    # the whole source's count was checked against IdxSize before the slice
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    duckdb.connect(path).close()
    query = "SELECT range AS id FROM range(5000000000)"
    lf = pl.scan_database(query, f"duckdb:///{path}")
    assert lf.head(3).select(pl.len()).collect(engine=engine).item() == 3
    assert lf.slice(4_999_999_998).select(pl.len()).collect(engine=engine).item() == 2
    # a negative slice was resolved from the count narrowed to IdxSize
    assert lf.tail(2).select(pl.len()).collect(engine=engine).item() == 2
    assert lf.slice(-3, 2).select(pl.len()).collect(engine=engine).item() == 2
    # the in-memory engine read (and numbered) the whole source below the row index
    out = lf.with_row_index().slice(3, 2).collect(engine=engine)
    assert out.rows() == [(3, 3), (4, 4)]


class ScanTestExtension(pl.datatypes.BaseExtension):
    """A custom extension dtype for testing scan schemas."""


@pytest.mark.parametrize("name", ["schema", "schema_overrides"])
@pytest.mark.parametrize(
    "dtype",
    [
        pl.Object,
        pl.List(pl.Object),
        pl.Array(pl.Object, 2),
        pl.Struct({"child": pl.Object}),
        pl.List(pl.Array(pl.Struct({"child": pl.List(pl.Object)}), 2)),
        pl.Map(pl.String, pl.Object),
        pl.Map(pl.Object, pl.String),
        pl.Struct({"child": pl.Map(pl.String, pl.List(pl.Object))}),
        pl.Map(pl.Struct({"child": pl.Object}), pl.String),
        pl.Extension("testing.scan_object", pl.Object),
        pl.Extension("testing.scan_struct_object", pl.Struct({"child": pl.Object})),
        pl.List(ScanTestExtension("testing.scan_list_object", pl.Object)),
    ],
)
def test_object_dtype_is_refused(uri: str, name: str, dtype: Any) -> None:
    # Nested Objects used to read silently as Binary, or panic during Arrow export.
    kwargs: dict[str, Any] = {name: {"s": dtype}}
    with pytest.raises(TypeError, match=f"{name}: column 's' can't be read as") as exc:
        pl.scan_database("t", uri, **kwargs)
    assert str(dtype) in str(exc.value)


@pytest.mark.parametrize("name", ["schema", "schema_overrides"])
@pytest.mark.parametrize(
    "dtype",
    [
        pl.List(pl.Array(pl.Struct({"child": pl.List(pl.String)}), 2)),
        pl.Map(pl.String, pl.List(pl.Int64)),
    ],
)
def test_valid_nested_scan_dtype(name: str, dtype: Any) -> None:
    # Constructing a lazy scan with both schemas avoids DB I/O while still exporting
    # the requested nested dtype to Arrow, and preserves it in the output schema.
    kwargs: dict[str, Any] = {"schema": {"s": dtype}, name: {"s": dtype}}
    lf = pl.scan_database("t", "sqlite:///unused-nested-schema.db", **kwargs)
    assert lf.collect_schema() == {"s": dtype}


@pytest.mark.parametrize("name", ["schema", "schema_overrides"])
def test_valid_scan_extension_dtype(name: str) -> None:
    dtype = ScanTestExtension("testing.scan_extension", pl.String)
    pl.register_extension_type(dtype.ext_name(), ScanTestExtension)
    try:
        kwargs: dict[str, Any] = {"schema": {"s": pl.String}, name: {"s": dtype}}
        lf = pl.scan_database("t", "sqlite:///unused-extension-schema.db", **kwargs)
        assert lf.collect_schema() == {"s": dtype}
    finally:
        pl.unregister_extension_type(dtype.ext_name())


@pytest.mark.parametrize(
    "call",
    [
        'lf.group_by("id").map_groups(lambda df: df, schema=None)',
        'lf.rolling(index_column=pl.col("id") * 1, period="2i")',
        'lf.group_by_dynamic(index_column=pl.col("id") * 1, every="2i")',
    ],
)
def test_plan_building_resolves_the_schema_without_the_gil(db: Path, call: str) -> None:
    # The plugin needs the GIL to find the schema: building these plans held it, and
    # waited on the plugin forever. In a subprocess, so that a hang fails the test.
    code = f"""
import polars as pl
lf = pl.scan_database("t", "sqlite:///" + {str(db)!r})
{call}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize("unbuffered", [True, False])
def test_factory_returning_one_pymysql_connection_twice_is_refused(
    unbuffered: bool,
) -> None:
    # PyMySQL ends an unbuffered result silently when another statement runs on its
    # connection: two reads can't stream from one. A session opens an unbuffered
    # cursor on any connection of the factory, so a default (buffered) one is refused
    # too
    pymysql = pytest.importorskip("pymysql")
    from polars_database._bridge import _Session

    # no server: only the connection's class and cursor class matter
    conn = pymysql.connections.Connection.__new__(pymysql.connections.Connection)
    conn.cursorclass = (
        pymysql.cursors.SSCursor if unbuffered else pymysql.cursors.Cursor
    )

    kw: dict[str, Any] = {"execute_options": {}, "params": None}
    first = _Session(conn, owned=True, factory=True, **kw)
    try:
        with pytest.raises(ValueError, match="new connection on each call"):
            _Session(conn, owned=True, factory=True, **kw)
    finally:
        first._release_factory_connection()
    # once the first is closed, the connection may be returned again
    _Session(conn, owned=True, factory=True, **kw)._release_factory_connection()


def test_factory_returning_one_connection_twice_is_refused_any_driver() -> None:

    from polars_database._bridge import _FACTORY_CONNECTIONS, _Session

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    kw: dict[str, Any] = {"execute_options": {}, "params": None}
    first = _Session(conn, owned=True, factory=True, **kw)
    with pytest.raises(ValueError, match="new connection on each call"):
        _Session(conn, owned=True, factory=True, **kw)
    first._release_factory_connection()

    # a session dropped without `close()` doesn't keep the connection refused
    dropped = _Session(conn, owned=True, factory=True, **kw)
    assert id(conn) in _FACTORY_CONNECTIONS
    del dropped
    gc.collect()
    assert id(conn) not in _FACTORY_CONNECTIONS
    _Session(conn, owned=True, factory=True, **kw)._release_factory_connection()
    conn.close()


def test_connection_info_leaves_an_open_sessions_connection_open() -> None:
    from polars_database._bridge import _Session, connection_info

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    kw: dict[str, Any] = {"execute_options": {}, "params": None}
    session = _Session(conn, owned=True, factory=True, **kw)
    cur = conn.execute("SELECT 1 UNION ALL SELECT 2")
    assert cur.fetchone() == (1,)
    # the factory returns the connection the session holds: it can still be read from
    connection_info(lambda: conn, "factory")
    assert cur.fetchone() == (2,)
    assert conn.execute("SELECT 3").fetchone() == (3,)
    session._release_factory_connection()

    # one no session holds is closed, as ever
    connection_info(lambda: conn, "factory")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 4")


def test_session_keeps_its_connection_until_the_driver_close_returns() -> None:
    from polars_database._bridge import _FACTORY_CONNECTIONS, _Session

    started, gate = threading.Event(), threading.Event()

    class Slow(sqlite3.Connection):
        fail = False

        def close(self) -> None:
            started.set()
            gate.wait(timeout=30)
            super().close()
            if self.fail:
                msg = "close failed"
                raise RuntimeError(msg)

    kw: dict[str, Any] = {"execute_options": {}, "params": None}
    conn = sqlite3.connect(":memory:", check_same_thread=False, factory=Slow)
    first = _Session(conn, owned=True, factory=True, **kw)
    closer = threading.Thread(target=first.close)
    closer.start()
    try:
        assert started.wait(timeout=30)
        # still held while the driver closes it: a second session is refused
        with pytest.raises(ValueError, match="new connection on each call"):
            _Session(conn, owned=True, factory=True, **kw)
    finally:
        gate.set()
        closer.join(timeout=30)
    assert id(conn) not in _FACTORY_CONNECTIONS

    # a close that fails still releases it
    started.clear()
    gate.set()
    conn = sqlite3.connect(":memory:", check_same_thread=False, factory=Slow)
    conn.fail = True
    session = _Session(conn, owned=True, factory=True, **kw)
    with pytest.raises(RuntimeError, match="close failed"):
        session.close()
    assert id(conn) not in _FACTORY_CONNECTIONS


def test_session_releases_a_pooled_connection_before_giving_it_back() -> None:
    from polars_database._bridge import _FACTORY_CONNECTIONS, _Session

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    held_at_return: list[bool] = []

    def back_to_the_pool() -> None:
        # once it is in a pool, another reader may be handed it and open a session
        held_at_return.append(id(conn) in _FACTORY_CONNECTIONS)

    kw: dict[str, Any] = {"execute_options": {}, "params": None}
    session = _Session(conn, owned=True, close=back_to_the_pool, factory=True, **kw)
    session.close()
    assert held_at_return == [False]
    assert id(conn) not in _FACTORY_CONNECTIONS

    # a failing return still releases it
    def fails() -> None:
        msg = "pool closed"
        raise RuntimeError(msg)

    session = _Session(conn, owned=True, close=fails, factory=True, **kw)
    with pytest.raises(RuntimeError, match="pool closed"):
        session.close()
    assert id(conn) not in _FACTORY_CONNECTIONS
    conn.close()


def _fake(module: str, name: str = "Obj", **attrs: Any) -> Any:
    """An instance of a class that claims to be from a driver module."""
    return type(name, (), {"__module__": module, **attrs})()


def test_databricks_empty_result_is_typed_from_the_description() -> None:
    # #384, #422: the description is ambiguous (`TIMESTAMP_NTZ` reads `timestamp`,
    # `TIME` `string`): ambiguous types stay `Null` (typed from values), decimals have
    # their precision
    from polars_database._bridge import _empty_frame

    description = [
        ("id", "bigint", None, None, None, None, None),
        ("g", "int", None, None, None, None, None),
        ("s", "string", None, None, None, None, None),
        ("t", "timestamp", None, None, None, None, None),
        ("n", "timestamp", None, None, None, None, None),
        ("d", "decimal", None, None, 10, 2, None),
        ("u", "decimal", None, None, None, None, None),
        ("a", "array", None, None, None, None, None),
    ]
    cursor = _fake("databricks.sql.client", "Cursor", description=description)
    expected: dict[str, Any] = {
        "id": pl.Int64,
        "g": pl.Int32,
        "s": pl.Null,
        "t": pl.Null,
        "n": pl.Null,
        "d": pl.Decimal(10, 2),
        "u": pl.Null,
        "a": pl.Null,
    }
    assert _empty_frame(cursor, utc=True).schema == pl.Schema(expected)


def test_empty_result_description_dtypes_of_duckdb_and_pymysql_bit() -> None:
    # #438: DuckDB's nested types, and pymysql's BIT (bytes, as with rows)
    from polars_database._bridge import _empty_frame

    description = [
        ("id", "INTEGER", None, None, None, None, None),
        ("st", "STRUCT(a INTEGER, b VARCHAR[])", None, None, None, None, None),
        ("l", "BIGINT[]", None, None, None, None, None),
    ]
    cursor = _fake("duckdb", "DuckDBPyConnection", description=description)
    expected: dict[str, Any] = {
        "id": pl.Null,  # (a scalar column is typed from the catalog)
        "st": pl.Struct({"a": pl.Int32, "b": pl.List(pl.String)}),
        "l": pl.List(pl.Int64),
    }
    assert _empty_frame(cursor, utc=True).schema == pl.Schema(expected)
    bit = [("b", 16, None, None, None, None, True)]
    cursor = _fake("pymysql.cursors", "Cursor", description=bit)
    assert _empty_frame(cursor, utc=True).schema == pl.Schema({"b": pl.Binary})


def test_trino_empty_result_is_typed_from_the_description() -> None:
    # a precision above 6 is read as text, an instant as UTC
    from polars_database._bridge import _empty_frame

    types: dict[str, tuple[str, PolarsDataType]] = {
        "i": ("bigint", pl.Int64),
        "d": ("decimal(10, 2)", pl.Decimal(10, 2)),
        "ts": ("timestamp(3)", pl.Datetime("us")),
        "tz": ("timestamp(6) with time zone", pl.Datetime("us", "UTC")),
        "ts9": ("timestamp(9)", pl.String),
        "tz9": ("timestamp(9) with time zone", pl.String),
        "t": ("time(3)", pl.Time),
        "t9": ("time(9)", pl.String),
        "c": ("char(3)", pl.String),
        "u": ("uuid", pl.Null),
    }
    description = [(n, t, None, None, None, None, None) for n, (t, _) in types.items()]
    cursor = _fake("trino.dbapi", "Cursor", description=description)
    expected = {n: dtype for n, (_, dtype) in types.items()}
    assert _empty_frame(cursor, utc=True).schema == pl.Schema(expected)


def test_trino_batch_of_nulls_takes_the_described_dtype() -> None:
    from polars_database._bridge import _cast_null_columns

    frames = iter([pl.DataFrame({"a": [None], "b": [None], "c": [1]})])
    schema = pl.Schema({"a": pl.Int32, "b": pl.Null, "c": pl.Int64})
    (out,) = _cast_null_columns(frames, schema)
    assert out.schema == pl.Schema({"a": pl.Int32, "b": pl.Null, "c": pl.Int64})


@pytest.mark.parametrize(
    ("dbms", "dialect"),
    [("Microsoft SQL Server", "mssql"), ("PostgreSQL", "")],
)
def test_pyodbc_connection_of_sql_server_is_found(dbms: str, dialect: str) -> None:
    from polars_database._bridge import _connection_info

    conn = _fake("pyodbc", "Connection", getinfo=lambda self, key: dbms)
    assert _connection_info(conn)[0] == dialect


def test_databricks_empty_result_takes_the_drivers_arrow_schema() -> None:
    # #384, #422: an exhausted result still has the Arrow schema of its rows
    import pyarrow as pa
    from polars_database._bridge import _empty_frame

    names = ["i", "ntz", "ltz", "d", "a", "t"]
    table = pa.table(
        {
            "i": pa.array([], pa.int32()),
            "ntz": pa.array([], pa.timestamp("us")),
            "ltz": pa.array([], pa.timestamp("us", "America/Los_Angeles")),
            "d": pa.array([], pa.decimal128(10, 2)),
            "a": pa.array([], pa.list_(pa.int32())),
            "t": pa.array([], pa.time64("ns")),
        }
    )
    description = [(n, "string", None, None, None, None, None) for n in names]
    cursor = _fake(
        "databricks.sql.client",
        "Cursor",
        description=description,
        fetchmany_arrow=lambda _self, size: table.slice(0, size),
    )
    frame = _empty_frame(cursor, utc=True)
    expected: dict[str, Any] = {
        "i": pl.Int32,
        "ntz": pl.Datetime("us"),
        "ltz": pl.Datetime("us", "America/Los_Angeles"),
        "d": pl.Decimal(10, 2),
        "a": pl.List(pl.Int32),
        "t": pl.Time,
    }
    assert frame.schema == pl.Schema(expected)
    assert frame.height == 0

    # a failing fetch, or other columns: the description's types stay
    def fails(_self: Any, size: int) -> Any:
        raise RuntimeError

    broken = _fake(
        "databricks.sql.client",
        "Cursor",
        description=description,
        fetchmany_arrow=fails,
    )
    assert _empty_frame(broken, utc=True).schema == (
        pl.Schema(dict.fromkeys(names, pl.Null))
    )


def _transaction_idle_of(conn: Any) -> bool | None:
    from polars_database._bridge import _transaction_idle

    return _transaction_idle(conn)


def test_mariadb_connector_transaction_status() -> None:
    # #411: the status flags of the last result (SERVER_STATUS_IN_TRANS is 1)
    in_trans = _fake("mariadb.connections", "Connection", server_status=1 | 2)
    idle = _fake("mariadb.connections", "Connection", server_status=2)
    assert _transaction_idle_of(in_trans) is False
    assert _transaction_idle_of(idle) is True


class _FakeMySQLdbCursor:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.closed = False

    def execute(self, sql: str) -> None:
        assert sql == "SELECT @@in_transaction"
        if isinstance(self.result, Exception):
            raise self.result

    def fetchone(self) -> Any:
        return self.result

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("result", "idle"),
    [((1,), False), ((0,), True), (RuntimeError("Unknown system variable"), None)],
)
def test_mysqldb_transaction_status(result: Any, idle: bool | None) -> None:
    # mysqlclient has no status attribute: a MariaDB server answers `@@in_transaction`,
    # a MySQL one errors and the status is unknown
    cursor = _FakeMySQLdbCursor(result)
    conn = _fake("MySQLdb.connections", "Connection", cursor=lambda _self: cursor)
    assert _transaction_idle_of(conn) is idle
    assert cursor.closed


def test_arrow_odbc_refuses_params_and_execute_options() -> None:
    # #412: they were dropped silently
    from polars_database._bridge import _open

    kind, target = "odbc:arrow_odbc", "Driver={x};Server=h"
    sessions: list[dict[str, Any]] = [
        {"params": [5], "execute_options": {}},
        {"params": None, "execute_options": {"timeout": 5}},
    ]
    for refused in sessions:
        with pytest.raises(ValueError, match="arrow_odbc takes no"):
            _open(kind, "odbc", target, None, {}, "mssql", refused)
    ok: dict[str, Any] = {"params": None, "execute_options": {}}
    assert _open(kind, "odbc", target, None, {}, "mssql", ok) is not None


def test_psycopg_pool_connection_goes_back_with_putconn() -> None:
    # #418: `close()` of a `getconn()` connection doesn't give the slot back
    from polars_database._bridge import _pool_checkout

    class ConnectionPool:
        __module__ = "psycopg_pool.pool"

        def __init__(self) -> None:
            self.returned: list[Any] = []

        def putconn(self, conn: Any) -> None:
            self.returned.append(conn)

    class Connection:
        __module__ = "psycopg"

        def __init__(self, pool: Any) -> None:
            self._pool = pool
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pool = ConnectionPool()
    conn = Connection(pool)
    driver, close, invalidate = _pool_checkout(conn)
    assert driver is conn
    assert close is not None
    assert invalidate is not None
    close()
    assert pool.returned == [conn]
    assert not conn.closed
    invalidate()
    assert pool.returned == [conn, conn]
    assert conn.closed

    # psycopg_pool 3.2 `close_returns=True`: `close()` returns it itself, once
    pool2 = ConnectionPool()
    pool2._close_returns = True  # type: ignore[attr-defined]
    conn2 = Connection(pool2)
    _, _, invalidate2 = _pool_checkout(conn2)
    assert invalidate2 is not None
    invalidate2()
    assert conn2.closed
    assert pool2.returned == []

    # #426: the public attribute (psycopg_pool 3.3) is the one that counts
    pool3 = ConnectionPool()
    pool3.close_returns = True  # type: ignore[attr-defined]
    conn3 = Connection(pool3)
    _, _, invalidate3 = _pool_checkout(conn3)
    assert invalidate3 is not None
    invalidate3()
    assert conn3.closed
    assert pool3.returned == []

    # another pool's (oracledb, mysql.connector) connection is left to `close()`
    other = Connection(_fake("oracledb.pool", "ConnectionPool"))
    assert _pool_checkout(other) == (other, None, None)


def test_adbc_release_closes_the_statement_before_a_concurrent_close() -> None:
    # #416: `close()` found `_open` empty while `release` was still closing the
    # statement, and closed the connection under it
    from polars_database._bridge import _AdbcSession

    events: list[str] = []
    in_stmt_close = threading.Event()
    go = threading.Event()

    class Stmt:
        def close(self) -> None:
            events.append("stmt.close")
            in_stmt_close.set()
            go.wait(5)
            events.append("stmt.closed")

    class Closer:
        def close(self) -> None:
            events.append("closer.close")

    session: Any = _AdbcSession.__new__(_AdbcSession)
    session._lock = threading.Lock()
    session._handle_lock = threading.Lock()
    session._open = []
    session._closed = False
    session._issuing = False
    session._conn = Closer()
    session._db = Closer()
    stmt = Stmt()
    session._open.append(stmt)

    releaser = threading.Thread(target=session.release, args=(stmt,))
    releaser.start()
    assert in_stmt_close.wait(5)
    closer = threading.Thread(target=session.close)
    closer.start()
    time.sleep(0.2)
    go.set()
    releaser.join(5)
    closer.join(5)
    assert events == [
        "stmt.close",
        "stmt.closed",
        "closer.close",
        "closer.close",
    ]
