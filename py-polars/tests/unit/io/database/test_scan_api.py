"""The `scan_database` arguments: their checks, and how each connection form is used."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

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
        {"partition_num": 0},
        {"partition_num": True},
        {"partition_num": -1},
        {"partition_num": 2.0},
        {"partition_num": "2"},
        {"max_connections": 0},
        {"batch_size": 0},
        {"partition_on": None, "partition_num": 4},
        {"partition_on": None, "partition_range": (1, 10)},
        {"partition_range": (1, 10)},  # with "auto"
        {"partition_predicates": ["id < 5"], "partition_on": "id"},
        {"partition_predicates": ["id < 5"], "partition_num": 2},
        {"partition_on": "id", "partition_range": (100, 1)},
        {"consistency": "strict"},
        {"source_type": "view"},
        {"dialect": "not_a_dialect"},
        {"pushdown": False, "partition_predicates": ["id < 5", "id >= 5"]},
        {"pushdown": False, "partition_on": "id"},
        {"pushdown": False, "partition_num": 2},
        {"partition_predicates": ["id < 5", "id >= 5"], "consistency": "snapshot"},
    ]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            scan(**kwargs)
    invalid_types: list[dict[str, Any]] = [
        # a str would be split into one predicate per character
        {"partition_predicates": "id < 5"},
        {"partition_on": 5},
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
        partition_num=np.int64(4),
        max_connections=np.int32(2),
        batch_size=np.uint16(10),
    )
    assert lf.collect().height == 100
    with pytest.raises(ValueError):
        pl.scan_database("t", uri, partition_num=np.float64(4))


def test_integer_arguments_have_an_upper_bound(uri: str) -> None:
    for kwargs in (
        {"partition_on": "id", "partition_num": 65537},
        {"batch_size": 2**70},
        {"max_connections": 2**70},
    ):
        with pytest.raises(ValueError, match="must be at most"):
            pl.scan_database("t", uri, **kwargs)  # type: ignore[arg-type]
    pl.scan_database("t", uri, partition_on="id", partition_num=65536)


def test_partition_iterables_are_materialized(uri: str) -> None:
    lf = pl.scan_database(
        "t",
        uri,
        partition_on="id",
        partition_range=(x for x in (1, 100)),  # type: ignore[arg-type]
    )
    assert lf.collect().height == 100
    lf = pl.scan_database(
        "t",
        uri,
        partition_predicates=(p for p in ("id <= 50", "id > 50")),  # type: ignore[arg-type]
    )
    assert lf.collect().height == 100
    lf = pl.scan_database(
        "t",
        uri,
        partition_predicates=(p for p in ("id > 0",)),  # type: ignore[arg-type]
        consistency="snapshot",
    )
    assert lf.collect().height == 100
    with pytest.raises(ValueError, match="snapshot"):
        pl.scan_database(
            "t",
            uri,
            partition_predicates=(p for p in ("id <= 50", "id > 50")),  # type: ignore[arg-type]
            consistency="snapshot",
        )
    with pytest.raises(TypeError, match="partition_range must be a sequence"):
        pl.scan_database("t", uri, partition_on="id", partition_range=5)  # type: ignore[arg-type]


def test_partition_range_must_fit_the_column(uri: str) -> None:
    lf = pl.scan_database(
        "t",
        uri,
        partition_on="id",
        partition_range=(date(2020, 1, 1), date(2021, 1, 1)),
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="numeric"):
        lf.collect()


def test_partition_on_column_named_auto(uri: str) -> None:
    lf = pl.scan_database("t", uri, partition_on=pl.col("auto"), partition_num=4)
    assert 'range on "auto"' in lf.explain()
    assert lf.collect().height == 100


def test_snapshot_reads_with_one_query(uri: str) -> None:
    lf = pl.scan_database("t", uri, consistency="snapshot", partition_num=4)
    assert "partitions: 1" in lf.explain()
    assert lf.collect().height == 100
    one = pl.scan_database(
        "t", uri, partition_predicates=["id < 50"], consistency="snapshot"
    )
    assert one.collect().height == 49


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


# the factory's first two calls find the schema, the third plans the partitions
@pytest.mark.parametrize("fail_at", [1, 3])
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
    args: dict[str, Any] = {"schema": schema, "partition_on": "id", "partition_num": 2}
    a = pl.scan_database("t", bad, **args)
    b = pl.scan_database("t", slow, **args)
    with pytest.raises(pl.exceptions.ComputeError, match="bad connection"):
        a.join(b, on="id").collect()
    with lock:
        assert running == 0
        n_calls = calls["slow"]
    time.sleep(1)
    assert calls["slow"] == n_calls


@pytest.mark.parametrize("stop", ["head", "error"])
def test_stop_cancels_a_partition_waiting_for_its_first_rows(
    tmp_path: Path, stop: str
) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(i, i % 7) for i in range(1, 1001)]
        )
        conn.commit()

    cancelled = threading.Event()

    class Connection(sqlite3.Connection):
        def interrupt(self) -> None:
            cancelled.set()
            super().interrupt()

    # read as a `sqlite3` connection
    Connection.__module__ = "sqlite3"

    waited = threading.Lock()
    waiting = threading.Event()

    def wait_for_cancel() -> int:
        # the first row of the second partition blocks until the read is cancelled
        if waited.acquire(blocking=False):
            waiting.set()
            cancelled.wait(5)
        return 1

    first = threading.Lock()

    def second_is_waiting() -> int:
        # the first partition answers once the second one is blocked, if it starts
        if first.acquire(blocking=False):
            waiting.wait(2)
        return 1

    def factory() -> sqlite3.Connection:
        conn = sqlite3.connect(path, factory=Connection, check_same_thread=False)
        conn.create_function("wait_for_cancel", 0, wait_for_cancel)
        conn.create_function("second_is_waiting", 0, second_is_waiting)
        return conn

    lf = pl.scan_database(
        "t",
        factory,
        schema={"id": pl.Int64, "a": pl.Int64},
        partition_predicates=[
            "id <= 500 AND second_is_waiting() = 1",
            "id > 500 AND wait_for_cancel() = 1",
        ],
    )
    start = time.monotonic()
    # the first partition alone answers the query, or fails it (an early stop of the
    # streaming engine; the in-memory engine reads the whole scan first)
    if stop == "head":
        q = lf.filter(pl.col("a") == 1).head(3)
        assert q.collect(engine="streaming")["id"].to_list() == [1, 8, 15]
    else:
        q = lf.select(pl.col("id").cast(pl.UInt8, strict=True))
        with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion"):
            q.collect(engine="streaming")
    # a statement blocked in the second partition was cancelled, and none starts later
    assert cancelled.is_set() or not waiting.is_set()
    assert time.monotonic() - start < 4
    time.sleep(0.5)
    assert cancelled.is_set() or not waiting.is_set()


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
        # The first partition fails the query with its first batch, but its read blocks
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
        "t",
        factory,
        schema={"id": pl.Int64},
        partition_predicates=["first_rows(id) = 1 AND id <= 500", "id > 500"],
        # one read at a time: the second partition waits for the first
        max_connections=1,
        batch_size=5,
    )
    # Delay the failure, so the first partition's read reaches the blocking row first.
    q = lf.select(
        pl.col("id")
        .map_batches(lambda s: (time.sleep(0.5), s)[1], return_dtype=pl.Int64)
        .mul(100)
        .cast(pl.UInt8, strict=True)
    )
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


def test_query_that_picks_rows_is_not_partitioned(uri: str) -> None:
    for query in [
        "SELECT * FROM t LIMIT 10",
        "SELECT * FROM t ORDER BY id LIMIT 10 OFFSET 5",
        "SELECT * FROM t WHERE id > (SELECT id FROM t LIMIT 1)",
        "SELECT id, random() AS r FROM t",
    ]:
        lf = pl.scan_database(query, uri, partition_on="id", partition_num=2)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
        assert pl.scan_database(query, uri).collect().height > 0


def test_arrow_type_polars_cannot_read_is_an_error(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT INTERVAL 1 DAY AS iv")
    lf = pl.scan_database("SELECT iv FROM t", f"duckdb:///{path}")
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
    assert lf.tail(2).collect(engine=engine)["id"].to_list() == [
        4_999_999_998,
        4_999_999_999,
    ]
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
