from __future__ import annotations

import io
import os
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.database._scan_corpus import check_filter_ids, planned_sql

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import PolarsDataType

pytest.importorskip("polars_database")
pytest.importorskip("sqlglot")

N_ROWS = 1000


def _rows(n: int) -> list[tuple[Any, ...]]:
    return [
        (
            i,
            f"c{i % 7}",
            None if i % 11 == 0 else i * 1.5,
            "paid" if i % 3 else "open",
            f"2026-01-{1 + i % 28:02d}",
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE orders ("
        "id INTEGER PRIMARY KEY, customer TEXT, amount REAL, status TEXT, day TEXT)"
    )
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _rows(N_ROWS))
    conn.execute("CREATE TABLE empty (a INTEGER, b TEXT)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def shop_uri(shop_db: Path) -> str:
    return f"sqlite:///{shop_db}"


@pytest.fixture
def expected(shop_db: Path) -> pl.DataFrame:
    conn = sqlite3.connect(shop_db)
    try:
        return pl.read_database("SELECT * FROM orders", conn)
    finally:
        conn.close()


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns)


@pytest.mark.parametrize(
    "driver", [None, "adbc", "sqlite3", "connectorx", "sqlalchemy"]
)
def test_scan_table_drivers(
    shop_uri: str, expected: pl.DataFrame, driver: str | None
) -> None:
    pytest.importorskip(
        {"connectorx": "connectorx", "sqlalchemy": "sqlalchemy"}.get(
            driver or "", "sqlite3"
        )
    )
    if driver in (None, "adbc"):
        pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri, driver=driver)
    assert lf.collect_schema() == expected.schema
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))


@pytest.mark.parametrize(
    ("predicate", "exact"),
    [
        (pl.col("status") == "paid", True),
        (pl.col("status") != "paid", True),
        (pl.col("id") > 500, True),
        (pl.col("id").is_between(10, 20), True),
        (pl.col("id").is_between(10, 20, closed="none"), True),
        (pl.col("customer").is_in(["c1", "c3"]), True),
        (pl.col("customer").is_in(["c1", None], nulls_equal=True), True),
        (pl.col("amount").is_null(), True),
        (pl.col("amount").is_not_null() & (pl.col("amount") < 30), True),
        ((pl.col("id") < 10) | (pl.col("id") > 990), True),
        (~(pl.col("id") < 990), True),
        # not pushed on SQLite: its LIKE stops at a NUL
        (pl.col("customer").str.starts_with("c1"), False),
        (pl.col("customer").str.ends_with("3"), False),
        (pl.col("customer").str.contains("_", literal=True), False),
        (pl.col("amount").ne_missing(3.0), True),
        (pl.col("amount").eq_missing(None), True),
        # not pushed: arithmetic is not in tier 1
        ((pl.col("id") * 2) > 100, False),
        # partly pushed: the AND keeps its translatable side as a pre-filter
        ((pl.col("id") > 900) & ((pl.col("id") % 2) == 0), False),
    ],
)
def test_scan_filter_pushdown(
    shop_uri: str, expected: pl.DataFrame, predicate: pl.Expr, exact: bool
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri).filter(predicate)
    out = lf.collect()
    assert_frame_equal(_sorted(out), _sorted(expected.filter(predicate)))
    plan = lf.explain()
    if exact:
        assert "WHERE" in planned_sql(lf)
        assert "(1 exact)" in plan or "exact)" in plan


def test_scan_partitioning(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "orders", shop_uri, partition_on="id", partition_num=4, max_connections=2
    )
    plan = lf.explain()
    assert "partitions: 4 (range on" in plan
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))

    # bounds use the pushed filters, and every row lands in exactly one partition
    q = lf.filter(pl.col("id") > 700)
    assert_frame_equal(
        _sorted(q.collect()), _sorted(expected.filter(pl.col("id") > 700))
    )


def test_scan_partition_column_with_nulls(
    shop_uri: str, expected: pl.DataFrame
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri, partition_on="amount", partition_num=3)
    assert lf.explain().count("range on") == 1
    out = lf.collect()
    assert out.height == N_ROWS
    assert out["amount"].null_count() == expected["amount"].null_count()


def test_scan_partition_range_and_predicates(
    shop_uri: str, expected: pl.DataFrame
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    # the range only sets the stride: rows outside it still land in the end partitions
    lf = pl.scan_database(
        "orders",
        shop_uri,
        partition_on="id",
        partition_num=4,
        partition_range=(400, 600),
    )
    assert "bounds from partition_range" in lf.explain()
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))

    lf = pl.scan_database(
        "orders",
        shop_uri,
        partition_predicates=["id % 2 = 0", "id % 2 = 1"],
    )
    assert "2 (partition_predicates)" in lf.explain()
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))


def test_scan_order_by(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "orders", shop_uri, order_by="id", partition_on="id", partition_num=4
    )
    assert "ORDER BY" in planned_sql(lf)
    assert_frame_equal(lf.collect(), expected.sort("id"))


def test_scan_slice_and_count(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri)
    assert "LIMIT 5" in planned_sql(lf.head(5))
    assert lf.head(5).collect().height == 5
    assert lf.slice(10, 5).collect().height == 5
    assert lf.select(pl.len()).collect().item() == N_ROWS
    assert lf.filter(pl.col("status") == "paid").select(pl.len()).collect().item() == (
        N_ROWS - N_ROWS // 3
    )


def test_scan_row_index_keeps_filters_in_polars(
    shop_uri: str, expected: pl.DataFrame
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = (
        pl.scan_database("orders", shop_uri, order_by="id", partition_on=None)
        .with_row_index()
        .filter(pl.col("id") > 990)
    )
    out = lf.collect()
    assert out["index"].to_list() == list(range(990, 1000))


def test_scan_query(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "SELECT id, customer, amount FROM orders WHERE status = 'paid'", shop_uri
    ).filter(pl.col("id") < 50)
    assert (
        "FROM (SELECT id, customer, amount FROM orders WHERE status = 'paid')"
        in planned_sql(lf)
    )
    exp = expected.filter(pl.col("status") == "paid", pl.col("id") < 50).select(
        "id", "customer", "amount"
    )
    assert_frame_equal(_sorted(lf.collect()), _sorted(exp))


def test_scan_query_without_pushdown(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    # duplicate output names can't be wrapped, nor held in a DataFrame
    lf = pl.scan_database("SELECT id, id FROM orders", shop_uri)
    with pytest.raises(pl.exceptions.DuplicateError):
        lf.collect_schema()

    # a query that sqlglot can't parse runs as given
    lf = pl.scan_database(
        "SELECT id FROM orders WHERE id < 3 /* trailing */",
        shop_uri,
        source_type="query",
    )
    assert lf.collect()["id"].to_list() == [1, 2]

    lf = pl.scan_database("orders", shop_uri, pushdown=False).filter(pl.col("id") < 3)
    assert "pushdown: off (pushdown=False)" in lf.explain()
    assert lf.collect()["id"].to_list() == [1, 2]


def test_scan_user_schema_and_overrides(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    schema = {"id": pl.Int32, "customer": pl.String}
    lf = pl.scan_database("SELECT id, customer FROM orders", shop_uri, schema=schema)
    assert lf.collect_schema() == pl.Schema(schema)
    assert lf.collect().schema == pl.Schema(schema)

    lf = pl.scan_database("orders", shop_uri, schema_overrides={"id": pl.UInt16})
    assert lf.collect_schema()["id"] == pl.UInt16
    assert lf.collect()["id"].dtype == pl.UInt16


def test_scan_empty_table(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    out = pl.scan_database("empty", shop_uri).collect()
    assert out.height == 0
    assert out.columns == ["a", "b"]


def test_scan_connection_factory_and_live(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT range AS id, range % 5 AS g, "
            "DATE '2026-01-01' + CAST(range AS INTEGER) AS d FROM range(100)"
        )

    def factory() -> Any:
        return duckdb.connect(path, read_only=True)

    lf = pl.scan_database("t", factory).filter(
        pl.col("g") == 2, pl.col("d") >= date(2026, 2, 1)
    )
    assert "python:duckdb" in lf.explain()
    out = lf.collect()
    assert sorted(out["id"].to_list()) == [i for i in range(31, 100) if i % 5 == 2]

    live = duckdb.connect(path, read_only=True)
    try:
        lf = pl.scan_database("t", live).select("id").filter(pl.col("id") < 3)
        assert sorted(lf.collect()["id"].to_list()) == [0, 1, 2]
    finally:
        live.close()

    # a URI picks the duckdb module too
    out = pl.scan_database("t", f"duckdb:///{path}").select(pl.len()).collect()
    assert out.item() == 100


def test_scan_live_connection_is_not_serializable(shop_db: Path) -> None:
    conn = sqlite3.connect(shop_db, check_same_thread=False)
    try:
        lf = pl.scan_database("orders", conn, schema={"id": pl.Int64})
        with pytest.raises(Exception, match="live connection"):
            lf.serialize()
    finally:
        conn.close()


def test_scan_serialize_roundtrip(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri).filter(pl.col("id") <= 10)
    buf = io.BytesIO(lf.serialize())
    out = pl.LazyFrame.deserialize(buf).collect()
    assert_frame_equal(_sorted(out), _sorted(expected.filter(pl.col("id") <= 10)))


def test_scan_secrets_not_in_explain(shop_db: Path) -> None:
    lf = pl.scan_database(
        "orders",
        # a host that can't resolve fails at once; a refused connect is retried
        "postgresql://user:hunter2@nohost.invalid:1/db",
        schema={"id": pl.Int64},
        driver="sqlalchemy",
    )
    with pytest.raises(Exception) as exc:
        lf.collect()
    assert "hunter2" not in str(exc.value)


def test_scan_errors(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    with pytest.raises(Exception, match="not installed"):
        pl.scan_database(
            "orders", shop_uri, driver="python:not_a_real_module"
        ).collect()
    with pytest.raises(Exception, match="range partitioning"):
        pl.scan_database(
            "orders", shop_uri, partition_on="customer", partition_num=2
        ).collect()
    with pytest.raises(ValueError, match="connection URI"):
        pl.scan_database("orders", "not a uri")
    with pytest.raises(Exception, match="snapshot"):
        pl.scan_database(
            "orders",
            shop_uri,
            partition_on="id",
            partition_num=2,
            consistency="snapshot",
        ).collect()


def test_show_drivers() -> None:
    df = pl.io.database.show_drivers("sqlite")
    assert df.columns == [
        "dialect",
        "backend",
        "available",
        "version",
        "location",
        "install",
    ]
    assert "python:sqlite3" in df["backend"].to_list()
    assert df.filter(pl.col("backend") == "python:sqlite3")["available"].item()
    assert pl.io.database.show_drivers("postgresql://u@h/db")[
        "dialect"
    ].unique().to_list() == ["postgresql"]


def test_read_database_uri_adbc_manifest_driver(
    shop_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an ADBC driver installed with `dbc` (a manifest, not a pip package) is found
    adbc_sqlite = pytest.importorskip("adbc_driver_sqlite")
    pytest.importorskip("adbc_driver_manager")
    manifest_dir = tmp_path / "drivers"
    manifest_dir.mkdir()
    library = adbc_sqlite._driver_path()
    (manifest_dir / "sqlite.toml").write_text(
        f'name = "SQLite"\nversion = "1.0.0"\n[Driver]\nshared = "{library}"\n'
    )
    monkeypatch.setenv("ADBC_DRIVER_PATH", str(manifest_dir))

    import polars.io.database._utils as utils

    real_import: Callable[..., Any] = utils._import_optional_adbc_driver

    def no_pip_driver(module_name: str, **kwargs: Any) -> Any:
        if module_name == "adbc_driver_sqlite":
            msg = "ADBC 'adbc_driver_sqlite' driver not detected."
            raise ModuleNotFoundError(msg)
        return real_import(module_name, **kwargs)

    monkeypatch.setattr(utils, "_import_optional_adbc_driver", no_pip_driver)
    out = pl.read_database_uri(
        "SELECT COUNT(*) AS n FROM orders", f"sqlite:///{shop_db}", engine="adbc"
    )
    assert out.item() == N_ROWS


# The differential corpus on SQLite and DuckDB, the in-process databases, on every PR.
# The server variant is in `test_scan_servers.py`.

EDGE_OVERRIDES: dict[str, PolarsDataType] = {
    "dec": pl.Decimal(18, 4),
    "d": pl.Date,
    "ts": pl.Datetime("us"),
    "b": pl.Boolean,
}


@pytest.fixture(scope="module")
def edge_sqlite(tmp_path_factory: pytest.TempPathFactory) -> str:
    from tests.unit.io.database._scan_corpus import edge_rows

    path = tmp_path_factory.mktemp("edge") / "edge.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE edge (id INTEGER PRIMARY KEY, i16 INTEGER, i32 INTEGER, "
        "i64 INTEGER, f64 REAL, dec TEXT, s TEXT, sb TEXT, d TEXT, ts TEXT, b INTEGER)"
    )
    rows = [
        tuple(str(v) if isinstance(v, (Decimal, date)) else v for v in row)
        for row in edge_rows()
    ]
    conn.executemany("INSERT INTO edge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


@pytest.fixture(scope="module")
def edge_duckdb(tmp_path_factory: pytest.TempPathFactory) -> str:
    duckdb = pytest.importorskip("duckdb")
    from tests.unit.io.database._scan_corpus import edge_frame

    path = str(tmp_path_factory.mktemp("edge") / "edge.duckdb")
    frame = edge_frame()
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE edge AS SELECT * FROM frame")
    return path


# as `_scan_corpus.CORPUS_MODES`, which is imported lazily
_CORPUS_MODES = ["table", "query", "table-p4", "query-p4"]


def _corpus_scan(
    source_uri: Any, driver: str | None, mode: str, **kwargs: Any
) -> pl.LazyFrame:
    source = "SELECT * FROM edge" if mode.startswith("query") else "edge"
    partitions: dict[str, Any] = (
        {"partition_on": "id", "partition_num": 4}
        if mode.endswith("p4")
        else {"partition_on": None}
    )
    return pl.scan_database(source, source_uri, driver=driver, **partitions, **kwargs)


def _cell_corpus(
    mode: str,
    cell: int,
    scan: Callable[[str | None], pl.LazyFrame],
    base_scan: Callable[[str | None], pl.LazyFrame] | None,
) -> list[pl.Expr]:
    """
    The corpus of a cell (see `_scan_corpus.cell_predicates`).

    `scan(source)` scans the "table" or "query" source, or the whole table without
    pushdown for None; `base_scan` is the base cell's, None on the base cell. The base
    cell's table runs the whole corpus; its query is compared with it.
    """
    from tests.unit.io.database._scan_corpus import (
        FULL,
        SAMPLE,
        CorpusPlans,
        cell_predicates,
        corpus_for_mode,
    )

    source = mode.removesuffix("-p4")
    if FULL or (base_scan is None and source == "table"):
        return corpus_for_mode(mode)

    def plans(scan: Callable[[str | None], pl.LazyFrame], source: str) -> CorpusPlans:
        return CorpusPlans.of(scan(source), scan(None).collect())

    base = plans(scan, "table") if base_scan is None else plans(base_scan, source)
    return cell_predicates(
        mode, plans(scan, source), base, sample=SAMPLE[mode], offset=cell
    )


def _corpus_params(cells: list[str]) -> list[Any]:
    """(cell, mode); outside the full run, partitioned modes on the base cell only."""
    full = bool(os.environ.get("POLARS_TEST_DATABASE_FULL"))
    return [
        pytest.param(cell, mode, id=f"{cell}-{mode}")
        for cell in cells
        for mode in _CORPUS_MODES
        if full or cell == cells[0] or not mode.endswith("p4")
    ]


# the first is the base cell
_SQLITE_DRIVERS = ["sqlite3", "adbc", "connectorx", "sqlalchemy"]


@pytest.mark.parametrize(("driver", "mode"), _corpus_params(_SQLITE_DRIVERS))
def test_corpus_sqlite(edge_sqlite: str, driver: str, mode: str) -> None:
    from tests.unit.io.database._scan_corpus import check_corpus, ids

    pytest.importorskip(
        {"adbc": "adbc_driver_sqlite", "sqlite3": "sqlite3"}.get(driver, driver)
    )

    def scan(driver: str, mode: str | None) -> pl.LazyFrame:
        # mode None: the whole table, without pushdown
        if mode is None:
            return pl.scan_database(
                "edge",
                edge_sqlite,
                driver=driver,
                pushdown=False,
                schema_overrides=EDGE_OVERRIDES,
            )
        return _corpus_scan(edge_sqlite, driver, mode, schema_overrides=EDGE_OVERRIDES)

    reference = scan(driver, None).collect()
    assert ids(reference) == list(range(60))
    predicates = _cell_corpus(
        mode,
        _SQLITE_DRIVERS.index(driver),
        lambda m: scan(driver, m),
        None if driver == "sqlite3" else lambda m: scan("sqlite3", m),
    )
    lf = scan(driver, mode)
    failures, _ = check_corpus(lambda: lf, reference, predicates)
    assert not failures, "\n".join(failures[:40])


# DuckDB refuses connections to one file with different configurations at once: the
# base cell connects per query, read-only like the live connection
_DUCKDB_CONNECTIONS = ["factory", "uri", "live"]


@pytest.mark.parametrize(("connection", "mode"), _corpus_params(_DUCKDB_CONNECTIONS))
def test_corpus_duckdb(edge_duckdb: str, connection: str, mode: str) -> None:
    import duckdb

    from tests.unit.io.database._scan_corpus import check_corpus, ids

    def connect() -> Any:
        # one thread: the base cell connects per query, and a pool of threads per
        # connection is costly next to the other test workers
        return duckdb.connect(edge_duckdb, read_only=True, config={"threads": 1})

    live = connect() if connection == "live" else None
    connections: dict[str, Any] = {
        "uri": f"duckdb:///{edge_duckdb}",
        "factory": connect,
        "live": live,
    }

    def scan(connection: str, mode: str | None) -> pl.LazyFrame:
        conn = connections[connection]
        if mode is None:
            return pl.scan_database("edge", conn, pushdown=False, partition_on=None)
        return _corpus_scan(conn, None, mode)

    try:
        reference = scan(connection, None).collect()
        assert ids(reference) == list(range(60))
        predicates = _cell_corpus(
            mode,
            _DUCKDB_CONNECTIONS.index(connection),
            lambda m: scan(connection, m),
            None if connection == "factory" else lambda m: scan("factory", m),
        )
        lf = scan(connection, mode)
        failures, _ = check_corpus(lambda: lf, reference, predicates)
        assert not failures, "\n".join(failures[:40])
    finally:
        if live is not None:
            live.close()


def test_property_sqlite(edge_sqlite: str) -> None:
    """Random predicate trees on SQLite; failing seeds belong in the fixed corpus."""
    pytest.importorskip("hypothesis")
    pytest.importorskip("adbc_driver_sqlite")
    from hypothesis import HealthCheck, given, settings

    from tests.unit.io.database._scan_corpus import (
        expected_ids,
        ids,
        predicate_strategy,
    )

    reference = pl.scan_database(
        "edge", edge_sqlite, pushdown=False, schema_overrides=EDGE_OVERRIDES
    ).collect()
    lf = pl.scan_database("edge", edge_sqlite, schema_overrides=EDGE_OVERRIDES)

    @settings(
        max_examples=int(os.environ.get("POLARS_TEST_DATABASE_EXAMPLES", "50")),
        deadline=None,
        suppress_health_check=list(HealthCheck),
        database=None,
    )
    @given(predicate_strategy())
    def check(p: pl.Expr) -> None:
        expected = expected_ids(reference, p)
        if isinstance(expected, Exception):
            with pytest.raises(Exception):  # noqa: B017
                lf.filter(p).select("id").collect()
            return
        assert ids(lf.filter(p).select("id").collect()) == expected, str(p)

    check()


# Pushdown and planning regressions.


def test_scan_overridden_column_is_not_pushed(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT, r REAL, a TEXT, b TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?)",
            [
                (1, "5", 1.0, "2020-1-5", "2020-01-10"),
                (2, "10", 1.7, "2020-01-01", "2020-01-02"),
            ],
        )
        conn.commit()
    lf = pl.scan_database(
        "t",
        f"sqlite:///{path}",
        schema_overrides={"s": pl.Int64, "r": pl.Int64, "a": pl.Date, "b": pl.Date},
    )
    table = lf.collect()
    for p in [pl.col("s") > 5, pl.col("r") == 1, pl.col("a") < pl.col("b")]:
        assert_frame_equal(lf.filter(p).collect(), table.filter(p))


def test_scan_decimal_literal_is_not_rounded(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 1.50::DECIMAL(10,2)), "
            "(2, 2.00::DECIMAL(10,2))) v(id, d)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    table = lf.collect()
    for p in [
        pl.col("d") < pl.lit(Decimal("1.501")),
        pl.col("d") > pl.lit(Decimal("1.499")),
        pl.col("d") == pl.lit(Decimal("1.5")),
    ]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


def test_scan_enum_column_and_infinite_partition_bounds(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TYPE mood AS ENUM ('a', 'b')")
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'::mood, 'inf'::DOUBLE), "
            "(2, 'b'::mood, '-inf'::DOUBLE), (3, 'a'::mood, 0.0)) v(id, e, f)"
        )
    uri = f"duckdb:///{path}"
    for pushdown in [True, False]:
        lf = pl.scan_database("t", uri, pushdown=pushdown)
        out = lf.filter(pl.col("e") == "a").collect()
        assert sorted(out["id"].to_list()) == [1, 3]
    lf = pl.scan_database("t", uri, partition_on="f", partition_num=3)
    assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3]


def test_scan_query_with_trailing_comment(shop_uri: str) -> None:
    lf = pl.scan_database("SELECT * FROM orders WHERE id <= 10 -- note", shop_uri)
    assert lf.filter(pl.col("id") > 5).select(pl.len()).collect().item() == 5


def test_scan_order_by_needs_one_ordered_read(shop_uri: str) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="global order"):
        pl.scan_database(
            "orders",
            shop_uri,
            order_by="id",
            partition_predicates=["id > 500", "id <= 500"],
        ).collect()
    # without pushdown a table scan still orders; a query can't be wrapped
    out = pl.scan_database("orders", shop_uri, order_by="id", pushdown=False).collect()
    assert out["id"].is_sorted()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="order_by"):
        pl.scan_database(
            "SELECT * FROM orders", shop_uri, order_by="id", pushdown=False
        ).collect()


def test_scan_query_that_isnt_repeatable_counts_its_own_rows(shop_uri: str) -> None:
    # each run picks other rows: the count that a negative slice needs is taken from
    # the rows read, not from a second run (which "changed the table")
    lf = pl.scan_database("SELECT * FROM orders WHERE abs(random()) % 2 = 0", shop_uri)
    tail = lf.tail(5).collect()
    assert tail.height == 5
    assert tail["id"].is_sorted()
    n = lf.select(pl.len()).collect().item()
    assert 0 < n < N_ROWS


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_sampled_integer_column_with_a_later_float(
    tmp_path: Path, driver: str
) -> None:
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, x NUMERIC)")
        # typed Int64 from the first 1000 rows
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, i if i < 1000 else i + 0.5) for i in range(1100)],
        )
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    assert lf.collect_schema()["x"] == pl.Int64
    # never truncated to 1000
    with pytest.raises(pl.exceptions.PolarsError, match=r"1000\.5|Float64"):
        lf.collect()
    out = pl.scan_database(
        "t", f"sqlite:///{path}", driver=driver, schema_overrides={"x": pl.Float64}
    ).collect()
    assert out.filter(pl.col("id") == 1000)["x"].item() == 1000.5


@pytest.mark.parametrize("partition_num", [1, 4])
def test_scan_order_by_with_range_partitions_keeps_nulls_first(
    tmp_path: Path, partition_num: int
) -> None:
    # SQLite sorts NULLs first: they must come from the first partition, not the last
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, v TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, f"v{i}") for i in range(1, 101)] + [(None, "n1"), (None, "n2")],
        )
        conn.commit()
    lf = pl.scan_database(
        "t",
        f"sqlite:///{path}",
        order_by="id",
        partition_on="id",
        partition_num=partition_num,
    )
    ids = lf.collect()["id"].to_list()
    assert ids == [None, None, *range(1, 101)]


def test_scan_errors_dont_leak_passwords() -> None:
    # unencoded `#`, `?` and `/` in a URI password, and `;` in a braced ODBC value
    for uri in [
        "teradata://u:pa#Sekr3t@h/db",
        "teradata://u:pa?Sekr3t@h/db",
        "teradata://u:pa/Sekr3t@h/db",
        "Driver={No Such Driver 1};Server=s;UID=u;PWD={a;Sekr3t}",
    ]:
        with pytest.raises(Exception) as exc:
            pl.scan_database("t", uri).collect()
        assert "Sekr3t" not in str(exc.value), uri


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_string_literal_with_nul(tmp_path: Path, driver: str) -> None:
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a\0b"), (2, "ab")])
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    table = lf.collect()
    for p in [
        pl.col("s") == "a\0b",
        pl.col("s") != "a\0b",
        pl.col("s").is_in(["a\0b"]),
        pl.col("s").str.starts_with("a\0"),
    ]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, unicode(randomblob(1)) AS b FROM orders",
        "SELECT id, random() % 256 AS b FROM orders",
    ],
)
def test_scan_query_with_a_volatile_function_isnt_partitioned(
    shop_uri: str, query: str
) -> None:
    lf = pl.scan_database(
        query, shop_uri, partition_on="b", partition_num=4, partition_range=(0, 255)
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        lf.collect()


def test_scan_hugeint_literals(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    big = 99_999_999_999_999_999_999_999_999_999_999_999_999
    with duckdb.connect(path) as conn:
        # HUGEINT is read as Decimal(38, 0): a value of 39 digits fails the read
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 5::HUGEINT), "
            f"(2, {big}::HUGEINT), (3, NULL::HUGEINT), (4, {-big}::HUGEINT)) "
            "v(id, h)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    table = lf.collect()
    assert table.height == 4
    values = pl.Series([5, big], dtype=pl.Int128).cast(pl.Decimal(38, 0))
    for p in [
        pl.col("h") == 5,
        pl.col("h") >= pl.lit(big, dtype=pl.Decimal(38, 0)),
        pl.col("h").is_in(values),
    ]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))
    with duckdb.connect(path) as conn:
        conn.execute(f"INSERT INTO t VALUES (5, {2**127 - 1}::HUGEINT)")
    with pytest.raises(pl.exceptions.ComputeError, match="can't represent"):
        lf.collect()


def _assert_filters_match(lf: pl.LazyFrame, predicates: list[pl.Expr]) -> None:
    table = lf.collect()
    for p in predicates:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


@pytest.mark.parametrize("source", ["t", "SELECT * FROM t"])
def test_scan_duckdb_collations_and_non_text_columns(
    tmp_path: Path, source: str
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER, s VARCHAR COLLATE NOCASE, v VARCHAR, "
            "u UUID, e ENUM('a', 'b'))"
        )
        conn.execute(
            "INSERT INTO t VALUES "
            "(1, 'abc', 'abc', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a'), "
            "(2, 'ABC', 'ABC', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'b')"
        )
    lf = pl.scan_database(source, f"duckdb:///{path}")
    assert lf.collect_schema()["u"] == pl.String
    _assert_filters_match(
        lf,
        [
            pl.col("s") == "abc",
            pl.col("s") != "abc",
            pl.col("s") < "b",
            pl.col("s").is_in(["abc"]),
            # the database casts the literal to UUID / the enum
            pl.col("u") == "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11",
            pl.col("u") == "zzz",
            pl.col("u") < "b",
            pl.col("e") == "nope",
        ],
    )
    # a VARCHAR is still compared, as a superset: the table declares a collation
    assert "filters 1/1" in lf.filter(pl.col("v") == "abc").explain()


def test_scan_duckdb_session_default_collation() -> None:
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (id INTEGER, v VARCHAR)")
    conn.execute("INSERT INTO t VALUES (1, 'abc'), (2, 'ABC')")
    conn.execute("SET default_collation = 'nocase'")
    lf = pl.scan_database("t", conn)
    _assert_filters_match(lf, [pl.col("v") == "abc", pl.col("v") < "b"])
    assert "(0 exact)" in lf.filter(pl.col("v") == "abc").explain()


def _importorskip_sqlite_driver(driver: str) -> None:
    pytest.importorskip(
        {"adbc": "adbc_driver_sqlite", "sqlite3": "sqlite3"}.get(driver, driver)
    )


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
@pytest.mark.parametrize("source", ["t", "SELECT * FROM t"])
def test_scan_sqlite_strings_without_text_affinity(
    tmp_path: Path, source: str, driver: str
) -> None:
    # a column whose values have several storage classes (ConnectorX reads it as text)
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, n NUMERIC, x, s TEXT COLLATE NOCASE)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            [
                (1, "abc", "10", "abc"),
                (2, 1, 10, "ABC"),
                (3, 10, None, "b"),
                (4, 9, None, None),
                (5, "x", None, None),
            ],
        )
        conn.commit()
    lf = pl.scan_database(source, f"sqlite:///{path}", driver=driver)
    assert lf.collect_schema()["n"] == pl.String
    assert lf.collect()["n"].to_list() == ["abc", "1", "10", "9", "x"]
    _assert_filters_match(
        lf,
        [
            pl.col("n") == "01",
            pl.col("n") == "1.0",
            pl.col("n") > "9",
            pl.col("x") == "10",
            pl.col("s") == "abc",
            pl.col("s") != "abc",
            pl.col("s") < "b",
        ],
    )


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_scan_sqlite_large_integers_next_to_reals(
    tmp_path: Path, driver: str, batch_size: int | None
) -> None:
    # a batch typed by its own values read `2**53 + 1` next to `1.5` as a float, also
    # when retyped to String; alone in a batch, it was an error
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, 2**53 + 1), (2, 1.5)])
        conn.commit()
    uri = f"sqlite:///{path}"
    text = pl.scan_database(
        "t",
        uri,
        driver=driver,
        batch_size=batch_size,
        schema_overrides={"x": pl.String},
    )
    assert text.collect()["x"].to_list() == [str(2**53 + 1), "1.5"]
    floats = pl.scan_database("t", uri, driver=driver, batch_size=batch_size)
    assert floats.collect()["x"].to_list() == [float(2**53 + 1), 1.5]
    # also when retyped to a decimal, through Float64 otherwise
    decimals = pl.scan_database(
        "t",
        uri,
        driver=driver,
        batch_size=batch_size,
        schema_overrides={"x": pl.Decimal(30, 1)},
    )
    assert decimals.collect()["x"].to_list() == [
        Decimal(2**53 + 1),
        Decimal("1.5"),
    ]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_scan_sqlite_large_integers_next_to_integral_reals(
    tmp_path: Path, driver: str, batch_size: int | None
) -> None:
    # retyped to an integer, `2**53 + 1` next to `2.0` or `1.5` was read through Float64
    # (the cast to an override cuts `1.5`)
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(1, 2**53 + 1), (2, 2.0), (3, 1.5), (4, -2.5)],
        )
        conn.commit()
    for dtype in [pl.Int64, pl.UInt64]:
        lf = pl.scan_database(
            "t",
            f"sqlite:///{path}",
            driver=driver,
            batch_size=batch_size,
            schema_overrides={"x": dtype},
        ).filter(pl.col("id") < 4 if dtype == pl.UInt64 else True)
        assert (
            lf.collect()["x"].to_list()
            == [2**53 + 1, 2, 1, -2][: 3 if dtype == pl.UInt64 else 4]
        ), dtype


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_view_union_of_reals_and_integers(
    tmp_path: Path, driver: str
) -> None:
    # `pragma_table_info` types a compound view's column by its first branch: REAL here,
    # while the other branch holds integers
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE a (id INTEGER, x REAL)")
        conn.execute("CREATE TABLE b (id INTEGER, x INTEGER)")
        conn.executemany("INSERT INTO a VALUES (?, ?)", [(1, 1.5), (3, 2.0)])
        conn.executemany("INSERT INTO b VALUES (?, ?)", [(2, 2**53 + 1), (4, 7)])
        conn.execute(
            "CREATE VIEW v AS SELECT id, x FROM a UNION ALL SELECT id, x FROM b"
        )
        conn.execute(
            "CREATE VIEW w AS SELECT id, x FROM b UNION ALL SELECT id, x FROM a"
        )
        conn.commit()
    uri = f"sqlite:///{path}"
    for view in ["v", "w"]:
        for batch_size in [None, 1]:
            lf = pl.scan_database(view, uri, driver=driver, batch_size=batch_size)
            df = lf.collect().sort("id")
            assert df["x"].to_list() == [1.5, float(2**53), 2.0, 7.0], (
                view,
                batch_size,
            )
            got = lf.filter(pl.col("x") == 2.0**53).collect()["id"].to_list()
            assert got == [2], view


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_filter_on_large_integers_read_as_floats(
    tmp_path: Path, driver: str
) -> None:
    # the database compared `2**53 + 1` exactly; Polars reads it as `2.0**53`
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x, r REAL)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(1, 2**53 + 1, 1.0), (2, 1.5, 2.5)]
        )
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    big = float(2**53)
    for predicate, expected in [
        (pl.col("x") == big, [1]),
        (pl.col("x") > big, []),
        (pl.col("x") >= big, [1]),
        (pl.col("x").is_in([big]), [1]),
        (pl.col("x").is_between(big, big), [1]),
        (pl.col("x") < 2, [2]),
    ]:
        assert lf.filter(predicate).collect()["id"].to_list() == expected, predicate
    assert "WHERE" in planned_sql(lf.filter(pl.col("x") == big))
    # a REAL column holds no integers: compared as it is (an index on it serves)
    assert 'WHERE "t"."r" > 2.0' in planned_sql(lf.filter(pl.col("r") > 2.0))


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_numbers_text_past_the_sample(tmp_path: Path, driver: str) -> None:
    # a column without affinity typed Int64 / Float64 from its sample: the read parses
    # an integer's or a real's text, which SQLite compares as text, above every number
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, i, f)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(k, k, k + 0.5) for k in range(3000)]
        )
        conn.execute("INSERT INTO t VALUES (3000, '1', '1.5')")
        conn.execute("INSERT INTO t VALUES (3001, '2', '2')")
        conn.commit()
    uri = f"sqlite:///{path}"
    lf = pl.scan_database("t", uri, driver=driver)
    df = lf.filter(pl.col("id") >= 3000).collect()
    assert df.rows() == [(3000, 1, 1.5), (3001, 2, 2.0)]
    for predicate in [
        pl.col("i") == 1,
        (pl.col("i") > 3) & (pl.col("id") >= 3000),
        pl.col("i").is_in([1, 2]),
        pl.col("f") < 2.5,
        pl.col("f") == 2.0,
    ]:
        got = lf.filter(predicate).collect().sort("id")
        expected = lf.collect().filter(predicate).sort("id")
        assert_frame_equal(got, expected)
    # other text (`'+1'`, `'1e0'`) isn't read as a number: it fails, as `'abc'` does
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO t VALUES (3002, '+1', '1e0')")
        conn.commit()
    for column in ["i", "f"]:
        with pytest.raises(Exception):  # noqa: B017
            pl.scan_database("t", uri, driver=driver).select(column).collect()


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_blobs_in_a_text_column(tmp_path: Path, driver: str) -> None:
    # a blob past the sample reads as its text, which the database compared as a blob
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT COLLATE NOCASE)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(k, f"v{k % 50}") for k in range(3000)]
        )
        conn.execute("INSERT INTO t VALUES (3000, X'7631')")
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    assert lf.filter(pl.col("id") == 3000).collect()["s"].to_list() == ["v1"]
    for predicate in [
        pl.col("s") == "v1",
        pl.col("s") == "V1",  # the column's collation still applies
        pl.col("s") != "v1",
        (pl.col("s") >= "v1") & (pl.col("s") < "v10"),
        pl.col("s").is_in(["v1", "v2"]),
        pl.col("s").str.starts_with("v1"),
    ]:
        assert lf.filter(predicate).select(pl.len()).collect().item() == (
            pl.scan_database("t", f"sqlite:///{path}", driver=driver, pushdown=False)
            .filter(predicate)
            .select(pl.len())
            .collect()
            .item()
        ), predicate
    assert "(CAST(\"s\" AS TEXT)) = 'v1'" in planned_sql(lf.filter(pl.col("s") == "v1"))


def test_scan_sqlite_connectorx_quotes_and_backslashes(tmp_path: Path) -> None:
    # ConnectorX parses the SQL itself, reading `\'` as an escaped quote
    pytest.importorskip("connectorx")
    values = ["a\\'b", "\\'", "''", "a\\b", "it's"]
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver="connectorx")
    for i, v in enumerate(values):
        assert lf.filter(pl.col("s") == v).collect()["id"].to_list() == [i], v
        got = lf.filter(pl.col("s").str.contains(v, literal=True)).collect()
        assert i in got["id"].to_list(), v


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_reals_as_text_as_polars_writes_them(
    tmp_path: Path, driver: str
) -> None:
    # (a table `s` and a column `v`: names in the conversion's SQL)
    # SQLite writes `1.0e+20`, `1.0e-05`, `Inf`; some of its builds round a real's text
    # to another real
    import random

    _importorskip_sqlite_driver(driver)
    values = [1e20, 1e-5, float("inf"), -float("inf"), 1.0, 0.1, 1 / 3, 1e-7, 1e15]
    values += [1e16, 123456789012345680.0, 1234567890123456.7, -2.5, 2.0**53]
    values += [5e-324, 1e-310, 1.7976931348623157e308, 2.0**63, -9.846175033433891]
    values += [2.2790121708605243e274]
    rng = random.Random(7)
    values += [rng.uniform(-1, 1) * 10 ** rng.randint(-300, 300) for _ in range(2000)]
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE s (id INTEGER, x REAL, v)")
        conn.executemany(
            "INSERT INTO s VALUES (?, ?, ?)", [(i, v, v) for i, v in enumerate(values)]
        )
        conn.execute("INSERT INTO s VALUES (10000, NULL, 7)")
        # text that starts as the reader's marker of a real does
        conn.execute("INSERT INTO s VALUES (10001, NULL, char(1) || '1:0')")
        conn.commit()
    lf = pl.scan_database(
        "s",
        f"sqlite:///{path}",
        driver=driver,
        schema_overrides={"x": pl.String, "v": pl.String},
    )
    expected = pl.Series([*values, None, None]).cast(pl.String).to_list()
    df = lf.collect()
    assert df["x"].to_list() == expected
    assert df["v"].to_list() == [*expected[:-2], "7", "\x011:0"]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_numbers_then_text_in_small_batches(
    tmp_path: Path, driver: str
) -> None:
    # row drivers type each batch of the planning sample on its own: Int64, then String
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", enumerate([1, 1, 1, "abc"]))
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver, batch_size=2)
    assert lf.collect()["x"].to_list() == ["1", "1", "1", "abc"]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_filter_on_a_volatile_column(tmp_path: Path, driver: str) -> None:
    # SQLite merges the wrapped query into the filter's and calls random() again for it
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(200)])
        conn.commit()
    lf = pl.scan_database(
        "SELECT random() AS r, id FROM t", f"sqlite:///{path}", driver=driver
    )
    df = lf.filter(pl.col("r") > 0, pl.col("id") < 100).collect()
    assert (df["r"] > 0).all()
    assert (df["id"] < 100).all()
    # the filter on `id` is still pushed (through the conversion of a number's text)
    assert 'ELSE "id" END) < 100' in planned_sql(
        lf.filter(pl.col("r") > 0, pl.col("id") < 100)
    )
    assert "WHERE" not in planned_sql(lf.filter(pl.col("r") > 0))


def test_scan_sqlite_filter_on_a_user_function_column(tmp_path: Path) -> None:
    # a function sqlglot doesn't know may be volatile: `create_function` makes it so
    import random

    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(200)])
        conn.commit()
    with closing(sqlite3.connect(path, check_same_thread=False)) as live:
        live.create_function("myrand", 0, random.random)
        lf = pl.scan_database("SELECT id, myrand() AS r FROM t", live)
        assert "WHERE" not in planned_sql(lf.filter(pl.col("r") > 0.5))
        assert (lf.filter(pl.col("r") > 0.5).collect()["r"] > 0.5).all()


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
@pytest.mark.parametrize("source", ["v", "w", "SELECT * FROM v", "SELECT id, r FROM w"])
def test_scan_sqlite_volatile_views(tmp_path: Path, driver: str, source: str) -> None:
    # SQLite merges the view into the filter's query and calls random() again for it;
    # each partition reads the view again
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(300)])
        conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
        conn.execute("CREATE VIEW w AS SELECT * FROM v")
        conn.commit()
    uri = f"sqlite:///{path}"
    lf = pl.scan_database(source, uri, driver=driver)
    df = lf.filter(pl.col("r") > 0, pl.col("id") < 200).collect()
    assert (df["r"] > 0).all()
    assert (df["id"] < 200).all()
    if source == "v":
        # the filter on `id` is still pushed (through the conversion of a number's
        # text: a view's column may hold one)
        sql = planned_sql(lf.filter(pl.col("r") > 0, pl.col("id") < 200))
        assert 'ELSE "id" END) < 200' in sql
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        pl.scan_database(
            source, uri, driver=driver, partition_on="id", partition_num=4
        ).collect()


def test_scan_sqlite_views_without_volatile_functions() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY, x)")
            conn.executemany(
                "INSERT INTO ids VALUES (?, ?)", [(i, i) for i in range(50)]
            )
            # `abs` and `printf` are deterministic; `(a, b)` names the columns
            conn.execute(
                "CREATE VIEW v (a, b) AS SELECT id, printf('%d', abs(x)) FROM ids"
            )
            conn.commit()
        lf = pl.scan_database(
            "v", f"sqlite:///{path}", partition_on="a", partition_num=4
        )
        assert sorted(lf.collect()["a"].to_list()) == list(range(50))
        assert 'ELSE "a" END) < 5' in planned_sql(lf.filter(pl.col("a") < 5))


def test_scan_sqlite_partition_over_a_user_function(tmp_path: Path) -> None:
    # a function sqlglot doesn't know may be volatile: `create_function` makes it so,
    # unless `deterministic=True`; each partition would call it again
    import random

    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(200)])
        conn.commit()
    with closing(sqlite3.connect(path, check_same_thread=False)) as live:
        live.create_function("myrand", 0, random.Random(1).random)
        live.create_function("twice", 1, lambda v: 2 * v, deterministic=True)
        for query in [
            "SELECT id, myrand() AS r FROM t",
            "SELECT id FROM t WHERE myrand() < 0.5",
        ]:
            lf = pl.scan_database(query, live, partition_on="id", partition_num=4)
            with pytest.raises(pl.exceptions.InvalidOperationError, match="myrand"):
                lf.collect()
        lf = pl.scan_database(
            "SELECT id, twice(id) AS d FROM t", live, partition_on="d", partition_num=4
        )
        assert sorted(lf.collect()["d"].to_list()) == [2 * i for i in range(200)]
        # a temporary view over it, scanned as a table
        live.execute("CREATE TEMP VIEW uv AS SELECT id, myrand() AS r FROM t")
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            pl.scan_database("uv", live, partition_on="id", partition_num=4).collect()
        df = pl.scan_database("uv", live).filter(pl.col("r") > 0.5).collect()
        assert (df["r"] > 0.5).all()


def test_scan_sqlite_volatile_views_in_attached_databases(tmp_path: Path) -> None:
    # an unqualified name resolves in `temp`, `main`, then the attached databases; a
    # view's own names in its database first
    with closing(sqlite3.connect(tmp_path / "aux.db")) as conn:
        conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(300)])
        conn.execute("CREATE VIEW av AS SELECT id, random() AS r FROM ids")
        conn.execute("CREATE VIEW aw AS SELECT * FROM av")
        conn.commit()
    with closing(
        sqlite3.connect(tmp_path / "main.db", check_same_thread=False)
    ) as live:
        live.execute("CREATE TABLE av (id INTEGER, r REAL)")
        live.execute(f"ATTACH '{tmp_path / 'aux.db'}' AS aux")
        for source in ["aw", "aux.aw", "aux.av", "SELECT * FROM aw"]:
            df = pl.scan_database(source, live).filter(pl.col("r") > 0).collect()
            assert (df["r"] > 0).all(), source
            lf = pl.scan_database(source, live, partition_on="id", partition_num=4)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()


def test_scan_partition_with_schema_and_unknown_functions(tmp_path: Path) -> None:
    # without I/O nothing tells whether `quote` is volatile: taken to be as before
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(50)])
        conn.commit()
    lf = pl.scan_database(
        "SELECT id, quote(id) AS q FROM ids",
        f"sqlite:///{path}",
        schema={"id": pl.Int64, "q": pl.String},
        partition_on="id",
        partition_num=4,
    )
    assert sorted(lf.collect()["id"].to_list()) == list(range(50))


def test_scan_duckdb_volatile_views(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
        conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
        conn.execute("CREATE VIEW c AS SELECT id, abs(id) AS a FROM ids")
        # a view's own names resolve in its schema
        conn.execute("CREATE SCHEMA s")
        conn.execute("CREATE VIEW s.v AS SELECT id FROM main.ids WHERE random() < 0.5")
        conn.execute("CREATE VIEW s.w AS SELECT * FROM v")
        # a macro is as volatile as its body
        conn.execute("CREATE MACRO twice(x) AS 2 * x")
        conn.execute("CREATE MACRO noisy(x) AS x + random()")
    uri = f"duckdb:///{path}"
    for source in ["s.w", "SELECT id, noisy(id) AS n FROM ids"]:
        lf = pl.scan_database(source, uri, partition_on="id", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
    lf = pl.scan_database(
        "SELECT id, twice(id) AS d FROM ids", uri, partition_on="d", partition_num=4
    )
    assert sorted(lf.collect()["d"].to_list()) == [2 * i for i in range(300)]
    df = pl.scan_database("v", uri).filter(pl.col("r") > 0.5).collect()
    assert (df["r"] > 0.5).all()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        pl.scan_database("v", uri, partition_on="id", partition_num=4).collect()
    lf = pl.scan_database("c", uri, partition_on="id", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == list(range(300))


def test_scan_query_with_order_by_is_not_partitioned(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, x)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(i, (i * 37) % 101) for i in range(100)]
        )
        conn.commit()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="ORDER BY"):
        pl.scan_database(
            "SELECT * FROM t ORDER BY x DESC",
            f"sqlite:///{path}",
            partition_on="id",
            partition_num=4,
        ).collect()


@pytest.mark.parametrize(
    "source",
    ["e", "SELECT * FROM e", "SELECT * FROM s LIMIT 3", "SELECT i, f, d FROM s"],
)
def test_scan_duckdb_unsampled_columns_keep_their_types(
    tmp_path: Path, source: str
) -> None:
    # the Python driver types an empty result's columns Null; an empty table, or a query
    # that isn't sampled (LIMIT), was read as all String
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE e (i INT, f DOUBLE, d DATE)")
        conn.execute(
            "CREATE TABLE s AS SELECT i::INT AS i, i * 1.5 AS f, "
            "DATE '2020-01-01' + i::INT AS d FROM range(5) r(i)"
        )
    lf = pl.scan_database(source, f"duckdb:///{path}")
    schema = lf.collect_schema()
    assert schema["i"] == pl.Int32
    assert schema["d"] == pl.Date
    assert schema["f"].is_float() or schema["f"].is_decimal()
    lf.filter(pl.col("i") > 1).collect()


@pytest.mark.parametrize("driver", ["sqlite3", "sqlalchemy", "live"])
def test_scan_row_driver_with_leading_nulls(tmp_path: Path, driver: str) -> None:
    # more leading NULLs than a row driver's default inference length (100)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, None if i < 150 else "x") for i in range(200)],
        )
        conn.commit()
    if driver == "live":
        with closing(sqlite3.connect(path, check_same_thread=False)) as live:
            df = pl.scan_database("t", live).collect()
    else:
        df = pl.scan_database("t", f"sqlite:///{path}", driver=driver).collect()
    assert df.schema["s"] == pl.String
    assert df["s"].null_count() == 150


def test_scan_sqlalchemy_literals_with_colons(tmp_path: Path) -> None:
    # SQLAlchemy's `text()` would read `:name` in a pushed literal as a placeholder,
    # and `\:` as an escaped `:`
    pytest.importorskip("sqlalchemy")
    path = tmp_path / "t.db"
    rows = [(1, "\\:x"), (2, ":x"), (3, "y"), (4, ":abc"), (5, ":lo"), (6, "0")]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", rows)
        conn.commit()
    uri = f"sqlite:///{path}"
    expected = pl.DataFrame(rows, schema=["id", "s"], orient="row")
    predicates = [
        pl.col("s") == "\\:x",
        pl.col("s") != "\\:x",
        pl.col("s").is_in(["\\:x"]),
        pl.col("s") == ":abc",
        pl.col("s").str.starts_with(":"),
        pl.col("s") == ":lo",
    ]
    lf = pl.scan_database("t", uri, driver="sqlalchemy")
    # with SQLAlchemy's `:name` parameters, the query's own placeholders are bound
    q = pl.scan_database(
        "SELECT id, s FROM t WHERE id > :lo", uri, driver="sqlalchemy", params={"lo": 0}
    )
    for p in predicates:
        want = sorted(expected.filter(p)["id"].to_list())
        assert sorted(lf.filter(p).collect()["id"].to_list()) == want, p
        assert sorted(q.filter(p).collect()["id"].to_list()) == want, p


def test_scan_sqlalchemy_live_connection_left_outside_a_transaction(
    shop_uri: str,
) -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(shop_uri)
    try:
        with engine.connect() as conn:
            assert pl.scan_database("orders", conn).collect().height == N_ROWS
            assert not conn.in_transaction()
            # a transaction the user opened is theirs: it stays open
            conn.exec_driver_sql("SELECT 1")
            pl.scan_database("orders", conn).head(1).collect()
            assert conn.in_transaction()
    finally:
        engine.dispose()


def test_scan_engine_with_a_small_pool(shop_uri: str, expected: pl.DataFrame) -> None:
    # more partitions than the pool has connections: they share them, rather than
    # wait for the pool to time out
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(
        shop_uri,
        poolclass=sqlalchemy.pool.QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=2,
    )
    try:
        lf = pl.scan_database(
            "orders", engine, partition_on="id", partition_num=4, max_connections=4
        )
        assert_frame_equal(_sorted(lf.collect()), expected)
    finally:
        engine.dispose()


def test_connectorx_checks_the_connection_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a failed login fails `connect`, rather than every catalog lookup in turn
    from polars_database import _bridge

    from polars.io.database import _utils

    calls: list[str] = []

    def read(sql: str, *, connection_uri: str, protocol: str | None) -> Any:
        calls.append(connection_uri)
        if "wrong" in connection_uri:
            msg = "timed out waiting for connection: password authentication failed"
            raise RuntimeError(msg)
        return pl.DataFrame({"x": [1]})

    monkeypatch.setattr(_utils, "_read_sql_connectorx", read)
    monkeypatch.setattr(_bridge, "_CONNECTORX_CHECKED", set())
    with pytest.raises(RuntimeError, match="password authentication failed"):
        _bridge._ConnectorXSession("postgresql://u:wrong@h/db", "postgresql")
    good = "postgresql://u:right@h/db"
    _bridge._ConnectorXSession(good, "postgresql")
    _bridge._ConnectorXSession(good, "postgresql")
    assert calls == ["postgresql://u:wrong@h/db", good]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_sqlite_uri_paths_mean_one_file_for_every_driver(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        {"adbc": "adbc_driver_sqlite", "sqlite3": "sqlite3"}.get(driver, driver)
    )
    monkeypatch.chdir(tmp_path)
    for name, x in [("rel.db", 1), ("a b.db", 2)]:
        with closing(sqlite3.connect(tmp_path / name)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (?)", (x,))
            conn.commit()

    def read(uri: str) -> int:
        return int(pl.scan_database("t", uri, driver=driver).collect().item())

    # as SQLAlchemy reads them: three slashes are relative, four absolute
    assert read("sqlite:///rel.db") == 1
    assert read(f"sqlite:///{tmp_path}/rel.db") == 1
    assert read("sqlite:rel.db") == 1
    assert read("sqlite:///a%20b.db") == 2
    if driver == "connectorx":
        with pytest.raises(Exception, match="connectorx can't open SQLite with URI"):
            read("sqlite:///rel.db?mode=ro")
    else:
        assert read("sqlite:///rel.db?mode=ro") == 1
    # no file was created elsewhere
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a b.db", "rel.db"]


def _skip_without_duckdb_adbc() -> None:
    try:
        pl.scan_database("SELECT 1 AS x", "duckdb://", driver="adbc").collect()
    except pl.exceptions.ComputeError as e:
        if "is not installed" in str(e):
            pytest.skip("no DuckDB ADBC driver")
        raise


@pytest.mark.parametrize("driver", ["duckdb", "adbc"])
def test_scan_duckdb_string_literal_with_nul(tmp_path: Path, driver: str) -> None:
    duckdb = pytest.importorskip("duckdb")
    if driver == "adbc":
        _skip_without_duckdb_adbc()
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a' || chr(0) || 'b'), "
            "(2, 'ab')) v(id, s)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}", driver=driver)
    table = lf.collect()
    for p in [pl.col("s") == "a\0b", pl.col("s") != "a\0b"]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))
        # the Python module reads the whole query text; C drivers stop at the NUL
        pushed = "filters 1/1" in lf.filter(p).explain()
        assert pushed is (driver == "duckdb")


@pytest.mark.parametrize("driver", ["duckdb", "adbc"])
def test_scan_duckdb_values_arrow_cant_carry(tmp_path: Path, driver: str) -> None:
    # TIME '24:00:00' read as NULL, ±infinity as i64::MAX, the UHUGEINT maximum as -1;
    # a UHUGEINT of 39 digits broke its Decimal(38, 0)
    duckdb = pytest.importorskip("duckdb")
    if driver == "adbc":
        _skip_without_duckdb_adbc()
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER, tm TIME, ts TIMESTAMP, tns TIMESTAMP_NS, "
            "tz TIMESTAMPTZ, d DATE, uh UHUGEINT)"
        )
        conn.execute(
            "INSERT INTO t VALUES (1, '12:00:00', '2024-01-01', '2024-01-01', "
            f"'2024-01-01 00:00:00+00', '2024-01-01', {10**38 - 1})"
        )
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, f"duckdb:///{path}", driver=driver)
        assert lf.collect()["uh"].cast(pl.String).item() == str(10**38 - 1)
    values = {
        "tm": "'24:00:00'",
        "ts": "'infinity'",
        "tns": "'-infinity'",
        "tz": "'infinity'",
        "d": "'-infinity'",
        "uh": str(10**38),
    }
    with duckdb.connect(path) as conn:
        for i, (column, value) in enumerate(values.items(), start=2):
            conn.execute(f"INSERT INTO t (id, {column}) VALUES ({i}, {value})")
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, f"duckdb:///{path}", driver=driver)
        for column in values:
            with pytest.raises(
                pl.exceptions.ComputeError,
                match=f'column "{column}" holds a value Polars can\'t represent',
            ):
                lf.select(column).collect()
        assert lf.select("id").collect().height == 7


@pytest.mark.parametrize("driver", ["duckdb", "adbc"])
def test_scan_duckdb_types_read_as_text(tmp_path: Path, driver: str) -> None:
    # BIT and VARINT read as their internal bytes; an INTERVAL or UNION column failed
    # every scan of its table, even unprojected
    duckdb = pytest.importorskip("duckdb")
    if driver == "adbc":
        _skip_without_duckdb_adbc()
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER, b BIT, v VARINT, iv INTERVAL, "
            "u UNION(n INTEGER, s VARCHAR), lb BIT[], st STRUCT(a INTERVAL))"
        )
        conn.execute(
            "INSERT INTO t VALUES (1, '0101', "
            "123456789012345678901234567890123456789012, INTERVAL 3 DAY, 1, "
            "['11'::BIT], {'a': INTERVAL 1 DAY}), "
            "(2, '1', -1, NULL, 'x', NULL, NULL)"
        )
    # (a query's columns are probed as the driver reads them: an INTERVAL fails there)
    for source in ["t"]:
        lf = pl.scan_database(source, f"duckdb:///{path}", driver=driver)
        assert lf.select("id").collect().height == 2
        df = lf.collect().sort("id")
        assert df["b"].to_list() == ["0101", "1"]
        assert df["v"].to_list() == ["123456789012345678901234567890123456789012", "-1"]
        assert df["iv"].to_list() == ["3 days", None]
        assert df["u"].to_list() == ["1", "x"]
        assert df["lb"].to_list() == ["[11]", None]
        assert df["st"].to_list() == ["{'a': 1 day}", None]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_reals_in_a_string_column_keep_every_digit(
    tmp_path: Path, driver: str
) -> None:
    # SQLite's text of a real has 15 digits (ConnectorX), ADBC's 6
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    values = ["label", 1234.5678901234567, 0.1, 2.0 / 3, 1e300, None]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
        conn.commit()
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, f"sqlite:///{path}", driver=driver)
        read = lf.collect().sort("id")["x"].to_list()
        assert read[0] == "label"
        assert read[-1] is None
        assert [float(v) for v in read[1:-1]] == values[1:-1]


def test_scan_duckdb_timestamptz_is_planned_in_utc(tmp_path: Path) -> None:
    # DuckDB returns TIMESTAMPTZ in the session time zone, the machine's by default
    from datetime import datetime

    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT TIMESTAMPTZ '2024-01-01 00:00:00+00' AS tz"
        )

    def connect() -> Any:
        return duckdb.connect(path, config={"TimeZone": "Asia/Tokyo"})

    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, connect)
        assert lf.collect_schema()["tz"] == pl.Datetime("us", "UTC")
        assert lf.collect()["tz"].item() == datetime(2024, 1, 1, tzinfo=UTC)


def test_scan_engine_with_a_shared_connection_pool() -> None:
    # a StaticPool's one connection is the user's: used as a live connection, so its
    # temporary tables are visible and it isn't discarded
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=sqlalchemy.pool.StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("CREATE TEMPORARY TABLE tmp (id INTEGER, s TEXT)")
            conn.exec_driver_sql("INSERT INTO tmp VALUES (1, 'a'), (2, 'b')")
            conn.commit()
        lf = pl.scan_database("tmp", engine)
        assert sorted(lf.collect()["id"].to_list()) == [1, 2]
        assert lf.filter(pl.col("s") == "b").collect()["id"].to_list() == [2]
        assert sorted(lf.collect()["id"].to_list()) == [1, 2]
    finally:
        engine.dispose()


def test_engine_pool_capacity() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from polars.io.database._scan import _engine_pool_capacity

    def capacity(**kwargs: Any) -> int | None:
        engine = sqlalchemy.create_engine(
            "sqlite:///unused.db", poolclass=sqlalchemy.pool.QueuePool, **kwargs
        )
        return _engine_pool_capacity(engine)

    assert capacity(pool_size=2, max_overflow=3) == 5
    # unbounded
    assert capacity(pool_size=0) is None
    assert capacity(pool_size=2, max_overflow=-1) is None


def test_connectorx_missing_sqlite_file(tmp_path: Path) -> None:
    # ConnectorX would retry opening it for minutes
    pytest.importorskip("connectorx")
    path = tmp_path / "missing.db"
    with pytest.raises(Exception, match="no SQLite database file"):
        pl.scan_database("t", f"sqlite:///{path}", driver="connectorx").collect()
    assert not path.exists()


def test_connectorx_sqlite_windows_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # ConnectorX parses the URI as a URL: `sqlite://C:\\Users\\a` would come out as
    # `C/\\Users\\a`. Fully encoded, nothing in the path is a URL delimiter
    import ntpath

    from polars_database import _bridge

    monkeypatch.setattr(os.path, "abspath", ntpath.abspath)
    path = "C:\\Users\\A b#1\\t.db"
    uri = _bridge._connectorx_uri(f"sqlite:///{path}", "sqlite")
    rest = uri.removeprefix("sqlite://")
    assert not set(rest) & set(":/\\?# "), uri
    assert unquote(rest) == path


def test_connectorx_sqlite_path_with_url_characters(tmp_path: Path) -> None:
    pytest.importorskip("connectorx")
    path = tmp_path / "A b#1%2F" / "t.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1 AS id")
        conn.commit()
    uri = f"sqlite:///{quote(str(path))}"
    lf = pl.scan_database("t", uri, driver="connectorx")
    assert lf.collect()["id"].to_list() == [1]


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_empty_sqlite_table_is_typed_by_affinity(
    tmp_path: Path, driver: str
) -> None:
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER, s VARCHAR(10), f DOUBLE, b BLOB, d DATE)"
        )
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    assert dict(lf.collect_schema()) == {
        "id": pl.Int64,
        "s": pl.String,
        "f": pl.Float64,
        "b": pl.Binary,
        "d": pl.String,
    }
    # rows added after planning fit the planned schema
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO t VALUES (1, 'x', 1.5, x'00', '2020-01-02')")
        conn.commit()
    assert lf.collect().rows() == [(1, "x", 1.5, b"\x00", "2020-01-02")]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_user_schema_string_isnt_taken_as_text(
    tmp_path: Path, driver: str
) -> None:
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x, c CHAR(3))")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(1, "10", "a"), (2, 10, "a ")]
        )
        conn.commit()
    schema = {"id": pl.Int64, "x": pl.String, "c": pl.String}
    lf = pl.scan_database("t", f"sqlite:///{path}", schema=schema, driver=driver)
    _assert_filters_match(lf, [pl.col("x") == "10", pl.col("x") != "10"])
    # SQLite doesn't pad CHAR(n): its strings compare exactly
    exact = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    exact = exact.filter(pl.col("c") == "a")
    assert "(1 exact)" in exact.explain()
    assert exact.collect()["id"].to_list() == [1]


def test_scan_duckdb_names_resolve_case_insensitively(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, u UUID, v VARCHAR)")
        conn.execute(
            "INSERT INTO t VALUES (1, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a')"
        )
    for source in ["T", "MAIN.T", "t"]:
        lf = pl.scan_database(source, f"duckdb:///{path}")
        _assert_filters_match(lf, [pl.col("u") == "zzz", pl.col("v") != "a"])
        assert "(1 exact)" in lf.filter(pl.col("v") == "a").explain(), source


def test_scan_duckdb_uri_paths_via_adbc(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    _skip_without_duckdb_adbc()
    with duckdb.connect(str(tmp_path / "d.db")) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1 AS id")
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        # relative, as the Python backends (and SQLAlchemy) read it
        for uri in ["duckdb:///d.db", "duckdb:///d.db?read_only=true"]:
            lf = pl.scan_database("t", uri, driver="adbc")
            assert lf.collect()["id"].to_list() == [1], uri
    finally:
        os.chdir(cwd)


def test_scan_shared_connection_pool_keeps_the_users_transaction() -> None:
    # the pool's reset on return would roll back the one shared connection
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=sqlalchemy.pool.StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE t (id INTEGER)")
            conn.exec_driver_sql("INSERT INTO t VALUES (1), (2)")
        user = engine.connect()
        user.exec_driver_sql("INSERT INTO t VALUES (3)")  # not committed
        assert pl.scan_database("t", engine).collect().height == 3
        assert user.in_transaction()
        user.commit()
        user.close()
        with engine.connect() as conn:
            assert conn.exec_driver_sql("SELECT count(*) FROM t").scalar() == 3
    finally:
        engine.dispose()


def test_restored_setting_literals() -> None:
    from polars_database._bridge import _sql_str

    assert _sql_str("UTC", dialect="postgresql") == "'UTC'"
    # independent of `standard_conforming_strings`
    assert _sql_str("a\\b'c", dialect="postgresql") == "E'a\\\\b''c'"
    assert _sql_str("a\\b", dialect="mysql") == "'a\\\\b'"


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_numbers_of_two_storage_classes(
    tmp_path: Path, driver: str
) -> None:
    # integers and reals in a NUMERIC column (ConnectorX reads them as reals)
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, n NUMERIC, b)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)",
            [(1, 1, b"\x01"), (2, 1.5, b"\x02"), (3, None, None)],
        )
        conn.commit()
    df = pl.scan_database("t", f"sqlite:///{path}", driver=driver).collect()
    assert df.schema == pl.Schema({"id": pl.Int64, "n": pl.Float64, "b": pl.Binary})
    assert df["n"].to_list() == [1.0, 1.5, None]


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_sqlite_integer_booleans(tmp_path: Path, driver: str) -> None:
    # SQLite booleans are integers: Polars reads any but 0 as true, `TRUE` is 1
    pytest.importorskip({"adbc": "adbc_driver_sqlite"}.get(driver, driver))
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(1, 1), (2, 2), (3, 0), (4, -1), (5, None)]
        )
        conn.commit()
    schema = {"id": pl.Int64, "b": pl.Boolean}
    lf = pl.scan_database("t", f"sqlite:///{path}", schema=schema, driver=driver)
    _assert_filters_match(
        lf,
        [
            pl.col("b"),
            ~pl.col("b"),
            pl.col("b") == True,  # noqa: E712
            pl.col("b") != True,  # noqa: E712
            pl.col("b") == False,  # noqa: E712
            pl.col("b").eq_missing(False),
            pl.col("b").ne_missing(True),
        ],
    )
    assert '"b" <> 0' in lf.filter(pl.col("b")).explain()


def test_scan_query_ending_in_a_semicolon_and_comments(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    for query in [
        "SELECT id FROM orders WHERE id < 5; -- done",
        "SELECT id FROM orders WHERE id < 5 /* done */ ;;",
        "SELECT id FROM orders WHERE id < 5 -- done",
    ]:
        lf = pl.scan_database(query, shop_uri).filter(pl.col("id") > 2)
        assert "(1 exact)" in lf.explain(), query
        assert sorted(lf.collect()["id"].to_list()) == [3, 4], query


def test_scan_unparsable_query_must_be_a_plain_select(shop_uri: str) -> None:
    # sqlglot can't parse `USING <`; SQLite sessions aren't read-only, so only a
    # plain SELECT is taken
    for query in [
        "WITH d AS (SELECT id FROM orders) SELECT * FROM d ORDER BY id USING <",
        "SELECT id INTO t2 FROM orders ORDER BY id USING <",
    ]:
        lf = pl.scan_database(query, shop_uri, source_type="query")
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="can't be parsed"
        ):
            lf.collect_schema()


def test_scan_now_dependent_query_isnt_partitioned(shop_uri: str) -> None:
    lf = pl.scan_database(
        "SELECT * FROM orders WHERE day < date('now')",
        shop_uri,
        partition_on="id",
        partition_num=4,
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="current time"):
        lf.collect()


def test_scan_duckdb_session_null_order(tmp_path: Path) -> None:
    # `order_by` writes its NULL order: a session's `default_null_order` doesn't move
    # the NULLs between partitions
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        conn.execute(
            "CREATE TABLE t AS SELECT CASE WHEN range % 10 = 0 THEN NULL "
            "ELSE range END AS k FROM range(100)"
        )
        conn.execute("SET default_null_order = 'nulls_first'")
        for partition_num in [1, 4]:
            lf = pl.scan_database(
                "t", conn, order_by="k", partition_on="k", partition_num=partition_num
            )
            k = lf.collect()["k"].to_list()
            assert k == sorted(v for v in k if v is not None) + [None] * 10
    finally:
        conn.close()


def test_scan_numpy_params_arent_merged() -> None:
    # two arrays whose reprs are the same (NumPy elides the middle)
    duckdb = pytest.importorskip("duckdb")
    np = pytest.importorskip("numpy")
    conn = duckdb.connect()
    try:
        a = np.arange(2000)
        b = a.copy()
        b[1000] = 4999
        assert repr(a) == repr(b)
        query = "SELECT range AS id FROM range(5000) WHERE list_contains(?, range)"

        def scan(params: Any) -> pl.LazyFrame:
            return pl.scan_database(query, conn, params=[params], partition_on=None)

        both = pl.concat([scan(a).select(pl.max("id")), scan(b).select(pl.max("id"))])
        assert both.collect()["id"].to_list() == [1999, 4999]
    finally:
        conn.close()


def test_scan_plain_params_are_copied() -> None:
    # the key and the values run are the ones the scan was built with, whatever the
    # caller's list holds by then
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        query = "SELECT ?::INTEGER AS x"
        p: list[Any] = [1]
        a = pl.scan_database(query, conn, params=p, partition_on=None)
        p[0] = 2
        b = pl.scan_database(query, conn, params=[1], partition_on=None)
        nested: dict[str, Any] = {"x": [1]}
        c = pl.scan_database(
            "SELECT $x[1]::INTEGER AS x", conn, params=nested, partition_on=None
        )
        nested["x"][0] = 3
        out = pl.collect_all([a, b, c])
        assert [df["x"].to_list() for df in out] == [[1], [1], [1]]
    finally:
        conn.close()


def test_scan_sqlite_declared_types_through_connectorx(tmp_path: Path) -> None:
    # ConnectorX read a BOOLEAN's 2 and -1 as true, and failed on a DATE's or a
    # BLOB's integers: every driver reads the values by their storage class
    pytest.importorskip("connectorx")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN, d DATE, ts TIMESTAMP, "
            "bl BLOB, i INT)"
        )
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, None, None, None, None, b"\x01"),
                (2, 2, 20240101, 5, 7, b""),
                (3, -1, 1, 2, 8, None),
                (4, 0, None, None, None, b"\x02"),
            ],
        )
        conn.commit()
    uri = f"sqlite:///{path}"
    for source in ["t", "SELECT * FROM t"]:
        expected = pl.scan_database(source, uri, driver="sqlite3").collect()
        assert expected["b"].to_list() == [None, 2, -1, 0]
        lf = pl.scan_database(source, uri, driver="connectorx")
        assert_frame_equal(lf.collect(), expected)
        _assert_filters_match(lf, [pl.col("b") == 1, pl.col("b") != 0, pl.col("d") > 1])


def test_scan_duckdb_nested_values_arrow_cant_carry(tmp_path: Path) -> None:
    # lists, structs and maps of infinities, 24:00:00 and UHUGEINT values
    # beyond i128 read i64::MAX, NULL or -1; a TIMETZ of 24:00:00 read NULL; (U)HUGEINT
    # values of 39 digits broke their Decimal(38, 0)
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    values = {
        "lts": ("TIMESTAMP[]", "['2024-01-01'::TIMESTAMP]", "['infinity'::TIMESTAMP]"),
        "lld": ("DATE[][]", "[['2024-01-01'::DATE]]", "[[NULL, '-infinity'::DATE]]"),
        "st": (
            'STRUCT(a TIMESTAMPTZ, "b c" INTEGER)',
            "{'a': '2024-01-01 00:00:00+00', 'b c': 1}",
            "{'a': 'infinity', 'b c': 1}",
        ),
        "ls": ("STRUCT(t TIME)[]", "[{'t': '12:00:00'}]", "[{'t': '24:00:00'}]"),
        "m": (
            "MAP(VARCHAR, DATE)",
            "MAP {'k': '2024-01-01'::DATE}",
            "MAP {'k': 'infinity'::DATE}",
        ),
        "luh": ("UHUGEINT[]", "[1]", f"[{2**128 - 1}]"),
        # more than Decimal(38, 0)'s digits, within i128
        "uh": ("UHUGEINT", "1", str(10**38 + 1)),
        "hi": ("HUGEINT", str(-(10**38) + 1), str(-(10**38))),
        "ttz": ("TIMETZ", "'12:00:00+00'", "'24:00:00+00'"),
    }
    with duckdb.connect(path) as conn:
        columns = ", ".join(f'"{c}" {t}' for c, (t, _, _) in values.items())
        conn.execute(f"CREATE TABLE t (id INTEGER, {columns})")
        good = ", ".join(v for _, v, _ in values.values())
        conn.execute(f"INSERT INTO t VALUES (1, {good})")
    for source in ["t", "SELECT * FROM t"]:
        df = pl.scan_database(source, f"duckdb:///{path}").collect()
        assert df["lts"].to_list() == [[datetime(2024, 1, 1)]]
        assert df["ls"].to_list() == [[{"t": time(12)}]]
        assert df["ttz"].to_list() == [time(12)]
    with duckdb.connect(path) as conn:
        for i, (column, (_, _, bad)) in enumerate(values.items(), start=2):
            conn.execute(f'INSERT INTO t (id, "{column}") VALUES ({i}, {bad})')
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, f"duckdb:///{path}")
        for column in values:
            with pytest.raises(
                pl.exceptions.ComputeError,
                match=f'column "{column}" holds a value Polars can\'t represent',
            ):
                lf.select(column).collect()
        assert lf.select("id").collect().height == 1 + len(values)


def test_scan_duckdb_nanosecond_timestamps_compare_at_nanoseconds(
    tmp_path: Path,
) -> None:
    # a microsecond literal compared with a TIMESTAMP_NS column at microseconds
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES "
            "(1, '2020-01-01 00:00:00.000000500'::TIMESTAMP_NS), "
            "(2, '2020-01-01 00:00:00'::TIMESTAMP_NS), "
            "(3, '2020-01-01 00:00:00.000001'::TIMESTAMP_NS), "
            "(4, NULL::TIMESTAMP_NS)) v(id, ts)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    assert lf.collect_schema()["ts"] == pl.Datetime("ns")
    t0 = datetime(2020, 1, 1)
    t1 = datetime(2020, 1, 1, 0, 0, 0, 1)
    check_filter_ids(
        lf,
        [
            pl.col("ts") == t0,
            pl.col("ts") != t0,
            pl.col("ts") > t0,
            pl.col("ts") <= t0,
            pl.col("ts") < t1,
            pl.col("ts").is_in([t0, t1]),
            pl.col("ts").is_between(t0, t1, closed="none"),
        ],
    )


def test_scan_duckdb_float_literals_compare_exactly(tmp_path: Path) -> None:
    # `repr(float)` is a DECIMAL literal to DuckDB, cast to DOUBLE inexactly
    import random

    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(140)
    values = [rng.random() for _ in range(300)]
    values += [rng.uniform(-1, 1) * 10.0 ** rng.randint(-40, 40) for _ in range(300)]
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, f DOUBLE)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
    lf = pl.scan_database("t", f"duckdb:///{path}")
    got = lf.filter(pl.col("f").is_in(values)).collect()
    assert sorted(got["id"].to_list()) == list(range(len(values)))
    for i, v in list(enumerate(values))[::20]:
        assert lf.filter(pl.col("f") == v).collect()["id"].to_list() == [i], v
        n_le = sum(x <= v for x in values)
        assert lf.filter(pl.col("f") <= v).collect().height == n_le, v


def test_scan_sqlite_utf16_text_is_not_ordered_in_the_database(
    tmp_path: Path,
) -> None:
    # BINARY compares UTF-16 bytes (memcmp) in a UTF-16 database: not code point order
    path = tmp_path / "utf16.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA encoding = 'UTF-16le'")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(1, "a"), (2, "Ā"), (3, "b"), (4, "\U0001f600"), (5, "Ａ")],  # noqa: RUF001
        )
        conn.commit()
        assert conn.execute("PRAGMA encoding").fetchone()[0] == "UTF-16le"
    lf = pl.scan_database("t", f"sqlite:///{path}")
    check_filter_ids(
        lf,
        [
            pl.col("s") > "a",
            pl.col("s") < "b",
            pl.col("s") >= "Ａ",  # noqa: RUF001
            pl.col("s") == "Ā",
        ],
    )


def test_scan_duckdb_negative_wide_decimal_literals(tmp_path: Path) -> None:
    # sqlglot wrote a negative literal through a 28-digit Decimal: rounded
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    v = "0.1234567890123456789012345678901234"
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES "
            f"(1, -{v}::DECIMAL(38, 34)), (2, {v}::DECIMAL(38, 34)), "
            "(3, -0.1234567890123456789012345679::DECIMAL(38, 34))) v(id, d)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    neg = Decimal("-" + v)
    check_filter_ids(
        lf,
        [
            pl.col("d") == neg,
            pl.col("d") < neg,
            pl.col("d") > neg,
            pl.col("d").is_in([neg, -neg]),
        ],
    )
    assert "(1 exact)" in lf.filter(pl.col("d") == neg).explain()


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_numbers_text_in_a_query_or_under_schema(
    tmp_path: Path, driver: str
) -> None:
    # a column without affinity holding a number's text (`'2'`) reads as that number,
    # but a query's column (or one under `schema=`), and any two compared columns, were
    # compared as stored: text, which SQLite orders above every number
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a, b)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(i, i, i) for i in range(1, 1001)]
        )
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(1001, 2, "2"), (1002, 5, "7")]
        )
        conn.commit()
    uri = f"sqlite:///{path}"
    schema = {"id": pl.Int64, "a": pl.Int64, "b": pl.Int64}
    for lf in [
        pl.scan_database("t", uri, driver=driver),
        pl.scan_database("t", uri, driver=driver, schema=schema),
        pl.scan_database("SELECT id, a, b FROM t", uri, driver=driver),
    ]:
        check_filter_ids(
            lf,
            [
                pl.col("a") == pl.col("b"),
                pl.col("a") != pl.col("b"),
                pl.col("b") == 2,
                pl.col("b").is_in([2, 7]),
                pl.col("b") > 6,
            ],
        )


@pytest.mark.parametrize("driver", ["sqlite3", "sqlalchemy"])
def test_scan_sqlite_temporary_table_hiding_a_table(
    tmp_path: Path, driver: str
) -> None:
    # an unqualified name resolves in `temp` first, but `main`'s table was described:
    # its binary collation, where the temporary table's column is NOCASE
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'zzz')")
        conn.commit()
    engine: Any = None
    if driver == "sqlalchemy":
        sqlalchemy = pytest.importorskip("sqlalchemy")
        engine = sqlalchemy.create_engine(f"sqlite:///{path}")
        live: Any = engine.connect()
        live.exec_driver_sql("CREATE TEMP TABLE t (id INTEGER, s TEXT COLLATE NOCASE)")
        live.exec_driver_sql("INSERT INTO t VALUES (1, 'abc'), (2, 'ABC')")
    else:
        live = sqlite3.connect(path, check_same_thread=False)
        live.execute("CREATE TEMP TABLE t (id INTEGER, s TEXT COLLATE NOCASE)")
        live.execute("INSERT INTO t VALUES (1, 'abc'), (2, 'ABC')")
    try:
        lf = pl.scan_database("t", live)
        assert lf.filter(pl.col("s") == "abc").collect()["id"].to_list() == [1]
        assert lf.filter(pl.col("s") != "abc").collect()["id"].to_list() == [2]
    finally:
        live.close()
        if engine is not None:
            engine.dispose()


def test_scan_sqlite_volatility_under_an_explicit_schema(tmp_path: Path) -> None:
    # `schema=` plans without I/O, so views and user functions weren't inspected: a
    # filter on a view's `random()` returned rows Polars would drop, and partitions over
    # a user function read other rows; the read checks it now
    import random

    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(300)])
        conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
        conn.commit()
    uri = f"sqlite:///{path}"
    schema = {"id": pl.Int64, "r": pl.Int64}
    lf = pl.scan_database("v", uri, schema=schema)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="schema="):
        lf.filter(pl.col("r") > 0).collect()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        pl.scan_database(
            "v", uri, schema=schema, partition_on="id", partition_num=4
        ).collect()
    # nothing relies on it: read as planned; a filter on a column of no volatile
    # function is still pushed
    assert lf.collect().height == 300
    assert lf.filter(pl.col("id") < 10).collect().height == 10
    # a view that isn't repeatable is counted from the rows it reads: its tail is the
    # read's, not cut by a count that saw other rows
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE more (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO more VALUES (?)", [(i,) for i in range(1000)])
        conn.execute("CREATE VIEW half AS SELECT id FROM more WHERE random() < 0")
        conn.commit()
    for _ in range(3):
        # (a count of other rows cut the read ~30 ids early, or failed "table changed";
        # no id in 980-999 kept: 2**-20)
        tail = pl.scan_database("half", uri, schema={"id": pl.Int64}).tail(5).collect()
        ids = tail["id"].to_list()
        assert len(ids) == 5, ids
        assert ids == sorted(ids), ids
        assert ids[-1] >= 980, ids
    # automatic partitions of a volatile view (without schema=) read it once
    lf = pl.scan_database("v", uri, order_by=["id"], partition_num=4)
    assert lf.collect()["id"].n_unique() == 300
    assert (
        pl.scan_database("ids", uri, schema={"id": pl.Int64})
        .filter(pl.col("id") < 10)
        .collect()
        .height
        == 10
    )
    with closing(sqlite3.connect(path, check_same_thread=False)) as live:
        live.create_function("myrand", 0, random.Random(1).random)
        lf = pl.scan_database(
            "SELECT id, myrand() AS r FROM ids",
            live,
            schema={"id": pl.Int64, "r": pl.Float64},
            partition_on="id",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="myrand"):
            lf.collect()


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_scan_sqlite_override_cuts_reals_as_compared(
    tmp_path: Path, driver: str
) -> None:
    # an Int64 override of an Int64 column reads 1.5 as 1, but `i == 1` compared 1.5
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, i)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(k, k) for k in range(2, 1001)]
        )
        conn.execute("INSERT INTO t VALUES (1001, 1.5)")
        conn.commit()
    lf = pl.scan_database(
        "t", f"sqlite:///{path}", driver=driver, schema_overrides={"i": pl.Int64}
    )
    check_filter_ids(lf, [pl.col("i") == 1, pl.col("i") < 2, pl.col("i").is_in([1, 3])])


def test_scan_duckdb_volatility_through_ctes_and_macros(tmp_path: Path) -> None:
    # a CTE named as the view it reads hid the view; table macros weren't inspected, nor
    # the views a macro's body reads: partitions read other rows each
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(1000)")
        conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
        conn.execute("CREATE MACRO tm() AS TABLE SELECT id, random() AS r FROM ids")
        conn.execute("CREATE MACRO dm() AS TABLE SELECT id, id * 2 AS r FROM ids")
        conn.execute("CREATE MACRO vr(x) AS (SELECT r FROM v WHERE id = x)")
    uri = f"duckdb:///{path}"
    for query in [
        "WITH v AS (SELECT * FROM v) SELECT * FROM v",
        "SELECT * FROM tm()",
        "SELECT id, vr(id) AS r FROM ids",
    ]:
        lf = pl.scan_database(query, uri, partition_on="r", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
    lf = pl.scan_database("SELECT * FROM dm()", uri, partition_on="r", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == list(range(1000))
    # a scalar macro whose body starts like a query's keyword is still an expression
    with duckdb.connect(path) as conn:
        conn.execute("CREATE MACRO h(x) AS from_base64(x)")
    lf = pl.scan_database(
        "SELECT id, h('AA==') AS p FROM ids", uri, partition_on="id", partition_num=4
    )
    assert lf.collect().height == 1000
    # a built-in table function is stable
    lf = pl.scan_database(
        "SELECT * FROM range(100) t(id)", uri, partition_on="id", partition_num=2
    )
    assert sorted(lf.collect()["id"].to_list()) == list(range(100))


def test_scan_sqlite_cte_naming_itself_is_recursive(tmp_path: Path) -> None:
    # SQLite takes a CTE that names itself as recursive, without RECURSIVE: it doesn't
    # read the volatile view of its name
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE VIEW v AS SELECT random() AS x")
    query = (
        "WITH v AS (SELECT 1 AS x UNION ALL SELECT x + 1 FROM v WHERE x < 1000) "
        "SELECT x FROM v"
    )
    lf = pl.scan_database(query, f"sqlite:///{path}", partition_on="x", partition_num=4)
    assert sorted(lf.collect()["x"].to_list()) == list(range(1, 1001))


def test_scan_decimal_override_rounds_as_asked(tmp_path: Path) -> None:
    # a decimal retyped to fewer places by `schema_overrides` failed "would round",
    # though Polars' own cast rounds it; a planned decimal still may not round
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1.234::DECIMAL(10, 3) AS x")
    uri = f"duckdb:///{path}"
    for source in ["t", "SELECT x FROM t"]:
        lf = pl.scan_database(source, uri, schema_overrides={"x": pl.Decimal(10, 2)})
        expected = pl.Series([Decimal("1.234")]).cast(pl.Decimal(10, 2))
        assert lf.collect()["x"].to_list() == expected.to_list(), source


def _sql_of(lf: pl.LazyFrame) -> str:
    return "\n".join(line for line in lf.explain().splitlines() if "sql[" in line)


def test_scan_sqlite_schema_columns_matched_to_the_views_by_name(
    tmp_path: Path,
) -> None:
    # Under `schema=`, a view's volatile columns were matched to the user's by position:
    # `{"r": ..., "id": ...}` took `id` for the view's `random()`, and a filter on `r`
    # was run in the database, which calls `random()` again for it (rows with r <= 0 for
    # `r > 0`); a schema of `id` alone took every column for volatile, failing `id < 10`
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(300)])
        conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
        conn.commit()
    uri = f"sqlite:///{path}"
    lf = pl.scan_database("v", uri, schema={"r": pl.Int64, "id": pl.Int64})
    with pytest.raises(pl.exceptions.InvalidOperationError, match="a filter on 'r'"):
        lf.filter(pl.col("r") > 0).collect()
    assert lf.filter(pl.col("id") < 10).collect().height == 10
    lf = pl.scan_database("v", uri, schema={"id": pl.Int64})
    assert lf.filter(pl.col("id") < 10).collect().height == 10


@pytest.mark.parametrize("database", ["sqlite", "duckdb"])
def test_scan_automatic_partitions_of_a_volatile_view_under_schema(
    tmp_path: Path, database: str
) -> None:
    # `schema=` plans without I/O: automatic partitions (`order_by` and `partition_num`)
    # split a view of `random()`, and the read then failed "pass partition_on=None",
    # which the user never passed. They are one, as without `schema=`.
    if database == "sqlite":
        path = tmp_path / "t.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
            conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(300)])
            conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
            conn.commit()
        uri = f"sqlite:///{path}"
    else:
        duckdb = pytest.importorskip("duckdb")
        path = tmp_path / "t.duckdb"
        with duckdb.connect(str(path)) as conn:
            conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
            conn.execute("CREATE VIEW v AS SELECT id, random() AS r FROM ids")
        uri = f"duckdb:///{path}"
    schema = {"id": pl.Int64, "r": pl.Float64}
    lf = pl.scan_database("v", uri, schema=schema, order_by=["id"], partition_num=4)
    assert "partitions: 1" in lf.explain()
    assert lf.collect()["id"].to_list() == list(range(300))
    # a table still splits
    lf = pl.scan_database(
        "ids", uri, schema={"id": pl.Int64}, order_by=["id"], partition_num=4
    )
    assert "partitions: 4" in lf.explain()
    assert lf.collect()["id"].to_list() == list(range(300))


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
@pytest.mark.parametrize("source", ["t", "SELECT id, i FROM t"])
def test_scan_sqlite_override_reads_integer_text(
    tmp_path: Path, driver: str, source: str
) -> None:
    # A same-dtype integer override cut reals to integers, but left an integer's text
    # ('7') as text: `i == 7` missed it (sqlite3, SQLAlchemy), ADBC and ConnectorX
    # failed the read. It is read, and compared, as 7.
    _importorskip_sqlite_driver(driver)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, i)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(k, k) for k in range(10, 3001)]
        )
        conn.execute("INSERT INTO t VALUES (3001, '7')")
        conn.execute("INSERT INTO t VALUES (3002, 7.0)")
        conn.commit()
    lf = pl.scan_database(
        source, f"sqlite:///{path}", driver=driver, schema_overrides={"i": pl.Int64}
    )
    assert sorted(lf.filter(pl.col("i") == 7).collect()["id"].to_list()) == [
        3001,
        3002,
    ]
    check_filter_ids(
        lf,
        [
            pl.col("i") < 8,
            pl.col("i").is_in([7, 11]),
            pl.col("i").is_between(5, 9),
        ],
    )


def test_scan_duckdb_hugeint_of_39_digits_retyped(tmp_path: Path) -> None:
    # A HUGEINT of 39 digits fails the `Decimal(38, 0)` it is read as, but failed the
    # read also when retyped to a type that holds it (and the error advised one of them)
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    big = 10**38 + 7
    with duckdb.connect(path) as conn:
        conn.execute(
            f"CREATE TABLE t AS SELECT * FROM (VALUES (1, 5::HUGEINT), "
            f"(2, {big}::HUGEINT), (3, -{big}::HUGEINT)) v(id, h)"
        )
    uri = f"duckdb:///{path}"
    with pytest.raises(Exception, match="can't represent"):
        pl.scan_database("t", uri).collect()
    for source in ["t", "SELECT id, h FROM t"]:
        for dtype, values in [
            (pl.String, ["5", str(big), str(-big)]),
            (pl.Int128, [5, big, -big]),
            (pl.Float64, [5.0, float(big), -float(big)]),
        ]:
            lf = pl.scan_database(source, uri, schema_overrides={"h": dtype})
            assert lf.collect().sort("id")["h"].to_list() == values, (source, dtype)
            assert lf.filter(pl.col("id") == 1).collect()["h"].to_list() == values[:1]
    lf = pl.scan_database("t", uri, schema={"id": pl.Int64, "h": pl.Int128})
    assert lf.collect().sort("id")["h"].to_list() == [5, big, -big]
    assert lf.filter(pl.col("h") > 5).collect()["id"].to_list() == [2]


def test_scan_sqlite_params_dont_hide_volatile_views(tmp_path: Path) -> None:
    # With params=, every catalog lookup was given the parameters too, and failed: the
    # view of `random()` went unchecked, and a filter on it was run in the database,
    # which calls `random()` again for it (half the rows had `r >= 0` for `r < 0`)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(1, 1001)])
        conn.execute("CREATE VIEW vr AS SELECT id, random() AS r FROM t")
        conn.commit()
    uri = f"sqlite:///{path}"
    query = "SELECT * FROM vr WHERE id > ?"
    with closing(sqlite3.connect(path, check_same_thread=False)) as live:
        cases: list[tuple[Any, dict[str, Any]]] = [
            (uri, {"driver": "sqlite3", "params": [0]}),
            (uri, {"driver": "sqlite3", "execute_options": {"parameters": [0]}}),
            (live, {"params": [0]}),
        ]
        for connection, kwargs in cases:
            lf = pl.scan_database(query, connection, **kwargs)
            df = lf.filter(pl.col("r") < 0).collect()
            assert df.height > 0
            assert (df["r"] < 0).all(), kwargs
            lf = pl.scan_database(
                query,
                connection,
                partition_on="id",
                partition_num=4,
                **kwargs,
            )
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()


def test_scan_duckdb_live_connection_sees_its_temporary_objects() -> None:
    # The read ran on a cursor, a connection of its own without the connection's
    # temporary tables and views: a temporary table hiding a table read the hidden one,
    # a temporary view of `random()` hiding a table read the table, and a temporary
    # table alone failed "does not exist"
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE tt AS SELECT 1 AS id")
        conn.execute(
            "CREATE TEMP TABLE tt AS SELECT range AS id, 'abc' AS s FROM range(5)"
        )
        conn.execute("CREATE TEMP TABLE only_temp AS SELECT range AS id FROM range(3)")
        conn.execute("CREATE TABLE vt AS SELECT range AS id FROM range(100)")
        conn.execute("CREATE TABLE src AS SELECT range AS id FROM range(100)")
        conn.execute(
            "CREATE TEMP VIEW vt AS SELECT id, random() AS r FROM src WHERE id < 10"
        )
        df = pl.scan_database("tt", conn).collect()
        assert df.columns == ["id", "s"]
        assert df["id"].to_list() == list(range(5))
        assert (
            pl.scan_database("tt", conn).filter(pl.col("s") == "abc").collect().height
            == 5
        )
        lf = pl.scan_database("SELECT * FROM only_temp WHERE id > 0", conn)
        assert lf.collect()["id"].to_list() == [1, 2]
        assert pl.scan_database("vt", conn).collect().height == 10
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            pl.scan_database("vt", conn, partition_on="id", partition_num=2).collect()
        # two scans of the connection read in one query
        joined = (
            pl.scan_database("tt", conn)
            .join(pl.scan_database("only_temp", conn), on="id")
            .collect()
        )
        assert sorted(joined["id"].to_list()) == [0, 1, 2]
        # the connection is the user's to go on with
        assert conn.execute("SELECT count(*) FROM tt").fetchall() == [(5,)]
    finally:
        conn.close()


def test_scan_duckdb_connections_are_read_as_they_are(tmp_path: Path) -> None:
    # The read ran on a cursor, a connection of its own: a live connection's uncommitted
    # row wasn't read, and a factory's connection with a temporary table of the name of
    # a stored one read the stored one
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.execute("BEGIN")
        conn.execute("INSERT INTO t VALUES (2)")
        assert sorted(pl.scan_database("t", conn).collect()["id"].to_list()) == [1, 2]
        assert conn.execute("SELECT count(*) FROM t").fetchall() == [(2,)]
        conn.execute("COMMIT")
    finally:
        conn.close()
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as stored:
        stored.execute("CREATE TABLE t AS SELECT 1 AS id")

    def factory() -> Any:
        c = duckdb.connect(path)
        c.execute("CREATE TEMP TABLE t AS SELECT 99 AS id")
        return c

    assert pl.scan_database("t", factory).collect()["id"].to_list() == [99]


def test_scan_duckdb_live_connection_with_an_unread_result() -> None:
    # Without temporary objects or a transaction the scan reads on a cursor, which works
    # while the user's own result isn't read to the end (`fetchone()` of a count)
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(100)")
        conn.execute("SELECT id FROM t").fetchone()
        assert pl.scan_database("t", conn).collect().height == 100
        lf = pl.scan_database("t", conn, partition_on="id", partition_num=4)
        assert lf.collect().height == 100
    finally:
        conn.close()


def test_scan_duckdb_catalog_lookups_survive_macros_of_their_helpers() -> None:
    # A user's macro named like a function the plugin's own lookups call must not make
    # a view look like a table
    duckdb = pytest.importorskip("duckdb")
    for helper in [
        "lower(x) AS NULL",
        "list_position(l, x) AS 0",
        "list_append(l, x) AS l",
        "current_database() AS 'zzz'",
        "nullif(a, b) AS NULL",
    ]:
        conn = duckdb.connect()
        try:
            conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(1000)")
            conn.execute("CREATE VIEW rv AS SELECT id, random() AS r FROM t")
            conn.execute("CREATE TEMP TABLE o AS SELECT 1 AS id")
            conn.execute(f"CREATE MACRO {helper}")
            lf = pl.scan_database("rv", conn, partition_on="id", partition_num=3)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()
        finally:
            conn.close()


def test_scan_duckdb_macros_of_operators_and_rewritten_functions(
    tmp_path: Path,
) -> None:
    # sqlglot turns `LIKE`, `extract(…)`, `count(*)` into other calls than the user's
    # macros of the names DuckDB runs for them (`~~`, `date_part`, `count_star`)
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT range AS id, 'a' || range AS s, "
            "DATE '2020-01-01' + range::INT AS dt FROM range(300)"
        )
        conn.execute('CREATE MACRO "~~"(a, b) AS (random() < 0.5)::BIGINT::BOOLEAN')
        conn.execute("CREATE MACRO date_part(a, b) AS (random() * 10)::BIGINT")
        conn.execute('CREATE MACRO "!~~"(a, b) AS (random() < 0.5)')
        conn.execute('CREATE MACRO "/"(a, b) AS (random() * 10)::BIGINT')
    uri = f"duckdb:///{path}"
    for query in [
        "SELECT id FROM t WHERE s LIKE 'a%'",
        "SELECT id FROM t WHERE s NOT LIKE 'a%'",
        "SELECT id FROM t WHERE id / 3 = 1",
        "SELECT id FROM t WHERE extract(year FROM dt) = 1",
    ]:
        lf = pl.scan_database(query, uri, partition_on="id", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()


def test_scan_sqlite_regexp_and_match_operators_call_the_users_functions(
    tmp_path: Path,
) -> None:
    import random

    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path, check_same_thread=False)) as live:
        live.execute("CREATE TABLE t (id INTEGER, s TEXT)")
        live.executemany("INSERT INTO t VALUES (?, 'a')", [(i,) for i in range(200)])
        live.create_function("regexp", 2, lambda *_: random.randint(0, 1))
        lf = pl.scan_database(
            "SELECT id FROM t WHERE s REGEXP 'a'",
            live,
            partition_on="id",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
        # (a user's `count()` doesn't hijack the lookup of the functions)
        live.create_aggregate(
            "count",
            0,
            type(
                "C",
                (),
                {
                    "step": lambda self: None,
                    "finalize": lambda self: random.randint(0, 1),
                },
            ),
        )
        lf = pl.scan_database(
            "SELECT id FROM t WHERE s REGEXP 'a'",
            live,
            partition_on="id",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()


def test_scan_duckdb_views_are_found_along_the_search_path() -> None:
    # A name resolves in `temp`, then the current schema only: with a temporary object
    # present and `search_path = 's1,s2'`, a view of `random()` in `s2` was taken for a
    # table, and partitions of it returned other rows each (lost and duplicated ids)
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    try:
        conn.execute("CREATE SCHEMA s1")
        conn.execute("CREATE SCHEMA s2")
        conn.execute("CREATE TABLE s2.ids AS SELECT range AS id FROM range(1000)")
        conn.execute("CREATE VIEW s2.v AS SELECT id, random() AS r FROM s2.ids")
        conn.execute("CREATE TEMP TABLE other AS SELECT 1 AS id")
        conn.execute("SET search_path = 's1,s2'")
        assert pl.scan_database("v", conn).collect().height == 1000
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            pl.scan_database("v", conn, partition_on="id", partition_num=4).collect()
        # a table of the name earlier in the path hides it
        conn.execute("CREATE TABLE s1.v AS SELECT range AS id FROM range(10)")
        lf = pl.scan_database("v", conn, partition_on="id", partition_num=2)
        assert sorted(lf.collect()["id"].to_list()) == list(range(10))
    finally:
        conn.close()


def test_scan_duckdb_cursor_has_the_connections_session_state(tmp_path: Path) -> None:
    # A cursor starts with the defaults: `SET search_path`, `USE`, `SET TimeZone`,
    # `integer_division`, `SET VARIABLE` were lost, and the scan read `main.t` where the
    # connection's own `SELECT * FROM t` reads `s.t`
    duckdb = pytest.importorskip("duckdb")
    cases = [
        (["SET search_path = 's'"], "SELECT id FROM t"),
        (["USE s"], "SELECT id FROM t"),
        (["USE memory.s"], "SELECT id FROM t"),
        (
            ["SET TimeZone = 'Asia/Tokyo'"],
            "SELECT CAST(TIMESTAMPTZ '2020-01-01 00:00:00+00' AS VARCHAR) AS x",
        ),
        (["SET integer_division = true"], "SELECT 7 / 2 AS x"),
        (["SET VARIABLE v = 42"], "SELECT getvariable('v') AS x"),
        (["SET VARIABLE w = 'it''s'"], "SELECT getvariable('w') AS x"),
        (["SET VARIABLE k = NULL"], "SELECT getvariable('k') IS NULL AS x"),
    ]
    for setup, query in cases:
        conn = duckdb.connect()
        try:
            conn.execute("CREATE TABLE main.t AS SELECT 1 AS id")
            conn.execute("CREATE SCHEMA s")
            conn.execute("CREATE TABLE s.t AS SELECT 99 AS id")
            default_zone = conn.execute("SELECT current_setting('TimeZone')").fetchone()
            for statement in setup:
                conn.execute(statement)
            native = conn.execute(query).fetchall()
            settings = "SELECT name, value FROM duckdb_settings() ORDER BY name"
            before = conn.execute(settings).fetchall()
            assert pl.scan_database(query, conn).collect().rows() == native, setup
            if setup[0].startswith(("SET search_path", "USE")):
                table = pl.scan_database("t", conn).collect()
                assert table["id"].to_list() == [99], setup
            # the cursor's settings don't reach the connection, nor its other cursors
            assert conn.execute(settings).fetchall() == before
            assert (
                conn.cursor().execute("SELECT current_setting('TimeZone')").fetchone()
                == default_zone
            )
        finally:
            conn.close()
    # a factory that sets the search path itself
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as stored:
        stored.execute("CREATE TABLE t AS SELECT 1 AS id")
        stored.execute("CREATE SCHEMA s")
        stored.execute("CREATE TABLE s.t AS SELECT 99 AS id")

    def factory() -> Any:
        c = duckdb.connect(path)
        c.execute("SET search_path = 's'")
        return c

    assert pl.scan_database("t", factory).collect()["id"].to_list() == [99]


@pytest.mark.parametrize("shadow", [False, True])
def test_scan_duckdb_session_copy_handles_null_variables(shadow: bool) -> None:
    duckdb = pytest.importorskip("duckdb")
    from polars_database._bridge import _duckdb_copy_session

    conn = duckdb.connect()
    try:
        variables = {
            "untyped": "NULL",
            "text_null": "NULL::VARCHAR",
            "integer_null": "NULL::INTEGER",
            "list_null": "NULL::INTEGER[]",
            "text": "'NULL'",
            "number": "1",
        }
        for name, expression in variables.items():
            conn.execute(f"SET VARIABLE {name} = {expression}")
        if shadow:
            conn.execute("CREATE MACRO getvariable(x) AS 'not-null'")
        cursor = conn.cursor()
        try:
            assert _duckdb_copy_session(conn, cursor)
            for name in variables:
                query = (
                    f"SELECT system.main.getvariable('{name}'), "
                    f"typeof(system.main.getvariable('{name}')), system.main.getvariable('{name}') IS NULL"
                )
                assert (
                    cursor.execute(query).fetchall() == conn.execute(query).fetchall()
                )
                assert (
                    pl.scan_database(
                        f"SELECT system.main.getvariable('{name}') IS NULL AS x", conn
                    )
                    .collect()
                    .rows()
                    == conn.execute(
                        f"SELECT system.main.getvariable('{name}') IS NULL"
                    ).fetchall()
                )
        finally:
            cursor.close()
    finally:
        conn.close()


def test_scan_duckdb_volatile_shadow_message_names_only_the_volatile_macros(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
        conn.execute('CREATE MACRO "+"(a, b) AS a - (0 - b)')
        conn.execute('CREATE MACRO "-"(a, b) AS random()')
    lf = pl.scan_database(
        "SELECT id FROM ids", f"duckdb:///{path}", partition_on="id", partition_num=2
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match='"-"') as e:
        lf.collect()
    assert '"+"' not in str(e.value)


def test_scan_duckdb_search_path_with_quoted_entries_and_a_list_apply_macro() -> None:
    # The search path was split at every comma, and read with a list comprehension,
    # which DuckDB runs as the user's `list_apply`: a `random()` view in a schema
    # `"s,2"`, or with such a macro in the way, was taken for a table (partitions lost
    # rows)
    duckdb = pytest.importorskip("duckdb")
    for temp, macro, path in [
        (True, False, "'s1,\"s,2\"'"),
        (False, False, "'s1,\"s,2\"'"),
        (True, False, "'S1,\"S,2\"'"),
        (True, True, "'s1,\"s,2\"'"),
        (False, True, "'s1,\"s,2\"'"),
    ]:
        conn = duckdb.connect()
        try:
            conn.execute("CREATE SCHEMA s1")
            conn.execute('CREATE SCHEMA "s,2"')
            conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(1000)")
            conn.execute(
                'CREATE VIEW "s,2".v AS SELECT id, random() AS r FROM main.ids'
            )
            if temp:
                conn.execute("CREATE TEMP TABLE other AS SELECT 1 AS id")
            if macro:
                conn.execute("CREATE MACRO list_apply(l, f) AS [false]")
            conn.execute(f"SET search_path = {path}")
            lf = pl.scan_database("v", conn, partition_on="id", partition_num=4)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()
            # a table earlier on the path hides the view
            conn.execute("CREATE TABLE s1.v AS SELECT range AS id FROM range(10)")
            lf = pl.scan_database("v", conn, partition_on="id", partition_num=2)
            assert sorted(lf.collect()["id"].to_list()) == list(range(10))
        finally:
            conn.close()


def test_scan_duckdb_macros_of_operators_and_built_ins_the_query_doesnt_name(
    tmp_path: Path,
) -> None:
    # `id + 1`, `id * 2`, `trim(LEADING ..)` (`ltrim`) and `[e FOR e IN l]`
    # (`list_apply`) run the user's macros of those names: a volatile one makes the
    # sources of the database volatile, where the parser can't say which names a query
    # runs
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
    uri = f"duckdb:///{path}"
    queries = {
        "+": "SELECT id, id + 1 AS r FROM ids",
        "*": "SELECT id, id * 2 AS r FROM ids",
        "ltrim": "SELECT id, trim(id::VARCHAR, '0') AS a, "
        "trim(LEADING '0' FROM id::VARCHAR) AS b FROM ids",
        "list_apply": "SELECT id, [e + 1 FOR e IN [1, 2]] AS l FROM ids",
    }

    def partitioned(query: str) -> list[int]:
        lf = pl.scan_database(query, uri, partition_on="id", partition_num=4)
        return sorted(lf.collect()["id"].to_list())

    # no such macro: the sources are as repeatable as before (`SELECT *` too)
    for query in [*queries.values(), "SELECT * FROM ids"]:
        assert partitioned(query) == list(range(300))
    # a macro that is stable doesn't matter
    with duckdb.connect(path) as conn:
        conn.execute('CREATE MACRO "+"(a, b) AS a - (0 - b)')
    assert partitioned(queries["+"]) == list(range(300))
    bodies = {
        "+": 'CREATE OR REPLACE MACRO "+"(a, b) AS random()',
        "*": 'CREATE MACRO "*"(a, b) AS random()::BIGINT',
        "ltrim": "CREATE MACRO ltrim(a, b) AS random()::VARCHAR",
        "list_apply": "CREATE MACRO list_apply(l, f) AS [random()]",
    }
    for name, create in bodies.items():
        with duckdb.connect(path) as conn:
            conn.execute(create)
        for query in queries.values():
            # (the macro of `name` reaches every query of the database)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                partitioned(query)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            partitioned("SELECT * FROM ids")
        # a read without partitions is as it was
        assert pl.scan_database(queries[name], uri).collect().height == 300
        with duckdb.connect(path) as conn:
            conn.execute(f'DROP MACRO "{name}"')
    assert partitioned("SELECT * FROM ids") == list(range(300))


def test_scan_duckdb_query_sqlglot_cant_parse_still_checks_its_functions(
    tmp_path: Path,
) -> None:
    # `(3 !)` (the factorial postfix) isn't parsed: the functions the query calls by
    # name went unchecked, and a volatile macro of one was partitioned
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
        conn.execute("CREATE MACRO myrand(x) AS random()::BIGINT")
    uri = f"duckdb:///{path}"
    query = "SELECT id FROM ids WHERE (3 !) = 6 AND myrand(id) >= 0"
    assert pl.scan_database(query, uri).collect().height == 300
    lf = pl.scan_database(query, uri, partition_on="id", partition_num=4)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        lf.collect()
    # a query that parses and calls no volatile macro is partitioned
    with duckdb.connect(path) as conn:
        conn.execute("DROP MACRO myrand")
        conn.execute("CREATE MACRO myrand(x) AS x")
    lf = pl.scan_database(query, uri, partition_on="id", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == list(range(300))


def test_scan_duckdb_python_functions_with_side_effects_are_volatile() -> None:
    # A Python function registered without `side_effects=True` is CONSISTENT in the
    # catalog and counts as repeatable; with it, VOLATILE, and a partitioned read of
    # a query calling it is refused
    duckdb = pytest.importorskip("duckdb")
    import random

    try:
        from duckdb.sqltypes import BIGINT, DOUBLE
    except ImportError:  # DuckDB before 1.4
        from duckdb.typing import BIGINT, DOUBLE  # type: ignore[no-redef]

    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
        conn.create_function(
            "py_rand", lambda x: random.random(), [BIGINT], DOUBLE, side_effects=True
        )
        lf = pl.scan_database(
            "SELECT id, py_rand(id) AS r FROM ids",
            conn,
            partition_on="id",
            partition_num=4,
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()
    finally:
        conn.close()


def test_scan_sqlite_user_function_replacing_a_built_in(tmp_path: Path) -> None:
    # `abs` is a function sqlglot knows, but `create_function` replaces it: a query of
    # it was taken as stable (partitions lost rows, a filter on it was run in the
    # database, which calls it again)
    import random

    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path, check_same_thread=False)) as live:
        live.execute("CREATE TABLE t (id INTEGER)")
        live.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(200)])
        live.commit()
        live.execute("CREATE VIEW va AS SELECT id, abs(id) AS r FROM t")
        live.execute("CREATE VIEW vl AS SELECT id, lower(id) AS r FROM t")
        live.create_function("abs", 1, lambda _: random.uniform(-1, 1))
        # (the lookup of functions by name doesn't call the user's `lower()` either)
        live.create_function("lower", 1, lambda _: random.uniform(-1, 1))
        for source in ["va", "vl"]:
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                pl.scan_database(
                    source, live, partition_on="id", partition_num=4
                ).collect()
        # a function sqlglot knows under another name, and an operator that calls one
        live.create_function("instr", 2, lambda *_: random.uniform(-1, 1))
        live.create_function("like", 2, lambda *_: random.randint(0, 1))
        for source in [
            "SELECT id, instr(id, 1) AS r FROM t",
            "SELECT id, id LIKE '1%' AS r FROM t",
        ]:
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                pl.scan_database(
                    source, live, partition_on="id", partition_num=4
                ).collect()
        query = "SELECT id, abs(id) AS r FROM t"
        with pytest.raises(pl.exceptions.InvalidOperationError, match="abs"):
            pl.scan_database(query, live, partition_on="id", partition_num=4).collect()
        df = pl.scan_database(query, live).filter(pl.col("r") > 0).collect()
        assert (df["r"] > 0).all()
    # the built-in one is as stable as before
    lf = pl.scan_database(
        "SELECT id, abs(id) AS d FROM t",
        f"sqlite:///{path}",
        partition_on="id",
        partition_num=4,
    )
    assert sorted(lf.collect()["id"].to_list()) == list(range(200))


def test_scan_duckdb_macro_replacing_a_built_in(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE ids AS SELECT range AS id FROM range(300)")
    uri = f"duckdb:///{path}"
    query = "SELECT id, abs(id) AS a FROM ids"
    lf = pl.scan_database(query, uri, partition_on="id", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == list(range(300))
    with duckdb.connect(path) as conn:
        conn.execute("CREATE MACRO abs(x) AS random()")
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        pl.scan_database(query, uri, partition_on="id", partition_num=4).collect()
    # names sqlglot knows as another (`ltrim`), and `lower`, which the lookups call too
    with duckdb.connect(path) as conn:
        conn.execute("CREATE MACRO ltrim(x) AS random()::VARCHAR")
        conn.execute("CREATE MACRO lower(x) AS random()::VARCHAR")
    for query in [
        "SELECT id, ltrim(id::VARCHAR) AS a FROM ids",
        "SELECT id, lower(id::VARCHAR) AS a FROM ids",
    ]:
        lf = pl.scan_database(query, uri, partition_on="id", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()


def test_scan_sqlite_partly_pushed_filter_on_a_volatile_column_under_schema(
    tmp_path: Path,
) -> None:
    # Only the stable `id` conditions of `(id < 10 & s.starts_with(..)) | id == 20` are
    # pushed; the filter was rejected for the column `s` it names, which isn't compared
    # in the database
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in range(300)])
        conn.execute(
            "CREATE VIEW v AS SELECT id, printf('%d', random()) AS s, random() AS r FROM ids"
        )
        conn.commit()
    lf = pl.scan_database(
        "v",
        f"sqlite:///{path}",
        schema={"id": pl.Int64, "s": pl.String, "r": pl.Int64},
    )
    pred = ((pl.col("id") < 10) & pl.col("s").str.starts_with("-")) | (
        pl.col("id") == 20
    )
    df = lf.filter(pred).collect()
    assert set(df["id"].to_list()) <= set(range(10)) | {20}
    assert (df.filter(pl.col("id") != 20)["s"].str.starts_with("-")).all()
    # a filter on the column itself is still refused
    with pytest.raises(pl.exceptions.InvalidOperationError, match="a filter on 'r'"):
        lf.filter(pl.col("r") > 0).collect()


def test_scan_duckdb_two_part_names_of_an_attached_database(tmp_path: Path) -> None:
    # `db.v` names the schema `db` of the current database, or else the attached
    # database `db` (its schema `main`), as DuckDB resolves it; it was only taken for a
    # schema: a view of `nextval` there wasn't found (partitions accepted, each read
    # other values), nor a table's types (a string filter wasn't pushed)
    duckdb = pytest.importorskip("duckdb")
    other = str(tmp_path / "other.duckdb")
    with duckdb.connect(other) as conn:
        conn.execute(
            "CREATE TABLE base AS SELECT range AS id, 'a' AS s FROM range(1000)"
        )
        conn.execute("CREATE SEQUENCE seq")
        conn.execute("CREATE VIEW vseq AS SELECT id, nextval('seq') AS n FROM base")
        conn.execute("CREATE VIEW vr AS SELECT id, random() AS r FROM base")
    conn = duckdb.connect()
    try:
        conn.execute(f"ATTACH '{other}' AS otherdb")
        for source in [
            "otherdb.vseq",
            "otherdb.vr",
            "SELECT * FROM otherdb.vr",
            "otherdb.main.vr",
        ]:
            lf = pl.scan_database(source, conn, partition_on="id", partition_num=4)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="partitioned"
            ):
                lf.collect()
        lf = pl.scan_database("otherdb.base", conn).filter(pl.col("s") == "a")
        assert "WHERE" in _sql_of(lf)
        assert lf.collect().height == 1000
        # a schema of the current database by another name is still found
        conn.execute("CREATE SCHEMA s2")
        conn.execute("CREATE TABLE s2.base AS SELECT 1 AS id, 2 AS x")
        lf = pl.scan_database("s2.base", conn).filter(pl.col("x") == 2)
        assert lf.collect().columns == ["id", "x"]
        assert "WHERE" in _sql_of(lf)
    finally:
        conn.close()


def test_scan_duckdb_views_with_lambdas(tmp_path: Path) -> None:
    # DuckDB writes a view's lambdas (and list comprehensions) parenthesised, which
    # sqlglot didn't parse: the view couldn't be inspected, so no filter was pushed on
    # it and partitions were refused
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE base AS SELECT range AS id, [range, range + 1] AS l "
            "FROM range(1000)"
        )
        conn.execute(
            "CREATE VIEW v AS SELECT id, list_transform(l, lambda x: x + 1) AS g, "
            "[x * 2 FOR x IN l IF x > 0] AS c FROM base"
        )
        conn.execute("CREATE VIEW vr AS SELECT id, random() AS r FROM base")
    uri = f"duckdb:///{path}"
    lf = pl.scan_database("v", uri).filter(pl.col("id") < 10)
    assert "WHERE" in _sql_of(lf)
    assert lf.collect().height == 10
    lf = pl.scan_database("v", uri, partition_on="id", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == list(range(1000))
    # a filter on a column of `random()` is run in the database too: DuckDB filters the
    # values the view returns, as Postgres does (SQLite and MySQL call it again)
    lf = pl.scan_database("vr", uri).filter(pl.col("r") < 0.5)
    assert "WHERE" in _sql_of(lf)
    df = lf.collect()
    assert df.height > 0
    assert (df["r"] < 0.5).all()


@pytest.mark.parametrize("driver", ["sqlite3"])
def test_scan_query_that_isnt_repeatable_counts_batches_of_other_dtypes(
    tmp_path: Path, driver: str
) -> None:
    # the rows a count reads (and the query's schema sample) were stacked as the driver
    # typed each batch: an integer batch and a float batch didn't stack. (ADBC refuses
    # the column's change of type itself.)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, x NUMERIC)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, i if i < 100 else i + 0.5) for i in range(250)],
        )
        conn.commit()
    lf = pl.scan_database(
        "SELECT * FROM t WHERE abs(random()) % 1 = 0",
        f"sqlite:///{path}",
        driver=driver,
        batch_size=50,
        schema_overrides={"x": pl.Float64},
    )
    assert lf.collect_schema()["x"] == pl.Float64
    tail = lf.tail(3).collect()
    assert tail["x"].to_list() == [247.5, 248.5, 249.5]
    assert lf.select(pl.len()).collect().item() == 250
    # a repeatable query is sampled the same way
    lf = pl.scan_database(
        "SELECT * FROM t", f"sqlite:///{path}", driver=driver, batch_size=50
    )
    assert lf.collect()["x"].dtype == pl.Float64


@pytest.mark.parametrize("stop", ["head", "collect_batches", "held_iterator"])
def test_scan_duckdb_stopped_early_exits_cleanly(tmp_path: Path, stop: str) -> None:
    # The cleanup of a stopped read ran on a thread of its own and outlived `collect()`;
    # a query of a dropped `collect_batches()` iterator runs on until it notices. Their
    # driver calls then ran while Python finalized, which aborts the process (psycopg2,
    # MySQL over ADBC). DuckDB's module reported the connections leaked, also after a
    # complete read: the scan's last session was released only at pyo3's next use of the
    # GIL. A `collect_batches()` iterator held until exit stops its query only once
    # Python finalizes: its driver calls are skipped then (its connection is left
    # unclosed).
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(2000000)")
    read = {
        "head": "assert lf.head(3).collect().height == 3",
        "collect_batches": "for df in lf.collect_batches():\n    break",
        "held_iterator": "it = lf.collect_batches()\nnext(it)",
    }[stop]
    code = (
        textwrap.dedent(
            f"""
        import polars as pl
        lf = pl.scan_database("t", {f"duckdb:///{path}"!r}, batch_size=1000)
        assert lf.collect().height == 2000000
        """
        )
        + read
    )
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr
        if stop != "held_iterator":
            assert "leaked" not in out.stderr


def test_scan_sqlalchemy_older_than_2_is_refused(
    shop_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On SQLAlchemy 1.4 the session setup failed (`Connection` has no `commit`), and so
    # did bound parameters and SQLite connections used across threads
    sa = pytest.importorskip("sqlalchemy")
    engine = sa.create_engine(f"sqlite:///{shop_db}")
    monkeypatch.setattr(sa, "__version__", "1.4.54")
    try:
        for connection, driver in (
            (f"sqlite:///{shop_db}", "sqlalchemy"),
            (engine, None),
        ):
            with pytest.raises(Exception, match=r"sqlalchemy.* 2\.0"):
                pl.scan_database("orders", connection, driver=driver).collect()
    finally:
        engine.dispose()


def test_scan_connectorx_older_than_the_minimum_is_refused(
    shop_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ConnectorX 0.3 read decimals wider than a float's digits rounded, silently
    cx = pytest.importorskip("connectorx")
    from polars_database import _bridge

    monkeypatch.setattr(cx, "__version__", "0.3.3")
    with pytest.raises(Exception, match=rf"connectorx {_bridge._MIN_CONNECTORX}"):
        pl.scan_database("orders", shop_uri, driver="connectorx").collect()


@pytest.mark.parametrize("scheme", ["sqlite+x.toml", "sqlite+Evil.so"])
def test_scan_uri_driver_part_is_never_a_path(tmp_path: Path, scheme: str) -> None:
    # `+x.toml` read a manifest from the cwd and loaded the library it names: a URI
    # often comes from configuration (`DATABASE_URL`)
    (tmp_path / "x.toml").write_text('[Driver]\nshared = "/nonexistent/x.so"\n')
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(Exception, match="is not a driver name"):
            pl.scan_database("t", f"{scheme}:///{tmp_path}/a.db").collect()
    finally:
        os.chdir(cwd)


def test_scan_odbc_connect_password_is_redacted() -> None:
    from polars_database import _bridge

    uri = "mssql+pyodbc://@h/?odbc_connect=DRIVER%3D%7BX%7D%3BPWD%3DHunter2Secret"
    secrets = _bridge._secrets(uri, None, {})
    assert "Hunter2Secret" in secrets
    assert "Hunter2Secret" not in _bridge._scrub(f"failed: {uri}", secrets)


@pytest.mark.parametrize("driver", ["adbc", "connectorx", "sqlite3", "sqlalchemy"])
def test_scan_sqlite_missing_column_is_an_error(tmp_path: Path, driver: str) -> None:
    # SQLite reads an unknown double-quoted name as a string: a `schema=` column the
    # table doesn't have read as its own name in every row
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, a INTEGER)")
        conn.execute("INSERT INTO t VALUES (1, 2)")
        conn.commit()
    _skip_without_driver(driver)
    lf = pl.scan_database(
        "t",
        f"sqlite:///{path}",
        driver=driver,
        schema={"id": pl.Int64, "a": pl.Int64, "nope": pl.String},
    )
    with pytest.raises(Exception, match="nope"):
        lf.collect()


def _skip_without_driver(driver: str) -> None:
    module = {
        "adbc": "adbc_driver_manager",
        "connectorx": "connectorx",
        "sqlalchemy": "sqlalchemy",
    }.get(driver)
    if module is not None:
        pytest.importorskip(module)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Datetime("ms"), ["2026-01-01 10:00:00.789123"]),
        (pl.Date, ["2026-01-01 10:00:00"]),
    ],
)
def test_scan_schema_coarser_than_the_column_is_an_error(
    tmp_path: Path, dtype: PolarsDataType, values: list[str]
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t (ts TIMESTAMP)")
        conn.executemany("INSERT INTO t VALUES (?)", [[v] for v in values])
    lf = pl.scan_database("t", f"duckdb:///{path}", schema={"ts": dtype})
    with pytest.raises(Exception, match="would cut"):
        lf.collect()
    # exact values read
    with duckdb.connect(path) as conn:
        conn.execute("UPDATE t SET ts = date_trunc('day', ts)")
    assert lf.collect()["ts"].dtype == dtype


def test_scan_schema_float32_overflow_is_an_error(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1e300::DOUBLE AS f UNION ALL SELECT 0.5")
    lf = pl.scan_database("t", f"duckdb:///{path}", schema={"f": pl.Float32})
    with pytest.raises(Exception, match="Float32"):
        lf.collect()


def test_scan_in_an_atexit_hook_registered_before_the_first_scan(
    shop_db: Path,
) -> None:
    # `polars_database` is imported at the first scan, so its `atexit` hook ran before a
    # user's registered earlier, and stopped every later scan ("Python is exiting")
    code = textwrap.dedent(
        f"""
        import atexit
        import polars as pl

        def hook():
            n = pl.scan_database("orders", {f"sqlite:///{shop_db}"!r}).collect().height
            print("rows", n)

        atexit.register(hook)
        pl.scan_database("orders", {f"sqlite:///{shop_db}"!r}).head(1).collect()
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert f"rows {N_ROWS}" in out.stdout, out.stderr


def test_scan_live_duckdb_connection_is_released_with_the_frame(tmp_path: Path) -> None:
    # The scan's Python objects were released only at the plugin's next use of the GIL:
    # the connection kept its file open after `del lf`
    import gc

    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE t AS SELECT 1 AS id")
    before = sys.getrefcount(con)
    lf = pl.scan_database("t", con)
    assert lf.collect().height == 1
    del lf
    gc.collect()
    assert sys.getrefcount(con) == before


def test_scan_sqlite3_factory_subclass_from_main(shop_db: Path) -> None:
    # classified by its own module (`__main__`), it had no backend
    code = textwrap.dedent(
        f"""
        import sqlite3
        import polars as pl

        class MyConn(sqlite3.Connection):
            pass

        conn = sqlite3.connect(
            {str(shop_db)!r}, factory=MyConn, check_same_thread=False
        )
        print("rows", pl.scan_database("orders", conn).collect().height)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert f"rows {N_ROWS}" in out.stdout


@pytest.mark.parametrize(
    ("dtype", "lo", "hi"),
    [
        ("BIGINT", "-9223372036854775808", "9223372036854775807"),
        ("UBIGINT", "0", "18446744073709551615"),
        ("DECIMAL(38,0)", "-" + "9" * 38, "9" * 38),
    ],
)
def test_scan_partitions_over_a_range_wider_than_usize(
    tmp_path: Path, dtype: str, lo: str, hi: str
) -> None:
    # the range's width overflowed the partition count
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            f"CREATE TABLE t AS SELECT * FROM (VALUES (({lo})::{dtype}), "
            f"(({hi})::{dtype})) v(x)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}", partition_on="x", partition_num=2)
    assert sorted(str(v) for v in lf.collect()["x"]) == sorted([lo, hi])


def test_scan_fixed_size_array_column(tmp_path: Path) -> None:
    # the plugin was built without `dtype-array`
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, [1, 2]::INTEGER[2]), "
            "(2, NULL), (3, [NULL, 4]::INTEGER[2])) v(id, a)"
        )
    uri = f"duckdb:///{path}"
    expected = pl.DataFrame(
        {"id": [1, 2, 3], "a": [[1, 2], None, [None, 4]]},
        schema={"id": pl.Int32, "a": pl.Array(pl.Int32, 2)},
    )
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, uri)
        assert_frame_equal(lf.collect().sort("id"), expected)
        out = lf.filter(pl.col("a").arr.first() == 1).collect()
        assert out["id"].to_list() == [1]


def test_scan_duckdb_bit_strings_in_a_query_are_text(tmp_path: Path) -> None:
    # a query planned the type's bytes and read their text: `b'0101'`
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT 1 AS id, '0101'::BIT AS b, "
            "123456789012345678901234567890123456789012::VARINT AS v"
        )
    uri = f"duckdb:///{path}"
    expected = pl.DataFrame(
        {"id": [1], "b": ["0101"], "v": ["123456789012345678901234567890123456789012"]},
        schema={"id": pl.Int32, "b": pl.String, "v": pl.String},
    )
    for source in ["t", "SELECT * FROM t"]:
        assert_frame_equal(pl.scan_database(source, uri).collect(), expected)


def test_scan_duckdb_maps_and_nested_arrays(tmp_path: Path) -> None:
    # without `dtype-map` / `dtype-array`, a MAP read as a list of structs and a list of
    # arrays failed
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT 1 AS id, MAP {'k': 1} AS m, "
            "[[1, 2]::INTEGER[2], NULL] AS la"
        )
    uri = f"duckdb:///{path}"
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, uri)
        expected = pl.read_database(
            "SELECT * FROM t", duckdb.connect(path, read_only=True)
        )
        assert lf.collect_schema() == expected.schema, source
        assert_frame_equal(lf.collect(), expected)
        assert lf.filter(pl.col("m").map.len() == 1).collect().height == 1


@pytest.mark.parametrize(
    "dtype", [pl.Float16, pl.Array(pl.Int64, 2), pl.Map(pl.String, pl.Int64)]
)
def test_scan_overrides_of_dtypes_the_plugin_was_built_without(
    shop_uri: str, dtype: pl.DataType
) -> None:
    # Polars panicked on Arrow types the plugin's own Polars had no dtype for
    lf = pl.scan_database("orders", shop_uri, schema_overrides={"status": dtype})
    assert lf.collect_schema()["status"] == dtype


# Round 11 review findings.


def test_sqlite_query_without_pushdown(shop_uri: str, expected: pl.DataFrame) -> None:
    # the select items were qualified by `"_pl"`, the query aliased `pl_q`: "no such
    # column"
    for sql in [
        "SELECT id, amount FROM orders",
        "SELECT * FROM orders WHERE id > 0",
    ]:
        out = pl.scan_database(sql, shop_uri, pushdown=False).collect().sort("id")
        assert_frame_equal(out, expected.select(out.columns), check_dtypes=False)


def test_query_columns_differing_in_case(shop_uri: str, tmp_path: Path) -> None:
    # SQLite renamed the second to `x:1` (DuckDB `x_1`), and `"_pl"."x"` read the first
    sql = 'SELECT id AS "X", amount AS "x" FROM orders'
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="differ only in case"
    ):
        pl.scan_database(sql, shop_uri).collect()
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "case.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE orders AS SELECT 1 AS id, 2.5 AS amount")
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="differ only in case"
    ):
        pl.scan_database(sql, f"duckdb:///{path}").collect()


def test_datetime_read_as_time_is_refused(tmp_path: Path) -> None:
    # `schema={"ts": pl.Time}` over a timestamp dropped the date
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "time.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT TIMESTAMP '2020-01-02 03:04:05' AS ts "
            "UNION ALL SELECT TIMESTAMP '1970-01-01 01:00:00'"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}", schema={"ts": pl.Time})
    with pytest.raises(Exception, match="would drop"):
        lf.collect()


_DUCK_SLOW = (
    "(SELECT count(*) FROM range(40000) a, range(40000) b "
    "WHERE a.range + b.range < 0) >= 0"
)


@pytest.mark.parametrize("kind", ["uri", "factory"])
def test_duckdb_error_cancels_the_other_partitions(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Event

    from polars.io.database._executor import ConnectionExecutor

    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range::INTEGER AS id FROM range(100)")

    running = Event()
    release = Event()
    interrupted = Event()
    slow_cursor: Any = None
    execute = ConnectionExecutor.execute
    interrupt = duckdb.DuckDBPyConnection.interrupt

    def wait_for_interrupt() -> bool:
        running.set()
        assert release.wait(15), "the running DuckDB cursor was never interrupted"
        return True

    def interrupt_cursor(cursor: Any) -> None:
        interrupt(cursor)
        if cursor is slow_cursor:
            release.set()

    def execute_partition(self: Any, query: str, **kwargs: Any) -> Any:
        nonlocal slow_cursor
        if "error('boom')" in query:
            assert running.wait(15), "the other DuckDB partition never started"
        if "wait_for_interrupt()" in query:
            slow_cursor = self.cursor
            slow_cursor.create_function(
                "wait_for_interrupt",
                wait_for_interrupt,
                return_type="BOOLEAN",
                side_effects=True,
            )
            # Execute eagerly so the interrupt is observed here on the actual cursor.
            try:
                slow_cursor.execute(query)
            except duckdb.InterruptException:
                interrupted.set()
                raise
            msg = "the DuckDB statement completed without interruption"
            raise AssertionError(msg)
        return execute(self, query, **kwargs)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "interrupt", interrupt_cursor)
    monkeypatch.setattr(ConnectionExecutor, "execute", execute_partition)
    source: Any = f"duckdb:///{path}" if kind == "uri" else lambda: duckdb.connect(path)
    lf = pl.scan_database(
        "t",
        source,
        partition_predicates=[
            "id < 10 AND error('boom') IS NULL",
            "id >= 10 AND wait_for_interrupt()",
        ],
        max_connections=2,
    )
    with pytest.raises(Exception, match="boom"):
        lf.collect()
    assert interrupted.is_set(), "the DuckDB statement did not report an interrupt"


def test_duckdb_dropped_batches_dont_abort_the_exit(tmp_path: Path) -> None:
    # the exit cancel interrupted the wrong connection: DuckDB aborted the process
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range::INTEGER AS id FROM range(100)")
    slow = _DUCK_SLOW.replace("40000", "60000")
    preds = ["id < 10"] + [
        f"id >= {10 * i} AND id < {10 * (i + 1)} AND {slow}" for i in range(1, 4)
    ]
    code = textwrap.dedent(
        f"""
        import polars as pl
        lf = pl.scan_database(
            "t",
            {f"duckdb:///{path}"!r},
            partition_predicates={preds!r},
            max_connections=4,
        )
        it = lf.collect_batches(engine="streaming")
        next(it)
        del it
        """
    )
    start = monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert monotonic() - start < 20


def test_cancel_before_the_statement_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a cancel in the setup before the driver call found nothing to stop, and the
    # statement then ran to its end
    from polars_database import _bridge

    path = str(tmp_path / "t.db")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(100)])
        conn.execute("CREATE TABLE w (x INTEGER)")
        conn.executemany("INSERT INTO w VALUES (?)", [(i,) for i in range(30000)])
        conn.commit()
    slow = "(SELECT count(*) FROM w a, w b) > 0"
    preds = ["id < 10 AND abs(-9223372036854775807-1) > 0"] + [
        f"id >= {10 * i} AND id < {10 * (i + 1)} AND {slow}" for i in range(1, 4)
    ]
    original_cursor = _bridge._Session._server_cursor
    original_execute = _bridge._Session.execute

    def execute(self: Any, sql: str, batch_size: Any, bind: bool = True) -> Any:
        # the planner's own lookups (`bind=False`) run at the normal speed
        self.slow_setup = bind
        return original_execute(self, sql, batch_size, bind)

    def slow_setup(self: Any, batch_size: int) -> Any:
        if getattr(self, "slow_setup", False):
            sleep(0.5)
        return original_cursor(self, batch_size)

    monkeypatch.setattr(_bridge._Session, "execute", execute)
    monkeypatch.setattr(_bridge._Session, "_server_cursor", slow_setup)
    lf = pl.scan_database(
        "t",
        f"sqlite:///{path}",
        driver="sqlite3",
        partition_predicates=preds,
        max_connections=4,
    )
    start = monotonic()
    with pytest.raises(Exception, match="integer overflow"):
        lf.collect()
    assert monotonic() - start < 3


def test_cancelled_session_starts_no_statement(tmp_path: Path) -> None:
    # `cancel` is sticky: one that came before the statement had nothing to stop
    from polars_database._bridge import _Session

    path = str(tmp_path / "t.db")
    with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
        session = _Session(conn, owned=True, execute_options={})
        assert session.can_cancel
        session.cancel()
        with pytest.raises(RuntimeError, match="cancelled"):
            list(session.execute("SELECT 1", None))


def test_cancelled_adbc_session_starts_no_statement(tmp_path: Path) -> None:
    adbc_sqlite = pytest.importorskip("adbc_driver_sqlite")
    from polars_database._bridge import adbc_connect

    session = adbc_connect(
        adbc_sqlite._driver_path(), None, [("uri", f"file:{tmp_path}/a.db")], []
    )
    try:
        session.cancel()
        with pytest.raises(RuntimeError, match="cancelled"):
            session.execute("SELECT 1", None)
        with pytest.raises(RuntimeError, match="cancelled"):
            session.execute_update("SELECT 1")
        assert not session._open
    finally:
        session.close()


class _FakeHandle:
    def __arrow_c_stream__(self) -> Any:  # pragma: no cover
        msg = "a cancelled statement's stream was taken"
        raise AssertionError(msg)


def _fake_adbc_session(stmt: Any) -> Any:
    import threading

    from polars_database._bridge import _AdbcSession

    session = object.__new__(_AdbcSession)
    session._lock = threading.Lock()
    session._handle_lock = threading.Lock()
    session._open = []
    session._closed = False
    session._cancelled = False
    session._issuing = False
    session._conn = None  # type: ignore[assignment]
    session._statement = lambda sql, hint: stmt  # type: ignore[method-assign]
    return session


@pytest.mark.parametrize("startup_delay", [0.15, 1.6])
def test_adbc_cancel_is_repeated_until_the_statement_runs(startup_delay: float) -> None:
    # before the driver sent the query a cancel was refused (INVALID_STATE) and lost
    import threading

    pytest.importorskip("adbc_driver_manager")
    from adbc_driver_manager import AdbcStatusCode, ProgrammingError

    class Stmt:
        def __init__(self) -> None:
            self.running = False
            self.stopped = threading.Event()

        def cancel(self) -> None:
            if not self.running:
                msg = "no query"
                raise ProgrammingError(msg, status_code=AdbcStatusCode.INVALID_STATE)
            self.stopped.set()

        def execute_query(self) -> Any:
            sleep(startup_delay)  # the driver is not yet running the query
            self.running = True
            if not self.stopped.wait(5):
                msg = "the cancel was lost"
                raise AssertionError(msg)
            return _FakeHandle(), 0

        def close(self) -> None:
            pass

    stmt = Stmt()
    session = _fake_adbc_session(stmt)
    thread = threading.Thread(
        target=lambda: (sleep(0.05), session.cancel())  # type: ignore[func-returns-value]
    )
    thread.start()
    with pytest.raises(RuntimeError, match="cancelled"):
        session.execute("SELECT 1", None)
    thread.join()
    assert stmt.stopped.is_set()


def test_adbc_cancelled_statement_closes_after_its_handle_is_gone() -> None:
    import weakref

    pytest.importorskip("adbc_driver_manager")
    refs: list[Any] = []
    closed: list[bool] = []

    class Stmt:
        def execute_query(self) -> Any:
            session._cancelled = True
            handle = _FakeHandle()
            refs.append(weakref.ref(handle))
            return handle, 0

        def close(self) -> None:
            closed.append(refs[0]() is None)

    session = _fake_adbc_session(Stmt())
    with pytest.raises(RuntimeError, match="cancelled"):
        session.execute("SELECT 1", None)
    assert closed == [True]


def test_cancel_is_repeated_through_delayed_sqlite_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from polars_database._bridge import _Session

    from polars.io.database._executor import ConnectionExecutor

    entered = threading.Event()
    original = ConnectionExecutor.execute

    def delayed_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        sleep(1.6)  # longer than the former cancellation retry deadline
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConnectionExecutor, "execute", delayed_execute)
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        # Bound a regression's runtime even if every interrupt misses query startup.
        started = monotonic()
        conn.set_progress_handler(lambda: monotonic() - started > 6, 10000)
        session = _Session(conn, owned=True, execute_options={})

        def cancel() -> None:
            assert entered.wait(5)
            session.cancel()

        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            with pytest.raises(Exception, match=r"interrupted|cancelled"):
                list(
                    session.execute(
                        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n "
                        "WHERE x < 100000000) SELECT sum(x) FROM n",
                        None,
                    )
                )
        finally:
            thread.join(timeout=10)
        assert not thread.is_alive()
        assert monotonic() - started < 4
        assert not session._issuing


def test_cancel_at_exit_gives_up_on_a_driver_call_that_ignores_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from polars_database import _bridge

    pytest.importorskip("adbc_driver_manager")
    from adbc_driver_manager import AdbcStatusCode, ProgrammingError

    class Ignoring:
        def cancel(self) -> None:
            msg = "no query"
            raise ProgrammingError(msg, status_code=AdbcStatusCode.INVALID_STATE)

    monkeypatch.setattr(_bridge, "_EXIT_CANCEL_SECONDS", 0.3)
    monkeypatch.setattr(_bridge, "_exit_cancel_deadline", None)
    adbc = _fake_adbc_session(Ignoring())
    adbc._open = [adbc._statement("", None)]
    adbc._issuing = True
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        session = _bridge._Session(conn, owned=True, execute_options={})
        session._issuing = True
        _bridge.begin_exit_cancel()
        started = monotonic()
        for s in (session, adbc):
            s.cancel()
            assert s._cancelled
        # the deadline is shared: the second cancel found it expired
        assert monotonic() - started < 2
        assert _bridge._cancel_expired()
        # after the exit, cancels retry until the driver call returns again
        _bridge.end_exit_cancel()
        assert not _bridge._cancel_expired()
        calls = 0

        class Slow:
            def cancel(self) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    sleep(0.5)  # past the former deadline: still asked again

        adbc = _fake_adbc_session(Slow())
        adbc._open = [adbc._statement("", None)]
        adbc._issuing = True
        threading.Timer(1.0, lambda: setattr(adbc, "_issuing", False)).start()
        adbc.cancel()
        assert calls > 2


def test_unsupported_adbc_cancel_returns_while_issuing() -> None:
    pytest.importorskip("adbc_driver_manager")
    from adbc_driver_manager import NotSupportedError

    class Unsupported:
        def cancel(self) -> None:
            msg = "cancel is unsupported"
            raise NotSupportedError(msg)

    session = _fake_adbc_session(Unsupported())
    session._conn = Unsupported()
    session._open = [session._statement("", None)]
    session._issuing = True
    started = monotonic()
    session.cancel()
    assert monotonic() - started < 1
    assert session._cancelled
