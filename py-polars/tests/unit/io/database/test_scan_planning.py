"""
`scan_database` planning checks that need no database server.

SQL and classifier snapshots per dialect are planned offline: with `schema=` given and
one partition, planning does no I/O, so SQL Server, Oracle and the warehouses are
covered here without servers. Regenerate the snapshot after a reviewed change with
`POLARS_UPDATE_SNAPSHOTS=1`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from tests.unit.io.database._scan_corpus import EDGE_SCHEMA, planned_sql

pytest.importorskip("polars_database")
pytest.importorskip("sqlglot")
pytest.importorskip("sqlalchemy")

SNAPSHOT = Path(__file__).parent / "snapshots" / "scan_sql.json"

DIALECT_URIS = {
    "postgresql": "postgresql://u@h/db",
    "mysql": "mysql://u@h/db",
    "mssql": "mssql://u@h/db",
    "oracle": "oracle://u@h/db",
    "sqlite": "sqlite:///x.db",
    "duckdb": "duckdb:///x.duckdb",
    "snowflake": "snowflake://u@acct/db",
    "bigquery": "bigquery://project/dataset",
    "databricks": "databricks://h/db",
    "clickhouse": "clickhouse://u@h/db",
    "trino": "trino://u@h/catalog",
    "redshift": "redshift://u@h/db",
}

CASES: dict[str, Any] = {
    "all": None,
    "head": "head",
    "select": ["id", "s"],
    "int_cmp": pl.col("i32") > 42,
    "int_between": pl.col("i64").is_between(0, 10),
    "float_cmp": pl.col("f64") > 1.5,
    "decimal_cmp": pl.col("dec") >= Decimal("12.5"),
    "str_eq": pl.col("s") == "abc",
    "str_ne": pl.col("s") != "abc",
    "str_lt": pl.col("s") < "b",
    "str_quote": pl.col("s") == "it's",
    "str_backslash": pl.col("s") == "a\\b",
    "str_unicode": pl.col("s") == "é",
    "str_empty": pl.col("s") == "",
    "like_prefix": pl.col("s").str.starts_with("a%_"),
    "like_contains": pl.col("s").str.contains("[x]", literal=True),
    "is_in": pl.col("s").is_in(["a", "b", None], nulls_equal=True),
    "is_in_long": pl.col("i64").is_in(list(range(5000))),
    "is_null": pl.col("d").is_null(),
    "eq_missing_null": pl.col("i16").eq_missing(None),
    "ne_missing": pl.col("i16").ne_missing(3),
    "not_exact": ~(pl.col("i16") > 0),
    "not_superset": ~(pl.col("s") == "abc"),
    "or": (pl.col("i16") == 1) | (pl.col("b")),
    "date_cmp": pl.col("d") >= date(2026, 1, 1),
    "ts_cmp": pl.col("ts") < datetime(2026, 1, 1, 12, 30, 0, 123456),
    "bool_col": pl.col("b"),
    "arithmetic": (pl.col("i16") * 2) > 4,
}


def _plan(dialect: str, case: str) -> dict[str, str]:
    lf = pl.scan_database(
        "s.t",
        DIALECT_URIS[dialect],
        schema=EDGE_SCHEMA,
        driver="sqlalchemy",
        partition_on=None,
    )
    op = CASES[case]
    if isinstance(op, str):
        lf = lf.head(10)
    elif isinstance(op, list):
        lf = lf.select(op)
    elif op is not None:
        lf = lf.filter(op)
    lines = {
        k.strip(): v.strip()
        for k, _, v in (
            line.partition(": ")
            for line in lf.explain().splitlines()
            if line.strip().startswith(("pushed:", "sql[0]:"))
        )
    }
    return {"pushed": lines["pushed"], "sql": lines["sql[0]"]}


def _all_plans() -> dict[str, dict[str, dict[str, str]]]:
    return {d: {c: _plan(d, c) for c in CASES} for d in DIALECT_URIS}


def test_sql_snapshots() -> None:
    """Quoting, literals, LIMIT forms, IN chunking and exactness, per dialect."""
    plans = _all_plans()
    if os.environ.get("POLARS_UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(plans, indent=2, ensure_ascii=False) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    diffs = [
        f"{d}/{c}: {plans[d][c]} != {expected.get(d, {}).get(c)}"
        for d in plans
        for c in plans[d]
        if plans[d][c] != expected.get(d, {}).get(c)
    ]
    assert not diffs, "\n".join(diffs)


def test_sql_snapshot_details() -> None:
    plans = _all_plans()
    # LIMIT forms
    assert plans["postgresql"]["head"]["sql"].endswith("LIMIT 10")
    assert "SELECT TOP 10" in plans["mssql"]["head"]["sql"]
    assert plans["oracle"]["head"]["sql"].endswith("FETCH FIRST 10 ROWS ONLY")
    # literals
    assert "N'é'" in plans["mssql"]["str_unicode"]["sql"]
    assert "'it''s'" in plans["postgresql"]["str_quote"]["sql"]
    assert "'a\\\\b'" in plans["mysql"]["str_backslash"]["sql"]
    assert "DATE '2026-01-01'" in plans["oracle"]["date_cmp"]["sql"]
    # LIKE under an unknown collation: a superset only where it matches per character
    # (MySQL) or compares `COLLATE "C"` (Postgres)
    assert "filters 0/1" in plans["mssql"]["like_contains"]["pushed"]
    assert "filters 1/1 (0 exact)" in plans["mysql"]["like_contains"]["pushed"]
    # SQL Server lists are kept short: a long IN becomes a BETWEEN superset
    assert "BETWEEN 0 AND 4999" in plans["mssql"]["is_in_long"]["sql"]
    assert "(0 exact)" in plans["mssql"]["is_in_long"]["pushed"]
    assert " IN (" in plans["postgresql"]["is_in_long"]["sql"]
    # classification: NOT of a superset is never pushed, NOT of an exact one is
    assert "filters 0/1" in plans["mysql"]["not_superset"]["pushed"]
    assert "filters 1/1 (1 exact)" in plans["mysql"]["not_exact"]["pushed"]
    assert "filters 1/1 (0 exact)" in plans["mysql"]["str_eq"]["pushed"]
    # with `schema=` collations are unknown: string equality is a superset everywhere
    assert "filters 1/1 (0 exact)" in plans["postgresql"]["str_eq"]["pushed"]
    # ... and so are types the dtype doesn't show: a MySQL integer may be a YEAR (which
    # reads `42` as 2042), a SQLite string column may hold numbers (affinity)
    assert "filters 0/1" in plans["mysql"]["int_cmp"]["pushed"]
    assert "filters 1/1" in plans["mysql"]["not_exact"]["pushed"]
    assert "filters 0/1" in plans["sqlite"]["str_eq"]["pushed"]
    # floats and decimals: with `schema=` their columns may be of another numeric type
    # (`real`, `numeric`), which the database compares as the values it holds
    assert "filters 0/1" in plans["bigquery"]["float_cmp"]["pushed"]
    assert "filters 0/1" in plans["postgresql"]["float_cmp"]["pushed"]
    assert "filters 0/1" in plans["postgresql"]["decimal_cmp"]["pushed"]
    # Oracle: '' is NULL, so an empty-string literal isn't pushed
    assert "filters 0/1" in plans["oracle"]["str_empty"]["pushed"]
    # arithmetic is not tier 1
    assert "filters 0/1" in plans["postgresql"]["arithmetic"]["pushed"]
    # an empty projection selects a constant
    assert "projection 2/11" in plans["postgresql"]["select"]["pushed"]


@pytest.mark.parametrize(
    ("dialect", "query", "reason"),
    [
        ("mssql", "WITH a AS (SELECT 1 AS x) SELECT x FROM a", "CTE"),
        ("mssql", "SELECT x FROM t ORDER BY x", "ORDER BY"),
        ("postgresql", "SELECT x, x FROM t", "duplicate"),
        # Trino drops the ORDER BY of a derived table without LIMIT, and the others
        # aren't known to keep it
        ("trino", "SELECT x FROM t ORDER BY x DESC", "ORDER BY"),
        ("snowflake", "SELECT x FROM t ORDER BY x", "ORDER BY"),
        ("bigquery", "SELECT x FROM t ORDER BY x", "ORDER BY"),
        ("databricks", "SELECT x FROM t ORDER BY x", "ORDER BY"),
        ("redshift", "SELECT x FROM t ORDER BY x", "ORDER BY"),
    ],
)
def test_query_wrapping_falls_back(dialect: str, query: str, reason: str) -> None:
    lf = pl.scan_database(
        query, DIALECT_URIS[dialect], schema={"x": pl.Int64}, driver="sqlalchemy"
    )
    plan = lf.filter(pl.col("x") > 1).explain()
    assert "pushdown: off" in plan
    assert reason in plan
    assert f"sql[0]: {query}" in plan


@pytest.mark.parametrize(
    ("dialect", "query"),
    [
        # kept inside a derived table (checked on the servers)
        ("postgresql", "SELECT x FROM t ORDER BY x"),
        ("oracle", "SELECT x FROM t ORDER BY x"),
        ("clickhouse", "SELECT x FROM t ORDER BY x"),
        # with a LIMIT, every database keeps it
        ("trino", "SELECT x FROM t ORDER BY x LIMIT 5"),
    ],
)
def test_query_order_by_is_wrapped_where_kept(dialect: str, query: str) -> None:
    lf = pl.scan_database(
        query, DIALECT_URIS[dialect], schema={"x": pl.Int64}, driver="sqlalchemy"
    )
    plan = lf.filter(pl.col("x") > 1).explain()
    assert "pushdown: off" not in plan
    assert "filters 1/1" in plan


def test_query_wrapping() -> None:
    lf = pl.scan_database(
        "SELECT x FROM t ORDER BY x OFFSET 0 ROWS",
        DIALECT_URIS["mssql"],
        schema={"x": pl.Int64},
        driver="sqlalchemy",
    ).filter(pl.col("x") > 1)
    assert "FROM (SELECT x FROM t ORDER BY x OFFSET 0 ROWS) AS [_pl]" in lf.explain()


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM t",
        "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
        "SELECT * INTO copy FROM t",
        "SELECT 1; DROP TABLE t",
        "/* sneaky */ UPDATE t SET x = 1",
    ],
)
@pytest.mark.parametrize("pushdown", [True, False])
def test_write_queries_are_rejected(query: str, pushdown: bool) -> None:
    lf = pl.scan_database(
        query,
        DIALECT_URIS["postgresql"],
        schema={"x": pl.Int64},
        driver="sqlalchemy",
        pushdown=pushdown,
        source_type="query",
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="read"):
        lf.collect_schema()


def test_cse_of_identical_scans(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, "g") for i in range(10)])
    lf = pl.scan_database("t", f"sqlite:///{db}")
    joined = lf.join(lf, on="id")
    assert "CACHE" in joined.explain()
    assert joined.collect().height == 10


# secrets never appear in plans, errors or verbose logs

SECRET = "s3cr3t-Pa55word"


@pytest.mark.parametrize(
    ("connection", "driver"),
    [
        # a host that can't resolve fails at once; a refused connect is retried with
        # backoff (2.6s), which the MySQL cell covers: its retry log mustn't leak either
        (f"postgresql://user:{SECRET}@nohost.invalid:1/db", "adbc"),
        # ConnectorX's pool waits 30s on a refused connection; a bad option fails now
        (f"postgresql://user:{SECRET}@127.0.0.1:1/db?sslmode=bogus", "connectorx"),
        (f"postgresql://user:{SECRET}@nohost.invalid:1/db", "psycopg"),
        (f"postgresql://user:{SECRET}@nohost.invalid:1/db", "sqlalchemy"),
        (f"mysql://user:{SECRET}@127.0.0.1:1/db", "pymysql"),
        (f"Driver={{Nope}};Server=127.0.0.1,1;UID=u;PWD={SECRET}", "pyodbc"),
    ],
)
def test_secrets_absent(
    connection: str, driver: str, capfd: pytest.CaptureFixture[str]
) -> None:
    module = {"adbc": "adbc_driver_postgresql", "sqlalchemy": "sqlalchemy"}.get(
        driver, driver
    )
    pytest.importorskip(module)
    lf = pl.scan_database("t", connection, driver=driver)
    with pl.Config(verbose=True):
        assert SECRET not in repr(lf)
        with pytest.raises(Exception) as exc:
            lf.collect()
    out, err = capfd.readouterr()
    assert SECRET not in str(exc.value)
    assert SECRET not in out + err


def test_secret_from_credential_provider_absent(
    capfd: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("psycopg")
    lf = pl.scan_database(
        "t",
        "postgresql://user@nohost.invalid:1/db",
        driver="psycopg",
        credential_provider=lambda: {"password": SECRET},
    )
    with pl.Config(verbose=True), pytest.raises(Exception) as exc:
        lf.collect()
    out, err = capfd.readouterr()
    assert SECRET not in str(exc.value) + out + err


def test_plan_with_password_hides_it() -> None:
    lf = pl.scan_database(
        "t",
        f"postgresql://user:{SECRET}@h/db",
        schema={"x": pl.Int64},
        driver="sqlalchemy",
    )
    assert SECRET not in lf.explain()
    assert SECRET not in str(lf)


# driver resolution


@pytest.mark.parametrize(
    ("uri", "path"),
    [
        # as SQLAlchemy reads them: three slashes are relative, four absolute
        ("sqlite:////tmp/a.db", "/tmp/a.db"),
        ("sqlite:///rel.db", "rel.db"),
        ("sqlite:rel.db", "rel.db"),
        ("sqlite:///a%20b.db?mode=ro", "a b.db"),
        ("duckdb:///C:\\Users\\a\\t.duckdb", "C:\\Users\\a\\t.duckdb"),
        ("sqlite:///d:/a/b.db", "d:/a/b.db"),
        ("sqlite://", ":memory:"),
        ("duckdb:///:memory:", ":memory:"),
    ],
)
def test_file_path_of_uri(uri: str, path: str) -> None:
    # on Windows, `/C:/...` would be a network path
    from polars_database._bridge import _file_path

    assert _file_path(uri) == path


def _hide_modules(monkeypatch: pytest.MonkeyPatch, *hidden: str) -> None:
    from polars_database import _bridge

    real = _bridge._has_module
    monkeypatch.setattr(
        _bridge,
        "_has_module",
        lambda m: False if m.split(".")[0] in hidden or m in hidden else real(m),
    )


def test_resolution_picks_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("psycopg")
    _hide_modules(monkeypatch, "adbc_driver_postgresql", "connectorx")
    lf = pl.scan_database(
        "t", "postgresql://u@h/db", schema={"x": pl.Int64}, partition_on=None
    )
    plan = lf.explain()
    assert "via python:psycopg" in plan

    df = pl.io.database.show_drivers("postgresql")
    available = dict(zip(df["backend"], df["available"], strict=True))
    assert available["connectorx"] is False
    assert available["python:psycopg"] is True


def test_pin_is_final(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_modules(monkeypatch, "connectorx")
    lf = pl.scan_database(
        "t", "postgresql://u@h/db", schema={"x": pl.Int64}, driver="connectorx"
    )
    with pytest.raises(pl.exceptions.ComputeError) as exc:
        lf.collect_schema()
    msg = str(exc.value)
    assert "pinned database driver 'connectorx' is not installed" in msg
    assert "pip install connectorx" in msg


def test_no_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from polars_database import _bridge

    _hide_modules(monkeypatch, "connectorx", "trino", "sqlalchemy", "adbc_driver_trino")
    # the hint is the same whether or not adbc_driver_manager is installed (it isn't on
    # Windows CI)
    monkeypatch.setattr(_bridge, "has_adbc_driver_manager", lambda: True)
    lf = pl.scan_database("t", "trino://u@h/c", schema={"x": pl.Int64})
    with pytest.raises(pl.exceptions.ComputeError) as exc:
        lf.collect_schema()
    msg = re.sub(
        r"Checked ADBC manifests in .*",
        "Checked ADBC manifests in <dirs>",
        str(exc.value),
    )
    msg = msg.split("\n\nThis error occurred")[0]
    assert msg == (
        "no driver available for database dialect 'trino' (trino://u@h/c)\n"
        "  Install one of (best first):\n"
        "    dbc install trino\n"
        "    pip install connectorx\n"
        "    pip install trino\n"
        "    pip install sqlalchemy  (plus the dialect's DB-API driver)\n"
        "  Or pin one with driver=..., or pass a connection factory.\n"
        "  Checked ADBC manifests in <dirs>"
    )


def test_probing_imports_no_heavy_modules() -> None:
    code = (
        "import sys, polars as pl\n"
        "before = set(sys.modules)\n"
        "pl.io.database.show_drivers()\n"
        "new = set(sys.modules) - before\n"
        "heavy = {'sqlalchemy', 'pyarrow', 'pandas', 'psycopg', 'pymysql', 'oracledb',"
        " 'connectorx', 'duckdb', 'sqlglot', 'clickhouse_connect'}\n"
        "print(sorted({m.split('.')[0] for m in new} & heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout


# 1.6, 1.7, 1.9: partitioning, concurrency and the reader


@pytest.fixture
def numbers_db(tmp_path: Path) -> str:
    db = tmp_path / "n.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE n (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany(
            "INSERT INTO n VALUES (?, ?)", [(i, str(i)) for i in range(1000)]
        )
    return f"sqlite:///{db}"


def test_bounds_use_exact_filters(
    numbers_db: str, capfd: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("n", numbers_db, partition_on="id", partition_num=4)
    with pl.Config(verbose=True):
        out = lf.filter(pl.col("id") > 700).collect()
    _, err = capfd.readouterr()
    assert out.height == 299
    bounds = next(line for line in err.splitlines() if "partition bounds" in line)
    assert 'MAX("id")' in bounds
    assert '"id" > 700' in bounds


@pytest.mark.parametrize("cap", [1, 3])
def test_max_connections_caps_concurrency(
    numbers_db: str, cap: int, capfd: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "n", numbers_db, partition_on="id", partition_num=12, max_connections=cap
    )
    with pl.Config(verbose=True):
        assert lf.collect().height == 1000
    _, err = capfd.readouterr()
    assert f"max_concurrent_scans: {cap}," in err


@pytest.mark.parametrize(("env", "expected"), [("5", 2), ("1", 1)])
def test_env_override_lowers_but_never_raises_connection_cap(
    numbers_db: str,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    env: str,
    expected: int,
) -> None:
    # more readers than sessions could deadlock an ordered read
    pytest.importorskip("adbc_driver_sqlite")
    monkeypatch.setenv("POLARS_MAX_CONCURRENT_SCANS", env)
    lf = pl.scan_database(
        "n", numbers_db, partition_on="id", partition_num=12, max_connections=2
    )
    with pl.Config(verbose=True):
        assert lf.collect().height == 1000
    _, err = capfd.readouterr()
    assert f"max_concurrent_scans: {expected}," in err


def test_env_override_above_connection_cap_in_order(
    numbers_db: str, tmp_path: Path
) -> None:
    # An override above the session count, and morsels small enough that a reader's
    # output fills before it is done: a later partition must never hold the session the
    # first one waits for. In a subprocess, so a deadlock fails instead of hanging.
    pytest.importorskip("adbc_driver_sqlite")
    out = tmp_path / "out.parquet"
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_database('n', {numbers_db!r}, partition_on='id', "
        "partition_num=12, max_connections=2)\n"
        f"lf.sink_parquet({str(out)!r})\n"
    )
    env = {
        **os.environ,
        "POLARS_MAX_CONCURRENT_SCANS": "5",
        "POLARS_IDEAL_MORSEL_SIZE": "4",
    }
    try:
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("the ordered read with more readers than sessions hung")
    assert pl.read_parquet(out)["id"].to_list() == list(range(1000))


def test_live_connection_reads_one_partition_at_a_time(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(100)")
    live = duckdb.connect(path, read_only=True)
    try:
        lf = pl.scan_database(
            "t", live, partition_on="id", partition_num=4, max_connections=4
        )
        with pl.Config(verbose=True):
            assert lf.collect().height == 100
        _, err = capfd.readouterr()
        assert "max_concurrent_scans: 1," in err
    finally:
        live.close()


def test_connect_is_retried(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(10)")
    fail = {"next": 0}

    def factory() -> Any:
        if fail["next"] > 0:
            fail["next"] -= 1
            msg = "transient"
            raise ConnectionError(msg)
        return duckdb.connect(path, read_only=True)

    lf = pl.scan_database("t", factory, partition_on=None)
    lf.collect_schema()
    fail["next"] = 3  # fails three times, then works: within the retry budget
    assert lf.collect().height == 10
    fail["next"] = 4  # one more than the budget
    with pytest.raises(pl.exceptions.ComputeError, match="transient"):
        lf.collect()


def test_failed_login_is_not_retried(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(10)")
    calls = {"n": 0, "fail": False}

    def factory() -> Any:
        calls["n"] += 1
        if calls["fail"]:
            msg = 'password authentication failed for user "polars"'
            raise PermissionError(msg)
        return duckdb.connect(path, read_only=True)

    lf = pl.scan_database("t", factory, partition_on=None)
    lf.collect_schema()
    calls.update(n=0, fail=True)
    with pytest.raises(pl.exceptions.ComputeError, match="password"):
        lf.collect()
    assert calls["n"] == 1

    # a query error fails the same way again: it isn't re-run either
    calls.update(n=0, fail=False)
    bad = pl.scan_database(
        "SELECT CAST('x' || id AS INTEGER) AS x FROM t", factory, partition_on=None
    )
    with pytest.raises(pl.exceptions.PolarsError):
        bad.collect()
    assert calls["n"] <= 3  # dialect detection, planning, then one read


def test_cast_error_names_everything(numbers_db: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("n", numbers_db, schema={"id": pl.Int64, "s": pl.Date})
    with pytest.raises(pl.exceptions.SchemaError) as exc:
        lf.collect()
    msg = str(exc.value)
    assert "'s'" in msg
    assert "date" in msg
    assert "partition 0" in msg
    assert "adbc:sqlite" in msg


@pytest.mark.parametrize("n", [1, 2, 7, 64])
@pytest.mark.parametrize("table", ["with_nulls", "empty", "single"])
def test_every_row_read_once(tmp_path: Path, n: int, table: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    db = tmp_path / "p.db"
    all_rows: dict[str, list[tuple[int, int | None]]] = {
        "with_nulls": [(i, None if i % 9 == 0 else i * 3 - 50) for i in range(300)],
        "empty": [],
        "single": [(i, 7) for i in range(20)],
    }
    rows = all_rows[table]
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER)")
        conn.executemany("INSERT INTO p VALUES (?, ?)", rows)
    uri = f"sqlite:///{db}"
    cases: list[dict[str, Any]] = [
        {"partition_on": "k", "partition_num": n},
        # bounds that don't cover the data: rows outside land in the end partitions
        {"partition_on": "k", "partition_num": n, "partition_range": (100, 200)},
    ]
    for kwargs in cases:
        out = pl.scan_database(
            "p", uri, schema={"id": pl.Int64, "k": pl.Int64}, **kwargs
        )
        ids = out.select("id").collect()["id"].to_list()
        assert sorted(ids) == [r[0] for r in rows], kwargs


# params and consistency


def test_params(numbers_db: str) -> None:
    lf = pl.scan_database(
        "SELECT id, s FROM n WHERE id < ? AND s != ?",
        numbers_db,
        params=[10, "3"],
        driver="sqlite3",
    )
    out = lf.filter(pl.col("id") > 5).collect()
    assert sorted(out["id"].to_list()) == [6, 7, 8, 9]
    # every partition binds the same parameters
    lf = pl.scan_database(
        "SELECT id FROM n WHERE id < ?",
        numbers_db,
        params=[100],
        driver="sqlite3",
        partition_on="id",
        partition_num=4,
    )
    assert lf.select(pl.len()).collect().item() == 100


def test_params_errors(numbers_db: str) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="query source"):
        pl.scan_database("n", numbers_db, params=[1], driver="sqlite3").collect()
    pytest.importorskip("adbc_driver_sqlite")
    with pytest.raises(pl.exceptions.InvalidOperationError, match="binds parameters"):
        pl.scan_database(
            "SELECT id FROM n WHERE id < ?", numbers_db, params=[1], driver="adbc"
        ).collect()


def test_params_change_the_scan(numbers_db: str) -> None:
    q = "SELECT id FROM n WHERE id < ?"
    a = pl.scan_database(q, numbers_db, params=[5], driver="sqlite3")
    b = pl.scan_database(q, numbers_db, params=[7], driver="sqlite3")
    out = pl.concat([a, b]).select(pl.len()).collect()
    assert out.item() == 12


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sqlite3 takes check_same_thread as an int before Python 3.12",
)
def test_typed_driver_options_change_the_scan(numbers_db: str) -> None:
    # the string forms of `False` and `"False"` (truthy) are the same: the scans differ
    # by the values the driver gets
    def scan(value: Any) -> pl.LazyFrame:
        return pl.scan_database(
            "n",
            numbers_db,
            driver="sqlite3",
            connect_options={"check_same_thread": value},
        )

    assert "CACHE" in scan(False).join(scan(False), on="id").explain()
    assert "CACHE" not in scan(False).join(scan("False"), on="id").explain()


def test_snapshot_consistency_with_partitions(numbers_db: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "n", numbers_db, partition_on="id", partition_num=2, consistency="snapshot"
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="snapshot"):
        lf.collect()
    one = pl.scan_database("n", numbers_db, partition_on=None, consistency="snapshot")
    assert one.collect().height == 1000


def test_all_null_first_batch(tmp_path: Path) -> None:
    # sqlite3 types each batch on its own: the first one is all NULL
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, None if i <= 500 else i) for i in range(1, 1001)],
        )
        conn.commit()
    lf = pl.scan_database(
        "SELECT id, v FROM t ORDER BY id",
        f"sqlite:///{path}",
        driver="sqlite3",
        batch_size=100,
    )
    assert lf.collect_schema()["v"] == pl.Int64
    assert lf.collect()["v"].null_count() == 500


# connection URIs of the Python backends


def test_credential_provider_keeps_the_uri_password() -> None:
    from polars_database._bridge import _with_credentials

    uri = "postgresql://u:pw@h/db"
    assert _with_credentials(uri, ("v", None)) == "postgresql://v:pw@h/db"
    assert _with_credentials(uri, (None, "p@ss")) == "postgresql://u:p%40ss@h/db"
    # an unencoded `#` in the password: encoded, so that drivers read the host
    assert (
        _with_credentials("postgresql://u:pa#ss@h/db?x=1", None)
        == "postgresql://u:pa%23ss@h/db?x=1"
    )


def test_credential_provider_tuples() -> None:
    from polars_database._bridge import credentials

    with pytest.raises(TypeError, match=r"not a \(user, password\) tuple"):
        credentials(lambda: ("u", "p"))
    # `(credentials, expiry)`
    assert credentials(lambda: ({"user": "u"}, None)) == ("u", None)


@pytest.mark.parametrize("password", ["pa#ss", "pa?ss", "pa/ss", "p@ss"])
def test_url_parts_with_unencoded_password(password: str) -> None:
    from polars_database._bridge import _url_parts

    assert _url_parts(f"postgresql+pg8000://u:{password}@h:5/db?x=1") == {
        "host": "h",
        "port": 5,
        "user": "u",
        "password": password,
        "database": "db",
    }
    assert _url_parts("postgresql://h:5/db?x=a@b")["host"] == "h"


@pytest.mark.parametrize(
    ("kind", "uri", "added"),
    [
        ("python:psycopg", "postgresql://h/db", True),
        ("python:psycopg2", "redshift://h/db", True),
        ("connectorx", "postgresql://h/db", True),
        ("sqlalchemy", "postgresql://h/db", True),
        ("sqlalchemy", "postgresql+psycopg://h/db", True),
        # pg8000 and redshift_connector reject `connect_timeout`
        ("sqlalchemy", "postgresql+pg8000://h/db", False),
        ("sqlalchemy", "redshift+redshift_connector://h/db", False),
        ("python:pg8000", "postgresql://h/db", False),
        ("python:redshift_connector", "redshift://h/db", False),
    ],
)
def test_connect_timeout_only_for_libpq(kind: str, uri: str, added: bool) -> None:
    from polars_database._bridge import _with_connect_timeout

    dialect = uri.split(":")[0].split("+")[0]
    out = _with_connect_timeout(uri, dialect, kind)
    assert ("connect_timeout=" in out) == added


def test_redshift_uri_for_libpq() -> None:
    from polars_database._bridge import _libpq_uri

    assert _libpq_uri("redshift+psycopg2://u@h/db?x=1") == "postgresql://u@h/db?x=1"
    assert _libpq_uri("postgres://u@h/db") == "postgres://u@h/db"


def test_uri_query_parameters_to_driver_arguments() -> None:
    import ssl

    from polars_database._bridge import _query_kwargs

    kw = _query_kwargs(
        "pymysql",
        "mysql://h/db?charset=latin1&unix_socket=/s&ssl_ca=/ca&connect_timeout=3",
    )
    assert kw == {
        "charset": "latin1",
        "unix_socket": "/s",
        "ssl_ca": "/ca",
        "connect_timeout": 3.0,
    }
    kw = _query_kwargs("pymysql", "mysql://h/db?ssl_mode=VERIFY_IDENTITY")
    assert kw == {"ssl_verify_cert": True, "ssl_verify_identity": True}
    kw = _query_kwargs("MySQLdb", "mysql://h/db?ssl_ca=/ca&ssl_mode=required")
    assert kw == {"ssl_mode": "REQUIRED", "ssl": {"ca": "/ca"}}

    kw = _query_kwargs(
        "pg8000", "postgresql://h/db?sslmode=verify-full&host=/run", 5433
    )
    ctx = kw.pop("ssl_context")
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert kw == {"unix_sock": "/run/.s.PGSQL.5433"}
    # `prefer` (libpq's default): pg8000 can't fall back, so no TLS, as without it
    assert _query_kwargs("pg8000", "postgresql://h/db?sslmode=prefer") == {}

    assert _query_kwargs("trino", "trino://h/c?SSL=true&verify=/ca.pem") == {
        "http_scheme": "https",
        "verify": "/ca.pem",
    }
    assert _query_kwargs("pymssql", "mssql://h/db?encrypt=yes&charset=UTF-8") == {
        "encryption": "require",
        "charset": "UTF-8",
    }


@pytest.mark.parametrize(
    ("module", "uri", "match"),
    [
        ("pymysql", "mysql://h/db?charset=x&bogus=1", r"pymysql .* \['bogus'\]"),
        ("pg8000", "postgresql://h/db?gssencmode=require", "gssencmode"),
        ("pg8000", "postgresql://h/db?target_session_attrs=read-write", "honour"),
        ("pg8000", "postgresql://h/db?client_encoding=latin1", "honour"),
        (
            "pg8000",
            "postgresql://h/db?sslmode=verify-full&sslrootcert=/nonexistent",
            "/nonexistent",
        ),
        ("redshift_connector", "redshift://h/db?sslmode=require", "verify-ca"),
        ("clickhouse_connect", "clickhouse://h/db?secure=maybe", "boolean"),
    ],
)
def test_uri_query_parameters_not_mapped_raise(
    module: str, uri: str, match: str
) -> None:
    from polars_database._bridge import _query_kwargs

    with pytest.raises(ValueError, match=match):
        _query_kwargs(module, uri)


def test_driver_errors_are_scrubbed_of_secrets() -> None:
    from polars_database._bridge import _scrubbed, _secrets

    secrets = _secrets("postgresql://u:pa%23Sekr3t@h/db", ("u", "Pr0v1ded"), {})
    for secret in ["pa#Sekr3t", "pa%23Sekr3t", "Pr0v1ded"]:
        msg = f"invalid dsn: 'postgresql://u:{secret}@h/db'"
        with pytest.raises(OSError, match="invalid dsn") as exc, _scrubbed(secrets):
            raise OSError(msg)
        assert "Sekr3t" not in str(exc.value)
        assert "Pr0v1ded" not in str(exc.value)

    odbc = _secrets("Driver={x};Server=s;UID=u;PWD={a;Sekr3t}", None, {})
    msg = "login failed: PWD={a;Sekr3t}, a;Sekr3t"
    with pytest.raises(OSError) as exc, _scrubbed(odbc):
        raise OSError(msg)
    assert "Sekr3t" not in str(exc.value)

    class Echo(Exception):
        def __str__(self) -> str:
            return "echoes pa#Sekr3t"

    # a message not built from `args`: replaced
    with pytest.raises(RuntimeError, match=r"Echo: echoes \*\*\*"), _scrubbed(secrets):
        raise Echo


def test_uri_query_parameters_libpq_and_mysql_ssl() -> None:
    from polars_database import _bridge
    from polars_database._bridge import _query_kwargs

    # libpq parameters pg8000 honours, or that change nothing for it
    uri = (
        "postgresql://h/db?connect_timeout=5&target_session_attrs=any"
        "&client_encoding=UTF8&gssencmode=prefer&keepalives_idle=30"
        "&fallback_application_name=app"
    )
    assert _query_kwargs("pg8000", uri) == {
        _bridge._PG8000_CONNECT_TIMEOUT: 5.0,
        "application_name": "app",
    }
    # `ssl=true` requires TLS; mysql.connector only prefers it, so it's checked
    assert _query_kwargs("pymysql", "mysql://h/db?ssl=false") == {"ssl_disabled": True}
    assert "ssl" in _query_kwargs("pymysql", "mysql://h/db?ssl=true")
    for q in ["ssl=true", "ssl_mode=REQUIRED", "ssl_mode=VERIFY_CA"]:
        kw = _query_kwargs("mysql.connector", f"mysql://h/db?{q}")
        assert kw[_bridge._REQUIRE_TLS] is True
        assert kw.get("ssl_disabled") is False or kw.get("ssl_verify_cert") is True


def test_mysql_tls_check_rejects_a_plain_connection() -> None:
    from polars_database._bridge import _check_mysql_tls

    class Conn:
        closed = False

        def __init__(self, cipher: str) -> None:
            self.cipher = cipher

        def cursor(self) -> Any:
            conn = self

            class Cursor:
                def execute(self, sql: str) -> None:
                    pass

                def fetchone(self) -> tuple[str, str]:
                    return ("Ssl_cipher", conn.cipher)

                def close(self) -> None:
                    pass

            return Cursor()

        def close(self) -> None:
            self.closed = True

    _check_mysql_tls(Conn("TLS_AES_256_GCM_SHA384"))
    plain = Conn("")
    with pytest.raises(ValueError, match="requires TLS"):
        _check_mysql_tls(plain)
    assert plain.closed


@pytest.mark.parametrize(
    ("dialect", "query", "repeatable"),
    [
        ("sqlite", "SELECT unicode(randomblob(1)) AS b FROM t", False),
        ("postgres", "SELECT get_byte(uuid_send(gen_random_uuid()), 0) FROM t", False),
        ("postgres", "SELECT clock_timestamp(), statement_timestamp()", False),
        ("postgres", "SELECT nextval('s')", False),
        ("tsql", "SELECT NEWID() AS u FROM t", False),
        ("oracle", "SELECT DBMS_RANDOM.VALUE FROM dual", False),
        ("oracle", "SELECT s.nextval FROM dual", False),
        ("mysql", "SELECT UUID(), SYSDATE()", False),
        ("clickhouse", "SELECT rand64()", False),
        ("snowflake", "SELECT seq4()", False),
        # the transaction's time: a count and its read run in two transactions
        ("postgres", "SELECT * FROM t WHERE ts > now() - interval '1 hour'", False),
        ("postgres", "SELECT id, CURRENT_TIMESTAMP, CURRENT_DATE FROM t", False),
        ("postgres", "SELECT 'now'::timestamp", False),
        ("mysql", "SELECT NOW()", False),
        ("mysql", "SELECT * FROM t WHERE d > CURDATE()", False),
        ("sqlite", "SELECT datetime('now')", False),
        ("tsql", "SELECT GETDATE()", False),
        ("oracle", "SELECT SYSTIMESTAMP FROM dual", False),
        ("clickhouse", "SELECT today()", False),
        # a column named nextval; text 'now' that isn't a time
        ("postgres", "SELECT t.nextval FROM t", True),
        ("oracle", "SELECT x.nextval FROM t x", True),
        ("sqlite", "SELECT concat(s, 'now'), instr(s, 'now') FROM t", True),
        (
            "postgres",
            "SELECT replace(s, 'today', ''), CAST('now' AS text) FROM t",
            True,
        ),
        # compared with a column, 'now' may be cast to its type: taken as the time
        ("postgres", "SELECT id FROM t WHERE created < 'now'", False),
        (
            "postgres",
            "SELECT id FROM t WHERE d BETWEEN 'yesterday' AND 'tomorrow'",
            False,
        ),
        ("postgres", "SELECT 'today'::date", False),
        # SQLite's time functions without a time value
        ("sqlite", "SELECT unixepoch()", False),
        ("sqlite", "SELECT datetime(), date()", False),
        ("sqlite", "SELECT julianday('now')", False),
        ("sqlite", "SELECT unixepoch(ts) FROM t", True),
        # 'now' typed by what it meets is a time; one-argument age() is relative to
        # today
        ("postgres", "SELECT extract(epoch FROM ts - 'now') FROM t", False),
        (
            "postgres",
            "SELECT date_trunc('day', 'now'), greatest(ts, 'now') FROM t",
            False,
        ),
        ("postgres", "SELECT coalesce(ts, 'now') FROM t", False),
        ("postgres", "SELECT age(ts) FROM t", False),
        ("postgres", "SELECT age(a, b) FROM t", True),
        (
            "postgres",
            "SELECT 'now' AS label, s || 'now' FROM t WHERE s LIKE 'now%'",
            True,
        ),
        # rows numbered or picked in no set order
        ("postgres", "SELECT row_number() OVER () AS rn, id FROM t", False),
        ("postgres", "SELECT lag(x) OVER (ORDER BY ts) FROM t", False),
        (
            "postgres",
            "SELECT sum(x) OVER (ORDER BY y ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
            False,
        ),
        ("oracle", "SELECT id FROM t WHERE ROWNUM <= 5", False),
        # windows that treat ties alike; a function of its argument
        (
            "postgres",
            "SELECT sum(x) OVER (PARTITION BY g), count(*) OVER () FROM t",
            True,
        ),
        ("postgres", "SELECT rank() OVER (ORDER BY a) FROM t", True),
        ("mysql", "SELECT UNIX_TIMESTAMP(d) FROM t", True),
        ("mysql", "SELECT UNIX_TIMESTAMP() FROM t", False),
        ("postgres", "SELECT DISTINCT ON (a) a, b FROM t", False),
        ("postgres", "SELECT DISTINCT a FROM t", True),
        ("duckdb", "SELECT any_value(a), first(b) FROM t GROUP BY c", False),
        ("clickhouse", "SELECT any(a), anyLast(b) FROM t", False),
        ("clickhouse", "SELECT rowNumberInAllBlocks(), id FROM t", False),
        # the size of the block read; the query's or the session's identity
        ("clickhouse", "SELECT id, blockSize() AS bs FROM t", False),
        ("clickhouse", "SELECT id, queryID() AS q FROM t", False),
        ("clickhouse", "SELECT id, nowInBlock() AS n FROM t", False),
        ("postgres", "SELECT id, pg_backend_pid() AS p FROM t", False),
        ("mysql", "SELECT id, CONNECTION_ID() AS c FROM t", False),
    ],
)
def test_volatile_functions_make_a_query_not_repeatable(
    dialect: str, query: str, repeatable: bool
) -> None:
    from polars_database._sqlglot import inspect_query

    assert inspect_query(query, dialect)["repeatable"] is repeatable


@pytest.mark.parametrize(
    ("query", "columns", "all_columns"),
    [
        ("SELECT id, x FROM t WHERE random() < 0.5", [], False),
        ("SELECT random() AS r, id FROM t", ["r"], False),
        ("SELECT id, (SELECT abs(random())) AS r FROM t", ["r"], False),
        ("SELECT * FROM (SELECT random() AS r FROM t) AS s", [], True),
        ("SELECT random() AS r FROM t UNION ALL SELECT 1", [], True),
        ("SELECT random(), * FROM t", [], True),
    ],
)
def test_volatile_output_columns(
    query: str, columns: list[str], all_columns: bool
) -> None:
    from polars_database._sqlglot import inspect_query

    facts = inspect_query(query, "sqlite")
    assert facts["volatile_columns"] == columns
    assert facts["all_columns_volatile"] is all_columns


def test_duckdb_view_lambdas_are_inspected() -> None:
    # DuckDB writes a view's lambdas (and list comprehensions) parenthesised, which
    # sqlglot didn't parse: the view couldn't be inspected, so no filter was pushed on
    # it and partitions were refused
    from polars_database._sqlglot import inspect_query, inspect_view

    view = (
        "CREATE VIEW v AS SELECT id, list_apply(list_filter(list_apply(l, (lambda x: "
        'struct_pack("filter" := (x > 0), result := (x + 1)))), (lambda elem: '
        "struct_extract(elem, 'filter'))), (lambda elem: struct_extract(elem, 'result'))) "
        "AS g, list_reduce(l, (lambda a, b: (a + b))) AS s FROM base;"
    )
    facts = inspect_view(view, "duckdb")
    assert facts["parses"]
    assert facts["repeatable"]
    assert facts["output_columns"] == ["id", "g", "s"]
    assert facts["tables"] == [["base"]]
    # a query's text is kept as written; a call's parentheses and text are left alone
    query = "SELECT f((lambda x: x)) AS a, '(lambda' AS b FROM t"
    assert inspect_query(query, "duckdb")["body"] == query


@pytest.mark.parametrize(
    ("dialect", "query"),
    [
        ("mysql", "SELECT * FROM v WHERE id > %s"),
        ("mysql", "SELECT * FROM v WHERE id > %(low)s AND s LIKE '%s'"),
        ("oracle", "SELECT * FROM v WHERE id > :1"),
    ],
)
def test_queries_with_driver_placeholders_are_inspected(
    dialect: str, query: str
) -> None:
    # sqlglot doesn't parse MySQL's `%s` or Oracle's `:1`: a query with `params=` wasn't
    # inspected, so the volatile views it reads went unchecked (partitions accepted)
    from polars_database._sqlglot import inspect_query

    facts = inspect_query(query, dialect)
    assert facts["parses"]
    assert [t[-1].lower() for t in facts["tables"]] == ["v"]
    assert facts["body"] == query
    # a modulo is no placeholder
    assert inspect_query("SELECT a % s FROM t", "mysql")["parses"]


def test_unknown_functions_and_tables() -> None:
    from polars_database._sqlglot import inspect_query

    facts = inspect_query(
        "WITH c AS (SELECT * FROM s.t) SELECT myrand(), abs(x), Other(1) "
        "FROM c JOIN v ON c.id = v.id JOIN c AS d ON 1 = 1",
        "sqlite",
    )
    assert facts["unknown_functions"] == ["myrand", "other"]
    # where a user's function may replace a built-in, the known ones are listed too
    assert "abs" in facts["known_functions"]
    # (also those as written: `instr` is `STRPOS` to sqlglot)
    assert (
        "instr"
        in inspect_query("SELECT instr(a, b) FROM t", "sqlite")["known_functions"]
    )
    assert inspect_query("SELECT abs(x) FROM t", "postgres")["known_functions"] == []
    assert sorted(facts["tables"]) == [["s", "t"], ["v"]]
    # a CTE hides a table of its name only where it is in scope; unquoted names fold
    # as the dialect folds them; a table function may be a (parameterized) view
    facts = inspect_query(
        "SELECT * FROM V WHERE id IN (WITH v AS (SELECT 1 AS id) SELECT id FROM v) "
        'UNION ALL SELECT * FROM "Q"',
        "postgres",
    )
    assert sorted(facts["tables"]) == [["Q"], ["v"]]
    assert inspect_query("SELECT * FROM v", "oracle")["tables"] == [["V"]]
    facts = inspect_query("SELECT * FROM pv(p = 0.5)", "clickhouse")
    assert facts["tables"] == [["pv"]]
    # and a table function, which the catalog may know to be volatile (a table macro)
    assert facts["unknown_functions"] == ["pv"]


@pytest.mark.parametrize(
    ("dialect", "cte", "reference", "tables"),
    [
        ("postgres", '"V"', "v", [["v"]]),
        ("postgres", "v", '"V"', [["V"]]),
        ("postgres", "V", "v", []),
        ("postgres", '"v"', "V", []),
        ("postgres", '"V"', '"V"', []),
        ("oracle", '"v"', "v", [["V"]]),
        ("oracle", "v", '"v"', [["v"]]),
        ("oracle", "v", "V", []),
        ("oracle", '"V"', "v", []),
        ("oracle", '"v"', '"v"', []),
        ("sqlite", '"V"', "v", []),
        ("sqlite", "v", '"V"', []),
        ("duckdb", '"V"', "v", []),
        ("duckdb", "v", '"V"', []),
        ("clickhouse", '"V"', "v", [["v"]]),
        ("clickhouse", "V", "v", [["v"]]),
        ("clickhouse", '"V"', '"V"', []),
        ("mysql", "V", "v", [["v"]]),
        ("mysql", "`V`", "v", [["v"]]),
        ("mysql", "v", "`V`", [["V"]]),
        ("mysql", "`V`", "`V`", []),
        # SQL Server's collation may be case-sensitive: names stay as written, and a CTE
        # hides only the exact name
        ("tsql", "V", "v", [["v"]]),
        ("tsql", "[V]", "v", [["v"]]),
        ("tsql", "v", "[V]", [["V"]]),
        ("tsql", "V", "V", []),
        ("tsql", "[V]", "[V]", []),
    ],
)
def test_cte_identifier_case(
    dialect: str, cte: str, reference: str, tables: list[list[str]]
) -> None:
    from polars_database._sqlglot import inspect_query

    facts = inspect_query(
        f"WITH {cte} AS (SELECT 1) SELECT * FROM {reference}", dialect
    )
    assert facts["parses"]
    assert facts["tables"] == tables


def test_tsql_names_keep_their_case() -> None:
    from polars_database._sqlglot import inspect_query

    assert inspect_query("SELECT * FROM Orders", "tsql")["tables"] == [["Orders"]]
    facts = inspect_query("SELECT * FROM [Dbo].[Orders] o JOIN v ON 1 = 1", "tsql")
    assert sorted(facts["tables"]) == [["Dbo", "Orders"], ["v"]]
    # a nested reference too: the view's query is inspected as written
    facts = inspect_query(
        "SELECT * FROM [Dbo].[V] WHERE id IN (SELECT id FROM W)", "tsql"
    )
    assert sorted(facts["tables"]) == [["Dbo", "V"], ["W"]]


@pytest.mark.parametrize("recursive", [False, True])
def test_cte_identifier_case_preserves_scope(recursive: bool) -> None:
    from polars_database._sqlglot import inspect_query

    keyword = "RECURSIVE " if recursive else ""
    facts = inspect_query(
        f'WITH {keyword}"V" AS (SELECT * FROM "V") SELECT * FROM "V"', "postgres"
    )
    assert facts["tables"] == ([] if recursive else [["V"]])
    # A prior CTE is visible in the next definition; a later one is not.
    facts = inspect_query(
        'WITH "V" AS (SELECT * FROM v), v AS (SELECT * FROM "V") SELECT * FROM v',
        "postgres",
    )
    assert facts["tables"] == [["v"]]


@pytest.mark.parametrize(
    ("view", "dialect", "outputs", "columns", "all_columns", "repeatable"),
    [
        (
            "CREATE VIEW v AS SELECT id, random() AS r FROM t",
            "sqlite",
            ["id", "r"],
            ["r"],
            False,
            False,
        ),
        (
            "CREATE VIEW v (a, b) AS SELECT id, random() FROM t",
            "sqlite",
            ["a", "b"],
            ["b"],
            False,
            False,
        ),
        (
            "CREATE VIEW v (a, b) AS SELECT * FROM (SELECT id, random() AS r FROM t)",
            "sqlite",
            ["a", "b"],
            [],
            True,
            False,
        ),
        (
            "CREATE VIEW v AS SELECT id FROM t WHERE random() > 0",
            "sqlite",
            ["id"],
            [],
            False,
            False,
        ),
        (
            "CREATE VIEW v AS SELECT id FROM t LIMIT 5",
            "sqlite",
            ["id"],
            [],
            False,
            False,
        ),
        ("SELECT id, abs(x) AS a FROM t", "postgres", ["id", "a"], [], False, True),
        (
            "select `db`.`t`.`id` AS `id`,rand() AS `r` from `db`.`t`",
            "mysql",
            ["id", "r"],
            ["r"],
            False,
            False,
        ),
        ("", "mysql", None, [], False, False),
        ("CREATE TABLE t (id INT)", "sqlite", None, [], False, False),
    ],
)
def test_inspect_view(
    view: str,
    dialect: str,
    outputs: list[str] | None,
    columns: list[str],
    all_columns: bool,
    repeatable: bool,
) -> None:
    from polars_database._sqlglot import inspect_view

    facts = inspect_view(view, dialect)
    assert facts["parses"] is (outputs is not None or all_columns)
    assert facts["output_columns"] == outputs
    assert facts["volatile_columns"] == columns
    assert facts["all_columns_volatile"] is all_columns
    assert facts["repeatable"] is repeatable


@pytest.mark.parametrize(
    ("uri", "userinfo", "host"),
    [
        # an `@` in a parameter isn't the end of the userinfo, port or not
        (
            "postgresql://u:pw@127.0.0.1:5432/db?application_name=a@b",
            "u:pw",
            "127.0.0.1:5432",
        ),
        ("postgresql://h:5432/db?application_name=a@b", None, "h:5432"),
        ("clickhouse://h:8123/db?user=a@b", None, "h:8123"),
        # a password that starts like a port
        ("postgresql://u:12#Sekr3t@h:5432/db", "u:12#Sekr3t", "h:5432"),
        # `host:port/` ends the authority: hidden (see below), not connected to
        ("postgresql://u:12/Sekr3t@h:5432/db", None, "u:12"),
        ("postgresql://localhost:5432/my@db", None, "localhost:5432"),
        ("postgresql://u:pa/ss@h:5432/db", "u:pa/ss", "h:5432"),
        ("postgresql://u:12?Sekr3t@h:5432/db", "u:12?Sekr3t", "h:5432"),
        ("postgresql://u:pw@[::1]:5432/db", "u:pw", "[::1]:5432"),
        ("sqlite:///tmp/a:b@c.db", None, ""),
    ],
)
def test_uri_userinfo(uri: str, userinfo: str | None, host: str) -> None:
    from polars_database._bridge import _split_authority

    split = _split_authority(uri)
    assert split is not None
    assert (split[1], split[2]) == (userinfo, host)


def test_short_secrets_are_scrubbed_only_as_values() -> None:
    from polars_database._bridge import _raise_unmapped, _scrub, _secrets

    secrets = _secrets("postgresql://u:a@h/db?sslkey=k&password=b", None, {})
    assert "k" not in secrets
    message = "OperationalError: bad dsn postgresql://u:a@h/db?password=b"
    assert _scrub(message, secrets) == (
        "OperationalError: bad dsn postgresql://u:***@h/db?password=***"
    )
    assert _secrets("x", None, {"private_key_passphrase": "pp"}) == ["pp"]
    assert _secrets("postgresql://u:12/Sekr3t@h:5432/db", None, {}) == [
        "12/Sekr3t",
        "Sekr3t",
        "12",
    ]
    # base64-style: `/` then `=`
    uri = "mysql://polars:kq7/Zx9+Ab==@127.0.0.1:3306/polars"
    assert _secrets(uri, None, {}) == ["kq7/Zx9+Ab==", "Zx9+Ab==", "kq7"]
    assert _scrub('invalid port ":kq7", value "kq7"', _secrets(uri, None, {})) == (
        'invalid port ":***", value "***"'
    )
    for key in ["key", "encryption_key", "azure_storage_connection_string"]:
        assert _secrets("x", None, {key: "s3cret"}) == ["s3cret"], key
    assert "PrivateKeyPassphrase" not in str(
        _secrets("Driver={x};PrivateKeyPassphrase=Sekr3t", None, {})
    )
    assert _secrets("Driver={x};PrivateKeyPassphrase=Sekr3t", None, {}) == ["Sekr3t"]
    # a parameter name that isn't one may be a password the URI parse missed
    with pytest.raises(ValueError, match=r"parameter\(s\) \(2\)"):
        _raise_unmapped("pg8000", ["Sekr3t@h:5432/db", "sslmode"])


def test_order_by_puts_nulls_first_where_the_database_is_unknown() -> None:
    # range partitions put NULLs first: an unknown database is told to as well, in
    # SQL that every database runs
    lf = pl.scan_database(
        "t",
        "sqlite:///x.db",
        dialect="generic",
        schema={"id": pl.Int64},
        order_by="id",
        driver="sqlalchemy",
        partition_on=None,
    )
    sql = planned_sql(lf)
    assert 'ORDER BY CASE WHEN "id" IS NULL THEN 0 ELSE 1 END ASC, "id" ASC' in sql
    assert "NULLS" not in sql


def test_mysql_integer_of_unknown_type_keeps_literals_1_to_99() -> None:
    # with `schema=` an integer may be a YEAR, which reads 1-99 as 2001-2069, 1970-1999
    lf = pl.scan_database(
        "s.t",
        DIALECT_URIS["mysql"],
        schema=EDGE_SCHEMA,
        driver="sqlalchemy",
        partition_on=None,
    )
    for predicate, pushed in [
        (pl.col("i32") > 42, "filters 0/1"),
        (pl.col("i32").is_in([5, 1970]), "filters 0/1"),
        (pl.col("i32") > 500, "filters 1/1 (1 exact)"),
        (pl.col("i32") == 0, "filters 1/1 (1 exact)"),
        (pl.col("i32").is_between(100, 1950), "filters 1/1 (1 exact)"),
    ]:
        assert pushed in lf.filter(predicate).explain(), str(predicate)


def test_url_parts_errors_dont_echo_the_uri() -> None:
    from polars_database._bridge import _url_parts

    with pytest.raises(ValueError, match="invalid port") as exc:
        _url_parts("mysql://polars:kq7/Zx9+Ab==@127.0.0.1:3306/polars")
    assert "kq7" not in str(exc.value)
    with pytest.raises(ValueError, match="several"):
        _url_parts("postgresql://u@h1:5432,h2:5433/db")


@pytest.mark.parametrize("driver", ["sqlite3", "adbc"])
def test_sqlite_string_filters_with_nul(tmp_path: Path, driver: str) -> None:
    # SQLite's LIKE stops at a NUL: `'a\0b' LIKE '%b'` is false, which Polars can't
    # filter back in
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    db = tmp_path / "nul.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)", [(1, "a\0b"), (2, "ab"), (3, "c")]
        )
    lf = pl.scan_database("t", f"sqlite:///{db}", driver=driver)
    df = lf.collect()
    for predicate in (
        pl.col("s").str.ends_with("b"),
        pl.col("s").str.starts_with("a"),
        pl.col("s").str.contains("b", literal=True),
    ):
        got = lf.filter(predicate).collect().sort("id")
        assert got["id"].to_list() == df.filter(predicate).sort("id")["id"].to_list()


def test_duckdb_time_with_time_zone_filters(tmp_path: Path) -> None:
    # TIMETZ reads as its local time; DuckDB compares it with a TIME literal in UTC (and
    # can't order it against one): such filters stay in Polars
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "tz.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, t TIMETZ)")
        conn.execute(
            "INSERT INTO t VALUES (1, TIMETZ '12:00:00+02'), (2, TIMETZ '09:00:00+00')"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    df = lf.collect()
    for predicate in (
        pl.col("t") == time(12),
        pl.col("t") < time(11),
        pl.col("t") >= time(10),
    ):
        got = lf.filter(predicate).collect().sort("id")
        assert got["id"].to_list() == df.filter(predicate).sort("id")["id"].to_list()
        assert "filters 0/1" in lf.filter(predicate).explain()


def test_clickhouse_negated_in_is_not_pushed() -> None:
    # `NULL IN (…)` is 0 under ClickHouse's `transform_null_in`: NOT would keep NULLs
    for dialect, pushed in [
        ("clickhouse", "filters 0/1"),
        ("postgresql", "filters 1/1"),
    ]:
        lf = pl.scan_database(
            "s.t",
            DIALECT_URIS[dialect],
            schema=EDGE_SCHEMA,
            driver="sqlalchemy",
            partition_on=None,
        )
        for p in [
            ~pl.col("i64").is_in([1, 2]),
            ~(pl.col("i64").is_in([1, 2]) & (pl.col("i16") == 1)),
        ]:
            assert pushed in lf.filter(p).explain(), (dialect, p)
        # positively, 0 drops the NULL rows as NULL does
        assert "filters 1/1 (1 exact)" in lf.filter(pl.col("i64").is_in([1])).explain()


@pytest.mark.parametrize(
    ("dialect", "pushed"),
    [
        # per character (MySQL), or compared `COLLATE "C"` (Postgres)
        ("mysql", "filters 1/1 (0 exact)"),
        ("postgresql", "filters 1/1 (0 exact)"),
        # a collation may match by its collation elements (Czech `ch`)
        ("oracle", "filters 0/1"),
        ("mssql", "filters 0/1"),
        ("snowflake", "filters 0/1"),
    ],
)
def test_like_under_an_unknown_collation(dialect: str, pushed: str) -> None:
    lf = pl.scan_database(
        "s.t",
        DIALECT_URIS[dialect],
        schema=EDGE_SCHEMA,
        driver="sqlalchemy",
        partition_on=None,
    )
    assert pushed in lf.filter(pl.col("s").str.ends_with("c")).explain()
    if dialect == "mysql":
        # an index range for a prefix under PAD SPACE skips `'c\tx'` (#211)
        assert "filters 0/1" in lf.filter(pl.col("s").str.starts_with("c")).explain()


@pytest.mark.parametrize("dtype", [pl.Float64, pl.Float32, pl.Decimal(20, 4)])
def test_user_schema_floats_and_decimals_are_not_compared(dtype: Any) -> None:
    # `schema=` doesn't say the column's type: a Float64 over `real` or `numeric` is
    # compared by the database as the value it holds, not the one Polars reads
    lf = pl.scan_database(
        "t",
        DIALECT_URIS["postgresql"],
        schema={"id": pl.Int64, "x": dtype},
        driver="sqlalchemy",
        partition_on=None,
    )
    plan = lf.filter(pl.col("x") > 1).explain()
    assert "filters 0/1" in plan
    assert "filters 1/1" in lf.filter(pl.col("id") > 1).explain()


def test_clickhouse_identifiers_escape_backslashes() -> None:
    # sqlglot escapes `\` in ClickHouse's quoted identifiers only from 30.22: a column
    # `x\` was written `"x\"`, which ClickHouse reads as an escaped quote
    from polars_database import _sqlglot

    column = {"Column": "x\\"}
    query = {
        "columns": [column],
        "from": {"Table": ["t\\"]},
        "filter": [{"Binary": [column, "Eq", {"Literal": {"Int": 1}}]}],
        "order_by": [{"expr": column, "descending": False}],
        "limit": None,
        "qualified": [],
    }
    sql = _sqlglot.render(json.dumps(query), "clickhouse")
    assert sql.count('"x\\\\"') == 3, sql
    assert '"t\\\\"' in sql, sql
    assert '"x\\"' not in sql.replace('"x\\\\"', ""), sql


def test_sqlglot_older_than_the_minimum_is_refused() -> None:
    # pyproject.toml asks for sqlglot >= 28.6; an older one (installed without its
    # dependencies' resolution) read Oracle's SYSTIMESTAMP as repeatable
    code = (
        "import sqlglot\n"
        "sqlglot.__version__ = '28.5.0'\n"
        "import polars_database._sqlglot\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode != 0
    assert "ModuleUpgradeRequiredError" in out.stderr, out.stderr
    assert "28.6" in out.stderr, out.stderr


@pytest.mark.parametrize(
    ("value", "sql"),
    [
        ("1.50", "NUMERIC '1.50'"),
        (
            "-12345678901234567890123456789.123456789",
            "NUMERIC '-12345678901234567890123456789.123456789'",
        ),
        ("0.1234567890", "BIGNUMERIC '0.1234567890'"),
        (
            "123456789012345678901234567890",
            "BIGNUMERIC '123456789012345678901234567890'",
        ),
    ],
)
def test_bigquery_decimal_literals_are_typed(value: str, sql: str) -> None:
    # a bare `1.50` is a FLOAT64 on BigQuery
    from polars_database import _sqlglot

    assert _sqlglot._literal({"Decimal": value}, "bigquery").sql("bigquery") == sql
    assert _sqlglot._literal({"Decimal": value}, "postgres").sql("postgres") == value


def test_scrub_query_after_a_password_with_a_question_mark() -> None:
    from polars_database._bridge import _scrub, _secrets

    uri = "postgresql://u:ab?c=d@db.example.com/app?sslpassword=s3cr3tpw"
    secrets = _secrets(uri, None, {})
    assert "s3cr3tpw" in secrets
    assert "s3cr3tpw" not in _scrub(f"failed: {uri}", secrets)
    assert "s3cr3tpw" in _secrets(
        "postgresql://u@h/db?ssl%50assword=s3cr3tpw", None, {}
    )


def test_scrub_message_with_lone_surrogate() -> None:
    from polars_database._bridge import _scrub

    message = _scrub("no such file \udcff pw=hunter2hunter2", ["hunter2hunter2"])
    assert message.startswith("no such file ")
    assert message.endswith(" pw=***")
