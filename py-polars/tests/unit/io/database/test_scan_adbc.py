"""
ADBC reads through Python's `adbc_driver_manager`.

The data paths are in `test_scan.py` and the server suite; these tests cover what the
driver manager adds: closing early, errors, a missing or unusable driver manager, and
the values the ADBC Postgres driver can't convert (when `POLARS_TEST_POSTGRES_URI` is
set).
"""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("polars_database")
pytest.importorskip("sqlglot")
pytest.importorskip("adbc_driver_manager")
pytest.importorskip("adbc_driver_sqlite")


@pytest.fixture
def edge_db(tmp_path: Path) -> str:
    path = tmp_path / "edge.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, i INTEGER, f REAL, s TEXT, b BLOB)")
    rows: list[tuple[Any, ...]] = [
        (1, 2**63 - 1, float("inf"), "", b""),
        (2, -(2**63), -0.0, "émoji 🦆", b"\x00\xff"),
        (3, 0, 1.5, "it's %_\\", None),
        (4, None, None, None, b"x"),
    ]
    rows += [(i, i, i / 7, f"s{i}", None) for i in range(5, 20_005)]
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


def test_adbc_reads_edge_values(edge_db: str) -> None:
    df = pl.scan_database("t", edge_db, driver="adbc").collect()
    expected = pl.scan_database("t", edge_db, driver="sqlite3").collect()
    assert_frame_equal(df.sort("id"), expected.sort("id"))


def test_adbc_rejects_text_that_isnt_utf8(tmp_path: Path) -> None:
    # the driver passed SQLite's bytes on: a Polars String must be UTF-8
    path = tmp_path / "u.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, s TEXT)")
    conn.execute("INSERT INTO t VALUES (1, CAST(X'FF' AS TEXT)), (2, 'a')")
    conn.commit()
    conn.close()
    for source in ["t", "SELECT * FROM t"]:
        lf = pl.scan_database(source, f"sqlite:///{path}", driver="adbc")
        with pytest.raises(pl.exceptions.ComputeError, match="'s' holds text that is"):
            lf.collect()


def test_adbc_stops_early(edge_db: str) -> None:
    # an early stop drops the stream before its end: the statement and connection close
    lf = pl.scan_database("t", edge_db, driver="adbc", batch_size=100).head(3)
    assert lf.collect(engine="streaming").height == 3


def test_adbc_reports_sql_errors(edge_db: str) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="no_such_column"):
        pl.scan_database(
            "SELECT no_such_column FROM t", edge_db, driver="adbc"
        ).collect()


def test_adbc_without_driver_manager(
    edge_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "adbc_driver_manager", None)
    # a pinned ADBC driver is not available: the error names the package, not the driver
    with pytest.raises(pl.exceptions.ComputeError) as e:
        pl.scan_database("t", edge_db, driver="adbc").collect()
    msg = str(e.value)
    assert "is installed, but" in msg
    assert msg.endswith("pip install adbc-driver-manager")
    # unpinned, the next backend reads
    lf = pl.scan_database("t", edge_db)
    assert "adbc:" not in lf.explain()
    assert lf.select(pl.len()).collect().item() == 20_004
    # nor is it reported as available, and the install hint names the package
    adbc = pl.io.database.show_drivers("sqlite").filter(
        pl.col("backend") == "adbc:sqlite"
    )
    assert not adbc["available"].item()
    assert adbc["install"].item() == "pip install adbc-driver-manager"


def test_adbc_redacts_rejected_credentials(edge_db: str) -> None:
    # the SQLite driver quotes a rejected option, value included
    with pytest.raises(pl.exceptions.ComputeError) as e:
        pl.scan_database(
            "t", edge_db, driver="adbc", connect_options={"password": "s3cr3t-pw"}
        ).collect()
    assert "s3cr3t-pw" not in str(e.value)


def test_adbc_rejects_driver_manager_arguments(edge_db: str) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="its own arguments"):
        pl.scan_database(
            "t", edge_db, driver="adbc", connect_options={"entrypoint": "x"}
        ).collect()


@pytest.mark.xdist_group("adbc_postgres")
def test_adbc_rejects_postgres_infinity() -> None:
    # the ADBC Postgres driver wraps ±infinity to dates around 290 000 BC
    uri = os.environ.get("POLARS_TEST_POSTGRES_URI")
    if uri is None:
        pytest.skip("needs POLARS_TEST_POSTGRES_URI")
    pytest.importorskip("adbc_driver_postgresql")
    for value in [
        "'infinity'::timestamp",
        "'-infinity'::timestamptz",
        "'infinity'::date",
        "'294276-12-31 23:59:59'::timestamp",  # past what the driver's shift can hold
        "ARRAY['2020-01-01'::timestamp, 'infinity']",
        "ARRAY['-infinity'::date]",
    ]:
        query = f"SELECT 1 AS id, {value} AS v"
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="'v' holds a Postgres"
        ):
            pl.scan_database(query, uri, driver="adbc").collect()
    # the earliest and latest Postgres values read as they are
    finite = (
        "SELECT 1 AS id, '4714-11-24 00:00:00 BC'::timestamp AS lo, "
        "'4714-11-24 BC'::date AS dlo, '200000-12-31'::date AS dhi"
    )
    df = pl.scan_database(finite, uri, driver="adbc").collect()
    assert df["dlo"].dt.year().item() == -4713
    assert df["dhi"].dt.year().item() == 200000
