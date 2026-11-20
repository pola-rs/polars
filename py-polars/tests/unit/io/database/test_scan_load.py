"""
Load and safety checks for `scan_database` on Postgres 17 and the `big` table.

Run against a Postgres server seeded with the `big` table:

    POLARS_TEST_DATABASE_LOAD=1 POLARS_TEST_POSTGRES_URI=... pytest test_scan_load.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("polars_database")

POSTGRES = os.environ.get("POLARS_TEST_POSTGRES_URI")
pytestmark = [
    pytest.mark.skipif(
        POSTGRES is None or not os.environ.get("POLARS_TEST_DATABASE_LOAD"),
        reason="needs POLARS_TEST_DATABASE_LOAD=1 and a Postgres URI with `big`",
    ),
    pytest.mark.slow,
    # counts the server's Polars sessions and measures memory: alone, not in parallel
    pytest.mark.skipif(
        os.environ.get("PYTEST_XDIST_WORKER") is not None,
        reason="run serially, without -n, and with no other tests on the server",
    ),
]

DRIVERS = ["adbc", "psycopg"]
RSS_LIMIT = 1536 * 1024 * 1024


def _uri() -> str:
    assert POSTGRES is not None
    return POSTGRES


def _sessions(*, active_only: bool) -> int:
    state = "AND state = 'active'" if active_only else ""
    # client backends only: a parallel query's workers carry its `application_name`
    return int(
        pl.read_database_uri(
            "SELECT COUNT(*) AS n FROM pg_stat_activity "
            "WHERE application_name = 'polars' AND backend_type = 'client backend' "
            f"AND pid != pg_backend_pid() {state}",
            _uri(),
        ).item()
    )


def _scanning() -> int:
    """Polars sessions running (or between fetches of) a read of `big`."""
    return int(
        pl.read_database_uri(
            "SELECT COUNT(*) AS n FROM pg_stat_activity "
            "WHERE application_name = 'polars' AND backend_type = 'client backend' "
            "AND state != 'idle' "
            "AND (query LIKE '%big%' OR query LIKE '%polars_scan%') "
            "AND query NOT LIKE '%1 = 0%'",
            _uri(),
        ).item()
    )


def _wait_gone(timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while (n := _sessions(active_only=False)) and time.monotonic() < deadline:
        time.sleep(0.1)
    return n


def _wait_idle(timeout: float = 2.0) -> int:
    deadline = time.monotonic() + timeout
    while (n := _sessions(active_only=True)) and time.monotonic() < deadline:
        time.sleep(0.1)
    return n


@pytest.fixture(autouse=True)
def _baseline() -> None:
    assert _wait_idle(5) == 0, "a polars query is still running before the test"


@pytest.mark.parametrize("driver", DRIVERS)
def test_cancel_on_head(driver: str) -> None:
    start = time.monotonic()
    out = (
        pl.scan_database("big", _uri(), driver=driver, batch_size=10_000)
        .filter(pl.col("g") >= 0)  # keeps the limit out of the SQL
        .head(10)
        .collect()
    )
    assert out.height == 10
    assert time.monotonic() - start < 20
    assert _wait_idle() == 0


@pytest.mark.parametrize("driver", DRIVERS)
def test_cancel_on_downstream_error(driver: str) -> None:
    baseline = _sessions(active_only=False)

    # fails on the first batch: `s` holds hex digests, not integers
    lf = pl.scan_database("big", _uri(), driver=driver, batch_size=10_000)
    with pytest.raises(pl.exceptions.PolarsError):
        lf.select(pl.col("s").str.to_integer()).collect()
    assert _wait_idle() == 0
    time.sleep(0.5)
    assert _sessions(active_only=False) <= baseline


@pytest.mark.parametrize("driver", DRIVERS)
def test_cancel_on_keyboard_interrupt(driver: str) -> None:
    code = (
        "import polars as pl, sys\n"
        f"pl.scan_database('big', {_uri()!r}, driver={driver!r}, batch_size=10_000)"
        ".select(pl.col('x').sum()).collect()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 20
        # a second look could fall between two fetches: the first one that saw it counts
        while (scanning := _scanning()) == 0 and time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                pytest.fail(f"the scan exited early: {stderr}")
            time.sleep(0.1)
        assert scanning > 0, "the scan never started"
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("the scan did not stop on SIGINT")
        assert proc.returncode != 0
        # the process is gone, and so are its sessions and statements
        assert _wait_gone() == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()


def test_connection_cap() -> None:
    peak = [0]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], _sessions(active_only=False))
            time.sleep(0.05)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        out = (
            pl.scan_database(
                "big",
                _uri(),
                driver="adbc",
                partition_on="id",
                partition_num=12,
                max_connections=3,
            )
            .filter(pl.col("id") <= 3_000_000)
            .select(pl.len())
            .collect()
        )
    finally:
        stop.set()
        sampler.join()
    assert out.item() == 3_000_000
    assert 1 <= peak[0] <= 3, peak[0]


def _sink_rss(driver: str, n_rows: int, out: Path) -> dict[str, Any]:
    code = (
        "import json, re, time, polars as pl\n"
        "t = time.monotonic()\n"
        f"pl.scan_database('big', {_uri()!r}, driver={driver!r})"
        f".filter(pl.col('id') <= {n_rows}).sink_parquet({str(out)!r})\n"
        # VmHWM: this process's peak RSS. ru_maxrss would include the forked parent's.
        "hwm = re.search(r'VmHWM:\\s+(\\d+)', open('/proc/self/status').read())\n"
        "print(json.dumps({'rss': int(hwm.group(1)) * 1024,"
        " 'seconds': time.monotonic() - t}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.parametrize("driver", DRIVERS)
def test_sink_parquet_memory_is_bounded(driver: str, tmp_path: Path) -> None:
    results = {}
    for n in (10_000_000, 50_000_000):
        out = tmp_path / f"{n}.parquet"
        results[n] = _sink_rss(driver, n, out)
        assert pl.scan_parquet(out).select(pl.len()).collect().item() == n
        out.unlink()
    print(f"\n{driver}: {results}")
    small, large = results[10_000_000]["rss"], results[50_000_000]["rss"]
    assert large < RSS_LIMIT, results
    # streaming: 5x the rows must not need much more memory
    assert large < small * 1.5 + 256 * 1024 * 1024, results


@pytest.mark.parametrize("driver", DRIVERS)
def test_session_setup(driver: str) -> None:
    out = pl.scan_database(
        "SELECT current_setting('application_name') AS app, "
        "current_setting('TimeZone') AS tz, "
        "current_setting('transaction_read_only') AS ro",
        _uri(),
        driver=driver,
    ).collect()
    assert out.row(0) == ("polars", "UTC", "on")


@pytest.mark.parametrize("driver", DRIVERS)
def test_write_query_is_rejected(driver: str) -> None:
    before = pl.read_database_uri("SELECT COUNT(*) AS n FROM edge", _uri()).item()
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.scan_database("DELETE FROM edge", _uri(), driver=driver).collect()
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.scan_database(
            "WITH d AS (DELETE FROM edge RETURNING *) SELECT * FROM d",
            _uri(),
            driver=driver,
        ).collect()
    after = pl.read_database_uri("SELECT COUNT(*) AS n FROM edge", _uri()).item()
    assert before == after


def test_throughput_against_read_database_uri() -> None:
    # recorded, not gating: > 2x slower than ConnectorX gets an issue
    query = "SELECT * FROM big WHERE id <= 5000000"
    t = time.monotonic()
    a = pl.read_database_uri(query, _uri(), partition_on="id", partition_num=8)
    cx = time.monotonic() - t
    del a
    t = time.monotonic()
    b = (
        pl.scan_database(
            "big",
            _uri(),
            driver="adbc",
            partition_on="id",
            partition_num=8,
            max_connections=8,
        )
        .filter(pl.col("id") <= 5_000_000)
        .collect()
    )
    scan = time.monotonic() - t
    assert b.height == 5_000_000
    print(
        f"\nconnectorx: {cx:.2f}s, scan_database adbc: {scan:.2f}s ({scan / cx:.2f}x)"
    )


def _run(code: str, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"hung: {code}")


@pytest.mark.parametrize("driver", DRIVERS)
def test_negative_slice_with_more_partitions_than_sessions(driver: str) -> None:
    # readers counted from the end must not keep every session while the next one waits
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_database('big', {_uri()!r}, driver={driver!r},"
        " partition_on='id', partition_num=8, max_connections=2)"
        ".filter(pl.col('id') <= 100_000)\n"
        "print(lf.tail(30_000).select(pl.len()).collect().item(),"
        " lf.slice(-50_010, 20).collect().height)\n"
    )
    assert _run(code, 60).stdout.split() == ["30000", "20"]


@pytest.mark.parametrize("driver", ["psycopg2", "sqlalchemy"])
def test_python_drivers_stream(driver: str) -> None:
    # head(10) of 5M rows: a buffering cursor would fetch everything first. Live
    # connections read whole results: Polars may release a read after `collect()`
    # returned, and a late FETCH would land in the user's next transaction
    source = {
        "psycopg2": f"{_uri()!r}, driver='psycopg2'",
        "sqlalchemy": f"{_uri()!r}.replace('postgresql://', 'postgresql+psycopg://'),"
        " driver='sqlalchemy'",
    }[driver]
    code = (
        "import re, polars as pl\n"
        f"lf = pl.scan_database('big', {source}, partition_on=None)\n"
        "out = lf.filter(pl.col('id') <= 5_000_000, pl.col('g') >= 0).head(10).collect()\n"
        "hwm = re.search(r'VmHWM:\\s+(\\d+)', open('/proc/self/status').read())\n"
        "print(out.height, int(hwm.group(1)) * 1024)\n"
    )
    height, rss = _run(code, 120).stdout.split()
    assert height == "10"
    assert int(rss) < 450 * 1024 * 1024, rss
