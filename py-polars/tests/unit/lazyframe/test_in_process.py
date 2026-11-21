from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from polars.lazyframe.in_process import InProcessQuery

# A source that would take this long to read every batch.
N_BATCHES = 200
BATCH_SLEEP = 0.05


class SlowSource:
    """A Python source that sleeps between batches and records when it is closed."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()
        self.batches = 0

    def scan(self) -> pl.LazyFrame:
        def source(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            try:
                for i in range(N_BATCHES):
                    self.started.set()
                    self.batches += 1
                    yield pl.DataFrame({"a": [i]})
                    time.sleep(BATCH_SLEEP)
            finally:
                self.closed.set()

        return register_io_source(source, schema={"a": pl.Int64})


def _cancel_when_started(src: SlowSource, q: InProcessQuery) -> float:
    assert src.started.wait(30)
    start = time.perf_counter()
    q.cancel()
    elapsed = time.perf_counter() - start
    # Nothing of the query runs any more: the source closed before `cancel` returned.
    assert src.closed.is_set()
    batches = src.batches
    time.sleep(3 * BATCH_SLEEP)
    assert src.batches == batches
    return elapsed


def test_cancel_waits_for_source_cleanup() -> None:
    src = SlowSource()
    q = src.scan().select(pl.col("a") + 1).collect(background=True)
    elapsed = _cancel_when_started(src, q)
    # It stopped between batches, well before the whole source was read.
    assert elapsed < N_BATCHES * BATCH_SLEEP / 4
    assert src.batches < N_BATCHES


def test_cancel_waits_for_streaming_sink(tmp_path: Path) -> None:
    # A sink to a file runs on the streaming engine, which honours the cancellation too.
    src = SlowSource()
    q = (
        src.scan()
        .sink_parquet(tmp_path / "out.parquet", lazy=True)
        .collect(background=True)
    )
    elapsed = _cancel_when_started(src, q)
    assert elapsed < N_BATCHES * BATCH_SLEEP / 4
    assert src.batches < N_BATCHES


def test_cancel_after_completion_and_twice() -> None:
    q = pl.LazyFrame({"a": [1, 2, 3]}).select(pl.col("a") * 2).collect(background=True)
    assert q.fetch_blocking()["a"].to_list() == [2, 4, 6]
    start = time.perf_counter()
    q.cancel()
    q.cancel()
    assert time.perf_counter() - start < 5


def test_cancel_twice_while_running() -> None:
    src = SlowSource()
    q = src.scan().collect(background=True)
    _cancel_when_started(src, q)
    q.cancel()
    with pytest.raises(pl.exceptions.ComputeError, match="query interrupted"):
        q.fetch_blocking()


def test_cancel_from_inside_the_query_does_not_deadlock() -> None:
    handle: list[InProcessQuery] = []
    ready = threading.Event()
    cancelled = threading.Event()

    def udf(s: pl.Series) -> pl.Series:
        assert ready.wait(30)
        # Called from the query's own thread: requests the cancellation, doesn't wait.
        handle[0].cancel()
        cancelled.set()
        return s

    lf: Any = pl.LazyFrame({"a": [1, 2, 3]}).select(
        pl.col("a").map_batches(udf, return_dtype=pl.Int64)
    )
    q = lf.collect(background=True)
    handle.append(q)
    ready.set()
    assert cancelled.wait(30)
    q.cancel()


def test_cancel_from_a_polars_python_thread_pool_does_not_wait() -> None:
    # Polars' Python thread pools (Python readers, scan resolvers) run work the query
    # may wait for: a cancel there only requests the cancellation.
    from polars._utils.threading import PyThreadPool

    started = threading.Event()
    closed = threading.Event()

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        try:
            for i in range(10):
                started.set()
                yield pl.DataFrame({"a": [i]})
                time.sleep(2)
        finally:
            closed.set()

    q = register_io_source(source, schema={"a": pl.Int64}).collect(background=True)
    assert started.wait(30)
    pool = PyThreadPool(1)
    start = time.perf_counter()
    pool.pool.submit(q.cancel).result(timeout=30)
    elapsed = time.perf_counter() - start
    pool.pool.shutdown()
    # It returned without waiting for the batch being read.
    assert elapsed < 1
    q.cancel()
    assert closed.is_set()


def test_cancel_stops_between_callbacks_of_a_callback_sink() -> None:
    calls: list[float] = []
    started = threading.Event()

    def callback(df: pl.DataFrame) -> None:
        calls.append(time.perf_counter())
        started.set()
        time.sleep(0.5)

    q = (
        pl.LazyFrame({"a": range(20)})
        .sink_batches(callback, chunk_size=1, lazy=True)
        .collect(background=True)
    )
    assert started.wait(30)
    q.cancel()
    returned = time.perf_counter()
    n = len(calls)
    time.sleep(1)
    # The callback running when cancel was called ended, and no other one started.
    assert len(calls) == n
    assert all(t < returned for t in calls)
    assert n < 20


def _run_with_slow_source(
    batch_sleep: float, script: str
) -> subprocess.CompletedProcess[str]:
    """Run `script` in a new Python process, with `lf` scanning a slow source."""
    script = SLOW_SOURCE_SCRIPT % {"batch_sleep": batch_sleep} + script
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )


SLOW_SOURCE_SCRIPT = """
    import os, signal, threading, time
    import polars as pl
    from polars.io.plugins import register_io_source

    started = threading.Event()

    def source(with_columns, predicate, n_rows, batch_size):
        for i in range(50):
            started.set()
            yield pl.DataFrame({"a": [i]})
            time.sleep(%(batch_sleep)s)

    lf = register_io_source(source, schema={"a": pl.Int64})
"""


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT")
def test_ctrl_c_while_cancel_waits() -> None:
    # Ctrl-C interrupts the wait (the query still stops), and doesn't abort the process.
    script = """
    q = lf.collect(background=True)
    started.wait(30)
    time.sleep(0.2)

    def interrupt():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    start = time.perf_counter()
    try:
        q.cancel()
        print("cancel returned")
    except KeyboardInterrupt:
        print("cancel raised KeyboardInterrupt", time.perf_counter() - start < 2.5)
    q.cancel()
    try:
        q.fetch_blocking()
    except pl.exceptions.ComputeError as e:
        print("fetch:", e)
    print("done")
"""
    out = _run_with_slow_source(3.0, script)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [
        "cancel raised KeyboardInterrupt True",
        "fetch: query interrupted",
        "done",
    ]


def test_dropping_a_running_query() -> None:
    script = """
    import gc
    q = lf.collect(background=True)
    started.wait(30)
    del q
    gc.collect()
    time.sleep(1)
    print("done")
"""
    out = _run_with_slow_source(0.1, script)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["done"]
