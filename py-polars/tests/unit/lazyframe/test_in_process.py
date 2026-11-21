from __future__ import annotations

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
