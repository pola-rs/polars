from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sink_batches(engine: EngineType) -> None:
    df = pl.DataFrame({"a": range(100)})
    frames: list[pl.DataFrame] = []

    df.lazy().sink_batches(lambda df: frames.append(df), engine=engine)  # type: ignore[call-overload]

    assert_frame_equal(pl.concat(frames), df)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sink_batches_early_stop(engine: EngineType) -> None:
    df = pl.DataFrame({"a": range(1000)})
    stopped = False

    def cb(_: pl.DataFrame) -> bool | None:
        nonlocal stopped
        assert not stopped
        stopped = True
        return True

    df.lazy().sink_batches(cb, chunk_size=100, engine=engine)  # type: ignore[call-overload]
    assert stopped


def test_collect_batches() -> None:
    df = pl.DataFrame({"a": range(100)})
    frames = []

    for f in df.lazy().collect_batches():
        frames += [f]

    assert_frame_equal(pl.concat(frames), df)


def test_chunk_size() -> None:
    df = pl.DataFrame({"a": range(113)})

    for f in df.lazy().collect_batches(chunk_size=17):
        expected = df.head(17)
        df = df.slice(17)

        assert_frame_equal(f, expected)

    df = pl.DataFrame({"a": range(10)})

    for f in df.lazy().collect_batches(chunk_size=10):
        assert not f.is_empty()

        expected = df.head(10)
        df = df.slice(10)

        assert_frame_equal(f, expected)
