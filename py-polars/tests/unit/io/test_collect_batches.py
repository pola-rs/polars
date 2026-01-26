from __future__ import annotations

import subprocess
import sys
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


@pytest.mark.slow
def test_collect_batches_releases_gil_26031() -> None:
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import polars as pl
from polars.testing import assert_frame_equal

def reentrant_add(x: int):
    return next(
        pl.DataFrame({"": x})
        .lazy()
        .select(pl.first().map_elements(lambda x: x + 1, return_dtype=pl.UInt32))
        .collect_batches(engine="streaming")
    ).item()

assert_frame_equal(
    pl.concat(
        pl.LazyFrame({"a": range(10)})
        .with_columns(
            out=pl.col("a").map_elements(reentrant_add, return_dtype=pl.UInt32)
        )
        .collect_batches(engine="streaming")
    ),
    pl.DataFrame(
        [
            pl.Series("a", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pl.Int64),
            pl.Series("out", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=pl.UInt32),
        ]
    ),
)

print("OK", end="")
""",
        ],
        timeout=5,
    )

    assert out == b"OK"
