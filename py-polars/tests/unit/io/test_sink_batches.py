import pytest

import polars as pl
from polars._typing import EngineType
from polars.testing import assert_frame_equal


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sink_batches(engine: EngineType) -> None:
    df = pl.DataFrame({"a": range(100)})
    frames = []

    df.lazy().sink_batches(lambda df: frames.append(df))

    assert_frame_equal(pl.concat(frames), df)


def test_sink_batches_early_stop() -> None:
    df = pl.DataFrame({"a": range(100)})
    stopped = False

    def cb(_: pl.DataFrame) -> bool:
        nonlocal stopped
        assert not stopped
        stopped = True
        return False

    df.lazy().sink_batches(cb)
