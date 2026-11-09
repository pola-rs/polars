from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime
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


@pytest.mark.slow
def test_collect_batches_arrow_c_stream_python_dataset_28628() -> None:
    # Resolving a `PythonDataset` scan (`scan_delta` / `scan_iceberg`) calls back into
    # Python from another thread, so `__arrow_c_stream__` must not hold the GIL while
    # it resolves the schema. Run in a subprocess: on regression this deadlocks.
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import polars as pl
import pyarrow as pa
from polars._plr import PyLazyFrame
from polars._utils.wrap import wrap_ldf

SCHEMA = pa.schema([pa.field("id", pa.int64())])
BATCH = pa.record_batch([pa.array(range(1000), type=pa.int64())], schema=SCHEMA)


class DatasetObject:
    def schema(self):
        return SCHEMA

    def to_dataset_scan(self, **kwargs):
        def impl(*args, **kw):
            return iter([pl.from_arrow(BATCH)]), False

        lf = pl.LazyFrame._scan_python_function(SCHEMA, impl, pyarrow=True, is_pure=True)
        return lf, "v1"


lf = wrap_ldf(PyLazyFrame.new_from_dataset_object(DatasetObject()))
reader = pa.RecordBatchReader.from_stream(lf.collect_batches(engine="streaming"))
assert reader.read_all().num_rows == 1000

print("OK", end="")
""",
        ],
        timeout=30,
    )

    assert out == b"OK"


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_collect_batches_split_by(engine: EngineType) -> None:
    df = pl.DataFrame({"ts": [1, 1, 2, 2, 2, 3], "x": range(6)})

    frames = list(df.lazy().collect_batches(split_by="ts", engine=engine))

    assert [f["ts"].to_list() for f in frames] == [[1, 1], [2, 2, 2], [3]]
    assert all(f["ts"].n_unique() == 1 for f in frames)
    assert_frame_equal(pl.concat(frames), df)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_collect_batches_split_by_across_morsels(engine: EngineType) -> None:
    n = 100_000
    df = pl.DataFrame({"ts": [i // 1000 for i in range(n)], "x": range(n)})

    frames = list(df.lazy().collect_batches(split_by="ts", engine=engine))

    assert len(frames) == 100
    assert all(f["ts"].n_unique() == 1 for f in frames)
    assert_frame_equal(pl.concat(frames), df)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sink_batches_split_by(engine: EngineType) -> None:
    df = pl.DataFrame({"ts": [1, 1, 2, 2, 2, 3], "x": range(6)})
    frames: list[pl.DataFrame] = []

    df.lazy().sink_batches(lambda f: frames.append(f), split_by="ts", engine=engine)

    assert [f["ts"].to_list() for f in frames] == [[1, 1], [2, 2, 2], [3]]
    assert_frame_equal(pl.concat(frames), df)


def test_split_by_validation() -> None:
    lf = pl.DataFrame({"ts": [1, 1, 2], "x": range(3)}).lazy()

    with pytest.raises(pl.exceptions.InvalidOperationError, match="maintain_order"):
        lf.collect_batches(split_by="ts", maintain_order=False)

    with pytest.raises(pl.exceptions.InvalidOperationError, match="combined"):
        lf.collect_batches(split_by="ts", chunk_size=2)

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        list(lf.collect_batches(split_by="missing"))

    with pytest.raises(pl.exceptions.InvalidOperationError, match="maintain_order"):
        lf.sink_batches(lambda _: None, split_by="ts", maintain_order=False)

    with pytest.raises(pl.exceptions.InvalidOperationError, match="combined"):
        lf.sink_batches(lambda _: None, split_by="ts", chunk_size=2)

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        lf.sink_batches(lambda _: None, split_by="missing")


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_collect_batches_split_by_single_key_across_morsels(engine: EngineType) -> None:
    n = 200_000
    df = pl.DataFrame({"ts": [1] * n, "x": range(n)})

    frames = list(df.lazy().collect_batches(split_by="ts", engine=engine))

    assert len(frames) == 1
    assert frames[0].height == n
    assert_frame_equal(frames[0], df)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_split_by_empty(engine: EngineType) -> None:
    df = pl.DataFrame(schema={"ts": pl.Int64, "x": pl.Int64})

    assert list(df.lazy().collect_batches(split_by="ts", engine=engine)) == []

    frames: list[pl.DataFrame] = []
    df.lazy().sink_batches(lambda f: frames.append(f), split_by="ts", engine=engine)
    assert frames == []


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (pl.Int64, [1, 1, 2, 2, 2, 3], [[1, 1], [2, 2, 2], [3]]),
        (pl.UInt32, [1, 1, 2, 2, 2, 3], [[1, 1], [2, 2, 2], [3]]),
        (
            pl.String,
            ["a", "a", "b", "b", "b", "c"],
            [["a", "a"], ["b", "b", "b"], ["c"]],
        ),
        (
            pl.Boolean,
            [True, True, False, False, False, True],
            [[True, True], [False, False, False], [True]],
        ),
        (
            pl.Categorical,
            ["a", "a", "b", "b", "b", "c"],
            [["a", "a"], ["b", "b", "b"], ["c"]],
        ),
        (
            pl.Date,
            [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 2),
                date(2020, 1, 2),
                date(2020, 1, 3),
            ],
            [
                [date(2020, 1, 1), date(2020, 1, 1)],
                [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 2)],
                [date(2020, 1, 3)],
            ],
        ),
        (
            pl.Datetime,
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 2),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
            ],
            [
                [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                [
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 2),
                ],
                [datetime(2020, 1, 3)],
            ],
        ),
    ],
)
def test_split_by_dtypes(
    dtype: pl.DataType, values: list[object], expected: list[list[object]]
) -> None:
    df = pl.DataFrame({"key": pl.Series(values, dtype=dtype), "x": range(6)})

    frames = list(df.lazy().collect_batches(split_by="key"))

    assert [f["key"].to_list() for f in frames] == expected
    assert_frame_equal(pl.concat(frames), df)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sink_batches_split_by_early_stop(engine: EngineType) -> None:
    df = pl.DataFrame({"ts": [1, 1, 2, 2, 2, 3], "x": range(6)})
    seen: list[int] = []

    def callback(frame: pl.DataFrame) -> bool:
        seen.append(frame["ts"][0])
        return True

    df.lazy().sink_batches(callback, split_by="ts", engine=engine)  # type: ignore[call-overload]

    assert seen == [1]


def test_sink_batches_split_by_lazy() -> None:
    df = pl.DataFrame({"ts": [1, 1, 2, 2, 2, 3], "x": range(6)})
    frames: list[pl.DataFrame] = []

    out = df.lazy().sink_batches(  # type: ignore[call-overload]
        lambda f: frames.append(f), split_by="ts", lazy=True
    )

    assert isinstance(out, pl.LazyFrame)
    out.collect()
    assert [f["ts"].to_list() for f in frames] == [[1, 1], [2, 2, 2], [3]]
    assert_frame_equal(pl.concat(frames), df)


def test_split_by_null_keys() -> None:
    df = pl.DataFrame({"ts": [1, None, None, 2], "x": range(4)})

    frames = list(df.lazy().collect_batches(split_by="ts"))

    assert [f["ts"].to_list() for f in frames] == [[1], [None, None], [2]]
    assert_frame_equal(pl.concat(frames), df)
