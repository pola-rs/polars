import io
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars._typing import EngineType
from polars.testing import assert_frame_equal

SINKS = [
    (pl.scan_ipc, pl.LazyFrame.sink_ipc),
    (pl.scan_parquet, pl.LazyFrame.sink_parquet),
    (pl.scan_csv, pl.LazyFrame.sink_csv),
    (pl.scan_ndjson, pl.LazyFrame.sink_ndjson),
]


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.write_disk
def test_mkdir(tmp_path: Path, scan: Any, sink: Any, engine: EngineType) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    with pytest.raises(FileNotFoundError):
        sink(df.lazy(), tmp_path / "a" / "b" / "c" / "file", engine=engine)

    f = tmp_path / "a" / "b" / "c" / "file2"
    sink(df.lazy(), f, mkdir=True)

    assert_frame_equal(scan(f).collect(), df)


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.write_disk
def test_lazy_sinks(tmp_path: Path, scan: Any, sink: Any, engine: EngineType) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    lf1 = sink(df.lazy(), tmp_path / "a", lazy=True)
    lf2 = sink(df.lazy(), tmp_path / "b", lazy=True)

    assert not Path(tmp_path / "a").exists()
    assert not Path(tmp_path / "b").exists()

    pl.collect_all([lf1, lf2], engine=engine)

    assert_frame_equal(scan(tmp_path / "a").collect(), df)
    assert_frame_equal(scan(tmp_path / "b").collect(), df)


@pytest.mark.parametrize(
    "sink",
    [
        pl.LazyFrame.sink_ipc,
        pl.LazyFrame.sink_parquet,
        pl.LazyFrame.sink_csv,
        pl.LazyFrame.sink_ndjson,
    ],
)
@pytest.mark.write_disk
def test_double_lazy_error(sink: Any) -> None:
    df = pl.DataFrame({})

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="cannot create a sink on top of another sink",
    ):
        sink(sink(df.lazy(), "a", lazy=True), "b")


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_to_memory(sink: Any, scan: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    f = io.BytesIO()
    sink(df.lazy(), f)

    f.seek(0)
    assert_frame_equal(
        scan(f).collect(),
        df,
    )


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.write_disk
def test_sink_to_file(tmp_path: Path, sink: Any, scan: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    with (tmp_path / "f").open("w+") as f:
        sink(df.lazy(), f, sync_on_close="all")
        f.seek(0)
        assert_frame_equal(
            scan(f).collect(),
            df,
        )
