from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars._typing import EngineType
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("scan", "sink"),
    [
        (pl.scan_ipc, pl.LazyFrame.sink_ipc),
        (pl.scan_parquet, pl.LazyFrame.sink_parquet),
        (pl.scan_csv, pl.LazyFrame.sink_csv),
        (pl.scan_ndjson, pl.LazyFrame.sink_ndjson),
    ],
)
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
