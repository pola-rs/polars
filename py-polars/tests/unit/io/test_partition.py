from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypedDict

import pytest

import polars as pl
from polars.io.partition import MaxSizePartitioning

if TYPE_CHECKING:
    from pathlib import Path


class IOType(TypedDict):
    """A type of IO."""

    ext: str
    scan: Any
    sink: Any


io_types: list[IOType] = [
    {"ext": "csv", "scan": pl.scan_csv, "sink": pl.LazyFrame.sink_csv},
    {"ext": "jsonl", "scan": pl.scan_ndjson, "sink": pl.LazyFrame.sink_ndjson},
    {"ext": "parquet", "scan": pl.scan_parquet, "sink": pl.LazyFrame.sink_parquet},
    {"ext": "ipc", "scan": pl.scan_ipc, "sink": pl.LazyFrame.sink_ipc},
]


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.parametrize("length", [0, 1, 4, 5, 6, 7])
@pytest.mark.parametrize("max_size", [1, 2, 3])
@pytest.mark.write_disk
def test_max_size_partition(
    tmp_path: Path,
    io_type: IOType,
    length: int,
    max_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lf = pl.Series("a", range(length), pl.Int64).to_frame().lazy()

    monkeypatch.setenv("POLARS_FORCE_NEW_STREAMING", "1")
    (io_type["sink"])(
        lf,
        MaxSizePartitioning(tmp_path / f"{{part}}.{io_type['ext']}", max_size=max_size),
    )
    os.sync()

    i = 0
    while length > 0:
        assert (io_type["scan"])(tmp_path / f"{i}.{io_type['ext']}").select(
            pl.len()
        ).collect()[0, 0] == min(max_size, length)

        length -= max_size
        i += 1
